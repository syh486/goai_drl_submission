"""Collect raw dual-LiDAR replay from generated SRU MuJoCo terrain.

Only metric ray returns and root poses are stored. Noise, dropout, pooling,
world-z construction, and normalization happen in the trainer. Each terrain
seed cycles through every atlas tile, so a small number of environments does
not silently collect only one difficulty row.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .replay import write_metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--samples", type=int, default=10000)
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--sample-every", type=int, default=2)
    parser.add_argument(
        "--terrain-seed", type=int, action="append", default=None,
        help="Terrain seed; repeat the option to collect multiple independent atlases.",
    )
    parser.add_argument("--surface-seed", type=int, default=20260905)
    parser.add_argument("--terrain-rows", type=int, default=6)
    parser.add_argument("--terrain-cols", type=int, default=30)
    parser.add_argument("--tile-cycle-steps", type=int, default=128)
    parser.add_argument("--require-full-coverage", action="store_true")
    parser.add_argument("--low-level-checkpoint", type=Path, required=True)
    parser.add_argument("--low-level-profile", choices=("legacy", "official_20260828"), default="official_20260828")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20260905)
    parser.add_argument("--grass-fraction", type=float, default=0.25)
    parser.add_argument("--gravel-fraction", type=float, default=0.25)
    return parser.parse_args()


def _save_chunk(output_dir: Path, chunk_id: int, rows: list[dict[str, np.ndarray]]) -> None:
    payload = {key: np.concatenate([row[key] for row in rows], axis=0) for key in rows[0]}
    temporary = output_dir / f"chunk_{chunk_id:06d}.npz.part"
    target = output_dir / f"chunk_{chunk_id:06d}.npz"
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **payload)
    temporary.replace(target)


def _next_tile_batch(
    stream: list[int], cursor: int, *, tile_count: int, num_envs: int, rng: np.random.Generator
) -> tuple[np.ndarray, int, list[int]]:
    while cursor + num_envs > len(stream):
        stream.extend(rng.permutation(tile_count).tolist())
    return np.asarray(stream[cursor:cursor + num_envs], dtype=np.int64), cursor + num_envs, stream


def main() -> int:
    args = parse_args()
    if args.num_envs < 1 or args.samples < 1 or args.chunk_size < 1 or args.sample_every < 1 or args.tile_cycle_steps < 1:
        raise ValueError("num-envs, samples, chunk-size, sample-every, and tile-cycle-steps must be positive")
    if not str(args.device).startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("random-terrain Warp collection requires an available CUDA device")

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if any(output_dir.glob("chunk_*.npz")):
        raise FileExistsError(
            f"replay directory already contains chunks: {output_dir}; use a new directory"
        )
    terrain_seeds = args.terrain_seed or [20260905]
    write_metadata(output_dir, terrain_seeds=terrain_seeds, num_envs=args.num_envs)
    from sru_training.s10_warp_raycast import WarpStaticTerrainLidar
    from sru_training.s10_mujoco_backend import S10NativeMujocoBackend

    rng = np.random.default_rng(args.seed)
    rows: list[dict[str, np.ndarray]] = []
    chunk_id = 0
    collected = 0
    coverage: dict[str, object] = {}
    target_per_seed = int(np.ceil(args.samples / len(terrain_seeds)))

    for seed_index, terrain_seed in enumerate(terrain_seeds):
        backend = S10NativeMujocoBackend(
            num_envs=args.num_envs,
            task_mode="random_goal_sru",
            terrain_seed=terrain_seed,
            surface_seed=args.surface_seed + seed_index,
            terrain_rows=args.terrain_rows,
            terrain_cols=args.terrain_cols,
            grass_fraction=args.grass_fraction,
            gravel_fraction=args.gravel_fraction,
            device=args.device,
            low_level="official_onnx",
            low_level_checkpoint=args.low_level_checkpoint,
            low_level_profile=args.low_level_profile,
            use_lidar=False,
            use_height=False,
            reset_mode="fixed",
            max_episode_length=300,
            sensor_backend="warp",
            seed=args.seed + seed_index,
        )
        scanner = WarpStaticTerrainLidar(
            backend.model, horizontal_samples=900, device=str(args.device)
        )
        atlas = backend.terrain_atlas
        tile_count = atlas.config.num_rows * atlas.config.num_cols
        tile_stream = rng.permutation(tile_count).tolist()
        tile_cursor = 0
        tile_samples = np.zeros(tile_count, dtype=np.int64)
        episode_ids = np.arange(args.num_envs, dtype=np.int64) + seed_index * 1_000_000_000
        local_collected = 0
        step = 0
        try:
            tile_batch, tile_cursor, tile_stream = _next_tile_batch(
                tile_stream, tile_cursor, tile_count=tile_count, num_envs=args.num_envs, rng=rng
            )
            backend.set_environment_terrain_tiles(tile_batch)
            while local_collected < target_per_seed and collected < args.samples:
                if step > 0 and step % args.tile_cycle_steps == 0:
                    tile_batch, tile_cursor, tile_stream = _next_tile_batch(
                        tile_stream, tile_cursor, tile_count=tile_count, num_envs=args.num_envs, rng=rng
                    )
                    backend.set_environment_terrain_tiles(tile_batch)

                poses = np.asarray([data.qpos[:7] for data in backend.data], dtype=np.float32)
                front, rear, _, _ = scanner.capture_with_world_z(
                    torch.as_tensor(poses, dtype=torch.float32, device=args.device)
                )
                selected = np.arange(args.num_envs, dtype=np.int64) if step % args.sample_every == 0 else np.empty(0, dtype=np.int64)
                selected = selected[: min(len(selected), target_per_seed - local_collected, args.samples - collected)]
                if len(selected):
                    tile_samples[backend.environment_tile_indices[selected]] += 1
                    rows.append({
                        "front_raw_d": front.detach().cpu().numpy()[selected].astype(np.float32, copy=False),
                        "rear_raw_d": rear.detach().cpu().numpy()[selected].astype(np.float32, copy=False),
                        "root_qpos": poses[selected].astype(np.float32, copy=False),
                        "episode_id": episode_ids[selected].copy(),
                        "terrain_tile": backend.environment_tile_indices[selected].astype(np.int64, copy=True),
                    })
                    local_collected += len(selected)
                    collected += len(selected)
                commands = np.zeros((args.num_envs, 3), dtype=np.float32)
                commands[:, 0] = rng.uniform(-0.35, 0.95, args.num_envs)
                commands[:, 2] = rng.uniform(-0.8, 0.8, args.num_envs)
                _, _, dones, _ = backend.step(torch.as_tensor(commands, device=args.device))
                done_indices = torch.nonzero(dones, as_tuple=False).flatten().cpu().numpy()
                episode_ids[done_indices] += args.num_envs
                step += 1
                if sum(len(row["episode_id"]) for row in rows) >= args.chunk_size:
                    _save_chunk(output_dir, chunk_id, rows)
                    chunk_id += 1
                    rows.clear()
                    print(f"[collect] seed={terrain_seed} samples={collected}/{args.samples} steps={step} chunks={chunk_id}", flush=True)
            if rows:
                _save_chunk(output_dir, chunk_id, rows)
                chunk_id += 1
                rows.clear()
            type_counts = {name: 0 for name in ("maze", "non_maze", "stairs", "pits")}
            for tile, count in zip(atlas.tiles, tile_samples):
                type_counts[tile.terrain_type] += int(count)
            coverage[str(terrain_seed)] = {
                "samples": int(local_collected),
                "tile_sample_min": int(tile_samples.min()),
                "tile_sample_max": int(tile_samples.max()),
                "tiles_seen": int(np.count_nonzero(tile_samples)),
                "tile_count": int(tile_count),
                "terrain_type_sample_counts": type_counts,
                "terrain_type_tile_counts": atlas.type_counts,
                "surface_area_m2": atlas.surface_counts,
                "tile_difficulties": [float(tile.difficulty) for tile in atlas.tiles],
            }
            print(f"[coverage] seed={terrain_seed} tiles={np.count_nonzero(tile_samples)}/{tile_count} min={tile_samples.min()} max={tile_samples.max()} types={type_counts}", flush=True)
            if args.require_full_coverage and np.any(tile_samples == 0):
                raise RuntimeError(f"terrain seed {terrain_seed} did not cover all {tile_count} atlas tiles")
            if args.require_full_coverage and any(value == 0 for value in atlas.type_counts.values()):
                raise RuntimeError(f"terrain seed {terrain_seed} did not generate all terrain types: {atlas.type_counts}")
            if args.require_full_coverage:
                if args.grass_fraction > 0.0 and coverage[str(terrain_seed)]["surface_area_m2"]["grass"] <= 0.0:
                    raise RuntimeError(f"terrain seed {terrain_seed} generated no grass surface")
                if args.gravel_fraction > 0.0 and coverage[str(terrain_seed)]["surface_area_m2"]["gravel"] <= 0.0:
                    raise RuntimeError(f"terrain seed {terrain_seed} generated no gravel surface")
        finally:
            backend.close()

    (output_dir / "coverage.json").write_text(json.dumps(coverage, indent=2), encoding="utf-8")
    print(f"[collect] complete samples={collected} chunks={chunk_id}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
