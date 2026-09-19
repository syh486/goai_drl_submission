"""Compare Warp static-terrain rays with MuJoCo's native CPU ray caster."""

from __future__ import annotations

import argparse

import mujoco
import numpy as np
import torch

from src.S10_sdk_deploy.interface.robot.simulation.s10_lidar import S10LidarSampler
from sru_training.s10_warp_raycast import WarpStaticTerrainLidar, _build_static_terrain
from training.terrains import SruAtlasConfig, build_sru_mujoco_model, generate_sru_atlas
from training.terrains.constants import VERTICAL_SCALE


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=1)
    parser.add_argument("--cols", type=int, default=4)
    parser.add_argument("--horizontal-samples", type=int, default=90)
    parser.add_argument("--random-rays", type=int, default=512)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def cpu_raycast(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    origins: np.ndarray,
    directions: np.ndarray,
    max_range: float,
) -> np.ndarray:
    group = np.asarray((1, 0, 0, 0, 0, 0), dtype=np.uint8)
    result = np.full(origins.shape[0], max_range, dtype=np.float64)
    for index, (origin, direction) in enumerate(zip(origins, directions)):
        distance = mujoco.mj_ray(
            model, data, origin, direction, group, 1, -1, None, None
        )
        if 0.0 < distance < max_range:
            result[index] = distance
    return result


def compare_distances(label: str, cpu: np.ndarray, warp: np.ndarray) -> None:
    max_range = 10.0
    cpu_hit = cpu < max_range
    warp_hit = warp < max_range
    hit_agreement = float(np.mean(cpu_hit == warp_hit))
    common = cpu_hit & warp_hit
    differences = np.abs(cpu[common] - warp[common])
    p99 = float(np.quantile(differences, 0.99)) if differences.size else 0.0
    maximum = float(differences.max()) if differences.size else 0.0
    print(
        f"{label}: rays={cpu.size} common_hits={differences.size} "
        f"hit_agreement={hit_agreement:.6f} p99_abs_m={p99:.6g} max_abs_m={maximum:.6g}"
    )
    if differences.size:
        common_indices = np.flatnonzero(common)
        worst = np.argsort(differences)[-5:][::-1]
        print(
            "  worst="
            + ", ".join(
                f"idx={common_indices[index]} cpu={cpu[common_indices[index]]:.6f} "
                f"warp={warp[common_indices[index]]:.6f} diff={differences[index]:.6f}"
                for index in worst
            )
        )
    assert hit_agreement >= 0.995, (label, hit_agreement)
    assert p99 <= 2.0e-3, (label, p99)


def cpu_lidar_float64(
    sampler: S10LidarSampler,
    data: mujoco.MjData,
    root: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    sampler.capture_pose(data, root[:3], root[3:])
    scans = []
    for view in range(2):
        mujoco.mj_multiRay(
            sampler.model,
            data,
            sampler.origins[view],
            np.ascontiguousarray(sampler.world_directions[view].reshape(-1)),
            sampler.geom_group,
            1,
            sampler.base_body_id,
            sampler.geom_ids,
            sampler.distances,
            None,
            sampler.rays_per_view if hasattr(sampler, "rays_per_view") else 96 * sampler.horizontal_samples,
            sampler.max_range_m,
        )
        scan = sampler.distances.copy()
        scan[(scan <= 0.0) | (scan > sampler.max_range_m)] = sampler.max_range_m
        scans.append(scan.reshape(96, sampler.horizontal_samples))
    return scans[0], scans[1]


def main() -> None:
    args = parse_args()
    atlas = generate_sru_atlas(
        SruAtlasConfig(seed=42, num_rows=args.rows, num_cols=args.cols)
    )
    model = build_sru_mujoco_model(atlas)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    geometry = _build_static_terrain(model)
    assert geometry.geom_counts == {
        "mesh": 0,
        "hfield": args.rows * args.cols,
        "box": 4 + len(atlas.surface_patches),
        "plane": 0,
    }
    assert not geometry.has_plane
    assert float(geometry.vertices[:, 2].min()) < -0.1

    raycaster = WarpStaticTerrainLidar(
        model,
        horizontal_samples=args.horizontal_samples,
        device=args.device,
    )
    print(
        f"geometry: vertices={geometry.vertices.shape[0]} "
        f"triangles={geometry.faces.shape[0]} patches={geometry.hfield_patch_count}"
    )

    rng = np.random.default_rng(20260905)
    x_extent = args.rows * atlas.config.tile_size[0]
    y_extent = args.cols * atlas.config.tile_size[1]
    origins = np.column_stack(
        (
            rng.uniform(-2.0, x_extent + 2.0, args.random_rays),
            rng.uniform(-2.0, y_extent + 2.0, args.random_rays),
            np.full(args.random_rays, 3.5),
        )
    )
    directions = rng.normal(size=(args.random_rays, 3))
    directions[:, 2] = -np.abs(directions[:, 2]) - 0.1
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)

    # Force one vertical ray onto the deepest sample. This catches the old
    # unconditional z=0 plane, which made every negative pit appear flat.
    pit_tile = min(atlas.tiles, key=lambda tile: int(tile.height_full.min()))
    pit_index = np.unravel_index(np.argmin(pit_tile.height_full), pit_tile.height_full.shape)
    pit_origin = np.asarray(
        (
            pit_tile.origin[0] + pit_index[0] * 0.1 - 15.0,
            pit_tile.origin[1] + pit_index[1] * 0.1 - 15.0,
            3.0,
        ),
        dtype=np.float64,
    )
    origins[0] = pit_origin
    directions[0] = (0.0, 0.0, -1.0)

    cpu = cpu_raycast(model, data, origins, directions, 10.0)
    warp = raycaster.raycast_world(
        torch.as_tensor(origins, device=args.device),
        torch.as_tensor(directions, device=args.device),
    ).cpu().numpy()
    expected_pit_distance = 3.0 - float(pit_tile.height_full[pit_index]) * VERTICAL_SCALE
    np.testing.assert_allclose(cpu[0], expected_pit_distance, atol=2.0e-5)
    np.testing.assert_allclose(warp[0], expected_pit_distance, atol=2.0e-4)
    compare_distances("world_rays", cpu, warp)

    cpu_sampler = S10LidarSampler(
        model, None, horizontal_samples=args.horizontal_samples
    )
    roots = []
    cpu_scans = []
    for tile in atlas.tiles[: min(3, len(atlas.tiles))]:
        # The tile center can be occupied by a generated maze wall. Keep the
        # synthetic sensor origin above the complete tile so this contract
        # tests ray geometry, not the undefined inside-solid case.
        root_z = float(tile.height_full.max()) * VERTICAL_SCALE + 0.8
        root = np.asarray((*tile.origin[:2], root_z, 1.0, 0.0, 0.0, 0.0))
        roots.append(root)
        cpu_scans.extend(cpu_lidar_float64(cpu_sampler, data, root))
    warp_front, warp_rear = raycaster.capture(
        torch.as_tensor(np.stack(roots), dtype=torch.float32, device=args.device)
    )
    warp_scans = []
    for front, rear in zip(warp_front.cpu().numpy(), warp_rear.cpu().numpy()):
        warp_scans.extend((front, rear))
    compare_distances(
        "dual_lidar",
        np.asarray(cpu_scans, dtype=np.float32).reshape(-1),
        np.asarray(warp_scans, dtype=np.float32).reshape(-1),
    )
    print("WARP_RAYCAST_CONTRACT_OK")


if __name__ == "__main__":
    main()
