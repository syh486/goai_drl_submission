"""Measure full-atlas Warp BVH construction and batched LiDAR capture."""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch
import warp as wp

from sru_training.s10_warp_raycast import WarpStaticTerrainLidar
from training.terrains import SruAtlasConfig, build_sru_mujoco_model, generate_sru_atlas


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=6)
    parser.add_argument("--cols", type=int, default=30)
    parser.add_argument("--num-envs", type=int, default=128)
    parser.add_argument("--horizontal-samples", type=int, default=900)
    parser.add_argument("--captures", type=int, default=5)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--terrain-profile",
        choices=(
            "legacy_full",
            "stage1_flat",
            "stage2_low_density_obstacles",
            "stage3_reduced_height",
            "stage4_full_no_pits",
            "stage5_lower_density_stairs",
        ),
        default="legacy_full",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    atlas = generate_sru_atlas(
        SruAtlasConfig(
            seed=args.seed,
            num_rows=args.rows,
            num_cols=args.cols,
            terrain_profile=args.terrain_profile,
        )
    )
    atlas_seconds = time.perf_counter() - started
    model_started = time.perf_counter()
    model = build_sru_mujoco_model(atlas)
    model_seconds = time.perf_counter() - model_started

    device = wp.get_device(args.device)
    free_before = int(device.free_memory)
    build_started = time.perf_counter()
    lidar = WarpStaticTerrainLidar(
        model,
        horizontal_samples=args.horizontal_samples,
        device=args.device,
    )
    torch.cuda.synchronize(args.device)
    build_seconds = time.perf_counter() - build_started
    free_after_build = int(device.free_memory)

    roots = []
    for index in range(args.num_envs):
        tile = atlas.tiles[index % len(atlas.tiles)]
        root_z = float(tile.height_full.max()) * 0.005 + 0.8
        roots.append((*tile.origin[:2], root_z, 1.0, 0.0, 0.0, 0.0))
    root_qpos = torch.as_tensor(
        np.asarray(roots), dtype=torch.float32, device=args.device
    )
    lidar.capture(root_qpos)
    torch.cuda.synchronize(args.device)

    capture_started = time.perf_counter()
    for _ in range(args.captures):
        front, rear = lidar.capture(root_qpos)
    torch.cuda.synchronize(args.device)
    capture_seconds = (time.perf_counter() - capture_started) / args.captures
    free_after_capture = int(device.free_memory)

    geometry = lidar.geometry
    print(
        "WARP_TERRAIN_SMOKE_OK\n"
        f"atlas={args.rows}x{args.cols} geoms={geometry.geom_counts}\n"
        f"vertices={geometry.vertices.shape[0]} triangles={geometry.faces.shape[0]} "
        f"host_geometry_mib={(geometry.vertices.nbytes + geometry.faces.nbytes) / 2**20:.1f}\n"
        f"timing_s atlas={atlas_seconds:.3f} model={model_seconds:.3f} "
        f"warp_bvh={build_seconds:.3f} capture_mean={capture_seconds:.6f}\n"
        f"capture={args.num_envs}x2x96x{args.horizontal_samples} "
        f"rays={args.num_envs * 2 * 96 * args.horizontal_samples}\n"
        f"cuda_used_mib build={(free_before - free_after_build) / 2**20:.1f} "
        f"after_capture={(free_before - free_after_capture) / 2**20:.1f}\n"
        f"distance_mean_m={torch.cat((front, rear)).mean().item():.6f}"
    )


if __name__ == "__main__":
    main()
