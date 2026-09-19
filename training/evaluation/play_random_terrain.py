"""Inspect one migrated SRU terrain tile with the native S10 MuJoCo backend."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import time

import mujoco
import mujoco.viewer
import numpy as np
import torch

from sru_training.s10_mujoco_backend import S10NativeMujocoBackend


REPO_ROOT = Path(__file__).resolve().parents[2]
NEW_LOW_LEVEL = REPO_ROOT / "src/S10_sdk_deploy/policy/policy_official_20260828.onnx"


def _yaw_from_wxyz(quat: np.ndarray) -> float:
    w, x, y, z = quat
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _wrap(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def _goal_command(
    position: np.ndarray,
    yaw: float,
    goal: np.ndarray,
    *,
    max_forward: float,
    max_yaw: float,
) -> tuple[np.ndarray, float]:
    delta = goal[:2] - position[:2]
    distance = float(np.linalg.norm(delta))
    yaw_error = _wrap(math.atan2(float(delta[1]), float(delta[0])) - yaw)
    yaw_rate = float(np.clip(1.4 * yaw_error, -max_yaw, max_yaw))
    forward = 0.0
    if abs(yaw_error) < 0.75:
        forward = max_forward * min(1.0, distance / 1.5) * max(0.0, math.cos(yaw_error))
    return np.asarray((forward, 0.0, yaw_rate), dtype=np.float32), distance


def _update_markers(viewer: mujoco.viewer.Handle, spawn: np.ndarray, goal: np.ndarray) -> None:
    colors = (
        np.asarray((0.2, 0.55, 1.0, 0.9), dtype=np.float32),
        np.asarray((0.1, 0.9, 0.2, 0.9), dtype=np.float32),
    )
    with viewer.lock():
        viewer.user_scn.ngeom = 2
        for index, (position, color) in enumerate(zip((spawn, goal), colors)):
            marker = np.asarray(position, dtype=np.float64).copy()
            marker[2] += 0.25
            mujoco.mjv_initGeom(
                viewer.user_scn.geoms[index],
                mujoco.mjtGeom.mjGEOM_SPHERE,
                np.asarray((0.22, 0.22, 0.22), dtype=np.float64),
                marker,
                np.eye(3, dtype=np.float64).reshape(-1),
                color,
            )


def main() -> None:
    parser = argparse.ArgumentParser()
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
    parser.add_argument("--terrain-seed", type=int, default=42)
    parser.add_argument(
        "--surface-seed",
        type=int,
        help="Independent grass/gravel assignment seed; defaults to terrain seed.",
    )
    parser.add_argument("--grass-fraction", type=float, default=0.25)
    parser.add_argument("--gravel-fraction", type=float, default=0.25)
    parser.add_argument("--terrain-rows", type=int, default=2)
    parser.add_argument("--row", type=int, default=0)
    parser.add_argument("--col", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=3000)
    parser.add_argument("--max-forward-speed", type=float, default=0.4)
    parser.add_argument("--max-yaw-rate", type=float, default=0.7)
    parser.add_argument("--realtime-factor", type=float, default=1.0)
    parser.add_argument(
        "--control",
        choices=("goal_follower", "stand"),
        default="goal_follower",
    )
    parser.add_argument("--low-level-checkpoint", type=Path)
    parser.add_argument(
        "--low-level", choices=("official_onnx", "pim_him"), default="official_onnx"
    )
    parser.add_argument(
        "--low-level-profile",
        choices=("legacy", "official_20260828"),
        default="legacy",
    )
    args = parser.parse_args()
    if args.terrain_rows < 1 or not 0 <= args.row < args.terrain_rows:
        raise ValueError("row must lie inside the generated terrain rows")
    if not 0 <= args.col < 30:
        raise ValueError("col must be in [0, 29]")
    if args.max_steps < 1 or args.realtime_factor <= 0.0:
        raise ValueError("step count and realtime factor must be positive")

    low_level_checkpoint = args.low_level_checkpoint
    if args.low_level == "pim_him" and low_level_checkpoint is None:
        raise ValueError("--low-level pim_him requires --low-level-checkpoint")
    if low_level_checkpoint is None and args.low_level_profile == "official_20260828":
        low_level_checkpoint = NEW_LOW_LEVEL
    backend = S10NativeMujocoBackend(
        num_envs=1,
        task_mode="random_goal_sru",
        terrain_seed=args.terrain_seed,
        surface_seed=args.surface_seed,
        terrain_profile=args.terrain_profile,
        grass_fraction=args.grass_fraction,
        gravel_fraction=args.gravel_fraction,
        terrain_rows=args.terrain_rows,
        terrain_cols=30,
        device="cpu",
        low_level=args.low_level,
        low_level_checkpoint=low_level_checkpoint,
        low_level_profile=args.low_level_profile,
        reset_mode="fixed",
        max_episode_length=args.max_steps + 1,
        single_episode_length=args.max_steps + 1,
        use_lidar=False,
        use_height=False,
        physics_workers=1,
        seed=args.terrain_seed,
    )
    backend.set_environment_terrain_tile(0, row=args.row, col=args.col)
    tile = backend.terrain_atlas.tile(args.row, args.col)
    print(
        "[SRU terrain] "
        f"seed={args.terrain_seed} tile=({args.row},{args.col}) "
        f"type={tile.terrain_type} difficulty={tile.difficulty:.4f} "
        f"surface_patches={len(backend.terrain_atlas.surface_patches)}; "
        "blue=spawn green=goal",
        flush=True,
    )

    period = 1.0 / backend.action_spec.policy_hz / args.realtime_factor
    try:
        with mujoco.viewer.launch_passive(backend.model, backend.data[0]) as viewer:
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            viewer.cam.trackbodyid = backend.base_body_id
            viewer.cam.distance = 8.0
            viewer.cam.azimuth = 145.0
            viewer.cam.elevation = -35.0
            for step in range(args.max_steps):
                if not viewer.is_running():
                    break
                started = time.monotonic()
                spawn = backend.random_spawn_positions[0].copy()
                goal = backend.random_goal_positions[0].copy()
                _update_markers(viewer, spawn, goal)
                if args.control == "stand":
                    command = np.zeros(3, dtype=np.float32)
                    distance = float(np.linalg.norm(goal[:2] - backend.data[0].qpos[:2]))
                else:
                    command, distance = _goal_command(
                        backend.data[0].qpos[:3],
                        _yaw_from_wxyz(backend.data[0].qpos[3:7]),
                        goal,
                        max_forward=args.max_forward_speed,
                        max_yaw=args.max_yaw_rate,
                    )
                _, reward, done, info = backend.step(torch.from_numpy(command).unsqueeze(0))
                if step % 10 == 0 or bool(done[0]):
                    print(
                        f"[SRU terrain] step={step:4d} distance={distance:5.2f}m "
                        f"reward={float(reward[0]):+.4f} reason={info['done_reason'][0]}",
                        flush=True,
                    )
                viewer.sync()
                time.sleep(max(0.0, period - (time.monotonic() - started)))
    finally:
        backend.close()


if __name__ == "__main__":
    main()
