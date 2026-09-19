"""Play a trained PPO policy on the SRU random-terrain MuJoCo task."""

from __future__ import annotations

import argparse
import os
from collections import Counter
from pathlib import Path
import sys
import time

import mujoco
import mujoco.viewer
import numpy as np
import torch

from training.evaluation.policy import build_policy
from sru_training.s10_mujoco_backend import DONE_NONE, S10NativeMujocoBackend
from sru_training.s10_mujoco_env import S10MujocoVecEnv
from sru_training.s10_policy_config import observation_spec_from_policy_state
from sru_training.s10_viewer_overlay import update_perception_overlay


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ONNX = REPO_ROOT / "src/S10_sdk_deploy/policy/policy_official_20260828.onnx"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--policy-index", type=int, choices=(1,), default=1)
    parser.add_argument(
        "--low-level", choices=("official_onnx", "pim_him"), default="official_onnx"
    )
    parser.add_argument("--low-level-checkpoint", type=Path)
    parser.add_argument("--terrain-seed", type=int, default=42)
    parser.add_argument("--surface-seed", type=int, default=20260905)
    parser.add_argument(
        "--terrain-profile",
        choices=(
            "stage1_flat",
            "stage2_low_density_obstacles",
            "stage3_reduced_height",
            "stage4_full_no_pits",
            "stage5_lower_density_stairs",
        ),
        default="stage4_full_no_pits",
    )
    parser.add_argument("--grass-fraction", type=float, default=0.25)
    parser.add_argument("--gravel-fraction", type=float, default=0.25)
    parser.add_argument("--terrain-rows", type=int, default=6)
    parser.add_argument("--terrain-col", type=int)
    parser.add_argument("--terrain-row", type=int)
    parser.add_argument("--episodes", type=int, default=12)
    parser.add_argument("--max-episode-length", type=int, default=300)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--realtime-factor", type=float, default=1.0)
    parser.add_argument("--contact-threshold", type=float, default=500.0)
    parser.add_argument(
        "--filter-alpha",
        type=float,
        default=0.5,
        help="fixed PLAY low-pass alpha; 0.5 matches the upstream MX default",
    )
    parser.add_argument("--camera-distance", type=float, default=4.5)
    parser.add_argument("--camera-azimuth", type=float, default=145.0)
    parser.add_argument("--camera-elevation", type=float, default=-45.0)
    parser.add_argument(
        "--near-clip",
        type=float,
        default=0.05,
        help="GUI near clipping distance in metres (independent of terrain-atlas extent)",
    )
    args = parser.parse_args()

    checkpoint = args.checkpoint.expanduser().resolve()
    low_level_checkpoint = (
        DEFAULT_ONNX
        if args.low_level_checkpoint is None
        else args.low_level_checkpoint.expanduser().resolve()
    )
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if not low_level_checkpoint.is_file():
        raise FileNotFoundError(low_level_checkpoint)
    if args.low_level == "pim_him" and args.low_level_checkpoint is None:
        raise ValueError("--low-level pim_him requires --low-level-checkpoint")
    if args.episodes < 1 or args.max_episode_length < 1:
        raise ValueError("episodes and max episode length must be positive")
    if args.realtime_factor <= 0.0:
        raise ValueError("realtime factor must be positive")
    if args.camera_distance <= 0.0:
        raise ValueError("camera distance must be positive")
    if args.contact_threshold < 0.0:
        raise ValueError("contact threshold must be non-negative")
    if not 0.0 <= args.filter_alpha < 1.0:
        raise ValueError("filter alpha must be in [0, 1)")
    if args.near_clip <= 0.0:
        raise ValueError("near clip distance must be positive")
    if (args.terrain_row is None) != (args.terrain_col is None):
        raise ValueError("--terrain-row and --terrain-col must be provided together")

    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state_key = "model_state_dict"
    if state_key not in payload:
        raise KeyError(f"{checkpoint} does not contain {state_key}")
    state_dict = payload[state_key]
    spec = observation_spec_from_policy_state(state_dict)

    torch.manual_seed(args.terrain_seed)
    np.random.seed(args.terrain_seed)
    backend = S10NativeMujocoBackend(
        num_envs=1,
        device=args.device,
        task_mode="random_goal_sru",
        terrain_seed=args.terrain_seed,
        surface_seed=args.surface_seed,
        terrain_profile=args.terrain_profile,
        grass_fraction=args.grass_fraction,
        gravel_fraction=args.gravel_fraction,
        terrain_rows=args.terrain_rows,
        terrain_cols=30,
        reset_mode="fixed",
        max_episode_length=args.max_episode_length,
        single_episode_length=args.max_episode_length,
        low_level=args.low_level,
        low_level_checkpoint=low_level_checkpoint,
        low_level_profile="official_20260828",
        use_lidar=True,
        use_height=True,
        sensor_backend="warp",
        lidar_horizontal_samples=900,
        sensor_workers=1,
        physics_workers=1,
        contact_threshold=args.contact_threshold,
        contact_persistence_steps=1,
        record_lidar_for_visualization=True,
        seed=args.terrain_seed,
    )
    if args.terrain_row is not None:
        backend.set_environment_terrain_tile(
            0, row=args.terrain_row, col=args.terrain_col
        )

    model_extent = float(backend.model.stat.extent)
    default_near_clip = model_extent * float(backend.model.vis.map.znear)
    backend.model.vis.map.znear = min(
        float(backend.model.vis.map.znear), args.near_clip / model_extent
    )
    actual_near_clip = model_extent * float(backend.model.vis.map.znear)

    env = S10MujocoVecEnv(backend, obs_spec=spec, device=args.device)
    env._filter_alpha.fill_(args.filter_alpha)
    policy = build_policy(state_dict, args.device)
    policy.eval()
    obs, _ = env.get_observations()
    reasons: Counter[str] = Counter()
    episode_return = 0.0
    episode_step = 0
    completed_episodes = 0
    policy_period = 1.0 / backend.action_spec.policy_hz

    try:
        with mujoco.viewer.launch_passive(backend.model, backend.data[0]) as viewer:
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            viewer.cam.trackbodyid = backend.base_body_id
            viewer.cam.distance = args.camera_distance
            viewer.cam.azimuth = args.camera_azimuth
            viewer.cam.elevation = args.camera_elevation
            tile = backend.terrain_atlas.tile(
                int(backend.terrain_levels[0]), int(backend.terrain_types[0])
            )
            print(
                f"[S10 random play] checkpoint={checkpoint} iter={payload.get('iter')} "
                f"tile=({tile.row},{tile.col}) type={tile.terrain_type} "
                f"device={args.device} low_level={args.low_level}; "
                f"contact={args.contact_threshold:.0f}N/1step; "
                f"filter_alpha={args.filter_alpha:.2f}; "
                f"near_clip={default_near_clip:.3f}->{actual_near_clip:.3f}m; "
                "green=goal cyan=front_lidar magenta=rear_lidar",
                flush=True,
            )
            with torch.inference_mode():
                while viewer.is_running() and completed_episodes < args.episodes:
                    started = time.monotonic()
                    action = policy.act_inference(obs)
                    obs, reward, dones, info = env.step(action)
                    episode_return += float(reward[0].item())
                    episode_step += 1

                    goal = backend._goal_point(0).copy()
                    front_points, rear_points = backend.lidar_hit_points()
                    update_perception_overlay(viewer, goal, front_points, rear_points)
                    viewer.sync()

                    if episode_step % 5 == 0 or bool(dones[0].item()):
                        distance = float(info["terminal_goal_distance_xy"][0])
                        command = backend.last_cmd[0]
                        print(
                            f"\rstep={episode_step:3d} tile=({int(backend.terrain_levels[0])},"
                            f"{int(backend.terrain_types[0])}) dist={distance:5.2f}m "
                            f"cmd=({command[0]:+.2f},{command[2]:+.2f}) "
                            f"reward={episode_return:+7.2f}",
                            end="",
                            flush=True,
                        )

                    if bool(dones[0].item()):
                        reason = str(info["done_reason"][0])
                        if reason != DONE_NONE:
                            reasons[reason] += 1
                        completed_episodes += 1
                        print(
                            f"  DONE={reason} episodes={completed_episodes}/{args.episodes}",
                            flush=True,
                        )
                        policy.memory_a.reset(dones, use_random_init=False)
                        env._filter_alpha[dones].fill_(args.filter_alpha)
                        episode_return = 0.0
                        episode_step = 0

                    remaining = policy_period / args.realtime_factor - (
                        time.monotonic() - started
                    )
                    if remaining > 0.0:
                        time.sleep(remaining)
    finally:
        env.close()

    print(f"[S10 random play] terminations={dict(reasons)}", flush=True)


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
