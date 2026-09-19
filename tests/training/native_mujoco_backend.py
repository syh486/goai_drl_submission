"""Smoke a real multi-environment MuJoCo backend and the rsl_rl VecEnv adapter."""

from __future__ import annotations

import argparse
import time

import torch

from sru_training.s10_mujoco_backend import S10NativeMujocoBackend
from sru_training.s10_mujoco_env import S10MujocoVecEnv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-envs", type=int, default=2)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--no-lidar", action="store_true")
    parser.add_argument("--no-height", action="store_true")
    parser.add_argument("--sensor-workers", type=int, default=None)
    parser.add_argument("--physics-workers", type=int, default=None)
    parser.add_argument("--lidar-horizontal-samples", type=int, default=900)
    parser.add_argument("--sensor-backend", choices=("cpu", "warp"), default="cpu")
    args = parser.parse_args()

    backend = S10NativeMujocoBackend(
        num_envs=args.num_envs,
        device=args.device,
        task_mode="random_goal_sru",
        terrain_profile="stage4_full_no_pits",
        reset_mode="fixed",
        use_lidar=not args.no_lidar,
        use_height=not args.no_height,
        sensor_workers=args.sensor_workers,
        physics_workers=args.physics_workers,
        lidar_horizontal_samples=args.lidar_horizontal_samples,
        sensor_backend=args.sensor_backend,
        max_episode_length=max(args.steps + 1, 8),
    )
    expected_low_level_calls = backend.physics_steps // backend.low_level_decimation * args.num_envs
    assert backend.physics_steps == 200, backend.physics_steps
    assert backend.low_level_decimation == 20, backend.low_level_decimation
    env = S10MujocoVecEnv(backend, device=args.device)
    observations, extras = env.get_observations()
    assert observations.shape == (args.num_envs, 2575)
    assert extras["observations"]["critic"].shape == (args.num_envs, 5712)
    start = time.perf_counter()
    for step in range(args.steps):
        actions = torch.zeros((args.num_envs, 2), device=args.device)
        actions[:, 0] = 0.1
        obs, rewards, dones, info = env.step(actions)
        assert obs.shape == (args.num_envs, 2575)
        assert rewards.shape == (args.num_envs,)
        assert dones.shape == (args.num_envs,)
        assert torch.isfinite(obs).all() and torch.isfinite(rewards).all()
        assert info["timing_config"]["onnx_inference_calls_this_step"] == expected_low_level_calls, info
        print("NATIVE_STEP", step, info, flush=True)
    elapsed = time.perf_counter() - start
    print(
        "NATIVE_MUJOCO_OK",
        {"num_envs": args.num_envs, "steps": args.steps, "wall_s": round(elapsed, 3), "device": args.device},
        flush=True,
    )
    env.close()


if __name__ == "__main__":
    main()
