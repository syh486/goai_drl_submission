"""Headless deterministic checkpoint evaluation on the SRU random-goal task."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import time

import numpy as np
import torch

from training.evaluation.policy import build_policy
from sru_training.s10_mujoco_backend import DONE_COMPLETE, S10NativeMujocoBackend
from sru_training.s10_mujoco_env import S10MujocoVecEnv
from sru_training.s10_policy_config import observation_spec_from_policy_state


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ONNX = ROOT / "src/S10_sdk_deploy/policy/policy_official_20260828.onnx"
DEFAULT_ENCODER = ROOT / "checkpoints/lidar_encoder_random_terrain_ft/best.pt"


def evaluate(args: argparse.Namespace, checkpoint: Path) -> dict[str, object]:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state_dict = payload["model_state_dict"]
    spec = observation_spec_from_policy_state(state_dict)
    policy = build_policy(state_dict, args.device)
    policy.eval()
    low_level_checkpoint = (
        DEFAULT_ONNX
        if args.low_level_checkpoint is None
        else args.low_level_checkpoint.expanduser().resolve()
    )
    backend = S10NativeMujocoBackend(
        num_envs=args.num_envs,
        device=args.device,
        task_mode="random_goal_sru",
        terrain_profile=args.terrain_profile,
        terrain_seed=args.terrain_seed,
        surface_seed=args.surface_seed,
        grass_fraction=0.25,
        gravel_fraction=0.25,
        terrain_rows=6,
        terrain_cols=30,
        reset_mode="fixed",
        max_episode_length=args.max_episode_length,
        single_episode_length=args.max_episode_length,
        low_level=args.low_level,
        low_level_checkpoint=low_level_checkpoint,
        low_level_profile="official_20260828",
        low_level_ready_after_reset=args.low_level == "official_onnx",
        lidar_encoder_checkpoint=DEFAULT_ENCODER,
        use_lidar=True,
        use_height=True,
        sensor_backend="warp",
        lidar_horizontal_samples=900,
        sensor_workers=8,
        physics_workers=8,
        contact_threshold=args.contact_threshold,
        contact_persistence_steps=args.contact_persistence_steps,
        seed=args.seed,
    )
    env = S10MujocoVecEnv(backend, obs_spec=spec, device=args.device)
    obs, _ = env.get_observations()
    reasons: Counter[str] = Counter()
    lengths: list[int] = []
    timeout_distances: list[float] = []
    action_abs_sum = np.zeros(2, dtype=np.float64)
    action_samples = 0
    started = time.monotonic()
    try:
        with torch.inference_mode():
            while sum(reasons.values()) < args.episodes:
                actions = policy.act_inference(obs)
                action_abs_sum += np.abs(actions.detach().cpu().numpy()).sum(axis=0)
                action_samples += args.num_envs
                obs, _, dones, info = env.step(actions)
                done_indices = torch.nonzero(dones, as_tuple=False).flatten().cpu().numpy()
                for index in done_indices:
                    if sum(reasons.values()) >= args.episodes:
                        break
                    reason = str(info["done_reason"][index])
                    reasons[reason] += 1
                    lengths.append(int(info["terminal_episode_steps"][index]))
                    if reason == "timeout":
                        timeout_distances.append(
                            float(info["terminal_goal_distance_xy"][index])
                        )
                policy.memory_a.reset(dones, use_random_init=False)
    finally:
        env.close()
    total = sum(reasons.values())
    return {
        "checkpoint": str(checkpoint),
        "low_level": args.low_level,
        "low_level_checkpoint": str(low_level_checkpoint),
        "iteration": payload.get("iter"),
        "episodes": total,
        "success_rate": reasons[DONE_COMPLETE] / max(total, 1),
        "reasons": dict(reasons),
        "mean_episode_length": float(np.mean(lengths)),
        "mean_timeout_distance": (
            float(np.mean(timeout_distances)) if timeout_distances else None
        ),
        "mean_abs_action": (action_abs_sum / max(action_samples, 1)).tolist(),
        "contact_threshold": args.contact_threshold,
        "contact_persistence_steps": args.contact_persistence_steps,
        "wall_seconds": time.monotonic() - started,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoints", nargs="+", type=Path)
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--episodes", type=int, default=64)
    parser.add_argument("--max-episode-length", type=int, default=300)
    parser.add_argument("--contact-threshold", type=float, default=500.0)
    parser.add_argument("--contact-persistence-steps", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--low-level", choices=("official_onnx", "pim_him"), default="official_onnx"
    )
    parser.add_argument("--low-level-checkpoint", type=Path)
    parser.add_argument("--seed", type=int, default=20260907)
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
        default="stage5_lower_density_stairs",
    )
    args = parser.parse_args()
    if args.num_envs < 1 or args.episodes < 1 or args.max_episode_length < 1:
        raise ValueError("environment, episode, and step counts must be positive")
    if args.contact_threshold < 0.0 or args.contact_persistence_steps < 1:
        raise ValueError("contact threshold must be non-negative and persistence positive")
    if args.low_level == "pim_him" and args.low_level_checkpoint is None:
        raise ValueError("--low-level pim_him requires --low-level-checkpoint")
    for value in args.checkpoints:
        checkpoint = value.expanduser().resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        print("RANDOM_TERRAIN_EVAL", evaluate(args, checkpoint), flush=True)


if __name__ == "__main__":
    main()
