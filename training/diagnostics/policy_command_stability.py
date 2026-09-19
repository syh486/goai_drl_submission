"""Measure deterministic high-level command oscillation on random terrain."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
import time

import numpy as np
import torch

from training.evaluation.policy import build_policy
from sru_training.s10_mujoco_backend import S10NativeMujocoBackend
from sru_training.s10_mujoco_env import S10MujocoVecEnv
from sru_training.s10_policy_config import observation_spec_from_policy_state


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ONNX = ROOT / "src/S10_sdk_deploy/policy/policy_official_20260828.onnx"
DEFAULT_ENCODER = ROOT / "checkpoints/lidar_encoder_random_terrain_ft/best.pt"


def _yaw_from_wxyz(quaternion: np.ndarray) -> float:
    w, x, y, z = quaternion
    return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def _wrap_pi(angle: float) -> float:
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


def _safe_mean(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def evaluate(checkpoint: Path, args: argparse.Namespace) -> dict[str, object]:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state_dict = payload["model_state_dict"]
    spec = observation_spec_from_policy_state(state_dict)
    policy = build_policy(state_dict, args.device)
    policy.eval()
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
        low_level="official_onnx",
        low_level_checkpoint=DEFAULT_ONNX,
        low_level_profile="official_20260828",
        low_level_ready_after_reset=True,
        lidar_encoder_checkpoint=DEFAULT_ENCODER,
        use_lidar=True,
        use_height=True,
        sensor_backend="warp",
        lidar_horizontal_samples=900,
        sensor_workers=args.workers,
        physics_workers=args.workers,
        contact_threshold=args.contact_threshold,
        contact_persistence_steps=1,
        seed=args.seed,
    )
    env = S10MujocoVecEnv(backend, obs_spec=spec, device=args.device)
    # IsaacLab PLAY disables alpha randomization and uses the action term's
    # configured default. Fixing alpha also makes checkpoint comparisons fair.
    env._filter_alpha.fill_(args.filter_alpha)
    obs, _ = env.get_observations()

    reasons: Counter[str] = Counter()
    aligned_raw_yaw: list[float] = []
    aligned_cmd_yaw: list[float] = []
    aligned_cmd_delta: list[float] = []
    all_raw_yaw: list[float] = []
    all_cmd_yaw: list[float] = []
    saturated = 0
    aligned_saturated = 0
    aligned_flips = 0
    aligned_pairs = 0
    previous_cmd = np.zeros(args.num_envs, dtype=np.float64)
    previous_aligned = np.zeros(args.num_envs, dtype=bool)
    started = time.monotonic()

    try:
        with torch.inference_mode():
            while sum(reasons.values()) < args.episodes:
                actions = policy.act_inference(obs)
                raw_yaw = actions[:, 1].detach().cpu().numpy().astype(np.float64)
                heading_error = np.zeros(args.num_envs, dtype=np.float64)
                distance = np.zeros(args.num_envs, dtype=np.float64)
                for index, data in enumerate(backend.data):
                    delta = backend._goal_point(index)[:2] - data.qpos[:2]
                    distance[index] = np.linalg.norm(delta)
                    goal_heading = np.arctan2(delta[1], delta[0])
                    heading_error[index] = _wrap_pi(goal_heading - _yaw_from_wxyz(data.qpos[3:7]))
                aligned = (
                    (np.abs(heading_error) <= np.deg2rad(args.aligned_angle_deg))
                    & (distance >= args.min_goal_distance)
                )

                obs, _, dones, info = env.step(actions)
                cmd_yaw = backend.last_cmd[:, 2].astype(np.float64, copy=True)
                tanh_saturated = np.abs(np.tanh(raw_yaw)) >= args.saturation_threshold
                all_raw_yaw.extend(raw_yaw.tolist())
                all_cmd_yaw.extend(cmd_yaw.tolist())
                saturated += int(tanh_saturated.sum())

                aligned_indices = np.flatnonzero(aligned)
                aligned_raw_yaw.extend(raw_yaw[aligned_indices].tolist())
                aligned_cmd_yaw.extend(cmd_yaw[aligned_indices].tolist())
                aligned_saturated += int(tanh_saturated[aligned_indices].sum())
                paired = aligned & previous_aligned
                paired_indices = np.flatnonzero(paired)
                if paired_indices.size:
                    delta = np.abs(cmd_yaw[paired_indices] - previous_cmd[paired_indices])
                    aligned_cmd_delta.extend(delta.tolist())
                    flips = (
                        (cmd_yaw[paired_indices] * previous_cmd[paired_indices] < 0.0)
                        & (np.abs(cmd_yaw[paired_indices]) >= args.flip_threshold)
                        & (np.abs(previous_cmd[paired_indices]) >= args.flip_threshold)
                    )
                    aligned_flips += int(flips.sum())
                    aligned_pairs += int(paired_indices.size)

                done_indices = torch.nonzero(dones, as_tuple=False).flatten().cpu().numpy()
                for index in done_indices:
                    if sum(reasons.values()) >= args.episodes:
                        break
                    reasons[str(info["done_reason"][index])] += 1
                policy.memory_a.reset(dones, use_random_init=False)
                env._filter_alpha[dones].fill_(args.filter_alpha)
                previous_cmd[:] = cmd_yaw
                previous_aligned[:] = aligned
                previous_aligned[done_indices] = False
    finally:
        env.close()

    total_samples = max(len(all_raw_yaw), 1)
    aligned_samples = max(len(aligned_raw_yaw), 1)
    return {
        "checkpoint": str(checkpoint),
        "iteration": payload.get("iter"),
        "episodes": sum(reasons.values()),
        "reasons": dict(reasons),
        "filter_alpha": args.filter_alpha,
        "all": {
            "samples": len(all_raw_yaw),
            "mean_abs_raw_yaw": _safe_mean(np.abs(all_raw_yaw).tolist()),
            "mean_abs_cmd_yaw": _safe_mean(np.abs(all_cmd_yaw).tolist()),
            "tanh_saturation_rate": saturated / total_samples,
        },
        "aligned": {
            "angle_deg": args.aligned_angle_deg,
            "samples": len(aligned_raw_yaw),
            "mean_abs_raw_yaw": _safe_mean(np.abs(aligned_raw_yaw).tolist()),
            "mean_abs_cmd_yaw": _safe_mean(np.abs(aligned_cmd_yaw).tolist()),
            "mean_abs_cmd_delta": _safe_mean(aligned_cmd_delta),
            "tanh_saturation_rate": aligned_saturated / aligned_samples,
            "large_sign_flip_rate": aligned_flips / max(aligned_pairs, 1),
        },
        "wall_seconds": time.monotonic() - started,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoints", nargs="+", type=Path)
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--episodes", type=int, default=48)
    parser.add_argument("--max-episode-length", type=int, default=300)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260910)
    parser.add_argument("--terrain-seed", type=int, default=42)
    parser.add_argument("--surface-seed", type=int, default=20260905)
    parser.add_argument("--terrain-profile", default="stage5_lower_density_stairs")
    parser.add_argument("--contact-threshold", type=float, default=500.0)
    parser.add_argument("--filter-alpha", type=float, default=0.5)
    parser.add_argument("--aligned-angle-deg", type=float, default=10.0)
    parser.add_argument("--min-goal-distance", type=float, default=1.0)
    parser.add_argument("--saturation-threshold", type=float, default=0.95)
    parser.add_argument("--flip-threshold", type=float, default=0.2)
    args = parser.parse_args()
    if not 0.0 <= args.filter_alpha < 1.0:
        raise ValueError("filter alpha must be in [0, 1)")
    if args.num_envs < 1 or args.episodes < 1 or args.workers < 1:
        raise ValueError("environment, episode, and worker counts must be positive")
    for value in args.checkpoints:
        checkpoint = value.expanduser().resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        print("POLICY_COMMAND_STABILITY", json.dumps(evaluate(checkpoint, args)), flush=True)


if __name__ == "__main__":
    main()
