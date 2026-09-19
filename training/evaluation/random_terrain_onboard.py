"""Headless pure-onboard evaluation on the random SRU training terrain.

The sampled target is treated as a waypoint stored in the initial local
frame.  During an episode, the policy goal is rebuilt from the KISS/ESKF
odometry estimate and the stored target only. All actor proprioception and
LiDAR world-height features use onboard estimates; MuJoCo truth is used only
for initial route anchoring, physical simulation, terminal labels, and errors.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import replace
from pathlib import Path
import json
import time

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from deployment.evaluation.simulation_utils import (
    angular_goal_error,
    goal_body,
    perturb_history,
    perturb_scans,
)
from deployment.localization.local_odometry import DualLidarImuWheelEskfOdometry, LocalOdometryConfig
from training.evaluation.policy import build_policy
from sru_training.s10_mujoco_backend import DONE_COMPLETE, S10NativeMujocoBackend
from sru_training.s10_mujoco_env import S10MujocoVecEnv
from sru_training.s10_lidar_encoder import (
    build_sensor_frame_directions,
    gather_aux_at_min_distance,
    native_to_90,
    world_z_native,
)
from sru_training.s10_policy_config import observation_spec_from_policy_state


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ONNX = ROOT / "src/S10_sdk_deploy/policy/policy_official_20260828.onnx"
DEFAULT_ENCODER = ROOT / "checkpoints/lidar_encoder_random_terrain_ft/best.pt"


def _new_odometry(backend: S10NativeMujocoBackend, args: argparse.Namespace):
    if backend._visual_lidar_scans is None:
        raise RuntimeError("visual LiDAR scans are required for onboard evaluation")
    initial_pose = np.asarray(backend.data[0].qpos[:7], dtype=np.float64).copy()
    odometry = DualLidarImuWheelEskfOdometry(
        initial_pose,
        np.asarray(backend.data[0].sensordata[:4]),
        horizontal_samples=900,
        config=LocalOdometryConfig(
            wheel_radius=args.wheel_radius,
            voxel_size=args.voxel_size,
            vertical_stride=args.vertical_stride,
            horizontal_stride=args.horizontal_stride,
            icp_threads=args.icp_threads,
            adaptive_wheel_weighting=True,
            enable_motion_constraints=False,
            enable_zero_velocity_update=True,
            enable_point_coupling=True,
        ),
    )
    odometry.initialize_scan(backend._visual_lidar_scans)
    return odometry, initial_pose


def _prepare_onboard_state(
    state,
    odometry: DualLidarImuWheelEskfOdometry,
    target: np.ndarray,
    gyro_body: np.ndarray,
    scans: tuple[np.ndarray, np.ndarray],
    sensor_dirs: np.ndarray,
    encoder,
):
    # Match ros2_navigation._control: ESKF velocity is stored in the initial
    # frame, then rotated into map and finally into the current body frame.
    velocity_map = odometry.initial_rotation_wb @ odometry.filter.velocity
    velocity_body = odometry.rotation_wb.T @ velocity_map
    gravity_body = odometry.rotation_wb.T @ np.asarray((0.0, 0.0, -1.0))
    q_xyzw = Rotation.from_matrix(odometry.rotation_wb).as_quat()
    estimated_pose = np.concatenate((odometry.position_w, q_xyzw[[3, 0, 1, 2]]))
    front, rear = scans
    front_d, _ = native_to_90(front)
    rear_d, _ = native_to_90(rear)
    front_z = gather_aux_at_min_distance(
        world_z_native(front, estimated_pose, front=True, sensor_dirs=sensor_dirs), front
    )
    rear_z = gather_aux_at_min_distance(
        world_z_native(rear, estimated_pose, front=False, sensor_dirs=sensor_dirs), rear
    )
    latent = encoder.encode_maps(front_d, rear_d, front_z, rear_z)
    for name, value in (
        ("velocity", velocity_body),
        ("gyro", gyro_body),
        ("gravity", gravity_body),
        ("goal", odometry.goal_body(target)),
    ):
        if not np.isfinite(value).all():
            raise RuntimeError(f"nonfinite onboard actor input: {name}")

    def tensor(value: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(value, dtype=torch.float32, device=state.base_lin_vel.device).unsqueeze(0)

    return replace(
        state,
        base_lin_vel=tensor(velocity_body),
        base_ang_vel=tensor(gyro_body),
        projected_gravity=tensor(gravity_body),
        goal_body=tensor(odometry.goal_body(target)),
        lidar_latent=latent,
    )


def _summarize(values: list[float]) -> dict[str, float | int | None]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "samples": int(len(array)),
        "mean": float(array.mean()) if len(array) else None,
        "p50": float(np.percentile(array, 50)) if len(array) else None,
        "p95": float(np.percentile(array, 95)) if len(array) else None,
        "max": float(array.max()) if len(array) else None,
        "last_valid": float(array[-1]) if len(array) else None,
    }


def evaluate(args: argparse.Namespace) -> dict[str, object]:
    checkpoint = args.checkpoint.expanduser().resolve()
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state_dict = payload["model_state_dict"]
    spec = observation_spec_from_policy_state(state_dict)
    policy = build_policy(state_dict, args.device)
    policy.eval()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    backend = S10NativeMujocoBackend(
        num_envs=1,
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
        sensor_workers=1,
        physics_workers=1,
        record_lidar_for_visualization=True,
        record_imu_history=True,
        imu_sample_hz=200.0,
        seed=args.seed,
    )
    env = S10MujocoVecEnv(backend, obs_spec=spec, device=args.device)
    reasons: Counter[str] = Counter()
    records: list[dict[str, object]] = []
    all_xy_errors: list[float] = []
    all_3d_errors: list[float] = []
    all_goal_errors: list[float] = []
    all_velocity_errors: list[float] = []
    sensor_rng = np.random.default_rng(args.seed + 1009)
    tile_rng = np.random.default_rng(args.seed + 2017)
    sensor_dirs = build_sensor_frame_directions(900)
    assert backend.terrain_atlas is not None
    tile_count = (
        backend.terrain_atlas.config.num_rows
        * backend.terrain_atlas.config.num_cols
    )
    tile_stream: list[int] = []
    started = time.monotonic()

    def next_tile_index() -> int:
        nonlocal tile_stream
        if not tile_stream:
            tile_stream = tile_rng.permutation(tile_count).tolist()
        return int(tile_stream.pop())

    def initialize_episode():
        tile_index = next_tile_index()
        row, col = divmod(tile_index, backend.terrain_atlas.config.num_cols)
        state = backend.set_environment_terrain_tile(0, row=row, col=col)
        env._last_state = state
        target = backend.random_goal_positions[0].copy()
        odometry, initial_pose = _new_odometry(backend, args)
        env._last_state = _prepare_onboard_state(
            state, odometry, target,
            np.asarray(backend.data[0].sensordata[7:10], dtype=np.float64),
            backend._visual_lidar_scans, sensor_dirs, backend.lidar_encoder,
        )
        env.episode_length_buf.zero_()
        env._filtered_cmd.zero_()
        env._filter_alpha.fill_(args.filter_alpha)
        env._policy_bias.zero_()
        env._policy_scale[:, 0] = env.action_spec.policy_scale_vx
        env._policy_scale[:, 1] = env.action_spec.policy_scale_yaw
        obs, _ = env.get_observations()
        tile = backend.terrain_atlas.tile(row, col)
        return obs, odometry, target, initial_pose, {
            "tile_index": tile_index,
            "terrain_row": row,
            "terrain_col": col,
            "terrain_type": tile.terrain_type,
            "terrain_difficulty": float(tile.difficulty),
        }

    obs, odometry, target, initial_pose, terrain = initialize_episode()
    episode_xy_errors: list[float] = []
    episode_3d_errors: list[float] = []
    episode_goal_errors: list[float] = []
    episode_velocity_errors: list[float] = []
    episode_started = time.monotonic()
    try:
        with torch.inference_mode():
            while sum(reasons.values()) < args.episodes:
                action = policy.act_inference(obs)
                obs, _, dones, info = env.step(action)
                if bool(dones[0].item()):
                    reason = str(info["done_reason"][0])
                    reasons[reason] += 1
                    terminal_pose = np.asarray(info["terminal_base_position"][0])
                    records.append({
                        "episode": sum(reasons.values()),
                        "reason": reason,
                        "success": reason == DONE_COMPLETE,
                        "steps": int(info["terminal_episode_steps"][0]),
                        **terrain,
                        "terminal_goal_distance_xy_m": float(
                            info["terminal_goal_distance_xy"][0]
                        ),
                        "terminal_base_position": terminal_pose.tolist(),
                        "start_position": initial_pose[:3].tolist(),
                        "position_error_xy_m": _summarize(episode_xy_errors),
                        "position_error_3d_m": _summarize(episode_3d_errors),
                        "goal_vector_error_deg": _summarize(episode_goal_errors),
                        "body_velocity_error_mps": _summarize(episode_velocity_errors),
                        "wall_seconds": time.monotonic() - episode_started,
                    })
                    policy.memory_a.reset(dones, use_random_init=False)
                    if sum(reasons.values()) >= args.episodes:
                        break
                    obs, odometry, target, initial_pose, terrain = initialize_episode()
                    episode_xy_errors = []
                    episode_3d_errors = []
                    episode_goal_errors = []
                    episode_velocity_errors = []
                    episode_started = time.monotonic()
                    continue

                assert backend._visual_lidar_scans is not None
                scans = perturb_scans(
                    backend._visual_lidar_scans, sensor_rng,
                    range_noise_std=0.0, dropout_rate=0.0,
                )
                history = perturb_history(
                    backend.imu_history(), sensor_rng,
                    accel_bias=np.zeros(3), gyro_bias=np.zeros(3),
                    accel_noise_std=0.0, gyro_noise_std=0.0,
                    wheel_scale_noise_std=0.0,
                )
                odometry.update(
                    scans, history,
                )
                state = env._last_state
                assert state is not None
                onboard = _prepare_onboard_state(
                    state, odometry, target, history["gyro"][-1],
                    scans, sensor_dirs, backend.lidar_encoder,
                )
                truth_pose = np.asarray(backend.data[0].qpos[:7], dtype=np.float64)
                error_vector = odometry.position_w - truth_pose[:3]
                xy_error = float(np.linalg.norm(error_vector[:2]))
                error_3d = float(np.linalg.norm(error_vector))
                goal_error = angular_goal_error(
                    odometry.goal_body(target),
                    goal_body(truth_pose, target),
                )
                episode_xy_errors.append(xy_error)
                episode_3d_errors.append(error_3d)
                episode_goal_errors.append(goal_error)
                all_xy_errors.append(xy_error)
                all_3d_errors.append(error_3d)
                all_goal_errors.append(goal_error)
                velocity_error = float(torch.linalg.vector_norm(
                    onboard.base_lin_vel - state.base_lin_vel
                ).item())
                episode_velocity_errors.append(velocity_error)
                all_velocity_errors.append(velocity_error)
                env._last_state = onboard
                obs, _ = env.get_observations()
    finally:
        env.close()

    total = sum(reasons.values())
    result = {
        "checkpoint": str(checkpoint),
        "checkpoint_iteration": payload.get("iter"),
        "episodes": total,
        "successes": reasons[DONE_COMPLETE],
        "success_rate": reasons[DONE_COMPLETE] / max(total, 1),
        "reasons": dict(reasons),
        "terrain_profile": args.terrain_profile,
        "terrain_sampling": "random_without_replacement_across_6x30_atlas",
        "goal_source": "stored_initial_local_target_plus_kiss_eskf",
        "actor_input_sources": {
            "base_lin_vel": "kiss_eskf_velocity_rotated_into_body_frame",
            "base_ang_vel": "latest_simulated_body_imu_gyro",
            "projected_gravity": "kiss_eskf_rotation",
            "goal_body": "stored_initial_target_plus_kiss_eskf_pose",
            "lidar_latent": "simulated_range_scans_plus_kiss_eskf_world_height",
            "last_action": "policy_history",
        },
        "simulation_assumptions": (
            "exact initial route-frame anchor; ideal synchronized simulated sensors "
            "without calibration errors, missing points, latency, or injected noise; "
            "MuJoCo truth is used only for physics, reset, terminal labels, and diagnostics"
        ),
        "action_filter_alpha": args.filter_alpha,
        "coordinate_error_definition": (
            "KISS/ESKF estimate minus MuJoCo truth on non-terminal control steps; "
            "last_valid is the final sample before backend auto-reset"
        ),
        "position_error_xy_m": _summarize(all_xy_errors),
        "position_error_3d_m": _summarize(all_3d_errors),
        "goal_vector_error_deg": _summarize(all_goal_errors),
        "body_velocity_error_mps": _summarize(all_velocity_errors),
        "official_low_level": str(DEFAULT_ONNX),
        "lidar_encoder": str(DEFAULT_ENCODER),
        "records": records,
        "wall_seconds": time.monotonic() - started,
    }
    print("RANDOM_TERRAIN_ONBOARD_EVAL", json.dumps(result, ensure_ascii=False), flush=True)
    if args.json_out:
        output = args.json_out.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--episodes", type=int, default=25)
    parser.add_argument("--max-episode-length", type=int, default=300)
    parser.add_argument("--num-envs", type=int, default=1, help="Reserved for interface symmetry; must be 1.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20260907)
    parser.add_argument("--terrain-seed", type=int, default=42)
    parser.add_argument("--surface-seed", type=int, default=20260905)
    parser.add_argument(
        "--terrain-profile",
        choices=("stage1_flat", "stage2_low_density_obstacles", "stage3_reduced_height", "stage4_full_no_pits", "stage5_lower_density_stairs"),
        default="stage4_full_no_pits",
    )
    parser.add_argument("--wheel-radius", type=float, default=0.0825)
    parser.add_argument("--voxel-size", type=float, default=0.10)
    parser.add_argument("--vertical-stride", type=int, default=2)
    parser.add_argument("--horizontal-stride", type=int, default=6)
    parser.add_argument("--icp-threads", type=int, default=4)
    parser.add_argument(
        "--filter-alpha",
        type=float,
        default=0.5,
        help="fixed high-level command filter; 0.5 matches SRU PLAY and deployment",
    )
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()
    if args.num_envs != 1 or args.episodes < 1 or args.max_episode_length < 1:
        raise ValueError("pure onboard random-terrain evaluation requires one env and positive limits")
    if not 0.0 <= args.filter_alpha < 1.0:
        raise ValueError("filter alpha must be in [0, 1)")
    evaluate(args)


if __name__ == "__main__":
    main()
