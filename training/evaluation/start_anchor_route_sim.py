"""End-to-end simulation of route collection and onboard replay.

The ground-truth policy driver and the onboard waypoint recorder are kept as
separate data paths. Three collection passes start from perturbed poses and
share one LiDAR start anchor. A final pass replays the robustly fused route
using only onboard actor inputs and onboard waypoint switching.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import itertools
import json
import math
from pathlib import Path
import time

import mujoco
import mujoco.viewer
import numpy as np
import torch
from scipy.ndimage import binary_erosion, label
from scipy.spatial.transform import Rotation

from deployment.localization.local_odometry import DualLidarImuWheelEskfOdometry, LocalOdometryConfig
from deployment.common.math_utils import quat_wxyz_to_rotmat
from deployment.navigation.core import estimate_support_height_map
from deployment.localization.start_alignment import (
    StartAlignmentConfig,
    StartAnchor,
    StartAnchorAccumulator,
    align_start_anchor,
    compose_aligned_initial_pose,
    save_start_anchor,
)
from deployment.waypoints.collection import (
    CollectionLimits,
    CollectionRejected,
    PoseSample,
    WaypointCollectionSession,
)
from sru_training.s10_lidar_encoder import (
    INVALID_RANGE_THRESHOLD_M,
    MIN_RANGE_M,
    S10_FRONT_POS,
    S10_FRONT_ROT_WXYZ,
    S10_REAR_POS,
    S10_REAR_ROT_WXYZ,
    build_sensor_frame_directions,
)
from sru_training.s10_mujoco_backend import (
    DONE_COMPLETE,
    DONE_NONE,
    STANDING_BASE_CLEARANCE,
    S10NativeMujocoBackend,
)
from sru_training.s10_mujoco_env import S10MujocoVecEnv
from sru_training.s10_policy_config import observation_spec_from_policy_state
from sru_training.s10_viewer_overlay import update_perception_overlay
from training.evaluation.policy import build_policy
from training.evaluation.random_terrain_onboard import _prepare_onboard_state
from training.terrains.atlas import mask_index_to_local_xy
from training.terrains.constants import VERTICAL_SCALE


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHECKPOINT = (
    ROOT / "checkpoints/navigation/sru_deploy_model_2750.pt"
)
DEFAULT_ONNX = ROOT / "src/S10_sdk_deploy/policy/policy_official_20260828.onnx"
DEFAULT_ENCODER = ROOT / "checkpoints/lidar_encoder_random_terrain_ft/best.pt"


@dataclass(frozen=True)
class ReferenceFrame:
    body_position_world: np.ndarray
    rotation_world_reference: np.ndarray
    route_origin_world: np.ndarray
    anchor: StartAnchor


@dataclass
class RuntimeState:
    odometry: DualLidarImuWheelEskfOdometry
    sim_time_s: float


def _rotation_z(yaw: float) -> np.ndarray:
    cosine, sine = np.cos(yaw), np.sin(yaw)
    return np.asarray(
        ((cosine, -sine, 0.0), (sine, cosine, 0.0), (0.0, 0.0, 1.0)),
        dtype=np.float64,
    )


def _quaternion_wxyz(rotation: np.ndarray) -> np.ndarray:
    xyzw = Rotation.from_matrix(rotation).as_quat()
    return xyzw[[3, 0, 1, 2]]


def _yaw_from_wxyz(quaternion: np.ndarray) -> float:
    rotation = quat_wxyz_to_rotmat(quaternion)
    return float(np.arctan2(rotation[1, 0], rotation[0, 0]))


def _wrap_degrees(value: float) -> float:
    return float((value + 180.0) % 360.0 - 180.0)


def _scan_points(scans: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    directions = build_sensor_frame_directions(900)
    points = []
    for scan, sensor_position, sensor_quaternion in (
        (scans[0], S10_FRONT_POS, S10_FRONT_ROT_WXYZ),
        (scans[1], S10_REAR_POS, S10_REAR_ROT_WXYZ),
    ):
        values = np.asarray(scan, dtype=np.float64)
        valid = (values > MIN_RANGE_M) & (values < INVALID_RANGE_THRESHOLD_M)
        rays_body = np.einsum(
            "ij,hwj->hwi",
            quat_wxyz_to_rotmat(sensor_quaternion),
            directions,
        )
        hits = np.asarray(sensor_position, dtype=np.float64) + values[..., None] * rays_body
        points.append(hits[valid])
    return np.ascontiguousarray(np.concatenate(points, axis=0))


def _route_position(reference: ReferenceFrame, world_position: np.ndarray) -> np.ndarray:
    return reference.rotation_world_reference.T @ (
        np.asarray(world_position, dtype=np.float64) - reference.route_origin_world
    )


def _world_position(reference: ReferenceFrame, route_position: np.ndarray) -> np.ndarray:
    return reference.route_origin_world + reference.rotation_world_reference @ np.asarray(
        route_position, dtype=np.float64
    )


def _terrain_point(tile, index: np.ndarray) -> np.ndarray:
    x, y = mask_index_to_local_xy(int(index[0]), int(index[1]))
    z = float(tile.height_inner[int(index[0]), int(index[1])]) * VERTICAL_SCALE
    return np.asarray((tile.origin[0] + x, tile.origin[1] + y, z), dtype=np.float64)


def _points_from_indices(tile, indices: np.ndarray) -> np.ndarray:
    return np.stack([_terrain_point(tile, item) for item in indices])


def _sample_route(
    tile,
    rng: np.random.Generator,
    count: int,
    *,
    min_spacing_m: float = 2.8,
    max_spacing_m: float = 5.2,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    spawn_indices = np.argwhere(tile.spawn_mask)
    valid_indices = np.argwhere(tile.valid_mask)
    center = tile.origin[:2]
    spawn_world = _points_from_indices(tile, spawn_indices)
    start_pool = np.flatnonzero(np.linalg.norm(spawn_world[:, :2] - center, axis=1) <= 6.0)
    if not len(start_pool):
        start_pool = np.arange(len(spawn_indices))
    start = spawn_world[int(rng.choice(start_pool))]

    valid_world = _points_from_indices(tile, valid_indices)
    route = []
    previous = start
    for _ in range(count):
        distances = np.linalg.norm(valid_world[:, :2] - previous[:2], axis=1)
        mask = (distances >= min_spacing_m) & (distances <= max_spacing_m)
        if route:
            prior = np.stack(route)
            separation = np.min(
                np.linalg.norm(valid_world[:, None, :2] - prior[None, :, :2], axis=2), axis=1
            )
            mask &= separation >= 1.2
        candidates = np.flatnonzero(mask)
        if not len(candidates):
            raise RuntimeError("could not sample a distance-constrained 10-point route")
        previous = valid_world[int(rng.choice(candidates))]
        route.append(previous.copy())
    return start, np.stack(route), ["valid_surface"] * count


def _sample_elevated_route(
    tile,
    rng: np.random.Generator,
    count: int,
    *,
    min_spacing_m: float,
    max_spacing_m: float,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Build approach/top/exit triplets for every elevated structure."""

    if count != 15:
        raise ValueError("elevated route profile currently defines exactly 15 waypoints")
    ground_mask = (
        tile.valid_mask
        & ~tile.platform_mask
        & (tile.height_inner == 0)
    )
    ground_mask = binary_erosion(ground_mask, iterations=5, border_value=0)
    ground_indices = np.argwhere(ground_mask)
    if not len(ground_indices):
        raise RuntimeError("terrain tile has no edge-safe ground targets")
    ground_pool = _points_from_indices(tile, ground_indices)

    component_labels, component_count = label(tile.platform_mask)
    features = []
    for component in range(1, component_count + 1):
        raw_indices = np.argwhere(component_labels == component)
        component_mask = binary_erosion(
            component_labels == component, iterations=5, border_value=0
        )
        indices = np.argwhere(component_mask)
        if not len(indices):
            continue
        lower = raw_indices.min(axis=0)
        upper = raw_indices.max(axis=0)
        safe_y = indices[:, 1]
        top_index = np.asarray(
            (
                int((lower[0] + upper[0]) // 2),
                int(safe_y[int(rng.integers(len(safe_y)))]),
            ),
            dtype=np.int64,
        )
        top = _terrain_point(tile, top_index)
        side_points = []
        for side_x in (int(lower[0]) - 35, int(upper[0]) + 35):
            desired_index = np.asarray(
                (np.clip(side_x, 0, tile.height_inner.shape[0] - 1), top_index[1]),
                dtype=np.int64,
            )
            desired = _terrain_point(tile, desired_index)
            distances = np.linalg.norm(
                ground_pool[:, :2] - desired[None, :2], axis=1
            )
            side_points.append(ground_pool[int(np.argmin(distances))])
        kind = "stair" if float(top[2]) >= 0.35 else "platform"
        features.append((kind, top, tuple(side_points)))
    if sum(feature[0] == "stair" for feature in features) < 2 or sum(
        feature[0] == "platform" for feature in features
    ) < 3:
        raise RuntimeError(
            "elevated route needs at least two stair tops and three low platforms"
        )

    best = None
    for order in itertools.permutations(range(len(features))):
        for orientations in itertools.product((0, 1), repeat=len(features)):
            route = []
            categories = []
            for feature_index, orientation in zip(order, orientations):
                kind, top, sides = features[feature_index]
                route.extend((sides[orientation], top, sides[1 - orientation]))
                categories.extend((f"{kind}_approach", f"{kind}_top", f"{kind}_exit"))
            positions = np.stack(route)
            distances = np.linalg.norm(np.diff(positions[:, :2], axis=0), axis=1)
            violations = np.maximum(min_spacing_m - distances, 0.0).sum()
            violations += np.maximum(distances - max_spacing_m, 0.0).sum()
            score = (float(violations), float(np.max(distances)), float(np.sum(distances)))
            if best is None or score < best[0]:
                best = (score, positions, categories)
    assert best is not None
    if best[0][0] > 1.0e-9:
        raise RuntimeError(
            "elevated structures cannot satisfy the requested spacing bounds; "
            f"best violation={best[0][0]:.3f}m"
        )

    route = best[1]
    categories = best[2]
    spawn_indices = np.argwhere(tile.spawn_mask & (tile.height_inner == 0))
    spawn_world = _points_from_indices(tile, spawn_indices)
    first_distances = np.linalg.norm(spawn_world[:, :2] - route[0, :2], axis=1)
    start_candidates = np.flatnonzero(
        (first_distances >= min_spacing_m) & (first_distances <= max_spacing_m)
    )
    if not len(start_candidates):
        raise RuntimeError("no valid spawn satisfies elevated-route initial spacing")
    midpoint = 0.5 * (min_spacing_m + max_spacing_m)
    mismatch = np.abs(first_distances[start_candidates] - midpoint)
    near_midpoint = start_candidates[np.argsort(mismatch)[: min(100, len(mismatch))]]
    start = spawn_world[int(rng.choice(near_midpoint))]
    return start, route, categories


def _place_robot(
    backend: S10NativeMujocoBackend,
    position_xy: np.ndarray,
    yaw: float,
    *,
    fallback_surface_z: float,
) -> None:
    data = backend.data[0]
    surface_z = backend._terrain_z(
        data,
        np.asarray(position_xy, dtype=np.float64),
        ray_start_z=fallback_surface_z + 2.0,
        fallback_height=fallback_surface_z,
    )
    data.qpos[:] = backend.model.qpos0
    # The generated terrain joints are static and must retain the template values.
    from sru_training.s10_mujoco_backend import DOF, ROOT_QPOS

    data.qpos[ROOT_QPOS + DOF:] = backend.static_qpos
    data.qvel[:] = 0.0
    data.qpos[:3] = (
        float(position_xy[0]),
        float(position_xy[1]),
        surface_z + 0.05 + STANDING_BASE_CLEARANCE,
    )
    data.qpos[3:7] = (np.cos(yaw * 0.5), 0.0, 0.0, np.sin(yaw * 0.5))
    data.qpos[7:7 + DOF] = backend.joint_target
    data.ctrl[:] = 0.0
    mujoco.mj_forward(backend.model, data)
    if backend.onnx_controller is not None:
        backend.onnx_controller.reset(backend.num_envs, np.asarray((0,), dtype=np.int64))
    backend._settle_reset_one(0)
    backend.episode_steps[0] = 0
    backend.target_steps[0] = 0
    backend._goal_was_reached[0] = False
    backend._goal_hold_steps[0] = 0
    backend._illegal_contact_steps[0] = 0
    backend.last_done_reason[0] = DONE_NONE
    backend.last_cmd[0] = 0.0
    backend.last_action[0].zero_()
    backend._previous_high_level_action[0] = 0.0
    backend._finalize_reset_state(0)
    backend._observe()


def _odometry_config(args: argparse.Namespace) -> LocalOdometryConfig:
    return LocalOdometryConfig(
        wheel_radius=args.wheel_radius,
        voxel_size=args.odometry_voxel_size,
        vertical_stride=2,
        horizontal_stride=6,
        icp_threads=args.icp_threads,
        adaptive_wheel_weighting=True,
        enable_motion_constraints=False,
        enable_zero_velocity_update=True,
        enable_point_coupling=True,
    )


def _make_reference(
    backend: S10NativeMujocoBackend,
    start_world: np.ndarray,
    yaw: float,
    alignment_config: StartAlignmentConfig,
) -> ReferenceFrame:
    _place_robot(
        backend, start_world[:2], yaw, fallback_surface_z=float(start_world[2])
    )
    data = backend.data[0]
    state = backend._observe()
    del state
    if backend._visual_lidar_scans is None:
        raise RuntimeError("reference anchor has no simulated LiDAR scans")
    accumulator = StartAnchorAccumulator(alignment_config)
    points = _scan_points(backend._visual_lidar_scans)
    for _ in range(alignment_config.capture_frames):
        accumulator.add(points, stationary=True)
    body_position = np.asarray(data.qpos[:3], dtype=np.float64).copy()
    rotation = quat_wxyz_to_rotmat(np.asarray(data.qpos[3:7], dtype=np.float64))
    route_origin = body_position.copy()
    route_origin[2] -= STANDING_BASE_CLEARANCE
    return ReferenceFrame(
        body_position_world=body_position,
        rotation_world_reference=rotation,
        route_origin_world=route_origin,
        anchor=StartAnchor(
            accumulator.merged(),
            np.asarray(data.sensordata[:4], dtype=np.float64).copy(),
            alignment_config.capture_frames,
            alignment_config.voxel_size_m,
        ),
    )


def _initialize_localizer(
    backend: S10NativeMujocoBackend,
    reference: ReferenceFrame,
    alignment_config: StartAlignmentConfig,
    args: argparse.Namespace,
) -> tuple[RuntimeState, dict[str, float]]:
    if backend._visual_lidar_scans is None:
        backend._observe()
    assert backend._visual_lidar_scans is not None
    accumulator = StartAnchorAccumulator(alignment_config)
    live_points = _scan_points(backend._visual_lidar_scans)
    for _ in range(alignment_config.capture_frames):
        accumulator.add(live_points, stationary=True)
    alignment = align_start_anchor(reference.anchor, accumulator.merged(), alignment_config)
    reference_pose = np.asarray(
        (0.0, 0.0, STANDING_BASE_CLEARANCE, 1.0, 0.0, 0.0, 0.0),
        dtype=np.float64,
    )
    initial_position, initial_rotation = compose_aligned_initial_pose(reference_pose, alignment)
    initial_pose = np.concatenate((initial_position, _quaternion_wxyz(initial_rotation)))
    initial_imu = np.asarray(backend.data[0].sensordata[:4], dtype=np.float64).copy()
    odometry = DualLidarImuWheelEskfOdometry(
        initial_pose,
        initial_imu,
        horizontal_samples=900,
        config=_odometry_config(args),
    )
    odometry.initialize_scan(backend._visual_lidar_scans)

    truth_translation = reference.rotation_world_reference.T @ (
        np.asarray(backend.data[0].qpos[:3], dtype=np.float64)
        - reference.body_position_world
    )
    truth_rotation = reference.rotation_world_reference.T @ quat_wxyz_to_rotmat(
        backend.data[0].qpos[3:7]
    )
    truth_yaw = np.degrees(np.arctan2(truth_rotation[1, 0], truth_rotation[0, 0]))
    return RuntimeState(odometry, 0.0), {
        "estimated_yaw_deg": alignment.yaw_deg,
        "truth_yaw_deg": float(truth_yaw),
        "yaw_error_deg": _wrap_degrees(alignment.yaw_deg - truth_yaw),
        "estimated_translation_m": alignment.translation_reference_live_m.tolist(),
        "truth_translation_m": truth_translation.tolist(),
        "translation_error_m": float(
            np.linalg.norm(alignment.translation_reference_live_m - truth_translation)
        ),
        "rmse_m": alignment.rmse_m,
        "overlap": alignment.overlap_fraction,
    }


def _update_localizer(
    backend: S10NativeMujocoBackend,
    runtime: RuntimeState,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    scans = backend._visual_lidar_scans
    if scans is None:
        raise RuntimeError("simulated LiDAR scans were not recorded")
    history = backend.imu_history()
    if len(history["time"]) < 2:
        raise RuntimeError("simulated high-rate IMU history is incomplete")
    runtime.odometry.update(scans, history)
    runtime.sim_time_s += 1.0 / backend.action_spec.policy_hz
    return _scan_points(scans), history


def _append_pose_sample(
    session: WaypointCollectionSession,
    runtime: RuntimeState,
    points_body: np.ndarray,
    gyro: np.ndarray,
) -> None:
    odometry = runtime.odometry
    velocity_route = odometry.initial_rotation_wb @ odometry.filter.velocity
    velocity_body = odometry.rotation_wb.T @ velocity_route
    support_height, _, _ = estimate_support_height_map(
        points_body,
        odometry.position_w,
        odometry.rotation_wb,
        expected_clearance_m=STANDING_BASE_CLEARANCE,
    )
    diagnostics = {
        "covariance_trace": odometry.covariance_trace,
        "icp_accepted": odometry.icp_update_accepted,
    }
    if np.isfinite(support_height):
        diagnostics["support_height_map_m"] = float(support_height)
    session.append(PoseSample(
        stamp_s=runtime.sim_time_s,
        receipt_s=runtime.sim_time_s,
        position=odometry.position_w.copy(),
        quaternion_wxyz=_quaternion_wxyz(odometry.rotation_wb),
        linear_velocity=velocity_body,
        angular_velocity=np.asarray(gyro, dtype=np.float64).copy(),
        healthy=True,
        quality_state="GOOD",
        diagnostics=diagnostics,
    ))


def _configure_goal(backend: S10NativeMujocoBackend, target_world: np.ndarray) -> None:
    backend.random_goal_positions[0] = np.asarray(target_world, dtype=np.float64)
    backend.target_steps[0] = 0
    backend._goal_was_reached[0] = False
    backend._goal_hold_steps[0] = 0
    backend._previous_goal_distance_xy[0] = backend._distance_to_goal(0, flat=True)


def _reset_policy(policy, env: S10MujocoVecEnv) -> None:
    done = torch.ones(1, dtype=torch.bool, device=env.device)
    with torch.inference_mode():
        policy.memory_a.reset(done, use_random_init=False)
    env._filtered_cmd.zero_()
    env._last_state = env.backend._observe()


def _policy_step(
    policy,
    env: S10MujocoVecEnv,
    obs: torch.Tensor,
) -> tuple[torch.Tensor, np.ndarray, bool, dict, object]:
    with torch.inference_mode():
        action = policy.act_inference(obs)
    command = env._process_actions(action)
    env.backend.set_policy_actions(action)
    state, _, dones, info = env.backend.step(command)
    env._last_state = state
    return action, command[0].detach().cpu().numpy(), bool(dones[0]), info, state


def _zero_step(
    env: S10MujocoVecEnv,
) -> tuple[bool, dict, object]:
    env._filtered_cmd.zero_()
    zero_action = torch.zeros((1, 2), dtype=torch.float32, device=env.device)
    env.backend.set_policy_actions(zero_action)
    state, _, dones, info = env.backend.step(
        torch.zeros((1, 3), dtype=torch.float32, device=env.device)
    )
    env._last_state = state
    return bool(dones[0]), info, state


def _collection_pass(
    pass_index: int,
    perturbation: tuple[float, float, float],
    backend: S10NativeMujocoBackend,
    env: S10MujocoVecEnv,
    policy,
    reference: ReferenceFrame,
    targets_world: np.ndarray,
    output_dir: Path,
    alignment_config: StartAlignmentConfig,
    args: argparse.Namespace,
) -> dict[str, object]:
    dx, dy, yaw_deg = perturbation
    reference_yaw = np.arctan2(
        reference.rotation_world_reference[1, 0],
        reference.rotation_world_reference[0, 0],
    )
    start_xy = reference.body_position_world[:2] + reference.rotation_world_reference[:2, :2] @ np.asarray((dx, dy))
    _place_robot(
        backend,
        start_xy,
        reference_yaw + np.deg2rad(yaw_deg),
        fallback_surface_z=float(reference.route_origin_world[2]),
    )
    runtime, alignment_stats = _initialize_localizer(
        backend, reference, alignment_config, args
    )
    route_path = output_dir / f"collection_pass_{pass_index}.yaml"
    for path in (route_path, route_path.with_suffix(".quality.json"), route_path.with_suffix(".anchor.npz")):
        path.unlink(missing_ok=True)
    save_start_anchor(
        route_path.with_suffix(".anchor.npz"),
        reference.anchor.points_reference_body_m,
        reference.anchor.initial_imu_quaternion_wxyz,
        frame_count=reference.anchor.frame_count,
        voxel_size_m=reference.anchor.voxel_size_m,
    )
    session = WaypointCollectionSession(
        route_path,
        limits=CollectionLimits(
            window_s=1.0,
            min_samples=5,
            max_linear_speed_mps=0.10,
            max_angular_speed_rps=0.15,
            min_waypoint_spacing_m=0.20,
        ),
        description="simulated independent ground-truth drive / onboard collection",
    )
    _reset_policy(policy, env)
    records = []
    failure = None
    obs, _ = env.get_observations()
    for target_index, target_world in enumerate(targets_world):
        _configure_goal(backend, target_world)
        env._last_state = backend._observe()
        obs, _ = env.get_observations()
        reached = False
        travel_steps = 0
        for travel_steps in range(1, args.max_steps_per_waypoint + 1):
            _, _, done, info, _ = _policy_step(policy, env, obs)
            if done:
                failure = {
                    "target_index": target_index,
                    "reason": str(info["done_reason"][0]),
                    "travel_steps": travel_steps,
                }
                break
            points, history = _update_localizer(backend, runtime)
            _append_pose_sample(session, runtime, points, history["gyro"][-1])
            distance = float(np.linalg.norm(backend.data[0].qpos[:2] - target_world[:2]))
            if distance <= args.collection_reach_m:
                reached = True
                break
            obs, _ = env.get_observations()
        if failure is not None:
            break
        if not reached:
            failure = {
                "target_index": target_index,
                "reason": "waypoint_timeout",
                "travel_steps": travel_steps,
            }
            break

        marked = None
        last_rejection = None
        hold_steps = 0
        for hold_steps in range(1, args.max_collection_hold_steps + 1):
            done, info, _ = _zero_step(env)
            if done:
                failure = {
                    "target_index": target_index,
                    "reason": str(info["done_reason"][0]),
                    "phase": "collection_hold",
                }
                break
            points, history = _update_localizer(backend, runtime)
            _append_pose_sample(session, runtime, points, history["gyro"][-1])
            try:
                marked = session.mark(
                    name=f"waypoint_{target_index:02d}",
                    tags=["simulated"],
                    now_s=runtime.sim_time_s,
                )
                break
            except CollectionRejected as error:
                last_rejection = str(error)
                continue
        if failure is not None:
            break
        if marked is None:
            failure = {
                "target_index": target_index,
                "reason": "collector_never_became_stationary",
                "hold_steps": hold_steps,
                "last_rejection": last_rejection,
                "truth_body_linear_speed_mps": float(
                    np.linalg.norm(backend.data[0].qvel[:3])
                ),
                "truth_body_angular_speed_rps": float(
                    np.linalg.norm(backend.data[0].qvel[3:6])
                ),
            }
            break

        recorded = np.asarray(session.nodes[-1]["position"], dtype=np.float64)
        truth_support_world = np.asarray(backend.data[0].qpos[:3], dtype=np.float64).copy()
        truth_support_world[2] -= STANDING_BASE_CLEARANCE
        truth_route = _route_position(reference, truth_support_world)
        intended_route = _route_position(reference, target_world)
        records.append({
            "target_index": target_index,
            "travel_steps": travel_steps,
            "hold_steps": hold_steps,
            "recorded_route_m": recorded.tolist(),
            "truth_stop_route_m": truth_route.tolist(),
            "intended_target_route_m": intended_route.tolist(),
            "localization_error_m": float(np.linalg.norm(recorded - truth_route)),
            "target_stop_error_xy_m": float(np.linalg.norm(truth_route[:2] - intended_route[:2])),
        })
        print(
            "COLLECTED_WAYPOINT",
            f"pass={pass_index}",
            f"index={target_index}",
            f"travel_steps={travel_steps}",
            f"hold_steps={hold_steps}",
            f"localization_error_m={records[-1]['localization_error_m']:.4f}",
            flush=True,
        )
        backend._goal_was_reached[0] = False
        backend._goal_hold_steps[0] = 0

    return {
        "pass": pass_index,
        "requested_start_perturbation": {"x_m": dx, "y_m": dy, "yaw_deg": yaw_deg},
        "alignment": alignment_stats,
        "collected": len(records),
        "success": failure is None and len(records) == len(targets_world),
        "failure": failure,
        "route_file": str(route_path),
        "records": records,
    }


def _onboard_replay(
    route_positions: np.ndarray,
    perturbation: tuple[float, float, float],
    backend: S10NativeMujocoBackend,
    env: S10MujocoVecEnv,
    policy,
    reference: ReferenceFrame,
    alignment_config: StartAlignmentConfig,
    args: argparse.Namespace,
    viewer=None,
) -> dict[str, object]:
    dx, dy, yaw_deg = perturbation
    reference_yaw = np.arctan2(
        reference.rotation_world_reference[1, 0],
        reference.rotation_world_reference[0, 0],
    )
    start_xy = reference.body_position_world[:2] + reference.rotation_world_reference[:2, :2] @ np.asarray((dx, dy))
    _place_robot(
        backend,
        start_xy,
        reference_yaw + np.deg2rad(yaw_deg),
        fallback_surface_z=float(reference.route_origin_world[2]),
    )
    runtime, alignment_stats = _initialize_localizer(
        backend, reference, alignment_config, args
    )
    _reset_policy(policy, env)
    sensor_dirs = build_sensor_frame_directions(900)
    records = []
    failure = None

    for target_index, target_route in enumerate(route_positions):
        target_world = _world_position(reference, target_route)
        _configure_goal(backend, target_world)
        state = backend._observe()
        assert backend._visual_lidar_scans is not None
        onboard_state = _prepare_onboard_state(
            state,
            runtime.odometry,
            target_route,
            np.asarray(backend.data[0].sensordata[7:10], dtype=np.float64),
            backend._visual_lidar_scans,
            sensor_dirs,
            backend.lidar_encoder,
        )
        env._last_state = onboard_state
        obs, _ = env.get_observations()
        reached = False
        for step in range(1, args.max_steps_per_waypoint + 1):
            if viewer is not None and not viewer.is_running():
                failure = {
                    "target_index": target_index,
                    "reason": "viewer_closed",
                    "steps": step,
                }
                break
            step_started = time.monotonic()
            _, _, done, info, state = _policy_step(policy, env, obs)
            if done:
                failure = {
                    "target_index": target_index,
                    "reason": str(info["done_reason"][0]),
                    "steps": step,
                }
                break
            _, history = _update_localizer(backend, runtime)
            support_position = runtime.odometry.position_w.copy()
            support_position[2] -= STANDING_BASE_CLEARANCE
            estimated_xy = float(np.linalg.norm(support_position[:2] - target_route[:2]))
            estimated_z = abs(float(support_position[2] - target_route[2]))
            if viewer is not None:
                front_points, rear_points = backend.lidar_hit_points()
                update_perception_overlay(
                    viewer, target_world, front_points, rear_points
                )
                viewer.sync()
                if step % 5 == 0:
                    truth_support = np.asarray(
                        backend.data[0].qpos[:3], dtype=np.float64
                    ).copy()
                    truth_support[2] -= STANDING_BASE_CLEARANCE
                    truth_route = _route_position(reference, truth_support)
                    position_error = float(
                        np.linalg.norm(runtime.odometry.position_w[:2] - truth_route[:2])
                    )
                    command = backend.last_cmd[0]
                    print(
                        f"\r[onboard GUI] target={target_index + 1}/{len(route_positions)} "
                        f"estimated_dist={estimated_xy:.2f}m "
                        f"position_error={position_error:.3f}m "
                        f"cmd=({command[0]:+.2f},{command[2]:+.2f})",
                        end="",
                        flush=True,
                    )
                remaining = (
                    1.0 / backend.action_spec.policy_hz / args.realtime_factor
                    - (time.monotonic() - step_started)
                )
                if remaining > 0.0:
                    time.sleep(remaining)
            if estimated_xy <= args.replay_reach_m and estimated_z <= 0.55:
                truth_support = np.asarray(backend.data[0].qpos[:3], dtype=np.float64).copy()
                truth_support[2] -= STANDING_BASE_CLEARANCE
                truth_route = _route_position(reference, truth_support)
                records.append({
                    "target_index": target_index,
                    "steps": step,
                    "estimated_reach_xy_m": estimated_xy,
                    "truth_reach_xy_m": float(np.linalg.norm(truth_route[:2] - target_route[:2])),
                    "position_error_xy_m": float(
                        np.linalg.norm(runtime.odometry.position_w[:2] - truth_route[:2])
                    ),
                })
                if viewer is not None:
                    print(
                        f"\n[onboard GUI] reached target {target_index + 1}/"
                        f"{len(route_positions)} truth_dist="
                        f"{records[-1]['truth_reach_xy_m']:.3f}m",
                        flush=True,
                    )
                reached = True
                break
            assert backend._visual_lidar_scans is not None
            onboard_state = _prepare_onboard_state(
                state,
                runtime.odometry,
                target_route,
                history["gyro"][-1],
                backend._visual_lidar_scans,
                sensor_dirs,
                backend.lidar_encoder,
            )
            env._last_state = onboard_state
            obs, _ = env.get_observations()
        if failure is not None:
            break
        if not reached:
            failure = {
                "target_index": target_index,
                "reason": "waypoint_timeout",
                "steps": args.max_steps_per_waypoint,
            }
            break
        backend._goal_was_reached[0] = False
        backend._goal_hold_steps[0] = 0

    return {
        "requested_start_perturbation": {"x_m": dx, "y_m": dy, "yaw_deg": yaw_deg},
        "alignment": alignment_stats,
        "reached": len(records),
        "waypoints": len(route_positions),
        "success": failure is None and len(records) == len(route_positions),
        "failure": failure,
        "records": records,
    }


def _summary(values: np.ndarray) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(array)),
        "p95": float(np.percentile(array, 95)),
        "max": float(np.max(array)),
    }


def evaluate(args: argparse.Namespace) -> dict[str, object]:
    replay_payload = None
    if args.replay_result is not None:
        replay_payload = json.loads(
            args.replay_result.expanduser().resolve().read_text(encoding="utf-8")
        )
        route_metadata = replay_payload.get("route", {})
        args.route_profile = route_metadata.get("profile", args.route_profile)
        args.route_seed = int(replay_payload.get("route_seed", args.route_seed))
        args.min_spacing_m = float(
            route_metadata.get("min_spacing_m", args.min_spacing_m)
        )
        args.max_spacing_m = float(
            route_metadata.get("max_spacing_m", args.max_spacing_m)
        )
        map_metadata = replay_payload.get("map", {})
        args.terrain_seed = int(map_metadata.get("terrain_seed", args.terrain_seed))
        args.surface_seed = int(map_metadata.get("surface_seed", args.surface_seed))
        args.row = int(map_metadata.get("row", args.row))
        args.col = int(map_metadata.get("col", args.col))
        comparison = replay_payload.get("comparison") or {}
        fused_route = comparison.get("fused_route_m")
        if fused_route:
            args.waypoints = len(fused_route)

    checkpoint = args.checkpoint.expanduser().resolve()
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state_dict = payload["model_state_dict"]
    spec = observation_spec_from_policy_state(state_dict)
    policy = build_policy(state_dict, args.device)
    backend = S10NativeMujocoBackend(
        num_envs=1,
        device=args.device,
        task_mode="random_goal_sru",
        terrain_profile="stage5_lower_density_stairs",
        terrain_seed=args.terrain_seed,
        surface_seed=args.surface_seed,
        grass_fraction=0.25,
        gravel_fraction=0.25,
        terrain_rows=2,
        terrain_cols=30,
        reset_mode="fixed",
        max_episode_length=args.max_total_steps,
        single_episode_length=args.max_total_steps,
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
        contact_threshold=500.0,
        contact_persistence_steps=1,
        seed=args.seed,
    )
    if args.gui:
        model_extent = float(backend.model.stat.extent)
        backend.model.vis.map.znear = min(
            float(backend.model.vis.map.znear), args.near_clip / model_extent
        )
    env = S10MujocoVecEnv(backend, obs_spec=spec, device=args.device)
    env._filter_alpha.fill_(0.5)
    backend.required_goal_hold_steps = args.max_total_steps * 2
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    alignment_config = StartAlignmentConfig(
        capture_frames=12,
        voxel_size_m=0.12,
        yaw_search_range_deg=45.0,
        yaw_search_step_deg=2.0,
        max_registration_points=3500,
    )
    started = time.monotonic()
    try:
        backend.set_environment_terrain_tile(0, row=args.row, col=args.col)
        tile = backend.terrain_atlas.tile(args.row, args.col)
        rng = np.random.default_rng(args.route_seed)
        if args.route_profile == "elevated":
            nominal_start, targets_world, route_categories = _sample_elevated_route(
                tile,
                rng,
                args.waypoints,
                min_spacing_m=args.min_spacing_m,
                max_spacing_m=args.max_spacing_m,
            )
        else:
            nominal_start, targets_world, route_categories = _sample_route(
                tile,
                rng,
                args.waypoints,
                min_spacing_m=args.min_spacing_m,
                max_spacing_m=args.max_spacing_m,
            )
        initial_yaw = math.atan2(
            float(targets_world[0, 1] - nominal_start[1]),
            float(targets_world[0, 0] - nominal_start[0]),
        )
        reference = _make_reference(
            backend, nominal_start, initial_yaw, alignment_config
        )
        if args.replay_result is not None or args.preview_generated_route:
            if args.preview_generated_route:
                fused = np.stack([
                    _route_position(reference, target) for target in targets_world
                ])
                replay_source = "generated_route_ground_truth_coordinates"
            else:
                assert replay_payload is not None
                comparison = replay_payload.get("comparison")
                if not comparison or len(comparison.get("fused_route_m", ())) != args.waypoints:
                    raise ValueError(
                        f"replay result does not contain {args.waypoints} fused waypoints"
                    )
                fused = np.asarray(comparison["fused_route_m"], dtype=np.float64)
                replay_source = str(args.replay_result)
            if args.gui:
                with mujoco.viewer.launch_passive(backend.model, backend.data[0]) as viewer:
                    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
                    viewer.cam.trackbodyid = backend.base_body_id
                    viewer.cam.distance = args.camera_distance
                    viewer.cam.azimuth = args.camera_azimuth
                    viewer.cam.elevation = args.camera_elevation
                    print(
                        "[onboard GUI] green=current waypoint, cyan=front LiDAR, "
                        "magenta=rear LiDAR; actor and waypoint switching use onboard state",
                        flush=True,
                    )
                    replay = _onboard_replay(
                        fused,
                        (0.16, -0.18, -4.0),
                        backend,
                        env,
                        policy,
                        reference,
                        alignment_config,
                        args,
                        viewer=viewer,
                    )
            else:
                replay = _onboard_replay(
                    fused,
                    (0.16, -0.18, -4.0),
                    backend,
                    env,
                    policy,
                    reference,
                    alignment_config,
                    args,
                )
            print(
                "ONBOARD_REPLAY_ONLY",
                f"success={replay['success']}",
                f"reached={replay['reached']}/{replay['waypoints']}",
                f"alignment={replay['alignment']}",
                f"failure={replay['failure']}",
                flush=True,
            )
            return {"onboard_replay": replay, "source_result": replay_source}
        save_start_anchor(
            output_dir / "reference_start.anchor.npz",
            reference.anchor.points_reference_body_m,
            reference.anchor.initial_imu_quaternion_wxyz,
            frame_count=reference.anchor.frame_count,
            voxel_size_m=reference.anchor.voxel_size_m,
        )
        perturbations = (
            (0.18, -0.12, 2.0),
            (-0.22, 0.16, -3.5),
            (0.12, 0.20, 5.0),
        )
        passes = []
        for index, perturbation in enumerate(
            perturbations[:args.collection_passes], start=1
        ):
            result = _collection_pass(
                index,
                perturbation,
                backend,
                env,
                policy,
                reference,
                targets_world,
                output_dir,
                alignment_config,
                args,
            )
            passes.append(result)
            (output_dir / "progress.json").write_text(
                json.dumps({"collection_passes": passes}, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            print(
                "COLLECTION_PASS",
                index,
                f"success={result['success']}",
                f"collected={result['collected']}/{args.waypoints}",
                f"alignment={result['alignment']}",
                f"failure={result['failure']}",
                flush=True,
            )
            if not result["success"]:
                break

        completed = [item for item in passes if item["success"]]
        comparison = None
        replay = None
        if len(completed) == args.collection_passes:
            positions = np.asarray([
                [record["recorded_route_m"] for record in item["records"]]
                for item in completed
            ])
            fused = np.median(positions, axis=0)
            deviations = np.linalg.norm(positions[:, :, :2] - fused[None, :, :2], axis=2)
            localization_errors = np.asarray([
                record["localization_error_m"]
                for item in completed for record in item["records"]
            ])
            stop_errors = np.asarray([
                record["target_stop_error_xy_m"]
                for item in completed for record in item["records"]
            ])
            comparison = {
                "fused_route_m": fused.tolist(),
                "cross_pass_xy_deviation_m": _summary(deviations),
                "collection_localization_error_m": _summary(localization_errors),
                "ground_truth_stop_error_xy_m": _summary(stop_errors),
                "per_waypoint_max_cross_pass_xy_m": np.max(deviations, axis=0).tolist(),
            }
            if not args.skip_replay:
                replay = _onboard_replay(
                    fused,
                    (0.16, -0.18, -4.0),
                    backend,
                    env,
                    policy,
                    reference,
                    alignment_config,
                    args,
                )
                print(
                    "ONBOARD_REPLAY",
                    f"success={replay['success']}",
                    f"reached={replay['reached']}/{replay['waypoints']}",
                    f"alignment={replay['alignment']}",
                    f"failure={replay['failure']}",
                    flush=True,
                )

        result = {
            "checkpoint": str(checkpoint),
            "checkpoint_iteration": payload.get("iter"),
            "map": {
                "terrain_profile": "stage5_lower_density_stairs",
                "terrain_seed": args.terrain_seed,
                "surface_seed": args.surface_seed,
                "row": args.row,
                "col": args.col,
                "type": tile.terrain_type,
                "difficulty": float(tile.difficulty),
            },
            "route_seed": args.route_seed,
            "route": {
                "profile": args.route_profile,
                "min_spacing_m": args.min_spacing_m,
                "max_spacing_m": args.max_spacing_m,
                "categories": route_categories,
            },
            "target_world_m": targets_world.tolist(),
            "collection_passes": passes,
            "comparison": comparison,
            "onboard_replay": replay,
            "wall_seconds": time.monotonic() - started,
        }
        output = output_dir / "result.json"
        output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print("START_ANCHOR_ROUTE_SIM", json.dumps({
            "output": str(output),
            "completed_collection_passes": len(completed),
            "comparison": comparison,
            "onboard_replay": replay,
        }, ensure_ascii=False), flush=True)
        return result
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--terrain-seed", type=int, default=123)
    parser.add_argument("--surface-seed", type=int, default=124)
    parser.add_argument("--row", type=int, default=0)
    parser.add_argument("--col", type=int, default=0)
    parser.add_argument("--route-seed", type=int, default=20260912)
    parser.add_argument(
        "--route-profile", choices=("random", "elevated"), default="random"
    )
    parser.add_argument("--min-spacing-m", type=float, default=2.8)
    parser.add_argument("--max-spacing-m", type=float, default=5.2)
    parser.add_argument("--seed", type=int, default=20260913)
    parser.add_argument("--waypoints", type=int, default=10)
    parser.add_argument("--collection-passes", type=int, choices=(1, 2, 3), default=3)
    parser.add_argument("--skip-replay", action="store_true")
    parser.add_argument(
        "--replay-result",
        type=Path,
        help="skip collection and replay fused_route_m from an existing result.json",
    )
    parser.add_argument(
        "--preview-generated-route",
        action="store_true",
        help="skip collection and replay the generated route with onboard observations",
    )
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--realtime-factor", type=float, default=1.0)
    parser.add_argument("--camera-distance", type=float, default=5.0)
    parser.add_argument("--camera-azimuth", type=float, default=145.0)
    parser.add_argument("--camera-elevation", type=float, default=-40.0)
    parser.add_argument("--near-clip", type=float, default=0.05)
    parser.add_argument("--max-steps-per-waypoint", type=int, default=350)
    parser.add_argument("--max-collection-hold-steps", type=int, default=30)
    parser.add_argument("--max-total-steps", type=int, default=10000)
    parser.add_argument("--collection-reach-m", type=float, default=0.45)
    parser.add_argument("--replay-reach-m", type=float, default=0.50)
    parser.add_argument("--wheel-radius", type=float, default=0.0825)
    parser.add_argument("--odometry-voxel-size", type=float, default=0.10)
    parser.add_argument("--icp-threads", type=int, default=4)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "training/evidence/start_anchor_route_sim_model2750",
    )
    args = parser.parse_args()
    if args.waypoints < 2 or args.max_steps_per_waypoint < 1:
        raise ValueError("waypoints and step limits must be positive")
    if args.min_spacing_m <= 0.0 or args.max_spacing_m <= args.min_spacing_m:
        raise ValueError("waypoint spacing bounds are invalid")
    if args.realtime_factor <= 0.0 or args.camera_distance <= 0.0:
        raise ValueError("realtime factor and camera distance must be positive")
    if args.near_clip <= 0.0:
        raise ValueError("near clip must be positive")
    if args.replay_result is not None and args.preview_generated_route:
        raise ValueError("choose either --replay-result or --preview-generated-route")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    evaluate(args)


if __name__ == "__main__":
    main()
