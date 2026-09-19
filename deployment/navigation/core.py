"""Hardware-facing SRU navigation primitives independent of ROS2 transport."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import hashlib
from pathlib import Path
import threading
from typing import Any

import numpy as np
import yaml

from deployment.common.lidar_geometry import (
    AIRY_VERTICAL_RAY_ANGLES,
    MAX_RANGE_M,
    MIN_RANGE_M,
    NATIVE_HEIGHT,
)
from deployment.common.math_utils import quat_wxyz_to_rotmat


NATIVE_WIDTH = 900


def quaternion_wxyz_from_rpy_deg(rpy_deg: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = np.deg2rad(np.asarray(rpy_deg, dtype=np.float64)) * 0.5
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    return np.asarray((
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    ))


def quaternion_wxyz_from_yaw(yaw_rad: float) -> np.ndarray:
    return np.asarray((np.cos(yaw_rad * 0.5), 0.0, 0.0, np.sin(yaw_rad * 0.5)))


@dataclass(frozen=True)
class SensorExtrinsic:
    position_body: np.ndarray
    rotation_body_sensor: np.ndarray

    @classmethod
    def from_config(cls, value: dict[str, Any]) -> "SensorExtrinsic":
        position = np.asarray(value["position_body_m"], dtype=np.float64)
        quaternion = np.asarray(value["quaternion_body_sensor_wxyz"], dtype=np.float64)
        if position.shape != (3,) or quaternion.shape != (4,):
            raise ValueError("sensor extrinsic must contain a 3D position and WXYZ quaternion")
        return cls(position, quat_wxyz_to_rotmat(quaternion))


@dataclass(frozen=True)
class CloudFrame:
    points_sensor: np.ndarray
    point_timestamps: np.ndarray
    rings: np.ndarray | None
    stamp_s: float
    receipt_s: float


@dataclass(frozen=True)
class AdaptedCloud:
    points_body: np.ndarray
    point_timestamps: np.ndarray
    distance_native: np.ndarray
    world_z_native: np.ndarray
    has_point_timestamps: bool


class AiryPointCloudAdapter:
    """Convert XYZIRT Airy points to KISS points and the trained 96x900 raster."""

    def __init__(
        self,
        extrinsic: SensorExtrinsic,
        *,
        ring_flip: bool = False,
        points_in_body_frame: bool = False,
    ):
        self.extrinsic = extrinsic
        self.ring_flip = bool(ring_flip)
        self.points_in_body_frame = bool(points_in_body_frame)

    def convert(
        self,
        frame: CloudFrame,
        position_map: np.ndarray,
        rotation_map_body: np.ndarray,
        *,
        build_raster: bool = True,
    ) -> AdaptedCloud:
        input_points = np.asarray(frame.points_sensor, dtype=np.float64)
        if input_points.ndim != 2 or input_points.shape[1] != 3:
            raise ValueError(f"Airy points must have shape [N,3], got {input_points.shape}")
        if not build_raster:
            if self.points_in_body_frame:
                points_body_all = input_points
                relative = input_points - self.extrinsic.position_body
                ranges = np.sqrt(np.einsum("ni,ni->n", relative, relative))
            else:
                points_body_all = (
                    self.extrinsic.position_body
                    + input_points @ self.extrinsic.rotation_body_sensor.T
                )
                ranges = np.sqrt(np.einsum("ni,ni->n", input_points, input_points))
            valid = (
                np.isfinite(points_body_all).all(axis=1)
                & np.isfinite(ranges)
                & (ranges > MIN_RANGE_M)
                & (ranges < 9.9)
            )
            points_body = points_body_all[valid]
            if len(points_body) < 100:
                raise ValueError(f"Airy frame has too few valid points: {len(points_body)}")
            stamps = np.asarray(frame.point_timestamps, dtype=np.float64).reshape(-1)
            has_stamps = len(stamps) == len(valid)
            stamps = stamps[valid] if has_stamps else np.empty(0, dtype=np.float64)
            if len(stamps) and (
                not np.isfinite(stamps).all() or np.ptp(stamps) <= 1.0e-6
            ):
                stamps = np.empty(0, dtype=np.float64)
                has_stamps = False
            return AdaptedCloud(
                points_body=np.ascontiguousarray(points_body, dtype=np.float64),
                point_timestamps=np.ascontiguousarray(stamps, dtype=np.float64),
                distance_native=np.empty((0, 0), dtype=np.float32),
                world_z_native=np.empty((0, 0), dtype=np.float32),
                has_point_timestamps=has_stamps,
            )

        if self.points_in_body_frame:
            points_body_all = input_points
            points_sensor_all = (
                input_points - self.extrinsic.position_body
            ) @ self.extrinsic.rotation_body_sensor
        else:
            points_sensor_all = input_points
            points_body_all = (
                self.extrinsic.position_body
                + input_points @ self.extrinsic.rotation_body_sensor.T
            )
        ranges = np.linalg.norm(points_sensor_all, axis=1)
        valid = (
            np.isfinite(points_sensor_all).all(axis=1)
            & np.isfinite(points_body_all).all(axis=1)
            & (ranges > MIN_RANGE_M)
            & (ranges < 9.9)
        )
        points = points_sensor_all[valid]
        points_body = points_body_all[valid]
        ranges = ranges[valid]
        if len(points) < 100:
            raise ValueError(f"Airy frame has too few valid points: {len(points)}")

        stamps = np.asarray(frame.point_timestamps, dtype=np.float64).reshape(-1)
        has_stamps = len(stamps) == len(valid)
        stamps = stamps[valid] if has_stamps else np.empty(0, dtype=np.float64)
        if len(stamps) and (not np.isfinite(stamps).all() or np.ptp(stamps) <= 1.0e-6):
            stamps = np.empty(0, dtype=np.float64)
            has_stamps = False

        if frame.rings is not None:
            rings = np.asarray(frame.rings).reshape(-1)
            rings = rings[valid] if len(rings) == len(valid) else np.empty(0)
        else:
            rings = np.empty(0)
        if len(rings) == len(points) and np.isfinite(rings).all() and ((rings >= 0) & (rings < NATIVE_HEIGHT)).all():
            rows = rings.astype(np.int64)
            if self.ring_flip:
                rows = NATIVE_HEIGHT - 1 - rows
        else:
            elevation = np.degrees(np.arctan2(points[:, 2], np.linalg.norm(points[:, :2], axis=1)))
            rows = np.abs(elevation[:, None] - AIRY_VERTICAL_RAY_ANGLES[None, :]).argmin(axis=1)

        yaw_deg = np.degrees(np.arctan2(points[:, 1], points[:, 0]))
        columns = np.rint((yaw_deg + 180.0) * (NATIVE_WIDTH / 360.0)).astype(np.int64) % NATIVE_WIDTH
        flat = rows * NATIVE_WIDTH + columns
        order = np.lexsort((ranges, flat))
        ordered_flat = flat[order]
        first = np.r_[True, ordered_flat[1:] != ordered_flat[:-1]]
        selected = order[first]

        points_map = np.asarray(position_map) + points_body @ np.asarray(rotation_map_body).T
        distance_native = np.full(NATIVE_HEIGHT * NATIVE_WIDTH, MAX_RANGE_M, dtype=np.float32)
        world_z_native = np.zeros(NATIVE_HEIGHT * NATIVE_WIDTH, dtype=np.float32)
        selected_flat = flat[selected]
        distance_native[selected_flat] = ranges[selected].astype(np.float32)
        world_z_native[selected_flat] = np.clip(points_map[selected, 2], -3.0, 3.0).astype(np.float32)

        return AdaptedCloud(
            points_body=np.ascontiguousarray(points_body, dtype=np.float64),
            point_timestamps=np.ascontiguousarray(stamps, dtype=np.float64),
            distance_native=distance_native.reshape(NATIVE_HEIGHT, NATIVE_WIDTH),
            world_z_native=world_z_native.reshape(NATIVE_HEIGHT, NATIVE_WIDTH),
            has_point_timestamps=has_stamps,
        )


class DualCloudSynchronizer:
    def __init__(self, max_skew_s: float = 0.03, queue_size: int = 8):
        self.max_skew_s = float(max_skew_s)
        self.front: deque[CloudFrame] = deque(maxlen=queue_size)
        self.rear: deque[CloudFrame] = deque(maxlen=queue_size)

    def push(self, side: str, frame: CloudFrame) -> tuple[CloudFrame, CloudFrame] | None:
        queue = self.front if side == "front" else self.rear
        other = self.rear if side == "front" else self.front
        queue.append(frame)
        if not other:
            return None
        closest = min(other, key=lambda item: abs(item.stamp_s - frame.stamp_s))
        if abs(closest.stamp_s - frame.stamp_s) > self.max_skew_s:
            return None
        self._remove_identity(queue, frame)
        self._remove_identity(other, closest)
        return (frame, closest) if side == "front" else (closest, frame)

    @staticmethod
    def _remove_identity(queue: deque[CloudFrame], target: CloudFrame) -> None:
        for index, item in enumerate(queue):
            if item is target:
                del queue[index]
                return
        raise RuntimeError("paired cloud disappeared from synchronization queue")


class ImuWheelBuffer:
    """Zero-order hold wheel data onto high-rate robot IMU samples."""

    def __init__(self, max_samples: int = 2000):
        self.samples: deque[tuple[Any, ...]] = deque(maxlen=max_samples)
        self.wheel_qvel = np.zeros(4, dtype=np.float64)
        self.wheel_torque = np.zeros(4, dtype=np.float64)
        self._lock = threading.Lock()

    def update_wheels(self, qvel: np.ndarray, torque: np.ndarray) -> None:
        with self._lock:
            self.wheel_qvel = np.asarray(qvel, dtype=np.float64).copy()
            self.wheel_torque = np.asarray(torque, dtype=np.float64).copy()

    def append_imu(
        self,
        stamp_s: float,
        receipt_s: float,
        orientation_wxyz: np.ndarray,
        accelerometer: np.ndarray,
        gyro: np.ndarray,
    ) -> None:
        with self._lock:
            self.samples.append((
                float(stamp_s),
                float(receipt_s),
                np.asarray(orientation_wxyz, dtype=np.float64).copy(),
                np.asarray(accelerometer, dtype=np.float64).copy(),
                np.asarray(gyro, dtype=np.float64).copy(),
                self.wheel_qvel.copy(),
                self.wheel_torque.copy(),
            ))

    def latest(self) -> tuple[Any, ...] | None:
        with self._lock:
            return self.samples[-1] if self.samples else None

    def history(self, previous_stamp_s: float, current_stamp_s: float) -> dict[str, np.ndarray]:
        """Return IMU samples covering a LiDAR interval in sensor-clock time."""
        with self._lock:
            all_samples = list(self.samples)
        return self._history_between(all_samples, previous_stamp_s, current_stamp_s, 0)

    def history_by_receipt(
        self, previous_receipt_s: float, current_receipt_s: float
    ) -> dict[str, np.ndarray]:
        """Associate sensors in AGX monotonic time while preserving IMU integration time."""
        with self._lock:
            all_samples = list(self.samples)
        return self._history_between(
            all_samples, previous_receipt_s, current_receipt_s, 1
        )

    @staticmethod
    def _history_between(
        all_samples: list[tuple[Any, ...]],
        previous_s: float,
        current_s: float,
        clock_index: int,
    ) -> dict[str, np.ndarray]:
        selected = [
            sample for sample in all_samples
            if previous_s < sample[clock_index] <= current_s
        ]
        prior = [sample for sample in all_samples if sample[clock_index] <= previous_s]
        if prior:
            selected.insert(0, prior[-1])
        if len(selected) < 2:
            raise RuntimeError(f"only {len(selected)} IMU samples cover the LiDAR interval")
        return {
            "time": np.asarray([sample[0] for sample in selected]),
            "orientation_wxyz": np.stack([sample[2] for sample in selected]),
            "accelerometer": np.stack([sample[3] for sample in selected]),
            "gyro": np.stack([sample[4] for sample in selected]),
            "wheel_qvel": np.stack([sample[5] for sample in selected]),
            "wheel_torque": np.stack([sample[6] for sample in selected]),
            "illegal_contact_force": np.zeros(len(selected)),
        }


class RouteManager:
    def __init__(self, config: dict[str, Any]):
        self.waypoints = np.asarray(config["waypoints_map_m"], dtype=np.float64)
        self.skipped = frozenset(int(value) for value in config.get("skipped_waypoints", ()))
        self.start_index = int(config.get("start_waypoint", 0))
        self.final_index = int(config.get("final_waypoint", len(self.waypoints) - 1))
        self.reach_xy_m = float(config.get("reach_xy_m", 0.5))
        self.reach_z_m = float(config.get("reach_z_m", 0.55))
        self.hold_s = float(config.get("hold_s", 0.4))
        self.initial_base_height_m = float(config.get("initial_base_height_m", 0.425))
        self.initial_map_yaw_deg = float(config.get("initial_map_yaw_deg", 0.0))
        self.active_index = self._next(self.start_index)
        self._inside_since: float | None = None
        self.complete = False
        bindings = config.get("waypoint_topometric_bindings", ())
        self.topometric_bindings = tuple(bindings) if bindings else ()
        if self.topometric_bindings and len(self.topometric_bindings) != len(self.waypoints):
            raise ValueError("waypoint topometric bindings must match waypoint count")

    def active_route_index_hint(self) -> int | None:
        if not self.topometric_bindings:
            return None
        return int(self.topometric_bindings[self.active_index]["route_index"])

    def _next(self, index: int) -> int:
        for candidate in range(index + 1, self.final_index + 1):
            if candidate not in self.skipped:
                return candidate
        return self.final_index

    def initial_pose(self) -> np.ndarray:
        initial = np.asarray(self.waypoints[self.start_index], dtype=np.float64).copy()
        initial[2] += self.initial_base_height_m
        return np.concatenate((initial, quaternion_wxyz_from_yaw(np.deg2rad(self.initial_map_yaw_deg))))

    @staticmethod
    def _encode_target(position: np.ndarray, rotation: np.ndarray, target: np.ndarray) -> np.ndarray:
        delta = rotation.T @ (target - position)
        distance = max(float(np.linalg.norm(delta)), 1.0e-6)
        return np.concatenate((delta / distance, (np.log1p(distance),))).astype(np.float32)

    def goal_body(self, position: np.ndarray, rotation: np.ndarray) -> np.ndarray:
        return self._encode_target(position, rotation, self.waypoints[self.active_index])

    def update(self, position: np.ndarray, now_s: float) -> bool:
        target = self.waypoints[self.active_index]
        inside = (
            np.linalg.norm(np.asarray(position)[:2] - target[:2]) <= self.reach_xy_m
            and abs(float(position[2] - target[2])) <= self.reach_z_m
        )
        self._inside_since = now_s if inside and self._inside_since is None else self._inside_since
        if not inside:
            self._inside_since = None
            return False
        if now_s - float(self._inside_since) < self.hold_s:
            return False
        if self.active_index >= self.final_index:
            self.complete = True
        else:
            self.active_index = self._next(self.active_index)
            self._inside_since = None
        return True


@dataclass(frozen=True)
class LocalizationQuality:
    healthy: bool
    state: str
    reasons: tuple[str, ...]


class LocalizationQualityGate:
    def __init__(
        self,
        *,
        warmup_frames: int = 5,
        max_covariance_trace: float = 25.0,
        min_imu_samples: int = 5,
        min_imu_span_s: float = 0.05,
    ):
        self.warmup_frames = int(warmup_frames)
        self.max_covariance_trace = float(max_covariance_trace)
        self.min_imu_samples = int(min_imu_samples)
        self.min_imu_span_s = float(min_imu_span_s)
        if self.min_imu_samples < 2 or self.min_imu_span_s <= 0.0:
            raise ValueError("IMU quality thresholds must be positive and use at least two samples")
        self.good_frames = 0
        self.consecutive_icp_rejections = 0

    def evaluate(
        self,
        *,
        pair_skew_s: float,
        points_per_lidar: tuple[int, int],
        point_timestamps_present: bool,
        imu_samples: int,
        imu_span_s: float,
        covariance_trace: float,
        icp_accepted: bool,
    ) -> LocalizationQuality:
        reasons = []
        if pair_skew_s > 0.03:
            reasons.append("dual_lidar_skew")
        if min(points_per_lidar) < 1000:
            reasons.append("too_few_lidar_points")
        if not point_timestamps_present:
            reasons.append("missing_point_timestamps")
        # Sensor-clock coverage is the primary condition. ROS may deliver the
        # 200 Hz IMU in batches while cloud callbacks are active, so requiring
        # ten callbacks in addition to a 50 ms span caused false degradation.
        if imu_samples < self.min_imu_samples or imu_span_s < self.min_imu_span_s:
            reasons.append("insufficient_imu_history")
        if covariance_trace > self.max_covariance_trace or not np.isfinite(covariance_trace):
            reasons.append("filter_covariance")
        self.consecutive_icp_rejections = 0 if icp_accepted else self.consecutive_icp_rejections + 1
        if self.consecutive_icp_rejections >= 2:
            reasons.append("repeated_icp_rejection")
        if reasons:
            self.good_frames = 0
            return LocalizationQuality(False, "LOST" if "repeated_icp_rejection" in reasons else "DEGRADED", tuple(reasons))
        self.good_frames += 1
        if self.good_frames < self.warmup_frames:
            return LocalizationQuality(False, "WARMUP", ("localization_warmup",))
        return LocalizationQuality(True, "GOOD", ())


def estimate_support_height_map(
    points_body: np.ndarray,
    position_map: np.ndarray,
    rotation_map_body: np.ndarray,
    *,
    expected_clearance_m: float = 0.425,
) -> tuple[float, int, float]:
    """Fit a local support plane around the chassis and return its map-frame Z."""
    points = np.asarray(points_body, dtype=np.float64)
    radial = np.linalg.norm(points[:, :2], axis=1)
    expected_z = -float(expected_clearance_m)
    valid = (
        np.isfinite(points).all(axis=1)
        & (radial >= 0.25)
        & (radial <= 0.90)
        & (points[:, 2] >= expected_z - 0.30)
        & (points[:, 2] <= expected_z + 0.20)
    )
    selected = points[valid]
    if len(selected) < 50:
        return float("nan"), len(selected), float("nan")
    mapped = np.asarray(position_map) + selected @ np.asarray(rotation_map_body).T
    local_xy = mapped[:, :2] - np.asarray(position_map)[:2]
    design = np.column_stack((local_xy, np.ones(len(local_xy))))
    inliers = np.ones(len(mapped), dtype=bool)
    coefficients = np.zeros(3)
    for _ in range(3):
        if int(np.count_nonzero(inliers)) < 50:
            return float("nan"), int(np.count_nonzero(inliers)), float("nan")
        coefficients, *_ = np.linalg.lstsq(design[inliers], mapped[inliers, 2], rcond=None)
        residual = mapped[:, 2] - design @ coefficients
        median = float(np.median(residual[inliers]))
        mad = float(np.median(np.abs(residual[inliers] - median)))
        threshold = max(0.025, 2.5 * 1.4826 * mad)
        inliers = np.abs(residual - median) <= threshold
    residual = mapped[inliers, 2] - design[inliers] @ coefficients
    rmse = float(np.sqrt(np.mean(residual ** 2)))
    slope = float(np.linalg.norm(coefficients[:2]))
    count = int(np.count_nonzero(inliers))
    if count < 50 or rmse > 0.08 or slope > 1.0:
        return float("nan"), count, rmse
    return float(coefficients[2]), count, rmse


def apply_route_file(config: dict[str, Any], waypoint_path: Path) -> None:
    route = config.get("route")
    if not isinstance(route, dict):
        raise ValueError("hardware config must contain a route mapping")
    route_payload = yaml.safe_load(waypoint_path.read_text(encoding="utf-8"))
    nodes = route_payload.get("nodes") if isinstance(route_payload, dict) else None
    if not isinstance(nodes, list) or not nodes:
        raise ValueError(f"route file has no nodes: {waypoint_path}")
    route["waypoints_map_m"] = [node["position"] for node in nodes]
    bindings = [node.get("topometric_binding") for node in nodes]
    if any(binding is not None for binding in bindings):
        if not all(isinstance(binding, dict) for binding in bindings):
            raise ValueError("every route node must have a topometric binding")
        route["waypoint_topometric_bindings"] = bindings
    topometric_map = route_payload.get("topometric_map")
    if topometric_map:
        map_path = Path(str(topometric_map)).expanduser()
        if not map_path.is_absolute():
            map_path = waypoint_path.expanduser().resolve().parent / map_path
        map_path = map_path.resolve()
        if not (map_path / "localization_map_manifest.json").is_file():
            raise FileNotFoundError(f"topometric localization map not found: {map_path}")
        map_config = config.setdefault("map_localization", {})
        map_config["enabled"] = True
        map_config["map_dir"] = str(map_path)
    route["start_waypoint"] = 0
    route["final_waypoint"] = len(nodes) - 1
    route["skipped_waypoints"] = list(route_payload.get("skipped_waypoints", []))
    route["initial_map_yaw_deg"] = float(route_payload.get(
        "initial_map_yaw_deg", route.get("initial_map_yaw_deg", 0.0)
    ))
    route["initial_base_height_m"] = float(route_payload.get(
        "initial_base_height_m", route.get("initial_base_height_m", 0.425)
    ))
    alignment = route_payload.get("initial_alignment", {})
    if isinstance(alignment, dict) and alignment.get("anchor_file"):
        anchor_path = Path(str(alignment["anchor_file"])).expanduser()
        if not anchor_path.is_absolute():
            anchor_path = waypoint_path.expanduser().resolve().parent / anchor_path
        anchor_path = anchor_path.resolve()
        if not anchor_path.is_file():
            raise FileNotFoundError(f"route start anchor not found: {anchor_path}")
        expected_sha256 = alignment.get("anchor_sha256")
        if expected_sha256:
            digest = hashlib.sha256(anchor_path.read_bytes()).hexdigest()
            if digest != str(expected_sha256):
                raise ValueError(
                    f"route start anchor checksum mismatch: {anchor_path}"
                )
        route["start_anchor_file"] = str(anchor_path)
        route["start_alignment_method"] = str(
            alignment.get("method", "dual_lidar_start_anchor_se3")
        )
    route["source_file"] = str(waypoint_path)


def load_hardware_config(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"hardware config must contain a mapping: {path}")
    route = value.get("route")
    if not isinstance(route, dict):
        raise ValueError("hardware config must contain a route mapping")
    waypoint_file = route.pop("waypoint_file", None)
    if waypoint_file is not None:
        waypoint_path = Path(waypoint_file)
        if not waypoint_path.is_absolute():
            waypoint_path = Path(__file__).resolve().parents[2] / waypoint_path
        apply_route_file(value, waypoint_path.resolve())
    return value
