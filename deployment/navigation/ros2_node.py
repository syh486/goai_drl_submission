"""ROS2 hardware entry point for dual-Airy SRU navigation."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import queue
import threading
import time

import numpy as np

from deployment.navigation.core import (
    AiryPointCloudAdapter,
    CloudFrame,
    DualCloudSynchronizer,
    ImuWheelBuffer,
    LocalizationQualityGate,
    RouteManager,
    SensorExtrinsic,
    apply_route_file,
    estimate_support_height_map,
    load_hardware_config,
    quaternion_wxyz_from_rpy_deg,
)
from deployment.localization.local_odometry import DualLidarImuWheelEskfOdometry, LocalOdometryConfig
from deployment.localization.continuous_map_localization import (
    ContinuousLocalizationConfig,
    ContinuousMapLocalizer,
)
from deployment.mapping.mapping_recording import MappingKeyframeRecorder, MappingRecordingConfig
from deployment.common.math_utils import quat_wxyz_to_rotmat
from deployment.localization.start_alignment import (
    StartAlignmentConfig,
    StartAnchorAccumulator,
    align_start_anchor,
    compose_aligned_initial_pose,
    load_start_anchor,
    save_start_anchor,
)
from deployment.waypoints.validate_route import validate_route


REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class _MapLocalizationRequest:
    points_body: np.ndarray
    odom_from_body: np.ndarray
    traveled_distance_m: float
    allow_large_relocalization: bool
    queued_monotonic_s: float


@dataclass(frozen=True)
class _MapLocalizationSnapshot:
    result: object | None
    candidate_audits: tuple[dict[str, object], ...]
    started_monotonic_s: float
    first_submitted_monotonic_s: float
    last_completed_monotonic_s: float
    last_accepted_monotonic_s: float
    last_update_seconds: float
    last_queue_age_seconds: float
    submitted: int
    completed: int
    dropped_requests: int
    result_sequence: int
    last_error: str | None


class _AsyncMapLocalizationWorker:
    """Run stateful map matching without blocking local odometry."""

    def __init__(self, logger) -> None:
        self.logger = logger
        self.queue: queue.Queue[_MapLocalizationRequest] = queue.Queue(maxsize=1)
        self.stop_event = threading.Event()
        self.lock = threading.Lock()
        self.localizer = None
        self.thread: threading.Thread | None = None
        self.result = None
        self.candidate_audits: tuple[dict[str, object], ...] = ()
        self.started_monotonic_s = 0.0
        self.first_submitted_monotonic_s = 0.0
        self.last_completed_monotonic_s = 0.0
        self.last_accepted_monotonic_s = 0.0
        self.last_update_seconds = 0.0
        self.last_queue_age_seconds = 0.0
        self.submitted = 0
        self.completed = 0
        self.dropped_requests = 0
        self.result_sequence = 0
        self.last_error: str | None = None

    def start(self, localizer) -> None:
        if self.thread is not None:
            raise RuntimeError("map-localization worker was already started")
        self.localizer = localizer
        self.started_monotonic_s = time.monotonic()
        self.thread = threading.Thread(
            target=self._loop,
            name="s10_route_map_localization",
            daemon=True,
        )
        self.thread.start()

    def submit(
        self,
        points_body: np.ndarray,
        odom_from_body: np.ndarray,
        *,
        traveled_distance_m: float,
        allow_large_relocalization: bool,
    ) -> None:
        request = _MapLocalizationRequest(
            points_body=np.ascontiguousarray(points_body, dtype=np.float64),
            odom_from_body=np.asarray(odom_from_body, dtype=np.float64).copy(),
            traveled_distance_m=float(traveled_distance_m),
            allow_large_relocalization=bool(allow_large_relocalization),
            queued_monotonic_s=time.monotonic(),
        )
        with self.lock:
            self.submitted += 1
            if self.first_submitted_monotonic_s == 0.0:
                self.first_submitted_monotonic_s = request.queued_monotonic_s
        if self.queue.full():
            try:
                self.queue.get_nowait()
                self.queue.task_done()
                with self.lock:
                    self.dropped_requests += 1
            except queue.Empty:
                pass
        self.queue.put_nowait(request)

    def _loop(self) -> None:
        while not self.stop_event.is_set():
            try:
                request = self.queue.get(timeout=0.1)
            except queue.Empty:
                continue
            started = time.monotonic()
            try:
                result = self.localizer.update(
                    request.points_body,
                    request.odom_from_body,
                    traveled_distance_m=request.traveled_distance_m,
                    allow_large_relocalization=(
                        request.allow_large_relocalization
                    ),
                )
                completed = time.monotonic()
                audits = tuple(
                    dict(item) for item in self.localizer.last_candidate_audits
                )
                with self.lock:
                    self.result = result
                    self.candidate_audits = audits
                    self.last_completed_monotonic_s = completed
                    if result.observation_accepted:
                        self.last_accepted_monotonic_s = completed
                    self.last_update_seconds = completed - started
                    self.last_queue_age_seconds = max(
                        0.0, started - request.queued_monotonic_s
                    )
                    self.completed += 1
                    self.result_sequence += 1
                    self.last_error = None
            except Exception as error:
                with self.lock:
                    self.last_error = f"{type(error).__name__}: {error}"
                self.logger.error(f"route-map localization failed: {error}")
            finally:
                self.queue.task_done()

    def snapshot(self) -> _MapLocalizationSnapshot:
        with self.lock:
            return _MapLocalizationSnapshot(
                result=self.result,
                candidate_audits=self.candidate_audits,
                started_monotonic_s=self.started_monotonic_s,
                first_submitted_monotonic_s=self.first_submitted_monotonic_s,
                last_completed_monotonic_s=self.last_completed_monotonic_s,
                last_accepted_monotonic_s=self.last_accepted_monotonic_s,
                last_update_seconds=self.last_update_seconds,
                last_queue_age_seconds=self.last_queue_age_seconds,
                submitted=self.submitted,
                completed=self.completed,
                dropped_requests=self.dropped_requests,
                result_sequence=self.result_sequence,
                last_error=self.last_error,
            )

    def close(self) -> None:
        self.stop_event.set()
        if self.thread is None:
            return
        self.thread.join(timeout=10.0)
        if self.thread.is_alive():
            self.logger.warning("route-map localization worker did not stop cleanly")


def _resolve_repo_path(value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else REPO_ROOT / path


def _stamp_seconds(stamp) -> float:
    return float(stamp.sec) + float(stamp.nanosec) * 1.0e-9


def _rotmat_to_quat_wxyz(rotation: np.ndarray) -> np.ndarray:
    matrix = np.asarray(rotation, dtype=np.float64)
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        quat = np.asarray((0.25 * scale, (matrix[2, 1] - matrix[1, 2]) / scale,
                           (matrix[0, 2] - matrix[2, 0]) / scale,
                           (matrix[1, 0] - matrix[0, 1]) / scale))
    else:
        index = int(np.argmax(np.diag(matrix)))
        if index == 0:
            scale = math.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
            quat = np.asarray(((matrix[2, 1] - matrix[1, 2]) / scale, 0.25 * scale,
                               (matrix[0, 1] + matrix[1, 0]) / scale,
                               (matrix[0, 2] + matrix[2, 0]) / scale))
        elif index == 1:
            scale = math.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
            quat = np.asarray(((matrix[0, 2] - matrix[2, 0]) / scale,
                               (matrix[0, 1] + matrix[1, 0]) / scale, 0.25 * scale,
                               (matrix[1, 2] + matrix[2, 1]) / scale))
        else:
            scale = math.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
            quat = np.asarray(((matrix[1, 0] - matrix[0, 1]) / scale,
                               (matrix[0, 2] + matrix[2, 0]) / scale,
                               (matrix[1, 2] + matrix[2, 1]) / scale, 0.25 * scale))
    return quat / np.linalg.norm(quat)


def _pose_matrix(position: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = np.asarray(rotation, dtype=np.float64)
    pose[:3, 3] = np.asarray(position, dtype=np.float64)
    return pose


def _map_corrected_pose(local_pose: np.ndarray, localization_result) -> np.ndarray:
    if localization_result is None:
        return np.asarray(local_pose, dtype=np.float64).copy()
    return (
        np.asarray(localization_result.map_from_odom, dtype=np.float64)
        @ np.asarray(local_pose, dtype=np.float64)
    )


def _read_cloud(
    message, *, max_points: int = 0
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    # KISS-ICP 1.3.0's ROS helper calls vars(PointField), which is incompatible
    # with the slot-based PointField messages shipped by ROS2 Jazzy.  Use the
    # ROS distribution's parser and pass only plain NumPy arrays to KISS-ICP.
    from sensor_msgs_py.point_cloud2 import read_points

    available = {field.name for field in message.fields}
    timestamp_name = next((name for name in ("timestamp", "time", "t") if name in available), None)
    names = ["x", "y", "z"]
    if "ring" in available:
        names.append("ring")
    if timestamp_name:
        names.append(timestamp_name)
    structured = np.asarray(read_points(message, field_names=names)).reshape(-1)
    if max_points < 0:
        raise ValueError("max_points must be nonnegative")
    if max_points and len(structured) > max_points:
        indices = np.linspace(
            0, len(structured) - 1, max_points, dtype=np.int64
        )
        structured = structured[indices]
    points = np.column_stack((structured["x"], structured["y"], structured["z"]))
    finite = np.isfinite(points).all(axis=1)
    points = points[finite].astype(np.float64, copy=False)
    stamps = (
        np.asarray(structured[timestamp_name], dtype=np.float64)[finite]
        if timestamp_name else np.empty(0, dtype=np.float64)
    )
    rings = np.asarray(structured["ring"])[finite] if "ring" in available else None
    return points, stamps, rings


def _pool_native_raster(
    distance_native: np.ndarray, world_z_native: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the trained 96x900 -> 96x90 nearest-return pooling in NumPy."""

    distances = np.asarray(distance_native, dtype=np.float32)
    world_z = np.asarray(world_z_native, dtype=np.float32)
    if distances.shape != (96, 900) or world_z.shape != distances.shape:
        raise ValueError(
            "ONNX raster input requires matching [96,900] distance and world-z maps"
        )
    grouped_distance = distances.reshape(96, 90, 10)
    grouped_z = world_z.reshape(96, 90, 10)
    valid = (grouped_distance > 0.05) & (grouped_distance < 9.9)
    safe = np.where(valid, grouped_distance, np.float32(10.0))
    indices = np.argmin(safe, axis=-1)
    pooled_distance = np.take_along_axis(
        safe, indices[..., None], axis=-1
    )[..., 0]
    pooled_z = np.take_along_axis(
        grouped_z, indices[..., None], axis=-1
    )[..., 0]
    return (
        np.ascontiguousarray(pooled_distance, dtype=np.float32),
        np.ascontiguousarray(pooled_z, dtype=np.float32),
    )


class S10HardwareNavigationNode:
    def __init__(
        self,
        node,
        config: dict,
        *,
        enable_motion_override: bool = False,
        localization_only: bool = False,
        onnx_dry_run: bool = False,
        record_start_anchor: Path | None = None,
        reference_start_anchor: Path | None = None,
        record_map: Path | None = None,
    ):
        from drdds.msg import ImuData, JointsData, Steer
        from geometry_msgs.msg import Twist
        from nav_msgs.msg import Odometry
        from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
        from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
        from sensor_msgs.msg import PointCloud2
        from std_msgs.msg import Bool, Float32MultiArray, Int32, String

        self.node = node
        self.messages = {
            "Twist": Twist,
            "Steer": Steer,
            "Odometry": Odometry,
            "Bool": Bool,
            "Float32MultiArray": Float32MultiArray,
            "Int32": Int32,
            "String": String,
        }
        topics = config["topics"]
        runtime = config["runtime"]
        sensors = config["sensors"]
        localization = config.get("localization", {})
        self.localization_only = bool(localization_only)
        self.onnx_dry_run = bool(onnx_dry_run)
        # Keep high-rate IMU ingestion independent from PointCloud2 decoding.
        # Front/rear clouds share one group because DualCloudSynchronizer is
        # deliberately single-writer; joints use a third group.
        self.cloud_callback_group = MutuallyExclusiveCallbackGroup()
        self.imu_callback_group = MutuallyExclusiveCallbackGroup()
        self.joints_callback_group = MutuallyExclusiveCallbackGroup()
        if self.localization_only and self.onnx_dry_run:
            raise ValueError("localization-only and ONNX dry-run modes are mutually exclusive")
        self.enable_motion = bool(
            not self.localization_only and not self.onnx_dry_run
            and (runtime.get("enable_motion", False) or enable_motion_override)
        )
        self.require_point_timestamps = bool(runtime.get("require_point_timestamps", True))
        self.max_staleness_s = float(runtime.get("max_sensor_staleness_s", 0.5))
        self.max_joint_staleness_s = float(
            runtime.get("max_joint_staleness_s", self.max_staleness_s)
        )
        self.control_period_s = 1.0 / float(runtime.get("control_hz", 5.0))
        self.publish_legacy_cmd_vel = bool(runtime.get("publish_legacy_cmd_vel", True))
        self.publish_dds_steer = bool(runtime.get("publish_dds_steer", True))
        self.dds_runner_command_scale = np.asarray(
            runtime.get("dds_runner_command_scale", (1.0, 1.0, 1.0)),
            dtype=np.float64,
        )
        if self.dds_runner_command_scale.shape != (3,) or np.any(
            self.dds_runner_command_scale <= 0.0
        ):
            raise ValueError("dds_runner_command_scale must contain three positive values")
        self.wheel_indices = np.asarray(runtime.get("wheel_indices", (3, 7, 11, 15)), dtype=np.int64)
        self.wheel_signs = tuple(float(value) for value in runtime["wheel_signs"])
        self.wheel_radius_m = float(runtime["wheel_radius_m"])
        cloud_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        imu_qos = QoSProfile(
            depth=400,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        joints_qos = QoSProfile(
            depth=100,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.localization_config = localization
        map_localization = config.get("map_localization", {})
        self.map_localization_enabled = bool(map_localization.get("enabled", False))
        self.map_localization_dir = (
            _resolve_repo_path(str(map_localization["map_dir"]))
            if self.map_localization_enabled else None
        )
        self.map_localization_stride = int(map_localization.get("update_stride", 2))
        self.map_localization_max_coast_s = float(
            map_localization.get("max_coast_s", 5.0)
        )
        if self.map_localization_stride < 1:
            raise ValueError("map_localization.update_stride must be positive")
        self.map_localization_config = ContinuousLocalizationConfig(
            registration_target=str(
                map_localization.get("registration_target", "route_submap")
            ),
            registration_backend=str(
                map_localization.get("registration_backend", "kiss_icp")
            ),
            beam_width=int(map_localization.get("beam_width", 2)),
            candidate_count=int(map_localization.get("candidate_count", 4)),
            odometry_recovery_candidate_count=int(
                map_localization.get("odometry_recovery_candidate_count", 2)
            ),
            query_voxel_m=float(map_localization.get("query_voxel_m", 0.22)),
            fine_query_voxel_m=float(
                map_localization.get("fine_query_voxel_m", 0.08)
            ),
            fine_max_query_points=int(
                map_localization.get("fine_max_query_points", 16000)
            ),
            query_submap_updates=int(
                map_localization.get("query_submap_updates", 5)
            ),
            max_icp_iterations=int(
                map_localization.get("max_icp_iterations", 20)
            ),
            icp_threads=int(map_localization.get("icp_threads", 2)),
            fine_max_icp_iterations=int(
                map_localization.get("fine_max_icp_iterations", 18)
            ),
            fine_max_correspondence_m=float(
                map_localization.get("fine_max_correspondence_m", 0.40)
            ),
            fine_min_fitness=float(
                map_localization.get("fine_min_fitness", 0.55)
            ),
            fine_max_rmse_m=float(
                map_localization.get("fine_max_rmse_m", 0.18)
            ),
            odometry_tracking_candidate_count=int(
                map_localization.get("odometry_tracking_candidate_count", 1)
            ),
            recovery_odometry_radius_submaps=int(
                map_localization.get("recovery_odometry_radius_submaps", 12)
            ),
            route_lag_cost=float(
                map_localization.get("route_lag_cost", 0.05)
            ),
            multisession_consensus_enabled=bool(
                map_localization.get("multisession_consensus_enabled", False)
            ),
            multisession_fitness_weight=float(
                map_localization.get("multisession_fitness_weight", 0.04)
            ),
            multisession_descriptor_weight=float(
                map_localization.get("multisession_descriptor_weight", 0.70)
            ),
            multisession_translation_innovation_weight=float(
                map_localization.get(
                    "multisession_translation_innovation_weight", 0.02
                )
            ),
            multisession_yaw_innovation_weight=float(
                map_localization.get(
                    "multisession_yaw_innovation_weight", 0.002
                )
            ),
            multisession_route_step_weight=float(
                map_localization.get("multisession_route_step_weight", 0.001)
            ),
            multisession_odometry_route_weight=float(
                map_localization.get(
                    "multisession_odometry_route_weight", 0.05
                )
            ),
            multisession_repeatability_bonus=float(
                map_localization.get("multisession_repeatability_bonus", 0.12)
            ),
            multisession_max_translation_disagreement_m=float(
                map_localization.get(
                    "multisession_max_translation_disagreement_m", 0.15
                )
            ),
            multisession_max_yaw_disagreement_deg=float(
                map_localization.get(
                    "multisession_max_yaw_disagreement_deg", 2.0
                )
            ),
            multisession_history_decay=float(
                map_localization.get("multisession_history_decay", 0.0)
            ),
            min_fitness=float(map_localization.get("min_fitness", 0.50)),
            max_rmse_m=float(map_localization.get("max_rmse_m", 0.48)),
            continuity_min_fitness=float(
                map_localization.get("continuity_min_fitness", 0.40)
            ),
            continuity_max_rmse_m=float(
                map_localization.get("continuity_max_rmse_m", 0.50)
            ),
            temporal_fusion_enabled=bool(
                map_localization.get("temporal_fusion_enabled", False)
            ),
            max_route_consistent_relocalization_translation_m=float(
                map_localization.get(
                    "max_route_consistent_relocalization_translation_m", 25.0
                )
            ),
            max_route_consistent_relocalization_yaw_deg=float(
                map_localization.get(
                    "max_route_consistent_relocalization_yaw_deg", 45.0
                )
            ),
        )
        self.max_registration_points = int(
            localization.get("max_registration_points", 0)
        )
        if self.max_registration_points < 0:
            raise ValueError("max_registration_points must be nonnegative")
        self.max_cloud_points_per_lidar = int(
            localization.get("max_cloud_points_per_lidar", 0)
        )
        if self.max_cloud_points_per_lidar < 0:
            raise ValueError("max_cloud_points_per_lidar must be nonnegative")
        if record_map is not None:
            # Mapping captures are the offline source of truth and retain the
            # full clouds. Runtime localization uses bounded, evenly sampled
            # clouds to keep latency below the sensor period.
            self.max_cloud_points_per_lidar = 0
        support_height_hz = float(localization.get("support_height_hz", 2.0))
        if support_height_hz <= 0.0:
            raise ValueError("support_height_hz must be positive")
        self.support_height_period_s = 1.0 / support_height_hz
        self.last_support_height_s = float("-inf")
        self.last_support_height = (float("nan"), 0, float("nan"))
        diagnostics_hz = float(localization.get("diagnostics_hz", 2.0))
        if diagnostics_hz <= 0.0:
            raise ValueError("diagnostics_hz must be positive")
        self.diagnostics_period_s = 1.0 / diagnostics_hz
        self.last_diagnostics_s = float("-inf")
        self.start_alignment_config = StartAlignmentConfig.from_config(
            config.get("start_alignment")
        )
        if record_map is not None and not self.localization_only:
            raise ValueError("map recording is only allowed in localization-only mode")
        self.map_recorder = (
            MappingKeyframeRecorder(
                record_map,
                MappingRecordingConfig.from_config(config.get("mapping_recording")),
            )
            if record_map is not None else None
        )

        self.front_adapter = AiryPointCloudAdapter(
            SensorExtrinsic.from_config(sensors["front"]),
            ring_flip=bool(sensors["front"].get("ring_flip", False)),
            points_in_body_frame=bool(sensors["front"].get("points_in_body_frame", False)),
        )
        self.rear_adapter = AiryPointCloudAdapter(
            SensorExtrinsic.from_config(sensors["rear"]),
            ring_flip=bool(sensors["rear"].get("ring_flip", False)),
            points_in_body_frame=bool(sensors["rear"].get("points_in_body_frame", False)),
        )
        self.synchronizer = DualCloudSynchronizer(float(runtime["max_dual_lidar_skew_s"]))
        self.sensor_buffer = ImuWheelBuffer()
        self.route = RouteManager(config["route"])
        self.record_start_anchor = (
            record_start_anchor.expanduser().resolve()
            if record_start_anchor is not None else None
        )
        anchor_source = (
            str(reference_start_anchor.expanduser().resolve())
            if reference_start_anchor is not None
            else config["route"].get("start_anchor_file")
        )
        if self.record_start_anchor is not None and anchor_source is not None:
            raise ValueError("cannot record and consume a start anchor in the same process")
        self.reference_start_anchor = (
            load_start_anchor(Path(anchor_source)) if anchor_source is not None else None
        )
        self.start_anchor_accumulator = (
            StartAnchorAccumulator(self.start_alignment_config)
            if self.record_start_anchor is not None or self.reference_start_anchor is not None
            else None
        )
        self.start_anchor_initial_imu: np.ndarray | None = None
        self.start_alignment_result = None
        self.gate = LocalizationQualityGate(
            warmup_frames=int(runtime.get("warmup_frames", 5)),
            min_imu_samples=int(runtime.get("min_imu_samples", 5)),
            min_imu_span_s=float(runtime.get("min_imu_span_s", 0.05)),
        )
        self.odometry = None
        self.map_localizer = None
        self.map_localization_worker = (
            _AsyncMapLocalizationWorker(self.node.get_logger())
            if self.map_localization_enabled else None
        )
        self.map_localization_updates = 0
        self.local_traveled_distance_m = 0.0
        self.previous_local_position: np.ndarray | None = None
        self.last_pair_receipt_s = 0.0
        self.last_pair_stamp_s = 0.0
        self.last_imu_receipt_s = 0.0
        self.last_joints_receipt_s = 0.0
        self.last_control_s = 0.0
        self.latest_gyro = np.zeros(3)
        self.last_quality = None
        self.dropped_lidar_pairs = 0
        self.last_cloud_decode_seconds = {"front": 0.0, "rear": 0.0}
        self.last_pair_queue_age_seconds = 0.0
        self.last_adapter_seconds = 0.0
        self.last_odometry_seconds = 0.0
        self.last_raster_seconds = 0.0
        self.last_publish_seconds = 0.0
        self.last_pair_total_seconds = 0.0
        self.last_imu_samples = 0
        self.last_imu_span_s = 0.0
        self.cloud_pair_queue: queue.Queue[tuple[CloudFrame, CloudFrame]] = (
            queue.Queue(maxsize=1)
        )
        self.processing_stop = threading.Event()
        self.last_mapping_frames = None

        if not self.localization_only and not self.onnx_dry_run:
            raise ValueError(
                "the legacy PyTorch policy runtime was removed; use --onnx-dry-run "
                "with deployment/scripts/navigation/run_onnx_navigation_stack.sh"
            )

        self.cmd_pub = (
            node.create_publisher(Twist, topics["cmd_vel"], 10)
            if self.publish_legacy_cmd_vel and not self.localization_only and not self.onnx_dry_run else None
        )
        self.steer_pub = (
            node.create_publisher(Steer, topics.get("steer", "/STEER"), 10)
            if self.publish_dds_steer and not self.localization_only and not self.onnx_dry_run else None
        )
        self.onnx_input_pub = (
            node.create_publisher(
                Float32MultiArray,
                topics.get("onnx_input", "/s10/navigation/onnx_input"),
                2,
            )
            if self.onnx_dry_run else None
        )
        self.odom_pub = node.create_publisher(Odometry, topics["odometry"], 20)
        self.health_pub = node.create_publisher(Bool, "/s10/localization/healthy", 10)
        self.goal_pub = node.create_publisher(Float32MultiArray, "/s10/navigation/goal_body", 10)
        self.waypoint_pub = node.create_publisher(Int32, "/s10/navigation/waypoint_index", 10)
        self.diagnostic_pub = node.create_publisher(
            String, topics.get("diagnostics", "/s10/localization/diagnostics"), 10
        )
        node.create_subscription(
            PointCloud2, topics["front_cloud"], lambda msg: self._cloud("front", msg),
            cloud_qos, callback_group=self.cloud_callback_group,
        )
        node.create_subscription(
            PointCloud2, topics["rear_cloud"], lambda msg: self._cloud("rear", msg),
            cloud_qos, callback_group=self.cloud_callback_group,
        )
        node.create_subscription(
            ImuData, topics["robot_imu"], self._imu, imu_qos,
            callback_group=self.imu_callback_group,
        )
        node.create_subscription(
            JointsData, topics["robot_joints"], self._joints, joints_qos,
            callback_group=self.joints_callback_group,
        )
        node.create_timer(0.1, self._watchdog)
        self.processing_thread = threading.Thread(
            target=self._processing_loop,
            name="s10_lidar_odometry",
            daemon=True,
        )
        self.processing_thread.start()
        mode = (
            "LOCALIZATION_ONLY" if self.localization_only
            else ("ONNX_DRY_RUN" if self.onnx_dry_run else "NAVIGATION")
        )
        alignment_mode = (
            f"recording start anchor {self.record_start_anchor}"
            if self.record_start_anchor is not None
            else (
                f"automatic start alignment from {anchor_source}"
                if self.reference_start_anchor is not None else "legacy configured start pose"
            )
        )
        node.get_logger().warning(
            f"Hardware {mode} ready with motion "
            f"{'ENABLED' if self.enable_motion else 'DISABLED'}; "
            f"nonzero commands remain quality-gated; {alignment_mode}"
        )

    def _imu(self, message) -> None:
        receipt = time.monotonic()
        self.last_imu_receipt_s = receipt
        data = message.data
        rpy_deg = np.asarray((data.roll, data.pitch, data.yaw), dtype=np.float64)
        quaternion = quaternion_wxyz_from_rpy_deg(rpy_deg)
        self.latest_gyro = np.asarray((data.omega_x, data.omega_y, data.omega_z), dtype=np.float64)
        acceleration = np.asarray((data.acc_x, data.acc_y, data.acc_z), dtype=np.float64)
        self.sensor_buffer.append_imu(
            _stamp_seconds(message.header.stamp), receipt, quaternion,
            acceleration, self.latest_gyro,
        )
        if self.map_recorder is not None:
            self.map_recorder.record_imu(
                sensor_stamp_s=_stamp_seconds(message.header.stamp),
                receipt_monotonic_s=receipt,
                rpy_deg=rpy_deg,
                acceleration=acceleration,
                angular_velocity=self.latest_gyro,
            )

    def _joints(self, message) -> None:
        self.last_joints_receipt_s = time.monotonic()
        joints = message.data.joints_data
        velocity = np.asarray([joints[index].velocity for index in self.wheel_indices])
        torque = np.asarray([joints[index].torque for index in self.wheel_indices])
        self.sensor_buffer.update_wheels(velocity, torque)

    def _cloud(self, side: str, message) -> None:
        receipt = time.monotonic()
        try:
            points, timestamps, rings = _read_cloud(
                message, max_points=self.max_cloud_points_per_lidar
            )
            self.last_cloud_decode_seconds[side] = time.monotonic() - receipt
            frame = CloudFrame(points, timestamps, rings, _stamp_seconds(message.header.stamp), receipt)
            pair = self.synchronizer.push(side, frame)
            if pair is not None:
                if self.cloud_pair_queue.full():
                    try:
                        self.cloud_pair_queue.get_nowait()
                        self.dropped_lidar_pairs += 1
                    except queue.Empty:
                        pass
                self.cloud_pair_queue.put_nowait(pair)
        except Exception as error:
            self._stop()
            self.node.get_logger().error(f"{side} cloud processing failed: {error}")

    def _processing_loop(self) -> None:
        while not self.processing_stop.is_set():
            try:
                pair = self.cloud_pair_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                self._process_pair(*pair)
            except Exception as error:
                if self.processing_stop.is_set() or not self.node.context.ok():
                    return
                self._stop()
                self.node.get_logger().error(f"LiDAR odometry processing failed: {error}")

    def close(self) -> None:
        self.processing_stop.set()
        self.processing_thread.join(timeout=3.0)
        if self.processing_thread.is_alive():
            self.node.get_logger().warning("LiDAR odometry worker did not stop cleanly")
        if self.map_localization_worker is not None:
            self.map_localization_worker.close()
        if self.map_recorder is not None:
            if self.last_mapping_frames is not None:
                self._record_mapping_keyframe(*self.last_mapping_frames, force=True)
            self.map_recorder.close()

    @staticmethod
    def _mapping_points_body(
        front_frame: CloudFrame,
        rear_frame: CloudFrame,
        front_adapter: AiryPointCloudAdapter,
        rear_adapter: AiryPointCloudAdapter,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        clouds = []
        timestamps = []
        rings = []
        has_timestamps = True
        has_rings = True
        for frame, adapter in (
            (front_frame, front_adapter), (rear_frame, rear_adapter)
        ):
            points = np.asarray(frame.points_sensor, dtype=np.float64)
            if not adapter.points_in_body_frame:
                points = (
                    adapter.extrinsic.position_body
                    + points @ adapter.extrinsic.rotation_body_sensor.T
                )
            clouds.append(points)
            frame_timestamps = np.asarray(frame.point_timestamps).reshape(-1)
            frame_rings = (
                np.asarray(frame.rings).reshape(-1)
                if frame.rings is not None else np.empty(0)
            )
            has_timestamps &= len(frame_timestamps) == len(points)
            has_rings &= len(frame_rings) == len(points)
            timestamps.append(frame_timestamps)
            rings.append(frame_rings)
        return (
            np.ascontiguousarray(np.concatenate(clouds, axis=0)),
            np.ascontiguousarray(np.concatenate(timestamps), dtype=np.float64)
            if has_timestamps else np.empty(0, dtype=np.float64),
            np.ascontiguousarray(np.concatenate(rings), dtype=np.int32)
            if has_rings else np.empty(0, dtype=np.int32),
        )

    def _record_mapping_keyframe(
        self,
        front_frame: CloudFrame,
        rear_frame: CloudFrame,
        quality,
        *,
        force: bool = False,
    ) -> None:
        if self.map_recorder is None or quality is None:
            return
        self.last_mapping_frames = (front_frame, rear_frame, quality)
        mapping_points, point_timestamps, rings = self._mapping_points_body(
            front_frame, rear_frame, self.front_adapter, self.rear_adapter
        )
        saved = self.map_recorder.consider(
            stamp_s=max(front_frame.stamp_s, rear_frame.stamp_s),
            position_odom_m=self.odometry.position_w,
            rotation_odom_body=self.odometry.rotation_wb,
            points_body_m=mapping_points,
            point_timestamps_s=point_timestamps,
            rings=rings,
            diagnostics={
                "covariance_trace": self.odometry.covariance_trace,
                "icp_accepted": self.odometry.icp_update_accepted,
                "icp_translation_correction_m": float(np.linalg.norm(
                    self.odometry.icp_translation_correction
                )),
                "icp_rotation_correction_deg": self.odometry.icp_residual_rotation_deg,
                "pair_skew_s": abs(front_frame.stamp_s - rear_frame.stamp_s),
                "quality_healthy": bool(quality.healthy),
                "quality_state": str(quality.state),
                "quality_reasons": list(quality.reasons),
            },
            force=force,
        )
        if saved and self.map_recorder.accepted_count % 20 == 0:
            self.node.get_logger().info(
                "Mapping keyframes queued: "
                f"{self.map_recorder.accepted_count}, "
                f"written={self.map_recorder.written_count}, "
                f"dropped={self.map_recorder.dropped_count}"
            )

    def _merge_clouds(self, front, rear) -> tuple[np.ndarray, np.ndarray]:
        points = np.concatenate((front.points_body, rear.points_body), axis=0)
        if front.has_point_timestamps and rear.has_point_timestamps:
            timestamps = np.concatenate((front.point_timestamps, rear.point_timestamps))
        else:
            timestamps = np.empty(0, dtype=np.float64)
        if self.max_registration_points and len(points) > self.max_registration_points:
            indices = np.linspace(
                0,
                len(points) - 1,
                self.max_registration_points,
                dtype=np.int64,
            )
            points = np.ascontiguousarray(points[indices])
            if len(timestamps):
                timestamps = np.ascontiguousarray(timestamps[indices])
        return points, timestamps

    def _process_pair(self, front_frame: CloudFrame, rear_frame: CloudFrame) -> None:
        pair_started = time.monotonic()
        receipt = max(front_frame.receipt_s, rear_frame.receipt_s)
        self.last_pair_queue_age_seconds = max(0.0, pair_started - receipt)
        latest_imu = self.sensor_buffer.latest()
        if latest_imu is None:
            return
        # Cloud discovery can precede the first DDS joint sample at startup.
        # Wait for a complete sensor set instead of reporting a false fault.
        if not self.last_joints_receipt_s:
            return
        if receipt - self.last_imu_receipt_s > self.max_staleness_s:
            raise RuntimeError("robot IMU topic is stale")
        if receipt - self.last_joints_receipt_s > self.max_joint_staleness_s:
            raise RuntimeError("robot joint topic is stale")
        if self.odometry is None:
            initial_pose = self.route.initial_pose()
            front = self.front_adapter.convert(
                front_frame, initial_pose[:3], quat_wxyz_to_rotmat(initial_pose[3:]),
                build_raster=False,
            )
            rear = self.rear_adapter.convert(
                rear_frame, initial_pose[:3], quat_wxyz_to_rotmat(initial_pose[3:]),
                build_raster=False,
            )
            points, timestamps = self._merge_clouds(front, rear)
            if self.start_anchor_accumulator is not None:
                wheel_linear = (
                    self.sensor_buffer.wheel_qvel
                    * np.asarray(self.wheel_signs, dtype=np.float64)
                    * self.wheel_radius_m
                )
                stationary = bool(
                    np.max(np.abs(wheel_linear))
                    <= self.start_alignment_config.stationary_wheel_speed_mps
                    and np.linalg.norm(self.latest_gyro)
                    <= self.start_alignment_config.stationary_gyro_rps
                )
                if stationary and not self.start_anchor_accumulator.frames:
                    self.start_anchor_initial_imu = np.asarray(latest_imu[2]).copy()
                complete = self.start_anchor_accumulator.add(
                    points, stationary=stationary
                )
                self.last_pair_receipt_s = receipt
                self.last_pair_stamp_s = max(front_frame.stamp_s, rear_frame.stamp_s)
                if not complete:
                    self._stop()
                    return
                merged_start = self.start_anchor_accumulator.merged()
                if self.record_start_anchor is not None:
                    if self.start_anchor_initial_imu is None:
                        raise RuntimeError("start anchor has no synchronized IMU orientation")
                    save_start_anchor(
                        self.record_start_anchor,
                        merged_start,
                        self.start_anchor_initial_imu,
                        frame_count=self.start_alignment_config.capture_frames,
                        voxel_size_m=self.start_alignment_config.voxel_size_m,
                    )
                    self.node.get_logger().warning(
                        f"Automatic start anchor saved: {self.record_start_anchor} "
                        f"({len(merged_start)} points)"
                    )
                else:
                    self.start_alignment_result = align_start_anchor(
                        self.reference_start_anchor,
                        merged_start,
                        self.start_alignment_config,
                    )
                    position, rotation = compose_aligned_initial_pose(
                        initial_pose, self.start_alignment_result
                    )
                    initial_pose = np.concatenate(
                        (position, _rotmat_to_quat_wxyz(rotation))
                    )
                    result = self.start_alignment_result
                    self.node.get_logger().warning(
                        "Automatic start alignment complete: "
                        f"yaw={result.yaw_deg:+.3f}deg "
                        f"translation={result.translation_reference_live_m.tolist()}m "
                        f"rmse={result.rmse_m:.3f}m "
                        f"overlap={result.overlap_fraction:.3f}"
                    )
                self.start_anchor_accumulator = None
            self.odometry = DualLidarImuWheelEskfOdometry(
                initial_pose,
                latest_imu[2],
                config=LocalOdometryConfig(
                    wheel_radius=self.wheel_radius_m,
                    wheel_signs=self.wheel_signs,
                    icp_threads=int(self.localization_config.get("icp_threads", 4)),
                    icp_max_iterations=int(
                        self.localization_config.get("icp_max_iterations", 80)
                    ),
                    enable_deskew=True,
                    voxel_size=float(self.localization_config.get("voxel_size_m", 0.15)),
                    orientation_nis_threshold=float(
                        self.localization_config.get("orientation_nis_threshold", 0.0)
                    ),
                    orientation_yaw_sigma_deg=float(
                        self.localization_config.get("orientation_yaw_sigma_deg", 0.5)
                    ),
                    lidar_nis_threshold=float(
                        self.localization_config.get("lidar_nis_threshold", 0.0)
                    ),
                    enable_point_coupling=bool(
                        self.localization_config.get("enable_point_coupling", False)
                    ),
                    point_coupling_sigma=float(
                        self.localization_config.get("point_coupling_sigma_m", 0.08)
                    ),
                    point_coupling_effective_points=float(
                        self.localization_config.get("point_coupling_effective_points", 120.0)
                    ),
                    point_coupling_max_points=int(
                        self.localization_config.get("point_coupling_max_points", 1200)
                    ),
                    enable_zero_velocity_update=bool(
                        self.localization_config.get("enable_zero_velocity_update", False)
                    ),
                    enable_motion_constraints=bool(
                        self.localization_config.get("enable_motion_constraints", False)
                    ),
                ),
            )
            initial_odom_pose = _pose_matrix(
                self.odometry.position_w, self.odometry.rotation_wb
            )
            self.previous_local_position = initial_odom_pose[:3, 3].copy()
            if self.map_localization_enabled:
                self.map_localizer = ContinuousMapLocalizer(
                    self.map_localization_dir,
                    initial_odom_pose,
                    initial_odom_pose,
                    config=self.map_localization_config,
                )
                self.map_localization_worker.start(self.map_localizer)
            front = self.front_adapter.convert(
                front_frame, initial_pose[:3], quat_wxyz_to_rotmat(initial_pose[3:]),
                build_raster=False,
            )
            rear = self.rear_adapter.convert(
                rear_frame, initial_pose[:3], quat_wxyz_to_rotmat(initial_pose[3:]),
                build_raster=False,
            )
            points, timestamps = self._merge_clouds(front, rear)
            self.odometry.initialize_points(points, timestamps)
            self.last_pair_receipt_s = receipt
            self.last_pair_stamp_s = max(front_frame.stamp_s, rear_frame.stamp_s)
            self.last_pair_total_seconds = time.monotonic() - pair_started
            self._publish_state(front, rear, None)
            return

        current_pair_stamp_s = max(front_frame.stamp_s, rear_frame.stamp_s)
        if current_pair_stamp_s <= self.last_pair_stamp_s:
            self.dropped_lidar_pairs += 1
            return
        try:
            history = self.sensor_buffer.history_by_receipt(
                self.last_pair_receipt_s, receipt
            )
        except RuntimeError as error:
            # A single-threaded ROS executor may deliver an old cloud callback
            # before the queued IMU callbacks that cover it.  Never propagate
            # or register such a frame against an incomplete time interval.
            self.dropped_lidar_pairs += 1
            if self.dropped_lidar_pairs <= 3 or self.dropped_lidar_pairs % 100 == 0:
                self.node.get_logger().warning(
                    f"dropping LiDAR pair without complete IMU coverage: {error}; "
                    f"dropped={self.dropped_lidar_pairs}"
                )
            return
        adapter_started = time.monotonic()
        front = self.front_adapter.convert(
            front_frame,
            self.odometry.position_w,
            self.odometry.rotation_wb,
            build_raster=False,
        )
        rear = self.rear_adapter.convert(
            rear_frame,
            self.odometry.position_w,
            self.odometry.rotation_wb,
            build_raster=False,
        )
        self.last_adapter_seconds = time.monotonic() - adapter_started
        points, timestamps = self._merge_clouds(front, rear)
        odometry_started = time.monotonic()
        self.odometry.update_points(points, history, timestamps)
        self.last_odometry_seconds = time.monotonic() - odometry_started
        local_pose = _pose_matrix(self.odometry.position_w, self.odometry.rotation_wb)
        # This scalar is only a weak, monotonic route-progress prior. Using
        # LiDAR-odometry pose increments makes it explode after sparse or
        # rejected scan matches. The adaptive wheel estimate remains bounded
        # and does not feed back into the metric map pose.
        self.local_traveled_distance_m += abs(
            float(self.odometry.wheel_prior_distance)
        )
        self.previous_local_position = local_pose[:3, 3].copy()
        self.map_localization_updates += 1
        if (
            self.map_localizer is not None
            and self.local_traveled_distance_m
            >= self.map_localization_config.minimum_motion_before_matching_m
            and (
                self.map_localization_worker.snapshot().submitted == 0
                or self.map_localization_updates
                % self.map_localization_stride == 0
            )
        ):
            wheel_linear = (
                self.sensor_buffer.wheel_qvel
                * np.asarray(self.wheel_signs, dtype=np.float64)
                * self.wheel_radius_m
            )
            stationary_for_relocalization = bool(
                np.max(np.abs(wheel_linear))
                <= self.start_alignment_config.stationary_wheel_speed_mps
                and np.linalg.norm(self.latest_gyro)
                <= self.start_alignment_config.stationary_gyro_rps
            )
            self.map_localization_worker.submit(
                points,
                local_pose,
                traveled_distance_m=self.local_traveled_distance_m,
                allow_large_relocalization=stationary_for_relocalization,
            )
        self.last_pair_receipt_s = receipt
        self.last_pair_stamp_s = current_pair_stamp_s
        raster_started = time.monotonic()
        if not self.localization_only:
            front = self.front_adapter.convert(
                front_frame, self.odometry.position_w, self.odometry.rotation_wb
            )
            rear = self.rear_adapter.convert(
                rear_frame, self.odometry.position_w, self.odometry.rotation_wb
            )
        self.last_raster_seconds = time.monotonic() - raster_started
        if not self.localization_only:
            map_pose = self._navigation_pose()
            if self._map_localization_healthy(receipt):
                self.route.update(map_pose[:3, 3], receipt)

        point_timestamps_present = front.has_point_timestamps and rear.has_point_timestamps
        quality = self.gate.evaluate(
            pair_skew_s=abs(front_frame.stamp_s - rear_frame.stamp_s),
            points_per_lidar=(len(front.points_body), len(rear.points_body)),
            point_timestamps_present=(point_timestamps_present or not self.require_point_timestamps),
            imu_samples=len(history["time"]),
            imu_span_s=float(history["time"][-1] - history["time"][0]),
            covariance_trace=self.odometry.covariance_trace,
            icp_accepted=self.odometry.icp_update_accepted,
        )
        self.last_imu_samples = len(history["time"])
        self.last_imu_span_s = float(history["time"][-1] - history["time"][0])
        self.last_quality = quality
        self._record_mapping_keyframe(front_frame, rear_frame, quality)
        publish_started = time.monotonic()
        self.last_pair_total_seconds = time.monotonic() - pair_started
        self._publish_state(front, rear, quality)
        self.last_publish_seconds = time.monotonic() - publish_started
        if not quality.healthy:
            self._stop()
        if receipt - self.last_control_s >= self.control_period_s:
            self.last_control_s = receipt
            self._control(front, rear, quality)

    def _control(self, front, rear, quality) -> None:
        if (
            self.localization_only
            or not quality.healthy
            or not self._map_localization_healthy(time.monotonic())
            or self.route.complete
        ):
            self._stop()
            return
        if self.onnx_dry_run:
            self._publish_onnx_input(front, rear)
            return
        raise RuntimeError("unsupported navigation mode without the ONNX runtime")

    def _publish_onnx_input(self, front, rear) -> None:
        """Publish a transport-only observation; no robot command topic is touched."""

        if self.onnx_input_pub is None:
            raise RuntimeError("ONNX dry-run publisher is unavailable")
        velocity_map = self.odometry.initial_rotation_wb @ self.odometry.filter.velocity
        velocity_body = self.odometry.rotation_wb.T @ velocity_map
        projected_gravity = self.odometry.rotation_wb.T @ np.asarray((0.0, 0.0, -1.0))
        map_pose = self._navigation_pose()
        goal_body = self.route.goal_body(map_pose[:3, 3], map_pose[:3, :3])
        front_distance, front_z = _pool_native_raster(
            front.distance_native, front.world_z_native
        )
        rear_distance, rear_z = _pool_native_raster(
            rear.distance_native, rear.world_z_native
        )
        # C++ runner inserts its previous raw action between gravity and goal.
        packed = np.concatenate((
            np.asarray(velocity_body, dtype=np.float32),
            np.asarray(self.latest_gyro, dtype=np.float32),
            np.asarray(projected_gravity, dtype=np.float32),
            np.asarray(goal_body, dtype=np.float32),
            front_distance.reshape(-1),
            rear_distance.reshape(-1),
            front_z.reshape(-1),
            rear_z.reshape(-1),
        ))
        if packed.shape != (34573,) or not np.isfinite(packed).all():
            raise RuntimeError(f"invalid packed ONNX observation: {packed.shape}")
        message = self.messages["Float32MultiArray"]()
        message.data = packed.tolist()
        self.onnx_input_pub.publish(message)

    def _publish_command(self, command: np.ndarray) -> None:
        if self.cmd_pub is not None:
            message = self.messages["Twist"]()
            message.linear.x = float(command[0])
            message.linear.y = float(command[1])
            message.angular.z = float(command[2])
            self.cmd_pub.publish(message)
        if self.steer_pub is not None:
            message = self.messages["Steer"]()
            message.data.x = float(command[0] / self.dds_runner_command_scale[0])
            message.data.y = float(command[1] / self.dds_runner_command_scale[1])
            message.data.yaw = float(command[2] / self.dds_runner_command_scale[2])
            self.steer_pub.publish(message)

    def _navigation_pose(self) -> np.ndarray:
        local_pose = _pose_matrix(
            self.odometry.position_w, self.odometry.rotation_wb
        )
        snapshot = (
            self.map_localization_worker.snapshot()
            if self.map_localization_worker is not None else None
        )
        result = snapshot.result if snapshot is not None else None
        return _map_corrected_pose(local_pose, result)

    def _map_localization_healthy(self, now_s: float) -> bool:
        if (
            not self.map_localization_enabled
            or self.local_traveled_distance_m
            < self.map_localization_config.minimum_motion_before_matching_m
        ):
            return True
        snapshot = self.map_localization_worker.snapshot()
        if snapshot.last_accepted_monotonic_s > 0.0:
            reference_s = snapshot.last_accepted_monotonic_s
        else:
            reference_s = snapshot.first_submitted_monotonic_s
        return bool(
            reference_s > 0.0
            and now_s - reference_s <= self.map_localization_max_coast_s
            and snapshot.last_error is None
        )

    def _stop(self) -> None:
        self._publish_command(np.zeros(3, dtype=np.float64))

    def _publish_state(self, front, rear, quality) -> None:
        if self.odometry is None:
            return
        now = self.node.get_clock().now().to_msg()
        odom = self.messages["Odometry"]()
        odom.header.stamp = now
        odom.header.frame_id = "s10_route_map"
        odom.child_frame_id = "base_link"
        map_pose = self._navigation_pose()
        odom.pose.pose.position.x, odom.pose.pose.position.y, odom.pose.pose.position.z = map(float, map_pose[:3, 3])
        quat = _rotmat_to_quat_wxyz(map_pose[:3, :3])
        odom.pose.pose.orientation.w, odom.pose.pose.orientation.x, odom.pose.pose.orientation.y, odom.pose.pose.orientation.z = map(float, quat)
        velocity_map = self.odometry.initial_rotation_wb @ self.odometry.filter.velocity
        velocity_body = self.odometry.rotation_wb.T @ velocity_map
        odom.twist.twist.linear.x, odom.twist.twist.linear.y, odom.twist.twist.linear.z = map(float, velocity_body)
        odom.twist.twist.angular.x, odom.twist.twist.angular.y, odom.twist.twist.angular.z = map(float, self.latest_gyro)
        self.odom_pub.publish(odom)

        healthy = self.messages["Bool"]()
        healthy.data = bool(
            quality and quality.healthy
            and self._map_localization_healthy(time.monotonic())
        )
        self.health_pub.publish(healthy)
        if not self.localization_only:
            goal = self.messages["Float32MultiArray"]()
            goal.data = self.route.goal_body(map_pose[:3, 3], map_pose[:3, :3]).tolist()
            self.goal_pub.publish(goal)
            waypoint = self.messages["Int32"]()
            waypoint.data = self.route.active_index
            self.waypoint_pub.publish(waypoint)
        diagnostic_now_s = time.monotonic()
        if diagnostic_now_s - self.last_diagnostics_s >= self.diagnostics_period_s:
            self.last_diagnostics_s = diagnostic_now_s
            self._publish_diagnostic(front, rear, quality)

    def _publish_diagnostic(self, front, rear, quality) -> None:
        now_s = time.monotonic()
        map_healthy = self._map_localization_healthy(now_s)
        map_snapshot = (
            self.map_localization_worker.snapshot()
            if self.map_localization_worker is not None else None
        )
        map_result = map_snapshot.result if map_snapshot is not None else None
        if now_s - self.last_support_height_s >= self.support_height_period_s:
            support_points = np.concatenate(
                (front.points_body, rear.points_body), axis=0
            )
            if len(support_points) > 6000:
                support_points = support_points[::max(1, len(support_points) // 6000)]
            self.last_support_height = estimate_support_height_map(
                support_points,
                self.odometry.position_w,
                self.odometry.rotation_wb,
                expected_clearance_m=self.route.initial_base_height_m,
            )
            self.last_support_height_s = now_s
        support_height, support_count, support_rmse = self.last_support_height
        values = {
            "runtime_mode": (
                "localization_only" if self.localization_only
                else ("onnx_dry_run" if self.onnx_dry_run else "navigation")
            ),
            "motion_enabled": self.enable_motion,
            "waypoint": "disabled" if self.localization_only else self.route.active_index,
            "local_traveled_distance_m": self.local_traveled_distance_m,
            "map_localization_updates": self.map_localization_updates,
            "front_points": len(front.points_body),
            "rear_points": len(rear.points_body),
            "covariance_trace": self.odometry.covariance_trace,
            "slip_score": self.odometry.slip_score,
            "wheel_sigma": self.odometry.wheel_velocity_sigma,
            "wheel_speed_mps": self.odometry.wheel_speed,
            "wheel_prior_distance_m": self.odometry.wheel_prior_distance,
            "icp_accepted": self.odometry.icp_update_accepted,
            "icp_translation_correction_m": float(np.linalg.norm(
                self.odometry.icp_translation_correction
            )),
            "icp_rotation_correction_deg": self.odometry.icp_residual_rotation_deg,
            "lidar_nis": self.odometry.lidar_innovation_nis,
            "lidar_noise_inflation": self.odometry.lidar_noise_inflation,
            "orientation_nis": self.odometry.orientation_innovation_nis,
            "orientation_noise_inflation": self.odometry.orientation_noise_inflation,
            "zero_velocity_update": self.odometry.zero_velocity_applied,
            "point_coupling_used": self.odometry.point_coupling_used,
            "point_coupling_count": self.odometry.point_coupling_count,
            "point_coupling_rmse": self.odometry.point_coupling_rmse,
            "point_coupling_condition": self.odometry.point_coupling_condition,
            "point_coupling_seconds": self.odometry.point_coupling_seconds,
            "registration_seconds": self.odometry.registration_seconds,
            "cloud_decode_front_seconds": self.last_cloud_decode_seconds["front"],
            "cloud_decode_rear_seconds": self.last_cloud_decode_seconds["rear"],
            "pair_queue_age_seconds": self.last_pair_queue_age_seconds,
            "adapter_seconds": self.last_adapter_seconds,
            "odometry_seconds": self.last_odometry_seconds,
            "raster_seconds": self.last_raster_seconds,
            "publish_seconds": self.last_publish_seconds,
            "pair_total_seconds": self.last_pair_total_seconds,
            "dropped_lidar_pairs": self.dropped_lidar_pairs,
            "imu_samples": self.last_imu_samples,
            "imu_span_s": self.last_imu_span_s,
            "support_height_map_m": support_height,
            "support_height_points": support_count,
            "support_height_rmse_m": support_rmse,
            "reasons": ",".join(quality.reasons) if quality else "initializing",
        }
        if map_snapshot is not None:
            values.update({
                "map_async_submitted": map_snapshot.submitted,
                "map_async_completed": map_snapshot.completed,
                "map_async_dropped_requests": map_snapshot.dropped_requests,
                "map_async_result_sequence": map_snapshot.result_sequence,
                "map_async_update_seconds": map_snapshot.last_update_seconds,
                "map_async_queue_age_seconds": map_snapshot.last_queue_age_seconds,
                "map_async_last_error": map_snapshot.last_error or "none",
            })
        if map_result is not None:
            values.update({
                "map_localization_mode": map_result.mode,
                "map_observation_accepted": map_result.observation_accepted,
                "map_route_index": map_result.selected_route_index,
                "map_selected_submap": map_result.selected_submap,
                "map_correction_translation_m": map_result.correction_translation_m,
                "map_correction_yaw_deg": map_result.correction_yaw_deg,
                "map_fitness": map_result.fitness,
                "map_rmse_m": map_result.rmse_m,
                "map_descriptor_distance": map_result.descriptor_distance,
                "map_control_healthy": self._map_localization_healthy(now_s),
            })
        if map_snapshot is not None:
            candidate_audits = map_snapshot.candidate_audits
            values["map_candidate_count"] = len(candidate_audits)
            if candidate_audits:
                best_candidate = min(
                    candidate_audits,
                    key=lambda item: (
                        len(item["rejection_reasons"]),
                        float(item["rmse_m"]),
                        -float(item["fitness"]),
                    ),
                )
                values.update({
                    "map_best_candidate_route_index": best_candidate[
                        "candidate_route_index"
                    ],
                    "map_best_candidate_fitness": best_candidate["fitness"],
                    "map_best_candidate_rmse_m": best_candidate["rmse_m"],
                    "map_best_candidate_descriptor_distance": best_candidate[
                        "descriptor_distance"
                    ],
                    "map_best_candidate_translation_innovation_m": best_candidate[
                        "translation_innovation_m"
                    ],
                    "map_best_candidate_yaw_innovation_deg": best_candidate[
                        "yaw_innovation_deg"
                    ],
                    "map_best_candidate_rejections": ",".join(
                        best_candidate["rejection_reasons"]
                    ) or "accepted",
                })
        if self.start_alignment_result is not None:
            values.update({
                "start_alignment_yaw_deg": self.start_alignment_result.yaw_deg,
                "start_alignment_translation_m": np.linalg.norm(
                    self.start_alignment_result.translation_reference_live_m
                ),
                "start_alignment_rmse_m": self.start_alignment_result.rmse_m,
                "start_alignment_overlap": self.start_alignment_result.overlap_fraction,
                "start_alignment_candidate_margin_m": (
                    self.start_alignment_result.candidate_margin_m
                ),
            })
        message = self.messages["String"]()
        message.data = json.dumps(
            {
                "name": "s10_local_navigation",
                "hardware_id": "S10_dual_Airy",
                "level": (
                    0 if quality and quality.healthy and map_healthy
                    else (2 if not map_healthy or (quality and quality.state == "LOST") else 1)
                ),
                "state": (
                    "MAP_LOST" if quality and quality.healthy and not map_healthy
                    else (quality.state if quality else "INITIALIZING")
                ),
                # Keep values scalar and transport-neutral.  The collector
                # parses these strings back into booleans/floats as needed.
                "values": {str(key): str(value) for key, value in values.items()},
            },
            ensure_ascii=False,
        )
        self.diagnostic_pub.publish(message)

    def _watchdog(self) -> None:
        now = time.monotonic()
        stale = (
            (self.last_pair_receipt_s and now - self.last_pair_receipt_s > self.max_staleness_s)
            or (self.last_imu_receipt_s and now - self.last_imu_receipt_s > self.max_staleness_s)
            or (
                self.last_joints_receipt_s
                and now - self.last_joints_receipt_s > self.max_joint_staleness_s
            )
        )
        if stale:
            self._stop()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path,
        default=REPO_ROOT / "deployment/config/hardware_navigation.yaml",
    )
    parser.add_argument("--enable-motion", action="store_true")
    parser.add_argument(
        "--route-file",
        type=Path,
        help="override the route in the hardware config with a collected route YAML",
    )
    parser.add_argument(
        "--localization-only",
        action="store_true",
        help="run localization without loading a policy or publishing commands",
    )
    parser.add_argument(
        "--map-dir",
        type=Path,
        help="enable frozen route-map correction from this directory",
    )
    parser.add_argument(
        "--onnx-dry-run",
        action="store_true",
        help=(
            "publish packed live observations for the isolated ONNX runner; "
            "never create robot command publishers"
        ),
    )
    parser.add_argument(
        "--record-start-anchor",
        type=Path,
        help="automatically save the initial stationary dual-LiDAR anchor",
    )
    parser.add_argument(
        "--start-anchor",
        type=Path,
        help=(
            "align the initial stationary dual-LiDAR frames to this anchor; "
            "intended for read-only route-map localization"
        ),
    )
    parser.add_argument(
        "--record-map",
        type=Path,
        help="write bounded offline-mapping keyframes (requires --localization-only)",
    )
    args, ros_args = parser.parse_known_args()
    if args.enable_motion and (args.localization_only or args.onnx_dry_run):
        parser.error("--enable-motion cannot be combined with a read-only mode")
    if args.localization_only and args.onnx_dry_run:
        parser.error("--localization-only and --onnx-dry-run are mutually exclusive")
    if not args.localization_only and not args.onnx_dry_run:
        parser.error(
            "the legacy PyTorch navigation mode was removed; use --onnx-dry-run "
            "or deployment/scripts/navigation/run_onnx_navigation_stack.sh"
        )
    if args.record_start_anchor is not None and not args.localization_only:
        parser.error("--record-start-anchor is only valid with --localization-only")
    if args.start_anchor is not None and not args.localization_only:
        parser.error("--start-anchor is only valid with --localization-only")
    if args.start_anchor is not None and not args.start_anchor.expanduser().is_file():
        parser.error(f"start anchor not found: {args.start_anchor.expanduser()}")
    if args.record_map is not None and not args.localization_only:
        parser.error("--record-map is only valid with --localization-only")
    if args.map_dir is not None and not args.localization_only:
        parser.error("--map-dir override is only valid with --localization-only")
    if not args.localization_only and args.route_file is None:
        parser.error("autonomous navigation requires --route-file from waypoint collection")

    import rclpy
    from rclpy._rclpy_pybind11 import RCLError
    from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor

    rclpy.init(args=ros_args)
    node = rclpy.create_node("s10_sru_hardware_navigation")
    config = load_hardware_config(args.config.expanduser().resolve())
    if args.map_dir is not None:
        map_dir = args.map_dir.expanduser().resolve()
        manifest = map_dir / "localization_map_manifest.json"
        if not manifest.is_file():
            parser.error(f"route map manifest not found: {manifest}")
        map_config = config.setdefault("map_localization", {})
        map_config["enabled"] = True
        map_config["map_dir"] = str(map_dir)
    if args.route_file is not None:
        route_file = args.route_file.expanduser().resolve()
        summary = validate_route(route_file)
        node.get_logger().info(
            f"Validated route with {summary['waypoint_count']} waypoints, "
            f"length={summary['total_length_m']:.2f}m"
        )
        apply_route_file(config, route_file)
    runtime = S10HardwareNavigationNode(
        node,
        config,
        enable_motion_override=args.enable_motion,
        localization_only=args.localization_only,
        onnx_dry_run=args.onnx_dry_run,
        record_start_anchor=args.record_start_anchor,
        reference_start_anchor=args.start_anchor,
        record_map=args.record_map,
    )
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException, RCLError):
        pass
    finally:
        executor.shutdown(timeout_sec=5.0)
        runtime.close()
        if rclpy.ok():
            runtime._stop()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
