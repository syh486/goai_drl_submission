"""Bounded, crash-tolerant keyframe recording for offline route mapping."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import queue
import shutil
import tempfile
import threading
import time

import numpy as np


@dataclass(frozen=True)
class MappingRecordingConfig:
    translation_m: float = 0.20
    rotation_deg: float = 4.0
    min_interval_s: float = 0.20
    record_every_interval: bool = False
    voxel_size_m: float = 0.15
    min_range_m: float = 0.45
    max_range_m: float = 30.0
    max_points: int = 24000
    queue_size: int = 8
    sync_interval_frames: int = 0
    reserve_free_gib: float = 1.0
    record_imu_stream: bool = True

    @classmethod
    def from_config(cls, values: dict | None) -> "MappingRecordingConfig":
        values = values or {}
        return cls(
            translation_m=float(values.get("translation_m", 0.20)),
            rotation_deg=float(values.get("rotation_deg", 4.0)),
            min_interval_s=float(values.get("min_interval_s", 0.20)),
            record_every_interval=bool(values.get("record_every_interval", False)),
            voxel_size_m=float(values.get("voxel_size_m", 0.15)),
            min_range_m=float(values.get("min_range_m", 0.45)),
            max_range_m=float(values.get("max_range_m", 30.0)),
            max_points=int(values.get("max_points", 24000)),
            queue_size=int(values.get("queue_size", 8)),
            sync_interval_frames=int(values.get("sync_interval_frames", 0)),
            reserve_free_gib=float(values.get("reserve_free_gib", 1.0)),
            record_imu_stream=bool(values.get("record_imu_stream", True)),
        )

    def validate(self) -> None:
        if self.translation_m <= 0.0 or self.rotation_deg <= 0.0:
            raise ValueError("mapping keyframe motion thresholds must be positive")
        if self.min_interval_s < 0.0 or self.voxel_size_m <= 0.0:
            raise ValueError("mapping keyframe timing/voxel values are invalid")
        if not 0.0 < self.min_range_m < self.max_range_m:
            raise ValueError("mapping point range is invalid")
        if self.max_points < 100 or self.queue_size < 1:
            raise ValueError("mapping keyframe limits are invalid")
        if self.sync_interval_frames < 0:
            raise ValueError("mapping sync interval must be non-negative")
        if self.reserve_free_gib < 0.25:
            raise ValueError("mapping recorder must reserve at least 0.25 GiB")


@dataclass(frozen=True)
class _PendingKeyframe:
    index: int
    stamp_s: float
    position_odom_m: np.ndarray
    rotation_odom_body: np.ndarray
    points_body_m: np.ndarray
    point_timestamps_s: np.ndarray
    rings: np.ndarray
    diagnostics: dict[str, object]


def _rotation_distance_deg(first: np.ndarray, second: np.ndarray) -> float:
    delta = np.asarray(first).T @ np.asarray(second)
    cosine = np.clip((float(np.trace(delta)) - 1.0) * 0.5, -1.0, 1.0)
    return math.degrees(math.acos(cosine))


def _voxel_downsample(points: np.ndarray, voxel_size_m: float, max_points: int) -> np.ndarray:
    cloud = np.asarray(points, dtype=np.float64)
    if cloud.ndim != 2 or cloud.shape[1] != 3:
        raise ValueError(f"mapping points must have shape [N,3], got {cloud.shape}")
    cloud = cloud[np.isfinite(cloud).all(axis=1)]
    if not len(cloud):
        return np.empty((0, 3), dtype=np.float32)
    keys = np.floor(cloud / float(voxel_size_m)).astype(np.int32)
    _, indices = np.unique(keys, axis=0, return_index=True)
    indices.sort()
    cloud = cloud[indices]
    if len(cloud) > max_points:
        sample = np.linspace(0, len(cloud) - 1, max_points, dtype=np.int64)
        cloud = cloud[sample]
    return np.ascontiguousarray(cloud, dtype=np.float32)


class MappingKeyframeRecorder:
    """Write sparse body-frame scans without blocking the localization worker."""

    SCHEMA_VERSION = 1

    def __init__(self, output_dir: Path, config: MappingRecordingConfig) -> None:
        config.validate()
        self.output_dir = output_dir.expanduser().resolve()
        self.keyframe_dir = self.output_dir / "keyframes"
        if self.output_dir.exists() and any(self.output_dir.iterdir()):
            raise FileExistsError(f"mapping output is not empty: {self.output_dir}")
        self.keyframe_dir.mkdir(parents=True, exist_ok=True)
        self.config = config
        self.reserve_bytes = int(config.reserve_free_gib * (1024 ** 3))
        self.pending: queue.Queue[_PendingKeyframe | None] = queue.Queue(config.queue_size)
        self.writer_error: Exception | None = None
        self.last_position: np.ndarray | None = None
        self.last_rotation: np.ndarray | None = None
        self.last_stamp_s = float("-inf")
        self.accepted_count = 0
        self.written_count = 0
        self.dropped_count = 0
        self.started_wall_s = time.time()
        self.imu_sample_count = 0
        self.imu_path = self.output_dir / "imu_samples.f64"
        self.imu_stream = (
            self.imu_path.open("wb", buffering=1024 * 1024)
            if config.record_imu_stream else None
        )
        self._write_session("recording")
        self.writer = threading.Thread(
            target=self._writer_loop, name="s10_mapping_writer", daemon=True
        )
        self.writer.start()

    def _write_session(self, state: str) -> None:
        payload = {
            "schema_version": self.SCHEMA_VERSION,
            "state": state,
            "created_unix_s": self.started_wall_s,
            "updated_unix_s": time.time(),
            "config": asdict(self.config),
            "accepted_keyframes": self.accepted_count,
            "written_keyframes": self.written_count,
            "dropped_keyframes": self.dropped_count,
            "imu_samples": self.imu_sample_count,
            "imu_stream": (
                {
                    "file": self.imu_path.name,
                    "dtype": "float64",
                    "columns": [
                        "sensor_stamp_s", "receipt_monotonic_s",
                        "roll_deg", "pitch_deg", "yaw_deg",
                        "acc_x", "acc_y", "acc_z",
                        "gyro_x", "gyro_y", "gyro_z",
                    ],
                }
                if self.imu_stream is not None else None
            ),
        }
        target = self.output_dir / "session.json"
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=self.output_dir,
            prefix=".session.", delete=False,
        ) as stream:
            temporary = Path(stream.name)
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)

    def _raise_writer_error(self) -> None:
        if self.writer_error is not None:
            raise RuntimeError(f"mapping writer failed: {self.writer_error}") from self.writer_error

    def consider(
        self,
        *,
        stamp_s: float,
        position_odom_m: np.ndarray,
        rotation_odom_body: np.ndarray,
        points_body_m: np.ndarray,
        point_timestamps_s: np.ndarray | None = None,
        rings: np.ndarray | None = None,
        diagnostics: dict[str, object] | None = None,
        force: bool = False,
    ) -> bool:
        self._raise_writer_error()
        position = np.asarray(position_odom_m, dtype=np.float64)
        rotation = np.asarray(rotation_odom_body, dtype=np.float64)
        if position.shape != (3,) or rotation.shape != (3, 3):
            raise ValueError("mapping keyframe pose has invalid shape")
        if not np.isfinite(position).all() or not np.isfinite(rotation).all():
            raise ValueError("mapping keyframe pose is not finite")
        # Shutdown may ask to force-save the most recently processed LiDAR
        # pair. Never duplicate or reorder a scan which was already accepted.
        if self.last_position is not None and float(stamp_s) <= self.last_stamp_s:
            return False
        if not force and float(stamp_s) - self.last_stamp_s < self.config.min_interval_s:
            return False

        translation = (
            float("inf") if self.last_position is None
            else float(np.linalg.norm(position - self.last_position))
        )
        rotation_deg = (
            float("inf") if self.last_rotation is None
            else _rotation_distance_deg(self.last_rotation, rotation)
        )
        if (
            not force
            and not self.config.record_every_interval
            and translation < self.config.translation_m
            and rotation_deg < self.config.rotation_deg
        ):
            return False

        points = np.asarray(points_body_m, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3 or len(points) < 100:
            raise ValueError(f"mapping points must have shape [N,3], got {points.shape}")
        # Keep voxelization and range filtering off the localization thread.
        # The writer has ample throughput for motion-triggered keyframes.
        points = np.ascontiguousarray(points, dtype=np.float32)
        point_timestamps = np.asarray(
            np.empty(0) if point_timestamps_s is None else point_timestamps_s,
            dtype=np.float64,
        ).reshape(-1)
        ring_values = np.asarray(
            np.empty(0) if rings is None else rings, dtype=np.int32
        ).reshape(-1)
        if len(point_timestamps) not in (0, len(points)):
            raise ValueError("point timestamps must be empty or match mapping points")
        if len(ring_values) not in (0, len(points)):
            raise ValueError("rings must be empty or match mapping points")

        item = _PendingKeyframe(
            index=self.accepted_count,
            stamp_s=float(stamp_s),
            position_odom_m=position.copy(),
            rotation_odom_body=rotation.copy(),
            points_body_m=points,
            point_timestamps_s=point_timestamps.copy(),
            rings=ring_values.copy(),
            diagnostics=dict(diagnostics or {}),
        )
        try:
            self.pending.put_nowait(item)
        except queue.Full:
            self.dropped_count += 1
            return False
        self.accepted_count += 1
        self.last_position = position.copy()
        self.last_rotation = rotation.copy()
        self.last_stamp_s = float(stamp_s)
        return True

    def record_imu(
        self,
        *,
        sensor_stamp_s: float,
        receipt_monotonic_s: float,
        rpy_deg: np.ndarray,
        acceleration: np.ndarray,
        angular_velocity: np.ndarray,
    ) -> None:
        if self.imu_stream is None:
            return
        row = np.concatenate((
            np.asarray((sensor_stamp_s, receipt_monotonic_s), dtype=np.float64),
            np.asarray(rpy_deg, dtype=np.float64).reshape(3),
            np.asarray(acceleration, dtype=np.float64).reshape(3),
            np.asarray(angular_velocity, dtype=np.float64).reshape(3),
        ))
        if not np.isfinite(row).all():
            return
        self.imu_stream.write(row.tobytes())
        self.imu_sample_count += 1
        # Flush once per second at the nominal 200 Hz rate so the detached
        # launcher can prove that IMU samples are reaching durable storage
        # before the operator starts moving.
        if self.imu_sample_count % 200 == 0:
            self.imu_stream.flush()

    def _writer_loop(self) -> None:
        try:
            while True:
                item = self.pending.get()
                try:
                    if item is None:
                        return
                    if shutil.disk_usage(self.output_dir).free <= self.reserve_bytes:
                        raise RuntimeError(
                            f"free disk space reached reserved {self.config.reserve_free_gib:.2f} GiB"
                        )
                    self._write_keyframe(item)
                    self.written_count += 1
                    if self.written_count % 20 == 0:
                        self._write_session("recording")
                    if (
                        self.config.sync_interval_frames > 0
                        and self.written_count % self.config.sync_interval_frames == 0
                    ):
                        os.sync()
                finally:
                    self.pending.task_done()
        except Exception as error:  # propagated on the localization thread
            self.writer_error = error

    def _write_keyframe(self, item: _PendingKeyframe) -> None:
        output = self.keyframe_dir / f"{item.index:06d}.npz"
        points = np.asarray(item.points_body_m, dtype=np.float32)
        point_timestamps = item.point_timestamps_s
        rings = item.rings
        ranges = np.linalg.norm(points, axis=1)
        valid = (
            np.isfinite(points).all(axis=1)
            & np.isfinite(ranges)
            & (ranges >= self.config.min_range_m)
            & (ranges <= self.config.max_range_m)
        )
        source_point_count = int(np.count_nonzero(valid))
        points = points[valid]
        if len(point_timestamps):
            point_timestamps = point_timestamps[valid]
        if len(rings):
            rings = rings[valid]
        keys = np.floor(points / self.config.voxel_size_m).astype(np.int32)
        _, indices = np.unique(keys, axis=0, return_index=True)
        indices.sort()
        if len(indices) > self.config.max_points:
            sample = np.linspace(0, len(indices) - 1, self.config.max_points, dtype=np.int64)
            indices = indices[sample]
        points = np.ascontiguousarray(points[indices], dtype=np.float32)
        if len(point_timestamps):
            point_timestamps = np.ascontiguousarray(point_timestamps[indices])
        if len(rings):
            rings = np.ascontiguousarray(rings[indices])
        if len(points) < 100:
            raise RuntimeError("mapping keyframe has too few valid points after filtering")
        payload = {
            "schema_version": np.asarray(self.SCHEMA_VERSION, dtype=np.int64),
            "index": np.asarray(item.index, dtype=np.int64),
            "stamp_s": np.asarray(item.stamp_s, dtype=np.float64),
            "position_odom_m": item.position_odom_m,
            "rotation_odom_body": item.rotation_odom_body,
            "points_body_m": points,
            "point_timestamps_s": point_timestamps,
            "rings": rings,
            "source_point_count": np.asarray(source_point_count, dtype=np.int64),
            "diagnostics_json": np.asarray(json.dumps(item.diagnostics, sort_keys=True)),
        }
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=self.keyframe_dir, prefix=f".{output.name}.", delete=False
        ) as stream:
            temporary = Path(stream.name)
            # Voxelized float32 scans are already compact; uncompressed NPZ keeps
            # CPU cost and localization interference predictable on the AGX.
            np.savez(stream, **payload)
            stream.flush()
        os.replace(temporary, output)

    def close(self) -> None:
        if self.writer.is_alive():
            self.pending.put(None)
            self.writer.join(timeout=120.0)
        if self.writer.is_alive():
            raise RuntimeError("mapping writer did not stop within 120 seconds")
        self._raise_writer_error()
        if self.imu_stream is not None and not self.imu_stream.closed:
            self.imu_stream.flush()
            os.fsync(self.imu_stream.fileno())
            self.imu_stream.close()
        # Keyframes are atomically renamed as they are written. One filesystem
        # sync here makes all buffered scan data durable before marking the
        # session complete without stalling localization on every frame.
        os.sync()
        self._write_session("complete")
