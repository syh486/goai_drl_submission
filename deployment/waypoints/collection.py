"""Transport-independent waypoint collection and route serialization."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any

import numpy as np
import yaml

from deployment.localization.start_alignment import anchor_path_for_route


@dataclass(frozen=True)
class PoseSample:
    stamp_s: float
    receipt_s: float
    position: np.ndarray
    quaternion_wxyz: np.ndarray
    linear_velocity: np.ndarray
    angular_velocity: np.ndarray
    healthy: bool
    quality_state: str
    diagnostics: dict[str, Any]


@dataclass(frozen=True)
class CollectionLimits:
    window_s: float = 1.0
    min_samples: int = 5
    start_min_samples: int = 1
    min_healthy_fraction: float = 0.9
    max_linear_speed_mps: float = 0.08
    max_angular_speed_rps: float = 0.12
    max_xy_spread_m: float = 0.08
    max_z_spread_m: float = 0.06
    max_covariance_trace: float = 25.0
    standing_base_clearance_m: float = 0.425
    min_waypoint_spacing_m: float = 0.25
    max_pose_age_s: float = 0.3

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "CollectionLimits":
        fields = cls.__dataclass_fields__
        return cls(**{key: config[key] for key in fields if key in config})


class CollectionRejected(RuntimeError):
    """The current localization window is not safe enough to record."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _normalized_quaternion_median(quaternions: np.ndarray) -> np.ndarray:
    values = np.asarray(quaternions, dtype=np.float64)
    reference = values[0]
    aligned = values.copy()
    aligned[(aligned @ reference) < 0.0] *= -1.0
    result = np.median(aligned, axis=0)
    norm = float(np.linalg.norm(result))
    if norm < 1.0e-8 or not np.isfinite(norm):
        raise CollectionRejected("orientation window is invalid")
    return result / norm


def _atomic_dump(path: Path, payload: dict[str, Any], *, json_format: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        delete=False,
    ) as stream:
        temporary = Path(stream.name)
        if json_format:
            json.dump(payload, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        else:
            yaml.safe_dump(payload, stream, sort_keys=False, allow_unicode=True)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class WaypointCollectionSession:
    """Collect robust waypoint positions from a quality-gated odometry stream."""

    def __init__(
        self,
        output_path: Path,
        *,
        limits: CollectionLimits | None = None,
        description: str = "route collected onboard",
        resume: bool = False,
        anchor_path: Path | None = None,
    ) -> None:
        self.output_path = output_path.expanduser().resolve()
        self.audit_path = self.output_path.with_suffix(".quality.json")
        self.anchor_path = (
            anchor_path.expanduser().resolve()
            if anchor_path is not None else anchor_path_for_route(self.output_path)
        )
        self.limits = limits or CollectionLimits()
        self.samples: deque[PoseSample] = deque(maxlen=1000)
        self.events: list[dict[str, Any]] = []
        self.nodes: list[dict[str, Any]] = []
        self.created_at = _utc_now()
        self.description = description
        if self.output_path.exists():
            if not resume:
                raise FileExistsError(
                    f"route already exists: {self.output_path}; use --resume to append"
                )
            self._load_existing()

    def _load_existing(self) -> None:
        payload = yaml.safe_load(self.output_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or payload.get("coordinate_frame") != "s10_route_map":
            raise ValueError("existing route is not an s10_route_map waypoint file")
        nodes = payload.get("nodes")
        if not isinstance(nodes, list):
            raise ValueError("existing route has no nodes list")
        self.nodes = list(nodes)
        self.description = str(payload.get("description", self.description))
        collection = payload.get("collection", {})
        self.created_at = str(collection.get("collected_at", self.created_at))
        if self.audit_path.exists():
            audit = json.loads(self.audit_path.read_text(encoding="utf-8"))
            self.events = list(audit.get("events", []))

    def append(self, sample: PoseSample) -> None:
        arrays = (
            np.asarray(sample.position),
            np.asarray(sample.quaternion_wxyz),
            np.asarray(sample.linear_velocity),
            np.asarray(sample.angular_velocity),
        )
        if [value.shape for value in arrays] != [(3,), (4,), (3,), (3,)]:
            raise ValueError("pose sample has an invalid vector shape")
        if not all(np.isfinite(value).all() for value in arrays):
            raise ValueError("pose sample contains non-finite values")
        self.samples.append(sample)

    def _window(
        self, now_s: float, *, min_samples: int | None = None
    ) -> list[PoseSample]:
        required_samples = self.limits.min_samples if min_samples is None else int(min_samples)
        if required_samples < 1:
            raise ValueError("min_samples must be positive")
        if not self.samples:
            raise CollectionRejected("no odometry samples received")
        if now_s - self.samples[-1].receipt_s > self.limits.max_pose_age_s:
            raise CollectionRejected("odometry stream is stale")
        end = self.samples[-1].receipt_s
        window = [
            sample for sample in self.samples
            if end - sample.receipt_s <= self.limits.window_s
        ]
        if len(window) < required_samples:
            raise CollectionRejected(
                f"only {len(window)} pose samples in the {self.limits.window_s:.2f}s window; "
                f"need {required_samples}"
            )
        # Use the newest independent estimates. Older moving samples should not
        # keep a waypoint blocked after enough stationary LiDAR updates arrive.
        return window[-required_samples:]

    def mark(
        self,
        *,
        name: str | None = None,
        tags: list[str] | None = None,
        force_close_spacing: bool = False,
        now_s: float | None = None,
        min_samples: int | None = None,
    ) -> dict[str, Any]:
        window = self._window(
            time.monotonic() if now_s is None else float(now_s),
            min_samples=min_samples,
        )
        healthy_fraction = float(np.mean([sample.healthy for sample in window]))
        if healthy_fraction < self.limits.min_healthy_fraction:
            raise CollectionRejected(
                f"healthy fraction {healthy_fraction:.2f} is below "
                f"{self.limits.min_healthy_fraction:.2f}"
            )

        linear = np.asarray([np.linalg.norm(sample.linear_velocity) for sample in window])
        angular = np.asarray([np.linalg.norm(sample.angular_velocity) for sample in window])
        if float(np.percentile(linear, 95)) > self.limits.max_linear_speed_mps:
            raise CollectionRejected("robot is still translating")
        if float(np.percentile(angular, 95)) > self.limits.max_angular_speed_rps:
            raise CollectionRejected("robot is still rotating")

        positions = np.stack([sample.position for sample in window])
        center = np.median(positions, axis=0)
        xy_spread = float(np.max(np.linalg.norm(positions[:, :2] - center[:2], axis=1)))
        z_spread = float(np.max(np.abs(positions[:, 2] - center[2])))
        if xy_spread > self.limits.max_xy_spread_m:
            raise CollectionRejected(f"XY pose spread {xy_spread:.3f}m is too large")
        if z_spread > self.limits.max_z_spread_m:
            raise CollectionRejected(f"Z pose spread {z_spread:.3f}m is too large")

        covariance_values = []
        for sample in window:
            try:
                covariance_values.append(float(sample.diagnostics["covariance_trace"]))
            except (KeyError, TypeError, ValueError):
                pass
        if not covariance_values:
            raise CollectionRejected("localization covariance is unavailable")
        covariance_trace = float(np.median(covariance_values))
        if not np.isfinite(covariance_trace) or covariance_trace > self.limits.max_covariance_trace:
            raise CollectionRejected(f"covariance trace {covariance_trace:.3f} is too large")

        support_heights = []
        for sample in window:
            try:
                value = float(sample.diagnostics["support_height_map_m"])
            except (KeyError, TypeError, ValueError):
                continue
            if np.isfinite(value):
                support_heights.append(value)
        terrain_position = center.copy()
        height_source = "base_height_minus_calibrated_clearance"
        if len(support_heights) >= max(3, len(window) // 2):
            support_height = float(np.median(support_heights))
            support_spread = float(np.max(np.abs(np.asarray(support_heights) - support_height)))
            if support_spread <= self.limits.max_z_spread_m:
                terrain_position[2] = support_height
                height_source = "local_lidar_support_plane"
            else:
                terrain_position[2] -= self.limits.standing_base_clearance_m
        else:
            terrain_position[2] -= self.limits.standing_base_clearance_m
        if self.nodes and not force_close_spacing:
            previous = np.asarray(self.nodes[-1]["position"], dtype=np.float64)
            spacing = float(np.linalg.norm(terrain_position[:2] - previous[:2]))
            if spacing < self.limits.min_waypoint_spacing_m:
                raise CollectionRejected(
                    f"waypoint is only {spacing:.3f}m from the previous point; "
                    "use --force only when a close point is intentional"
                )

        quaternion = _normalized_quaternion_median(
            np.stack([sample.quaternion_wxyz for sample in window])
        )
        index = len(self.nodes)
        node = {
            "name": name or f"waypoint_{index:03d}",
            "position": [round(float(value), 4) for value in terrain_position],
            "orientation_wxyz": [round(float(value), 7) for value in quaternion],
            "source": "onboard_pose_window_median",
            "height_source": height_source,
            "quality": "GOOD",
            "tags": list(tags or []),
        }
        event = {
            "event": "mark",
            "index": index,
            "name": node["name"],
            "recorded_at": _utc_now(),
            "sensor_stamp_s": float(np.median([sample.stamp_s for sample in window])),
            "position_base_median_m": center.tolist(),
            "position_terrain_m": terrain_position.tolist(),
            "height_source": height_source,
            "orientation_wxyz": quaternion.tolist(),
            "window": {
                "duration_s": self.limits.window_s,
                "sample_span_s": float(window[-1].receipt_s - window[0].receipt_s),
                "samples": len(window),
                "healthy_fraction": healthy_fraction,
                "linear_speed_p95_mps": float(np.percentile(linear, 95)),
                "angular_speed_p95_rps": float(np.percentile(angular, 95)),
                "xy_spread_m": xy_spread,
                "z_spread_m": z_spread,
                "covariance_trace_median": covariance_trace,
            },
            "latest_diagnostics": dict(window[-1].diagnostics),
        }
        self.nodes.append(node)
        self.events.append(event)
        self.save()
        return event

    def undo(self) -> dict[str, Any]:
        if not self.nodes:
            raise CollectionRejected("there is no waypoint to undo")
        removed = self.nodes.pop()
        event = {
            "event": "undo",
            "removed_index": len(self.nodes),
            "removed_name": removed.get("name"),
            "removed_lidar_observation": removed.get("lidar_observation"),
            "recorded_at": _utc_now(),
        }
        self.events.append(event)
        self.save()
        return event

    def attach_lidar_observation(
        self, waypoint_index: int, metadata: dict[str, Any]
    ) -> None:
        if not 0 <= waypoint_index < len(self.nodes):
            raise IndexError(f"waypoint index out of range: {waypoint_index}")
        self.nodes[waypoint_index]["lidar_observation"] = dict(metadata)
        for event in reversed(self.events):
            if event.get("event") == "mark" and event.get("index") == waypoint_index:
                event["lidar_observation"] = dict(metadata)
                break
        self.save()

    def route_payload(self) -> dict[str, Any]:
        alignment = {
            "method": "dual_lidar_start_anchor_se3",
            "anchor_file": os.path.relpath(
                self.anchor_path, start=self.output_path.parent
            ),
            "automatic_capture": True,
            "automatic_navigation_alignment": True,
        }
        if self.anchor_path.is_file():
            alignment["anchor_sha256"] = _sha256(self.anchor_path)
        return {
            "schema_version": 2,
            "description": self.description,
            "coordinate_frame": "s10_route_map",
            "initial_map_yaw_deg": 0.0,
            "initial_base_height_m": self.limits.standing_base_clearance_m,
            "collection": {
                "source": "onboard_kiss_icp_eskf",
                "start_pose_initialized": True,
                "collected_at": self.created_at,
                "last_saved_at": _utc_now(),
                "position_window_s": self.limits.window_s,
                "height_source": "local_lidar_support_plane_with_clearance_fallback",
                "standing_base_clearance_m": self.limits.standing_base_clearance_m,
                "external_localization": False,
            },
            "initial_alignment": alignment,
            "skipped_waypoints": [],
            "nodes": self.nodes,
        }

    def save(self) -> None:
        _atomic_dump(self.output_path, self.route_payload(), json_format=False)
        _atomic_dump(
            self.audit_path,
            {
                "schema_version": 1,
                "route_file": str(self.output_path),
                "waypoint_count": len(self.nodes),
                "events": self.events,
            },
            json_format=True,
        )

    def status(self) -> dict[str, Any]:
        latest = self.samples[-1] if self.samples else None
        return {
            "output": str(self.output_path),
            "waypoint_count": len(self.nodes),
            "pose_samples": len(self.samples),
            "localization_healthy": bool(latest and latest.healthy),
            "quality_state": latest.quality_state if latest else "NO_ODOMETRY",
            "last_waypoint": self.nodes[-1]["name"] if self.nodes else None,
        }
