"""Atomic front/rear Airy snapshots attached to collected waypoints."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import tempfile

import numpy as np

from deployment.waypoints.collection import CollectionRejected


@dataclass(frozen=True)
class LidarSnapshotFrame:
    points_xyz: np.ndarray
    point_timestamps: np.ndarray
    rings: np.ndarray | None
    stamp_s: float
    receipt_s: float
    frame_id: str


class WaypointLidarSnapshotStore:
    """Keep recent synchronized clouds and persist a few pairs for each mark."""

    def __init__(
        self,
        route_path: Path,
        *,
        pairs_per_waypoint: int = 3,
        max_pair_skew_s: float = 0.03,
        max_age_s: float = 1.5,
        queue_size: int = 30,
    ) -> None:
        if pairs_per_waypoint < 1 or queue_size < pairs_per_waypoint:
            raise ValueError("invalid LiDAR snapshot queue dimensions")
        if max_pair_skew_s <= 0.0 or max_age_s <= 0.0:
            raise ValueError("LiDAR snapshot timing limits must be positive")
        route = route_path.expanduser().resolve()
        self.output_dir = route.with_suffix(".observations")
        self.pairs_per_waypoint = int(pairs_per_waypoint)
        self.max_pair_skew_s = float(max_pair_skew_s)
        self.max_age_s = float(max_age_s)
        self.frames = {
            "front": deque(maxlen=queue_size),
            "rear": deque(maxlen=queue_size),
        }

    def append(self, side: str, frame: LidarSnapshotFrame) -> None:
        if side not in self.frames:
            raise ValueError(f"unknown LiDAR side: {side}")
        points = np.asarray(frame.points_xyz)
        if points.ndim != 2 or points.shape[1] != 3 or not len(points):
            raise ValueError("LiDAR snapshot points must have shape [N,3]")
        timestamps = np.asarray(frame.point_timestamps)
        if timestamps.shape not in {(0,), (len(points),)}:
            raise ValueError("LiDAR snapshot timestamps must be empty or match the point count")
        if timestamps.size and not np.isfinite(timestamps).all():
            raise ValueError("LiDAR snapshot timestamps must be finite")
        if frame.rings is not None:
            rings = np.asarray(frame.rings)
            if rings.shape not in {(0,), (len(points),)}:
                raise ValueError("LiDAR snapshot rings must be empty or match the point count")
        self.frames[side].append(frame)

    def prepare(self, now_s: float) -> list[tuple[LidarSnapshotFrame, LidarSnapshotFrame]]:
        recent = {
            side: [
                frame for frame in frames
                if float(now_s) - frame.receipt_s <= self.max_age_s
            ]
            for side, frames in self.frames.items()
        }
        if not recent["front"] or not recent["rear"]:
            raise CollectionRejected("fresh front/rear LiDAR observations are unavailable")

        pairs = []
        used_rear: set[int] = set()
        for front in reversed(recent["front"]):
            candidates = [
                (abs(front.stamp_s - rear.stamp_s), index, rear)
                for index, rear in enumerate(recent["rear"])
                if index not in used_rear
            ]
            if not candidates:
                continue
            skew, index, rear = min(candidates, key=lambda item: item[0])
            if skew <= self.max_pair_skew_s:
                used_rear.add(index)
                pairs.append((front, rear))
            if len(pairs) >= self.pairs_per_waypoint:
                break
        if len(pairs) < self.pairs_per_waypoint:
            raise CollectionRejected(
                f"only {len(pairs)}/{self.pairs_per_waypoint} synchronized LiDAR pairs "
                f"within {self.max_pair_skew_s:.3f}s"
            )
        return list(reversed(pairs))

    def save(
        self,
        waypoint_index: int,
        pairs: list[tuple[LidarSnapshotFrame, LidarSnapshotFrame]],
    ) -> dict[str, object]:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        output = self.output_dir / f"waypoint_{waypoint_index:03d}.npz"
        payload: dict[str, np.ndarray] = {
            "schema_version": np.asarray(1, dtype=np.int64),
            "waypoint_index": np.asarray(waypoint_index, dtype=np.int64),
            "pair_count": np.asarray(len(pairs), dtype=np.int64),
        }
        skews = []
        point_counts = {"front": [], "rear": []}
        timestamps_available = True
        rings_available = True
        for pair_index, (front, rear) in enumerate(pairs):
            skews.append(abs(front.stamp_s - rear.stamp_s))
            for side, frame in (("front", front), ("rear", rear)):
                prefix = f"pair_{pair_index:02d}_{side}"
                points = np.ascontiguousarray(frame.points_xyz, dtype=np.float32)
                timestamps = np.ascontiguousarray(frame.point_timestamps, dtype=np.float64)
                rings = (
                    np.empty(0, dtype=np.int16)
                    if frame.rings is None
                    else np.ascontiguousarray(frame.rings, dtype=np.int16)
                )
                timestamps_available &= len(timestamps) == len(points)
                rings_available &= len(rings) == len(points)
                payload[f"{prefix}_xyz"] = points
                payload[f"{prefix}_timestamp"] = timestamps
                payload[f"{prefix}_ring"] = rings
                payload[f"{prefix}_stamp_s"] = np.asarray(frame.stamp_s, dtype=np.float64)
                payload[f"{prefix}_frame_id"] = np.asarray(frame.frame_id)
                point_counts[side].append(len(points))

        with tempfile.NamedTemporaryFile(
            mode="wb", dir=self.output_dir, prefix=f".{output.name}.", delete=False
        ) as stream:
            temporary = Path(stream.name)
            np.savez_compressed(stream, **payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, output)
        digest_builder = hashlib.sha256()
        with output.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest_builder.update(chunk)
        digest = digest_builder.hexdigest()
        fields = ["x", "y", "z"]
        if rings_available:
            fields.append("ring")
        if timestamps_available:
            fields.append("timestamp")
        return {
            "file": os.path.relpath(output, start=self.output_dir.parent),
            "sha256": digest,
            "schema_version": 1,
            "pair_count": len(pairs),
            "max_pair_skew_s": float(max(skews)),
            "front_points": point_counts["front"],
            "rear_points": point_counts["rear"],
            "fields": fields,
            "ring_available": rings_available,
            "point_timestamps_available": timestamps_available,
        }

    def delete(self, waypoint_index: int) -> None:
        (self.output_dir / f"waypoint_{waypoint_index:03d}.npz").unlink(missing_ok=True)
