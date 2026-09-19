"""Validate and summarize a mapping keyframe session."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from deployment.localization.start_alignment import (
    StartAlignmentConfig,
    StartAnchor,
    align_start_anchor,
)


def _rotation_distance_deg(first: np.ndarray, second: np.ndarray) -> float:
    delta = np.asarray(first).T @ np.asarray(second)
    cosine = np.clip((float(np.trace(delta)) - 1.0) * 0.5, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def validate(
    session_dir: Path,
    start_frame: int = 0,
    end_frame: int | None = None,
) -> dict[str, object]:
    root = session_dir.expanduser().resolve()
    metadata_path = root / "session.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"mapping session metadata is missing: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    all_files = sorted((root / "keyframes").glob("*.npz"))
    files = all_files[start_frame:end_frame]
    if not files:
        raise ValueError("mapping session contains no keyframes")

    stamps = []
    positions = []
    rotations = []
    point_counts = []
    icp_rejected = 0
    unhealthy_keyframes = 0
    timestamped_keyframes = 0
    ring_keyframes = 0
    total_bytes = metadata_path.stat().st_size
    first_points = None
    last_points = None
    for selected_index, path in enumerate(files):
        total_bytes += path.stat().st_size
        with np.load(path, allow_pickle=False) as payload:
            if int(payload["schema_version"]) != 1:
                raise ValueError(f"unsupported keyframe schema: {path}")
            expected_index = int(path.stem)
            if int(payload["index"]) != expected_index:
                raise ValueError(f"non-contiguous keyframe index: {path}")
            points = np.asarray(payload["points_body_m"])
            position = np.asarray(payload["position_odom_m"], dtype=np.float64)
            rotation = np.asarray(payload["rotation_odom_body"], dtype=np.float64)
            if points.ndim != 2 or points.shape[1] != 3 or len(points) < 100:
                raise ValueError(f"invalid keyframe points: {path}")
            if position.shape != (3,) or rotation.shape != (3, 3):
                raise ValueError(f"invalid keyframe pose: {path}")
            if not np.isfinite(points).all() or not np.isfinite(position).all():
                raise ValueError(f"non-finite keyframe data: {path}")
            diagnostics = json.loads(str(payload["diagnostics_json"]))
            icp_rejected += int(not diagnostics.get("icp_accepted", True))
            unhealthy_keyframes += int(not diagnostics.get("quality_healthy", True))
            stamps.append(float(payload["stamp_s"]))
            positions.append(position)
            rotations.append(rotation)
            point_counts.append(len(points))
            if "point_timestamps_s" in payload.files:
                point_timestamps = np.asarray(payload["point_timestamps_s"])
                rings = np.asarray(payload["rings"])
                if point_timestamps.shape not in {(0,), (len(points),)}:
                    raise ValueError(f"invalid per-point timestamps: {path}")
                if rings.shape not in {(0,), (len(points),)}:
                    raise ValueError(f"invalid ring values: {path}")
                timestamped_keyframes += int(len(point_timestamps) == len(points))
                ring_keyframes += int(len(rings) == len(points))
            if selected_index == 0:
                first_points = points.copy()
            if selected_index == len(files) - 1:
                last_points = points.copy()

    stamps_array = np.asarray(stamps)
    if len(stamps_array) > 1 and np.any(np.diff(stamps_array) <= 0.0):
        raise ValueError("mapping keyframe timestamps are not strictly increasing")
    positions_array = np.asarray(positions)
    intervals = np.diff(stamps_array)
    steps = np.linalg.norm(np.diff(positions_array, axis=0), axis=1)
    rotations_step = np.asarray([
        _rotation_distance_deg(rotations[index - 1], rotations[index])
        for index in range(1, len(rotations))
    ])
    scan_closure: dict[str, object]
    try:
        alignment = align_start_anchor(
            StartAnchor(
                points_reference_body_m=first_points,
                initial_imu_quaternion_wxyz=np.asarray((1.0, 0.0, 0.0, 0.0)),
                frame_count=1,
                voxel_size_m=0.15,
            ),
            last_points,
            StartAlignmentConfig(
                capture_frames=3,
                voxel_size_m=0.15,
                min_range_m=0.45,
                max_range_m=30.0,
                structural_min_z_m=-0.20,
                max_registration_points=5000,
                yaw_search_range_deg=60.0,
                yaw_search_step_deg=2.0,
                coarse_candidates=5,
                max_initial_translation_m=2.0,
                max_correspondence_m=0.80,
                trim_fraction=0.70,
                icp_iterations=30,
            ),
        )
        scan_closure = {
            "matched": True,
            "translation_m": float(np.linalg.norm(
                alignment.translation_reference_live_m
            )),
            "yaw_deg": float(alignment.yaw_deg),
            "rmse_m": float(alignment.rmse_m),
            "overlap_fraction": float(alignment.overlap_fraction),
            "candidate_margin_m": float(alignment.candidate_margin_m),
        }
    except Exception as error:
        scan_closure = {"matched": False, "error": str(error)}

    imu_summary = None
    imu_metadata = metadata.get("imu_stream")
    if imu_metadata is not None:
        imu_path = root / str(imu_metadata["file"])
        if not imu_path.is_file():
            raise ValueError(f"IMU stream is missing: {imu_path}")
        columns = list(imu_metadata.get("columns", ()))
        values = np.fromfile(imu_path, dtype=np.float64)
        if not columns or len(values) % len(columns):
            raise ValueError("IMU stream size does not match its schema")
        rows = values.reshape(-1, len(columns))
        if len(rows) != int(metadata.get("imu_samples", -1)):
            raise ValueError("IMU stream sample count does not match session metadata")
        if len(rows) and (
            not np.isfinite(rows).all() or np.any(np.diff(rows[:, 0]) < 0.0)
        ):
            raise ValueError("IMU stream contains invalid or reordered samples")
        total_bytes += imu_path.stat().st_size
        gaps = np.diff(rows[:, 0]) if len(rows) > 1 else np.empty(0)
        imu_summary = {
            "samples": len(rows),
            "columns": columns,
            "duration_s": float(rows[-1, 0] - rows[0, 0]) if len(rows) > 1 else 0.0,
            "first_stamp_s": float(rows[0, 0]) if len(rows) else None,
            "last_stamp_s": float(rows[-1, 0]) if len(rows) else None,
            "lidar_start_minus_imu_start_s": (
                float(stamps_array[0] - rows[0, 0]) if len(rows) else None
            ),
            "imu_end_minus_lidar_end_s": (
                float(rows[-1, 0] - stamps_array[-1]) if len(rows) else None
            ),
            "max_gap_s": float(gaps.max(initial=0.0)),
            "p99_gap_s": float(np.percentile(gaps, 99)) if len(gaps) else 0.0,
        }

    summary = {
        "state": metadata.get("state", "unknown"),
        "source_start_frame": int(files[0].stem),
        "source_end_frame": int(files[-1].stem),
        "keyframes": len(files),
        "duration_s": float(stamps_array[-1] - stamps_array[0]),
        "median_keyframe_interval_s": float(np.median(intervals)) if len(intervals) else 0.0,
        "p95_keyframe_interval_s": float(np.percentile(intervals, 95)) if len(intervals) else 0.0,
        "max_keyframe_interval_s": float(intervals.max(initial=0.0)),
        "local_path_length_m": float(steps.sum()),
        "raw_closure_xyz_m": float(np.linalg.norm(positions_array[-1] - positions_array[0])),
        "raw_closure_xy_m": float(np.linalg.norm(positions_array[-1, :2] - positions_array[0, :2])),
        "raw_closure_rotation_deg": _rotation_distance_deg(rotations[0], rotations[-1]),
        "max_keyframe_translation_m": float(steps.max(initial=0.0)),
        "max_keyframe_rotation_deg": float(rotations_step.max(initial=0.0)),
        "points_min": int(min(point_counts)),
        "points_median": float(np.median(point_counts)),
        "points_max": int(max(point_counts)),
        "point_timestamp_keyframes": timestamped_keyframes,
        "ring_keyframes": ring_keyframes,
        "imu_stream": imu_summary,
        "icp_rejected_keyframes": icp_rejected,
        "unhealthy_keyframes": unhealthy_keyframes,
        "dropped_keyframes": int(metadata.get("dropped_keyframes", 0)),
        "size_mib": float(total_bytes / (1024 ** 2)),
        "first_last_scan_closure": scan_closure,
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("session_dir", type=Path)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--end-frame", type=int)
    args = parser.parse_args()
    print(json.dumps(
        validate(args.session_dir, args.start_frame, args.end_frame),
        indent=2,
        sort_keys=True,
    ))


if __name__ == "__main__":
    main()
