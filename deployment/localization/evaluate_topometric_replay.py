"""Leakage-free topological replay without a pseudo ground-truth trajectory."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time

import numpy as np

from deployment.localization.continuous_map_localization import (
    ContinuousLocalizationConfig,
    ContinuousMapLocalizer,
)
from deployment.mapping.mapping_geometry import load_points
from deployment.common.trajectory_io import load_aligned_trajectory, prepare_trajectory_poses


def _maximum_rejection_streak(accepted: np.ndarray) -> int:
    maximum = 0
    current = 0
    for value in accepted:
        current = 0 if value else current + 1
        maximum = max(maximum, current)
    return maximum


def evaluate_topometric_replay(
    map_dir: Path,
    heldout_session: Path,
    heldout_trajectory: Path,
    *,
    localization_stride: int = 2,
    localization_config: ContinuousLocalizationConfig = ContinuousLocalizationConfig(),
    output_report: Path | None = None,
    allow_reference_session: bool = False,
) -> dict[str, object]:
    root = map_dir.expanduser().resolve()
    session = heldout_session.expanduser().resolve()
    manifest = json.loads(
        (root / "localization_map_manifest.json").read_text(encoding="utf-8")
    )
    reference_sessions = {
        str(Path(value).expanduser().resolve())
        for value in manifest["reference_sessions"]
    }
    is_reference_session = str(session) in reference_sessions
    if is_reference_session and not allow_reference_session:
        raise ValueError("held-out session was used to build this map")
    if localization_stride < 1:
        raise ValueError("localization_stride must be positive")

    aligned = load_aligned_trajectory(session, heldout_trajectory)
    raw_kiss = aligned.poses
    files = list(aligned.keyframe_files)
    if len(files) < 100:
        raise ValueError("held-out trajectory has too few aligned frames")
    stamps = aligned.timestamps_s.tolist()
    intervals = np.diff(np.asarray(stamps, dtype=np.float64))
    intervals = intervals[intervals > 0.0]
    sensor_rate_hz = float(1.0 / np.median(intervals))

    pose_mode = str(
        manifest.get("config", {}).get(
            "trajectory_pose_mode", "lio"
        )
    )
    local_motion, pose_diagnostics = prepare_trajectory_poses(aligned, pose_mode)
    increments = np.linalg.norm(np.diff(local_motion[:, :3, 3], axis=0), axis=1)
    traveled = np.concatenate((np.zeros(1), np.cumsum(increments)))
    localizer = ContinuousMapLocalizer(
        root, None, local_motion[0], config=localization_config
    )
    records = []
    started_at = time.perf_counter()
    for frame in range(localization_stride, len(files), localization_stride):
        result = localizer.update(
            load_points(files[frame]),
            local_motion[frame],
            traveled_distance_m=float(traveled[frame]),
        )
        records.append({
            "frame": frame,
            "sensor_stamp_s": stamps[frame],
            "traveled_distance_m": float(traveled[frame]),
            "route_index": result.selected_route_index,
            "odometry_route_index": localizer.last_odometry_route_index,
            "submap_id": result.selected_submap,
            "accepted": result.observation_accepted,
            "mode": result.mode,
            "fitness": result.fitness,
            "rmse_m": result.rmse_m,
            "map_position_m": result.map_from_body[:3, 3].tolist(),
            "map_rotation_matrix": result.map_from_body[:3, :3].tolist(),
        })

    route_indices = np.asarray([item["route_index"] for item in records])
    route_steps = np.diff(route_indices)
    accepted = np.asarray([item["accepted"] for item in records], dtype=bool)
    maximum_rejections = _maximum_rejection_streak(accepted)
    max_route_index = max(
        int(entry.get("route_index", index))
        for index, entry in enumerate(manifest["submaps"])
    )
    elapsed_s = time.perf_counter() - started_at
    coverage = float(np.max(route_indices) / max_route_index)
    regressions = int(np.count_nonzero(route_steps < -1))
    large_jumps = int(np.count_nonzero(
        route_steps > localization_config.route_lookahead_submaps
    ))
    unexpected_large_jumps = int(sum(
        bool(step > localization_config.route_lookahead_submaps)
        and records[index + 1]["mode"] != "relocalized"
        for index, step in enumerate(route_steps)
    ))
    maximum_coast_s = float(
        maximum_rejections * localization_stride / sensor_rate_hz
    )
    startup_distance_m = localization_config.minimum_motion_before_matching_m
    in_motion = np.asarray(
        [item["traveled_distance_m"] >= startup_distance_m for item in records],
        dtype=bool,
    )
    if not np.any(in_motion):
        raise ValueError("held-out replay never reached the localization startup distance")
    in_motion_accepted = accepted[in_motion]
    in_motion_maximum_rejections = _maximum_rejection_streak(in_motion_accepted)
    in_motion_maximum_coast_s = float(
        in_motion_maximum_rejections * localization_stride / sensor_rate_hz
    )
    in_motion_accept_rate = float(np.mean(in_motion_accepted))
    full_record_continuity_passed = bool(maximum_coast_s <= 10.0)
    in_motion_continuity_passed = bool(in_motion_maximum_coast_s <= 10.0)
    summary = {
        "schema_version": 1,
        "evaluation_contract": (
            "development self-replay; scans may be present in the map and "
            "results do not qualify held-out accuracy"
            if is_reference_session else
            "held-out scans are absent from the map; no pseudo ground truth "
            "or held-out progress is passed to the localizer"
        ),
        "map_dir": str(root),
        "heldout_session": str(session),
        "trajectory_alignment": aligned.report(),
        "trajectory_pose_diagnostics": pose_diagnostics,
        "localization_config": asdict(localization_config),
        "reference_sessions": sorted(reference_sessions),
        "reference_session_replay": is_reference_session,
        "recorded_sensor_rate_hz": sensor_rate_hz,
        "effective_localization_rate_hz": sensor_rate_hz / localization_stride,
        "wall_clock_processing_rate_hz": len(records) / elapsed_s,
        "observation_accept_rate": float(np.mean(accepted)),
        "maximum_consecutive_rejections": maximum_rejections,
        "maximum_coast_s": maximum_coast_s,
        "startup_distance_m": startup_distance_m,
        "in_motion_observation_accept_rate": in_motion_accept_rate,
        "in_motion_maximum_consecutive_rejections": in_motion_maximum_rejections,
        "in_motion_maximum_coast_s": in_motion_maximum_coast_s,
        "full_record_continuity_passed": full_record_continuity_passed,
        "in_motion_continuity_passed": in_motion_continuity_passed,
        "route_index_final": int(route_indices[-1]),
        "route_index_max_reached": int(np.max(route_indices)),
        "route_index_count": max_route_index + 1,
        "route_coverage_fraction": coverage,
        "route_index_regressions": regressions,
        "route_index_large_jumps": large_jumps,
        "route_index_unexpected_large_jumps": unexpected_large_jumps,
        "metric_accuracy_qualified": False,
        "topological_replay_passed": bool(
            coverage >= 0.90
            and regressions == 0
            and unexpected_large_jumps == 0
            and in_motion_continuity_passed
        ),
        "qualification_note": (
            "The in-motion gate begins where the localizer is configured to start "
            "map matching. This can qualify a read-only field trial, never autonomous "
            "motion; metric accuracy still needs surveyed points or another independent "
            "reference."
        ),
    }
    output = {"summary": summary, "records": records}
    report_path = (
        root / "heldout_topometric_replay.json"
        if output_report is None
        else output_report.expanduser().resolve()
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map-dir", type=Path, required=True)
    parser.add_argument("--heldout-session", type=Path, required=True)
    parser.add_argument(
        "--heldout-trajectory", type=Path,
        help="timed GLIM traj_lidar.txt or an exactly aligned NPZ/NPY trajectory",
        required=True,
    )
    parser.add_argument("--localization-stride", type=int, default=2)
    parser.add_argument(
        "--registration-backend",
        choices=("kiss_icp", "kiss_vgicp", "gtsam_vgicp", "open3d_gicp"),
        default="kiss_icp",
    )
    parser.add_argument("--query-submap-updates", type=int, default=5)
    parser.add_argument(
        "--allow-reference-session",
        action="store_true",
        help="allow a non-qualifying development self-replay",
    )
    parser.add_argument(
        "--temporal-fusion",
        action="store_true",
        help="bound and damp ordinary tracking updates in SE(3)",
    )
    parser.add_argument(
        "--multisession-consensus",
        action="store_true",
        help=(
            "select route indices from the canonical map and fuse only "
            "repeatable aligned support observations"
        ),
    )
    parser.add_argument(
        "--output-report",
        type=Path,
        help="write the immutable replay report outside the map directory",
    )
    args = parser.parse_args()
    output = evaluate_topometric_replay(
        args.map_dir,
        args.heldout_session,
        args.heldout_trajectory,
        localization_stride=args.localization_stride,
        localization_config=ContinuousLocalizationConfig(
            registration_backend=args.registration_backend,
            query_submap_updates=args.query_submap_updates,
            temporal_fusion_enabled=args.temporal_fusion,
            multisession_consensus_enabled=args.multisession_consensus,
        ),
        output_report=args.output_report,
        allow_reference_session=args.allow_reference_session,
    )
    print(json.dumps(output["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
