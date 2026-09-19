"""Audit causal KISS/VGICP agreement without changing the online trajectory."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

from deployment.localization.continuous_map_localization import (
    ContinuousLocalizationConfig,
    ContinuousMapLocalizer,
)
from deployment.mapping.mapping_geometry import (
    load_points,
    rotation_distance_deg,
    voxel_downsample,
)
from deployment.common.trajectory_io import load_aligned_trajectory, prepare_trajectory_poses


def _pose(record: dict[str, object]) -> np.ndarray:
    result = np.eye(4, dtype=np.float64)
    result[:3, 3] = np.asarray(record["map_position_m"], dtype=np.float64)
    result[:3, :3] = np.asarray(record["map_rotation_matrix"], dtype=np.float64)
    return result


def _distance(first: np.ndarray, second: np.ndarray) -> tuple[float, float]:
    residual = np.linalg.inv(first) @ second
    return (
        float(np.linalg.norm(residual[:3, 3])),
        rotation_distance_deg(residual[:3, :3]),
    )


def _query(
    files: list[Path],
    poses: np.ndarray,
    frame: int,
    config: ContinuousLocalizationConfig,
) -> tuple[np.ndarray, np.ndarray]:
    current_from_odom = np.linalg.inv(poses[frame])
    chunks = []
    begin = max(0, frame - config.query_submap_updates + 1)
    for index in range(begin, frame + 1):
        points = load_points(files[index])
        ranges = np.linalg.norm(points, axis=1)
        points = points[
            np.isfinite(points).all(axis=1)
            & (ranges >= 0.45)
            & (ranges <= config.max_sensor_range_m)
        ]
        current_from_history = current_from_odom @ poses[index]
        chunks.append(
            points @ current_from_history[:3, :3].T
            + current_from_history[:3, 3]
        )
    fine = voxel_downsample(np.concatenate(chunks), config.fine_query_voxel_m)
    if len(fine) > config.fine_max_query_points:
        indices = np.linspace(
            0, len(fine) - 1, config.fine_max_query_points, dtype=np.int64
        )
        fine = np.ascontiguousarray(fine[indices])
    return voxel_downsample(fine, config.query_voxel_m), fine


def audit_registration_consensus(
    map_dir: Path,
    session: Path,
    trajectory: Path,
    baseline_replay: Path,
    frames: list[int],
    output: Path,
    *,
    query_submap_updates: int = 5,
) -> dict[str, object]:
    root = map_dir.expanduser().resolve()
    manifest = json.loads(
        (root / "localization_map_manifest.json").read_text(encoding="utf-8")
    )
    pose_mode = str(manifest.get("config", {}).get("trajectory_pose_mode", "lio"))
    aligned = load_aligned_trajectory(session, trajectory)
    poses, pose_diagnostics = prepare_trajectory_poses(aligned, pose_mode)
    files = list(aligned.keyframe_files)
    replay = json.loads(baseline_replay.expanduser().resolve().read_text(encoding="utf-8"))
    records = {int(record["frame"]): record for record in replay["records"]}

    common = ContinuousLocalizationConfig(
        query_submap_updates=query_submap_updates,
        registration_backend="kiss_icp",
    )
    kiss = ContinuousMapLocalizer(root, None, poses[0], config=common)
    vgicp_config = ContinuousLocalizationConfig(
        query_submap_updates=query_submap_updates,
        registration_backend="gtsam_vgicp",
    )
    vgicp = ContinuousMapLocalizer(root, None, poses[0], config=vgicp_config)
    kiss_submaps = {int(item["id"]): item for item in kiss.submaps}
    vgicp_submaps = {int(item["id"]): item for item in vgicp.submaps}

    results = []
    for frame in frames:
        record = records.get(frame)
        previous = records.get(frame - 1)
        if record is None or previous is None or record.get("submap_id") is None:
            results.append({"frame": frame, "failure": "missing baseline record"})
            continue
        submap_id = int(record["submap_id"])
        query_points, fine_points = _query(files, poses, frame, common)
        previous_map_from_body = _pose(previous)
        previous_map_from_odom = previous_map_from_body @ np.linalg.inv(poses[frame - 1])
        predicted = previous_map_from_odom @ poses[frame]
        anchor = np.asarray(kiss_submaps[submap_id]["anchor"], dtype=np.float64)
        initial = np.linalg.inv(anchor) @ predicted

        kiss_transform, kiss_fitness, kiss_rmse = kiss._register(
            query_points,
            fine_points,
            None,
            kiss_submaps[submap_id],
            initial,
        )
        vgicp_source = vgicp.vgicp.Frame(
            np.ascontiguousarray(
                voxel_downsample(fine_points, vgicp_config.vgicp_source_voxel_m),
                dtype=np.float64,
            ),
            vgicp_config.vgicp_covariance_neighbors,
            vgicp_config.vgicp_num_threads,
            vgicp_config.vgicp_covariance_scale,
        )
        vgicp_transform, vgicp_fitness, vgicp_rmse = vgicp._register(
            query_points,
            fine_points,
            vgicp_source,
            vgicp_submaps[submap_id],
            initial,
        )
        kiss_pose = anchor @ kiss_transform
        vgicp_pose = anchor @ vgicp_transform
        baseline_pose = _pose(record)
        disagreement_translation, disagreement_rotation = _distance(
            kiss_pose, vgicp_pose
        )
        kiss_correction_translation, kiss_correction_rotation = _distance(
            predicted, kiss_pose
        )
        vgicp_correction_translation, vgicp_correction_rotation = _distance(
            predicted, vgicp_pose
        )
        results.append({
            "frame": frame,
            "route_index": int(record["route_index"]),
            "submap_id": submap_id,
            "kiss_fitness": kiss_fitness,
            "kiss_rmse_m": kiss_rmse,
            "vgicp_fitness": vgicp_fitness,
            "vgicp_rmse_m": vgicp_rmse,
            "kiss_vgicp_translation_m": disagreement_translation,
            "kiss_vgicp_rotation_deg": disagreement_rotation,
            "kiss_prediction_correction_m": kiss_correction_translation,
            "kiss_prediction_correction_deg": kiss_correction_rotation,
            "vgicp_prediction_correction_m": vgicp_correction_translation,
            "vgicp_prediction_correction_deg": vgicp_correction_rotation,
            "kiss_vs_baseline_m": _distance(kiss_pose, baseline_pose)[0],
            "kiss_map_from_body": kiss_pose.tolist(),
            "vgicp_map_from_body": vgicp_pose.tolist(),
        })
        print(
            f"[audit] frame={frame} route={record['route_index']} "
            f"delta={disagreement_translation:.3f}m/"
            f"{disagreement_rotation:.2f}deg",
            flush=True,
        )

    report = {
        "schema_version": 1,
        "method": "read-only causal KISS/VGICP consensus audit",
        "map_dir": str(root),
        "session": str(session.expanduser().resolve()),
        "trajectory": str(trajectory.expanduser().resolve()),
        "baseline_replay": str(baseline_replay.expanduser().resolve()),
        "trajectory_alignment": aligned.report(),
        "pose_diagnostics": pose_diagnostics,
        "kiss_config": asdict(common),
        "vgicp_config": asdict(vgicp_config),
        "records": results,
    }
    destination = output.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def _frames(value: str) -> list[int]:
    result = []
    for item in value.split(","):
        if "-" in item:
            begin, end = (int(part) for part in item.split("-", 1))
            result.extend(range(begin, end + 1))
        else:
            result.append(int(item))
    return sorted(set(result))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map-dir", type=Path, required=True)
    parser.add_argument("--session", type=Path, required=True)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--baseline-replay", type=Path, required=True)
    parser.add_argument("--frames", type=_frames, required=True)
    parser.add_argument("--query-submap-updates", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audit_registration_consensus(
        args.map_dir,
        args.session,
        args.trajectory,
        args.baseline_replay,
        args.frames,
        args.output,
        query_submap_updates=args.query_submap_updates,
    )


if __name__ == "__main__":
    main()
