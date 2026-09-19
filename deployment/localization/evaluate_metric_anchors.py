"""Measure held-out metric repeatability at intermediate route anchors."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from itertools import combinations
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from deployment.mapping.mapping_geometry import load_points, rotation_distance_deg, voxel_downsample
from deployment.localization.start_alignment import (
    StartAlignmentConfig,
    StartAlignmentResult,
    StartAnchor,
    align_start_anchor,
)
from deployment.common.trajectory_io import load_aligned_trajectory, prepare_trajectory_poses


@dataclass(frozen=True)
class MetricAnchorConfig:
    route_fractions: tuple[float, ...] = tuple(
        index / 20.0 for index in range(1, 20)
    )
    causal_window_frames: tuple[int, ...] = (5, 7, 9)
    voxel_m: float = 0.08
    min_range_m: float = 0.20
    max_range_m: float = 30.0
    max_registration_points: int = 12000
    max_rmse_m: float = 0.18
    min_overlap: float = 0.55
    max_consensus_translation_m: float = 0.10
    max_consensus_rotation_deg: float = 2.0
    metric_gate_m: float = 0.10


def _aggregate_center_cloud(
    poses: np.ndarray,
    files: list[Path],
    center: int,
    radius: int,
    config: MetricAnchorConfig,
) -> np.ndarray:
    begin = max(0, center - radius)
    end = min(len(files), center + radius + 1)
    center_inverse = np.linalg.inv(poses[center])
    chunks = []
    for index in range(begin, end):
        points = load_points(files[index])
        ranges = np.linalg.norm(points, axis=1)
        points = points[
            np.isfinite(points).all(axis=1)
            & (ranges >= config.min_range_m)
            & (ranges <= config.max_range_m)
        ]
        center_from_frame = center_inverse @ poses[index]
        chunks.append(
            points @ center_from_frame[:3, :3].T + center_from_frame[:3, 3]
        )
    return voxel_downsample(np.concatenate(chunks), config.voxel_m)


def _aggregate_causal_cloud(
    poses: np.ndarray,
    files: list[Path],
    frame: int,
    frame_count: int,
    config: MetricAnchorConfig,
) -> np.ndarray:
    begin = max(0, frame - frame_count + 1)
    center_inverse = np.linalg.inv(poses[frame])
    chunks = []
    for index in range(begin, frame + 1):
        points = load_points(files[index])
        ranges = np.linalg.norm(points, axis=1)
        points = points[
            np.isfinite(points).all(axis=1)
            & (ranges >= config.min_range_m)
            & (ranges <= config.max_range_m)
        ]
        frame_from_history = center_inverse @ poses[index]
        chunks.append(
            points @ frame_from_history[:3, :3].T + frame_from_history[:3, 3]
        )
    return voxel_downsample(np.concatenate(chunks), config.voxel_m)


def _alignment_config(config: MetricAnchorConfig) -> StartAlignmentConfig:
    return StartAlignmentConfig(
        capture_frames=3,
        voxel_size_m=config.voxel_m,
        min_range_m=config.min_range_m,
        max_range_m=config.max_range_m,
        structural_min_z_m=-0.50,
        max_registration_points=config.max_registration_points,
        yaw_search_range_deg=180.0,
        yaw_search_step_deg=3.0,
        coarse_candidates=8,
        max_initial_translation_m=5.0,
        max_correspondence_m=0.60,
        trim_fraction=0.75,
        icp_iterations=40,
    )


def _register_local_anchor(
    reference_points: np.ndarray,
    query_points: np.ndarray,
    initial_anchor_from_query: np.ndarray,
    config: MetricAnchorConfig,
) -> tuple[np.ndarray, float, float]:
    try:
        from kiss_icp.mapping import VoxelHashMap
        from kiss_icp.registration import Registration
    except ImportError as error:
        raise RuntimeError("metric anchor evaluation requires kiss-icp") from error
    reference = voxel_downsample(reference_points, config.voxel_m)
    query = voxel_downsample(query_points, config.voxel_m)
    prior_aligned = (
        query @ initial_anchor_from_query[:3, :3].T
        + initial_anchor_from_query[:3, 3]
    )
    coarse_alignment = align_start_anchor(
        StartAnchor(
            reference,
            np.asarray((1.0, 0.0, 0.0, 0.0)),
            1,
            0.12,
        ),
        prior_aligned,
        StartAlignmentConfig(
            capture_frames=3,
            voxel_size_m=0.12,
            min_range_m=0.0,
            max_range_m=100.0,
            structural_min_z_m=-100.0,
            max_registration_points=8000,
            yaw_search_range_deg=30.0,
            yaw_search_step_deg=2.0,
            coarse_candidates=6,
            max_initial_translation_m=5.0,
            max_correspondence_m=0.90,
            trim_fraction=0.70,
            icp_iterations=30,
        ),
    )
    coarse_transform = (
        coarse_alignment.transform_reference_live @ initial_anchor_from_query
    )
    voxel_map = VoxelHashMap(
        voxel_size=config.voxel_m,
        max_distance=1000.0,
        max_points_per_voxel=20,
    )
    voxel_map.add_points(reference)
    registration = Registration(
        max_num_iterations=40,
        convergence_criterion=1.0e-4,
        max_num_threads=0,
    )
    transform = np.asarray(registration.align_points_to_map(
        points=query,
        voxel_map=voxel_map,
        initial_guess=coarse_transform,
        max_correspondance_distance=0.40,
        kernel=0.40 / 3.0,
    ))
    aligned = query @ transform[:3, :3].T + transform[:3, 3]
    distances, _ = cKDTree(reference).query(aligned, workers=-1)
    inliers = distances <= 0.40
    fitness = float(np.mean(inliers))
    rmse = (
        float(np.sqrt(np.mean(np.square(distances[inliers]))))
        if np.any(inliers) else float("inf")
    )
    return transform, fitness, rmse


def _best_reverse_registration(
    reference_points: np.ndarray,
    query_points: np.ndarray,
    forward_reference_from_query: np.ndarray,
    config: MetricAnchorConfig,
) -> dict[str, object]:
    """Validate a forward match in the reverse direction without one-shot bias."""

    candidates = []
    for initialization_name, initialization in (
        ("identity", np.eye(4, dtype=np.float64)),
        ("inverse_forward", np.linalg.inv(forward_reference_from_query)),
    ):
        reverse, fitness, rmse = _register_local_anchor(
            query_points,
            reference_points,
            initialization,
            config,
        )
        cycle = forward_reference_from_query @ reverse
        cycle_translation = float(np.linalg.norm(cycle[:3, 3]))
        cycle_rotation = rotation_distance_deg(cycle[:3, :3])
        candidates.append({
            "initialization": initialization_name,
            "transform": reverse,
            "fitness": fitness,
            "rmse_m": rmse,
            "cycle_translation_m": cycle_translation,
            "cycle_rotation_deg": cycle_rotation,
            "score": (
                rmse
                + 0.30 * (1.0 - fitness)
                + cycle_translation
                + 0.01 * cycle_rotation
            ),
        })
    return min(candidates, key=lambda item: float(item["score"]))


def _estimate_start_map_from_odom(
    root: Path,
    first_entry: dict[str, object],
    heldout_poses: np.ndarray,
    heldout_files: list[Path],
    config: MetricAnchorConfig,
) -> tuple[np.ndarray, dict[str, object]]:
    with np.load(root / first_entry["file"], allow_pickle=False) as payload:
        reference_points = np.asarray(
            payload.get("fine_points_anchor_m", payload["points_anchor_m"]),
            dtype=np.float64,
        )
        anchor_pose = np.asarray(payload["anchor_pose"], dtype=np.float64)
    align_config = _alignment_config(config)
    accepted_results = []
    windows = []
    for radius in (5, 7, 9):
        query = _aggregate_center_cloud(
            heldout_poses, heldout_files, 0, radius, config
        )
        result = align_start_anchor(
            StartAnchor(
                reference_points,
                np.asarray((1.0, 0.0, 0.0, 0.0)),
                radius + 1,
                config.voxel_m,
            ),
            query,
            align_config,
        )
        accepted = bool(
            result.rmse_m <= 0.20 and result.overlap_fraction >= 0.55
        )
        windows.append({
            "frame_count": radius + 1,
            "accepted": accepted,
            "rmse_m": result.rmse_m,
            "overlap": result.overlap_fraction,
            "translation_anchor_live_m": (
                result.translation_reference_live_m.tolist()
            ),
            "yaw_deg": result.yaw_deg,
        })
        if accepted:
            accepted_results.append(result)
    selected, translation_spread, rotation_spread, consensus_count = (
        _robust_consensus(
            accepted_results,
            config.max_consensus_translation_m,
            config.max_consensus_rotation_deg,
        )
    )
    map_from_start_body = anchor_pose @ selected.transform_reference_live
    return (
        map_from_start_body @ np.linalg.inv(heldout_poses[0]),
        {
            "windows": windows,
            "consensus_translation_spread_m": translation_spread,
            "consensus_rotation_spread_deg": rotation_spread,
            "consensus_window_count": consensus_count,
            "map_start_position_m": map_from_start_body[:3, 3].tolist(),
        },
    )


def _consensus(
    results: list[StartAlignmentResult],
) -> tuple[StartAlignmentResult, float, float]:
    if len(results) < 2:
        raise ValueError("metric anchor needs at least two accepted windows")
    costs = np.zeros(len(results), dtype=np.float64)
    maximum_translation = 0.0
    maximum_rotation = 0.0
    for first in range(len(results)):
        for second in range(first + 1, len(results)):
            translation = float(np.linalg.norm(
                results[first].translation_reference_live_m
                - results[second].translation_reference_live_m
            ))
            rotation = rotation_distance_deg(
                results[first].rotation_reference_live.T
                @ results[second].rotation_reference_live
            )
            maximum_translation = max(maximum_translation, translation)
            maximum_rotation = max(maximum_rotation, rotation)
            costs[first] += translation + 0.02 * rotation
            costs[second] += translation + 0.02 * rotation
    return results[int(np.argmin(costs))], maximum_translation, maximum_rotation


def _robust_consensus(
    results: list[StartAlignmentResult],
    max_translation_m: float,
    max_rotation_deg: float,
) -> tuple[StartAlignmentResult, float, float, int]:
    """Select the largest mutually consistent subset of at least two windows."""

    if len(results) < 2:
        raise ValueError("metric anchor needs at least two accepted windows")
    best_subset: tuple[int, ...] | None = None
    best_cost = float("inf")
    best_translation = float("inf")
    best_rotation = float("inf")
    for size in range(len(results), 1, -1):
        for subset in combinations(range(len(results)), size):
            maximum_translation = 0.0
            maximum_rotation = 0.0
            cost = 0.0
            valid = True
            for first, second in combinations(subset, 2):
                translation = float(np.linalg.norm(
                    results[first].translation_reference_live_m
                    - results[second].translation_reference_live_m
                ))
                rotation = rotation_distance_deg(
                    results[first].rotation_reference_live.T
                    @ results[second].rotation_reference_live
                )
                maximum_translation = max(maximum_translation, translation)
                maximum_rotation = max(maximum_rotation, rotation)
                cost += translation + 0.02 * rotation
                if (
                    translation > max_translation_m
                    or rotation > max_rotation_deg
                ):
                    valid = False
                    break
            if valid and cost < best_cost:
                best_subset = subset
                best_cost = cost
                best_translation = maximum_translation
                best_rotation = maximum_rotation
        if best_subset is not None:
            break
    if best_subset is None:
        raise ValueError("no two metric-anchor windows form a consistent pair")
    subset_results = [results[index] for index in best_subset]
    selected, _, _ = _consensus(subset_results)
    return selected, best_translation, best_rotation, len(best_subset)


def _pose_from_record(record: dict[str, object]) -> np.ndarray:
    pose = np.eye(4, dtype=np.float64)
    pose[:3, 3] = np.asarray(record["map_position_m"], dtype=np.float64)
    pose[:3, :3] = np.asarray(record["map_rotation_matrix"], dtype=np.float64)
    return pose


def evaluate_metric_anchors(
    map_dir: Path,
    heldout_session: Path,
    heldout_trajectory: Path,
    replay_report: Path,
    *,
    output_report: Path | None = None,
    config: MetricAnchorConfig = MetricAnchorConfig(),
) -> dict[str, object]:
    root = map_dir.expanduser().resolve()
    manifest = json.loads(
        (root / "localization_map_manifest.json").read_text(encoding="utf-8")
    )
    canonical_entries = {
        int(entry["route_index"]): entry
        for entry in manifest["submaps"]
        if int(entry["reference_session"]) == 0
    }
    maximum_route_index = max(canonical_entries)
    aligned = load_aligned_trajectory(heldout_session, heldout_trajectory)
    heldout_poses, pose_diagnostics = prepare_trajectory_poses(aligned, "lio")
    heldout_files = list(aligned.keyframe_files)
    replay = json.loads(replay_report.expanduser().resolve().read_text(
        encoding="utf-8"
    ))
    records = [
        record for record in replay["records"]
        if record.get("accepted") and "map_position_m" in record
    ]
    if not records:
        raise ValueError("replay report has no accepted metric poses")

    independent_map_from_odom, start_alignment = _estimate_start_map_from_odom(
        root,
        canonical_entries[0],
        heldout_poses,
        heldout_files,
        config,
    )
    accepted_by_submap: dict[int, list[dict[str, object]]] = {}
    for record in records:
        submap_id = int(record.get("submap_id", -1))
        if submap_id >= 0:
            accepted_by_submap.setdefault(submap_id, []).append(record)
    available_route_indices = [
        route_index for route_index, entry in canonical_entries.items()
        if int(entry["submap_id"]) in accepted_by_submap
    ]
    if not available_route_indices:
        raise ValueError("replay did not accept any canonical route submap")
    anchor_reports = []
    for fraction in config.route_fractions:
        requested_route_index = int(round(fraction * maximum_route_index))
        route_index = min(
            available_route_indices,
            key=lambda index: abs(index - requested_route_index),
        )
        entry = canonical_entries[route_index]
        exact_submap = accepted_by_submap[int(entry["submap_id"])]
        selected_record = min(
            exact_submap,
            key=lambda record: (
                float(record.get("rmse_m") or float("inf")),
                -float(record.get("fitness") or 0.0),
            ),
        )
        frame = int(selected_record["frame"])
        if frame >= len(heldout_files):
            raise ValueError(f"replay frame {frame} exceeds aligned trajectory")

        with np.load(root / entry["file"], allow_pickle=False) as payload:
            reference_points = np.asarray(
                payload.get("fine_points_anchor_m", payload["points_anchor_m"]),
                dtype=np.float64,
            )
            anchor_pose = np.asarray(payload["anchor_pose"], dtype=np.float64)

        windows = []
        accepted_transforms: list[np.ndarray] = []
        predicted_map_from_body = independent_map_from_odom @ heldout_poses[frame]
        initial_anchor_from_body = np.linalg.inv(anchor_pose) @ predicted_map_from_body
        for frame_count in config.causal_window_frames:
            query_points = _aggregate_causal_cloud(
                heldout_poses,
                heldout_files,
                frame,
                frame_count,
                config,
            )
            transform, fitness, rmse = _register_local_anchor(
                reference_points,
                query_points,
                initial_anchor_from_body,
                config,
            )
            accepted = bool(
                rmse <= config.max_rmse_m and fitness >= config.min_overlap
            )
            windows.append({
                "frame_count": frame_count,
                "accepted": accepted,
                "rmse_m": rmse,
                "fitness": fitness,
                "translation_anchor_live_m": transform[:3, 3].tolist(),
                "rotation_anchor_live": transform[:3, :3].tolist(),
            })
            if accepted:
                accepted_transforms.append(transform)

        anchor_report: dict[str, object] = {
            "requested_route_fraction": fraction,
            "requested_route_index": requested_route_index,
            "route_index": route_index,
            "heldout_frame": frame,
            "heldout_sensor_stamp_s": float(selected_record["sensor_stamp_s"]),
            "selection_source": "lowest_rmse_accepted_nearest_submap",
            "window_results": windows,
            "qualified": False,
        }
        try:
            alignment_results = [StartAlignmentResult(
                rotation_reference_live=transform[:3, :3],
                translation_reference_live_m=transform[:3, 3],
                yaw_deg=0.0,
                rmse_m=0.0,
                overlap_fraction=1.0,
                correspondence_count=0,
                candidate_margin_m=0.0,
            ) for transform in accepted_transforms]
            selected, translation_spread, rotation_spread, consensus_count = (
                _robust_consensus(
                    alignment_results,
                    config.max_consensus_translation_m,
                    config.max_consensus_rotation_deg,
                )
            )
            consensus_passed = True
            expected_pose = anchor_pose @ selected.transform_reference_live
            actual_pose = _pose_from_record(selected_record)
            residual = np.linalg.inv(expected_pose) @ actual_pose
            translation_error = float(np.linalg.norm(residual[:3, 3]))
            rotation_error = rotation_distance_deg(residual[:3, :3])
            anchor_report.update({
                "consensus_passed": consensus_passed,
                "consensus_translation_spread_m": translation_spread,
                "consensus_rotation_spread_deg": rotation_spread,
                "consensus_window_count": consensus_count,
                "expected_position_m": expected_pose[:3, 3].tolist(),
                "actual_position_m": actual_pose[:3, 3].tolist(),
                "translation_error_m": translation_error,
                "rotation_error_deg": rotation_error,
                "qualified": bool(
                    consensus_passed
                    and translation_error <= config.metric_gate_m
                ),
            })
            # Continue the independent sparse-anchor chain from the measured
            # pose, not from the online localizer output. This keeps the next
            # 10% route interval inside the local registration basin while
            # preserving independence of the metric reference.
            independent_map_from_odom = (
                expected_pose @ np.linalg.inv(heldout_poses[frame])
            )
        except ValueError as error:
            anchor_report["failure"] = str(error)
        anchor_reports.append(anchor_report)

    errors = np.asarray([
        float(item["translation_error_m"])
        for item in anchor_reports
        if "translation_error_m" in item
    ])
    repeatability_qualified = bool(
        len(errors) == len(config.route_fractions)
        and all(bool(item["qualified"]) for item in anchor_reports)
    )
    report = {
        "schema_version": 1,
        "evaluation_contract": (
            "Each held-out anchor is independently registered to the canonical "
            "fine submap with multi-window consensus; online map poses are not "
            "used to initialize the independent registration."
        ),
        "map_dir": str(root),
        "heldout_session": str(heldout_session.expanduser().resolve()),
        "heldout_trajectory": str(heldout_trajectory.expanduser().resolve()),
        "replay_report": str(replay_report.expanduser().resolve()),
        "config": asdict(config),
        "trajectory_alignment": aligned.report(),
        "trajectory_pose_diagnostics": pose_diagnostics,
        "independent_start_alignment": start_alignment,
        "metric_gate_m": config.metric_gate_m,
        "anchor_count": len(anchor_reports),
        "qualified_anchor_count": sum(
            bool(item["qualified"]) for item in anchor_reports
        ),
        "maximum_translation_error_m": (
            float(np.max(errors)) if len(errors) else None
        ),
        "mean_translation_error_m": (
            float(np.mean(errors)) if len(errors) else None
        ),
        "p95_translation_error_m": (
            float(np.percentile(errors, 95.0)) if len(errors) else None
        ),
        "cross_lap_repeatability_qualified": repeatability_qualified,
        "full_route_metric_accuracy_qualified": False,
        "qualification_note": (
            "This measures cross-lap repeatability relative to the canonical "
            "route map, not surveyed absolute error. It cannot qualify the full-route "
            "0.10 m gate even when every sampled registration agrees."
        ),
        "anchors": anchor_reports,
    }
    if output_report is not None:
        output = output_report.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map-dir", type=Path, required=True)
    parser.add_argument("--heldout-session", type=Path, required=True)
    parser.add_argument("--heldout-trajectory", type=Path, required=True)
    parser.add_argument("--replay-report", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    args = parser.parse_args()
    report = evaluate_metric_anchors(
        args.map_dir,
        args.heldout_session,
        args.heldout_trajectory,
        args.replay_report,
        output_report=args.output_report,
    )
    print(json.dumps({
        key: report[key]
        for key in (
            "anchor_count",
            "qualified_anchor_count",
            "maximum_translation_error_m",
            "mean_translation_error_m",
            "p95_translation_error_m",
            "cross_lap_repeatability_qualified",
            "full_route_metric_accuracy_qualified",
        )
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
