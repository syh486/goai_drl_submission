"""Evaluate causal online map poses against held-out multi-window references."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from itertools import combinations, product
import json
from pathlib import Path

import numpy as np

from deployment.localization.continuous_map_localization import (
    ContinuousLocalizationConfig,
    _scan_context_alignment,
)
from deployment.localization.evaluate_metric_anchors import (
    MetricAnchorConfig,
    _aggregate_causal_cloud,
    _best_reverse_registration,
    _register_local_anchor,
)
from deployment.mapping.mapping_geometry import (
    rotation_distance_deg,
    scan_context_descriptor,
)
from deployment.mapping.refine_route_multilap import (
    MultiLapRefinementConfig,
    _path_progress,
    _registration_cost,
    _selected_route_indices,
)
from deployment.common.trajectory_io import load_aligned_trajectory, prepare_trajectory_poses


CAUSAL_WINDOWS = (5, 7, 9)
CONSENSUS_TRANSLATION_M = 0.10
CONSENSUS_ROTATION_DEG = 2.0


def _pose_from_record(record: dict[str, object]) -> np.ndarray:
    pose = np.eye(4, dtype=np.float64)
    pose[:3, 3] = np.asarray(record["map_position_m"], dtype=np.float64)
    pose[:3, :3] = np.asarray(record["map_rotation_matrix"], dtype=np.float64)
    return pose


def _route_index_from_record(record: dict[str, object]) -> int | None:
    """Read route diagnostics from legacy or ordered-anchor replay reports."""

    for key in ("accepted_route_index", "target_route_index", "route_index"):
        value = record.get(key)
        if value is not None:
            return int(value)
    return None


def _transform_distance(first: np.ndarray, second: np.ndarray) -> tuple[float, float]:
    residual = np.linalg.inv(first) @ second
    return (
        float(np.linalg.norm(residual[:3, 3])),
        rotation_distance_deg(residual[:3, :3]),
    )


def _consensus_subset(
    results: list[dict[str, object]],
) -> tuple[list[dict[str, object]], dict[str, object]]:
    grouped: dict[int, list[dict[str, object]]] = {}
    for result in results:
        grouped.setdefault(int(result["window_frames"]), []).append(result)
    best: tuple[dict[str, object], ...] | None = None
    best_cost = float("inf")
    best_spread = (float("inf"), float("inf"))
    windows = sorted(grouped)
    for size in range(len(windows), 1, -1):
        for selected_windows in combinations(windows, size):
            for selected in product(*(grouped[window] for window in selected_windows)):
                maximum_translation = 0.0
                maximum_rotation = 0.0
                pair_cost = 0.0
                valid = True
                for first, second in combinations(selected, 2):
                    translation, rotation = _transform_distance(
                        np.asarray(first["transform"], dtype=np.float64),
                        np.asarray(second["transform"], dtype=np.float64),
                    )
                    maximum_translation = max(maximum_translation, translation)
                    maximum_rotation = max(maximum_rotation, rotation)
                    pair_cost += translation + 0.02 * rotation
                    if (
                        translation > CONSENSUS_TRANSLATION_M
                        or rotation > CONSENSUS_ROTATION_DEG
                    ):
                        valid = False
                        break
                registration_cost = sum(float(item["cost"]) for item in selected)
                total_cost = pair_cost + registration_cost
                if valid and total_cost < best_cost:
                    best = selected
                    best_cost = total_cost
                    best_spread = (maximum_translation, maximum_rotation)
        if best is not None:
            break
    if best is None:
        raise ValueError("fewer than two causal windows form an SE3 consensus")
    subset = list(best)
    medoid_costs = []
    for first, item in enumerate(subset):
        cost = 0.0
        for second, other in enumerate(subset):
            if first == second:
                continue
            translation, rotation = _transform_distance(
                np.asarray(item["transform"], dtype=np.float64),
                np.asarray(other["transform"], dtype=np.float64),
            )
            cost += translation + 0.02 * rotation
        medoid_costs.append(cost)
    medoid = subset[int(np.argmin(medoid_costs))]
    diagnostics = {
        "window_count": len(subset),
        "translation_spread_m": best_spread[0],
        "rotation_spread_deg": best_spread[1],
        "medoid_window_frames": int(medoid["window_frames"]),
        "medoid_initialization": str(medoid["initialization"]),
    }
    return subset, diagnostics


def _extract_causal_references(
    root: Path,
    entries: dict[int, dict[str, object]],
    heldout_session: Path,
    heldout_trajectory: Path,
    config: MultiLapRefinementConfig,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    aligned = load_aligned_trajectory(heldout_session, heldout_trajectory)
    poses, pose_diagnostics = prepare_trajectory_poses(aligned, "lio")
    files = list(aligned.keyframe_files)
    progress = _path_progress(poses)
    registration_config = MetricAnchorConfig(
        voxel_m=0.08,
        min_range_m=0.20,
        max_range_m=30.0,
        max_registration_points=12000,
        max_rmse_m=config.max_rmse_m,
        min_overlap=config.min_fitness,
    )
    cloud_cache: dict[tuple[int, int], np.ndarray] = {}

    def cloud(frame: int, window: int) -> np.ndarray:
        key = (frame, window)
        if key not in cloud_cache:
            cloud_cache[key] = _aggregate_causal_cloud(
                poses, files, frame, window, registration_config
            )
        return cloud_cache[key]

    references = []
    failures = []
    previous_frame = -1
    previous_map_from_support: np.ndarray | None = None
    route_indices = _selected_route_indices(entries, config.anchor_stride_submaps)
    for number, route_index in enumerate(route_indices, start=1):
        entry = entries[route_index]
        target_fraction = float(entry["anchor_progress_fraction"])
        target_progress_m = float(entry["anchor_progress_m"])
        insertion = int(np.searchsorted(progress, target_progress_m, side="left"))
        if insertion <= 0:
            center = 0
        elif insertion >= len(progress):
            center = len(progress) - 1
        else:
            center = min(
                (insertion - 1, insertion),
                key=lambda index: abs(float(progress[index]) - target_progress_m),
            )
        begin = max(max(CAUSAL_WINDOWS) - 1, center - config.candidate_radius_frames)
        end = min(len(files) - 1, center + config.candidate_radius_frames)
        candidates = list(range(begin, end + 1, config.candidate_step_frames))
        if center not in candidates and begin <= center <= end:
            candidates.append(center)
        candidates = sorted(frame for frame in set(candidates) if frame > previous_frame)
        with np.load(root / str(entry["file"]), allow_pickle=False) as payload:
            reference_points = np.asarray(
                payload.get("fine_points_anchor_m", payload["points_anchor_m"]),
                dtype=np.float64,
            )
            anchor_pose = np.asarray(payload["anchor_pose"], dtype=np.float64)
        reference_descriptor = scan_context_descriptor(reference_points, 20, 60)
        descriptor_rank = []
        for frame in candidates:
            descriptor, _ = _scan_context_alignment(
                reference_descriptor,
                scan_context_descriptor(cloud(frame, 5), 20, 60),
            )
            descriptor_rank.append((descriptor, frame))
        descriptor_rank.sort(key=lambda item: item[0])
        candidate_frames = {
            frame for _, frame in descriptor_rank[:2]
        }
        if center in candidates:
            candidate_frames.add(center)

        verified_candidates = []
        candidate_audit = []
        for descriptor_distance, frame in descriptor_rank:
            if frame not in candidate_frames:
                continue
            initializations = [("identity", np.eye(4, dtype=np.float64))]
            if previous_map_from_support is not None:
                motion = np.linalg.inv(poses[previous_frame]) @ poses[frame]
                initializations.append((
                    "ordered_local_odometry",
                    np.linalg.inv(anchor_pose) @ previous_map_from_support @ motion,
                ))
            window_results = []
            window_clouds: dict[int, np.ndarray] = {}
            for window in CAUSAL_WINDOWS:
                query = cloud(frame, window)
                window_clouds[window] = query
                hypotheses = []
                for initialization_name, initialization in initializations:
                    transform, fitness, rmse = _register_local_anchor(
                        reference_points,
                        query,
                        initialization,
                        registration_config,
                    )
                    hypotheses.append({
                        "window_frames": window,
                        "initialization": initialization_name,
                        "transform": transform.tolist(),
                        "fitness": fitness,
                        "rmse_m": rmse,
                        "cost": _registration_cost(fitness, rmse),
                    })
                accepted = [
                    result for result in hypotheses
                    if float(result["fitness"]) >= config.min_fitness
                    and float(result["rmse_m"]) <= config.max_rmse_m
                ]
                window_results.extend(accepted)
            audit = {
                "frame": frame,
                "descriptor_distance": descriptor_distance,
                "window_results": window_results,
            }
            try:
                subset, consensus = _consensus_subset(window_results)
                reverse_records = []
                for result in subset:
                    window = int(result["window_frames"])
                    forward = np.asarray(result["transform"], dtype=np.float64)
                    reverse = _best_reverse_registration(
                        reference_points,
                        window_clouds[window],
                        forward,
                        registration_config,
                    )
                    reverse_records.append({
                        "window_frames": window,
                        "initialization": reverse["initialization"],
                        "fitness": reverse["fitness"],
                        "rmse_m": reverse["rmse_m"],
                        "cycle_translation_m": reverse["cycle_translation_m"],
                        "cycle_rotation_deg": reverse["cycle_rotation_deg"],
                    })
                reverse_valid = [
                    record for record in reverse_records
                    if float(record["fitness"]) >= config.min_fitness
                    and float(record["rmse_m"]) <= config.max_rmse_m
                    and float(record["cycle_translation_m"])
                    <= config.max_cycle_translation_m
                    and float(record["cycle_rotation_deg"])
                    <= config.max_cycle_rotation_deg
                ]
                if len(reverse_valid) < 2:
                    raise ValueError("fewer than two windows pass reverse registration")
                medoid_window = int(consensus["medoid_window_frames"])
                medoid = next(
                    result for result in subset
                    if int(result["window_frames"]) == medoid_window
                )
                transform = np.asarray(medoid["transform"], dtype=np.float64)
                score = (
                    sum(float(result["cost"]) for result in subset)
                    + float(consensus["translation_spread_m"])
                    + 0.01 * float(consensus["rotation_spread_deg"])
                    + sum(float(record["cycle_translation_m"])
                          for record in reverse_valid)
                )
                candidate = {
                    "route_index": route_index,
                    "route_fraction": target_fraction,
                    "route_progress_m": target_progress_m,
                    "support_frame": frame,
                    "candidate_center_frame": center,
                    "candidate_frame_offset": frame - center,
                    "canonical_from_support": transform.tolist(),
                    "consensus": consensus,
                    "window_results": subset,
                    "reverse_results": reverse_records,
                    "score": score,
                }
                verified_candidates.append((score, candidate, anchor_pose))
                audit["consensus"] = consensus
                audit["reverse_results"] = reverse_records
                audit["score"] = score
            except ValueError as error:
                audit["failure"] = str(error)
            candidate_audit.append(audit)

        if not verified_candidates:
            failures.append({
                "route_index": route_index,
                "route_fraction": target_fraction,
                "route_progress_m": target_progress_m,
                "candidate_center_frame": center,
                "candidate_audit": candidate_audit,
                "failure": "no causal multi-window reference passed",
            })
            print(
                f"[holdout-causal] {number}/{len(route_indices)} "
                f"route={route_index} rejected",
                flush=True,
            )
            continue
        _, selected, anchor_pose = min(
            verified_candidates, key=lambda item: item[0]
        )
        selected["candidate_audit"] = candidate_audit
        references.append(selected)
        previous_frame = int(selected["support_frame"])
        previous_map_from_support = (
            anchor_pose
            @ np.asarray(selected["canonical_from_support"], dtype=np.float64)
        )
        print(
            f"[holdout-causal] {number}/{len(route_indices)} "
            f"route={route_index} frame={previous_frame} "
            f"windows={selected['consensus']['window_count']} "
            f"spread={selected['consensus']['translation_spread_m']:.3f}m",
            flush=True,
        )
    extraction = {
        "schema_version": 2,
        "method": (
            "causal odometry-distance scheduling, multi-window, "
            "multi-initialization, bidirectional consensus"
        ),
        "heldout_session": str(heldout_session.expanduser().resolve()),
        "heldout_trajectory": str(heldout_trajectory.expanduser().resolve()),
        "trajectory_alignment": aligned.report(),
        "pose_diagnostics": pose_diagnostics,
        "config": asdict(config),
        "causal_windows": list(CAUSAL_WINDOWS),
        "attempted_anchor_count": len(route_indices),
        "accepted_anchor_count": len(references),
        "failures": failures,
    }
    return references, extraction


def evaluate_multilap_holdout(
    map_dir: Path,
    heldout_session: Path,
    heldout_trajectory: Path,
    replay_report: Path,
    output_dir: Path,
    config: MultiLapRefinementConfig,
    *,
    metric_gate_m: float = 0.10,
) -> dict[str, object]:
    root = map_dir.expanduser().resolve()
    destination = output_dir.expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    references_path = destination / "heldout_causal_references.json"
    extraction_path = destination / "heldout_reference_report.json"
    if references_path.is_file() and extraction_path.is_file():
        references = json.loads(references_path.read_text(encoding="utf-8"))
        extraction = json.loads(extraction_path.read_text(encoding="utf-8"))
    else:
        if any(destination.iterdir()):
            raise ValueError(
                f"incomplete held-out evaluation output exists: {destination}"
            )
        manifest = json.loads(
            (root / "localization_map_manifest.json").read_text(encoding="utf-8")
        )
        entries = {
            int(entry["route_index"]): entry
            for entry in manifest["submaps"]
            if int(entry["reference_session"]) == 0
        }
        references, extraction = _extract_causal_references(
            root, entries, heldout_session, heldout_trajectory, config
        )
        references_path.write_text(
            json.dumps(references, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        extraction_path.write_text(
            json.dumps(extraction, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    manifest = json.loads(
        (root / "localization_map_manifest.json").read_text(encoding="utf-8")
    )
    entries = {
        int(entry["route_index"]): entry
        for entry in manifest["submaps"]
        if int(entry["reference_session"]) == 0
    }
    replay = json.loads(
        replay_report.expanduser().resolve().read_text(encoding="utf-8")
    )
    records = {
        int(record["frame"]): record for record in replay["records"]
        if "map_position_m" in record
    }
    startup_distance = ContinuousLocalizationConfig().minimum_motion_before_matching_m
    anchors = []
    excluded_startup = 0
    for reference in references:
        route_index = int(reference["route_index"])
        frame = int(reference["support_frame"])
        entry = entries[route_index]
        with np.load(root / str(entry["file"]), allow_pickle=False) as payload:
            anchor_pose = np.asarray(payload["anchor_pose"], dtype=np.float64)
        expected_pose = anchor_pose @ np.asarray(
            reference["canonical_from_support"], dtype=np.float64
        )
        record = records.get(frame)
        anchor_report: dict[str, object] = {
            "route_index": route_index,
            "route_fraction": float(reference["route_fraction"]),
            "frame": frame,
            "reference_consensus": reference["consensus"],
            "qualified": False,
        }
        if record is None:
            anchor_report["failure"] = "online replay has no matching frame"
            anchors.append(anchor_report)
            continue
        if float(record["traveled_distance_m"]) < startup_distance:
            anchor_report["excluded"] = "localization startup distance"
            excluded_startup += 1
            anchors.append(anchor_report)
            continue
        actual_pose = _pose_from_record(record)
        residual = np.linalg.inv(expected_pose) @ actual_pose
        translation_error = float(np.linalg.norm(residual[:3, 3]))
        rotation_error = rotation_distance_deg(residual[:3, :3])
        online_route_index = _route_index_from_record(record)
        anchor_report.update({
            "online_observation_accepted": bool(record["accepted"]),
            "online_route_index": online_route_index,
            "expected_position_m": expected_pose[:3, 3].tolist(),
            "actual_position_m": actual_pose[:3, 3].tolist(),
            "translation_error_m": translation_error,
            "rotation_error_deg": rotation_error,
            "qualified": bool(translation_error <= metric_gate_m),
        })
        anchors.append(anchor_report)

    evaluated = [item for item in anchors if "translation_error_m" in item]
    errors = np.asarray([
        float(item["translation_error_m"]) for item in evaluated
    ])
    qualified_count = sum(bool(item["qualified"]) for item in evaluated)
    attempted_after_startup = len(anchors) - excluded_startup
    repeatability_qualified = bool(
        attempted_after_startup >= config.minimum_constraints
        and len(evaluated) == attempted_after_startup
        and qualified_count == attempted_after_startup
    )
    report = {
        "schema_version": 2,
        "evaluation_contract": (
            "The held-out lap is absent from map refinement. References use only "
            "current/past scans and distance already accumulated from the start, "
            "ordered local-odometry propagation, identity initialization, 5/7/9-"
            "frame SE3 consensus, and reverse registration. They never use final "
            "lap length or the online map pose being evaluated."
        ),
        "map_dir": str(root),
        "heldout_session": str(heldout_session.expanduser().resolve()),
        "heldout_trajectory": str(heldout_trajectory.expanduser().resolve()),
        "replay_report": str(replay_report.expanduser().resolve()),
        "config": asdict(config),
        "metric_gate_m": metric_gate_m,
        "startup_distance_m": startup_distance,
        "reference_anchor_count": len(references),
        "excluded_startup_anchor_count": excluded_startup,
        "evaluated_anchor_count": len(evaluated),
        "qualified_anchor_count": qualified_count,
        "maximum_translation_error_m": float(np.max(errors)) if len(errors) else None,
        "mean_translation_error_m": float(np.mean(errors)) if len(errors) else None,
        "p95_translation_error_m": (
            float(np.percentile(errors, 95.0)) if len(errors) else None
        ),
        "heldout_map_repeatability_qualified": repeatability_qualified,
        "full_route_metric_accuracy_qualified": False,
        "qualification_note": (
            "This qualifies causal held-out localization repeatability against "
            "the route map, not surveyed absolute map accuracy."
        ),
        "reference_extraction": extraction,
        "anchors": anchors,
    }
    (destination / "heldout_metric_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map-dir", type=Path, required=True)
    parser.add_argument("--heldout-session", type=Path, required=True)
    parser.add_argument("--heldout-trajectory", type=Path, required=True)
    parser.add_argument("--replay-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--anchor-stride-submaps", type=int, default=4)
    parser.add_argument("--candidate-radius-frames", type=int, default=8)
    parser.add_argument("--candidate-step-frames", type=int, default=2)
    parser.add_argument("--minimum-constraints", type=int, default=20)
    args = parser.parse_args()
    report = evaluate_multilap_holdout(
        args.map_dir,
        args.heldout_session,
        args.heldout_trajectory,
        args.replay_report,
        args.output_dir,
        MultiLapRefinementConfig(
            anchor_stride_submaps=args.anchor_stride_submaps,
            candidate_radius_frames=args.candidate_radius_frames,
            candidate_step_frames=args.candidate_step_frames,
            minimum_constraints=args.minimum_constraints,
        ),
    )
    print(json.dumps({
        key: report[key]
        for key in (
            "reference_anchor_count",
            "excluded_startup_anchor_count",
            "evaluated_anchor_count",
            "qualified_anchor_count",
            "maximum_translation_error_m",
            "mean_translation_error_m",
            "p95_translation_error_m",
            "heldout_map_repeatability_qualified",
            "full_route_metric_accuracy_qualified",
        )
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
