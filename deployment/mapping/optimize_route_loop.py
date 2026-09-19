"""Estimate a verified terminal loop and smoothly correct one GLIM route."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from deployment.mapping.mapping_geometry import (
    load_points,
    rotation_distance_deg,
    transform_points,
    voxel_downsample,
)
from deployment.localization.start_alignment import (
    StartAlignmentConfig,
    StartAlignmentResult,
    StartAnchor,
    align_start_anchor,
)
from deployment.common.trajectory_io import load_aligned_trajectory, prepare_trajectory_poses


@dataclass(frozen=True)
class RouteLoopConfig:
    # The first/last 8-12 frames cover the common start area. Larger windows
    # include different inbound/outbound route geometry and bias the terminal
    # registration even though their point residual remains superficially low.
    window_frames: tuple[int, ...] = (8, 10, 12)
    aggregate_voxel_m: float = 0.10
    min_range_m: float = 0.20
    max_range_m: float = 40.0
    structural_min_z_m: float = -0.50
    max_registration_points: int = 12000
    yaw_search_step_deg: float = 3.0
    max_initial_translation_m: float = 5.0
    max_correspondence_m: float = 0.60
    trim_fraction: float = 0.75
    icp_iterations: int = 40
    max_rmse_m: float = 0.20
    min_overlap: float = 0.60
    max_consensus_translation_m: float = 0.10
    max_consensus_rotation_deg: float = 1.0
    same_floor_max_abs_z_m: float = 0.10


def _aggregate_endpoint_cloud(
    poses: np.ndarray,
    files: list[Path],
    *,
    terminal: bool,
    frame_count: int,
    voxel_m: float,
    min_range_m: float,
    max_range_m: float,
) -> np.ndarray:
    if frame_count < 3 or frame_count > len(files):
        raise ValueError("endpoint aggregate frame count is invalid")
    indices = (
        range(len(files) - frame_count, len(files))
        if terminal else range(frame_count)
    )
    center = len(files) - 1 if terminal else 0
    center_inverse = np.linalg.inv(poses[center])
    chunks = []
    for index in indices:
        points = load_points(files[index])
        ranges = np.linalg.norm(points, axis=1)
        points = points[
            np.isfinite(points).all(axis=1)
            & (ranges >= min_range_m)
            & (ranges <= max_range_m)
        ]
        chunks.append(transform_points(points, center_inverse @ poses[index]))
    return voxel_downsample(np.concatenate(chunks), voxel_m)


def _alignment_config(config: RouteLoopConfig) -> StartAlignmentConfig:
    return StartAlignmentConfig(
        capture_frames=3,
        voxel_size_m=config.aggregate_voxel_m,
        min_range_m=config.min_range_m,
        max_range_m=config.max_range_m,
        structural_min_z_m=config.structural_min_z_m,
        max_registration_points=config.max_registration_points,
        yaw_search_range_deg=180.0,
        yaw_search_step_deg=config.yaw_search_step_deg,
        coarse_candidates=8,
        max_initial_translation_m=config.max_initial_translation_m,
        max_correspondence_m=config.max_correspondence_m,
        trim_fraction=config.trim_fraction,
        icp_iterations=config.icp_iterations,
    )


def _alignment_record(
    frame_count: int,
    result: StartAlignmentResult,
    accepted: bool,
) -> dict[str, object]:
    return {
        "window_frames": frame_count,
        "accepted": accepted,
        "rmse_m": result.rmse_m,
        "overlap": result.overlap_fraction,
        "correspondence_count": result.correspondence_count,
        "candidate_margin_m": result.candidate_margin_m,
        "translation_start_terminal_m": result.translation_reference_live_m.tolist(),
        "rotation_start_terminal": result.rotation_reference_live.tolist(),
        "yaw_deg": result.yaw_deg,
    }


def _select_consensus_result(
    results: list[StartAlignmentResult],
) -> tuple[StartAlignmentResult, float, float]:
    if len(results) < 2:
        raise RuntimeError("terminal loop needs at least two accepted window sizes")
    translations = np.asarray(
        [item.translation_reference_live_m for item in results]
    )
    maximum_translation = 0.0
    maximum_rotation = 0.0
    costs = np.zeros(len(results), dtype=np.float64)
    for first in range(len(results)):
        for second in range(first + 1, len(results)):
            translation = float(np.linalg.norm(
                translations[first] - translations[second]
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


def estimate_endpoint_alignment(
    reference_poses: np.ndarray,
    reference_files: list[Path],
    live_poses: np.ndarray,
    live_files: list[Path],
    *,
    reference_terminal: bool,
    live_terminal: bool,
    config: RouteLoopConfig = RouteLoopConfig(),
) -> tuple[StartAlignmentResult, list[dict[str, object]], float, float]:
    """Estimate one endpoint frame transform with multi-window consensus."""

    align_config = _alignment_config(config)
    records = []
    accepted_results = []
    for frame_count in config.window_frames:
        reference = _aggregate_endpoint_cloud(
            reference_poses,
            reference_files,
            terminal=reference_terminal,
            frame_count=frame_count,
            voxel_m=config.aggregate_voxel_m,
            min_range_m=config.min_range_m,
            max_range_m=config.max_range_m,
        )
        live = _aggregate_endpoint_cloud(
            live_poses,
            live_files,
            terminal=live_terminal,
            frame_count=frame_count,
            voxel_m=config.aggregate_voxel_m,
            min_range_m=config.min_range_m,
            max_range_m=config.max_range_m,
        )
        result = align_start_anchor(
            StartAnchor(
                reference,
                np.asarray((1.0, 0.0, 0.0, 0.0)),
                frame_count,
                config.aggregate_voxel_m,
            ),
            live,
            align_config,
        )
        accepted = bool(
            result.rmse_m <= config.max_rmse_m
            and result.overlap_fraction >= config.min_overlap
            and abs(result.translation_reference_live_m[2])
            <= config.same_floor_max_abs_z_m
        )
        records.append(_alignment_record(frame_count, result, accepted))
        if accepted:
            accepted_results.append(result)
    selected, translation_spread, rotation_spread = _select_consensus_result(
        accepted_results
    )
    if (
        translation_spread > config.max_consensus_translation_m
        or rotation_spread > config.max_consensus_rotation_deg
    ):
        raise RuntimeError(
            "endpoint estimates disagree: "
            f"translation={translation_spread:.3f}m, "
            f"rotation={rotation_spread:.3f}deg"
        )
    return selected, records, translation_spread, rotation_spread


def apply_distributed_loop_correction(
    poses: np.ndarray,
    terminal_target: np.ndarray,
) -> tuple[np.ndarray, dict[str, float]]:
    """Distribute the terminal SE(3) correction with a smooth endpoint warp."""

    original = np.asarray(poses, dtype=np.float64)
    correction_end = terminal_target @ np.linalg.inv(original[-1])
    progress = np.concatenate((
        np.zeros(1),
        np.cumsum(np.linalg.norm(np.diff(original[:, :3, 3], axis=0), axis=1)),
    ))
    fraction = progress / progress[-1]
    blend = fraction * fraction * (3.0 - 2.0 * fraction)
    rotations = Slerp(
        (0.0, 1.0),
        Rotation.from_matrix(np.stack((np.eye(3), correction_end[:3, :3]))),
    )(blend).as_matrix()
    translations = blend[:, None] * correction_end[:3, 3]
    corrections = np.tile(np.eye(4), (len(original), 1, 1))
    corrections[:, :3, :3] = rotations
    corrections[:, :3, 3] = translations
    corrected = np.einsum("nij,njk->nik", corrections, original)

    original_steps = np.linalg.norm(np.diff(original[:, :3, 3], axis=0), axis=1)
    corrected_steps = np.linalg.norm(np.diff(corrected[:, :3, 3], axis=0), axis=1)
    step_change = np.abs(corrected_steps - original_steps)
    diagnostics = {
        "maximum_step_length_change_m": float(np.max(step_change)),
        "p95_step_length_change_m": float(np.percentile(step_change, 95.0)),
        "path_length_original_m": float(np.sum(original_steps)),
        "path_length_corrected_m": float(np.sum(corrected_steps)),
    }
    return corrected, diagnostics


def optimize_route_loop(
    session_dir: Path,
    trajectory_path: Path,
    output_dir: Path,
    config: RouteLoopConfig = RouteLoopConfig(),
) -> dict[str, object]:
    aligned = load_aligned_trajectory(session_dir, trajectory_path)
    poses, pose_diagnostics = prepare_trajectory_poses(aligned, "lio")
    files = list(aligned.keyframe_files)
    selected, records, translation_spread, rotation_spread = estimate_endpoint_alignment(
        poses,
        files,
        poses,
        files,
        reference_terminal=False,
        live_terminal=True,
        config=config,
    )
    terminal_target = selected.transform_reference_live
    corrected, correction_diagnostics = apply_distributed_loop_correction(
        poses, terminal_target
    )
    endpoint_residual = np.linalg.inv(terminal_target) @ corrected[-1]
    output = output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    trajectory_output = output / "optimized_route_trajectory.npz"
    np.savez_compressed(
        trajectory_output,
        schema_version=np.asarray(1, dtype=np.int64),
        poses=corrected,
        timestamps_s=aligned.timestamps_s,
        keyframe_indices=aligned.keyframe_indices,
    )
    raw_endpoint = np.linalg.inv(poses[0]) @ poses[-1]
    optimized_endpoint = np.linalg.inv(corrected[0]) @ corrected[-1]
    report = {
        "schema_version": 1,
        "method": "multi-window terminal submap registration plus smooth SE3 correction",
        "full_route_metric_accuracy_qualified": False,
        "qualification_note": (
            "The terminal loop is measured and corrected, but the 0.10 m full-route "
            "gate still requires independent intermediate repeated anchors."
        ),
        "session": str(session_dir.expanduser().resolve()),
        "source_trajectory": str(trajectory_path.expanduser().resolve()),
        "optimized_trajectory": str(trajectory_output),
        "config": asdict(config),
        "trajectory_alignment": aligned.report(),
        "trajectory_pose_diagnostics": pose_diagnostics,
        "window_alignments": records,
        "accepted_window_count": sum(bool(item["accepted"]) for item in records),
        "consensus_translation_spread_m": translation_spread,
        "consensus_rotation_spread_deg": rotation_spread,
        "terminal_constraint_qualified": True,
        "raw_endpoint_translation_m": raw_endpoint[:3, 3].tolist(),
        "raw_endpoint_rotation_rpy_deg": Rotation.from_matrix(
            raw_endpoint[:3, :3]
        ).as_euler("xyz", degrees=True).tolist(),
        "measured_endpoint_translation_m": terminal_target[:3, 3].tolist(),
        "measured_endpoint_rotation_rpy_deg": Rotation.from_matrix(
            terminal_target[:3, :3]
        ).as_euler("xyz", degrees=True).tolist(),
        "optimized_endpoint_translation_m": optimized_endpoint[:3, 3].tolist(),
        "optimized_endpoint_rotation_rpy_deg": Rotation.from_matrix(
            optimized_endpoint[:3, :3]
        ).as_euler("xyz", degrees=True).tolist(),
        "constructed_endpoint_residual_m": float(np.linalg.norm(
            endpoint_residual[:3, 3]
        )),
        "constructed_endpoint_residual_deg": rotation_distance_deg(
            endpoint_residual[:3, :3]
        ),
        "correction_diagnostics": correction_diagnostics,
    }
    report_path = output / "loop_closure_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", type=Path, required=True)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = optimize_route_loop(args.session, args.trajectory, args.output_dir)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
