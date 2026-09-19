"""Evaluate held-out terminal localization against independent endpoint scans."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path

import numpy as np

from deployment.mapping.mapping_geometry import rotation_distance_deg
from deployment.mapping.optimize_route_loop import RouteLoopConfig, estimate_endpoint_alignment
from deployment.common.trajectory_io import load_aligned_trajectory, prepare_trajectory_poses


def evaluate_metric_endpoint(
    map_dir: Path,
    reference_session: Path,
    reference_trajectory: Path,
    heldout_session: Path,
    heldout_trajectory: Path,
    replay_report: Path,
    *,
    output_report: Path | None = None,
    config: RouteLoopConfig = RouteLoopConfig(),
) -> dict[str, object]:
    reference_aligned = load_aligned_trajectory(
        reference_session, reference_trajectory
    )
    heldout_aligned = load_aligned_trajectory(heldout_session, heldout_trajectory)
    reference_poses, _ = prepare_trajectory_poses(reference_aligned, "lio")
    heldout_poses, _ = prepare_trajectory_poses(heldout_aligned, "lio")

    # Directly register the held-out terminal neighborhood into the reference
    # map start frame. This is independent of both GLIM terminal drift and the
    # live localizer output, and avoids composing two endpoint alignments.
    endpoint_config = replace(
        config,
        window_frames=(6, 8, 10),
        # Roll/pitch differ slightly with suspension loading even on the same
        # flat endpoint. Position repeatability remains the 0.10 m hard gate.
        max_consensus_rotation_deg=2.0,
    )
    expected_terminal, endpoint_records, endpoint_spread_m, endpoint_spread_deg = (
        estimate_endpoint_alignment(
            reference_poses,
            list(reference_aligned.keyframe_files),
            heldout_poses,
            list(heldout_aligned.keyframe_files),
            reference_terminal=False,
            live_terminal=True,
            config=endpoint_config,
        )
    )
    expected_map_from_terminal = expected_terminal.transform_reference_live

    replay = json.loads(replay_report.expanduser().resolve().read_text(
        encoding="utf-8"
    ))
    accepted = [item for item in replay["records"] if item["accepted"]]
    if not accepted or "map_position_m" not in accepted[-1]:
        raise ValueError("replay report does not contain localized map poses")
    actual_map_from_terminal = np.eye(4, dtype=np.float64)
    actual_map_from_terminal[:3, 3] = accepted[-1]["map_position_m"]
    actual_map_from_terminal[:3, :3] = accepted[-1]["map_rotation_matrix"]
    residual = np.linalg.inv(expected_map_from_terminal) @ actual_map_from_terminal
    translation_error = float(np.linalg.norm(residual[:3, 3]))
    report = {
        "schema_version": 1,
        "map_dir": str(map_dir.expanduser().resolve()),
        "heldout_session": str(heldout_session.expanduser().resolve()),
        "replay_report": str(replay_report.expanduser().resolve()),
        "metric_gate_m": 0.10,
        "endpoint_metric_qualified": bool(translation_error <= 0.10),
        "full_route_metric_accuracy_qualified": False,
        "qualification_note": (
            "Endpoint repeatability is necessary but intermediate independent "
            "anchors are still required for the full-route 0.10 m gate."
        ),
        "endpoint_observation_translation_spread_m": endpoint_spread_m,
        "endpoint_observation_rotation_spread_deg": endpoint_spread_deg,
        "expected_terminal_position_m": expected_map_from_terminal[:3, 3].tolist(),
        "actual_terminal_position_m": actual_map_from_terminal[:3, 3].tolist(),
        "terminal_translation_error_m": translation_error,
        "terminal_rotation_error_deg": rotation_distance_deg(residual[:3, :3]),
        "endpoint_observation_windows": endpoint_records,
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
    parser.add_argument("--reference-session", type=Path, required=True)
    parser.add_argument("--reference-trajectory", type=Path, required=True)
    parser.add_argument("--heldout-session", type=Path, required=True)
    parser.add_argument("--heldout-trajectory", type=Path, required=True)
    parser.add_argument("--replay-report", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    args = parser.parse_args()
    report = evaluate_metric_endpoint(
        args.map_dir,
        args.reference_session,
        args.reference_trajectory,
        args.heldout_session,
        args.heldout_trajectory,
        args.replay_report,
        output_report=args.output_report,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
