"""Validate a closed-loop route optimization before formal map building."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def validate_loop_optimization(output_dir: Path) -> dict[str, object]:
    root = output_dir.expanduser().resolve()
    report_path = root / "loop_closure_report.json"
    trajectory_path = root / "optimized_route_trajectory.npz"
    if not report_path.is_file():
        raise ValueError(f"loop closure report is missing: {report_path}")
    if not trajectory_path.is_file():
        raise ValueError(f"optimized route trajectory is missing: {trajectory_path}")

    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("terminal_constraint_qualified") is not True:
        raise ValueError("terminal loop constraint is not qualified")
    if int(report.get("accepted_window_count", 0)) < 2:
        raise ValueError("fewer than two endpoint windows accepted the loop")

    config = report.get("config", {})
    translation_spread = float(report["consensus_translation_spread_m"])
    rotation_spread = float(report["consensus_rotation_spread_deg"])
    if translation_spread > float(config["max_consensus_translation_m"]):
        raise ValueError("endpoint translation consensus exceeds its configured gate")
    if rotation_spread > float(config["max_consensus_rotation_deg"]):
        raise ValueError("endpoint rotation consensus exceeds its configured gate")
    if float(report["constructed_endpoint_residual_m"]) > 1.0e-6:
        raise ValueError("optimized trajectory does not satisfy the endpoint position factor")
    if float(report["constructed_endpoint_residual_deg"]) > 1.0e-5:
        raise ValueError("optimized trajectory does not satisfy the endpoint rotation factor")

    with np.load(trajectory_path) as archive:
        poses = np.asarray(archive["poses"], dtype=np.float64)
        timestamps = np.asarray(archive["timestamps_s"], dtype=np.float64)
        keyframe_indices = np.asarray(archive["keyframe_indices"], dtype=np.int64)
    if poses.ndim != 3 or poses.shape[1:] != (4, 4) or len(poses) < 100:
        raise ValueError(f"invalid optimized pose array shape: {poses.shape}")
    if timestamps.shape != (len(poses),) or keyframe_indices.shape != (len(poses),):
        raise ValueError("optimized trajectory metadata is not aligned with its poses")
    if not np.isfinite(poses).all() or not np.isfinite(timestamps).all():
        raise ValueError("optimized trajectory contains non-finite values")
    if np.any(np.diff(timestamps) <= 0.0):
        raise ValueError("optimized trajectory timestamps are not strictly increasing")
    if np.any(np.diff(keyframe_indices) <= 0):
        raise ValueError("optimized trajectory keyframe indices are not strictly increasing")

    summary = {
        "qualified": True,
        "output_dir": str(root),
        "pose_count": len(poses),
        "accepted_window_count": int(report["accepted_window_count"]),
        "consensus_translation_spread_m": translation_spread,
        "consensus_rotation_spread_deg": rotation_spread,
        "full_route_metric_accuracy_qualified": bool(
            report.get("full_route_metric_accuracy_qualified", False)
        ),
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    print(json.dumps(validate_loop_optimization(args.output_dir), indent=2))


if __name__ == "__main__":
    main()
