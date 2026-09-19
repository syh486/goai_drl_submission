"""Validate multi-lap route refinement before building a deployment map."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


QUALITY_LIMITS = {
    "canonical_shape_prior_translation_p95_m": 0.10,
    "canonical_shape_prior_translation_max_m": 0.15,
    "cross_lap_translation_p95_m": 0.10,
    "cross_lap_translation_max_m": 0.15,
    "cross_lap_rotation_p95_deg": 8.0,
    "support_odometry_translation_p95_m": 0.15,
    "support_odometry_translation_max_m": 0.25,
    "canonical_endpoint_translation_max_m": 0.03,
    "canonical_endpoint_rotation_max_deg": 1.0,
    "canonical_correction_translation_max_m": 2.50,
    "canonical_correction_rotation_max_deg": 5.0,
}


def _factor_value(
    statistics: dict[str, object], kind: str, field: str
) -> float:
    if kind not in statistics or field not in statistics[kind]:
        raise ValueError(f"missing optimization statistic: {kind}.{field}")
    return float(statistics[kind][field])


def multilap_quality_measurements(
    optimization: dict[str, object],
) -> dict[str, float]:
    statistics = optimization["factor_statistics_after"]
    return {
        "canonical_shape_prior_translation_p95_m": _factor_value(
            statistics, "canonical_shape_prior", "translation_p95_m"
        ),
        "canonical_shape_prior_translation_max_m": _factor_value(
            statistics, "canonical_shape_prior", "translation_max_m"
        ),
        "cross_lap_translation_p95_m": _factor_value(
            statistics, "cross_lap", "translation_p95_m"
        ),
        "cross_lap_translation_max_m": _factor_value(
            statistics, "cross_lap", "translation_max_m"
        ),
        "cross_lap_rotation_p95_deg": _factor_value(
            statistics, "cross_lap", "rotation_p95_deg"
        ),
        "support_odometry_translation_p95_m": _factor_value(
            statistics, "support_odometry", "translation_p95_m"
        ),
        "support_odometry_translation_max_m": _factor_value(
            statistics, "support_odometry", "translation_max_m"
        ),
        "canonical_endpoint_translation_max_m": _factor_value(
            statistics, "canonical_endpoint", "translation_max_m"
        ),
        "canonical_endpoint_rotation_max_deg": _factor_value(
            statistics, "canonical_endpoint", "rotation_max_deg"
        ),
        "canonical_correction_translation_max_m": float(
            optimization["canonical_correction_translation_max_m"]
        ),
        "canonical_correction_rotation_max_deg": float(
            optimization["canonical_correction_rotation_max_deg"]
        ),
    }


def multilap_quality_failures(
    measurements: dict[str, float],
) -> dict[str, dict[str, float]]:
    return {
        key: {"actual": value, "limit": QUALITY_LIMITS[key]}
        for key, value in measurements.items()
        if not np.isfinite(value) or value > QUALITY_LIMITS[key]
    }


def validate_multilap_refinement(output_dir: Path) -> dict[str, object]:
    root = output_dir.expanduser().resolve()
    report_path = root / "multilap_refinement_report.json"
    trajectory_path = root / "refined_route_trajectory.npz"
    if not report_path.is_file() or not trajectory_path.is_file():
        raise ValueError(f"incomplete multi-lap refinement output: {root}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    optimization = report["optimization"]
    extraction = report["constraint_extraction"]
    config = report["config"]
    if optimization.get("success") is not True:
        raise ValueError("multi-lap graph optimizer did not converge")
    accepted = int(extraction["accepted_constraints"])
    minimum = int(config["minimum_constraints"])
    if accepted < minimum:
        raise ValueError(f"only {accepted} cross-lap constraints passed")

    measurements = multilap_quality_measurements(optimization)
    failed = multilap_quality_failures(measurements)
    if failed:
        raise ValueError(f"multi-lap refinement quality gates failed: {failed}")

    with np.load(trajectory_path, allow_pickle=False) as archive:
        poses = np.asarray(archive["poses"], dtype=np.float64)
        timestamps = np.asarray(archive["timestamps_s"], dtype=np.float64)
        keyframes = np.asarray(archive["keyframe_indices"], dtype=np.int64)
    if poses.ndim != 3 or poses.shape[1:] != (4, 4) or len(poses) < 100:
        raise ValueError(f"invalid refined trajectory shape: {poses.shape}")
    if timestamps.shape != (len(poses),) or keyframes.shape != (len(poses),):
        raise ValueError("refined trajectory metadata is misaligned")
    if not np.isfinite(poses).all() or np.any(np.diff(timestamps) <= 0.0):
        raise ValueError("refined trajectory is non-finite or non-monotonic")
    return {
        "qualified": True,
        "output_dir": str(root),
        "accepted_constraints": accepted,
        "quality_limits": QUALITY_LIMITS,
        "measurements": measurements,
        "full_route_metric_accuracy_qualified": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    print(json.dumps(validate_multilap_refinement(args.output_dir), indent=2))


if __name__ == "__main__":
    main()
