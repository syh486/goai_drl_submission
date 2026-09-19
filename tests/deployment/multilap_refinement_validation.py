"""Contract checks for formal multi-lap refinement quality gates."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile

import numpy as np

from deployment.mapping.validate_multilap_refinement import (
    validate_multilap_refinement,
)


def _statistics() -> dict[str, object]:
    return {
        "canonical_shape_prior": {
            "translation_p95_m": 0.05,
            "translation_max_m": 0.08,
        },
        "cross_lap": {
            "translation_p95_m": 0.06,
            "translation_max_m": 0.09,
            "rotation_p95_deg": 2.0,
        },
        "support_odometry": {
            "translation_p95_m": 0.08,
            "translation_max_m": 0.12,
        },
        "canonical_endpoint": {
            "translation_max_m": 0.01,
            "rotation_max_deg": 0.2,
        },
    }


def main() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        report = {
            "config": {"minimum_constraints": 20},
            "constraint_extraction": {"accepted_constraints": 24},
            "optimization": {
                "success": True,
                "factor_statistics_after": _statistics(),
                "canonical_correction_translation_max_m": 1.0,
                "canonical_correction_rotation_max_deg": 1.0,
            },
        }
        (root / "multilap_refinement_report.json").write_text(
            json.dumps(report), encoding="utf-8"
        )
        poses = np.repeat(np.eye(4)[None], 100, axis=0)
        np.savez_compressed(
            root / "refined_route_trajectory.npz",
            poses=poses,
            timestamps_s=np.arange(100, dtype=np.float64) * 0.2,
            keyframe_indices=np.arange(100, dtype=np.int64),
        )
        result = validate_multilap_refinement(root)
        assert result["qualified"] is True
        assert result["full_route_metric_accuracy_qualified"] is False

        report["optimization"]["factor_statistics_after"]["cross_lap"][
            "translation_p95_m"
        ] = 0.11
        (root / "multilap_refinement_report.json").write_text(
            json.dumps(report), encoding="utf-8"
        )
        try:
            validate_multilap_refinement(root)
        except ValueError as error:
            assert "quality gates failed" in str(error)
        else:
            raise AssertionError("over-limit cross-lap residual was accepted")
    print("MULTILAP_REFINEMENT_VALIDATION_OK")


if __name__ == "__main__":
    main()
