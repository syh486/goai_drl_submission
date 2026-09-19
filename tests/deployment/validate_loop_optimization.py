"""Contract checks for formal loop-optimization qualification."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile

import numpy as np

from deployment.mapping.validate_loop_optimization import validate_loop_optimization


def _write_fixture(root: Path) -> None:
    count = 101
    poses = np.tile(np.eye(4), (count, 1, 1))
    poses[:, 0, 3] = np.linspace(0.0, 2.0, count)
    np.savez_compressed(
        root / "optimized_route_trajectory.npz",
        poses=poses,
        timestamps_s=np.arange(count, dtype=np.float64) * 0.1,
        keyframe_indices=np.arange(count, dtype=np.int64),
    )
    report = {
        "terminal_constraint_qualified": True,
        "accepted_window_count": 3,
        "consensus_translation_spread_m": 0.02,
        "consensus_rotation_spread_deg": 0.3,
        "constructed_endpoint_residual_m": 0.0,
        "constructed_endpoint_residual_deg": 0.0,
        "full_route_metric_accuracy_qualified": False,
        "config": {
            "max_consensus_translation_m": 0.10,
            "max_consensus_rotation_deg": 1.0,
        },
    }
    (root / "loop_closure_report.json").write_text(
        json.dumps(report), encoding="utf-8"
    )


def main() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        _write_fixture(root)
        result = validate_loop_optimization(root)
        assert result["qualified"] is True
        assert result["pose_count"] == 101
        assert result["full_route_metric_accuracy_qualified"] is False

        report_path = root / "loop_closure_report.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        report["terminal_constraint_qualified"] = False
        report_path.write_text(json.dumps(report), encoding="utf-8")
        try:
            validate_loop_optimization(root)
        except ValueError as error:
            assert "not qualified" in str(error)
        else:
            raise AssertionError("an unqualified loop was accepted")
    print("VALIDATE_LOOP_OPTIMIZATION_OK")


if __name__ == "__main__":
    main()
