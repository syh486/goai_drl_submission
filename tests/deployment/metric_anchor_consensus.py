"""Contract checks for robust multi-window metric-anchor consensus."""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from deployment.localization.evaluate_metric_anchors import _robust_consensus
from deployment.localization.start_alignment import StartAlignmentResult


def result(x: float, yaw_deg: float) -> StartAlignmentResult:
    return StartAlignmentResult(
        rotation_reference_live=Rotation.from_euler(
            "z", yaw_deg, degrees=True
        ).as_matrix(),
        translation_reference_live_m=np.asarray((x, 0.0, 0.0)),
        yaw_deg=yaw_deg,
        rmse_m=0.10,
        overlap_fraction=0.80,
        correspondence_count=1000,
        candidate_margin_m=0.02,
    )


def main() -> None:
    selected, translation, rotation, count = _robust_consensus(
        [result(0.00, 0.0), result(0.03, 0.5), result(0.40, 8.0)],
        0.10,
        2.0,
    )
    assert count == 2
    assert translation <= 0.03 + 1.0e-9
    assert rotation <= 0.5 + 1.0e-9
    assert selected.translation_reference_live_m[0] in (0.0, 0.03)
    try:
        _robust_consensus(
            [result(0.0, 0.0), result(0.3, 5.0), result(0.6, 10.0)],
            0.10,
            2.0,
        )
    except ValueError as error:
        assert "consistent pair" in str(error)
    else:
        raise AssertionError("mutually inconsistent windows were accepted")
    print("METRIC_ANCHOR_CONSENSUS_OK")


if __name__ == "__main__":
    main()
