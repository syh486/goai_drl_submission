"""Contract checks for causal held-out multi-lap evaluation."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile

import numpy as np

from deployment.localization.evaluate_multilap_holdout import (
    _consensus_subset,
    evaluate_multilap_holdout,
)
from deployment.mapping.refine_route_multilap import MultiLapRefinementConfig


def _result(x: float, window: int, cost: float) -> dict[str, object]:
    transform = np.eye(4, dtype=np.float64)
    transform[0, 3] = x
    return {
        "transform": transform.tolist(),
        "window_frames": window,
        "initialization": "identity",
        "cost": cost,
    }


def _write_cached_references(
    output: Path, references: list[dict[str, object]]
) -> None:
    output.mkdir()
    (output / "heldout_causal_references.json").write_text(
        json.dumps(references), encoding="utf-8"
    )
    (output / "heldout_reference_report.json").write_text(
        json.dumps({"method": "synthetic causal references"}), encoding="utf-8"
    )


def main() -> None:
    subset, diagnostics = _consensus_subset([
        _result(0.00, 5, 0.20),
        _result(0.03, 7, 0.10),
        _result(0.05, 9, 0.15),
        _result(0.40, 11, 0.01),
    ])
    assert len(subset) == 3
    assert diagnostics["medoid_window_frames"] == 7
    assert diagnostics["translation_spread_m"] <= 0.05 + 1.0e-9

    # A cheaper but inconsistent initialization must not displace the
    # cross-window-consistent hypothesis for the same temporal window.
    subset, diagnostics = _consensus_subset([
        _result(0.00, 5, 0.20),
        _result(0.50, 5, 0.01),
        _result(0.02, 7, 0.20),
        _result(0.52, 7, 0.01),
        _result(0.04, 9, 0.20),
    ])
    assert len(subset) == 3
    assert {int(item["window_frames"]) for item in subset} == {5, 7, 9}
    assert max(float(np.asarray(item["transform"])[0, 3]) for item in subset) < 0.1
    assert diagnostics["translation_spread_m"] <= 0.04 + 1.0e-9

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        map_dir = root / "map"
        submaps = map_dir / "submaps"
        submaps.mkdir(parents=True)
        np.savez_compressed(
            submaps / "submap_0000.npz", anchor_pose=np.eye(4)
        )
        (map_dir / "localization_map_manifest.json").write_text(
            json.dumps({
                "submaps": [{
                    "route_index": 0,
                    "reference_session": 0,
                    "file": "submaps/submap_0000.npz",
                }]
            }),
            encoding="utf-8",
        )
        replay = root / "replay.json"
        replay.write_text(
            json.dumps({
                "records": [
                    {
                        "frame": 9,
                        "map_position_m": [50.0, 0.0, 0.0],
                        "map_rotation_matrix": np.eye(3).tolist(),
                        "traveled_distance_m": 2.0,
                        "accepted": True,
                        "route_index": 0,
                    },
                    {
                        "frame": 10,
                        "map_position_m": [0.02, 0.0, 0.0],
                        "map_rotation_matrix": np.eye(3).tolist(),
                        "traveled_distance_m": 2.0,
                        "accepted": True,
                        "route_index": 0,
                    },
                    {
                        "frame": 12,
                        "map_position_m": [1.0, 0.0, 0.0],
                        "map_rotation_matrix": np.eye(3).tolist(),
                        "traveled_distance_m": 0.5,
                        "accepted": True,
                        "route_index": 0,
                    },
                ]
            }),
            encoding="utf-8",
        )
        common_reference = {
            "route_index": 0,
            "route_fraction": 0.5,
            "canonical_from_support": np.eye(4).tolist(),
            "consensus": {"window_count": 3},
        }
        qualified_output = root / "qualified"
        _write_cached_references(qualified_output, [
            {**common_reference, "support_frame": 10},
            {**common_reference, "support_frame": 12},
        ])
        report = evaluate_multilap_holdout(
            map_dir,
            root / "unused_session",
            root / "unused_trajectory",
            replay,
            qualified_output,
            MultiLapRefinementConfig(minimum_constraints=1),
        )
        assert report["evaluated_anchor_count"] == 1
        assert report["excluded_startup_anchor_count"] == 1
        assert report["maximum_translation_error_m"] == 0.02
        assert report["heldout_map_repeatability_qualified"] is True

        missing_output = root / "missing_exact_frame"
        _write_cached_references(missing_output, [
            {**common_reference, "support_frame": 11}
        ])
        report = evaluate_multilap_holdout(
            map_dir,
            root / "unused_session",
            root / "unused_trajectory",
            replay,
            missing_output,
            MultiLapRefinementConfig(minimum_constraints=1),
        )
        assert report["evaluated_anchor_count"] == 0
        assert report["anchors"][0]["failure"] == (
            "online replay has no matching frame"
        )
        assert report["heldout_map_repeatability_qualified"] is False
    print("MULTILAP_HOLDOUT_PROTOCOL_OK")


if __name__ == "__main__":
    main()
