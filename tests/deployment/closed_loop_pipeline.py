"""Static contract checks for the formal multi-lap mapping pipeline."""

from __future__ import annotations

from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    scripts = root / "deployment" / "scripts" / "mapping"
    finalizer = (scripts / "finalize_route_map.sh").read_text(encoding="utf-8")
    builder = (scripts / "build_multilap_route_map.sh").read_text(encoding="utf-8")
    evaluator = (scripts / "evaluate_multilap_map.sh").read_text(encoding="utf-8")

    assert "build_multilap_route_map.sh" in finalizer
    assert "refined_route_trajectory.npz" in finalizer
    assert "--support-lap" not in finalizer

    for required in (
        "optimize_route_loop.sh",
        "deployment.mapping.refine_route_multilap",
        "deployment.mapping.validate_multilap_refinement",
        "refined_route_trajectory.npz",
    ):
        assert required in builder
    assert builder.index("optimize_route_loop.sh") < builder.index(
        "deployment.mapping.refine_route_multilap"
    )
    assert builder.index("deployment.mapping.refine_route_multilap") < builder.index(
        "deployment.mapping.build_topometric_route_map",
        builder.index("deployment.mapping.refine_route_multilap"),
    )

    assert "deployment.localization.evaluate_topometric_replay" in evaluator
    assert "deployment.localization.evaluate_metric_endpoint" in evaluator
    assert "deployment.localization.evaluate_multilap_holdout" in evaluator

    # Do not reintroduce a raw-trajectory or endpoint-only formal-map shortcut.
    assert not (scripts / "build_topometric_map.sh").exists()
    print("CLOSED_LOOP_PIPELINE_OK")


if __name__ == "__main__":
    main()
