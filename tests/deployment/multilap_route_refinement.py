"""Synthetic contract checks for sparse multi-lap route refinement."""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from deployment.mapping.refine_route_multilap import (
    MultiLapRefinementConfig,
    interpolate_canonical_corrections,
    optimize_two_lap_graph,
)


def _poses(count: int, drift: np.ndarray | None = None) -> np.ndarray:
    values = np.repeat(np.eye(4)[None], count, axis=0)
    values[:, 0, 3] = np.linspace(0.0, 30.0, count)
    values[:, 1, 3] = 2.0 * np.sin(np.linspace(0.0, np.pi, count))
    values[:, :3, :3] = Rotation.from_euler(
        "z", np.linspace(0.0, 20.0, count), degrees=True
    ).as_matrix()
    if drift is not None:
        values[:, :3, 3] += np.linspace(0.0, 1.0, count)[:, None] * drift
    return values


def main() -> None:
    count = 21
    canonical = _poses(count)
    support = _poses(count, np.asarray((0.6, -0.4, 1.2)))
    constraints = []
    for index in range(count):
        measurement = np.linalg.inv(canonical[index]) @ _poses(count)[index]
        constraints.append({
            "canonical_frame": index,
            "support_frame": index,
            "canonical_from_support": measurement.tolist(),
            "cycle_translation_m": 0.01,
        })
    optimized, report = optimize_two_lap_graph(
        canonical,
        support,
        constraints,
        MultiLapRefinementConfig(
            minimum_constraints=5,
            max_nfev=150,
            optimizer_backend="scipy",
        ),
    )
    assert report["success"] is True
    assert report["factor_statistics_after"]["support_odometry"]["translation_p95_m"] < (
        report["factor_statistics_before"]["support_odometry"]["translation_p95_m"]
    )
    assert report["factor_statistics_after"]["cross_lap"]["translation_p95_m"] < 0.02
    assert report["canonical_correction_translation_max_m"] < 0.25
    refined = interpolate_canonical_corrections(
        canonical, np.arange(count, dtype=np.int64), optimized
    )
    assert refined.shape == canonical.shape
    np.testing.assert_allclose(refined[0], canonical[0], atol=1.0e-8)
    print("MULTILAP_ROUTE_REFINEMENT_OK")


if __name__ == "__main__":
    main()
