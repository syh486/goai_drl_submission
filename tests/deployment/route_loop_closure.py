"""Contract checks for smooth terminal loop correction."""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from deployment.mapping.optimize_route_loop import apply_distributed_loop_correction


def main() -> None:
    poses = np.tile(np.eye(4), (101, 1, 1))
    angle = np.linspace(0.0, 2.0 * np.pi, len(poses))
    poses[:, 0, 3] = 10.0 * np.sin(angle) + np.linspace(0.0, 0.4, len(poses))
    poses[:, 1, 3] = 10.0 * (1.0 - np.cos(angle))
    poses[:, 2, 3] = np.linspace(0.0, -1.2, len(poses))
    poses[:, :3, :3] = Rotation.from_euler(
        "z", np.linspace(0.0, 25.0, len(poses)), degrees=True
    ).as_matrix()
    target = np.eye(4)
    target[:3, :3] = Rotation.from_euler("z", 30.0, degrees=True).as_matrix()
    target[:3, 3] = (0.08, 0.04, 0.02)
    corrected, diagnostics = apply_distributed_loop_correction(poses, target)
    np.testing.assert_allclose(corrected[0], poses[0], atol=1.0e-12)
    np.testing.assert_allclose(corrected[-1], target, atol=1.0e-10)
    assert diagnostics["maximum_step_length_change_m"] < 0.05
    assert diagnostics["path_length_corrected_m"] > 1.0
    print("ROUTE_LOOP_CLOSURE_OK", diagnostics)


if __name__ == "__main__":
    main()
