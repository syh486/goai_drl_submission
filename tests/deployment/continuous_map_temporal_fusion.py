"""Contract checks for bounded causal map-to-odometry updates."""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from deployment.localization.continuous_map_localization import _bounded_se3_update


def main() -> None:
    current = np.eye(4, dtype=np.float64)
    target = np.eye(4, dtype=np.float64)
    target[:3, 3] = (2.0, 0.0, 0.0)
    target[:3, :3] = Rotation.from_euler("z", 20.0, degrees=True).as_matrix()

    updated = _bounded_se3_update(
        current,
        target,
        translation_gain=0.20,
        rotation_gain=0.20,
        max_translation_step_m=0.25,
        max_rotation_step_deg=2.0,
    )
    assert np.allclose(updated[:3, 3], (0.25, 0.0, 0.0), atol=1.0e-12)
    yaw = Rotation.from_matrix(updated[:3, :3]).as_euler("zyx", degrees=True)[0]
    assert np.isclose(yaw, 2.0, atol=1.0e-10)
    assert np.allclose(updated[3], (0.0, 0.0, 0.0, 1.0))

    small = target.copy()
    small[:3, 3] = (0.20, -0.10, 0.05)
    small[:3, :3] = Rotation.from_euler("xyz", (2.0, -1.0, 3.0), degrees=True).as_matrix()
    damped = _bounded_se3_update(
        current,
        small,
        translation_gain=0.50,
        rotation_gain=0.50,
        max_translation_step_m=1.0,
        max_rotation_step_deg=10.0,
    )
    assert np.allclose(damped[:3, 3], 0.5 * small[:3, 3])
    residual = Rotation.from_matrix(damped[:3, :3]).inv() * Rotation.from_matrix(
        small[:3, :3]
    )
    original_angle = np.linalg.norm(Rotation.from_matrix(small[:3, :3]).as_rotvec())
    assert np.isclose(np.linalg.norm(residual.as_rotvec()), 0.5 * original_angle)
    print("CONTINUOUS_MAP_TEMPORAL_FUSION_OK")


if __name__ == "__main__":
    main()
