"""Small rotation helpers shared by deployment modules."""

from __future__ import annotations

import numpy as np


def quat_wxyz_to_rotmat(quaternion: np.ndarray) -> np.ndarray:
    quat = np.asarray(quaternion, dtype=np.float64)
    if quat.shape != (4,):
        raise ValueError(f"quaternion must have shape (4,), got {quat.shape}")
    norm = float(np.linalg.norm(quat))
    if norm < 1.0e-12 or not np.isfinite(norm):
        raise ValueError("quaternion must be finite and non-zero")
    w, x, y, z = quat / norm
    return np.asarray((
        (1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)),
        (2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)),
        (2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)),
    ))
