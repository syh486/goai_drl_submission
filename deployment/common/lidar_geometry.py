"""Torch-free Airy geometry shared by localization and policy adapters."""

from __future__ import annotations

import numpy as np


NATIVE_HEIGHT = 96
MAX_RANGE_M = 10.0
MIN_RANGE_M = 0.2
INVALID_RANGE_THRESHOLD_M = 9.9

AIRY_VERTICAL_RAY_ANGLES = np.asarray(
    (
        -0.07, 0.88, 1.81, 2.76, 3.69, 4.62, 5.54, 6.48, 7.41, 8.34, 9.27, 10.21,
        11.15, 12.09, 13.03, 13.98, 14.92, 15.87, 16.82, 17.77, 18.72, 19.67,
        20.62, 21.57, 22.51, 23.45, 24.4, 25.33, 26.28, 27.21, 28.15, 29.08,
        30.02, 30.95, 31.88, 32.82, 33.74, 34.68, 35.62, 36.55, 37.5, 38.43,
        39.37, 40.31, 41.25, 42.21, 43.16, 44.09, 45.05, 46.0, 46.95, 47.9,
        48.85, 49.8, 50.73, 51.69, 52.62, 53.56, 54.5, 55.45, 56.37, 57.3,
        58.24, 59.18, 60.12, 61.05, 61.99, 62.93, 63.86, 64.81, 65.76, 66.69,
        67.65, 68.6, 69.56, 70.51, 71.46, 72.42, 73.37, 74.33, 75.29, 76.24,
        77.19, 78.14, 79.07, 80.02, 80.96, 81.9, 82.84, 83.78, 84.7, 85.64,
        86.57, 87.52, 88.46, 89.4,
    ),
    dtype=np.float32,
)

S10_FRONT_POS = np.asarray((0.22341, 0.0, -0.0001), dtype=np.float32)
S10_REAR_POS = np.asarray((-0.22341, 0.0, -0.0001), dtype=np.float32)
S10_FRONT_ROT_WXYZ = np.asarray(
    (0.00084463, 0.7071065, -0.00028154, 0.7071065), dtype=np.float32
)
S10_REAR_ROT_WXYZ = np.asarray(
    (0.70703477, 0.0, -0.70717879, 0.0), dtype=np.float32
)


def build_sensor_frame_directions(horizontal_samples: int = 900) -> np.ndarray:
    horizontal_deg = (
        np.arange(horizontal_samples, dtype=np.float32) * (360.0 / horizontal_samples)
        - 180.0
    )
    pitch_deg, yaw_deg = np.meshgrid(
        AIRY_VERTICAL_RAY_ANGLES, horizontal_deg, indexing="xy"
    )
    pitch = np.deg2rad(pitch_deg.reshape(-1)) + np.pi / 2.0
    yaw = np.deg2rad(yaw_deg.reshape(-1))
    directions = -np.stack(
        (
            np.sin(pitch) * np.cos(yaw),
            np.sin(pitch) * np.sin(yaw),
            np.cos(pitch),
        ),
        axis=1,
    )
    image = directions.reshape(horizontal_samples, NATIVE_HEIGHT, 3).transpose(1, 0, 2)
    return np.roll(image, -horizontal_samples // 2, axis=1).astype(
        np.float32, copy=False
    )
