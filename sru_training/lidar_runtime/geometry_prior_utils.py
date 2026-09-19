from __future__ import annotations

import numpy as np
import torch

from lidar_geometry import load_lidar_geometry


def build_geometry_prior_channels(
    *,
    target_height: int = 96,
    target_width: int = 90,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    geometry = load_lidar_geometry()

    vertical_source = np.asarray(geometry.vertical_ray_angles, dtype=np.float32)
    source_rows = np.arange(vertical_source.shape[0], dtype=np.float32)
    target_rows = np.linspace(0.0, float(vertical_source.shape[0] - 1), num=int(target_height), dtype=np.float32)
    elevation_deg = np.interp(target_rows, source_rows, vertical_source).astype(np.float32, copy=False)
    elevation_norm = elevation_deg / 90.0

    azimuth_deg = (
        np.arange(int(geometry.horizontal_samples), dtype=np.float32) * np.float32(geometry.horizontal_res)
        - np.float32(geometry.horizontal_fov / 2.0)
    )
    if geometry.horizontal_roll_columns != 0:
        azimuth_deg = np.roll(azimuth_deg, shift=-int(geometry.horizontal_roll_columns))

    if int(target_width) != int(geometry.horizontal_samples):
        source_cols = np.arange(int(geometry.horizontal_samples), dtype=np.float32)
        target_cols = np.linspace(0.0, float(geometry.horizontal_samples - 1), num=int(target_width), dtype=np.float32)
        azimuth_deg = np.interp(target_cols, source_cols, azimuth_deg).astype(np.float32, copy=False)

    azimuth_rad = np.deg2rad(azimuth_deg)
    azimuth_sin = np.sin(azimuth_rad, dtype=np.float32)
    azimuth_cos = np.cos(azimuth_rad, dtype=np.float32)

    elevation_map = np.repeat(elevation_norm[:, None], int(target_width), axis=1)
    azimuth_sin_map = np.repeat(azimuth_sin[None, :], int(target_height), axis=0)
    azimuth_cos_map = np.repeat(azimuth_cos[None, :], int(target_height), axis=0)
    stacked = np.stack((elevation_map, azimuth_sin_map, azimuth_cos_map), axis=0)
    return torch.from_numpy(stacked).to(dtype=dtype)
