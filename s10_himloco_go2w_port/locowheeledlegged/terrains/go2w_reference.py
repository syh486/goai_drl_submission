"""Composite terrain used by the locked Go2W reference."""

from __future__ import annotations

from dataclasses import MISSING

import numpy as np
from scipy import interpolate
from isaaclab.terrains.height_field.hf_terrains_cfg import HfTerrainBaseCfg
from isaaclab.terrains.height_field.utils import height_field_to_mesh
from isaaclab.utils import configclass


@height_field_to_mesh
def rough_pyramid_slope(difficulty: float, cfg: "HfGo2WRoughPyramidSlopeTerrainCfg") -> np.ndarray:
    """Positive pyramid slope plus uniform roughness, in the reference order."""

    width = int(cfg.size[0] / cfg.horizontal_scale)
    length = int(cfg.size[1] / cfg.horizontal_scale)
    slope = cfg.slope_range[0] + difficulty * (cfg.slope_range[1] - cfg.slope_range[0])
    height_max = int(slope * cfg.size[0] / 2.0 / cfg.vertical_scale)
    center_x, center_y = width // 2, length // 2
    x = (center_x - np.abs(center_x - np.arange(width))) / center_x
    y = (center_y - np.abs(center_y - np.arange(length))) / center_y
    heights = height_max * x.reshape(width, 1) * y.reshape(1, length)
    platform = int(cfg.platform_width / cfg.horizontal_scale / 2)
    platform_height = heights[center_x - platform, center_y - platform]
    heights = np.clip(heights, min(0, platform_height), max(0, platform_height))

    amplitude = cfg.noise_range[0] + difficulty * (cfg.noise_range[1] - cfg.noise_range[0])
    down_w = int(cfg.size[0] / cfg.downsampled_scale)
    down_l = int(cfg.size[1] / cfg.downsampled_scale)
    h_min = int(-amplitude / cfg.vertical_scale)
    h_max = int(amplitude / cfg.vertical_scale)
    h_step = max(1, int(cfg.noise_step / cfg.vertical_scale))
    choices = np.arange(h_min, h_max + h_step, h_step)
    rough_low = np.random.choice(choices, size=(down_w, down_l))
    interp = interpolate.RectBivariateSpline(
        np.linspace(0, cfg.size[0], down_w),
        np.linspace(0, cfg.size[1], down_l),
        rough_low,
    )
    rough = interp(np.linspace(0, cfg.size[0], width), np.linspace(0, cfg.size[1], length))
    return np.rint(heights + rough).astype(np.int16)


@configclass
class HfGo2WRoughPyramidSlopeTerrainCfg(HfTerrainBaseCfg):
    function = rough_pyramid_slope
    slope_range: tuple[float, float] = MISSING
    noise_range: tuple[float, float] = MISSING
    noise_step: float = MISSING
    downsampled_scale: float = 0.2
    platform_width: float = 3.0
