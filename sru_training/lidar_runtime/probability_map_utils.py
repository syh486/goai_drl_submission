from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch


FRONT_PROBABILITY_MAP_FILENAME = "nav_front_rslidar_points_probability_96x900.npy"
REAR_PROBABILITY_MAP_FILENAME = "nav_rear_rslidar_points_probability_96x900.npy"
DEFAULT_PROBABILITY_MAP_DIR = Path(__file__).resolve().parent / "assets/probability_maps"


def resolve_probability_map_dir(probability_map_dir: str | Path | None = None) -> Path | None:
    candidates: list[Path] = []
    env_dir = os.environ.get("SRU_LIDAR_PROBABILITY_MAP_DIR")
    if env_dir:
        candidates.append(Path(env_dir).expanduser().resolve())
    if probability_map_dir is not None:
        candidates.append(Path(probability_map_dir).expanduser().resolve())
    candidates.append(DEFAULT_PROBABILITY_MAP_DIR)

    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        if candidate.exists():
            return candidate
    return None


def resolve_probability_map_paths(
    *,
    probability_map_dir: str | Path | None = None,
    require_probability_maps: bool = False,
) -> tuple[str, str]:
    directory = resolve_probability_map_dir(probability_map_dir)
    if directory is None:
        if require_probability_maps:
            raise FileNotFoundError("Probability map directory could not be resolved.")
        return "", ""

    front_path = directory / FRONT_PROBABILITY_MAP_FILENAME
    rear_path = directory / REAR_PROBABILITY_MAP_FILENAME
    if require_probability_maps and (not front_path.exists() or not rear_path.exists()):
        raise FileNotFoundError(
            f"Probability maps not found under {directory}. "
            f"Expected {FRONT_PROBABILITY_MAP_FILENAME} and {REAR_PROBABILITY_MAP_FILENAME}."
        )
    return str(front_path) if front_path.exists() else "", str(rear_path) if rear_path.exists() else ""


def load_probability_map(
    path: str | Path | None,
    *,
    target_height: int = 96,
    target_width: int = 90,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    if path is None or str(path) == "":
        return torch.ones((target_height, target_width), dtype=dtype)

    source = np.load(Path(path).expanduser().resolve()).astype(np.float32, copy=False)
    if source.ndim != 2:
        raise RuntimeError(f"Probability map at {path} must be 2D, got shape {tuple(source.shape)}.")

    vertical_indices = np.round(np.linspace(0, source.shape[0] - 1, num=target_height)).astype(np.int64)
    horizontal_indices = np.round(np.linspace(0, source.shape[1] - 1, num=target_width)).astype(np.int64)
    downsampled = source[vertical_indices][:, horizontal_indices]
    return torch.from_numpy(downsampled).to(dtype=dtype)
