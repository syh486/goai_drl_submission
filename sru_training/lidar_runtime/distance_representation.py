from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


DEFAULT_FLAT_LIDAR_REFERENCE = (
    Path(__file__).resolve().parent / "assets/flat_lidar_reference/flat_lidar_reference.npz"
)


@dataclass(frozen=True)
class FlatLidarReference:
    front_d: np.ndarray
    rear_d: np.ndarray

    @classmethod
    def load(cls, reference_path: str | Path = DEFAULT_FLAT_LIDAR_REFERENCE) -> "FlatLidarReference":
        path = Path(reference_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Flat LiDAR reference not found: {path}")
        with np.load(path, allow_pickle=False) as data:
            front_d = data["front_d"].astype(np.float32, copy=True)
            rear_d = data["rear_d"].astype(np.float32, copy=True)
        return cls(front_d=front_d, rear_d=rear_d)


def flat_minus_distance(flat_reference_d: np.ndarray, distance_d: np.ndarray) -> np.ndarray:
    return flat_reference_d.astype(np.float32, copy=False) - distance_d.astype(np.float32, copy=False)


def flat_minus_distance_torch(flat_reference_d: torch.Tensor, distance_d: torch.Tensor) -> torch.Tensor:
    return flat_reference_d - distance_d
