"""Stable contracts for swapping onboard localization implementations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np


@dataclass(frozen=True)
class LocalizationEstimate:
    stamp_s: float
    position_map: np.ndarray
    rotation_map_body: np.ndarray
    velocity_body: np.ndarray
    covariance: np.ndarray
    healthy: bool
    state: str
    source: str

    def validate(self) -> None:
        if np.asarray(self.position_map).shape != (3,):
            raise ValueError("position_map must have shape (3,)")
        if np.asarray(self.rotation_map_body).shape != (3, 3):
            raise ValueError("rotation_map_body must have shape (3,3)")
        if np.asarray(self.velocity_body).shape != (3,):
            raise ValueError("velocity_body must have shape (3,)")
        if not np.isfinite(self.stamp_s):
            raise ValueError("localization stamp must be finite")


class OnboardLocalizer(Protocol):
    """Transport-independent localization backend used by route management."""

    def initialize(self, initial_pose: np.ndarray, initial_imu_wxyz: np.ndarray) -> None: ...

    def update(
        self,
        front_cloud: Any,
        rear_cloud: Any,
        imu_wheel_history: dict[str, np.ndarray],
    ) -> LocalizationEstimate: ...


class VelocityCommandSink(Protocol):
    """Safety-aware output boundary for the official low-level controller."""

    def publish(self, forward_mps: float, yaw_rate_rps: float) -> None: ...

    def stop(self) -> None: ...
