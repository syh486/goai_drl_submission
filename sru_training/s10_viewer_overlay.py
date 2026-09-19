"""MuJoCo viewer overlays for the target and sparse front/rear LiDAR hits."""

from __future__ import annotations

import mujoco
import numpy as np


GOAL_RGBA = np.asarray((0.1, 0.9, 0.2, 0.85), dtype=np.float32)
FRONT_RGBA = np.asarray((0.0, 0.85, 1.0, 0.85), dtype=np.float32)
REAR_RGBA = np.asarray((1.0, 0.15, 0.65, 0.85), dtype=np.float32)


def update_perception_overlay(
    viewer: mujoco.viewer.Handle,
    goal: np.ndarray,
    front_points: np.ndarray,
    rear_points: np.ndarray,
) -> None:
    """Draw one goal marker and as many LiDAR points as the viewer permits."""

    scene = viewer.user_scn
    scene.ngeom = 0
    marker = np.asarray(goal, dtype=np.float64).copy()
    marker[2] += 0.25
    _append_sphere(scene, marker, 0.22, GOAL_RGBA)
    for points, rgba in ((front_points, FRONT_RGBA), (rear_points, REAR_RGBA)):
        for point in np.asarray(points):
            if scene.ngeom >= scene.maxgeom:
                return
            _append_sphere(scene, point, 0.025, rgba)


def _append_sphere(scene: mujoco.MjvScene, position: np.ndarray, radius: float, rgba: np.ndarray) -> None:
    mujoco.mjv_initGeom(
        scene.geoms[scene.ngeom],
        mujoco.mjtGeom.mjGEOM_SPHERE,
        np.asarray((radius, radius, radius), dtype=np.float64),
        np.asarray(position, dtype=np.float64),
        np.eye(3, dtype=np.float64).reshape(-1),
        rgba,
    )
    scene.ngeom += 1
