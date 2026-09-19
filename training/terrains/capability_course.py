"""Controlled hfield courses for low-level locomotion capability calibration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROBOT_XML = (
    REPO_ROOT / "src/S10_sdk_deploy/S10_description/s10_mjcf/mjcf/S10.xml"
)


@dataclass(frozen=True)
class CapabilityCourse:
    kind: str
    obstacle_height: float
    total_height: float
    start: np.ndarray
    goal: np.ndarray
    top_entry_x: float
    stable_check_x: float
    horizontal_scale: float
    height_xy: np.ndarray


def build_capability_course(
    kind: str,
    obstacle_height: float,
    *,
    robot_xml: str | Path = DEFAULT_ROBOT_XML,
    horizontal_scale: float = 0.1,
    stair_steps: int = 5,
    stair_tread_depth: float = 0.4,
) -> tuple[mujoco.MjModel, CapabilityCourse]:
    """Build a straight single-platform or staircase hfield scene."""

    if kind not in {"platform", "stairs"}:
        raise ValueError("kind must be 'platform' or 'stairs'")
    if obstacle_height <= 0.0 or horizontal_scale <= 0.0:
        raise ValueError("course dimensions must be positive")
    if stair_steps < 1 or stair_tread_depth <= 0.0:
        raise ValueError("stair count and tread depth must be positive")

    x_min, x_max = -6.0, 6.0
    y_min, y_max = -3.0, 3.0
    x = np.linspace(
        x_min,
        x_max,
        int(round((x_max - x_min) / horizontal_scale)) + 1,
    )
    y = np.linspace(
        y_min,
        y_max,
        int(round((y_max - y_min) / horizontal_scale)) + 1,
    )
    height_xy = np.zeros((len(x), len(y)), dtype=np.float64)
    obstacle_start_x = -1.0
    if kind == "platform":
        height_xy[x >= obstacle_start_x, :] = obstacle_height
        total_height = obstacle_height
        top_entry_x = obstacle_start_x + 0.35
    else:
        for step in range(stair_steps):
            step_start = obstacle_start_x + step * stair_tread_depth
            height_xy[x >= step_start, :] = obstacle_height * (step + 1)
        total_height = obstacle_height * stair_steps
        top_entry_x = obstacle_start_x + stair_steps * stair_tread_depth

    start = np.asarray((-4.0, 0.0, 0.0), dtype=np.float64)
    goal = np.asarray((3.5, 0.0, total_height), dtype=np.float64)
    stable_check_x = max(top_entry_x + 0.5, 1.0)

    robot_xml = Path(robot_xml).expanduser().resolve()
    spec = mujoco.MjSpec.from_file(str(robot_xml))
    terrain_body = spec.worldbody.add_body(name="main_body")
    nrow, ncol = height_xy.T.shape
    spec.add_hfield(
        name="capability_hfield",
        nrow=nrow,
        ncol=ncol,
        size=(
            (x_max - x_min) * 0.5,
            (y_max - y_min) * 0.5,
            total_height,
            1.0,
        ),
        userdata=np.zeros(nrow * ncol, dtype=np.float32),
    )
    terrain_body.add_geom(
        name=f"capability_{kind}",
        type=mujoco.mjtGeom.mjGEOM_HFIELD,
        hfieldname="capability_hfield",
        group=0,
        contype=1,
        conaffinity=1,
        priority=1,
        condim=3,
        friction=(1.0, 0.01, 0.01),
        rgba=(0.45, 0.48, 0.52, 1.0),
    )
    for index, point in enumerate((start, goal)):
        spec.worldbody.add_geom(
            name=f"track_waypoint_{index}",
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            pos=point,
            size=(0.01, 0.01, 0.01),
            group=5,
            contype=0,
            conaffinity=0,
            rgba=(0.0, 0.0, 0.0, 0.0),
        )
    model = spec.compile()
    floor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    if floor_id >= 0:
        model.geom_contype[floor_id] = 0
        model.geom_conaffinity[floor_id] = 0
        model.geom_group[floor_id] = 5
        model.geom_rgba[floor_id, 3] = 0.0
    hfield_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_HFIELD, "capability_hfield"
    )
    address = int(model.hfield_adr[hfield_id])
    count = int(model.hfield_nrow[hfield_id] * model.hfield_ncol[hfield_id])
    model.hfield_data[address:address + count] = np.ascontiguousarray(
        height_xy.T / total_height, dtype=np.float32
    ).reshape(-1)

    return model, CapabilityCourse(
        kind=kind,
        obstacle_height=float(obstacle_height),
        total_height=float(total_height),
        start=start,
        goal=goal,
        top_entry_x=float(top_entry_x),
        stable_check_x=float(stable_check_x),
        horizontal_scale=float(horizontal_scale),
        height_xy=height_xy,
    )
