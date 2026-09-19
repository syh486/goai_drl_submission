"""Build a native MuJoCo model from an equivalent SRU terrain atlas."""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

from .atlas import SruTerrainAtlas
from .constants import VERTICAL_SCALE


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROBOT_XML = (
    REPO_ROOT
    / "src/S10_sdk_deploy/S10_description/s10_mjcf/mjcf/S10.xml"
)
TERRAIN_BODY_NAME = "main_body"
HFIELD_NAME_PREFIX = "sru_hfield_"
HFIELD_GEOM_PREFIX = "sru_terrain_"
SURFACE_PATCH_PREFIX = "sru_surface_"

SURFACE_MATERIALS = {
    "hard": {
        "friction": (1.0, 0.01, 0.01),
        "rgba": (0.42, 0.42, 0.42, 1.0),
    },
    # First-pass values are deliberately moderate. They are domain-randomized
    # surface classes, not claims of Isaac-to-MuJoCo friction equivalence.
    "grass": {
        "friction": (0.9, 0.01, 0.01),
        "rgba": (0.18, 0.48, 0.18, 1.0),
    },
    "gravel": {
        "friction": (0.72, 0.01, 0.01),
        "rgba": (0.45, 0.38, 0.28, 1.0),
    },
}


def _heightfield_encoding(height_units: np.ndarray) -> tuple[np.ndarray, float, float]:
    height_m = np.asarray(height_units, dtype=np.float64) * VERTICAL_SCALE
    z_min = float(height_m.min())
    z_max = float(height_m.max())
    z_span = z_max - z_min
    if z_span <= 1.0e-12:
        return np.zeros_like(height_m, dtype=np.float32), z_min, 0.005
    normalized = ((height_m - z_min) / z_span).astype(np.float32)
    return normalized, z_min, z_span


def build_sru_mujoco_model(
    atlas: SruTerrainAtlas,
    *,
    robot_xml: str | Path = DEFAULT_ROBOT_XML,
) -> mujoco.MjModel:
    """Compile S10 and add one native hfield geom per atlas tile."""

    robot_xml = Path(robot_xml).expanduser().resolve()
    if not robot_xml.is_file():
        raise FileNotFoundError(f"S10 robot XML does not exist: {robot_xml}")
    spec = mujoco.MjSpec.from_file(str(robot_xml))
    terrain_body = spec.worldbody.add_body(name=TERRAIN_BODY_NAME)
    encodings: dict[str, np.ndarray] = {}

    for tile in atlas.tiles:
        hfield_name = f"{HFIELD_NAME_PREFIX}{tile.index:03d}"
        geom_name = f"{HFIELD_GEOM_PREFIX}{tile.index:03d}_{tile.terrain_type}"
        normalized, z_min, z_span = _heightfield_encoding(tile.height_full)
        nrow, ncol = normalized.T.shape
        spec.add_hfield(
            name=hfield_name,
            nrow=nrow,
            ncol=ncol,
            size=(
                atlas.config.tile_size[0] * 0.5,
                atlas.config.tile_size[1] * 0.5,
                z_span,
                max(0.1, abs(z_min) + 0.1),
            ),
            # MjSpec requires allocated user data. Exact normalized values are
            # installed after compilation to avoid compiler normalization.
            userdata=np.zeros(nrow * ncol, dtype=np.float32),
        )
        terrain_body.add_geom(
            name=geom_name,
            type=mujoco.mjtGeom.mjGEOM_HFIELD,
            hfieldname=hfield_name,
            pos=(tile.origin[0], tile.origin[1], z_min),
            group=0,
            condim=3,
            contype=1,
            conaffinity=1,
            priority=1,
            friction=(1.0, 0.01, 0.01),
        )
        # MuJoCo's rows are world y and columns are world x.
        encodings[hfield_name] = np.ascontiguousarray(normalized.T)

    _add_atlas_border(terrain_body, atlas)
    _add_surface_patches(terrain_body, atlas)
    model = spec.compile()

    floor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    if floor_id >= 0:
        model.geom_contype[floor_id] = 0
        model.geom_conaffinity[floor_id] = 0
        model.geom_group[floor_id] = 5
        model.geom_rgba[floor_id, 3] = 0.0

    for name, normalized in encodings.items():
        hfield_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_HFIELD, name)
        if hfield_id < 0:
            raise RuntimeError(f"compiled model lost hfield {name}")
        address = int(model.hfield_adr[hfield_id])
        count = int(model.hfield_nrow[hfield_id] * model.hfield_ncol[hfield_id])
        model.hfield_data[address:address + count] = normalized.reshape(-1)

    return model


def _add_atlas_border(terrain_body: mujoco.MjsBody, atlas: SruTerrainAtlas) -> None:
    cfg = atlas.config
    inner_x = cfg.num_rows * cfg.tile_size[0]
    inner_y = cfg.num_cols * cfg.tile_size[1]
    border = cfg.border_width
    half_height = cfg.border_height * 0.5
    center_x = inner_x * 0.5
    center_y = inner_y * 0.5
    common = {
        "type": mujoco.mjtGeom.mjGEOM_BOX,
        "group": 0,
        "condim": 3,
        "contype": 1,
        "conaffinity": 1,
        "priority": 1,
        "friction": (1.0, 0.01, 0.01),
    }
    terrain_body.add_geom(
        name="sru_border_y_min",
        pos=(center_x, -border * 0.5, -half_height),
        size=((inner_x + 2 * border) * 0.5, border * 0.5, half_height),
        **common,
    )

    terrain_body.add_geom(
        name="sru_border_y_max",
        pos=(center_x, inner_y + border * 0.5, -half_height),
        size=((inner_x + 2 * border) * 0.5, border * 0.5, half_height),
        **common,
    )
    terrain_body.add_geom(
        name="sru_border_x_min",
        pos=(-border * 0.5, center_y, -half_height),
        size=(border * 0.5, inner_y * 0.5, half_height),
        **common,
    )
    terrain_body.add_geom(
        name="sru_border_x_max",
        pos=(inner_x + border * 0.5, center_y, -half_height),
        size=(border * 0.5, inner_y * 0.5, half_height),
        **common,
    )


def _add_surface_patches(terrain_body: mujoco.MjsBody, atlas: SruTerrainAtlas) -> None:
    """Add thin contact layers only over eligible flat-ground rectangles."""

    thickness = atlas.config.surface_layer_thickness
    for patch_index, patch in enumerate(atlas.surface_patches):
        material = SURFACE_MATERIALS[patch.surface_type]
        terrain_body.add_geom(
            name=f"{SURFACE_PATCH_PREFIX}{patch_index:05d}_{patch.surface_type}",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=tuple(float(value) for value in patch.center_world),
            size=(
                float(patch.half_size_xy[0]),
                float(patch.half_size_xy[1]),
                thickness * 0.5,
            ),
            group=0,
            condim=3,
            contype=1,
            conaffinity=1,
            priority=2,
            friction=material["friction"],
            rgba=material["rgba"],
        )
