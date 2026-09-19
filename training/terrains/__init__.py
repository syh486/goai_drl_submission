"""SRU terrain generation and MuJoCo integration."""

from .atlas import (
    SruAtlasConfig,
    SruTerrainAtlas,
    SruTerrainTile,
    SurfacePatch,
    generate_sru_atlas,
    mask_index_to_local_xy,
)
from .curriculum import (
    LEGACY_FULL,
    PROFILES,
    STAGE1_FLAT,
    STAGE2_LOW_DENSITY_OBSTACLES,
    STAGE3_REDUCED_HEIGHT,
    STAGE4_FULL_NO_PITS,
    STAGE5_LOWER_DENSITY_STAIRS,
    TerrainCurriculumProfile,
    get_terrain_profile,
)
from .mujoco_hfield import build_sru_mujoco_model
from .sampling import RandomGoalSample, SruPositionSampler, assign_terrain_tiles

__all__ = [
    "SruAtlasConfig",
    "SruTerrainAtlas",
    "SruTerrainTile",
    "SurfacePatch",
    "generate_sru_atlas",
    "mask_index_to_local_xy",
    "build_sru_mujoco_model",
    "RandomGoalSample",
    "SruPositionSampler",
    "assign_terrain_tiles",
    "LEGACY_FULL",
    "PROFILES",
    "STAGE1_FLAT",
    "STAGE2_LOW_DENSITY_OBSTACLES",
    "STAGE3_REDUCED_HEIGHT",
    "STAGE4_FULL_NO_PITS",
    "STAGE5_LOWER_DENSITY_STAIRS",
    "TerrainCurriculumProfile",
    "get_terrain_profile",
]
