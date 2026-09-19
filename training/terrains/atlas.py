# Copyright (c) 2022-2025, Fan Yang and Per Frivik, ETH Zurich.
# All rights reserved.
#
# SPDX-License-Identifier: MIT

"""Deterministic atlas assembly for the migrated SRU terrain generator."""

from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256

import numpy as np

from .constants import HORIZONTAL_SCALE, VERTICAL_SCALE
from .curriculum import get_terrain_profile
from .sru_generator import SruSubTerrainConfig, maze_terrain


TERRAIN_NAMES = ("maze", "non_maze", "stairs", "pits")
TERRAIN_PROPORTIONS = np.asarray((0.3, 0.2, 0.3, 0.2), dtype=np.float64)
SURFACE_TYPES = ("hard", "grass", "gravel")


@dataclass(frozen=True)
class SruAtlasConfig:
    """Configuration matching ``MAZE_TERRAIN_CFG`` in the MX project."""

    seed: int | None = None
    num_rows: int = 6
    num_cols: int = 30
    tile_size: tuple[float, float] = (30.0, 30.0)
    border_width: float = 30.0
    border_height: float = 1.0
    horizontal_scale: float = HORIZONTAL_SCALE
    vertical_scale: float = VERTICAL_SCALE
    difficulty_range: tuple[float, float] = (0.5, 1.0)
    proportions: tuple[float, float, float, float] = (0.3, 0.2, 0.3, 0.2)
    terrain_profile: str = "legacy_full"
    surface_seed: int | None = None
    grass_fraction: float = 0.25
    gravel_fraction: float = 0.25
    surface_block_cells: int = 20
    surface_layer_thickness: float = 0.002

    def validate(self) -> None:
        if self.num_rows < 1 or self.num_cols < 1:
            raise ValueError("atlas row and column counts must be positive")
        if self.tile_size != (30.0, 30.0):
            raise ValueError("the equivalent SRU generator requires 30 m tiles")
        if self.horizontal_scale != HORIZONTAL_SCALE:
            raise ValueError("the equivalent SRU generator requires 0.1 m horizontal scale")
        if self.vertical_scale != VERTICAL_SCALE:
            raise ValueError("the equivalent SRU generator requires 0.005 m vertical scale")
        get_terrain_profile(self.terrain_profile)
        proportions = np.asarray(self.proportions, dtype=np.float64)
        if proportions.shape != (4,) or np.any(proportions < 0.0) or proportions.sum() <= 0.0:
            raise ValueError("terrain proportions must contain four non-negative values")
        if not 0.0 <= self.grass_fraction <= 1.0:
            raise ValueError("grass_fraction must be in [0, 1]")
        if not 0.0 <= self.gravel_fraction <= 1.0:
            raise ValueError("gravel_fraction must be in [0, 1]")
        if self.grass_fraction + self.gravel_fraction > 1.0:
            raise ValueError("grass_fraction + gravel_fraction must be <= 1")
        if self.surface_block_cells < 1:
            raise ValueError("surface_block_cells must be positive")
        if self.surface_layer_thickness <= 0.0 or self.surface_layer_thickness > 0.01:
            raise ValueError("surface_layer_thickness must be in (0, 0.01]")


@dataclass(frozen=True)
class SruTerrainTile:
    """All geometry and sampling data for one atlas tile."""

    index: int
    row: int
    col: int
    terrain_type: str
    difficulty: float
    origin: np.ndarray
    height_inner: np.ndarray
    height_full: np.ndarray
    valid_mask: np.ndarray
    platform_mask: np.ndarray
    spawn_mask: np.ndarray

    def array_sha256(self) -> dict[str, str]:
        arrays = {
            "height_inner": self.height_inner,
            "height_full": self.height_full,
            "valid_mask": self.valid_mask,
            "platform_mask": self.platform_mask,
            "spawn_mask": self.spawn_mask,
        }
        return {
            name: sha256(np.ascontiguousarray(value).view(np.uint8)).hexdigest()
            for name, value in arrays.items()
        }


@dataclass(frozen=True)
class SurfacePatch:
    """A thin material patch placed on a flat, ground-level hfield region."""

    tile_index: int
    surface_type: str
    center_world: np.ndarray
    half_size_xy: np.ndarray
    surface_height: float

    @property
    def area(self) -> float:
        return float(4.0 * self.half_size_xy[0] * self.half_size_xy[1])


@dataclass(frozen=True)
class SruTerrainAtlas:
    config: SruAtlasConfig
    tiles: tuple[SruTerrainTile, ...]
    terrain_origins: np.ndarray
    surface_patches: tuple[SurfacePatch, ...] = ()
    eligible_flat_area: float = 0.0
    type_counts: dict[str, int] = field(init=False)
    surface_counts: dict[str, float] = field(init=False)

    def __post_init__(self) -> None:
        counts = {name: 0 for name in TERRAIN_NAMES}
        for tile in self.tiles:
            counts[tile.terrain_type] += 1
        object.__setattr__(self, "type_counts", counts)
        surface_counts = {name: 0.0 for name in SURFACE_TYPES}
        for patch in self.surface_patches:
            if patch.surface_type not in surface_counts:
                raise ValueError(f"unknown surface type {patch.surface_type!r}")
            surface_counts[patch.surface_type] += patch.area
        surface_counts["hard"] = max(
            0.0, float(self.eligible_flat_area) - surface_counts["grass"] - surface_counts["gravel"]
        )
        object.__setattr__(self, "surface_counts", surface_counts)

    def tile(self, row: int, col: int) -> SruTerrainTile:
        return self.tiles[row * self.config.num_cols + col]


def _subterrain_config(
    terrain_type: str,
    child_rng: np.random.Generator,
    *,
    obstacle_density_scale: float = 1.0,
    stair_obstacle_density_scale: float = 1.0,
    stair_height_scale: float = 1.0,
    flat_only: bool = False,
    pit_probability: float = 0.15,
) -> SruSubTerrainConfig:
    common = {
        # Preserve the wrapper's arithmetic instead of spelling 29.9. The two
        # floats round differently before ``int(size / scale)``.
        "size": (299 * HORIZONTAL_SCALE, 299 * HORIZONTAL_SCALE),
        "horizontal_scale": HORIZONTAL_SCALE,
        "vertical_scale": VERTICAL_SCALE,
        "grid_size": (15, 15),
        "cell_size": 2.0,
        "add_noise_to_flat": False,
        "add_goal": True,
        "rng": child_rng,
    }
    if terrain_type == "maze":
        return SruSubTerrainConfig(
            **common,
            randomize_wall=True,
            random_wall_ratio=0.5,
            add_stairs_to_maze=True,
            obstacle_density_scale=obstacle_density_scale,
            stair_obstacle_density_scale=stair_obstacle_density_scale,
            stair_height_scale=stair_height_scale,
            flat_only=flat_only,
            pit_probability=pit_probability,
        )
    if terrain_type == "non_maze":
        return SruSubTerrainConfig(
            **common,
            randomize_wall=True,
            random_wall_ratio=1.0,
            non_maze_terrain=True,
            obstacle_density_scale=obstacle_density_scale,
            stair_obstacle_density_scale=stair_obstacle_density_scale,
            stair_height_scale=stair_height_scale,
            flat_only=flat_only,
            pit_probability=pit_probability,
        )
    if terrain_type == "stairs":
        return SruSubTerrainConfig(
            **common,
            randomize_wall=False,
            random_wall_ratio=1.0,
            stairs=True,
            obstacle_density_scale=obstacle_density_scale,
            stair_obstacle_density_scale=stair_obstacle_density_scale,
            stair_height_scale=stair_height_scale,
            flat_only=flat_only,
            pit_probability=pit_probability,
        )
    if terrain_type == "pits":
        return SruSubTerrainConfig(
            **common,
            randomize_wall=True,
            random_wall_ratio=1.0,
            non_maze_terrain=True,
            dynamic_obstacles=True,
            obstacle_density_scale=obstacle_density_scale,
            stair_obstacle_density_scale=stair_obstacle_density_scale,
            stair_height_scale=stair_height_scale,
            flat_only=flat_only,
            pit_probability=pit_probability,
        )
    raise ValueError(f"unknown SRU terrain type: {terrain_type}")


def generate_sru_atlas(config: SruAtlasConfig | None = None) -> SruTerrainAtlas:
    """Generate an atlas with the same RNG call order as patched IsaacLab."""

    config = config or SruAtlasConfig()
    config.validate()
    profile = get_terrain_profile(config.terrain_profile)
    atlas_rng = np.random.default_rng(config.seed)
    reproducible = config.seed is not None
    # The legacy path preserves the exact old caller-controlled values. Named
    # profiles are self-contained and cannot accidentally inherit old pits or
    # difficulty settings from a copied config.
    proportions_source = (
        config.proportions
        if profile.name == "legacy_full"
        else profile.proportions
    )
    difficulty_range = (
        config.difficulty_range
        if profile.name == "legacy_full"
        else profile.difficulty_range
    )
    proportions = np.asarray(proportions_source, dtype=np.float64)
    proportions /= proportions.sum()

    expected_full_shape = (
        int(config.tile_size[0] / config.horizontal_scale) + 1,
        int(config.tile_size[1] / config.horizontal_scale) + 1,
    )
    border_pixels = 1
    expected_inner_shape = (
        expected_full_shape[0] - 2 * border_pixels,
        expected_full_shape[1] - 2 * border_pixels,
    )
    tiles: list[SruTerrainTile] = []
    origins = np.zeros((config.num_rows, config.num_cols, 3), dtype=np.float64)

    for index in range(config.num_rows * config.num_cols):
        row, col = np.unravel_index(index, (config.num_rows, config.num_cols))
        type_index = int(atlas_rng.choice(len(proportions), p=proportions))
        difficulty = float(atlas_rng.uniform(*difficulty_range))
        child_rng = atlas_rng.spawn(1)[0] if reproducible else np.random.default_rng()
        sub_cfg = _subterrain_config(
            TERRAIN_NAMES[type_index],
            child_rng,
            obstacle_density_scale=profile.obstacle_density_scale,
            stair_obstacle_density_scale=profile.stair_obstacle_density_scale,
            stair_height_scale=profile.stair_height_scale,
            flat_only=profile.flat_only,
            pit_probability=profile.pit_probability,
        )
        height_inner = maze_terrain(difficulty, sub_cfg)
        if height_inner.shape != expected_inner_shape:
            raise RuntimeError(
                f"SRU generator produced {height_inner.shape}, expected {expected_inner_shape}"
            )

        height_full = np.zeros(expected_full_shape, dtype=np.int16)
        height_full[border_pixels:-border_pixels, border_pixels:-border_pixels] = height_inner
        center = np.asarray(
            ((row + 0.5) * config.tile_size[0], (col + 0.5) * config.tile_size[1]),
            dtype=np.float64,
        )
        origin_z = float(
            np.max(height_full[140:160, 140:160]) * config.vertical_scale
        )
        origin = np.asarray((center[0], center[1], origin_z), dtype=np.float64)
        origins[row, col] = origin
        tiles.append(
            SruTerrainTile(
                index=index,
                row=int(row),
                col=int(col),
                terrain_type=TERRAIN_NAMES[type_index],
                difficulty=difficulty,
                origin=origin,
                height_inner=np.asarray(sub_cfg.height_field_visual[0], dtype=np.int16),
                height_full=height_full,
                valid_mask=np.asarray(sub_cfg.height_field_valid_mask[0], dtype=bool),
                platform_mask=np.asarray(sub_cfg.height_field_platform_mask[0], dtype=bool),
                spawn_mask=np.asarray(sub_cfg.height_field_spawn_mask[0], dtype=bool),
            )
        )

    from .surface_layers import generate_surface_patches

    surface_patches, eligible_flat_area = generate_surface_patches(
        tuple(tiles),
        tile_size=config.tile_size,
        horizontal_scale=config.horizontal_scale,
        vertical_scale=config.vertical_scale,
        seed=config.surface_seed if config.surface_seed is not None else config.seed,
        grass_fraction=config.grass_fraction,
        gravel_fraction=config.gravel_fraction,
        block_cells=config.surface_block_cells,
        layer_thickness=config.surface_layer_thickness,
    )
    return SruTerrainAtlas(
        config=config,
        tiles=tuple(tiles),
        terrain_origins=origins,
        surface_patches=surface_patches,
        eligible_flat_area=eligible_flat_area,
    )


def mask_index_to_local_xy(i: np.ndarray | int, j: np.ndarray | int) -> tuple[np.ndarray, np.ndarray]:
    """Map an inner 299-by-299 mask index to the centered tile frame."""

    x = (np.asarray(i, dtype=np.float64) + 1.0) * HORIZONTAL_SCALE - 15.0
    y = (np.asarray(j, dtype=np.float64) + 1.0) * HORIZONTAL_SCALE - 15.0
    return x, y
