"""Random-goal sampling with the same support as the IsaacLab SRU task."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .atlas import SruTerrainAtlas, mask_index_to_local_xy
from .constants import VERTICAL_SCALE


@dataclass(frozen=True)
class RandomGoalSample:
    tile_index: int
    spawn_world: np.ndarray
    goal_world: np.ndarray


class SruPositionSampler:
    """Precompute goal/spawn tables and sample independently within each tile."""

    def __init__(self, atlas: SruTerrainAtlas, *, platform_repeat_count: int = 10):
        self.atlas = atlas
        self.goal_indices: list[np.ndarray] = []
        self.spawn_indices: list[np.ndarray] = []
        for tile in atlas.tiles:
            valid = np.argwhere(tile.valid_mask)
            platform = tile.platform_mask[valid[:, 0], valid[:, 1]]
            platform_valid = valid[platform]
            if len(platform_valid):
                valid = np.concatenate(
                    (valid, np.repeat(platform_valid, platform_repeat_count, axis=0)),
                    axis=0,
                )
            spawn = np.argwhere(tile.spawn_mask)
            if not len(valid) or not len(spawn):
                raise RuntimeError(f"terrain tile {tile.index} has empty sampling support")
            self.goal_indices.append(valid)
            self.spawn_indices.append(spawn)

    def sample(self, tile_index: int, rng: np.random.Generator) -> RandomGoalSample:
        tile = self.atlas.tiles[int(tile_index)]
        goal_ij = self.goal_indices[tile.index][
            int(rng.integers(0, len(self.goal_indices[tile.index])))
        ]
        spawn_ij = self.spawn_indices[tile.index][
            int(rng.integers(0, len(self.spawn_indices[tile.index])))
        ]
        goal_x, goal_y = mask_index_to_local_xy(goal_ij[0], goal_ij[1])
        spawn_x, spawn_y = mask_index_to_local_xy(spawn_ij[0], spawn_ij[1])
        goal_surface = float(tile.height_inner[tuple(goal_ij)] * VERTICAL_SCALE)
        spawn_surface = float(tile.height_inner[tuple(spawn_ij)] * VERTICAL_SCALE)
        goal_height_offset = float(rng.random() * 0.6 + 0.2)
        return RandomGoalSample(
            tile_index=tile.index,
            spawn_world=np.asarray(
                (
                    tile.origin[0] + spawn_x,
                    tile.origin[1] + spawn_y,
                    spawn_surface,
                ),
                dtype=np.float64,
            ),
            goal_world=np.asarray(
                (
                    tile.origin[0] + goal_x,
                    tile.origin[1] + goal_y,
                    goal_surface + goal_height_offset,
                ),
                dtype=np.float64,
            ),
        )


def assign_terrain_tiles(
    num_envs: int,
    atlas: SruTerrainAtlas,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Match IsaacLab's fixed terrain type and random initial level assignment."""

    if num_envs < 1:
        raise ValueError("num_envs must be positive")
    levels = rng.integers(0, atlas.config.num_rows, size=num_envs, dtype=np.int64)
    types = np.floor(
        np.arange(num_envs, dtype=np.float64)
        / (float(num_envs) / atlas.config.num_cols)
    ).astype(np.int64)
    types = np.clip(types, 0, atlas.config.num_cols - 1)
    indices = levels * atlas.config.num_cols + types
    return levels, types, indices
