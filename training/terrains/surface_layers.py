"""Deterministic grass/gravel patches restricted to flat ground cells."""

from __future__ import annotations

from typing import Iterable

import numpy as np

from .atlas import SurfacePatch, SruTerrainTile
from .constants import HEIGHTS


def _merge_material_blocks(materials: np.ndarray) -> list[tuple[int, int, int, int, int]]:
    """Merge adjacent equal non-hard blocks into rectangles."""

    visited = np.zeros(materials.shape, dtype=bool)
    rectangles: list[tuple[int, int, int, int, int]] = []
    rows, cols = materials.shape
    for row in range(rows):
        for col in range(cols):
            material = int(materials[row, col])
            if material == 0 or visited[row, col]:
                continue
            width = 1
            while (
                col + width < cols
                and not visited[row, col + width]
                and int(materials[row, col + width]) == material
            ):
                width += 1
            height = 1
            while row + height < rows:
                if visited[row + height, col:col + width].any():
                    break
                if not np.all(materials[row + height, col:col + width] == material):
                    break
                height += 1
            visited[row:row + height, col:col + width] = True
            rectangles.append((row, row + height, col, col + width, material))
    return rectangles


def _tile_ground_blocks(
    tile: SruTerrainTile,
    *,
    block_cells: int,
    rng: np.random.Generator,
    grass_fraction: float,
    gravel_fraction: float,
) -> tuple[list[tuple[int, int, int, int, int]], int]:
    """Return block rectangles whose four corners are flat ground."""

    heights = np.asarray(tile.height_full, dtype=np.int16)
    valid = np.pad(np.asarray(tile.valid_mask, dtype=bool), 1, constant_values=False)
    platform = np.pad(
        np.asarray(tile.platform_mask, dtype=bool), 1, constant_values=False
    )
    # The first height axis is SRU x, the second is SRU y. A cell is eligible
    # only when all four corners are exactly ground level and valid.
    flat_ground = (
        (heights[:-1, :-1] == HEIGHTS.GROUND)
        & (heights[1:, :-1] == HEIGHTS.GROUND)
        & (heights[:-1, 1:] == HEIGHTS.GROUND)
        & (heights[1:, 1:] == HEIGHTS.GROUND)
        & valid[:-1, :-1]
        & valid[1:, :-1]
        & valid[:-1, 1:]
        & valid[1:, 1:]
        & ~platform[:-1, :-1]
        & ~platform[1:, :-1]
        & ~platform[:-1, 1:]
        & ~platform[1:, 1:]
    )
    block_rows = (flat_ground.shape[0] + block_cells - 1) // block_cells
    block_cols = (flat_ground.shape[1] + block_cells - 1) // block_cells
    materials = np.zeros((block_rows, block_cols), dtype=np.int8)
    eligible_area_cells = 0
    for row in range(block_rows):
        r0 = row * block_cells
        r1 = min(r0 + block_cells, flat_ground.shape[0])
        for col in range(block_cols):
            c0 = col * block_cells
            c1 = min(c0 + block_cells, flat_ground.shape[1])
            if not np.all(flat_ground[r0:r1, c0:c1]):
                continue
            eligible_area_cells += (r1 - r0) * (c1 - c0)
            draw = float(rng.random())
            if draw < grass_fraction:
                materials[row, col] = 1
            elif draw < grass_fraction + gravel_fraction:
                materials[row, col] = 2
    return _merge_material_blocks(materials), eligible_area_cells


def generate_surface_patches(
    tiles: Iterable[SruTerrainTile],
    *,
    tile_size: tuple[float, float],
    horizontal_scale: float,
    vertical_scale: float,
    seed: int | None,
    grass_fraction: float,
    gravel_fraction: float,
    block_cells: int,
    layer_thickness: float,
) -> tuple[tuple[SurfacePatch, ...], float]:
    """Assign approximately the requested fractions of eligible flat blocks."""

    rng = np.random.default_rng(seed)
    patches: list[SurfacePatch] = []
    eligible_area = 0.0
    for tile in tuple(tiles):
        rectangles, eligible_cells = _tile_ground_blocks(
            tile,
            block_cells=block_cells,
            rng=rng,
            grass_fraction=grass_fraction,
            gravel_fraction=gravel_fraction,
        )
        eligible_area += eligible_cells * horizontal_scale * horizontal_scale
        for block_row0, block_row1, block_col0, block_col1, material in rectangles:
            row0 = block_row0 * block_cells
            row1 = min(block_row1 * block_cells, tile.height_full.shape[0] - 1)
            col0 = block_col0 * block_cells
            col1 = min(block_col1 * block_cells, tile.height_full.shape[1] - 1)
            # Height array axes are SRU x/y. MuJoCo later transposes only the
            # hfield storage; patch world coordinates stay in SRU x/y order.
            x0 = -tile_size[0] * 0.5 + row0 * horizontal_scale
            x1 = -tile_size[0] * 0.5 + row1 * horizontal_scale
            y0 = -tile_size[1] * 0.5 + col0 * horizontal_scale
            y1 = -tile_size[1] * 0.5 + col1 * horizontal_scale
            height_units = int(tile.height_full[row0, col0])
            patches.append(
                SurfacePatch(
                    tile_index=tile.index,
                    surface_type=("hard", "grass", "gravel")[material],
                    center_world=np.asarray(
                        (
                            tile.origin[0] + 0.5 * (x0 + x1),
                            tile.origin[1] + 0.5 * (y0 + y1),
                            height_units * vertical_scale + 0.5 * layer_thickness,
                        ),
                        dtype=np.float64,
                    ),
                    half_size_xy=np.asarray(
                        (0.5 * (x1 - x0), 0.5 * (y1 - y0)), dtype=np.float64
                    ),
                    surface_height=height_units * vertical_scale,
                )
            )
    return tuple(patches), eligible_area
