"""Contract checks for the maintained four-stage terrain profiles."""

from __future__ import annotations

import numpy as np

from training.terrains import (
    SruAtlasConfig,
    generate_sru_atlas,
    get_terrain_profile,
)
from training.terrains.constants import HEIGHTS, STAIRS, VERTICAL_SCALE
from training.terrains.sru_generator import StairGenerator


def main() -> None:
    stage1 = generate_sru_atlas(
        SruAtlasConfig(
            seed=101,
            surface_seed=2026,
            num_rows=1,
            num_cols=4,
            terrain_profile="stage1_flat",
        )
    )
    assert stage1.type_counts == {"maze": 0, "non_maze": 4, "stairs": 0, "pits": 0}
    assert all(np.max(tile.height_full) == 0 for tile in stage1.tiles)
    assert all(np.min(tile.height_full) == 0 for tile in stage1.tiles)

    stage2 = generate_sru_atlas(
        SruAtlasConfig(seed=102, num_rows=2, num_cols=10, terrain_profile="stage2_low_density_obstacles")
    )
    assert stage2.type_counts["stairs"] == 0
    assert stage2.type_counts["pits"] == 0
    assert all(np.min(tile.height_full) >= 0 for tile in stage2.tiles)

    stage3_profile = get_terrain_profile("stage3_reduced_height")
    stage3 = StairGenerator(
        wall_height=int(1.5 / VERTICAL_SCALE),
        vertical_scale=VERTICAL_SCALE,
        height_scale=stage3_profile.stair_height_scale,
    )
    assert stage3.step_platform_height == round(HEIGHTS.PLATFORM * 0.6)
    assert stage3.ramp_platform_height == round(HEIGHTS.RAMP_PLATFORM * 0.6)

    stage4 = generate_sru_atlas(
        SruAtlasConfig(seed=103, num_rows=2, num_cols=10, terrain_profile="stage4_full_no_pits")
    )
    assert stage4.type_counts["pits"] == 0
    assert all(np.min(tile.height_full) >= 0 for tile in stage4.tiles)

    stage5_profile = get_terrain_profile("stage5_lower_density_stairs")
    assert stage5_profile.stair_obstacle_density_scale == 0.6
    assert stage5_profile.stair_height_scale == 0.75
    assert np.isclose(
        stage5_profile.proportions[2] / sum(stage5_profile.proportions), 0.5
    )
    stage5_stairs = StairGenerator(
        wall_height=int(1.5 / VERTICAL_SCALE),
        vertical_scale=VERTICAL_SCALE,
        height_scale=stage5_profile.stair_height_scale,
    )
    assert stage5_stairs.step_platform_height == round(HEIGHTS.PLATFORM * 0.75)
    assert stage5_stairs.ramp_platform_height == round(HEIGHTS.RAMP_PLATFORM * 0.75)

    # Every 3x3 stair structure is fully occupied. The old sparse layout left
    # two ground-height notches between stair blocks that trapped wheel pairs.
    stair_heights, _, _ = stage5_stairs.generate(np.random.default_rng(0))
    cell = STAIRS.SINGLE_CELL_PIXELS
    assert np.all(stair_heights[:cell, :] > 0)
    assert np.all(stair_heights[cell : 2 * cell, :] > 0)
    assert np.all(stair_heights[2 * cell :, :] > 0)
    stage5 = generate_sru_atlas(
        SruAtlasConfig(
            seed=103,
            num_rows=2,
            num_cols=10,
            terrain_profile="stage5_lower_density_stairs",
        )
    )
    assert stage5.type_counts["pits"] == 0
    assert all(np.min(tile.height_full) >= 0 for tile in stage5.tiles)

    repeat_a = generate_sru_atlas(
        SruAtlasConfig(seed=104, num_rows=1, num_cols=4, terrain_profile="stage3_reduced_height")
    )
    repeat_b = generate_sru_atlas(
        SruAtlasConfig(seed=104, num_rows=1, num_cols=4, terrain_profile="stage3_reduced_height")
    )
    for left, right in zip(repeat_a.tiles, repeat_b.tiles):
        assert left.array_sha256() == right.array_sha256()

    print(
        "CURRICULUM_CONTRACT_OK",
        {
            "stage1_types": stage1.type_counts,
            "stage2_types": stage2.type_counts,
            "stage4_types": stage4.type_counts,
            "stage5_types": stage5.type_counts,
            "stage3_step_height_m": stage3.step_platform_height * VERTICAL_SCALE,
            "stage5_step_height_m": (
                stage5_stairs.step_platform_height / 5 * VERTICAL_SCALE
            ),
        },
        flush=True,
    )


if __name__ == "__main__":
    main()
