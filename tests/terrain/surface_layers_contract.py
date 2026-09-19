"""Contract for deterministic flat-only grass and gravel surface assignment."""

from __future__ import annotations

import numpy as np

from training.terrains import SruAtlasConfig, generate_sru_atlas


def main() -> None:
    base = generate_sru_atlas(
        SruAtlasConfig(seed=42, surface_seed=20260905, num_rows=2, num_cols=8)
    )
    changed_surface = generate_sru_atlas(
        SruAtlasConfig(
            seed=42,
            surface_seed=7,
            grass_fraction=0.10,
            gravel_fraction=0.10,
            num_rows=2,
            num_cols=8,
        )
    )
    for left, right in zip(base.tiles, changed_surface.tiles):
        assert left.terrain_type == right.terrain_type
        assert left.difficulty == right.difficulty
        for name, left_hash in left.array_sha256().items():
            assert left_hash == right.array_sha256()[name], name

    assert base.eligible_flat_area > 0.0
    assert set(p.surface_type for p in base.surface_patches) <= {"grass", "gravel"}
    for patch in base.surface_patches:
        tile = base.tiles[patch.tile_index]
        x0 = int(round((patch.center_world[0] - patch.half_size_xy[0] - tile.origin[0] + 15.0) / 0.1))
        x1 = int(round((patch.center_world[0] + patch.half_size_xy[0] - tile.origin[0] + 15.0) / 0.1))
        y0 = int(round((patch.center_world[1] - patch.half_size_xy[1] - tile.origin[1] + 15.0) / 0.1))
        y1 = int(round((patch.center_world[1] + patch.half_size_xy[1] - tile.origin[1] + 15.0) / 0.1))
        region = tile.height_full[x0:x1 + 1, y0:y1 + 1]
        assert region.size and np.all(region == 0)
        assert not np.any(tile.platform_mask[max(0, x0 - 1):x1, max(0, y0 - 1):y1])
        np.testing.assert_allclose(patch.center_world[2], patch.surface_height + 0.001)

    ratios = {
        name: area / base.eligible_flat_area
        for name, area in base.surface_counts.items()
    }
    assert 0.10 <= ratios["grass"] <= 0.40, ratios
    assert 0.10 <= ratios["gravel"] <= 0.40, ratios
    assert ratios["hard"] > ratios["grass"]
    assert ratios["hard"] > ratios["gravel"]
    print(
        "SRU_SURFACE_LAYER_CONTRACT_OK",
        f"patches={len(base.surface_patches)}",
        f"eligible_area={base.eligible_flat_area:.1f}",
        f"ratios={ratios}",
    )


if __name__ == "__main__":
    main()
