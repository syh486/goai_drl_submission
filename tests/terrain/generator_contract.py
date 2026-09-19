"""Deterministic contract checks for the pure NumPy SRU terrain port."""

from __future__ import annotations

from hashlib import sha256

import numpy as np

from training.terrains import SruAtlasConfig, generate_sru_atlas, mask_index_to_local_xy


EXPECTED_HASH = "9324519639c520eb6320612b2a22534d5fb64edee74e7dd761773e505e9f7901"


def _atlas_hash() -> tuple[str, object]:
    atlas = generate_sru_atlas(SruAtlasConfig(seed=42, num_rows=2, num_cols=4))
    digest = sha256()
    for tile in atlas.tiles:
        digest.update(tile.terrain_type.encode("ascii"))
        digest.update(np.float64(tile.difficulty).tobytes())
        for value in (
            tile.height_inner,
            tile.height_full,
            tile.valid_mask,
            tile.platform_mask,
            tile.spawn_mask,
        ):
            digest.update(np.ascontiguousarray(value).view(np.uint8))
    return digest.hexdigest(), atlas


def main() -> None:
    actual, atlas = _atlas_hash()
    assert actual == EXPECTED_HASH, (actual, EXPECTED_HASH)
    assert atlas.type_counts == {"maze": 2, "non_maze": 2, "stairs": 3, "pits": 1}
    for tile in atlas.tiles:
        assert tile.height_inner.shape == (299, 299)
        assert tile.height_full.shape == (301, 301)
        assert np.array_equal(tile.height_full[1:-1, 1:-1], tile.height_inner)
        assert not tile.height_full[0].any()
        assert not tile.height_full[-1].any()
        assert tile.valid_mask.shape == tile.spawn_mask.shape == (299, 299)
        assert np.all(~tile.spawn_mask | tile.valid_mask)
        assert np.array_equal(
            tile.origin[:2],
            ((tile.row + 0.5) * 30.0, (tile.col + 0.5) * 30.0),
        )
    x, y = mask_index_to_local_xy(np.asarray((0, 149, 298)), np.asarray((0, 149, 298)))
    np.testing.assert_allclose(x, (-14.9, 0.0, 14.9), atol=1.0e-12)
    np.testing.assert_allclose(y, (-14.9, 0.0, 14.9), atol=1.0e-12)
    print("SRU_TERRAIN_GENERATOR_CONTRACT_OK", actual)


if __name__ == "__main__":
    main()
