"""Pure coverage checks for the random-terrain LiDAR sampler."""

from __future__ import annotations

import numpy as np

from training.lidar.collect_random_terrain import _next_tile_batch


def test_two_batches_cover_180_tiles_with_128_envs() -> None:
    rng = np.random.default_rng(20260905)
    stream = rng.permutation(180).tolist()
    cursor = 0
    first, cursor, stream = _next_tile_batch(
        stream, cursor, tile_count=180, num_envs=128, rng=rng
    )
    second, cursor, stream = _next_tile_batch(
        stream, cursor, tile_count=180, num_envs=128, rng=rng
    )
    assert cursor == 256
    assert len(set(first.tolist()) | set(second.tolist())) == 180


if __name__ == "__main__":
    test_two_batches_cover_180_tiles_with_128_envs()
    print("lidar_sampling_contract: PASS")
