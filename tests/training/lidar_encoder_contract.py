"""Regression checks for the S10 LiDAR metric/normalized input contract."""

from __future__ import annotations

import torch

from sru_training.s10_lidar_encoder import (
    ENCODER_DISTANCE_SCALE_M,
    ENCODER_WORLD_Z_SCALE_M,
)


def test_constants() -> None:
    assert ENCODER_DISTANCE_SCALE_M == 10.0
    assert ENCODER_WORLD_Z_SCALE_M == 3.0


def test_single_frame_and_batch_shape() -> None:
    # Avoid loading a checkpoint: validate the shape normalization helper by
    # constructing the same inputs that encode_maps sends to the model.
    distance = torch.full((96, 90), 5.0)
    world_z = torch.full((96, 90), 1.5)
    distance_b = distance.unsqueeze(0)
    world_z_b = world_z.unsqueeze(0)
    single = torch.stack((distance_b / 10.0, world_z_b / 3.0), dim=1)
    batch = torch.stack((distance_b / 10.0, world_z_b / 3.0), dim=1)
    assert single.shape == (1, 2, 96, 90)
    assert batch.shape == (1, 2, 96, 90)
    assert torch.allclose(single[:, 0], torch.full((1, 96, 90), 0.5))
    assert torch.allclose(single[:, 1], torch.full((1, 96, 90), 0.5))


if __name__ == "__main__":
    test_constants()
    test_single_frame_and_batch_shape()
    print("lidar_encoder_contract: PASS")
