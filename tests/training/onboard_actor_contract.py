"""The simulated deployment actor must not receive MuJoCo proprioception or world-z."""

from types import SimpleNamespace

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from sru_training.s10_lidar_encoder import build_sensor_frame_directions
from sru_training.s10_mujoco_env import S10RawState
from training.evaluation.random_terrain_onboard import _prepare_onboard_state


class _Encoder:
    def __init__(self):
        self.front_z = None

    def encode_maps(self, front_d, rear_d, front_z, rear_z):
        self.front_z = front_z.clone()
        assert front_d.shape == rear_d.shape == front_z.shape == rear_z.shape == (1, 96, 90)
        return torch.full((1, 64, 5, 8), 8.0)


def main() -> None:
    initial = Rotation.from_euler("z", 90, degrees=True).as_matrix()
    current = Rotation.from_euler("xyz", (15, -10, 180), degrees=True).as_matrix()
    odometry = SimpleNamespace(
        initial_rotation_wb=initial,
        rotation_wb=current,
        position_w=np.array((2.0, 3.0, 0.8)),
        filter=SimpleNamespace(velocity=np.array((1.0, 0.1, 0.2))),
    )
    odometry.goal_body = lambda target: np.array((0.1, 0.2, 0.3, 0.4))
    scans = (np.full((96, 900), 10.0, np.float32), np.full((96, 900), 10.0, np.float32))
    scans[0][10, 200] = 2.0
    scans[1][20, 300] = 3.0
    truth_state = S10RawState(
        base_lin_vel=torch.full((1, 3), 100.0),
        base_ang_vel=torch.full((1, 3), 100.0),
        projected_gravity=torch.full((1, 3), 100.0),
        last_action=torch.tensor([[0.2, -0.3]]),
        goal_body=torch.full((1, 4), 100.0),
        lidar_latent=torch.full((1, 64, 5, 8), 100.0),
    )
    encoder = _Encoder()
    state = _prepare_onboard_state(
        truth_state, odometry, np.array((3.0, 4.0, 1.0)),
        np.array((0.3, -0.1, 0.5)), scans,
        build_sensor_frame_directions(900), encoder,
    )
    np.testing.assert_allclose(
        state.base_lin_vel.numpy()[0],
        current.T @ initial @ odometry.filter.velocity,
        atol=1e-6,
    )
    np.testing.assert_allclose(state.base_ang_vel.numpy()[0], (0.3, -0.1, 0.5))
    np.testing.assert_allclose(
        state.projected_gravity.numpy()[0], current.T @ (0.0, 0.0, -1.0), atol=1e-6
    )
    np.testing.assert_allclose(state.goal_body.numpy()[0], (0.1, 0.2, 0.3, 0.4))
    np.testing.assert_allclose(state.last_action.numpy()[0], (0.2, -0.3))
    assert (state.lidar_latent == 8.0).all()
    assert encoder.front_z is not None
    z = encoder.front_z[0, 10, 20].item()
    assert abs(z - 0.8) > 0.01 and np.isfinite(z)
    truth_state.base_lin_vel.fill_(-100.0)
    truth_state.lidar_latent.fill_(-100.0)
    assert torch.all(state.base_lin_vel > -100.0) and torch.all(state.lidar_latent == 8.0)
    print("ONBOARD_ACTOR_CONTRACT_OK")


if __name__ == "__main__":
    main()
