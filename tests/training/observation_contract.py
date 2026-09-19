"""Regression checks for the positional SRU observation contract."""

from __future__ import annotations

import torch

from sru_training.s10_mujoco_env import (
    S10MujocoVecEnv,
    S10ObservationAdapter,
    S10RawState,
)
from sru_training.s10_policy_config import S10ObservationSpec


class _ClockBackend:
    num_envs = 8
    max_episode_length = 300

    def __init__(self, state: S10RawState):
        self.state = state
        self.lengths = None

    def reset(self):
        return self.state

    def set_initial_episode_lengths(self, lengths):
        self.lengths = lengths.copy()
        self.state.time_normalized = torch.as_tensor(
            lengths[:, None] / self.max_episode_length, dtype=torch.float32
        )
        return self.state

    def close(self):
        pass


def main() -> None:
    spec = S10ObservationSpec()
    adapter = S10ObservationAdapter(spec, "cpu")
    state = S10RawState(
        base_lin_vel=torch.full((1, 3), 1.0),
        base_ang_vel=torch.full((1, 3), 2.0),
        projected_gravity=torch.full((1, 3), 3.0),
        last_action=torch.full((1, 2), 4.0),
        goal_body=torch.full((1, 4), 5.0),
        lidar_latent=torch.full((1, *spec.latent_shape), 7.0),
        height_latent=torch.full((1, *spec.height_shape), 6.0),
        time_normalized=torch.full((1, 1), 0.25),
    )

    actor, critic, _ = adapter.encode(state)
    proprio_end = spec.proprio_dim
    time_end = proprio_end + 1
    height_end = time_end + spec.height_dim

    assert actor.shape == (1, spec.actor_obs_dim)
    assert critic.shape == (1, spec.critic_obs_dim)
    assert torch.equal(actor[:, :proprio_end], critic[:, :proprio_end])
    assert torch.all(critic[:, proprio_end:time_end] == 0.25)
    assert torch.all(critic[:, time_end:height_end] == 6.0)
    assert torch.all(critic[:, height_end:] == 7.0)
    assert torch.all(actor[:, proprio_end:] == 7.0)

    clock_state = S10RawState(
        base_lin_vel=torch.zeros(8, 3),
        base_ang_vel=torch.zeros(8, 3),
        projected_gravity=torch.zeros(8, 3),
        last_action=torch.zeros(8, 2),
        goal_body=torch.zeros(8, 4),
        lidar_latent=torch.zeros(8, *spec.latent_shape),
    )
    clock_backend = _ClockBackend(clock_state)
    env = S10MujocoVecEnv(clock_backend, device="cpu")
    env.randomize_episode_lengths()
    assert clock_backend.lengths is not None
    assert torch.equal(
        env.episode_length_buf,
        torch.as_tensor(clock_backend.lengths, dtype=torch.long),
    )
    _, clock_extras = env.get_observations()
    clock_critic = clock_extras["observations"]["critic"]
    expected_time = torch.as_tensor(
        clock_backend.lengths / clock_backend.max_episode_length,
        dtype=torch.float32,
    )
    torch.testing.assert_close(clock_critic[:, spec.proprio_dim], expected_time)
    env.close()
    print("OBSERVATION_CONTRACT_OK")


if __name__ == "__main__":
    main()
