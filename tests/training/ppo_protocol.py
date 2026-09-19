"""Check that the MuJoCo PPO protocol matches the upstream MX runner."""

from __future__ import annotations

import torch

import sru_training  # Select the repository's SRU-enabled rsl_rl package.
from rsl_rl.algorithms import PPO
from rsl_rl.modules import ActorCritic
from sru_training.s10_mujoco_backend import LOW_LEVEL_PROFILES
from sru_training.s10_mujoco_env import S10MujocoVecEnv, S10RawState
from sru_training.s10_policy_config import ppo_config


class _Backend:
    num_envs = 2
    max_episode_length = 8

    def _state(self):
        return S10RawState(
            base_lin_vel=torch.zeros(self.num_envs, 3),
            base_ang_vel=torch.zeros(self.num_envs, 3),
            projected_gravity=torch.tensor([[0.0, 0.0, -1.0]]).repeat(self.num_envs, 1),
            last_action=torch.zeros(self.num_envs, 2),
            goal_body=torch.zeros(self.num_envs, 4),
            lidar_latent=torch.zeros(self.num_envs, 64, 5, 8),
        )

    def reset(self):
        return self._state()

    def step(self, _cmd):
        return self._state(), torch.zeros(self.num_envs), torch.zeros(self.num_envs, dtype=torch.bool), {}

    def close(self):
        pass


def main() -> None:
    cfg = ppo_config()
    assert cfg["num_steps_per_env"] == 16
    # The shared upstream MX config carries this field, but the upstream
    # runner applies it only when the selected algorithm is MDPO.
    assert cfg["reward_shifting_value"] == 0.05
    assert cfg["algorithm"]["class_name"] == "PPO"
    assert cfg["algorithm"]["gamma"] == 0.995
    assert cfg["algorithm"]["schedule"] == "adaptive"

    env = S10MujocoVecEnv(_Backend(), device="cpu")
    base_scale = torch.tensor((1.5, 1.0))
    assert torch.allclose(env._policy_scale, base_scale.expand(2, -1))
    assert torch.count_nonzero(env._policy_bias) == 0
    assert torch.all((env._filter_alpha >= 0.1) & (env._filter_alpha <= 0.6))
    env.close()

    official = LOW_LEVEL_PROFILES["official_20260828"]
    assert official.command_scale == (1.5, 0.5, 0.6)
    assert official.wheel_kd == 0.8
    assert official.startup_ramp_steps == 150

    # MuJoCo VecEnv observations are borrowed, reusable buffers. PPO must
    # snapshot them in act(), before env.step() overwrites their contents.
    policy = ActorCritic(
        num_actor_obs=2,
        num_critic_obs=3,
        num_actions=1,
        actor_hidden_dims=[4],
        critic_hidden_dims=[4],
    )
    algorithm = PPO(policy, num_learning_epochs=1, num_mini_batches=1)
    actor_obs = torch.tensor([[1.0, 2.0]])
    critic_obs = torch.tensor([[3.0, 4.0, 5.0]])
    algorithm.act(actor_obs, critic_obs)
    actor_obs.fill_(99.0)
    critic_obs.fill_(99.0)
    torch.testing.assert_close(
        algorithm.transition.observations,
        torch.tensor([[1.0, 2.0]]),
    )
    torch.testing.assert_close(
        algorithm.transition.critic_observations,
        torch.tensor([[3.0, 4.0, 5.0]]),
    )
    print("PPO_PROTOCOL_OK")


if __name__ == "__main__":
    main()
