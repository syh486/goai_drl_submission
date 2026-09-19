"""Run a CPU-only migration smoke without a navigation task or LiDAR buffer.

This checks model construction, recurrent hidden-state handling, rollout
storage, GAE, PPO update, and the MuJoCo action/observation boundary.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import torch

from sru_training.s10_mujoco_env import S10MujocoVecEnv, S10RawState
from sru_training.s10_policy_config import ppo_config
from rsl_rl.runners import OnPolicyRunner


class NumericalBackend:
    """Numerical backend used only to exercise the framework contract."""

    def __init__(self, num_envs: int = 2, max_episode_length: int = 8):
        self.num_envs = num_envs
        self.max_episode_length = max_episode_length
        self.step_count = 0

    def _state(self) -> S10RawState:
        n = self.num_envs
        return S10RawState(
            base_lin_vel=torch.zeros(n, 3),
            base_ang_vel=torch.zeros(n, 3),
            projected_gravity=torch.tensor([[0.0, 0.0, -1.0]]).repeat(n, 1),
            last_action=torch.zeros(n, 2),
            goal_body=torch.zeros(n, 4),
            lidar_latent=torch.zeros(n, 64, 5, 8),
            time_normalized=torch.full((n, 1), min(self.step_count / self.max_episode_length, 1.0)),
        )

    def reset(self) -> S10RawState:
        self.step_count = 0
        return self._state()

    def step(self, cmd_vel: torch.Tensor):
        self.step_count += 1
        rewards = -(cmd_vel[:, 0].square() + cmd_vel[:, 2].square())
        dones = torch.zeros(self.num_envs, dtype=torch.bool)
        if self.step_count % self.max_episode_length == 0:
            dones[:] = True
        return self._state(), rewards, dones, {"backend": "numerical_smoke"}

    def close(self) -> None:
        pass


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    env = S10MujocoVecEnv(NumericalBackend(), device=device)
    cfg = ppo_config(smoke=True)
    log_dir = Path("/tmp/s10_sru_migration_smoke")
    shutil.rmtree(log_dir, ignore_errors=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    runner = OnPolicyRunner(env, cfg, log_dir=str(log_dir), device=str(device))
    runner.learn(num_learning_iterations=1)
    actor = runner.alg.actor_critic
    print(
        "SMOKE_OK",
        {
            "actor_obs": env.num_obs,
            "critic_obs": env.num_privileged_obs,
            "actions": env.num_actions,
            "actor_parameters": sum(p.numel() for p in actor.parameters()),
            "checkpoint": str(log_dir / "model_1.pt"),
        },
    )
    env.close()


if __name__ == "__main__":
    main()
