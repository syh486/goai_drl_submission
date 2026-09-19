#!/usr/bin/env python3
"""Fine-tune a strict Go2W checkpoint on the deployment-aligned S10 protocol."""

from __future__ import annotations

import argparse
import sys
import traceback
from datetime import datetime
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--num-envs", type=int, default=4096)
parser.add_argument("--max-iterations", type=int, default=3000)
parser.add_argument("--save-interval", type=int, default=250)
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--run-name", default="")
parser.add_argument(
    "--exploration-scale",
    type=float,
    default=1.0,
    help="Scale the checkpoint's per-action std; preserves its leg/wheel asymmetry.",
)
parser.add_argument(
    "--exploration-std",
    type=float,
    default=None,
    help="Optional scalar override; normally leave unset and use --exploration-scale.",
)
parser.add_argument("--learning-rate", type=float, default=1.0e-4)
parser.add_argument("--smoke", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
launcher = AppLauncher(args)
simulation_app = launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import locowheeledlegged  # noqa: E402,F401
from locowheeledlegged.config.s10.go2w_deployment_env_cfg import (  # noqa: E402
    Go2WDeploymentHIMEnvCfg,
)
from locowheeledlegged.him import HIMRunner  # noqa: E402


def main() -> None:
    cfg = Go2WDeploymentHIMEnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.seed = args.seed
    if args.device is not None:
        cfg.sim.device = args.device
    if args.smoke:
        cfg.scene.replicate_physics = True
        cfg.episode_length_s = 0.08
        cfg.scene.terrain.terrain_generator.num_rows = 1
        cfg.scene.terrain.terrain_generator.num_cols = 20
        cfg.scene.terrain.terrain_generator.border_width = 2.0
        cfg.scene.terrain.max_init_terrain_level = 0

    suffix = f"_{args.run_name}" if args.run_name else ""
    run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + f"_go2w_deploy_ft{suffix}"
    log_dir = ROOT / "logs" / "s10_go2w_deployment_finetune" / run_name
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    env = gym.make("Isaac-S10-Go2W-Deployment-HIM-v1", cfg=cfg)
    wrapped = RslRlVecEnvWrapper(env, clip_actions=100.0)
    runner = HIMRunner(
        wrapped,
        log_dir=log_dir,
        rollout_steps=48,
        save_interval=args.save_interval,
        device=cfg.sim.device,
        initial_noise_std=1.0,
        entropy_coef=0.005,
        learning_rate=args.learning_rate,
        policy_variant="blind",
    )
    if runner.one_step_dim != 57 or runner.history_dim != 342 or runner.critic_dim != 262:
        raise RuntimeError(
            f"protocol mismatch: one_step={runner.one_step_dim}, history={runner.history_dim}, "
            f"critic={runner.critic_dim}; expected 57/342/262"
        )
    try:
        # The observation and actuator protocol changed, so stale Adam moments
        # are more harmful than useful.  Transfer every network (including the
        # critic) but start both optimizers and curriculum state fresh.
        runner.load_transfer(args.checkpoint, reset_critic=False)
        if args.exploration_std is not None:
            runner.set_exploration_std(args.exploration_std)
        else:
            if args.exploration_scale <= 0.0:
                raise ValueError("exploration scale must be positive")
            runner.set_exploration_std(
                runner.actor_critic.std.detach().cpu() * args.exploration_scale
            )
        runner.algorithm.learning_rate = args.learning_rate
        for optimizer in (runner.algorithm.optimizer, runner.algorithm.estimator_optimizer):
            for group in optimizer.param_groups:
                group["lr"] = args.learning_rate
        print(
            "[INFO] deployment fine-tune protocol: measured wheel velocity, explicit clipped PD, "
            f"physics_dt={cfg.sim.dt:g}, decimation={cfg.decimation}, "
            f"learning_rate={args.learning_rate:g}, "
            f"exploration={'scalar ' + str(args.exploration_std) if args.exploration_std is not None else 'checkpoint x ' + str(args.exploration_scale)}",
            flush=True,
        )
        runner.learn(args.max_iterations, randomize_initial_episode_length=not args.smoke)
        runner.save(log_dir / f"model_{runner.iteration}.pt")
    finally:
        wrapped.close()


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        traceback.print_exc()
        raise
    finally:
        simulation_app.close()
