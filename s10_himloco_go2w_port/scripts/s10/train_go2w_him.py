#!/usr/bin/env python3
"""Train the faithful S10 port of HIMLoco-for-Go2W."""

from __future__ import annotations

import argparse
import sys
import traceback
from datetime import datetime
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--num-envs", type=int, default=4096)
parser.add_argument("--max-iterations", type=int, default=20000)
parser.add_argument("--save-interval", type=int, default=1000)
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--run-name", default="")
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
from locowheeledlegged.config.s10.go2w_him_env_cfg import Go2WHIMEnvCfg  # noqa: E402
from locowheeledlegged.him import HIMRunner  # noqa: E402


def main() -> None:
    cfg = Go2WHIMEnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.seed = args.seed
    if args.device is not None:
        cfg.sim.device = args.device
    if args.smoke:
        cfg.scene.replicate_physics = True
        # Force several time-out resets inside the 48-step rollout so the
        # IsaacGym-compatible history-preservation path is exercised.
        cfg.episode_length_s = 0.08
        cfg.scene.terrain.terrain_generator.num_rows = 1
        cfg.scene.terrain.terrain_generator.num_cols = 20
        cfg.scene.terrain.terrain_generator.border_width = 2.0
        cfg.scene.terrain.max_init_terrain_level = 0

    suffix = f"_{args.run_name}" if args.run_name else ""
    run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + f"_go2w_him{suffix}"
    log_dir = ROOT / "logs" / "s10_go2w_him" / run_name
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    env = gym.make("Isaac-S10-Go2W-HIM-v1", cfg=cfg)
    # The IsaacGym reference clips raw policy actions to +/-100 before the
    # delayed PD/velocity controller.  It is normally inactive, but keeping it
    # here makes the simulator contract exact under exploration spikes.
    wrapped = RslRlVecEnvWrapper(env, clip_actions=100.0)
    runner = HIMRunner(
        wrapped,
        log_dir=log_dir,
        rollout_steps=48,
        save_interval=args.save_interval,
        device=cfg.sim.device,
        initial_noise_std=1.0,
        entropy_coef=0.005,
        learning_rate=1.0e-3,
        policy_variant="blind",
    )
    if runner.one_step_dim != 57 or runner.history_dim != 342 or runner.critic_dim != 262:
        raise RuntimeError(
            f"protocol mismatch: one_step={runner.one_step_dim}, history={runner.history_dim}, "
            f"critic={runner.critic_dim}; expected 57/342/262"
        )
    try:
        runner.learn(args.max_iterations)
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
