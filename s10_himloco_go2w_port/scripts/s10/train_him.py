#!/usr/bin/env python3
"""Train one strict-port S10 HIMLoco A/B/C/Official/Hybrid variant."""

from __future__ import annotations

import argparse
import faulthandler
import signal
import sys
import traceback
from datetime import datetime
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--variant", choices=("a", "b", "c", "official", "hybrid"), required=True)
parser.add_argument("--num-envs", type=int, default=4096)
parser.add_argument("--max-iterations", type=int, default=80000)
parser.add_argument("--save-interval", type=int, default=500)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--initial-noise-std", type=float, default=1.0)
parser.add_argument("--entropy-coef", type=float, default=None)
parser.add_argument("--learning-rate", type=float, default=None)
parser.add_argument("--run-name", type=str, default="")
parser.add_argument(
    "--smoke",
    action="store_true",
    help="Use a runtime-only 2x6 terrain atlas; repository training defaults are unchanged.",
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

faulthandler.register(signal.SIGUSR1, all_threads=True)

if min(args.num_envs, args.max_iterations, args.save_interval) <= 0:
    parser.error("environment, iteration and save counts must be positive")
if args.initial_noise_std <= 0.0:
    parser.error("--initial-noise-std must be positive")

launcher = AppLauncher(args)
simulation_app = launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import locowheeledlegged  # noqa: E402,F401
from locowheeledlegged.config.s10.him_env_cfg import (  # noqa: E402
    HIMLocomotionAEnvCfg,
    HIMLocomotionBEnvCfg,
    HIMLocomotionCEnvCfg,
    HIMHybridReferenceEnvCfg,
    HIMOfficialReferenceEnvCfg,
)
from locowheeledlegged.config.s10.reference_env_cfg import (  # noqa: E402
    HYBRID_TERRAIN_FAMILIES,
    LEGACY_TERRAIN_FAMILIES,
    OFFICIAL_TERRAIN_FAMILIES,
    terrain_family_columns,
)
from locowheeledlegged.him import HIMRunner  # noqa: E402


SPECS = {
    "a": ("Isaac-LocomotionS10-HIM-A-v1", HIMLocomotionAEnvCfg, 0.01, 3.0e-4, LEGACY_TERRAIN_FAMILIES),
    "b": ("Isaac-LocomotionS10-HIM-B-v1", HIMLocomotionBEnvCfg, 0.01, 3.0e-4, LEGACY_TERRAIN_FAMILIES),
    "c": ("Isaac-LocomotionS10-HIM-C-v1", HIMLocomotionCEnvCfg, 0.01, 3.0e-4, LEGACY_TERRAIN_FAMILIES),
    "official": (
        "Isaac-LocomotionS10-HIM-Official-v1",
        HIMOfficialReferenceEnvCfg,
        0.003,
        1.0e-3,
        OFFICIAL_TERRAIN_FAMILIES,
    ),
    "hybrid": (
        "Isaac-LocomotionS10-HIM-Hybrid-v1",
        HIMHybridReferenceEnvCfg,
        0.005,
        5.0e-4,
        HYBRID_TERRAIN_FAMILIES,
    ),
}


def main() -> None:
    task_id, cfg_type, default_entropy, default_learning_rate, terrain_family_names = SPECS[args.variant]
    entropy_coef = default_entropy if args.entropy_coef is None else args.entropy_coef
    learning_rate = default_learning_rate if args.learning_rate is None else args.learning_rate
    cfg = cfg_type()
    cfg.scene.num_envs = args.num_envs
    cfg.seed = args.seed
    if args.smoke:
        cfg.scene.replicate_physics = True
        cfg.scene.terrain.terrain_generator.num_rows = 2
        cfg.scene.terrain.terrain_generator.num_cols = 6
        cfg.scene.terrain.terrain_generator.border_width = 2.0
        cfg.scene.terrain.max_init_terrain_level = 1
    if args.device is not None:
        cfg.sim.device = args.device
    terrain_families = terrain_family_columns(
        cfg.scene.terrain.terrain_generator,
        terrain_family_names,
    )

    run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + f"_{args.variant}_him"
    if args.run_name:
        run_name += f"_{args.run_name}"
    log_dir = PROJECT_ROOT / "logs" / "s10_strict_him" / run_name

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    env = gym.make(task_id, cfg=cfg)
    wrapped = RslRlVecEnvWrapper(env, clip_actions=None)
    runner = HIMRunner(
        wrapped,
        log_dir=log_dir,
        save_interval=args.save_interval,
        device=cfg.sim.device,
        initial_noise_std=args.initial_noise_std,
        entropy_coef=entropy_coef,
        learning_rate=learning_rate,
        terrain_family_columns=terrain_families,
        policy_variant="blind",
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
