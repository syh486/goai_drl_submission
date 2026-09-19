#!/usr/bin/env python3
"""Deterministic playback and fixed-scenario audit for strict-port S10 HIM checkpoints."""

from __future__ import annotations

import argparse
import faulthandler
import json
import math
import signal
import sys
import time
import traceback
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--variant", choices=("a", "b", "c", "official", "hybrid"), required=True)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument(
    "--randomized",
    action="store_true",
    help="Keep training-time observation corruption, domain randomization, and pushes enabled.",
)
parser.add_argument(
    "--terrain",
    choices=("flat", "boxes", "perlin", "random_rough", "stairs_down", "stairs_up", "ramp_up", "ramp_down"),
    default="stairs_up",
)
parser.add_argument("--level", type=int, default=0)
parser.add_argument("--vx", type=float, default=0.5)
parser.add_argument("--vy", type=float, default=0.0)
parser.add_argument("--omega", type=float, default=0.0)
parser.add_argument("--warmup-steps", type=int, default=100)
parser.add_argument("--playback-steps", type=int, default=600)
parser.add_argument("--print-every", type=int, default=100)
parser.add_argument("--no-real-time", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

faulthandler.register(signal.SIGUSR1, all_threads=True)

launcher = AppLauncher(args)
simulation_app = launcher.app

import gymnasium as gym  # noqa: E402
import isaaclab.terrains as terrain_gen  # noqa: E402
import torch  # noqa: E402
from isaaclab.utils.math import euler_xyz_from_quat  # noqa: E402
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
from locowheeledlegged.him import HIMRunner  # noqa: E402


TERRAIN_KEY = {
    "flat": "flat",
    "boxes": "boxes",
    "perlin": "perlin_rough",
    "random_rough": "random_rough",
    "stairs_down": "pyramid_stairs",
    "stairs_up": "pyramid_stairs_inv",
    "ramp_up": "hf_pyramid_slope_inv",
    "ramp_down": "hf_pyramid_slope",
}
CFG_BY_VARIANT = {
    "a": HIMLocomotionAEnvCfg,
    "b": HIMLocomotionBEnvCfg,
    "c": HIMLocomotionCEnvCfg,
    "official": HIMOfficialReferenceEnvCfg,
    "hybrid": HIMHybridReferenceEnvCfg,
}
TASK_BY_VARIANT = {
    "a": "Isaac-LocomotionS10-HIM-A-v1",
    "b": "Isaac-LocomotionS10-HIM-B-v1",
    "c": "Isaac-LocomotionS10-HIM-C-v1",
    "official": "Isaac-LocomotionS10-HIM-Official-v1",
    "hybrid": "Isaac-LocomotionS10-HIM-Hybrid-v1",
}


def _disable_randomization(cfg) -> None:
    for name in (
        "randomize_base_mass",
        "randomize_foot_physics_material",
        "randomize_rigid_body_inertia",
        "randomize_apply_external_force_torque",
        "randomize_actuator_gains",
        "push_robot",
    ):
        setattr(cfg.events, name, None)
    cfg.events.reset_base.params["pose_range"] = {
        "x": (0.0, 0.0),
        "y": (0.0, 0.0),
        "z": (0.0, 0.0),
        "roll": (0.0, 0.0),
        "pitch": (0.0, 0.0),
        "yaw": (0.0, 0.0),
    }
    cfg.events.reset_base.params["velocity_range"] = {
        axis: (0.0, 0.0) for axis in ("x", "y", "z", "roll", "pitch", "yaw")
    }


def _disable_curriculum(cfg) -> None:
    for name in ("command_x_levels", "command_y_levels", "command_z_levels", "terrain_levels"):
        if hasattr(cfg.curriculum, name):
            setattr(cfg.curriculum, name, None)
    cfg.scene.terrain.terrain_generator.curriculum = False


def _isolate_terrain(cfg, terrain_name: str) -> None:
    """Generate a deterministic single-family atlas for honest playback.

    The upstream terrain generator randomly distributes terrain families over
    columns according to their proportions.  A fixed column therefore cannot
    be used as a stable family selector.  Keeping only the requested family
    preserves its ten curriculum difficulty rows while removing that ambiguity.
    """

    selected = TERRAIN_KEY[terrain_name]
    sub_terrains = cfg.scene.terrain.terrain_generator.sub_terrains
    if selected not in sub_terrains:
        # The official M20 training recipe intentionally has no flat family.
        # A synthetic plane is nevertheless required for a common baseline
        # evaluation across Official and Hybrid checkpoints.  This only edits
        # the runtime playback config and never changes the training task.
        if selected == "flat":
            sub_terrains[selected] = terrain_gen.MeshPlaneTerrainCfg(proportion=1.0)
        else:
            raise ValueError(
                f"terrain {terrain_name!r} is unavailable for variant {args.variant!r}; "
                f"available keys={tuple(sub_terrains)}"
            )
    for name, terrain_cfg in sub_terrains.items():
        terrain_cfg.proportion = 1.0 if name == selected else 0.0
    cfg.scene.terrain.terrain_generator.num_cols = 1
    # Keep rows ordered by difficulty even though the runtime curriculum
    # manager is disabled.  Setting this False randomizes difficulty per row,
    # making a requested level index meaningless during evaluation.
    cfg.scene.terrain.terrain_generator.curriculum = True


def _set_command(command_term, command: torch.Tensor) -> None:
    command_term.vel_command_b[:] = command
    if hasattr(command_term, "vel_command_b_buffer"):
        command_term.vel_command_b_buffer[:] = command


def _mean(values: list[float]) -> float:
    return sum(values) / max(len(values), 1)


def main() -> None:
    print("[PLAY] loading checkpoint and environment configuration", flush=True)
    if args.level < 0 or args.level > 9:
        raise ValueError("terrain level must be in [0, 9]")
    if args.warmup_steps < 0 or args.playback_steps <= 0:
        raise ValueError("warmup must be non-negative and playback must be positive")

    checkpoint_path = args.checkpoint.expanduser().resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = CFG_BY_VARIANT[args.variant]()
    cfg.scene.num_envs = 1
    cfg.seed = args.seed
    cfg.observations.policy.enable_corruption = args.randomized
    cfg.commands.base_velocity.initial_zero_command_steps = 0
    cfg.commands.base_velocity.rel_standing_envs = 0.0
    cfg.commands.base_velocity.bang_bang_envs = 0.0
    cfg.commands.base_velocity.resampling_time_range = (1.0e9, 1.0e9)
    if not args.randomized:
        _disable_randomization(cfg)
    _disable_curriculum(cfg)
    _isolate_terrain(cfg, args.terrain)
    if args.device is not None:
        cfg.sim.device = args.device

    print(
        f"[PLAY] creating environment variant={args.variant} terrain={args.terrain} "
        f"level={args.level} device={cfg.sim.device}",
        flush=True,
    )
    env = gym.make(TASK_BY_VARIANT[args.variant], cfg=cfg)
    print("[PLAY] environment created", flush=True)
    wrapped = RslRlVecEnvWrapper(env, clip_actions=None)
    raw = wrapped.unwrapped

    runner = HIMRunner(
        wrapped,
        log_dir=Path("/tmp/s10_him_play"),
        save_interval=0,
        device=cfg.sim.device,
        initial_noise_std=checkpoint.get("initial_noise_std", 1.0),
        entropy_coef=float(checkpoint.get("entropy_coef", 0.01)),
        policy_variant="blind",
    )
    runner.actor_critic.load_state_dict(checkpoint["model_state_dict"])
    runner.actor_critic.eval()

    terrain = raw.scene.terrain
    column = 0
    terrain.terrain_levels.fill_(args.level)
    terrain.terrain_types.fill_(column)
    terrain.env_origins[:] = terrain.terrain_origins[args.level, column]

    wrapped.reset()
    command_term = raw.command_manager.get_term("base_velocity")
    command = torch.tensor([[args.vx, args.vy, args.omega]], device=cfg.sim.device)
    zero_command = torch.zeros_like(command)

    def policy_step(fixed_command: torch.Tensor):
        _set_command(command_term, fixed_command)
        _, extras = wrapped.get_observations()
        history = torch.flip(extras["observations"]["policy"], dims=(1,)).flatten(start_dim=1)
        with torch.inference_mode():
            actions = runner.actor_critic.act_inference(history)
            _, rewards, dones, info = wrapped.step(actions)
        return actions, rewards, dones, info

    for _ in range(args.warmup_steps):
        policy_step(zero_command)

    robot = raw.scene["robot"]
    start_position = robot.data.root_pos_w[0].clone()
    stats: dict[str, list[float]] = {
        key: [] for key in ("vx", "vy", "omega", "height", "roll", "pitch", "action_abs", "action_max")
    }
    reset_count = 0
    base_contact_count = 0
    hip_contact_count = 0
    step_dt = float(raw.step_dt)

    for step in range(args.playback_steps):
        started = time.perf_counter()
        actions, _, dones, _ = policy_step(command)
        reset_count += int(dones.sum().item())
        active_terms = raw.termination_manager.active_terms
        if "base_contact" in active_terms:
            base_contact_count += int(raw.termination_manager.get_term("base_contact").sum().item())
        if "hip_contact" in active_terms:
            hip_contact_count += int(raw.termination_manager.get_term("hip_contact").sum().item())

        lin_vel = robot.data.root_lin_vel_b[0]
        ang_vel = robot.data.root_ang_vel_b[0]
        roll, pitch, _ = euler_xyz_from_quat(robot.data.root_quat_w[0:1])
        roll_deg = math.degrees(math.atan2(math.sin(float(roll[0])), math.cos(float(roll[0]))))
        pitch_deg = math.degrees(math.atan2(math.sin(float(pitch[0])), math.cos(float(pitch[0]))))
        stats["vx"].append(float(lin_vel[0]))
        stats["vy"].append(float(lin_vel[1]))
        stats["omega"].append(float(ang_vel[2]))
        stats["height"].append(float(robot.data.root_pos_w[0, 2] - terrain.env_origins[0, 2]))
        stats["roll"].append(roll_deg)
        stats["pitch"].append(pitch_deg)
        stats["action_abs"].append(float(actions.abs().mean()))
        stats["action_max"].append(float(actions.abs().max()))

        if args.print_every > 0 and (step + 1) % args.print_every == 0:
            delta = robot.data.root_pos_w[0] - start_position
            print(
                f"[PLAY {step + 1:04d}] x={float(delta[0]):+.3f} y={float(delta[1]):+.3f} "
                f"vx={stats['vx'][-1]:+.3f} vy={stats['vy'][-1]:+.3f} "
                f"omega={stats['omega'][-1]:+.3f} z={stats['height'][-1]:.3f} "
                f"roll={roll_deg:+.2f} pitch={pitch_deg:+.2f} "
                f"|a|max={stats['action_max'][-1]:.2f} resets={reset_count}",
                flush=True,
            )
        if not args.no_real_time:
            time.sleep(max(0.0, step_dt - (time.perf_counter() - started)))

    delta = robot.data.root_pos_w[0] - start_position
    summary = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_iteration": int(checkpoint.get("iteration", -1)),
        "variant": args.variant,
        "seed": args.seed,
        "randomized": args.randomized,
        "terrain": args.terrain,
        "level": args.level,
        "command": [args.vx, args.vy, args.omega],
        "duration_s": args.playback_steps * step_dt,
        "reset_count": reset_count,
        "base_contact_count": base_contact_count,
        "hip_contact_count": hip_contact_count,
        "final_dx": float(delta[0]),
        "final_dy": float(delta[1]),
        "mean_vx": _mean(stats["vx"]),
        "mean_vy": _mean(stats["vy"]),
        "mean_omega": _mean(stats["omega"]),
        "mean_height": _mean(stats["height"]),
        "mean_abs_roll_deg": _mean([abs(v) for v in stats["roll"]]),
        "mean_abs_pitch_deg": _mean([abs(v) for v in stats["pitch"]]),
        "mean_action_abs": _mean(stats["action_abs"]),
        "max_action_abs": max(stats["action_max"]),
    }
    print("SUMMARY " + json.dumps(summary, sort_keys=True), flush=True)
    wrapped.close()


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        traceback.print_exc()
        # SimulationApp.close() may raise SystemExit(0), which would otherwise
        # mask the real playback error and make a failed audit look successful.
        try:
            simulation_app.close()
        except SystemExit:
            pass
        raise
    else:
        simulation_app.close()
