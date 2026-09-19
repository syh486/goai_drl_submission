#!/usr/bin/env python3
"""Deterministic GUI playback for the strict S10 Go2W HIM port."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import traceback
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--terrain", choices=("flat", "stairs_up", "platform"), default="stairs_up")
parser.add_argument("--level", type=int, default=5)
parser.add_argument("--vx", type=float, default=0.9)
parser.add_argument("--vy", type=float, default=0.0)
parser.add_argument("--omega", type=float, default=0.0)
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--warmup-steps", type=int, default=50)
parser.add_argument("--playback-steps", type=int, default=1000000)
parser.add_argument("--print-every", type=int, default=100)
parser.add_argument("--randomized", action="store_true")
parser.add_argument("--no-real-time", action="store_true")
parser.add_argument("--trace", type=Path, default=None, help="Write one JSON object per policy step.")
parser.add_argument("--action-delay", type=int, default=0, choices=range(4))
parser.add_argument("--print-asset-audit", action="store_true")
parser.add_argument(
    "--wheel-velocity-observation", choices=("auto", "zero", "measured"), default="auto"
)
parser.add_argument(
    "--actuator-integration", choices=("auto", "implicit", "explicit"), default="auto"
)
parser.add_argument("--physics-dt", type=float, choices=(0.001, 0.002, 0.005), default=None)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

launcher = AppLauncher(args)
simulation_app = launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from isaaclab.utils.math import euler_xyz_from_quat  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import locowheeledlegged  # noqa: E402,F401
from locowheeledlegged.config.s10.go2w_deployment_env_cfg import (  # noqa: E402
    Go2WDeploymentHIMEnvCfg,
)
from locowheeledlegged.config.s10.go2w_him_env_cfg import Go2WHIMEnvCfg  # noqa: E402
from locowheeledlegged.him import HIMRunner  # noqa: E402
from s10_policy_protocol import POLICY_JOINT_NAMES  # noqa: E402


TERRAIN_KEY = {"flat": "smooth_slope_up", "stairs_up": "stairs_up", "platform": "discrete"}


def _disable_randomization(cfg: Go2WHIMEnvCfg) -> None:
    for name in (
        "randomize_base_mass",
        "randomize_material",
        "randomize_actuator_gains",
        "disturbance",
        "push_robot",
    ):
        setattr(cfg.events, name, None)
    cfg.events.reset_base.params["pose_range"] = {
        axis: (0.0, 0.0) for axis in ("x", "y", "z", "roll", "pitch", "yaw")
    }
    cfg.events.reset_base.params["velocity_range"] = {
        axis: (0.0, 0.0) for axis in ("x", "y", "z", "roll", "pitch", "yaw")
    }
    cfg.events.reset_joints.params["position_range"] = (1.0, 1.0)


def _isolate_terrain(cfg: Go2WHIMEnvCfg, terrain: str) -> None:
    selected = TERRAIN_KEY[terrain]
    sub_terrains = cfg.scene.terrain.terrain_generator.sub_terrains
    for name, terrain_cfg in sub_terrains.items():
        terrain_cfg.proportion = 1.0 if name == selected else 0.0
    cfg.scene.terrain.terrain_generator.num_cols = 1
    cfg.scene.terrain.terrain_generator.curriculum = True
    cfg.scene.terrain.max_init_terrain_level = args.level


def _set_command(term, command: torch.Tensor) -> None:
    term.vel_command_b[:] = command
    if hasattr(term, "vel_command_b_buffer"):
        term.vel_command_b_buffer[:] = command
    # Heading mode updates yaw every policy step.  Disable it for an exact
    # fixed [vx, vy, omega] playback command.
    if hasattr(term, "is_heading_env"):
        term.is_heading_env.zero_()
    if hasattr(term, "is_standing_env"):
        term.is_standing_env.zero_()


def main() -> None:
    if not 0 <= args.level <= 9:
        raise ValueError("terrain level must be in [0, 9]")
    checkpoint_path = args.checkpoint.expanduser().resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if int(checkpoint.get("iteration", -1)) != 5000:
        print(f"[WARN] checkpoint iteration={checkpoint.get('iteration')} (expected 5000)", flush=True)

    checkpoint_actuator = checkpoint.get("actuator_integration", "implicit")
    actuator_integration = (
        checkpoint_actuator if args.actuator_integration == "auto" else args.actuator_integration
    )
    if actuator_integration not in ("implicit", "explicit"):
        raise ValueError(f"unsupported checkpoint actuator integration: {actuator_integration}")
    checkpoint_wheel_observation = checkpoint.get("wheel_velocity_observation", "zero-signal")
    if args.wheel_velocity_observation == "auto":
        wheel_velocity_observation = (
            "measured" if checkpoint_wheel_observation == "measured" else "zero"
        )
    else:
        wheel_velocity_observation = args.wheel_velocity_observation
    cfg = Go2WHIMEnvCfg() if actuator_integration == "implicit" else Go2WDeploymentHIMEnvCfg()
    cfg.scene.num_envs = 1
    cfg.seed = args.seed
    cfg.observe_wheel_velocity = wheel_velocity_observation == "measured"
    physics_dt = (
        float(checkpoint.get("training_physics_dt", cfg.sim.dt))
        if args.physics_dt is None
        else args.physics_dt
    )
    if actuator_integration == "implicit" and physics_dt != 0.005:
        raise ValueError("non-default physics dt is only supported for explicit actuator diagnostics")
    cfg.sim.dt = physics_dt
    cfg.decimation = round(0.02 / physics_dt)
    if abs(cfg.decimation * physics_dt - 0.02) > 1.0e-12:
        raise ValueError("physics dt must divide the 20 ms policy period exactly")
    cfg.sim.render_interval = cfg.decimation
    cfg.scene.height_scanner.update_period = cfg.sim.dt
    cfg.scene.contact_forces.update_period = cfg.sim.dt
    cfg.curriculum.terrain_levels = None
    cfg.curriculum.command_range = None
    cfg.commands.base_velocity.resampling_time_range = (1.0e9, 1.0e9)
    cfg.commands.base_velocity.rel_standing_envs = 0.0
    cfg.commands.base_velocity.rel_heading_envs = 0.0
    cfg.observations.policy.enable_corruption = args.randomized
    cfg.observations.policy.proprio.params["add_noise"] = args.randomized
    cfg.observations.critic.proprio.params["add_noise"] = args.randomized
    cfg.sim2sim_action_delay_override = args.action_delay
    for actuator_name in ("legs", "wheels"):
        actuator_cfg = cfg.scene.robot.actuators[actuator_name]
        actuator_cfg.min_delay = args.action_delay
        actuator_cfg.max_delay = args.action_delay
        actuator_cfg.resample_every_n_physics_steps = cfg.decimation
    if not args.randomized:
        _disable_randomization(cfg)
    _isolate_terrain(cfg, args.terrain)
    if args.device is not None:
        cfg.sim.device = args.device

    print(
        f"[PLAY] checkpoint={checkpoint_path} terrain={args.terrain} level={args.level} "
        f"command=({args.vx}, {args.vy}, {args.omega}) device={cfg.sim.device} "
        f"wheel_velocity={wheel_velocity_observation} actuator={actuator_integration} "
        f"physics_dt={physics_dt} decimation={cfg.decimation}",
        flush=True,
    )
    task_id = (
        "Isaac-S10-Go2W-HIM-v1"
        if actuator_integration == "implicit"
        else "Isaac-S10-Go2W-Deployment-HIM-v1"
    )
    env = gym.make(task_id, cfg=cfg)
    wrapped = RslRlVecEnvWrapper(env, clip_actions=100.0)
    raw = wrapped.unwrapped
    runner = HIMRunner(
        wrapped,
        log_dir=Path("/tmp/s10_go2w_him_play"),
        save_interval=0,
        device=cfg.sim.device,
        initial_noise_std=checkpoint.get("initial_noise_std", 1.0),
        entropy_coef=float(checkpoint.get("entropy_coef", 0.005)),
        learning_rate=float(checkpoint.get("learning_rate", 1.0e-3)),
        policy_variant="blind",
    )
    runner.actor_critic.load_state_dict(checkpoint["model_state_dict"])
    runner.actor_critic.eval()

    terrain = raw.scene.terrain
    terrain.terrain_levels.fill_(args.level)
    terrain.terrain_types.zero_()
    terrain.env_origins[:] = terrain.terrain_origins[args.level, 0]
    wrapped.reset()
    robot = raw.scene["robot"]
    if args.print_asset_audit:
        materials = robot.root_physx_view.get_material_properties()[0].detach().cpu()
        audit = {
            "body_names": list(robot.body_names),
            "mass_kg": robot.data.default_mass[0].detach().cpu().tolist(),
            "com_pos_b_xyz": robot.data.com_pos_b[0].detach().cpu().tolist(),
            "com_quat_b_wxyz": robot.data.com_quat_b[0].detach().cpu().tolist(),
            "inertia_body_3x3": robot.data.default_inertia[0].detach().cpu().reshape(-1, 3, 3).tolist(),
            "material_unique": torch.unique(materials, dim=0).tolist(),
            "joint_names": list(robot.joint_names),
            "joint_stiffness": robot.data.joint_stiffness[0].detach().cpu().tolist(),
            "joint_damping": robot.data.joint_damping[0].detach().cpu().tolist(),
            "joint_effort_limits": robot.data.joint_effort_limits[0].detach().cpu().tolist(),
            "joint_velocity_limits": robot.data.joint_vel_limits[0].detach().cpu().tolist(),
        }
        print("ISAAC_ASSET_AUDIT " + json.dumps(audit, sort_keys=True), flush=True)

    command_term = raw.command_manager.get_term("base_velocity")
    command = torch.tensor([[args.vx, args.vy, args.omega]], device=cfg.sim.device)
    zero_command = torch.zeros_like(command)

    def step(fixed_command: torch.Tensor):
        _set_command(command_term, fixed_command)
        history, _ = runner._extract(raw.obs_buf)
        # The cached frame was assembled at the end of the preceding physics
        # step.  Patch only its current command slice so the first inference
        # cannot accidentally consume the command sampled during reset.
        history = history.clone()
        history[:, 6:9] = fixed_command * fixed_command.new_tensor((2.0, 2.0, 0.25))
        with torch.inference_mode():
            estimated_velocity, latent = runner.actor_critic.estimator.encode(history)
            actions = runner.actor_critic.act_inference(history)
            _, rewards, dones, info = wrapped.step(actions)
        inference = {
            "observation": history[0, :57].detach().cpu().tolist(),
            "estimated_velocity": estimated_velocity[0].detach().cpu().tolist(),
            "latent": latent[0].detach().cpu().tolist(),
            "action": actions[0].detach().cpu().tolist(),
        }
        return actions, rewards, dones, info, inference

    for _ in range(args.warmup_steps):
        step(zero_command)

    start = robot.data.root_pos_w[0].clone()
    sums = {name: 0.0 for name in ("vx", "vy", "omega", "roll", "pitch")}
    resets = 0
    steps_done = 0
    joint_ids = [robot.joint_names.index(name) for name in POLICY_JOINT_NAMES]
    trace_path = args.trace.expanduser().resolve() if args.trace is not None else None
    if trace_path is not None:
        trace_path.parent.mkdir(parents=True, exist_ok=True)
    trace_file = trace_path.open("w", encoding="utf-8") if trace_path is not None else None
    try:
        for index in range(args.playback_steps):
            if not simulation_app.is_running():
                break
            started = time.perf_counter()
            actions, _, dones, _, inference = step(command)
            resets += int(dones.sum().item())
            lin_vel = robot.data.root_lin_vel_b[0]
            ang_vel = robot.data.root_ang_vel_b[0]
            roll, pitch, _ = euler_xyz_from_quat(robot.data.root_quat_w[0:1])
            roll_value = math.atan2(math.sin(float(roll[0])), math.cos(float(roll[0])))
            pitch_value = math.atan2(math.sin(float(pitch[0])), math.cos(float(pitch[0])))
            roll_deg = math.degrees(roll_value)
            pitch_deg = math.degrees(pitch_value)
            delta = robot.data.root_pos_w[0] - start
            sums["vx"] += float(lin_vel[0])
            sums["vy"] += float(lin_vel[1])
            sums["omega"] += float(ang_vel[2])
            sums["roll"] += abs(roll_deg)
            sums["pitch"] += abs(pitch_deg)
            steps_done += 1
            if trace_file is not None:
                record = {
                    "simulator": "isaaclab",
                    "step": index + 1,
                    "time_s": (index + 1) * float(raw.step_dt),
                    "dx": float(delta[0]),
                    "dy": float(delta[1]),
                    "base_height": float(robot.data.root_pos_w[0, 2]),
                    "true_base_lin_vel_b": lin_vel.detach().cpu().tolist(),
                    "true_base_ang_vel_b": ang_vel.detach().cpu().tolist(),
                    "root_quat_wxyz": robot.data.root_quat_w[0].detach().cpu().tolist(),
                    "joint_pos": robot.data.joint_pos[0, joint_ids].detach().cpu().tolist(),
                    "joint_vel": robot.data.joint_vel[0, joint_ids].detach().cpu().tolist(),
                    "applied_torque": robot.data.applied_torque[0, joint_ids].detach().cpu().tolist(),
                    "done": bool(dones[0]),
                    **inference,
                }
                trace_file.write(json.dumps(record, separators=(",", ":")) + "\n")
            if args.print_every > 0 and (index + 1) % args.print_every == 0:
                print(
                    f"[PLAY {index + 1:06d}] dx={float(delta[0]):+.2f} dy={float(delta[1]):+.2f} "
                    f"vx={float(lin_vel[0]):+.2f} omega={float(ang_vel[2]):+.2f} "
                    f"roll={roll_deg:+.1f} pitch={pitch_deg:+.1f} "
                    f"|a|max={float(actions.abs().max()):.2f} resets={resets}",
                    flush=True,
                )
            if not args.no_real_time:
                time.sleep(max(0.0, float(raw.step_dt) - (time.perf_counter() - started)))
    finally:
        if trace_file is not None:
            trace_file.close()

    denom = max(steps_done, 1)
    delta = robot.data.root_pos_w[0] - start
    print(
        "SUMMARY "
        + json.dumps(
            {
                "checkpoint_iteration": int(checkpoint.get("iteration", -1)),
                "terrain": args.terrain,
                "level": args.level,
                "command": [args.vx, args.vy, args.omega],
                "wheel_velocity_observation": wheel_velocity_observation,
                "actuator_integration": actuator_integration,
                "physics_dt": physics_dt,
                "decimation": cfg.decimation,
                "steps": steps_done,
                "resets": resets,
                "final_dx": float(delta[0]),
                "final_dy": float(delta[1]),
                "mean_vx": sums["vx"] / denom,
                "mean_vy": sums["vy"] / denom,
                "mean_omega": sums["omega"] / denom,
                "mean_abs_roll_deg": sums["roll"] / denom,
                "mean_abs_pitch_deg": sums["pitch"] / denom,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    wrapped.close()


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        traceback.print_exc()
        raise
    finally:
        simulation_app.close()
