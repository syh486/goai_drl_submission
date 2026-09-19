#!/usr/bin/env python3
"""Run an IsaacLab-trained S10 Go2W/HIM policy in the official MuJoCo model."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import time
from pathlib import Path

import mujoco
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from s10_policy_protocol import (  # noqa: E402
    DEFAULT_JOINT_POSITIONS,
    LEG_JOINT_NAMES,
    POLICY_JOINT_NAMES,
    POLICY_TO_ROBOT_INDICES,
    ROBOT_JOINT_NAMES,
    ROBOT_TO_POLICY_INDICES,
    WHEEL_JOINT_NAMES,
    audit_mujoco_s10_protocol,
)


def _load_him_core():
    """Load the simulator-independent network without importing IsaacLab packages."""

    path = ROOT / "locowheeledlegged/him/core.py"
    spec = importlib.util.spec_from_file_location("s10_go2w_him_core", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load HIM core from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_HIM_CORE = _load_him_core()
PIMHIMActorCritic = _HIM_CORE.PIMHIMActorCritic
PIMHIMCnnEstimator = _HIM_CORE.PIMHIMCnnEstimator


ONE_STEP_DIM = 57
HISTORY_LENGTH = 6
ACTION_DIM = 16
# The learned policy remains 50 Hz.  MuJoCo uses a finer integration step than
# PhysX/IsaacLab; the official S10 MuJoCo backend also runs at 1 kHz.  Holding
# each policy target for 20 substeps avoids changing the policy-side timing.
PHYSICS_DT = 0.001
POLICY_DT = 0.02
PHYSICS_STEPS_PER_POLICY = int(round(POLICY_DT / PHYSICS_DT))
LEG_COUNT = len(LEG_JOINT_NAMES)
WHEEL_COUNT = len(WHEEL_JOINT_NAMES)

LEG_KP = 80.0
LEG_KD = 2.0
WHEEL_KD = 0.8
LEG_EFFORT_LIMIT = 50.0
WHEEL_EFFORT_LIMIT = 14.0
WHEEL_VELOCITY_LIMIT = 65.5

LEG_ACTION_SCALE = np.asarray(
    [0.125 if "hipx" in name else 0.25 for name in LEG_JOINT_NAMES], dtype=np.float64
)
WHEEL_ACTION_SCALE = 5.0
DEFAULT_POLICY_POSITION = np.asarray(
    [DEFAULT_JOINT_POSITIONS[name] for name in POLICY_JOINT_NAMES], dtype=np.float64
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--xml",
        type=Path,
        default=ROOT / "locowheeledlegged/assets/s10/official/mjcf/S10.xml",
    )
    parser.add_argument("--terrain", choices=("flat", "stairs"), default="flat")
    parser.add_argument("--level", type=int, default=5, help="IsaacLab terrain row, in [0, 9].")
    parser.add_argument("--step-height", type=float, default=None)
    parser.add_argument(
        "--stair-count",
        type=int,
        default=9,
        help="IsaacLab's 8 m / 3 m-platform / 0.30 m inverted pyramid resolves to 9 steps.",
    )
    parser.add_argument("--stair-width", type=float, default=0.30)
    parser.add_argument("--terrain-size", type=float, default=8.0)
    parser.add_argument("--vx", type=float, default=0.9)
    parser.add_argument("--vy", type=float, default=0.0)
    parser.add_argument("--omega", type=float, default=0.0)
    # The reference initializes history as [current, 0, 0, 0, 0, 0].  Do not
    # insert a zero-command policy phase unless it is explicitly requested.
    parser.add_argument("--warmup-seconds", type=float, default=0.0)
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--physics-dt", type=float, choices=(0.001, 0.002, 0.005), default=PHYSICS_DT)
    parser.add_argument(
        "--integrator",
        choices=("euler", "implicit", "implicitfast", "rk4"),
        default="euler",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--no-real-time", action="store_true")
    parser.add_argument("--print-every", type=int, default=50)
    parser.add_argument("--trace", type=Path, default=None, help="Write one JSON object per policy step.")
    parser.add_argument(
        "--contact-friction",
        type=float,
        nargs=3,
        metavar=("SLIDING", "TORSIONAL", "ROLLING"),
        default=None,
        help="Override every collidable robot/terrain geom friction triple for sim2sim diagnosis.",
    )
    parser.add_argument(
        "--contact-condim",
        type=int,
        choices=(1, 3, 4, 6),
        default=None,
        help="Override collidable geom contact dimensionality; rolling friction requires condim=6.",
    )
    parser.add_argument(
        "--wheel-velocity-observation",
        choices=("auto", "zero", "measured"),
        default="auto",
        help="Use checkpoint metadata by default, or override the wheel-speed observation contract.",
    )
    parser.add_argument(
        "--startup-ramp-steps",
        type=int,
        default=0,
        help="Ramp emitted action and gains over this many 50 Hz policy steps.",
    )
    return parser


def _stairs_height(level: int) -> float:
    if not 0 <= level <= 9:
        raise ValueError("terrain level must be in [0, 9]")
    return 0.05 + (0.23 - 0.05) * level / 9.0


def _build_model(args: argparse.Namespace) -> tuple[mujoco.MjModel, float | None]:
    xml_path = args.xml.expanduser().resolve()
    if not xml_path.is_file():
        raise FileNotFoundError(xml_path)
    if args.terrain == "flat":
        model = mujoco.MjModel.from_xml_path(str(xml_path))
        step_height = None
    else:
        step_height = args.step_height if args.step_height is not None else _stairs_height(args.level)
        if step_height <= 0.0 or args.stair_count < 1 or args.stair_width <= 0.0:
            raise ValueError("stair dimensions must be positive")
        spec = mujoco.MjSpec.from_file(str(xml_path))
        # Match IsaacLab's MeshInvertedPyramidStairsTerrainCfg exactly along
        # the forward centerline.  Its ``+1`` step-count rule makes the final
        # center platform narrower than the nominal platform_width: for the
        # training values this is 8 - 2*9*.30 = 2.6 m, so the first riser is
        # at x=1.3 m rather than x=1.5 m.
        center_platform_width = args.terrain_size - 2.0 * args.stair_count * args.stair_width
        if center_platform_width <= 0.0:
            raise ValueError("stair count/width leave no center platform inside the terrain patch")
        start_x = center_platform_width * 0.5
        half_width_y = 1.5
        terrain = spec.worldbody.add_body(name="sim2sim_stairs")
        common = {
            "type": mujoco.mjtGeom.mjGEOM_BOX,
            "group": 0,
            "contype": 1,
            "conaffinity": 1,
            "priority": 1,
            "condim": 3,
            "friction": (1.0, 0.01, 0.01),
            "rgba": (0.42, 0.45, 0.50, 1.0),
        }
        for index in range(args.stair_count):
            top = step_height * (index + 1)
            center_x = start_x + args.stair_width * (index + 0.5)
            terrain.add_geom(
                name=f"sim2sim_stair_{index:02d}",
                pos=(center_x, 0.0, top * 0.5),
                size=(args.stair_width * 0.5, half_width_y, top * 0.5),
                **common,
            )
        total_height = step_height * args.stair_count
        platform_start = start_x + args.stair_count * args.stair_width
        platform_length = 100.0
        terrain.add_geom(
            name="sim2sim_stair_top",
            pos=(platform_start + platform_length * 0.5, 0.0, total_height * 0.5),
            size=(platform_length * 0.5, half_width_y, total_height * 0.5),
            **common,
        )
        model = spec.compile()
    model.opt.timestep = args.physics_dt
    model.opt.integrator = {
        "euler": mujoco.mjtIntegrator.mjINT_EULER,
        "implicit": mujoco.mjtIntegrator.mjINT_IMPLICIT,
        "implicitfast": mujoco.mjtIntegrator.mjINT_IMPLICITFAST,
        "rk4": mujoco.mjtIntegrator.mjINT_RK4,
    }[args.integrator]
    collidable = (model.geom_contype != 0) | (model.geom_conaffinity != 0)
    if args.contact_friction is not None:
        friction = np.asarray(args.contact_friction, dtype=np.float64)
        if friction.shape != (3,) or np.any(friction < 0.0):
            raise ValueError("contact friction values must be three non-negative numbers")
        model.geom_friction[collidable] = friction
    if args.contact_condim is not None:
        model.geom_condim[collidable] = args.contact_condim
    return model, step_height


def _load_policy(checkpoint_path: Path, device: torch.device) -> tuple[PIMHIMActorCritic, dict]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if int(checkpoint.get("history_length", -1)) != HISTORY_LENGTH:
        raise RuntimeError(f"checkpoint history mismatch: {checkpoint.get('history_length')}")
    if int(checkpoint.get("one_step_dim", -1)) != ONE_STEP_DIM:
        raise RuntimeError(f"checkpoint observation mismatch: {checkpoint.get('one_step_dim')}")
    if checkpoint.get("history_order") != "newest_first":
        raise RuntimeError(f"checkpoint history order mismatch: {checkpoint.get('history_order')}")
    if checkpoint.get("policy_variant") != "blind":
        raise RuntimeError(f"checkpoint policy variant mismatch: {checkpoint.get('policy_variant')}")

    estimator = PIMHIMCnnEstimator(
        history_length=HISTORY_LENGTH,
        one_step_dim=ONE_STEP_DIM,
        proprio_dim=ONE_STEP_DIM,
        lidar_start=ONE_STEP_DIM,
        image_channels=6,
        image_height=12,
        image_width=16,
        target_slices=((3, 60),),
        velocity_start=ONE_STEP_DIM,
        use_lidar=False,
    )
    actor_critic = PIMHIMActorCritic(
        history_dim=HISTORY_LENGTH * ONE_STEP_DIM,
        critic_dim=262,
        one_step_dim=ONE_STEP_DIM,
        action_dim=ACTION_DIM,
        estimator=estimator,
        initial_noise_std=checkpoint.get("initial_noise_std", 1.0),
        actor_observation_mode="proprio",
    ).to(device)
    actor_critic.load_state_dict(checkpoint["model_state_dict"], strict=True)
    actor_critic.eval()
    return actor_critic, checkpoint


def _rotation_matrix_wxyz(quaternion: np.ndarray) -> np.ndarray:
    w, x, y, z = quaternion
    return np.asarray(
        (
            (1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)),
            (2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)),
            (2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)),
        ),
        dtype=np.float64,
    )


class S10MujocoHIMController:
    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        policy: PIMHIMActorCritic,
        device: torch.device,
        *,
        wheel_velocity_observation: str,
        startup_ramp_steps: int,
    ) -> None:
        self.model = model
        self.data = data
        self.policy = policy
        self.device = device
        self.wheel_velocity_observation = wheel_velocity_observation
        self.startup_ramp_steps = int(startup_ramp_steps)
        if self.startup_ramp_steps < 0:
            raise ValueError("startup ramp steps must be non-negative")
        self.qpos_addresses = np.asarray(
            [model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)] for name in ROBOT_JOINT_NAMES]
        )
        self.qvel_addresses = np.asarray(
            [model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)] for name in ROBOT_JOINT_NAMES]
        )
        self.base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
        self.quat_sensor_address = int(model.sensor_adr[0])
        self.gyro_sensor_address = int(model.sensor_adr[2])
        self.history = torch.zeros(
            HISTORY_LENGTH, ONE_STEP_DIM, dtype=torch.float32, device=device
        )
        self.last_action = torch.zeros(ACTION_DIM, dtype=torch.float32, device=device)
        self.latest_emitted_action = np.zeros(ACTION_DIM, dtype=np.float64)
        self.run_count = 0
        self.kp_scale = 1.0
        self.kd_scale = 1.0
        self.latest_leg_target = DEFAULT_POLICY_POSITION[:LEG_COUNT].copy()
        self.latest_wheel_target = np.zeros(WHEEL_COUNT, dtype=np.float64)
        self.latest_observation = np.zeros(ONE_STEP_DIM, dtype=np.float32)
        self.latest_estimated_velocity = np.zeros(3, dtype=np.float32)
        self.latest_latent = np.zeros(16, dtype=np.float32)
        self.latest_effort_policy = np.zeros(ACTION_DIM, dtype=np.float64)

    def reset(self) -> None:
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[0:3] = (0.0, 0.0, 0.50)
        self.data.qpos[3:7] = (1.0, 0.0, 0.0, 0.0)
        robot_default = DEFAULT_POLICY_POSITION[np.asarray(ROBOT_TO_POLICY_INDICES)]
        self.data.qpos[self.qpos_addresses] = robot_default
        self.data.qvel[:] = 0.0
        self.history.zero_()
        self.last_action.zero_()
        self.latest_emitted_action.fill(0.0)
        self.run_count = 0
        self.kp_scale = 1.0
        self.kd_scale = 1.0
        self.latest_leg_target = DEFAULT_POLICY_POSITION[:LEG_COUNT].copy()
        self.latest_wheel_target.fill(0.0)
        self.latest_observation.fill(0.0)
        self.latest_estimated_velocity.fill(0.0)
        self.latest_latent.fill(0.0)
        self.latest_effort_policy.fill(0.0)
        mujoco.mj_forward(self.model, self.data)

    def _robot_state_policy_order(self) -> tuple[np.ndarray, np.ndarray]:
        q_robot = self.data.qpos[self.qpos_addresses]
        dq_robot = self.data.qvel[self.qvel_addresses]
        return (
            q_robot[np.asarray(POLICY_TO_ROBOT_INDICES)],
            dq_robot[np.asarray(POLICY_TO_ROBOT_INDICES)],
        )

    def observation(self, command: np.ndarray) -> torch.Tensor:
        q_policy, dq_policy = self._robot_state_policy_order()
        quaternion = self.data.sensordata[self.quat_sensor_address : self.quat_sensor_address + 4]
        gyro = self.data.sensordata[self.gyro_sensor_address : self.gyro_sensor_address + 3]
        projected_gravity = _rotation_matrix_wxyz(quaternion).T @ np.asarray((0.0, 0.0, -1.0))
        joint_position = q_policy - DEFAULT_POLICY_POSITION
        joint_velocity = dq_policy.copy()
        joint_position[-WHEEL_COUNT:] = 0.0
        if self.wheel_velocity_observation == "zero":
            joint_velocity[-WHEEL_COUNT:] = 0.0
        frame = np.concatenate(
            (
                gyro * 0.25,
                projected_gravity,
                command * np.asarray((2.0, 2.0, 0.25)),
                joint_position,
                joint_velocity * 0.05,
                self.last_action.detach().cpu().numpy(),
            )
        ).astype(np.float32)
        if frame.shape != (ONE_STEP_DIM,) or not np.isfinite(frame).all():
            raise RuntimeError(f"invalid MuJoCo observation frame: shape={frame.shape}")
        return torch.from_numpy(np.clip(frame, -100.0, 100.0)).to(self.device)

    @torch.inference_mode()
    def policy_step(self, command: np.ndarray) -> np.ndarray:
        current = self.observation(command)
        self.latest_observation = current.detach().cpu().numpy().copy()
        self.history[1:] = self.history[:-1].clone()
        self.history[0] = current
        history_flat = self.history.flatten().unsqueeze(0)
        estimated_velocity, latent = self.policy.estimator.encode(history_flat)
        action = self.policy.actor(
            self.policy._actor_input(history_flat, estimated_velocity, latent)
        ).squeeze(0)
        if action.shape != (ACTION_DIM,) or not torch.isfinite(action).all():
            raise RuntimeError("policy produced invalid action")
        self.last_action.copy_(action)
        self.latest_estimated_velocity = estimated_velocity[0].detach().cpu().numpy().copy()
        self.latest_latent = latent[0].detach().cpu().numpy().copy()
        action_np = action.detach().cpu().numpy().astype(np.float64)
        emitted_action = action_np.copy()
        if self.startup_ramp_steps > 0 and self.run_count < self.startup_ramp_steps:
            ratio = self.run_count / float(self.startup_ramp_steps)
            emitted_action *= ratio
            self.kp_scale = 0.2 + 0.8 * ratio
            self.kd_scale = 1.5
        else:
            self.kp_scale = 1.0
            self.kd_scale = 1.0
        self.run_count += 1
        self.latest_emitted_action = emitted_action
        self.latest_leg_target = (
            DEFAULT_POLICY_POSITION[:LEG_COUNT] + emitted_action[:LEG_COUNT] * LEG_ACTION_SCALE
        )
        self.latest_wheel_target = np.clip(
            emitted_action[LEG_COUNT:] * WHEEL_ACTION_SCALE,
            -WHEEL_VELOCITY_LIMIT,
            WHEEL_VELOCITY_LIMIT,
        )
        return action_np

    def physics_step(self) -> None:
        q_policy, dq_policy = self._robot_state_policy_order()
        effort_policy = np.empty(ACTION_DIM, dtype=np.float64)
        effort_policy[:LEG_COUNT] = np.clip(
            (LEG_KP * self.kp_scale) * (self.latest_leg_target - q_policy[:LEG_COUNT])
            - (LEG_KD * self.kd_scale) * dq_policy[:LEG_COUNT],
            -LEG_EFFORT_LIMIT,
            LEG_EFFORT_LIMIT,
        )
        effort_policy[LEG_COUNT:] = np.clip(
            (WHEEL_KD * self.kd_scale) * (self.latest_wheel_target - dq_policy[LEG_COUNT:]),
            -WHEEL_EFFORT_LIMIT,
            WHEEL_EFFORT_LIMIT,
        )
        self.latest_effort_policy = effort_policy.copy()
        self.data.ctrl[:] = effort_policy[np.asarray(ROBOT_TO_POLICY_INDICES)]
        mujoco.mj_step(self.model, self.data)

    def settle(self, seconds: float = 1.0) -> None:
        count = int(round(seconds / float(self.model.opt.timestep)))
        for _ in range(count):
            self.physics_step()

    def metrics(self) -> dict[str, float]:
        quaternion = self.data.qpos[3:7]
        rotation = _rotation_matrix_wxyz(quaternion)
        linear_velocity_b = rotation.T @ self.data.qvel[0:3]
        roll = math.atan2(rotation[2, 1], rotation[2, 2])
        pitch = math.atan2(-rotation[2, 0], math.hypot(rotation[2, 1], rotation[2, 2]))
        return {
            "x": float(self.data.qpos[0]),
            "y": float(self.data.qpos[1]),
            "z": float(self.data.qpos[2]),
            "vx_body": float(linear_velocity_b[0]),
            "vy_body": float(linear_velocity_b[1]),
            "omega_body": float(self.data.qvel[5]),
            "roll_deg": math.degrees(roll),
            "pitch_deg": math.degrees(pitch),
            "action_max": float(self.last_action.abs().max()),
        }


def _run(args: argparse.Namespace) -> None:
    physics_steps_per_policy = int(round(POLICY_DT / args.physics_dt))
    if abs(physics_steps_per_policy * args.physics_dt - POLICY_DT) > 1.0e-12:
        raise RuntimeError("MuJoCo substeps must exactly cover one 50 Hz policy interval")
    device = torch.device(args.device)
    model, step_height = _build_model(args)
    report = audit_mujoco_s10_protocol(model, mujoco)
    data = mujoco.MjData(model)
    policy, checkpoint = _load_policy(args.checkpoint.expanduser().resolve(), device)
    checkpoint_wheel_observation = checkpoint.get("wheel_velocity_observation", "zero-signal")
    if args.wheel_velocity_observation == "auto":
        wheel_velocity_observation = (
            "measured" if checkpoint_wheel_observation == "measured" else "zero"
        )
    else:
        wheel_velocity_observation = args.wheel_velocity_observation
    if checkpoint.get("actuator_integration") not in (None, "explicit"):
        print(
            "[WARN] checkpoint was trained with implicit actuator integration; "
            "MuJoCo evaluation is a transfer test, not a matched-protocol replay",
            flush=True,
        )
    controller = S10MujocoHIMController(
        model,
        data,
        policy,
        device,
        wheel_velocity_observation=wheel_velocity_observation,
        startup_ramp_steps=args.startup_ramp_steps,
    )
    controller.reset()
    controller.settle(1.0)

    zero_command = np.zeros(3, dtype=np.float64)
    command = np.asarray((args.vx, args.vy, args.omega), dtype=np.float64)
    total_policy_steps = int(round(args.duration / POLICY_DT))
    warmup_policy_steps = int(round(args.warmup_seconds / POLICY_DT))
    start_position = data.qpos[0:3].copy()
    sums = {name: 0.0 for name in ("vx_body", "vy_body", "omega_body", "roll_deg", "pitch_deg")}
    min_height = float("inf")
    max_abs_y = 0.0
    fell = False
    completed_policy_steps = 0
    trace_path = args.trace.expanduser().resolve() if args.trace is not None else None
    if trace_path is not None:
        trace_path.parent.mkdir(parents=True, exist_ok=True)
    trace_file = trace_path.open("w", encoding="utf-8") if trace_path is not None else None

    print(
        "SIM2SIM_START "
        + json.dumps(
            {
                "checkpoint_iteration": int(checkpoint.get("iteration", -1)),
                "terrain": args.terrain,
                "terrain_level": args.level if args.terrain == "stairs" else None,
                "step_height_m": step_height,
                "stair_count": args.stair_count if args.terrain == "stairs" else None,
                "first_riser_x_m": (
                    (args.terrain_size - 2.0 * args.stair_count * args.stair_width) * 0.5
                    if args.terrain == "stairs"
                    else None
                ),
                "command": command.tolist(),
                "physics_dt": args.physics_dt,
                "policy_dt": POLICY_DT,
                "integrator": args.integrator,
                "joint_order": list(report["robot_joint_order"]),
                "device": str(device),
                "contact_friction_override": args.contact_friction,
                "contact_condim_override": args.contact_condim,
                "wheel_velocity_observation": wheel_velocity_observation,
                "checkpoint_wheel_velocity_observation": checkpoint_wheel_observation,
                "checkpoint_actuator_integration": checkpoint.get("actuator_integration"),
                "checkpoint_training_physics_dt": checkpoint.get("training_physics_dt"),
                "checkpoint_training_decimation": checkpoint.get("training_decimation"),
                "startup_ramp_steps": args.startup_ramp_steps,
            },
            sort_keys=True,
        ),
        flush=True,
    )

    def advance(viewer=None) -> None:
        nonlocal min_height, max_abs_y, fell, completed_policy_steps
        for index in range(total_policy_steps + warmup_policy_steps):
            if viewer is not None and not viewer.is_running():
                break
            started = time.perf_counter()
            active_command = zero_command if index < warmup_policy_steps else command
            controller.policy_step(active_command)
            for _ in range(physics_steps_per_policy):
                controller.physics_step()
            metrics = controller.metrics()
            shown_step = index - warmup_policy_steps + 1
            if index >= warmup_policy_steps:
                completed_policy_steps += 1
                for name in sums:
                    sums[name] += metrics[name] if "deg" not in name else abs(metrics[name])
                min_height = min(min_height, metrics["z"])
                max_abs_y = max(max_abs_y, abs(metrics["y"] - start_position[1]))
                if metrics["z"] < 0.20 or abs(metrics["roll_deg"]) > 70.0 or abs(metrics["pitch_deg"]) > 70.0:
                    fell = True
                if trace_file is not None:
                    q_policy, dq_policy = controller._robot_state_policy_order()
                    quaternion = data.qpos[3:7]
                    rotation = _rotation_matrix_wxyz(quaternion)
                    record = {
                        "simulator": "mujoco",
                        "step": shown_step,
                        "time_s": shown_step * POLICY_DT,
                        "dx": float(data.qpos[0] - start_position[0]),
                        "dy": float(data.qpos[1] - start_position[1]),
                        "base_height": float(data.qpos[2]),
                        "true_base_lin_vel_b": (rotation.T @ data.qvel[0:3]).tolist(),
                        "true_base_ang_vel_b": data.sensordata[
                            controller.gyro_sensor_address : controller.gyro_sensor_address + 3
                        ].tolist(),
                        "root_quat_wxyz": quaternion.tolist(),
                        "joint_pos": q_policy.tolist(),
                        "joint_vel": dq_policy.tolist(),
                        "applied_torque": controller.latest_effort_policy.tolist(),
                        "done": bool(fell),
                        "observation": controller.latest_observation.tolist(),
                        "estimated_velocity": controller.latest_estimated_velocity.tolist(),
                        "latent": controller.latest_latent.tolist(),
                        "action": controller.last_action.detach().cpu().tolist(),
                        "emitted_action": controller.latest_emitted_action.tolist(),
                        "leg_position_target": controller.latest_leg_target.tolist(),
                        "wheel_velocity_target": controller.latest_wheel_target.tolist(),
                    }
                    trace_file.write(json.dumps(record, separators=(",", ":")) + "\n")
            if index >= warmup_policy_steps and args.print_every > 0 and shown_step % args.print_every == 0:
                print(f"[MUJOCO {shown_step:05d}] " + " ".join(f"{k}={v:+.3f}" for k, v in metrics.items()), flush=True)
            if viewer is not None:
                viewer.cam.lookat[:] = data.xpos[controller.base_id]
                viewer.sync()
            if not args.no_real_time:
                time.sleep(max(0.0, POLICY_DT - (time.perf_counter() - started)))

    try:
        if args.headless:
            advance()
        else:
            from mujoco import viewer as mj_viewer

            with mj_viewer.launch_passive(model, data) as viewer:
                viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
                viewer.cam.trackbodyid = controller.base_id
                viewer.cam.distance = 3.0
                viewer.cam.elevation = -18.0
                advance(viewer)
    finally:
        if trace_file is not None:
            trace_file.close()

    denominator = max(completed_policy_steps, 1)
    final = controller.metrics()
    summary = {
        "checkpoint_iteration": int(checkpoint.get("iteration", -1)),
        "terrain": args.terrain,
        "step_height_m": step_height,
        "duration_s": completed_policy_steps * POLICY_DT,
        "fell": fell,
        "dx": float(data.qpos[0] - start_position[0]),
        "dy": float(data.qpos[1] - start_position[1]),
        "mean_vx_body": sums["vx_body"] / denominator,
        "mean_vy_body": sums["vy_body"] / denominator,
        "mean_omega_body": sums["omega_body"] / denominator,
        "mean_abs_roll_deg": sums["roll_deg"] / denominator,
        "mean_abs_pitch_deg": sums["pitch_deg"] / denominator,
        "min_base_height": min_height,
        "max_abs_lateral_displacement": max_abs_y,
        "final": final,
    }
    print("SIM2SIM_SUMMARY " + json.dumps(summary, sort_keys=True), flush=True)


def main() -> None:
    _run(_parser().parse_args())


if __name__ == "__main__":
    main()
