"""MDP terms ported from TrackinBIT/HIMLoco-for-Go2W.

Reference commit: 011693738c61603c3f22f2bce755098dd36fa7eb.
Only simulator/API translations live here; reward and command semantics follow
the reference implementation.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch
from isaaclab.actuators import ImplicitActuator
from isaaclab.assets import Articulation
from isaaclab.envs.mdp.commands import UniformVelocityCommand, UniformVelocityCommandCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor, RayCaster
from isaaclab.utils import configclass
import isaaclab.utils.math as math_utils

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv


class Go2WVelocityCommand(UniformVelocityCommand):
    """Exact Go2W command mixture, including the high-speed 20% partition."""

    cfg: "Go2WVelocityCommandCfg"

    def _resample_command(self, env_ids: Sequence[int]):
        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        if env_ids.numel() == 0:
            return
        super()._resample_command(env_ids)

        # The reference always samples the ordinary partition from [-1, 1],
        # independent of the command curriculum's expanding high-speed range.
        self.vel_command_b[env_ids, 0].uniform_(-1.0, 1.0)

        high_ids = env_ids[env_ids < int(self.num_envs * self.cfg.high_speed_fraction)]
        if high_ids.numel() > 0:
            self.vel_command_b[high_ids, 0].uniform_(*self.cfg.ranges.lin_vel_x)
            keep_lateral = torch.abs(self.vel_command_b[high_ids, 0]) < 1.0
            self.vel_command_b[high_ids, 1] *= keep_lateral

        planar_norm = torch.linalg.norm(self.vel_command_b[env_ids, :2], dim=1)
        self.vel_command_b[env_ids, :2] *= (planar_norm > 0.2).unsqueeze(1)

    def _update_command(self):
        # legged_gym hard-codes heading-derived yaw to [-2, 2], independent
        # of the sampled ang_vel_yaw range in the configuration.
        if self.cfg.heading_command:
            env_ids = self.is_heading_env.nonzero(as_tuple=False).flatten()
            heading_error = math_utils.wrap_to_pi(
                self.heading_target[env_ids] - self.robot.data.heading_w[env_ids]
            )
            self.vel_command_b[env_ids, 2] = torch.clip(
                self.cfg.heading_control_stiffness * heading_error,
                min=-2.0,
                max=2.0,
            )
        standing_env_ids = self.is_standing_env.nonzero(as_tuple=False).flatten()
        self.vel_command_b[standing_env_ids, :] = 0.0


@configclass
class Go2WVelocityCommandCfg(UniformVelocityCommandCfg):
    class_type: type = Go2WVelocityCommand
    high_speed_fraction: float = 0.2


def run_still(
    env: "ManagerBasedRLEnv",
    command_name: str,
    asset_cfg: SceneEntityCfg,
    command_threshold: float = 0.1,
) -> torch.Tensor:
    """Reference ``run_still``: L1 default-pose error while moving."""

    asset: Articulation = env.scene[asset_cfg.name]
    error = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    moving = torch.linalg.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > command_threshold
    return torch.sum(torch.abs(error), dim=1) * moving


def scaled_velocity_commands(
    env: "ManagerBasedEnv",
    command_name: str,
) -> torch.Tensor:
    """Go2W observation scaling [2, 2, 0.25] before history packing."""

    command = env.command_manager.get_command(command_name)
    scale = command.new_tensor((2.0, 2.0, 0.25))
    return command * scale


def collision_count(
    env: "ManagerBasedRLEnv",
    sensor_cfg: SceneEntityCfg,
    threshold: float = 0.1,
) -> torch.Tensor:
    """Count penalized bodies whose net contact force exceeds the threshold."""

    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = sensor.data.net_forces_w[:, sensor_cfg.body_ids]
    return (torch.linalg.norm(forces, dim=-1) > threshold).float().sum(dim=1)


def feet_stumble(
    env: "ManagerBasedRLEnv",
    sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Reference wheel/foot stumble test: horizontal force > 3*vertical."""

    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = sensor.data.net_forces_w[:, sensor_cfg.body_ids]
    return torch.any(torch.linalg.norm(forces[..., :2], dim=-1) > 3.0 * torch.abs(forces[..., 2]), dim=1)


def height_scan_reference(
    env: "ManagerBasedEnv",
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reference critic scan: clip(base_z - 0.5 - terrain_z, -1, 1)."""

    asset: Articulation = env.scene[asset_cfg.name]
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
    hits = sensor.data.ray_hits_w[..., 2]
    fallback = asset.data.root_pos_w[:, 2].unsqueeze(1) - 0.5
    hits = torch.where(torch.isfinite(hits), hits, fallback)
    return torch.clamp(asset.data.root_pos_w[:, 2].unsqueeze(1) - 0.5 - hits, -1.0, 1.0)


def privileged_disturbance(env: "ManagerBasedEnv") -> torch.Tensor:
    buffer = getattr(env, "_go2w_disturbance", None)
    if buffer is None:
        return torch.zeros((env.num_envs, 3), device=env.device)
    return buffer


def normalized_wheel_contact_forces(
    env: "ManagerBasedEnv",
    sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Normalize [0,50] N to [-1,1], exactly as the Go2W critic."""

    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = sensor.data.net_forces_w[:, sensor_cfg.body_ids].reshape(env.num_envs, -1)
    return (forces - 25.0) * 0.04


def dof_acc_reference(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg,
    wheel_count: int = 4,
) -> torch.Tensor:
    """Reproduce Go2W's policy-rate acceleration term, including wheel semantics.

    The reference differences joint velocity once per 20 ms policy step, not
    once per 5 ms physics step.  Because its later ``dof_vel`` reward clears
    wheel velocity in-place, the previous wheel velocity stored for the next
    step is zero.  Consequently the wheel part penalizes ``wheel_vel / 0.02``.
    This function implements the observed execution semantics without mutating
    IsaacLab articulation state.
    """

    asset: Articulation = env.scene[asset_cfg.name]
    current = asset.data.joint_vel[:, asset_cfg.joint_ids]
    key = "_go2w_last_reward_joint_vel"
    previous = getattr(env, key, None)
    if previous is None or previous.shape != current.shape:
        previous = torch.zeros_like(current)
        setattr(env, key, previous)

    # IsaacGym clears this buffer on reset.  At the first post-reset reward
    # evaluation IsaacLab's episode counter is one.
    effective_previous = previous.clone()
    first_step = env.episode_length_buf <= 1
    if torch.any(first_step):
        effective_previous[first_step] = 0.0

    acceleration = (effective_previous - current) / env.step_dt
    previous.copy_(current)
    if wheel_count:
        previous[:, -wheel_count:] = 0.0
    return torch.sum(torch.square(acceleration), dim=1)


def pulsed_disturbance(
    env: "ManagerBasedEnv",
    env_ids: torch.Tensor | None,
    force_range: tuple[float, float],
    interval_steps: int,
    asset_cfg: SceneEntityCfg,
) -> None:
    """One-control-step force pulse every N steps, matching Isaac Gym forces."""

    asset: Articulation = env.scene[asset_cfg.name]
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    forces = torch.zeros((len(env_ids), 1, 3), device=env.device)
    if env.common_step_counter % interval_steps == 0:
        forces.uniform_(*force_range)
    torques = torch.zeros_like(forces)
    asset.set_external_force_and_torque(
        forces, torques, env_ids=env_ids, body_ids=asset_cfg.body_ids
    )
    if not hasattr(env, "_go2w_disturbance"):
        env._go2w_disturbance = torch.zeros((env.num_envs, 3), device=env.device)
    env._go2w_disturbance[env_ids] = forces[:, 0]


def overwrite_root_velocity_push(
    env: "ManagerBasedEnv",
    env_ids: torch.Tensor | None,
    max_vel_xy: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
    """Reference push overwrites, rather than increments, world XY velocity."""

    asset: Articulation = env.scene[asset_cfg.name]
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    velocity = asset.data.root_vel_w[env_ids].clone()
    velocity[:, :2].uniform_(-max_vel_xy, max_vel_xy)
    asset.write_root_velocity_to_sim(velocity, env_ids=env_ids)


def randomize_go2w_actuator_gains(
    env: "ManagerBasedEnv",
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    kp_range: tuple[float, float] = (0.9, 1.1),
    kd_range: tuple[float, float] = (0.9, 1.1),
    motor_strength_range: tuple[float, float] = (0.9, 1.1),
) -> None:
    """Reference per-environment Kp, Kd and common motor-strength factors.

    legged_gym samples one scalar for each factor and applies it to every
    actuator joint in that environment.  Scaling both stiffness and damping by
    the motor factor is equivalent to scaling the summed PD torque before
    clipping.  Implicit actuators additionally require writing the gains into
    PhysX; explicit actuators consume the updated tensors directly.
    """

    asset: Articulation = env.scene[asset_cfg.name]
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=env.device)
    count = len(env_ids)
    kp = torch.empty((count, 1), device=env.device).uniform_(*kp_range)
    kd = torch.empty((count, 1), device=env.device).uniform_(*kd_range)
    motor = torch.empty((count, 1), device=env.device).uniform_(*motor_strength_range)

    for actuator in asset.actuators.values():
        joint_ids = actuator.joint_indices
        stiffness = asset.data.default_joint_stiffness[env_ids][:, joint_ids] * kp * motor
        damping = asset.data.default_joint_damping[env_ids][:, joint_ids] * kd * motor
        actuator.stiffness[env_ids] = stiffness
        actuator.damping[env_ids] = damping
        if isinstance(actuator, ImplicitActuator):
            asset.write_joint_stiffness_to_sim(stiffness, joint_ids=joint_ids, env_ids=env_ids)
            asset.write_joint_damping_to_sim(damping, joint_ids=joint_ids, env_ids=env_ids)


def randomize_go2w_friction(
    env: "ManagerBasedEnv",
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    friction_range: tuple[float, float] = (0.25, 1.25),
) -> None:
    """Assign one shared friction coefficient to every shape of each robot."""

    asset: Articulation = env.scene[asset_cfg.name]
    if env_ids is None:
        env_ids_cpu = torch.arange(env.num_envs, device="cpu")
    else:
        env_ids_cpu = torch.as_tensor(env_ids, dtype=torch.long, device="cpu")
    materials = asset.root_physx_view.get_material_properties()
    friction = torch.empty((len(env_ids_cpu), 1), device="cpu").uniform_(*friction_range)
    materials[env_ids_cpu, :, 0] = friction
    materials[env_ids_cpu, :, 1] = friction
    materials[env_ids_cpu, :, 2] = 0.0
    asset.root_physx_view.set_material_properties(materials, env_ids_cpu)


def terrain_levels_reference(
    env: "ManagerBasedRLEnv",
    env_ids: Sequence[int],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Original legged_gym distance curriculum, without success labels."""

    asset: Articulation = env.scene[asset_cfg.name]
    terrain = env.scene.terrain
    command = env.command_manager.get_command("base_velocity")
    distance = torch.linalg.norm(asset.data.root_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2], dim=1)
    move_up = distance > terrain.cfg.terrain_generator.size[0] / 2.0
    move_down = distance < torch.linalg.norm(command[env_ids, :2], dim=1) * env.max_episode_length_s * 0.5
    move_down &= ~move_up
    terrain.update_env_origins(env_ids, move_up, move_down)
    return terrain.terrain_levels.float().mean()


def command_range_reference(
    env: "ManagerBasedRLEnv",
    env_ids: Sequence[int],
    reward_term_name: str,
    max_curriculum: float = 1.5,
) -> torch.Tensor:
    """Global Go2W x-command curriculum, evaluated once per full episode."""

    term = env.command_manager.get_term("base_velocity")
    last_step = getattr(term, "_go2w_last_curriculum_step", -1)
    if env.common_step_counter == 0 or env.common_step_counter % env.max_episode_length != 0:
        return torch.tensor(term.cfg.ranges.lin_vel_x[1], device=env.device)
    if last_step == env.common_step_counter:
        return torch.tensor(term.cfg.ranges.lin_vel_x[1], device=env.device)
    term._go2w_last_curriculum_step = env.common_step_counter

    env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=env.device)
    low = env_ids[env_ids > int(env.num_envs * 0.2)]
    high = env_ids[env_ids < int(env.num_envs * 0.2)]
    if low.numel() == 0 or high.numel() == 0:
        return torch.tensor(term.cfg.ranges.lin_vel_x[1], device=env.device)
    sums = env.reward_manager._episode_sums[reward_term_name]
    cfg = env.reward_manager.get_term_cfg(reward_term_name)
    threshold = 0.8 * cfg.weight * env.step_dt
    low_score = sums[low].mean() / env.max_episode_length
    high_score = sums[high].mean() / env.max_episode_length
    if low_score > threshold and high_score > threshold:
        lo, hi = term.cfg.ranges.lin_vel_x
        term.cfg.ranges.lin_vel_x = (max(lo - 0.2, -max_curriculum), min(hi + 0.2, max_curriculum))
    return torch.tensor(term.cfg.ranges.lin_vel_x[1], device=env.device)
