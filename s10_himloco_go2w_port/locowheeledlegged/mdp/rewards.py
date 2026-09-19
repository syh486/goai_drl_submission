from __future__ import annotations
import math
import torch
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg, ManagerTermBase, RewardTermCfg
from isaaclab.sensors import ContactSensor, RayCaster
from isaaclab.utils.math import quat_from_euler_xyz, quat_apply, quat_rotate_inverse, euler_xyz_from_quat, quat_inv, quat_mul
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def stand_still_without_cmd(
    env: ManagerBasedRLEnv,
    command_name: str,
    command_threshold: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),  # 需要修改 params["asset_cfg"].joint_names = leg_joint_names
    use_gravity_gating: bool = False,
    gating_max: float = 0.7,
) -> torch.Tensor:

    asset: Articulation = env.scene[asset_cfg.name]
    # compute out of limits constraints
    diff_angle = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    reward = torch.sum(torch.abs(diff_angle), dim=1)
    reward *= torch.linalg.norm(env.command_manager.get_command(command_name), dim=1) < command_threshold
    if use_gravity_gating:
        reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, gating_max) / gating_max
    return reward


def hip_deviation_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),   # params["asset_cfg"].joint_names = hip_joint_names
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]

    joint_ids = asset_cfg.joint_ids
    q = asset.data.joint_pos[:, joint_ids]
    q0 = asset.data.default_joint_pos[:, joint_ids]

    return torch.sum(torch.square(q - q0), dim=1)


def joint_deviation_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),   # params["asset_cfg"].joint_names = leg_joint_names
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]

    joint_ids = asset_cfg.joint_ids
    q = asset.data.joint_pos[:, joint_ids]
    q0 = asset.data.default_joint_pos[:, joint_ids]

    return torch.sum(torch.square(q - q0), dim=1)


def hip_action_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),  # params["asset_cfg"].joint_names = hip_joint_names
) -> torch.Tensor:
    """Penalize hip joint actions (L2 squared)."""
    action = env.action_manager.action
    joint_ids = asset_cfg.joint_ids 

    reward = torch.sum(torch.square(action[:, joint_ids]), dim=1)
    return reward


def custom_track_lin_vel_x_exp(
    env: ManagerBasedRLEnv, 
    std: float, 
    command_name: str, 
    gravity_z_power: float | None = None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    # compute the error
    lin_vel_error = torch.square(env.command_manager.get_command(command_name)[:, 0] - asset.data.root_lin_vel_b[:, 0])
    reward = torch.exp(-lin_vel_error / std**2)
    if gravity_z_power is not None:
        reward *= -(env.scene["robot"].data.projected_gravity_b[:, 2]) ** gravity_z_power
    else:
        reward *= -env.scene["robot"].data.projected_gravity_b[:, 2]
    return reward

def custom_track_lin_vel_y_exp(
    env: ManagerBasedRLEnv, 
    std: float, 
    command_name: str, 
    gravity_z_power: float | None = None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    # compute the error
    lin_vel_error = torch.square(env.command_manager.get_command(command_name)[:, 1] - asset.data.root_lin_vel_b[:, 1])
    reward = torch.exp(-lin_vel_error / std**2)
    if gravity_z_power is not None:
        reward *= -(env.scene["robot"].data.projected_gravity_b[:, 2]) ** gravity_z_power
    else:
        reward *= -env.scene["robot"].data.projected_gravity_b[:, 2]
    return reward

def custom_track_ang_vel_z_exp(
    env: ManagerBasedRLEnv, 
    std: float, 
    command_name: str, 
    gravity_z_power: float | None = None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    # compute the error
    ang_vel_error = torch.square(env.command_manager.get_command(command_name)[:, 2] - asset.data.root_ang_vel_b[:, 2])
    reward = torch.exp(-ang_vel_error / std**2)
    if gravity_z_power is not None:
        reward *= -(env.scene["robot"].data.projected_gravity_b[:, 2]) ** gravity_z_power
    else:
        reward *= -env.scene["robot"].data.projected_gravity_b[:, 2]
    return reward


def custom_action_rate_l2_with_clip(
    env: ManagerBasedRLEnv,
    threshold: float = 7.0,
) -> torch.Tensor:


    delta_action = env.action_manager.action - env.action_manager.prev_action
    if torch.max(torch.abs(delta_action)) > threshold:
        print(f"[WARN] custom_action_rate_l2_with_clip: delta_action exceeds threshold {threshold}!")
        delta_action = torch.clamp(delta_action, min=-threshold, max=threshold)
    pen = torch.sum(torch.square(delta_action), dim=1)
    return pen

def custom_base_height_l2(
    env: ManagerBasedRLEnv,
    target_height: float,
    terrain_height_threshold: Tuple[float, float] = (-0.2, 0.2),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    if sensor_cfg is not None:
        sensor: RayCaster = env.scene[sensor_cfg.name]
        # Adjust the target height using the sensor data
        base_ray_hits_w = sensor.data.ray_hits_w[..., 2]
        # Clamp base_ray_hits_w to avoid NaN and Inf (including -Inf/Inf) before usage
        base_ray_hits_w = torch.nan_to_num(base_ray_hits_w, nan=0.0, posinf=terrain_height_threshold[1], neginf=terrain_height_threshold[0])
        base_ray_hits_w = torch.clamp(base_ray_hits_w, min=terrain_height_threshold[0], max=terrain_height_threshold[1])
        adjusted_target_height = target_height + torch.mean(base_ray_hits_w, dim=1)
    else:
        # Use the provided target height directly for flat terrain
        adjusted_target_height = target_height
    # Compute the L2 squared penalty
    return torch.square(asset.data.root_pos_w[:, 2] - adjusted_target_height)


def update_terrain_curriculum_scale(
    env: ManagerBasedRLEnv,
    mean_level: torch.Tensor | float | None = None,
) -> torch.Tensor:
    """Update the cached official-M20 gait scale after terrain curriculum changes."""

    if mean_level is None:
        levels = getattr(env.scene.terrain, "terrain_levels", None)
        if levels is None or levels.numel() == 0:
            value = 1.0
        else:
            mean_level = levels.float().mean()
            value = float(mean_level.item())
    else:
        value = float(mean_level.item()) if isinstance(mean_level, torch.Tensor) else float(mean_level)
    if mean_level is not None:
        if not math.isfinite(value) or value <= 0.0:
            value = 0.0
        elif value < 3.0:
            value = math.exp(value - 3.0)
        else:
            value = 1.0
    scale = torch.full((env.num_envs,), value, device=env.device)
    setattr(env, "_s10_reference_gait_scale", scale)
    setattr(env, "_s10_reference_gait_scale_value", value)
    return scale


def terrain_curriculum_scale(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return the cached M20 gait scale without synchronizing once per reward term."""

    scale = getattr(env, "_s10_reference_gait_scale", None)
    if scale is None or scale.shape != (env.num_envs,) or scale.device != torch.device(env.device):
        scale = update_terrain_curriculum_scale(env)
    return scale


def terrain_scaled_joint_torques_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    penalty = torch.sum(torch.square(asset.data.applied_torque[:, asset_cfg.joint_ids]), dim=1)
    return penalty * terrain_curriculum_scale(env)


def terrain_scaled_action_rate_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    delta = env.action_manager.action - env.action_manager.prev_action
    return torch.sum(torch.square(delta), dim=1) * terrain_curriculum_scale(env)


def terrain_scaled_action_smooth_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Second-order action penalty used by the official M20 task."""

    cache_name = "_s10_reference_prev_prev_action"
    prev_prev = getattr(env, cache_name, None)
    if prev_prev is None or prev_prev.shape != env.action_manager.action.shape:
        prev_prev = torch.zeros_like(env.action_manager.action)
    delta = env.action_manager.action + prev_prev - 2.0 * env.action_manager.prev_action
    valid = (env.action_manager.prev_action != 0.0) & (prev_prev != 0.0)
    penalty = torch.sum(torch.square(delta) * valid, dim=1)
    setattr(env, cache_name, env.action_manager.prev_action.clone())
    return penalty * terrain_curriculum_scale(env)


def base_height_above_terrain_l2(
    env: ManagerBasedRLEnv,
    target_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """Height error using valid world-space ray hits without clipping absolute terrain elevation."""

    asset: RigidObject = env.scene[asset_cfg.name]
    if sensor_cfg is None:
        clearance = asset.data.root_pos_w[:, 2]
    else:
        sensor: RayCaster = env.scene[sensor_cfg.name]
        hits = sensor.data.ray_hits_w[..., 2]
        valid = torch.isfinite(hits) & (torch.abs(hits) < 1.0e6)
        valid_count = valid.sum(dim=1)
        ground_height = torch.where(valid, hits, torch.zeros_like(hits)).sum(dim=1)
        ground_height = ground_height / torch.clamp(valid_count, min=1)
        ground_height = torch.where(valid_count > 0, ground_height, asset.data.root_pos_w[:, 2] - target_height)
        clearance = asset.data.root_pos_w[:, 2] - ground_height
    return torch.square(clearance - target_height)


def joint_pos_penalty_with_terrain_relaxation(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    stand_still_scale: float,
    velocity_threshold: float,
    command_threshold: float,
    sensor_cfg: SceneEntityCfg | None = None,
    terrain_height_threshold: float = 0.06,
    high_terrain_penalty_scale: float = 0.1,
    ang_cmd_threshold: float | None = None,
    y_cmd_threshold: float | None = None,
    xy_norm_max: float = 0.1,
    xz_norm_max: float = 0.1,
) -> torch.Tensor:
    """Default-pose penalty with the official M20 high-terrain relaxation."""

    asset: Articulation = env.scene[asset_cfg.name]
    command_norm = torch.linalg.norm(env.command_manager.get_command(command_name), dim=1)
    body_speed = torch.linalg.norm(asset.data.root_lin_vel_b[:, :2], dim=1)
    deviation = torch.linalg.norm(
        asset.data.joint_pos[:, asset_cfg.joint_ids]
        - asset.data.default_joint_pos[:, asset_cfg.joint_ids],
        dim=1,
    )
    penalty = torch.where(
        (command_norm > command_threshold) | (body_speed > velocity_threshold),
        deviation,
        stand_still_scale * deviation,
    )
    command = env.command_manager.get_command(command_name)
    if ang_cmd_threshold is not None:
        pure_turn = (torch.abs(command[:, 2]) > ang_cmd_threshold) & (
            torch.linalg.norm(command[:, :2], dim=1) < xy_norm_max
        )
        penalty = penalty * (~pure_turn).float()
    if y_cmd_threshold is not None:
        pure_side = (torch.abs(command[:, 1]) > y_cmd_threshold) & (
            torch.linalg.norm(command[:, [0, 2]], dim=1) < xz_norm_max
        )
        penalty = penalty * (~pure_side).float()
    if sensor_cfg is not None:
        sensor: RayCaster = env.scene[sensor_cfg.name]
        hits = sensor.data.ray_hits_w[..., 2]
        valid = torch.isfinite(hits) & (torch.abs(hits) < 1.0e6)
        valid_count = valid.sum(dim=1)
        terrain_height = torch.where(valid, hits, torch.zeros_like(hits)).sum(dim=1)
        terrain_height = terrain_height / torch.clamp(valid_count, min=1)
        terrain_height = terrain_height - env.scene.env_origins[:, 2]
        high_terrain = (valid_count > 0) & (terrain_height > terrain_height_threshold)
        penalty = penalty * torch.where(
            high_terrain,
            torch.full_like(penalty, high_terrain_penalty_scale),
            torch.ones_like(penalty),
        )
    return penalty


def joint_mirror_relative_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    mirror_joints: tuple[tuple[str, str], ...],
    opposite_sign: bool = True,
) -> torch.Tensor:
    """Diagonal symmetry that is exactly zero at S10's asymmetric signed default pose."""

    asset: Articulation = env.scene[asset_cfg.name]
    cache_name = "_s10_reference_mirror_joint_ids"
    cached = getattr(env, cache_name, None)
    if cached is None:
        cached = tuple(
            (asset.find_joints(left)[0], asset.find_joints(right)[0])
            for left, right in mirror_joints
        )
        setattr(env, cache_name, cached)
    penalty = torch.zeros(env.num_envs, device=env.device)
    sign = -1.0 if opposite_sign else 1.0
    for left_ids, right_ids in cached:
        left = asset.data.joint_pos[:, left_ids] - asset.data.default_joint_pos[:, left_ids]
        right = asset.data.joint_pos[:, right_ids] - asset.data.default_joint_pos[:, right_ids]
        penalty += torch.sum(torch.square(left - sign * right), dim=1)
    penalty /= max(len(cached), 1)
    upright = torch.clamp(-asset.data.projected_gravity_b[:, 2], 0.0, 0.7) / 0.7
    return penalty * upright * terrain_curriculum_scale(env)


def contact_forces_over_limit(
    env: ManagerBasedRLEnv,
    threshold: float,
    sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids]
    excess = torch.max(torch.norm(forces, dim=-1), dim=1).values - threshold
    return torch.sum(torch.clamp(excess, min=0.0), dim=1) * terrain_curriculum_scale(env)


def bad_orientation_binary(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]
    gravity = asset.data.projected_gravity_b
    return ((gravity[:, 2] > 0.0) | (gravity[:, :2].abs() > 0.7).any(dim=1)).float()


def feet_air_time_yaw_with_clearance(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
    threshold: float,
    command_threshold: float = 0.1,
    clearance_threshold: float = 0.0,
) -> torch.Tensor:
    """Official M20 yaw air-time term with a per-swing clearance gate."""

    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    asset: RigidObject = env.scene[asset_cfg.name]
    first_contact = sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = sensor.data.last_air_time[:, sensor_cfg.body_ids]
    in_contact = sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0.0
    wheel_height = asset.data.body_pos_w[:, asset_cfg.body_ids, 2] - env.scene.env_origins[:, 2].unsqueeze(1)

    cache_name = "_s10_reference_yaw_air_time_height_state"
    state = getattr(env, cache_name, None)
    if state is None or state["ground_height"].shape != wheel_height.shape:
        state = {"ground_height": wheel_height.clone(), "max_air_height": wheel_height.clone()}
        setattr(env, cache_name, state)
    ground_height = state["ground_height"]
    max_air_height = torch.where(in_contact, state["max_air_height"], torch.maximum(state["max_air_height"], wheel_height))
    cleared = (max_air_height - ground_height) > clearance_threshold
    reward = torch.sum((last_air_time - threshold) * first_contact * cleared, dim=1)
    state["ground_height"] = torch.where(in_contact, wheel_height, ground_height)
    state["max_air_height"] = torch.where(in_contact, wheel_height, max_air_height)
    reward *= torch.abs(env.command_manager.get_command(command_name)[:, 2]) > command_threshold
    return reward * terrain_curriculum_scale(env)


def wheel_slide_yaw_command(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
    linear_command_threshold: float = 0.1,
    yaw_command_threshold: float = 0.1,
) -> torch.Tensor:
    """Penalize wheel-link lateral motion only for near-pure yaw commands."""

    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids].norm(dim=-1).max(dim=1).values > 1.0
    asset: RigidObject = env.scene[asset_cfg.name]
    relative_velocity_w = asset.data.body_lin_vel_w[:, asset_cfg.body_ids] - asset.data.root_lin_vel_w.unsqueeze(1)
    relative_velocity_b = quat_rotate_inverse(
        asset.data.root_quat_w.unsqueeze(1).expand(-1, relative_velocity_w.shape[1], -1).reshape(-1, 4),
        relative_velocity_w.reshape(-1, 3),
    ).reshape_as(relative_velocity_w)
    slide = torch.sum(torch.linalg.norm(relative_velocity_b[..., :2], dim=-1) * contacts, dim=1)
    command = env.command_manager.get_command(command_name)
    gate = (torch.linalg.norm(command[:, :2], dim=1) < linear_command_threshold) & (
        torch.abs(command[:, 2]) > yaw_command_threshold
    )
    return slide * gate.float() * terrain_curriculum_scale(env)


def rotation_gait_status(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
    group_a_body_names: tuple[str, ...],
    group_b_body_names: tuple[str, ...],
    target_height: float = 0.05,
    linear_command_threshold: float = 0.5,
    yaw_command_threshold: float = 0.05,
) -> torch.Tensor:
    """Official M20 diagonal support/lift reward for near-pure rotation."""

    cache_name = "_s10_reference_rotation_gait_ids"
    cached = getattr(env, cache_name, None)
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    asset: RigidObject = env.scene[asset_cfg.name]
    if cached is None:
        cached = {
            "sensor_a": [sensor.find_bodies(name)[0][0] for name in group_a_body_names],
            "sensor_b": [sensor.find_bodies(name)[0][0] for name in group_b_body_names],
            "body_a": [asset.find_bodies(name)[0][0] for name in group_a_body_names],
            "body_b": [asset.find_bodies(name)[0][0] for name in group_b_body_names],
        }
        setattr(env, cache_name, cached)
    contact_time = sensor.data.current_contact_time
    a_grounded = (contact_time[:, cached["sensor_a"]] > 0.0).all(dim=1)
    b_grounded = (contact_time[:, cached["sensor_b"]] > 0.0).all(dim=1)
    z_a = asset.data.body_pos_w[:, cached["body_a"], 2].mean(dim=1)
    z_b = asset.data.body_pos_w[:, cached["body_b"], 2].mean(dim=1)
    pattern = torch.maximum(
        (a_grounded & ((z_b - z_a) > target_height)).float(),
        (b_grounded & ((z_a - z_b) > target_height)).float(),
    )
    command = env.command_manager.get_command(command_name)
    gate = (torch.linalg.norm(command[:, :2], dim=1) < linear_command_threshold) & (
        torch.abs(command[:, 2]) > yaw_command_threshold
    )
    return pattern * gate.float() * terrain_curriculum_scale(env)


class RotationGaitSymmetry(ManagerTermBase):
    """Official M20 rolling contact-duty symmetry over a fixed time window."""

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.command_name = cfg.params["command_name"]
        self.linear_command_threshold = cfg.params["linear_command_threshold"]
        self.yaw_command_threshold = cfg.params["yaw_command_threshold"]
        self.target_duty = cfg.params["target_duty"]
        self.std = cfg.params["std"]
        self.sensor: ContactSensor = env.scene.sensors[cfg.params["sensor_cfg"].name]
        group_a = cfg.params["group_a_body_names"]
        group_b = cfg.params["group_b_body_names"]
        self.group_a_ids = [self.sensor.find_bodies(name)[0][0] for name in group_a]
        self.group_b_ids = [self.sensor.find_bodies(name)[0][0] for name in group_b]
        self.body_ids = self.group_a_ids + self.group_b_ids
        self.buffer_size = max(1, int(cfg.params["window_s"] / env.step_dt))
        self.contact_buffer = torch.zeros(env.num_envs, len(self.body_ids), self.buffer_size, device=env.device)
        self.buffer_index = 0
        self.buffer_filled = False

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is not None and len(env_ids) > 0:
            self.contact_buffer[env_ids] = 0.0

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        command_name: str,
        sensor_cfg: SceneEntityCfg,
        group_a_body_names: tuple[str, ...],
        group_b_body_names: tuple[str, ...],
        target_duty: float,
        std: float,
        linear_command_threshold: float,
        yaw_command_threshold: float,
        window_s: float,
    ) -> torch.Tensor:
        in_contact = (self.sensor.data.current_contact_time[:, self.body_ids] > 0.0).float()
        self.contact_buffer[:, :, self.buffer_index] = in_contact
        self.buffer_index = (self.buffer_index + 1) % self.buffer_size
        if self.buffer_index == 0:
            self.buffer_filled = True
        split = len(self.group_a_ids)
        duty_a = self.contact_buffer[:, :split].mean(dim=(1, 2))
        duty_b = self.contact_buffer[:, split:].mean(dim=(1, 2))
        reward = torch.exp(-torch.square(duty_a - self.target_duty) / self.std**2)
        reward *= torch.exp(-torch.square(duty_b - self.target_duty) / self.std**2)
        if not self.buffer_filled:
            reward *= self.buffer_index / self.buffer_size
        command = env.command_manager.get_command(self.command_name)
        gate = (torch.linalg.norm(command[:, :2], dim=1) < self.linear_command_threshold) & (
            torch.abs(command[:, 2]) > self.yaw_command_threshold
        )
        return reward * gate.float() * terrain_curriculum_scale(env)
