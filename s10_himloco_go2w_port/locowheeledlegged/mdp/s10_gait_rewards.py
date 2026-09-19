"""S10-only gait-correction rewards used by the A/B/C comparison."""

from __future__ import annotations

import torch
from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_rotate_inverse


def _body_frame_wheel_positions(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]
    wheel_delta_w = asset.data.body_pos_w[:, asset_cfg.body_ids] - asset.data.root_pos_w.unsqueeze(1)
    if wheel_delta_w.shape[1] != 4:
        raise ValueError(f"S10 gait rewards require four ordered wheel bodies, got {wheel_delta_w.shape[1]}")
    root_quat = asset.data.root_quat_w.unsqueeze(1).expand(-1, 4, -1)
    return quat_rotate_inverse(
        root_quat.reshape(-1, 4), wheel_delta_w.reshape(-1, 3)
    ).reshape(env.num_envs, 4, 3)


def wheel_lateral_clearance_l2(
    env,
    *,
    min_abs_y: float,
    max_abs_y: float,
    normalizer: float,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    wheel_y = _body_frame_wheel_positions(env, asset_cfg)[..., 1].abs()
    violation = torch.relu(min_abs_y - wheel_y) + torch.relu(wheel_y - max_abs_y)
    return torch.sum(torch.square(violation / normalizer), dim=1)


def wheel_lateral_target_deadband_l2(
    env,
    *,
    command_name: str,
    target_y: tuple[float, float, float, float],
    base_deadband: float,
    vx_deadband_gain: float,
    omega_deadband_gain: float,
    normalizer: float,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    wheel_y = _body_frame_wheel_positions(env, asset_cfg)[..., 1]
    target = wheel_y.new_tensor(target_y).unsqueeze(0)
    command = env.command_manager.get_command(command_name)
    deadband = (
        base_deadband
        + vx_deadband_gain * command[:, 0].abs()
        + omega_deadband_gain * command[:, 2].abs()
    ).unsqueeze(1)
    violation = torch.relu(torch.abs(wheel_y - target) - deadband)
    return torch.sum(torch.square(violation / normalizer), dim=1)


def wheel_pair_geometry_l2(
    env,
    *,
    command_name: str,
    target_width: tuple[float, float],
    target_center: tuple[float, float],
    width_base_deadband: float,
    width_vx_gain: float,
    width_omega_gain: float,
    center_base_deadband: float,
    center_vx_gain: float,
    center_omega_gain: float,
    width_normalizer: float,
    center_normalizer: float,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    wheel_y = _body_frame_wheel_positions(env, asset_cfg)[..., 1]
    widths = torch.stack((wheel_y[:, 0] - wheel_y[:, 1], wheel_y[:, 2] - wheel_y[:, 3]), dim=1)
    centers = torch.stack(
        (0.5 * (wheel_y[:, 0] + wheel_y[:, 1]), 0.5 * (wheel_y[:, 2] + wheel_y[:, 3])), dim=1
    )
    command = env.command_manager.get_command(command_name)
    abs_vx = command[:, 0].abs()
    abs_omega = command[:, 2].abs()
    width_deadband = (width_base_deadband + width_vx_gain * abs_vx + width_omega_gain * abs_omega).unsqueeze(1)
    center_deadband = (
        center_base_deadband + center_vx_gain * abs_vx + center_omega_gain * abs_omega
    ).unsqueeze(1)
    width_violation = torch.relu(
        torch.abs(widths - widths.new_tensor(target_width).unsqueeze(0)) - width_deadband
    )
    center_violation = torch.relu(
        torch.abs(centers - centers.new_tensor(target_center).unsqueeze(0)) - center_deadband
    )
    return torch.sum(torch.square(width_violation / width_normalizer), dim=1) + torch.sum(
        torch.square(center_violation / center_normalizer), dim=1
    )
