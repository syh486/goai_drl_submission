"""S10 observation helpers that enforce the official policy joint order."""

from __future__ import annotations

import torch
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg


def go2w_proprio_reference(
    env,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    cache_role: str,
    add_noise: bool = True,
) -> torch.Tensor:
    """Build the reference 57-D frame and share its noise across actor/critic.

    IsaacGym constructs ``current_obs`` once and uses the same noisy first 57
    values for the policy history and privileged critic observation.  Separate
    IsaacLab observation groups otherwise draw independent noise.  The policy
    group is the producer and the critic group consumes the exact cached frame.
    """

    if cache_role not in {"producer", "consumer"}:
        raise ValueError(f"invalid Go2W proprio cache role: {cache_role}")
    cache = getattr(env, "_go2w_shared_proprio", None)
    if cache_role == "consumer":
        if cache is None:
            raise RuntimeError("critic requested Go2W proprio before policy producer")
        return cache

    asset: Articulation = env.scene[asset_cfg.name]
    joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    joint_vel = asset.data.joint_vel[:, asset_cfg.joint_ids] - asset.data.default_joint_vel[:, asset_cfg.joint_ids]
    joint_pos = joint_pos.clone()
    joint_vel = joint_vel.clone()
    joint_pos[:, -4:] = 0.0
    if not getattr(env.cfg, "observe_wheel_velocity", False):
        joint_vel[:, -4:] = 0.0
    command = env.command_manager.get_command(command_name)
    last_action = env.action_manager.action
    frame = torch.cat(
        (
            asset.data.root_ang_vel_b * 0.25,
            asset.data.projected_gravity_b,
            command * command.new_tensor((2.0, 2.0, 0.25)),
            joint_pos,
            joint_vel * 0.05,
            last_action,
        ),
        dim=-1,
    )
    if frame.shape[1] != 57:
        raise RuntimeError(f"Go2W proprio frame must be 57-D, got {frame.shape[1]}")

    # Exact post-scale noise support from the IsaacGym reference:
    # angular velocity .2*.25, gravity .05, joint position .01,
    # joint velocity 1.5*.05; commands and previous actions are noiseless.
    if add_noise:
        noise_scale = frame.new_zeros(57)
        noise_scale[0:3] = 0.05
        noise_scale[3:6] = 0.05
        noise_scale[9:25] = 0.01
        noise_scale[25:41] = 0.075
        frame = frame + (2.0 * torch.rand_like(frame) - 1.0) * noise_scale
    env._go2w_shared_proprio = frame
    return frame


def joint_pos_rel_without_wheel_policy_order(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    value = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    value = value.clone()
    value[:, -4:] = 0.0
    return value


def joint_vel_rel_policy_order(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Go2W's effective joint-velocity observation in S10 policy order.

    The IsaacGym reference evaluates rewards before observations.  Its active
    ``dof_vel`` reward zeros ``self.dof_vel[:, wheel_indices]`` in-place, so the
    four wheel-velocity observation entries are always zero.  Reproducing that
    effective tensor here avoids depending on reward side effects and keeps the
    deployment contract explicit.
    """

    asset: Articulation = env.scene[asset_cfg.name]
    value = asset.data.joint_vel[:, asset_cfg.joint_ids] - asset.data.default_joint_vel[:, asset_cfg.joint_ids]
    value = value.clone()
    value[:, -4:] = 0.0
    return value
