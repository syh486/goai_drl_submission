"""Cross-simulator tensor contract for the S10 locomotion policy."""

from __future__ import annotations

from typing import Any

import torch

from s10_policy_protocol import (
    BASE_COM_BODY,
    DEFAULT_JOINT_POSITIONS,
    LEG_JOINT_NAMES,
    POLICY_JOINT_NAMES,
    POLICY_TO_ROBOT_INDICES,
    ROBOT_JOINT_NAMES,
    ROBOT_TO_POLICY_INDICES,
    WHEEL_BODY_NAMES,
    WHEEL_JOINT_NAMES,
    audit_mujoco_s10_protocol,
)


POLICY_OBSERVATION_TERMS = ("proprio",)
CRITIC_EXTRA_TERMS = (
    "base_lin_vel",
    "disturbance",
    "height_scan",
    "wheel_contact_forces",
)


def audit_isaaclab_go2w_protocol(env: Any) -> dict[str, Any]:
    """Fail fast if IsaacLab resolves any policy-facing tensor out of order."""

    raw = env.unwrapped
    robot = raw.scene["robot"]
    missing = [name for name in ROBOT_JOINT_NAMES if name not in robot.joint_names]
    if missing:
        raise RuntimeError(f"S10 USD is missing joints: {missing}")

    action_names = tuple(raw.action_manager.active_terms)
    if action_names != ("leg_joint_pos", "wheel_joint_vel"):
        raise RuntimeError(f"action term order mismatch: {action_names}")
    leg_term = raw.action_manager._terms["leg_joint_pos"]
    wheel_term = raw.action_manager._terms["wheel_joint_vel"]
    if tuple(leg_term._joint_names) != LEG_JOINT_NAMES:
        raise RuntimeError(f"leg action order mismatch: {tuple(leg_term._joint_names)}")
    if tuple(wheel_term._joint_names) != WHEEL_JOINT_NAMES:
        raise RuntimeError(f"wheel action order mismatch: {tuple(wheel_term._joint_names)}")

    expected_leg_scale = torch.tensor(
        [0.125 if "hipx" in name else 0.25 for name in LEG_JOINT_NAMES],
        device=raw.device,
    )
    actual_leg_scale = torch.as_tensor(leg_term._scale, device=raw.device)[0]
    actual_wheel_scale = torch.as_tensor(wheel_term._scale, device=raw.device)
    if actual_wheel_scale.ndim:
        actual_wheel_scale = actual_wheel_scale[0]
    if not torch.allclose(actual_leg_scale, expected_leg_scale):
        raise RuntimeError(f"leg action scale mismatch: {actual_leg_scale.tolist()}")
    if not torch.allclose(actual_wheel_scale, torch.full_like(actual_wheel_scale, 5.0)):
        raise RuntimeError(f"wheel action scale mismatch: {actual_wheel_scale.tolist()}")

    if abs(float(raw.cfg.decimation) * float(raw.cfg.sim.dt) - 0.02) > 1.0e-12:
        raise RuntimeError(
            f"policy timing mismatch: physics_dt={raw.cfg.sim.dt}, decimation={raw.cfg.decimation}, "
            f"step_dt={float(raw.cfg.decimation) * float(raw.cfg.sim.dt)}"
        )
    leg_actuator = robot.actuators["legs"]
    wheel_actuator = robot.actuators["wheels"]
    action_delay_override = getattr(raw.cfg, "sim2sim_action_delay_override", None)
    expected_min_delay = 0 if action_delay_override is None else int(action_delay_override)
    expected_max_delay = (
        int(round(0.015 / float(raw.cfg.sim.dt)))
        if action_delay_override is None
        else int(action_delay_override)
    )
    expected_actuators = (
        (leg_actuator, 80.0, 2.0, 50.0),
        (wheel_actuator, 0.0, 0.8, 14.0),
    )
    for actuator, stiffness, damping, effort in expected_actuators:
        cfg = actuator.cfg
        configured_effort = cfg.effort_limit_sim if actuator.is_implicit_model else cfg.effort_limit
        if (
            float(cfg.stiffness) != stiffness
            or float(cfg.damping) != damping
            or float(configured_effort) != effort
            or cfg.min_delay != expected_min_delay
            or cfg.max_delay != expected_max_delay
            or cfg.resample_every_n_physics_steps != raw.cfg.decimation
        ):
            raise RuntimeError(f"S10 actuator protocol mismatch: {cfg}")

    actuator_models = {actuator.is_implicit_model for actuator in (leg_actuator, wheel_actuator)}
    if len(actuator_models) != 1:
        raise RuntimeError("leg and wheel actuators use different integration models")
    uses_implicit_actuators = actuator_models.pop()
    if not uses_implicit_actuators:
        sim_stiffness = robot.root_physx_view.get_dof_stiffnesses().to(raw.device)
        sim_damping = robot.root_physx_view.get_dof_dampings().to(raw.device)
        controlled_ids = torch.cat((leg_actuator.joint_indices, wheel_actuator.joint_indices))
        if torch.any(sim_stiffness[:, controlled_ids] != 0.0) or torch.any(sim_damping[:, controlled_ids] != 0.0):
            raise RuntimeError("explicit S10 actuator still has non-zero PhysX joint-drive gains")

    expected_leg_reset = torch.tensor(
        [DEFAULT_JOINT_POSITIONS[name] for name in leg_actuator.joint_names],
        dtype=torch.float32,
        device=raw.device,
    )
    actual_leg_reset = leg_actuator.reset_target_tensor(leg_actuator.cfg.reset_position_target)
    if not torch.equal(actual_leg_reset, expected_leg_reset):
        raise RuntimeError(
            "leg actuator reset target was not resolved in IsaacLab's actual actuator joint order"
        )

    initial_targets = (
        (leg_actuator.position_delay, robot.data.joint_pos_target[:, leg_actuator.joint_indices]),
        (leg_actuator.velocity_delay, robot.data.joint_vel_target[:, leg_actuator.joint_indices]),
        (leg_actuator.effort_delay, robot.data.joint_effort_target[:, leg_actuator.joint_indices]),
        (wheel_actuator.position_delay, robot.data.joint_pos_target[:, wheel_actuator.joint_indices]),
        (wheel_actuator.velocity_delay, robot.data.joint_vel_target[:, wheel_actuator.joint_indices]),
        (wheel_actuator.effort_delay, robot.data.joint_effort_target[:, wheel_actuator.joint_indices]),
    )
    for delay, expected in initial_targets:
        actual = delay._circular_buffer._buffer
        if actual is None or not torch.equal(actual, expected.unsqueeze(0).expand_as(actual)):
            raise RuntimeError("initial action-delay ring does not contain the zero-raw-action targets")

    # The reference samples one Kp, Kd and motor-strength scalar per
    # environment, shared over every joint.  Kp*motor must therefore be
    # constant over leg joints, while Kd*motor is constant over legs+wheels.
    leg_ids = leg_actuator.joint_indices
    wheel_ids = wheel_actuator.joint_indices
    stiffness_ratio = leg_actuator.stiffness / robot.data.default_joint_stiffness[:, leg_ids]
    leg_damping_ratio = leg_actuator.damping / robot.data.default_joint_damping[:, leg_ids]
    wheel_damping_ratio = wheel_actuator.damping / robot.data.default_joint_damping[:, wheel_ids]
    damping_ratio = torch.cat((leg_damping_ratio, wheel_damping_ratio), dim=1)
    if torch.any(stiffness_ratio.max(dim=1).values - stiffness_ratio.min(dim=1).values > 1.0e-6):
        raise RuntimeError("Go2W Kp/motor randomization was not shared across leg joints")
    if torch.any(damping_ratio.max(dim=1).values - damping_ratio.min(dim=1).values > 1.0e-6):
        raise RuntimeError("Go2W Kd/motor randomization was not shared across all joints")

    materials = robot.root_physx_view.get_material_properties()
    static_friction = materials[:, :, 0]
    dynamic_friction = materials[:, :, 1]
    if torch.any(static_friction.max(dim=1).values - static_friction.min(dim=1).values > 1.0e-6):
        raise RuntimeError("Go2W friction was not shared across every robot shape")
    if not torch.allclose(static_friction, dynamic_friction, atol=1.0e-6, rtol=0.0):
        raise RuntimeError("Go2W static and dynamic friction coefficients differ")

    base_com = robot.data.com_pos_b[0, 0]
    expected_com = torch.tensor(BASE_COM_BODY, dtype=base_com.dtype, device=base_com.device)
    if not torch.allclose(base_com, expected_com, atol=1.0e-6, rtol=0.0):
        raise RuntimeError(
            f"IsaacLab base COM mismatch: actual={base_com.tolist()}, expected={list(BASE_COM_BODY)}"
        )

    obs_terms = raw.observation_manager.active_terms
    policy_terms = tuple(obs_terms["policy"])
    critic_terms = tuple(obs_terms["critic"])
    if policy_terms != POLICY_OBSERVATION_TERMS:
        raise RuntimeError(f"policy observation order mismatch: {policy_terms}")
    if critic_terms != POLICY_OBSERVATION_TERMS + CRITIC_EXTRA_TERMS:
        raise RuntimeError(f"critic observation order mismatch: {critic_terms}")

    policy_joint_ids = [robot.joint_names.index(name) for name in POLICY_JOINT_NAMES]
    default = robot.data.default_joint_pos[0, policy_joint_ids]
    expected_default = torch.tensor(
        [DEFAULT_JOINT_POSITIONS[name] for name in POLICY_JOINT_NAMES],
        dtype=default.dtype,
        device=default.device,
    )
    if not torch.allclose(default, expected_default, atol=1.0e-6, rtol=0.0):
        raise RuntimeError(
            f"S10 default pose mismatch in policy order: actual={default.tolist()} "
            f"expected={expected_default.tolist()}"
        )

    contact_sensor = raw.scene.sensors["contact_forces"]
    wheel_body_ids, wheel_names = contact_sensor.find_bodies(list(WHEEL_BODY_NAMES), preserve_order=True)
    if tuple(wheel_names) != WHEEL_BODY_NAMES:
        raise RuntimeError(f"wheel contact order mismatch: {tuple(wheel_names)}")

    return {
        "asset_joint_order": tuple(robot.joint_names),
        "policy_joint_ids": tuple(policy_joint_ids),
        "action_terms": action_names,
        "policy_terms": policy_terms,
        "critic_terms": critic_terms,
        "wheel_contact_ids": tuple(wheel_body_ids),
        "base_com_body_xyz": tuple(float(value) for value in base_com.tolist()),
        "policy_to_robot": POLICY_TO_ROBOT_INDICES,
        "robot_to_policy": ROBOT_TO_POLICY_INDICES,
        "actuator_integration": "implicit" if uses_implicit_actuators else "explicit",
    }
