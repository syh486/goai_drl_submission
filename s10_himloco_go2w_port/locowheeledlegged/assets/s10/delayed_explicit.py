"""Explicit S10 PD actuator with the Go2W intra-policy-step action delay.

Unlike PhysX joint drives, this model evaluates and clips the same PD torque
law used by the IsaacGym reference and the MuJoCo/real S10 controller at every
physics step.  The public delay-buffer interface intentionally matches
``DelayedImplicitActuator`` so reset/history protocol audits remain shared.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
from isaaclab.actuators import IdealPDActuator, IdealPDActuatorCfg
from isaaclab.utils import DelayBuffer, configclass
from isaaclab.utils.types import ArticulationActions


class DelayedExplicitPDActuator(IdealPDActuator):
    cfg: "DelayedExplicitPDActuatorCfg"

    def __init__(self, cfg: "DelayedExplicitPDActuatorCfg", *args: Any, **kwargs: Any) -> None:
        super().__init__(cfg, *args, **kwargs)
        self.position_delay = DelayBuffer(cfg.max_delay, self._num_envs, device=self._device)
        self.velocity_delay = DelayBuffer(cfg.max_delay, self._num_envs, device=self._device)
        self.effort_delay = DelayBuffer(cfg.max_delay, self._num_envs, device=self._device)
        self._delay_generator = torch.Generator(device=self._device)
        self._delay_generator.manual_seed(cfg.delay_seed)
        self._physics_step = 0

    def reset(self, env_ids: Sequence[int] | None) -> None:
        super().reset(env_ids)
        for buffer in (self.position_delay, self.velocity_delay, self.effort_delay):
            buffer.reset(env_ids)
        self.prime_reset_targets(env_ids)

    def prime_reset_targets(self, env_ids: Sequence[int] | None) -> None:
        self._prime_delay(self.position_delay, self.cfg.reset_position_target, env_ids)
        self._prime_delay(self.velocity_delay, self.cfg.reset_velocity_target, env_ids)
        self._prime_delay(self.effort_delay, self.cfg.reset_effort_target, env_ids)

    def prime_command_targets(
        self,
        position: torch.Tensor,
        velocity: torch.Tensor,
        effort: torch.Tensor,
        env_ids: Sequence[int],
    ) -> None:
        self._prime_delay_values(self.position_delay, position, env_ids)
        self._prime_delay_values(self.velocity_delay, velocity, env_ids)
        self._prime_delay_values(self.effort_delay, effort, env_ids)

    def _prime_delay(
        self,
        delay: DelayBuffer,
        target: float | tuple[float, ...] | dict[str, float],
        env_ids: Sequence[int] | None,
    ) -> None:
        value = self.reset_target_tensor(target)
        value = value.unsqueeze(0).repeat(self._num_envs, 1)
        selected = slice(None) if env_ids is None else env_ids
        ring = delay._circular_buffer
        if ring._buffer is None:
            ring._buffer = value.unsqueeze(0).repeat(ring.max_length, 1, 1)
            ring._pointer = ring.max_length - 1
            ring._num_pushes[:] = ring.max_length
            return
        self._prime_delay_values(delay, value[selected], selected)

    def _prime_delay_values(
        self,
        delay: DelayBuffer,
        value: torch.Tensor,
        env_ids: Sequence[int] | slice,
    ) -> None:
        if value.ndim != 2 or value.shape[1] != self.num_joints:
            raise ValueError(
                f"delay prime value must have shape [N,{self.num_joints}], got {tuple(value.shape)}"
            )
        ring = delay._circular_buffer
        if ring._buffer is None:
            if not isinstance(env_ids, slice) or value.shape[0] != self._num_envs:
                raise RuntimeError("partial delay priming cannot initialize an empty circular buffer")
            ring._buffer = value.unsqueeze(0).repeat(ring.max_length, 1, 1)
            ring._pointer = ring.max_length - 1
        else:
            ring._buffer[:, env_ids] = value.unsqueeze(0)
        ring._num_pushes[env_ids] = ring.max_length

    def reset_target_tensor(
        self,
        target: float | tuple[float, ...] | dict[str, float],
    ) -> torch.Tensor:
        if isinstance(target, dict):
            missing = tuple(name for name in self.joint_names if name not in target)
            if missing:
                raise ValueError(f"reset target is missing actuator joints: {missing}")
            target = tuple(target[name] for name in self.joint_names)
        value = torch.as_tensor(target, dtype=torch.float32, device=self._device)
        if value.ndim == 0:
            value = value.repeat(self.num_joints)
        if value.shape != (self.num_joints,):
            raise ValueError(
                f"reset target must be scalar or length {self.num_joints}, got {tuple(value.shape)}"
            )
        return value

    def compute(
        self,
        control_action: ArticulationActions,
        joint_pos: torch.Tensor,
        joint_vel: torch.Tensor,
    ) -> ArticulationActions:
        # legged_gym samples one 0--3 substep switch time for the complete
        # action once per policy step.  Identically seeded leg/wheel actuator
        # instances therefore apply one shared 16-D delay schedule.
        if self._physics_step % self.cfg.resample_every_n_physics_steps == 0:
            lags = torch.randint(
                self.cfg.min_delay,
                self.cfg.max_delay + 1,
                (self._num_envs,),
                dtype=torch.int,
                device=self._device,
                generator=self._delay_generator,
            )
            for buffer in (self.position_delay, self.velocity_delay, self.effort_delay):
                buffer.set_time_lag(lags)
        self._physics_step += 1
        control_action.joint_positions = self.position_delay.compute(control_action.joint_positions)
        control_action.joint_velocities = self.velocity_delay.compute(control_action.joint_velocities)
        control_action.joint_efforts = self.effort_delay.compute(control_action.joint_efforts)
        return super().compute(control_action, joint_pos, joint_vel)


@configclass
class DelayedExplicitPDActuatorCfg(IdealPDActuatorCfg):
    class_type: type = DelayedExplicitPDActuator
    min_delay: int = 0
    max_delay: int = 0
    resample_every_n_physics_steps: int = 4
    delay_seed: int = 137
    reset_position_target: float | tuple[float, ...] | dict[str, float] = 0.0
    reset_velocity_target: float | tuple[float, ...] | dict[str, float] = 0.0
    reset_effort_target: float | tuple[float, ...] | dict[str, float] = 0.0

    def __post_init__(self) -> None:
        if self.min_delay < 0 or self.max_delay < self.min_delay:
            raise ValueError("actuator delay must satisfy 0 <= min_delay <= max_delay")
        if self.resample_every_n_physics_steps < 1:
            raise ValueError("delay resampling period must be positive")
