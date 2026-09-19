"""MuJoCo-facing adapters for the migrated SRU training framework.

The backend protocol is deliberately small.  A real MuJoCo backend can be
connected without importing IsaacLab or ROS into ``rsl_rl``.  This file does
not choose the eventual reward or terrain sampling strategy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import torch

from rsl_rl.env import VecEnv
from .s10_policy_config import S10ActionSpec, S10ObservationSpec


@dataclass
class S10RawState:
    """Raw state required by the policy adapter, all tensors batch-first."""

    base_lin_vel: torch.Tensor
    base_ang_vel: torch.Tensor
    projected_gravity: torch.Tensor
    last_action: torch.Tensor
    goal_body: torch.Tensor
    lidar_latent: torch.Tensor
    height_latent: torch.Tensor | None = None
    time_normalized: torch.Tensor | None = None


class S10MujocoBackend(Protocol):
    """Minimal backend contract; implementation may be in-process or ROS-backed."""

    num_envs: int
    max_episode_length: int

    def reset(self) -> S10RawState: ...

    def step(
        self, cmd_vel: torch.Tensor
    ) -> tuple[S10RawState, torch.Tensor, torch.Tensor, dict[str, Any]]: ...

    def close(self) -> None: ...


class S10ObservationAdapter:
    """Build actor and asymmetric critic observations with explicit ordering."""

    def __init__(self, spec: S10ObservationSpec, device: torch.device | str):
        spec.validate()
        if spec.next_waypoint_dim:
            raise ValueError("only current-goal observations are supported")
        self.spec = spec
        self.device = torch.device(device)

    def encode(self, state: S10RawState) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        def batch(value: torch.Tensor, width: int, name: str) -> torch.Tensor:
            value = torch.as_tensor(value, dtype=torch.float32, device=self.device)
            if value.ndim == 1:
                value = value.unsqueeze(0)
            if value.ndim != 2 or value.shape[1] != width:
                raise ValueError(f"{name} must have shape [N,{width}], got {tuple(value.shape)}")
            return value

        base_lin = batch(state.base_lin_vel, 3, "base_lin_vel")
        base_ang = batch(state.base_ang_vel, 3, "base_ang_vel")
        gravity = batch(state.projected_gravity, 3, "projected_gravity")
        last_action = batch(state.last_action, 2, "last_action")
        goal = batch(state.goal_body, self.spec.goal_dim, "goal_body")
        proprio = torch.cat((base_lin, base_ang, gravity, last_action, goal), dim=-1)
        latent = torch.as_tensor(state.lidar_latent, dtype=torch.float32, device=self.device)
        if latent.ndim == 3:
            latent = latent.unsqueeze(0)
        if self.spec.num_cameras == 1:
            expected = (latent.shape[0], *self.spec.latent_shape)
            if tuple(latent.shape) != expected:
                raise ValueError(f"lidar_latent must have shape {expected}, got {tuple(latent.shape)}")
            latent_flat = latent.flatten(1)
        else:
            expected = (latent.shape[0], self.spec.num_cameras, *self.spec.latent_shape)
            if tuple(latent.shape) != expected:
                raise ValueError(f"dual-camera lidar_latent must have shape {expected}, got {tuple(latent.shape)}")
            latent_flat = latent.flatten(1)
        if latent_flat.shape[0] != proprio.shape[0]:
            raise ValueError("state fields have inconsistent batch sizes")

        actor_obs = torch.cat((proprio, latent_flat), dim=-1)

        height_is_placeholder = state.height_latent is None
        if height_is_placeholder:
            height_flat = torch.zeros(
                (proprio.shape[0], self.spec.height_dim), dtype=proprio.dtype, device=self.device
            )
        else:
            height = torch.as_tensor(state.height_latent, dtype=torch.float32, device=self.device)
            if height.ndim == 3:
                height = height.unsqueeze(0)
            expected_height = (height.shape[0], *self.spec.height_shape)
            if tuple(height.shape) != expected_height:
                raise ValueError(f"height_latent must have shape {expected_height}, got {tuple(height.shape)}")
            height_flat = height.flatten(1)

        if state.time_normalized is None:
            time_normalized = torch.zeros((proprio.shape[0], 1), dtype=proprio.dtype, device=self.device)
        else:
            time_normalized = batch(state.time_normalized, 1, "time_normalized")
        # ActorCriticSRU inherits the upstream SRU positional contract:
        # [proprioception, time, privileged height latent, visual latent].
        # In particular, the visual latent must remain the final block because
        # the network extracts it with a negative slice.
        critic_obs = torch.cat((proprio, time_normalized, height_flat, latent_flat), dim=-1)
        extras = {"s10/height_feature_is_placeholder": height_is_placeholder}
        return actor_obs, critic_obs, extras


def policy_action_to_cmd_vel(actions: torch.Tensor, spec: S10ActionSpec | None = None) -> torch.Tensor:
    """Map SRU ``[vx, yaw]`` actions to official ``[vx, vy, yaw]`` commands."""

    spec = spec or S10ActionSpec()
    actions = torch.as_tensor(actions, dtype=torch.float32)
    if actions.ndim != 2 or actions.shape[-1] != 2:
        raise ValueError(f"SRU actions must have shape [N,2], got {tuple(actions.shape)}")
    processed = torch.tanh(actions)
    cmd = torch.zeros((*actions.shape[:-1], 3), dtype=actions.dtype, device=actions.device)
    cmd[:, 0] = torch.clamp(processed[:, 0] * spec.policy_scale_vx, -spec.max_vx, spec.max_vx)
    cmd[:, 1] = 0.0
    cmd[:, 2] = torch.clamp(processed[:, 1] * spec.policy_scale_yaw, -spec.max_yaw, spec.max_yaw)
    return cmd


class S10MujocoVecEnv(VecEnv):
    """Adapt a MuJoCo backend to the original rsl_rl ``VecEnv`` contract."""

    def __init__(
        self,
        backend: S10MujocoBackend,
        obs_spec: S10ObservationSpec | None = None,
        action_spec: S10ActionSpec | None = None,
        randomize_action_scale: bool = False,
        device: torch.device | str = "cpu",
    ):
        self.backend = backend
        self.device = torch.device(device)
        self.obs_spec = obs_spec or S10ObservationSpec()
        self.action_spec = action_spec or S10ActionSpec()
        self.randomize_action_scale = bool(randomize_action_scale)
        self.adapter = S10ObservationAdapter(self.obs_spec, self.device)
        self.num_envs = backend.num_envs
        self.num_obs = self.obs_spec.actor_obs_dim
        self.num_privileged_obs = self.obs_spec.critic_obs_dim
        self.num_actions = 2
        self.max_episode_length = backend.max_episode_length
        self.cfg = {"name": "s10_mujoco_backend", "physics_dt": self.action_spec.mujoco_dt}
        self.episode_length_buf = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.reset_buf = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.rew_buf = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.obs_buf = torch.zeros((self.num_envs, self.num_obs), device=self.device)
        self.privileged_obs_buf = torch.zeros((self.num_envs, self.num_privileged_obs), device=self.device)
        self._filtered_cmd = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)
        self._filter_alpha = torch.zeros_like(self._filtered_cmd)
        self._policy_scale = torch.zeros((self.num_envs, 2), dtype=torch.float32, device=self.device)
        self._policy_bias = torch.zeros_like(self._policy_scale)
        self.extras: dict[str, Any] = {}
        self._last_state: S10RawState | None = None
        self.reset()

    def _reset_action_processing(self, env_mask: torch.Tensor) -> None:
        """Reset the MX action term state at the start of an episode.

        The upstream MX config disables ``randomize_action_scale`` and keeps
        only low-pass alpha randomization. The optional scale branch remains
        available for explicit legacy experiments.
        """

        count = int(env_mask.sum().item())
        if count == 0:
            return
        self._filtered_cmd[env_mask] = 0.0
        self._filter_alpha[env_mask] = 0.1 + 0.5 * torch.rand(
            (count, 3), dtype=torch.float32, device=self.device
        )
        base_scale = torch.tensor(
            (self.action_spec.policy_scale_vx, self.action_spec.policy_scale_yaw),
            dtype=torch.float32,
            device=self.device,
        )
        if self.randomize_action_scale:
            self._policy_scale[env_mask] = base_scale * (
                0.8 + 0.4 * torch.rand((count, 2), dtype=torch.float32, device=self.device)
            )
            self._policy_bias[env_mask] = -0.1 + 0.2 * torch.rand(
                (count, 2), dtype=torch.float32, device=self.device
            )
        else:
            self._policy_scale[env_mask] = base_scale
            self._policy_bias[env_mask] = 0.0

    def _restore_entry_action_processing(self, env_mask: torch.Tensor) -> None:
        state_getter = getattr(self.backend, "reset_action_processing_state", None)
        if state_getter is None:
            return
        state = state_getter()
        valid = torch.as_tensor(state["valid"], dtype=torch.bool, device=self.device)
        selected = torch.logical_and(env_mask, valid)
        if not torch.any(selected):
            return
        for name, target in (
            ("filtered_cmd", self._filtered_cmd),
            ("filter_alpha", self._filter_alpha),
            ("policy_scale", self._policy_scale),
            ("policy_bias", self._policy_bias),
        ):
            values = torch.as_tensor(
                state[name], dtype=torch.float32, device=self.device
            )
            target[selected] = values[selected]

    def _process_actions(self, actions: torch.Tensor) -> torch.Tensor:
        """Reproduce SRU tanh, velocity bias, scaling, and command filtering."""

        actions = actions.to(device=self.device, dtype=torch.float32)
        speed = torch.linalg.vector_norm(self._last_state.base_lin_vel, dim=1, keepdim=True)
        processed = (torch.tanh(actions) + speed * self._policy_bias) * self._policy_scale
        target = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)
        target[:, 0] = processed[:, 0]
        target[:, 2] = processed[:, 1]
        self._filtered_cmd.mul_(self._filter_alpha).add_(target * (1.0 - self._filter_alpha))
        cmd = self._filtered_cmd.clone()
        cmd[:, 0].clamp_(-self.action_spec.max_vx, self.action_spec.max_vx)
        cmd[:, 1].zero_()
        cmd[:, 2].clamp_(-self.action_spec.max_yaw, self.action_spec.max_yaw)
        return cmd

    def _refresh_observations(self) -> tuple[torch.Tensor, dict]:
        assert self._last_state is not None
        obs, critic_obs, adapter_extras = self.adapter.encode(self._last_state)
        self.obs_buf.copy_(obs)
        self.privileged_obs_buf.copy_(critic_obs)
        self.extras = {"observations": {"critic": self.privileged_obs_buf}, **adapter_extras}
        return self.obs_buf, self.extras

    def get_observations(self) -> tuple[torch.Tensor, dict]:
        return self._refresh_observations()

    def randomize_episode_lengths(self) -> None:
        """Apply the upstream runner's initial episode-phase randomization."""

        lengths = torch.randint_like(
            self.episode_length_buf, high=int(self.max_episode_length)
        )
        self.episode_length_buf.copy_(lengths)
        setter = getattr(self.backend, "set_initial_episode_lengths", None)
        if setter is not None:
            state = setter(lengths.detach().cpu().numpy())
            if state is not None:
                self._last_state = state

    def reset(self) -> tuple[torch.Tensor, dict]:
        self._last_state = self.backend.reset()
        self.episode_length_buf.zero_()
        self.reset_buf.zero_()
        reset_mask = torch.ones(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._reset_action_processing(reset_mask)
        self._restore_entry_action_processing(reset_mask)
        return self._refresh_observations()

    def step(self, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        actions = actions.to(device=self.device, dtype=torch.float32)
        cmd_vel = self._process_actions(actions)
        set_policy_actions = getattr(self.backend, "set_policy_actions", None)
        if set_policy_actions is not None:
            set_policy_actions(actions)
        state, rewards, dones, info = self.backend.step(cmd_vel)
        self._last_state = state
        self.episode_length_buf += 1
        rewards = torch.as_tensor(rewards, dtype=torch.float32, device=self.device).reshape(self.num_envs)
        dones = torch.as_tensor(dones, dtype=torch.bool, device=self.device).reshape(self.num_envs)
        backend_timeouts = info.get("time_outs")
        if backend_timeouts is None:
            timeouts = self.episode_length_buf >= self.max_episode_length
        else:
            timeouts = torch.as_tensor(backend_timeouts, dtype=torch.bool, device=self.device).reshape(self.num_envs)
            # A backend should classify success/fall/time-limit endings. Keep
            # the VecEnv safety timeout as a final guard for malformed backends.
            timeouts = torch.logical_or(
                timeouts, self.episode_length_buf >= self.max_episode_length
            )
        dones = torch.logical_or(dones, timeouts)
        self.reset_buf.copy_(dones)
        self.rew_buf.copy_(rewards)
        if torch.any(dones):
            self.episode_length_buf[dones] = 0
            self._reset_action_processing(dones)
            self._restore_entry_action_processing(dones)
        obs, extras = self._refresh_observations()
        info = dict(info)
        info["time_outs"] = timeouts
        info["observations"] = extras["observations"]
        return obs, rewards, dones, info

    def close(self) -> None:
        self.backend.close()

    def get_training_state(self) -> dict[str, Any]:
        state_dict = getattr(self.backend, "training_state_dict", None)
        return state_dict() if state_dict is not None else {}

    def load_training_state(self, state: dict[str, Any]) -> None:
        load_state_dict = getattr(self.backend, "load_training_state_dict", None)
        if load_state_dict is not None:
            load_state_dict(state)
