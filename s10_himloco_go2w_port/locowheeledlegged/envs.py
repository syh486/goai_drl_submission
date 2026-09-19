"""Environment classes specific to this independent port."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from isaaclab.envs import ManagerBasedRLEnv


class Go2WPositiveRewardEnv(ManagerBasedRLEnv):
    """Preserve the reference Go2W reward and six-frame buffer semantics.

    The IsaacGym task keeps the five pre-reset frames in ``obs_buf`` and only
    prepends the freshly reset state.  IsaacLab normally clears each history
    buffer during a reset and fills all six slots with the first new frame.
    Both differences are observable by HIM's temporal encoder, so this class
    restores the reference behavior for this task only.
    """

    def __init__(self, *args, **kwargs):
        self._go2w_initial_history_adjusted = False
        self._go2w_terminal_critic = None
        self._go2w_terminal_mask = None
        super().__init__(*args, **kwargs)

    def _policy_history_buffer(self):
        groups = getattr(self.observation_manager, "_group_obs_term_history_buffer", {})
        return groups.get("policy", {}).get("proprio")

    def _reset_idx(self, env_ids: Sequence[int]):
        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        history = self._policy_history_buffer()
        saved_buffer = None
        saved_pushes = None
        if history is not None and history._buffer is not None:
            saved_buffer = history._buffer[:, env_ids].clone()
            saved_pushes = history._num_pushes[env_ids].clone()

        # Capture the reference's reset-before-observation terminal critic
        # state.  The producer call refreshes the shared noisy proprio frame
        # without appending an artificial policy-history frame.
        policy_cfg = self.observation_manager._group_obs_term_cfgs["policy"][0]
        policy_cfg.func(self, **policy_cfg.params)
        terminal_critic = self.observation_manager.compute_group("critic")
        if self._go2w_terminal_critic is None:
            self._go2w_terminal_critic = torch.zeros_like(terminal_critic)
            self._go2w_terminal_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._go2w_terminal_critic[env_ids] = terminal_critic[env_ids]
        self._go2w_terminal_mask[env_ids] = True

        robot = self.scene["robot"]
        initial_reset = not self._go2w_initial_history_adjusted and self.common_step_counter == 0
        if initial_reset:
            # Before the first policy action, legged_gym's zero raw action maps
            # to the nominal leg pose and zero wheel velocity.
            saved_action = torch.zeros_like(self.action_manager._action[env_ids])
            saved_terms = {}
            for name, term in self.action_manager._terms.items():
                raw = torch.zeros_like(term._raw_actions[env_ids])
                if isinstance(term._offset, torch.Tensor):
                    processed = term._offset[env_ids].clone()
                else:
                    processed = torch.full_like(raw, float(term._offset))
                saved_terms[name] = (raw, processed)
            saved_position_target = robot.data.default_joint_pos[env_ids].clone()
            saved_velocity_target = robot.data.default_joint_vel[env_ids].clone()
            saved_effort_target = torch.zeros_like(saved_velocity_target)
        else:
            # legged_gym resets last_actions and then, after computing reset
            # observations, copies terminal self.actions back into it.  The
            # effective next-step action-rate and delay predecessor is therefore
            # the terminal action, which also appears in the reset observation.
            saved_action = self.action_manager._action[env_ids].clone()
            saved_terms = {
                name: (term._raw_actions[env_ids].clone(), term._processed_actions[env_ids].clone())
                for name, term in self.action_manager._terms.items()
            }
            saved_position_target = robot.data.joint_pos_target[env_ids].clone()
            saved_velocity_target = robot.data.joint_vel_target[env_ids].clone()
            saved_effort_target = robot.data.joint_effort_target[env_ids].clone()

        super()._reset_idx(env_ids)
        if saved_buffer is not None:
            # IsaacGym does not clear obs_buf in reset_idx().
            history._buffer[:, env_ids] = saved_buffer
            history._num_pushes[env_ids] = saved_pushes

        self.action_manager._action[env_ids] = saved_action
        for name, (raw, processed) in saved_terms.items():
            term = self.action_manager._terms[name]
            term._raw_actions[env_ids] = raw
            term._processed_actions[env_ids] = processed
        robot.set_joint_position_target(saved_position_target, env_ids=env_ids)
        robot.set_joint_velocity_target(saved_velocity_target, env_ids=env_ids)
        robot.set_joint_effort_target(saved_effort_target, env_ids=env_ids)
        for actuator in robot.actuators.values():
            joint_ids = actuator.joint_indices
            actuator.prime_command_targets(
                saved_position_target[:, joint_ids],
                saved_velocity_target[:, joint_ids],
                saved_effort_target[:, joint_ids],
                env_ids,
            )

    @staticmethod
    def _clip_observations(observations: dict[str, torch.Tensor]) -> None:
        # legged_gym clips both policy and privileged observations after the
        # full scaled vectors have been assembled.
        for value in observations.values():
            if isinstance(value, torch.Tensor):
                value.clamp_(min=-100.0, max=100.0)

    def reset(self, *args, **kwargs):
        observations, extras = super().reset(*args, **kwargs)
        if not self._go2w_initial_history_adjusted:
            history = self._policy_history_buffer()
            if history is None or history._buffer is None:
                raise RuntimeError("Go2W policy history was not initialized by the first reset")
            # CircularBuffer repeats the first append into every slot.  The
            # reference starts as [current, 0, 0, 0, 0, 0] newest-first.
            keep = history._pointer
            for index in range(history.max_length):
                if index != keep:
                    history._buffer[index].zero_()
            history._num_pushes.fill_(1)
            observations["policy"] = history.buffer
            self.obs_buf = observations
            self._go2w_initial_history_adjusted = True
        self._clip_observations(observations)
        return observations, extras

    def step(self, action: torch.Tensor):
        if self._go2w_terminal_mask is not None:
            self._go2w_terminal_mask.zero_()
        observations, rewards, terminated, truncated, extras = super().step(action)
        self._clip_observations(observations)
        rewards.clamp_(min=0.0)
        if self._go2w_terminal_mask is not None and torch.any(self._go2w_terminal_mask):
            extras["go2w_terminal_critic"] = self._go2w_terminal_critic
            extras["go2w_terminal_mask"] = self._go2w_terminal_mask
        self.obs_buf = observations
        self.reward_buf = rewards
        return observations, rewards, terminated, truncated, extras
