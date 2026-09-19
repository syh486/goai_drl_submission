"""Isaac Lab runner for the blind HIMLoco policy."""

from __future__ import annotations

import os
import time
from collections.abc import Sequence
from collections import defaultdict
import json
from pathlib import Path

import torch

from .core import PIMHIMActorCritic, PIMHIMCnnEstimator, PIMHIMPPO, PIMHIMRolloutStorage
from locowheeledlegged.s10_protocol import audit_isaaclab_go2w_protocol


class PIMHIMRunner:
    def __init__(
        self,
        env,
        *,
        log_dir: str | Path,
        history_length: int = 6,
        proprio_dim: int = 57,
        lidar_channels: int = 6,
        image_height: int = 12,
        image_width: int = 16,
        rollout_steps: int = 48,
        save_interval: int = 1000,
        device: str = "cuda:0",
        initial_noise_std: float | Sequence[float] = 1.0,
        entropy_coef: float = 0.005,
        learning_rate: float = 1.0e-3,
        terrain_family_columns: dict[str, tuple[int, int]] | None = None,
        tracking_only: bool = False,
        policy_variant: str = "pim_raw",
    ) -> None:
        self.env = env
        self.device = device
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.rollout_steps = rollout_steps
        self.save_interval = save_interval
        self.history_length = history_length
        self.terrain_family_columns = terrain_family_columns
        self.tracking_only = tracking_only
        if policy_variant != "blind":
            raise ValueError(f"unsupported S10 policy variant: {policy_variant}")
        self.policy_variant = policy_variant
        self.outcome_totals: dict[str, float] = defaultdict(float)
        self._episode_returns = torch.zeros(env.num_envs, device=device)
        self._episode_lengths = torch.zeros(env.num_envs, dtype=torch.long, device=device)
        self._episode_tracking_valid = torch.ones(env.num_envs, dtype=torch.bool, device=device)

        protocol = audit_isaaclab_go2w_protocol(env)
        print(
            "[INFO] S10 protocol audit passed: "
            f"actions={protocol['action_terms']}, "
            f"actuator_integration={protocol['actuator_integration']}, "
            f"policy_to_robot={protocol['policy_to_robot']}, "
            f"wheel_contact_ids={protocol['wheel_contact_ids']}",
            flush=True,
        )

        observations = self._cached_observations()
        history = observations["policy"]
        critic = observations["critic"]
        if history.ndim != 3 or history.shape[1] != history_length:
            raise ValueError(f"expected policy history [N,{history_length},D], got {tuple(history.shape)}")
        self.one_step_dim = history.shape[2]
        self.history_dim = history_length * self.one_step_dim
        self.critic_dim = critic.shape[1]
        uses_lidar = False
        expected_one_step_dim = proprio_dim + (lidar_channels * image_height * image_width if uses_lidar else 0)
        if expected_one_step_dim != self.one_step_dim:
            raise ValueError(
                f"S10 {policy_variant} one-step layout mismatch: expected {expected_one_step_dim}, "
                f"got {self.one_step_dim}"
            )
        velocity_start = self.one_step_dim
        if velocity_start + 3 > self.critic_dim:
            raise ValueError("critic requires true base velocity after the current proprioceptive frame")

        estimator = PIMHIMCnnEstimator(
            history_length=history_length,
            one_step_dim=self.one_step_dim,
            proprio_dim=proprio_dim,
            lidar_start=proprio_dim,
            image_channels=lidar_channels,
            image_height=image_height,
            image_width=image_width,
            # Exact Go2W target: critic[3:60] replaces the first three noisy
            # angular-velocity values with privileged base velocity.
            target_slices=((3, proprio_dim + 3),),
            velocity_start=velocity_start,
            use_lidar=uses_lidar,
        )
        self.actor_critic = PIMHIMActorCritic(
            history_dim=self.history_dim,
            critic_dim=self.critic_dim,
            one_step_dim=self.one_step_dim,
            action_dim=env.num_actions,
            estimator=estimator,
            initial_noise_std=initial_noise_std,
            actor_observation_mode="proprio",
        ).to(device)
        storage = PIMHIMRolloutStorage(
            env.num_envs,
            rollout_steps,
            self.history_dim,
            self.critic_dim,
            env.num_actions,
            device,
        )
        self.algorithm = PIMHIMPPO(
            self.actor_critic,
            storage,
            entropy_coef=entropy_coef,
            learning_rate=learning_rate,
        )
        initial_noise = torch.as_tensor(initial_noise_std, dtype=torch.float32).reshape(-1)
        if initial_noise.numel() == 1:
            initial_noise = initial_noise.repeat(env.num_actions)
        self.initial_noise_std = tuple(float(value) for value in initial_noise.tolist())
        self.iteration = 0
        self._runtime_protocol_audited = False
        self._runtime_reset_history_audited = False
        print(
            f"[INFO] S10 HIMLoco policy: "
            f"history={history_length}x{self.one_step_dim}={self.history_dim}, "
            f"LiDAR={'disabled' if not uses_lidar else f'{lidar_channels}x{image_height}x{image_width}'}, "
            f"critic={self.critic_dim}, action={env.num_actions}",
            flush=True,
        )
        print(
            f"[INFO] policy distribution={self.actor_critic.distribution_type}, "
            f"leg_noise_std={self.initial_noise_std[:12]}, wheel_noise_std={self.initial_noise_std[12:]}, "
            f"entropy_coef={self.algorithm.entropy_coef:g}, "
            f"learning_rate={self.algorithm.learning_rate:g}",
            flush=True,
        )

    def _extract(self, observations: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        # IsaacLab buffers are oldest-first; deployment and the estimator use
        # newest-first so that the first one-step slice is the current frame.
        history = torch.flip(observations["policy"], dims=(1,)).flatten(start_dim=1)
        return history.to(self.device), observations["critic"].to(self.device)

    def _cached_observations(self) -> dict[str, torch.Tensor]:
        """Read observations without appending a fake frame to IsaacLab history.

        IsaacLab 4.5's RSL-RL wrapper implements ``get_observations`` by
        calling ``observation_manager.compute()``.  On this version that call
        always appends to every history buffer, even though physics did not
        advance.  The environment's ``obs_buf`` is the current cached tensor
        and is the correct equivalent of legged_gym's observation buffer.
        """

        observations = getattr(self.env.unwrapped, "obs_buf", None)
        if not isinstance(observations, dict) or "policy" not in observations or "critic" not in observations:
            raise RuntimeError("IsaacLab environment has no cached policy/critic observations")
        return observations

    def _audit_first_runtime_step(
        self,
        previous_history: torch.Tensor,
        next_history: torch.Tensor,
        dones: torch.Tensor,
        actions: torch.Tensor,
    ) -> None:
        shifted = torch.equal(
            next_history[:, self.one_step_dim :],
            previous_history[:, : -self.one_step_dim],
        )
        if not shifted:
            raise RuntimeError("Go2W six-frame history did not shift by exactly one real policy frame")

        if not self._runtime_protocol_audited:
            history_frames = next_history.reshape(-1, self.history_length, self.one_step_dim)
            # Joint velocity starts at 25; the four wheel entries are 37:41.
            # The physical signal is zero and only scaled observation noise
            # with support [-0.075, 0.075] remains.
            measured_wheel_velocity = bool(
                getattr(self.env.unwrapped.cfg, "observe_wheel_velocity", False)
            )
            wheel_velocity_obs = history_frames[:, :, 37:41]
            if not measured_wheel_velocity and torch.any(torch.abs(wheel_velocity_obs) > 0.075001):
                raise RuntimeError(
                    "Go2W wheel-velocity observation exceeded its zero-signal noise support"
                )
            current_policy_frame = history_frames[:, 0]
            current_critic_frame = self._cached_observations()["critic"][:, : self.one_step_dim]
            if not torch.equal(current_policy_frame, current_critic_frame):
                raise RuntimeError("policy and critic did not receive the same noisy Go2W proprio frame")

            robot = self.env.unwrapped.scene["robot"]
            leg_actuator = robot.actuators["legs"]
            wheel_actuator = robot.actuators["wheels"]
            if not torch.equal(
                leg_actuator.position_delay.time_lags,
                wheel_actuator.position_delay.time_lags,
            ):
                raise RuntimeError("leg and wheel actions received different delay samples")
            if leg_actuator._physics_step != wheel_actuator._physics_step:
                raise RuntimeError("leg and wheel delay schedules are out of phase")
            self._runtime_protocol_audited = True
            print(
                "[INFO] S10 runtime protocol audit passed: real-frame history shift, "
                "shared actor/critic proprio, "
                f"wheel-velocity obs={'measured' if measured_wheel_velocity else 'zero-signal'}, "
                "shared 16-D action delay",
                flush=True,
            )

        reset_mask = dones.bool()
        if not self._runtime_reset_history_audited and torch.any(reset_mask):
            if not torch.equal(
                next_history[reset_mask, self.one_step_dim :],
                previous_history[reset_mask, : -self.one_step_dim],
            ):
                raise RuntimeError("Go2W pre-reset history was cleared instead of preserved")
            if not torch.equal(next_history[reset_mask, 41:57], actions[reset_mask]):
                raise RuntimeError(
                    "Go2W reset observation did not preserve the terminal action as the next delay predecessor"
                )
            robot = self.env.unwrapped.scene["robot"]
            for actuator_name in ("legs", "wheels"):
                actuator = robot.actuators[actuator_name]
                checks = (
                    ("position", actuator.position_delay, robot.data.joint_pos_target),
                    ("velocity", actuator.velocity_delay, robot.data.joint_vel_target),
                    ("effort", actuator.effort_delay, robot.data.joint_effort_target),
                )
                for channel, delay, target in checks:
                    expected = target[reset_mask][:, actuator.joint_indices].to(
                        dtype=delay._circular_buffer._buffer.dtype, device=self.device
                    )
                    actual = delay._circular_buffer._buffer[:, reset_mask]
                    if not torch.allclose(
                        actual,
                        expected.unsqueeze(0).expand_as(actual),
                        atol=0.0,
                        rtol=0.0,
                    ):
                        raise RuntimeError(
                            f"{actuator_name} {channel} delay buffer did not preserve the terminal command "
                            f"across reset: max_error={float((actual - expected.unsqueeze(0)).abs().max())}, "
                            f"actual_slots_for_first_reset_env={actual[:, 0].tolist()}, "
                            f"expected_first_reset_env={expected[0].tolist()}"
                        )
            self._runtime_reset_history_audited = True
            print(
                "[INFO] S10 reset-history and terminal-action delay carryover audit passed",
                flush=True,
            )

    @staticmethod
    def _sync() -> None:
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def learn(self, iterations: int, *, randomize_initial_episode_length: bool = True) -> list[dict[str, float]]:
        if randomize_initial_episode_length:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )
            # The first reset after randomizing age closes a partial episode;
            # do not report its partial return as a complete episode return.
            self._episode_tracking_valid[:] = False
        history, critic = self._extract(self._cached_observations())
        timings: list[dict[str, float]] = []
        raw_env = self.env.unwrapped
        reward_names = tuple(raw_env.reward_manager.active_terms)
        active_terminations = tuple(raw_env.termination_manager.active_terms)
        boundary_names = tuple(
            name
            for name in ("terrain_out_of_bounds", "out_of_lane", "out_of_patch")
            if name in active_terminations
        )
        boundary_metric_name = boundary_names[0] if boundary_names else None
        family_masks = self._terrain_family_masks()
        for iteration in range(iterations):
            self._sync()
            started = time.perf_counter()
            rollout_reward = 0.0
            rollout_dones = 0
            log_values: dict[str, list[float]] = defaultdict(list)
            reward_term_sums = torch.zeros(len(reward_names), device=self.device)
            termination_counts: dict[str, float] = defaultdict(float)
            family_sums: dict[str, float] = defaultdict(float)
            completed_returns: list[torch.Tensor] = []
            completed_lengths: list[torch.Tensor] = []
            action_abs_sum = 0.0
            action_sq_sum = 0.0
            action_above_1_count = 0
            action_above_5_count = 0
            action_above_10_count = 0
            action_count = 0
            policy_std_sum = 0.0
            leg_std_sum = 0.0
            wheel_std_sum = 0.0
            leg_action_abs_sum = 0.0
            wheel_action_abs_sum = 0.0
            with torch.inference_mode():
                for _ in range(self.rollout_steps):
                    actions = self.algorithm.act(history, critic)
                    action_abs_sum += float(actions.abs().sum().item())
                    action_sq_sum += float(actions.square().sum().item())
                    action_above_1_count += int((actions.abs() > 1.0).sum().item())
                    action_above_5_count += int((actions.abs() > 5.0).sum().item())
                    action_above_10_count += int((actions.abs() > 10.0).sum().item())
                    action_count += actions.numel()
                    policy_std_sum += float(self.actor_critic.action_std.mean().item())
                    leg_std_sum += float(self.actor_critic.action_std[..., :12].mean().item())
                    wheel_std_sum += float(self.actor_critic.action_std[..., 12:].mean().item())
                    leg_action_abs_sum += float(actions[:, :12].abs().sum().item())
                    wheel_action_abs_sum += float(actions[:, 12:].abs().sum().item())
                    _, rewards, dones, infos = self.env.step(actions)
                    rollout_reward += float(rewards.sum().item())
                    rollout_dones += int(dones.sum().item())
                    # RewardManager._step_reward stores each weighted term as
                    # a rate (value / dt). Convert it back to the exact per-step
                    # contribution so the term sum matches mean_step_reward.
                    reward_term_sums += raw_env.reward_manager._step_reward.sum(dim=0) * raw_env.step_dt
                    term_flags = {
                        name: raw_env.termination_manager.get_term(name)
                        for name in raw_env.termination_manager.active_terms
                    }
                    for name, flags in term_flags.items():
                        termination_counts[name] += float(flags.sum().item())

                    self._episode_returns += rewards
                    self._episode_lengths += 1
                    done_mask = dones.bool()
                    valid_done = done_mask & self._episode_tracking_valid
                    if valid_done.any():
                        completed_returns.append(self._episode_returns[valid_done].clone())
                        completed_lengths.append(self._episode_lengths[valid_done].clone())
                    if done_mask.any():
                        self._episode_returns[done_mask] = 0.0
                        self._episode_lengths[done_mask] = 0
                        self._episode_tracking_valid[done_mask] = True

                    commands = raw_env.command_manager.get_command("base_velocity")
                    robot = raw_env.scene["robot"]
                    zero_flags = torch.zeros(self.env.num_envs, dtype=torch.bool, device=self.device)
                    success = zero_flags.clone()
                    for success_name in ("obstacle_success", "platform_success"):
                        if success_name in term_flags:
                            success |= term_flags[success_name]
                    base_contact = term_flags.get("base_contact", zero_flags)
                    boundary = zero_flags.clone()
                    for boundary_name in boundary_names:
                        boundary |= term_flags[boundary_name]
                    # Only the episode clock is a neutral timeout.  Other
                    # terms may be marked time-out by Isaac Lab for value
                    # bootstrapping (for example global terrain bounds), but
                    # remain task failures for monitoring.
                    timeout = term_flags.get("time_out", zero_flags)
                    failure = zero_flags.clone()
                    for term_name, flags in term_flags.items():
                        if term_name not in ("time_out", "obstacle_success", "platform_success"):
                            failure |= flags
                    for family, mask in family_masks.items():
                        count = int(mask.sum().item())
                        if count == 0:
                            continue
                        family_sums[f"{family}/samples"] += count
                        family_sums[f"{family}/reward"] += float(rewards[mask].sum().item())
                        family_sums[f"{family}/vx_error"] += float(
                            (robot.data.root_lin_vel_b[mask, 0] - commands[mask, 0]).abs().sum().item()
                        )
                        family_sums[f"{family}/omega_error"] += float(
                            (robot.data.root_ang_vel_b[mask, 2] - commands[mask, 2]).abs().sum().item()
                        )
                        # A timeout is not a failure by itself.  The terrain
                        # curriculum evaluates distance at reset and may move
                        # the same episode up, down, or leave it unchanged.
                        # Keep timeouts separate instead of fabricating a
                        # failure signal that disagrees with the curriculum.
                        family_failure = failure & ~success
                        outcomes = [
                            ("success", success),
                            ("failure", family_failure),
                            ("base_contact", base_contact),
                            ("timeout", timeout & ~success),
                        ]
                        if boundary_metric_name is not None:
                            outcomes.append((boundary_metric_name, boundary))
                        for outcome, flags in outcomes:
                            value = float((flags & mask).sum().item())
                            family_sums[f"{family}/{outcome}"] += value
                            self.outcome_totals[f"{family}/{outcome}"] += value
                    for name, value in infos.get("log", {}).items():
                        if isinstance(value, torch.Tensor):
                            if value.numel() != 1:
                                continue
                            value = value.item()
                        if isinstance(value, (int, float)):
                            log_values[name].append(float(value))
                    next_history, next_critic = self._extract(infos["observations"])
                    self._audit_first_runtime_step(history, next_history, dones, actions)
                    estimator_next_critic = next_critic
                    terminal_mask = infos.get("go2w_terminal_mask")
                    terminal_critic = infos.get("go2w_terminal_critic")
                    if terminal_mask is not None:
                        terminal_mask = terminal_mask.to(self.device).bool()
                        if not torch.equal(terminal_mask, dones.bool()):
                            raise RuntimeError("terminal critic mask does not match environment dones")
                        estimator_next_critic = next_critic.clone()
                        estimator_next_critic[terminal_mask] = terminal_critic[terminal_mask].to(self.device)
                    elif torch.any(dones):
                        raise RuntimeError("missing pre-reset terminal critic observation for done transitions")
                    self.algorithm.process_step(
                        rewards.to(self.device), dones.to(self.device), infos, estimator_next_critic
                    )
                    history, critic = next_history, next_critic
                self.algorithm.compute_returns(critic)
            self._sync()
            collection_time = time.perf_counter() - started

            started = time.perf_counter()
            losses = self.algorithm.update()
            self._sync()
            learning_time = time.perf_counter() - started
            iteration_time = collection_time + learning_time
            transitions = self.rollout_steps * self.env.num_envs
            record = {
                "iteration": float(self.iteration),
                "collection": collection_time,
                "learning": learning_time,
                "total": iteration_time,
                "steps_per_second": transitions / iteration_time,
                "mean_step_reward": rollout_reward / transitions,
                "dones": float(rollout_dones),
                **losses,
            }
            for index, name in enumerate(reward_names):
                record[f"reward/{name}"] = float(reward_term_sums[index].item()) / transitions
            record["reward/decomposition_sum"] = sum(record[f"reward/{name}"] for name in reward_names)
            record["reward/decomposition_error"] = record["reward/decomposition_sum"] - record["mean_step_reward"]
            for name, count in termination_counts.items():
                record[f"termination/{name}_count"] = count
            record["policy/action_abs_mean"] = action_abs_sum / max(action_count, 1)
            record["policy/action_rms"] = (action_sq_sum / max(action_count, 1)) ** 0.5
            record["policy/action_above_1_fraction"] = action_above_1_count / max(action_count, 1)
            record["policy/action_above_5_fraction"] = action_above_5_count / max(action_count, 1)
            record["policy/action_above_10_fraction"] = action_above_10_count / max(action_count, 1)
            record["policy/std_mean"] = policy_std_sum / self.rollout_steps
            record["policy/leg_std_mean"] = leg_std_sum / self.rollout_steps
            record["policy/wheel_std_mean"] = wheel_std_sum / self.rollout_steps
            record["policy/leg_action_abs_mean"] = leg_action_abs_sum / (transitions * 12)
            record["policy/wheel_action_abs_mean"] = wheel_action_abs_sum / (transitions * 4)
            # End-of-rollout snapshots expose persistent lifted-wheel/side
            # support shortcuts that success/level averages can hide. These
            # are not time-integrated contact duty factors.
            contact_sensor = raw_env.scene.sensors.get("contact_forces")
            wheel_names = ("fl_wheel", "fr_wheel", "hl_wheel", "hr_wheel")
            if contact_sensor is not None and all(name in contact_sensor.body_names for name in wheel_names):
                wheel_ids = [contact_sensor.body_names.index(name) for name in wheel_names]
                air = contact_sensor.data.current_air_time[:, wheel_ids]
                support = contact_sensor.data.net_forces_w[:, wheel_ids, 2] > 1.0
                gravity = raw_env.scene["robot"].data.projected_gravity_b
                for family, mask in family_masks.items():
                    if not mask.any():
                        continue
                    prefix = f"posture_snapshot/{family}/"
                    record[prefix + "lateral_gravity_abs_mean"] = float(gravity[mask, 1].abs().mean())
                    record[prefix + "all_wheels_air_fraction"] = float((air[mask] > 0).all(dim=1).float().mean())
                    side_air = ((air[mask, 0] > 0.3) & (air[mask, 2] > 0.3)) | ((air[mask, 1] > 0.3) & (air[mask, 3] > 0.3))
                    record[prefix + "same_side_air_over_03_fraction"] = float(side_air.float().mean())
                    values = torch.stack((support[mask].float().mean(dim=0),
                                          (air[mask] > 1.0).float().mean(dim=0),
                                          air[mask].max(dim=0).values)).cpu().tolist()
                    for wheel, i in zip(("fl", "fr", "hl", "hr"), range(4)):
                        record[prefix + wheel + "_support_fraction"] = values[0][i]
                        record[prefix + wheel + "_air_over_1s_fraction"] = values[1][i]
                        record[prefix + wheel + "_current_air_max_s"] = values[2][i]
            if completed_returns:
                record["episode/completed_count"] = float(sum(value.numel() for value in completed_returns))
                record["episode/return_mean"] = float(torch.cat(completed_returns).mean().item())
                record["episode/length_mean"] = float(torch.cat(completed_lengths).float().mean().item())
            for family in family_masks:
                samples = family_sums[f"{family}/samples"]
                if samples:
                    record[f"terrain/{family}/mean_step_reward"] = family_sums[f"{family}/reward"] / samples
                    record[f"terrain/{family}/vx_abs_error"] = family_sums[f"{family}/vx_error"] / samples
                    record[f"terrain/{family}/omega_abs_error"] = family_sums[f"{family}/omega_error"] / samples
                outcomes = ["success", "failure", "base_contact", "timeout"]
                if boundary_metric_name is not None:
                    outcomes.append(boundary_metric_name)
                for outcome in outcomes:
                    record[f"terrain/{family}/{outcome}_count"] = family_sums[f"{family}/{outcome}"]
                completed = family_sums[f"{family}/success"] + family_sums[f"{family}/failure"]
                cumulative_completed = (
                    self.outcome_totals[f"{family}/success"] + self.outcome_totals[f"{family}/failure"]
                )
                active_terminations = set(raw_env.termination_manager.active_terms)
                success_available = (
                    "obstacle_success" in active_terminations
                    or (family == "platform" and "platform_success" in active_terminations)
                )
                if success_available and family != "flat" and completed:
                    record[f"terrain/{family}/success_rate_window"] = family_sums[f"{family}/success"] / completed
                if success_available and family != "flat" and cumulative_completed:
                    record[f"terrain/{family}/success_rate_cumulative"] = (
                        self.outcome_totals[f"{family}/success"] / cumulative_completed
                    )
            for name, values in log_values.items():
                record[f"environment/{name}"] = sum(values) / len(values)
            command_term = raw_env.command_manager.get_term("base_velocity")
            for label, aliases in (("x", ("x",)), ("yaw", ("yaw", "z"))):
                level = next(
                    (getattr(command_term, f"_command_{axis}_level") for axis in aliases
                     if hasattr(command_term, f"_command_{axis}_level")),
                    None,
                )
                tracking_ema = next(
                    (getattr(command_term, f"_tracking_{axis}_ema") for axis in aliases
                     if hasattr(command_term, f"_tracking_{axis}_ema")),
                    None,
                )
                if level is not None:
                    record[f"curriculum/command_{label}_level_mean"] = float(level.mean().item())
                if tracking_ema is not None:
                    record[f"curriculum/tracking_{label}_ema_mean"] = float(
                        tracking_ema.mean().item()
                    )
            terrain = self.env.unwrapped.scene.terrain
            obstacle = torch.zeros_like(terrain.terrain_types, dtype=torch.bool)
            for family, mask in family_masks.items():
                if family != "flat":
                    obstacle |= mask
            if obstacle.any():
                record["environment/obstacle_mean_level"] = float(
                    terrain.terrain_levels[obstacle].float().mean().item()
                )
                for level in range(terrain.max_terrain_level):
                    record[f"curriculum/level_{level}_fraction"] = float(
                        (terrain.terrain_levels[obstacle] == level).float().mean().item()
                    )
                for family, mask in family_masks.items():
                    if family != "flat" and mask.any():
                        record[f"curriculum/{family}_mean_level"] = float(
                            terrain.terrain_levels[mask].float().mean().item()
                        )
            gait_scale = getattr(raw_env, "_s10_reference_gait_scale_value", None)
            if gait_scale is not None:
                record["curriculum/gait_scale"] = float(gait_scale)
            timings.append(record)
            with (self.log_dir / "metrics.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, sort_keys=True) + "\n")
            print(
                f"[HIM {self.iteration:04d}] total={iteration_time:.3f}s "
                f"collection={collection_time:.3f}s learning={learning_time:.3f}s "
                f"throughput={record['steps_per_second']:.0f} steps/s "
                f"reward={record['mean_step_reward']:.4f} kl={losses['kl']:.4f} "
                f"velocity_loss={losses['velocity']:.5f} swap_loss={losses['swap']:.5f}",
                flush=True,
            )
            displayed_rewards = (
                "track_vx",
                "track_vy",
                "track_vx_vy",
                "track_omega",
                "orientation",
                "base_height",
                "hip_deviation",
                "joint_deviation",
                "joint_mirror",
                "wheel_lateral_clearance",
                "wheel_lateral_target",
                "wheel_pair_geometry",
                "stand_still",
                "leg_joint_acceleration",
                "wheel_joint_acceleration",
                "wheel_air_time",
                "feet_air_time_yaw",
                "wheel_slide_yaw",
                "rotation_gait_status",
                "rotation_gait_symmetry",
                "terminal_failure",
                "failure_penalty",
                "success_bonus",
                "effective_wheel_swing",
                "wheel_air_overstay",
                "wheel_same_side_flight",
            )
            displayed_rewards = tuple(name for name in displayed_rewards if name in reward_names)
            obstacle_success = sum(family_sums[f"{name}/success"] for name in family_masks if name != "flat")
            obstacle_failure = sum(family_sums[f"{name}/failure"] for name in family_masks if name != "flat")
            obstacle_timeouts = sum(family_sums[f"{name}/timeout"] for name in family_masks if name != "flat")
            success_available = any(
                name in active_terminations for name in ("obstacle_success", "platform_success")
            )
            if success_available:
                outcome_label = (
                    f"obstacle outcomes: success={obstacle_success:.0f} "
                    f"failure={obstacle_failure:.0f} timeout={obstacle_timeouts:.0f}"
                )
            else:
                outcome_label = (
                    f"terminal failures={obstacle_failure:.0f} "
                    f"neutral timeouts={obstacle_timeouts:.0f}"
                )
            print(
                "  rewards: "
                + " ".join(
                    f"{name}={record[f'reward/{name}']:+.4f}"
                    for name in displayed_rewards
                )
                + f" | {outcome_label}"
                + f" level={record.get('environment/obstacle_mean_level', 0.0):.2f}"
                + (
                    f" gait={record['curriculum/gait_scale']:.3f}"
                    if "curriculum/gait_scale" in record
                    else ""
                )
                + (
                    f" cmd_x={record['curriculum/command_x_level_mean']:.2f}"
                    f" cmd_yaw={record['curriculum/command_yaw_level_mean']:.2f}"
                    if "curriculum/command_x_level_mean" in record
                    else ""
                )
                + f" |a|>1={record['policy/action_above_1_fraction']:.3f}"
                + f" >5={record['policy/action_above_5_fraction']:.3f}"
                + f" >10={record['policy/action_above_10_fraction']:.3f}",
                flush=True,
            )
            self.iteration += 1
            if self.save_interval > 0 and self.iteration % self.save_interval == 0:
                self.save(self.log_dir / f"model_{self.iteration}.pt")
        return timings

    def _terrain_family_masks(self) -> dict[str, torch.Tensor]:
        """Return masks matching the deterministic 20-column terrain layout."""

        terrain_type = self.env.unwrapped.scene.terrain.terrain_types
        if self.terrain_family_columns is not None:
            return {
                name: (terrain_type >= start) & (terrain_type < stop)
                for name, (start, stop) in self.terrain_family_columns.items()
            }
        return {
            "flat": terrain_type < 7,
            "ramp_up": (terrain_type >= 7) & (terrain_type < 9),
            "ramp_down": (terrain_type >= 9) & (terrain_type < 11),
            "stairs_up": (terrain_type >= 11) & (terrain_type < 14),
            "stairs_down": (terrain_type >= 14) & (terrain_type < 16),
            "platform": terrain_type >= 16,
        }

    def save(self, path: str | Path) -> None:
        path = Path(path)
        os.makedirs(path.parent, exist_ok=True)
        raw_env = self.env.unwrapped
        robot = raw_env.scene["robot"]
        actuator_models = {actuator.is_implicit_model for actuator in robot.actuators.values()}
        if len(actuator_models) != 1:
            raise RuntimeError("cannot save checkpoint with mixed actuator integration models")
        actuator_integration = "implicit" if actuator_models.pop() else "explicit"
        command_term = self.env.unwrapped.command_manager.get_term("base_velocity")
        command_curriculum_state = None
        if hasattr(command_term, "curriculum_state_dict"):
            command_curriculum_state = command_term.curriculum_state_dict()
        torch.save(
            {
                "model_state_dict": self.actor_critic.state_dict(),
                "ppo_optimizer_state_dict": self.algorithm.optimizer.state_dict(),
                "estimator_optimizer_state_dict": self.algorithm.estimator_optimizer.state_dict(),
                "iteration": self.iteration,
                "history_length": self.history_length,
                "one_step_dim": self.one_step_dim,
                "history_order": "newest_first",
                "policy_distribution": self.actor_critic.distribution_type,
                "policy_variant": self.policy_variant,
                "wheel_velocity_observation": (
                    "measured" if getattr(raw_env.cfg, "observe_wheel_velocity", False) else "zero-signal"
                ),
                "actuator_integration": actuator_integration,
                "training_physics_dt": float(raw_env.cfg.sim.dt),
                "training_decimation": int(raw_env.cfg.decimation),
                "initial_noise_std": self.initial_noise_std,
                "entropy_coef": self.algorithm.entropy_coef,
                "terrain_levels": self.env.unwrapped.scene.terrain.terrain_levels.cpu(),
                "torch_rng_state": torch.get_rng_state(),
                "cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                "outcome_totals": dict(self.outcome_totals),
                "command_curriculum_state": command_curriculum_state,
            },
            path,
        )

    def load(self, path: str | Path, *, load_optimizers: bool = True) -> None:
        checkpoint = torch.load(Path(path).expanduser().resolve(), map_location=self.device, weights_only=False)
        for key, expected in (
            ("history_length", self.history_length),
            ("one_step_dim", self.one_step_dim),
        ):
            if int(checkpoint[key]) != int(expected):
                raise ValueError(f"checkpoint {key}={checkpoint[key]} does not match environment {expected}")
        if checkpoint.get("history_order") != "newest_first":
            raise ValueError("checkpoint history order is incompatible")
        if checkpoint.get("policy_distribution") != self.actor_critic.distribution_type:
            raise ValueError(
                "checkpoint policy distribution is incompatible: "
                f"expected {self.actor_critic.distribution_type}, "
                f"got {checkpoint.get('policy_distribution', 'legacy_unsquashed_normal')}"
            )
        if checkpoint.get("policy_variant", "pim_raw") != self.policy_variant:
            raise ValueError(
                f"checkpoint policy_variant={checkpoint.get('policy_variant', 'pim_raw')} "
                f"does not match runner {self.policy_variant}"
            )
        expected_wheel_observation = (
            "measured" if getattr(self.env.unwrapped.cfg, "observe_wheel_velocity", False) else "zero-signal"
        )
        saved_wheel_observation = checkpoint.get("wheel_velocity_observation")
        if saved_wheel_observation is not None and saved_wheel_observation != expected_wheel_observation:
            raise ValueError(
                f"checkpoint wheel_velocity_observation={saved_wheel_observation} does not match "
                f"environment {expected_wheel_observation}"
            )
        actuator_models = {
            actuator.is_implicit_model for actuator in self.env.unwrapped.scene["robot"].actuators.values()
        }
        if len(actuator_models) != 1:
            raise ValueError("environment mixes implicit and explicit actuator integration")
        expected_actuator_integration = "implicit" if actuator_models == {True} else "explicit"
        saved_actuator_integration = checkpoint.get("actuator_integration")
        if saved_actuator_integration is not None and saved_actuator_integration != expected_actuator_integration:
            raise ValueError(
                f"checkpoint actuator_integration={saved_actuator_integration} does not match "
                f"environment {expected_actuator_integration}"
            )
        saved_physics_dt = checkpoint.get("training_physics_dt")
        saved_decimation = checkpoint.get("training_decimation")
        if saved_physics_dt is not None and abs(float(saved_physics_dt) - float(self.env.unwrapped.cfg.sim.dt)) > 1e-12:
            raise ValueError(
                f"checkpoint training_physics_dt={saved_physics_dt} does not match "
                f"environment {self.env.unwrapped.cfg.sim.dt}"
            )
        if saved_decimation is not None and int(saved_decimation) != int(self.env.unwrapped.cfg.decimation):
            raise ValueError(
                f"checkpoint training_decimation={saved_decimation} does not match "
                f"environment {self.env.unwrapped.cfg.decimation}"
            )
        self.actor_critic.load_state_dict(checkpoint["model_state_dict"])
        if "entropy_coef" in checkpoint:
            self.algorithm.entropy_coef = float(checkpoint["entropy_coef"])
        if load_optimizers:
            self.algorithm.optimizer.load_state_dict(checkpoint["ppo_optimizer_state_dict"])
            self.algorithm.estimator_optimizer.load_state_dict(checkpoint["estimator_optimizer_state_dict"])
            self.algorithm.learning_rate = float(self.algorithm.optimizer.param_groups[0]["lr"])
        self.iteration = int(checkpoint["iteration"])
        self.outcome_totals.update(checkpoint.get("outcome_totals", {}))
        saved_levels = checkpoint.get("terrain_levels")
        terrain = self.env.unwrapped.scene.terrain
        if saved_levels is not None:
            if saved_levels.shape != terrain.terrain_levels.shape:
                raise ValueError(
                    f"checkpoint has {saved_levels.numel()} terrain levels, environment has {terrain.terrain_levels.numel()}"
                )
            terrain.terrain_levels.copy_(saved_levels.to(terrain.terrain_levels.device))
            # The M20 gait regularizers cache a scalar derived from the mean
            # terrain level.  Force the first reward evaluation after resume
            # to rebuild it from the restored levels rather than the fresh
            # environment's random initialization.
            raw_env = self.env.unwrapped
            for cache_name in ("_s10_reference_gait_scale", "_s10_reference_gait_scale_value"):
                if hasattr(raw_env, cache_name):
                    delattr(raw_env, cache_name)
            terrain.env_origins[:] = terrain.terrain_origins[terrain.terrain_levels, terrain.terrain_types]
        if "torch_rng_state" in checkpoint:
            torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
        if torch.cuda.is_available() and checkpoint.get("cuda_rng_state_all") is not None:
            torch.cuda.set_rng_state_all([state.cpu() for state in checkpoint["cuda_rng_state_all"]])
        # A resumed run starts fresh episodes on the restored curriculum rows;
        # otherwise robots would still be located at pre-restore origins.
        self.env.reset()
        command_state = checkpoint.get("command_curriculum_state")
        command_term = self.env.unwrapped.command_manager.get_term("base_velocity")
        if command_state is not None:
            if not hasattr(command_term, "load_curriculum_state_dict"):
                raise ValueError("checkpoint contains adaptive-command state but environment does not")
            command_term.load_curriculum_state_dict(command_state)
        print(f"[INFO] resumed PIM-HIM checkpoint={Path(path).expanduser().resolve()} at iteration={self.iteration}")

    def load_transfer(self, path: str | Path, *, reset_critic: bool = True) -> None:
        """Transfer Stage-1 representation/policy without its training state.

        Stage 2 changes both reward scale and terrain semantics, so optimizer,
        curriculum, RNG and outcome counters deliberately start fresh. The
        critic is reset by default because its Stage-1 return target contains
        large terminal success/failure rewards that no longer exist.
        """

        resolved = Path(path).expanduser().resolve()
        checkpoint = torch.load(resolved, map_location=self.device, weights_only=False)
        for key, expected in (("history_length", self.history_length), ("one_step_dim", self.one_step_dim)):
            if int(checkpoint[key]) != int(expected):
                raise ValueError(f"checkpoint {key}={checkpoint[key]} does not match environment {expected}")
        if checkpoint.get("history_order") != "newest_first":
            raise ValueError("checkpoint history order is incompatible")
        if checkpoint.get("policy_distribution") != self.actor_critic.distribution_type:
            raise ValueError(
                "checkpoint policy distribution is incompatible: "
                f"expected {self.actor_critic.distribution_type}, "
                f"got {checkpoint.get('policy_distribution', 'legacy_unsquashed_normal')}"
            )
        if checkpoint.get("policy_variant", "pim_raw") != self.policy_variant:
            raise ValueError(
                f"checkpoint policy_variant={checkpoint.get('policy_variant', 'pim_raw')} "
                f"does not match runner {self.policy_variant}"
            )

        state = checkpoint["model_state_dict"]
        if reset_critic:
            state = {key: value for key, value in state.items() if not key.startswith("critic.")}
        incompatible = self.actor_critic.load_state_dict(state, strict=not reset_critic)
        if reset_critic:
            unexpected = tuple(incompatible.unexpected_keys)
            invalid_missing = tuple(key for key in incompatible.missing_keys if not key.startswith("critic."))
            if unexpected or invalid_missing:
                raise ValueError(
                    f"unexpected transfer state: missing={invalid_missing}, unexpected={unexpected}"
                )
        self.iteration = 0
        self.outcome_totals.clear()
        self.env.reset()
        print(
            f"[INFO] transferred PIM-HIM checkpoint={resolved}; "
            f"critic={'reset' if reset_critic else 'transferred'}, optimizer/curriculum/iteration=fresh",
            flush=True,
        )

    def set_exploration_std(self, noise_std: float | Sequence[float]) -> None:
        """Re-open exploration after a transfer into a changed curriculum."""

        value = torch.as_tensor(noise_std, dtype=torch.float32).reshape(-1)
        if value.numel() == 1:
            value = value.repeat(self.env.num_actions)
        self.actor_critic.set_action_std(value)
        self.initial_noise_std = tuple(float(item) for item in value.tolist())
        print(f"[INFO] exploration std reset to {self.initial_noise_std}", flush=True)

    def load_finetune(
        self,
        path: str | Path,
        *,
        exploration_std: float | Sequence[float],
        preserve_terrain_levels: bool = False,
        keep_environment_terrain_levels: bool = False,
    ) -> None:
        """Continue a mature policy on a new curriculum without its first-step optimizer shock.

        Network and optimizer states remain continuous, while curriculum rows,
        outcome totals, iteration numbering and RNG start fresh.  Exploration
        is deliberately re-opened, so only the Adam moments belonging to
        ``log_std`` are discarded.
        """

        if preserve_terrain_levels and keep_environment_terrain_levels:
            raise ValueError("cannot preserve checkpoint and environment terrain levels simultaneously")
        resolved = Path(path).expanduser().resolve()
        checkpoint = torch.load(resolved, map_location=self.device, weights_only=False)
        for key, expected in (("history_length", self.history_length), ("one_step_dim", self.one_step_dim)):
            if int(checkpoint[key]) != int(expected):
                raise ValueError(f"checkpoint {key}={checkpoint[key]} does not match environment {expected}")
        if checkpoint.get("history_order") != "newest_first":
            raise ValueError("checkpoint history order is incompatible")
        if checkpoint.get("policy_distribution") != self.actor_critic.distribution_type:
            raise ValueError(
                "checkpoint policy distribution is incompatible: "
                f"expected {self.actor_critic.distribution_type}, "
                f"got {checkpoint.get('policy_distribution', 'legacy_unsquashed_normal')}"
            )
        if checkpoint.get("policy_variant", "pim_raw") != self.policy_variant:
            raise ValueError(
                f"checkpoint policy_variant={checkpoint.get('policy_variant', 'pim_raw')} "
                f"does not match runner {self.policy_variant}"
            )

        self.actor_critic.load_state_dict(checkpoint["model_state_dict"])
        self.algorithm.optimizer.load_state_dict(checkpoint["ppo_optimizer_state_dict"])
        self.algorithm.estimator_optimizer.load_state_dict(checkpoint["estimator_optimizer_state_dict"])
        self.algorithm.learning_rate = float(self.algorithm.optimizer.param_groups[0]["lr"])
        if "entropy_coef" in checkpoint:
            self.algorithm.entropy_coef = float(checkpoint["entropy_coef"])

        self.set_exploration_std(exploration_std)
        # This actor uses the reference's directly optimized standard-
        # deviation parameter (``std``), not a log-standard-deviation.
        self.algorithm.optimizer.state.pop(self.actor_critic.std, None)
        self.iteration = 0
        self.outcome_totals.clear()
        terrain = self.env.unwrapped.scene.terrain
        saved_levels = checkpoint.get("terrain_levels") if preserve_terrain_levels else None
        if saved_levels is not None:
            if saved_levels.shape != terrain.terrain_levels.shape:
                raise ValueError(
                    f"checkpoint has {saved_levels.numel()} terrain levels, "
                    f"environment has {terrain.terrain_levels.numel()}"
                )
            # Stage 1.5 and 1.8 share the same twelve normalized difficulty
            # rows.  A scalar row therefore remains meaningful even when a
            # few deterministic terrain columns change family.
            terrain.terrain_levels.copy_(saved_levels.to(terrain.terrain_levels.device))
        elif not keep_environment_terrain_levels:
            terrain.terrain_levels.zero_()
        terrain.env_origins[:] = terrain.terrain_origins[terrain.terrain_levels, terrain.terrain_types]
        self.env.reset()
        print(
            f"[INFO] fine-tuning checkpoint={resolved}; actor/critic/estimator/optimizers retained, "
            "curriculum="
            + (
                "preserved from checkpoint"
                if saved_levels is not None
                else "kept from environment"
                if keep_environment_terrain_levels
                else "reset to zero"
            )
            + ", "
            "RNG/outcomes/iteration fresh, std optimizer state fresh",
            flush=True,
        )
