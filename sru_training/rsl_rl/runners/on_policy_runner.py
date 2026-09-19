#  Copyright 2021 ETH Zurich, NVIDIA CORPORATION
#  Modified by Fan Yang, ETH Zurich 2025
#  SPDX-License-Identifier: BSD-3-Clause

"""On-policy runner for PPO, SPO, and MDPO algorithms."""

from __future__ import annotations

import math
import os
import random
import statistics
import time
from collections import Counter, deque

import numpy as np
import torch
try:
    from torch.utils.tensorboard import SummaryWriter as TensorboardSummaryWriter
except ModuleNotFoundError:  # TensorBoard is optional for framework-only smoke tests.
    class TensorboardSummaryWriter:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            pass

        def __getattr__(self, _name):
            return lambda *args, **kwargs: None

import rsl_rl
from rsl_rl.algorithms import MDPO, PPO, SPO
from rsl_rl.env import VecEnv
from rsl_rl.modules import ActorCritic, ActorCriticRecurrent, ActorCriticSRU, EmpiricalNormalization
from rsl_rl.utils import store_code_state, VideoRecorder


class OnPolicyRunner:
    """On-policy runner for training and evaluation.

    Supports PPO, SPO, and MDPO algorithms. Automatically detects the algorithm
    type and adapts behavior accordingly (e.g., MDPO uses two actor-critics).
    """

    def __init__(self, env: VecEnv, train_cfg, log_dir=None, device="cpu"):
        self.cfg = train_cfg
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.device = device
        self.env = env

        # Video recording (initialized lazily, enabled via set_video_recording)
        self.video_recorder: VideoRecorder | None = None

        obs, extras = self.env.get_observations()
        num_obs = obs.shape[1]
        if "critic" in extras["observations"]:
            num_critic_obs = extras["observations"]["critic"].shape[1]
        else:
            num_critic_obs = num_obs
        actor_critic_class = eval(self.policy_cfg.pop("class_name"))
        print("num obs", num_obs)
        print("num critic obs", num_critic_obs)

        alg_class = eval(self.alg_cfg.pop("class_name"))

        # Determine if this is MDPO (Multi-Distillation Policy Optimization)
        self.is_mdpo = alg_class == MDPO

        if self.is_mdpo:
            # MDPO uses two actor-critics
            actor_critic_1: ActorCritic | ActorCriticRecurrent | ActorCriticSRU = actor_critic_class(
                num_obs, num_critic_obs, self.env.num_actions, **self.policy_cfg
            ).to(self.device)
            actor_critic_2: ActorCritic | ActorCriticRecurrent | ActorCriticSRU = actor_critic_class(
                num_obs, num_critic_obs, self.env.num_actions, **self.policy_cfg
            ).to(self.device)
            self.alg = alg_class(actor_critic_1, actor_critic_2, device=self.device, **self.alg_cfg)
        else:
            # Standard algorithms use one actor-critic
            actor_critic: ActorCritic | ActorCriticRecurrent | ActorCriticSRU = actor_critic_class(
                num_obs, num_critic_obs, self.env.num_actions, **self.policy_cfg
            ).to(self.device)
            self.alg = alg_class(actor_critic, device=self.device, **self.alg_cfg)

        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]
        self.empirical_normalization = self.cfg["empirical_normalization"]
        if self.empirical_normalization:
            self.obs_normalizer = EmpiricalNormalization(shape=[num_obs], until=1.0e8).to(self.device)
            self.critic_obs_normalizer = EmpiricalNormalization(shape=[num_critic_obs], until=1.0e8).to(self.device)
        else:
            self.obs_normalizer = torch.nn.Identity()
            self.critic_obs_normalizer = torch.nn.Identity()

        # init storage and model
        self.alg.init_storage(
            self.env.num_envs,
            self.num_steps_per_env,
            [num_obs],
            [num_critic_obs],
            [self.env.num_actions],
        )

        # Log
        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0
        self.git_status_repos = [rsl_rl.__file__]

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False):
        """Run the training loop.

        Args:
            num_learning_iterations: Number of training iterations.
            init_at_random_ep_len: If True, randomize initial episode lengths.
        """
        # initialize writer
        if self.log_dir is not None and self.writer is None:
            self.logger_type = self.cfg.get("logger", "tensorboard")
            self.logger_type = self.logger_type.lower()

            if self.logger_type == "neptune":
                from rsl_rl.utils.neptune_utils import NeptuneSummaryWriter

                self.writer = NeptuneSummaryWriter(log_dir=self.log_dir, flush_secs=10, cfg=self.cfg)
                self.writer.log_config(self.env.cfg, self.cfg, self.alg_cfg, self.policy_cfg)
            elif self.logger_type == "wandb":
                from rsl_rl.utils.wandb_utils import WandbSummaryWriter

                self.writer = WandbSummaryWriter(log_dir=self.log_dir, flush_secs=10, cfg=self.cfg)
                self.writer.log_config(self.env.cfg, self.cfg, self.alg_cfg, self.policy_cfg)
            elif self.logger_type == "tensorboard":
                self.writer = TensorboardSummaryWriter(log_dir=self.log_dir, flush_secs=10)
            else:
                raise AssertionError("logger type not found")

        if init_at_random_ep_len:
            randomize_episode_lengths = getattr(
                self.env, "randomize_episode_lengths", None
            )
            if randomize_episode_lengths is None:
                self.env.episode_length_buf = torch.randint_like(
                    self.env.episode_length_buf,
                    high=int(self.env.max_episode_length),
                )
            else:
                randomize_episode_lengths()
        obs, extras = self.env.get_observations()
        critic_obs = extras["observations"].get("critic", obs)
        obs, critic_obs = obs.to(self.device), critic_obs.to(self.device)
        self.train_mode()

        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        # Reward shifting is configured by the selected training preset.
        reward_shifting_value = self.cfg.get("reward_shifting_value", 0.0)

        start_iter = self.current_learning_iteration
        tot_iter = start_iter + num_learning_iterations
        schedule_max_iterations = int(
            self.cfg.get("schedule_max_iterations", tot_iter)
        )
        if schedule_max_iterations <= 0:
            raise ValueError("schedule_max_iterations must be positive")
        if schedule_max_iterations < tot_iter:
            raise ValueError(
                "schedule_max_iterations cannot end before the requested training run: "
                f"schedule={schedule_max_iterations}, requested_end={tot_iter}"
            )

        for it in range(start_iter, tot_iter):
            start = time.time()
            iteration_done_reason_counts = Counter()
            iteration_policy_done_reason_counts = (Counter(), Counter())
            iteration_policy_segment_attempts = (Counter(), Counter())
            iteration_policy_segment_successes = (Counter(), Counter())
            iteration_reward_component_sums = Counter()
            iteration_step_reward_sum = 0.0
            iteration_episode_rewards = []
            iteration_episode_lengths = []
            iteration_segment_sampling = None
            iteration_entry_state_sampling = None
            iteration_single_waypoints_reached = 0

            # Check if we should start recording video this iteration
            if self.video_recorder:
                # Debug: print every 100 iterations to track progress
                if it % 100 == 0:
                    print(f"[DEBUG VideoRecorder] Iteration {it}, checking should_record...")
                if self.video_recorder.should_record(it):
                    self.video_recorder.start_recording()

            # Rollout
            with torch.no_grad():
                for i in range(self.num_steps_per_env):
                    actions = self.alg.act(obs, critic_obs)
                    obs, rewards, dones, infos = self.env.step(actions)

                    iteration_step_reward_sum += float(rewards.mean().item())
                    for name, value in infos.get("reward_components", {}).items():
                        iteration_reward_component_sums[str(name)] += float(value)

                    done_reasons = infos.get("done_reason", ())
                    terminal_starts = infos.get("start_waypoint_indices", ())
                    for env_index, reason in enumerate(done_reasons):
                        if reason != "none":
                            iteration_done_reason_counts[str(reason)] += 1
                            if self.is_mdpo:
                                policy_group = 0 if env_index % 2 else 1
                                iteration_policy_done_reason_counts[policy_group][
                                    str(reason)
                                ] += 1
                                if env_index < len(terminal_starts):
                                    segment = int(terminal_starts[env_index])
                                    iteration_policy_segment_attempts[policy_group][
                                        segment
                                    ] += 1
                                    if reason == "goal_complete":
                                        iteration_policy_segment_successes[policy_group][
                                            segment
                                        ] += 1
                    iteration_single_waypoints_reached += int(
                        sum(infos.get("waypoints_reached_this_step", ()))
                    )
                    if infos.get("segment_sampling") is not None:
                        iteration_segment_sampling = infos["segment_sampling"]
                    if infos.get("entry_state_sampling") is not None:
                        iteration_entry_state_sampling = infos[
                            "entry_state_sampling"
                        ]

                    # Capture video frame if recording
                    # Continue capturing until video_length is reached
                    if self.video_recorder and self.video_recorder.is_recording:
                        self.video_recorder.capture_frame()

                    # Match the upstream runner: reward shifting is effective
                    # only for the dual-policy MDPO path, even though the
                    # shared IsaacLab config object contains the field for PPO.
                    if self.is_mdpo and reward_shifting_value != 0.0:
                        rewards = rewards + reward_shifting_value

                    obs = self.obs_normalizer(obs)
                    if "critic" in infos["observations"]:
                        critic_obs = self.critic_obs_normalizer(infos["observations"]["critic"])
                    else:
                        critic_obs = obs
                    obs, critic_obs, rewards, dones = (
                        obs.to(self.device),
                        critic_obs.to(self.device),
                        rewards.to(self.device),
                        dones.to(self.device),
                    )
                    self.alg.process_env_step(rewards, dones, infos)

                    if self.log_dir is not None:
                        if "episode" in infos:
                            ep_infos.append(infos["episode"])
                        elif "log" in infos:
                            ep_infos.append(infos["log"])

                        # Shift rewards back for logging
                        log_rewards = rewards
                        if self.is_mdpo and reward_shifting_value != 0.0:
                            log_rewards = rewards - reward_shifting_value

                        cur_reward_sum += log_rewards
                        cur_episode_length += 1
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        if len(new_ids):
                            finished_rewards = cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist()
                            finished_lengths = cur_episode_length[new_ids][:, 0].cpu().numpy().tolist()
                            rewbuffer.extend(finished_rewards)
                            lenbuffer.extend(finished_lengths)
                            iteration_episode_rewards.extend(finished_rewards)
                            iteration_episode_lengths.extend(finished_lengths)
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0

                stop = time.time()
                collection_time = stop - start

                # Learning step
                start = stop
                self.alg.compute_returns(critic_obs)

                # update dropout masks
                self.alg.update_dropout_masks()

            # Update returns different values based on algorithm type
            update_result = self.alg.update(it, schedule_max_iterations)
            if self.is_mdpo:
                mean_value_loss, mean_surrogate_loss, mean_kl_divergence = update_result
            else:
                mean_value_loss, mean_surrogate_loss = update_result
                mean_kl_divergence = None

            stop = time.time()
            learn_time = stop - start
            self.current_learning_iteration = it

            # reset dropout masks
            self.alg.reset_dropout_masks()

            if self.log_dir is not None:
                iteration_episode_count = len(iteration_episode_rewards)
                iteration_episode_mean_reward = (
                    statistics.mean(iteration_episode_rewards) if iteration_episode_rewards else float("nan")
                )
                iteration_episode_mean_length = (
                    statistics.mean(iteration_episode_lengths) if iteration_episode_lengths else float("nan")
                )
                self.log(locals())

                # Log video only if recording is complete (reached video_length frames)
                if self.video_recorder and self.video_recorder.is_recording and self.video_recorder.is_complete():
                    self.video_recorder.log_video(self.writer, it, self.logger_type)

            if it % self.save_interval == 0:
                self.save(os.path.join(self.log_dir, f"model_{it}.pt"))
            ep_infos.clear()
            if it == start_iter:
                git_file_paths = store_code_state(self.log_dir, self.git_status_repos)
                if self.logger_type in ["wandb", "neptune"] and git_file_paths:
                    for path in git_file_paths:
                        self.writer.save_file(path)

        self.save(os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt"))

    def log(self, locs: dict, width: int = 80, pad: int = 35):
        """Log training statistics."""
        self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
        self.tot_time += locs["collection_time"] + locs["learn_time"]
        iteration_time = locs["collection_time"] + locs["learn_time"]

        ep_string = ""
        if locs["ep_infos"]:
            for key in locs["ep_infos"][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs["ep_infos"]:
                    if key not in ep_info:
                        continue
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                if "/" in key:
                    self.writer.add_scalar(key, value, locs["it"])
                    ep_string += f"""{f'{key}:':>{pad}} {value:.4f}\n"""
                else:
                    self.writer.add_scalar("Episode/" + key, value, locs["it"])
                    ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""

        # Get action std from appropriate actor-critic
        if self.is_mdpo:
            mean_std = self.alg.actor_critic_1.action_std.mean()
            mean_std_2 = self.alg.actor_critic_2.action_std.mean()
        else:
            mean_std = self.alg.actor_critic.action_std.mean()
            mean_std_2 = None

        collection_fps = self.num_steps_per_env * self.env.num_envs / max(locs["collection_time"], 1.0e-9)
        fps = int(self.num_steps_per_env * self.env.num_envs / max(locs["collection_time"] + locs["learn_time"], 1.0e-9))

        self.writer.add_scalar("Loss/value_function", locs["mean_value_loss"], locs["it"])
        self.writer.add_scalar("Loss/surrogate", locs["mean_surrogate_loss"], locs["it"])
        if self.is_mdpo and locs["mean_kl_divergence"] is not None:
            self.writer.add_scalar("Loss/kl_divergence", locs["mean_kl_divergence"], locs["it"])
            self.writer.add_scalar(
                "Loss/distill_coefficient",
                self.alg.current_distill_coef,
                locs["it"],
            )
        elif hasattr(self.alg, "last_kl_divergence"):
            self.writer.add_scalar("Loss/policy_kl", self.alg.last_kl_divergence, locs["it"])
            self.writer.add_scalar("Loss/clip_fraction", self.alg.last_clip_fraction, locs["it"])
        self.writer.add_scalar("Loss/learning_rate", self.alg.learning_rate, locs["it"])
        self.writer.add_scalar("Policy/mean_noise_std", mean_std.item(), locs["it"])
        if mean_std_2 is not None:
            self.writer.add_scalar(
                "PolicyGroup/policy_1_noise_std", mean_std.item(), locs["it"]
            )
            self.writer.add_scalar(
                "PolicyGroup/policy_2_noise_std", mean_std_2.item(), locs["it"]
            )
        self.writer.add_scalar("Perf/total_fps", fps, locs["it"])
        self.writer.add_scalar("Perf/collection_fps", collection_fps, locs["it"])
        self.writer.add_scalar("Perf/collection time", locs["collection_time"], locs["it"])
        self.writer.add_scalar("Perf/learning_time", locs["learn_time"], locs["it"])
        self.writer.add_scalar("Train/iteration_episode_count", locs["iteration_episode_count"], locs["it"])
        if locs["iteration_episode_count"]:
            self.writer.add_scalar(
                "Train/iteration_mean_reward", locs["iteration_episode_mean_reward"], locs["it"]
            )
            self.writer.add_scalar(
                "Train/iteration_mean_episode_length", locs["iteration_episode_mean_length"], locs["it"]
            )
        for reason in ("timeout", "goal_complete", "terrain_fall", "large_angle", "base_contact", "nonfinite"):
            self.writer.add_scalar(
                f"Termination/{reason}", locs["iteration_done_reason_counts"].get(reason, 0), locs["it"]
            )
        if self.is_mdpo:
            for policy_group, counts in enumerate(
                locs["iteration_policy_done_reason_counts"], start=1
            ):
                terminal_count = sum(counts.values())
                self.writer.add_scalar(
                    f"PolicyGroup/policy_{policy_group}_terminal_count",
                    terminal_count,
                    locs["it"],
                )
                if terminal_count:
                    self.writer.add_scalar(
                        f"PolicyGroup/policy_{policy_group}_success_rate",
                        counts.get("goal_complete", 0) / terminal_count,
                        locs["it"],
                    )
                attempts = locs["iteration_policy_segment_attempts"][
                    policy_group - 1
                ].get(14, 0)
                successes = locs["iteration_policy_segment_successes"][
                    policy_group - 1
                ].get(14, 0)
                self.writer.add_scalar(
                    f"PolicyGroup/policy_{policy_group}_segment_14_attempts",
                    attempts,
                    locs["it"],
                )
                if attempts:
                    self.writer.add_scalar(
                        f"PolicyGroup/policy_{policy_group}_segment_14_success_rate",
                        successes / attempts,
                        locs["it"],
                    )
        curriculum = locs.get("iteration_segment_sampling")
        if curriculum is not None and curriculum.get("enabled"):
            success_values = curriculum["success_ema"]
            self.writer.add_scalar(
                "Curriculum/effective_segments", curriculum["effective_segments"], locs["it"]
            )
            self.writer.add_scalar(
                "Curriculum/mean_success_ema", statistics.mean(success_values), locs["it"]
            )
            self.writer.add_scalar(
                "Curriculum/min_success_ema", min(success_values), locs["it"]
            )
            self.writer.add_scalar(
                "Curriculum/total_episodes", curriculum["total_episodes"], locs["it"]
            )
            for segment, success, probability in zip(
                curriculum["segments"],
                success_values,
                curriculum["probabilities"],
            ):
                self.writer.add_scalar(
                    f"Curriculum/segment_{segment:02d}_success_ema", success, locs["it"]
                )
                self.writer.add_scalar(
                    f"Curriculum/segment_{segment:02d}_probability", probability, locs["it"]
                )
            for segment, progress in zip(
                curriculum["segments"],
                curriculum.get(
                    "learning_progress", [0.0] * len(curriculum["segments"])
                ),
            ):
                self.writer.add_scalar(
                    f"Curriculum/segment_{segment:02d}_learning_progress",
                    progress,
                    locs["it"],
                )
        entry_sampling = locs.get("iteration_entry_state_sampling")
        if entry_sampling is not None and entry_sampling.get("enabled"):
            self.writer.add_scalar(
                "Reset/entry_state_applied_total",
                entry_sampling["applied_total"],
                locs["it"],
            )
            self.writer.add_scalar(
                "Reset/entry_state_missing_fallbacks_total",
                entry_sampling["missing_segment_fallbacks"],
                locs["it"],
            )
        self.writer.add_scalar(
            "Train/mean_step_reward",
            locs["iteration_step_reward_sum"] / self.num_steps_per_env,
            locs["it"],
        )
        self.writer.add_scalar(
            "Waypoint/single_reached",
            locs["iteration_single_waypoints_reached"],
            locs["it"],
        )
        for name, value in locs["iteration_reward_component_sums"].items():
            self.writer.add_scalar(f"Reward/{name}", value / self.num_steps_per_env, locs["it"])
        if len(locs["rewbuffer"]) > 0:
            self.writer.add_scalar("Train/mean_reward", statistics.mean(locs["rewbuffer"]), locs["it"])
            self.writer.add_scalar("Train/mean_episode_length", statistics.mean(locs["lenbuffer"]), locs["it"])
            if self.logger_type != "wandb":
                self.writer.add_scalar("Train/mean_reward/time", statistics.mean(locs["rewbuffer"]), self.tot_time)
                self.writer.add_scalar(
                    "Train/mean_episode_length/time", statistics.mean(locs["lenbuffer"]), self.tot_time
                )

        log_str = f" \033[1m Learning iteration {locs['it']}/{locs['tot_iter']} \033[0m "

        if len(locs["rewbuffer"]) > 0:
            log_string = (
                f"""{'#' * width}\n"""
                f"""{log_str.center(width, ' ')}\n\n"""
                f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs['collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
            )
            if self.is_mdpo and locs["mean_kl_divergence"] is not None:
                log_string += f"""{'KL divergence:':>{pad}} {locs['mean_kl_divergence']:.4f}\n"""
                log_string += f"""{'Distill coefficient:':>{pad}} {self.alg.current_distill_coef:.6f}\n"""
            log_string += (
                f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
                f"""{'Rolling100 reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
                f"""{'Rolling100 episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n"""
            )
        else:
            log_string = (
                f"""{'#' * width}\n"""
                f"""{log_str.center(width, ' ')}\n\n"""
                f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs['collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
            )

        log_string += (
            f"""{'Collection FPS:':>{pad}} {collection_fps:.2f}\n"""
            f"""{'Mean step reward:':>{pad}} {locs['iteration_step_reward_sum'] / self.num_steps_per_env:.4f}\n"""
            f"""{'Episodes this iteration:':>{pad}} {locs['iteration_episode_count']}\n"""
        )
        if locs["iteration_episode_count"]:
            log_string += (
                f"""{'Iteration episode reward:':>{pad}} {locs['iteration_episode_mean_reward']:.2f}\n"""
                f"""{'Iteration episode length:':>{pad}} {locs['iteration_episode_mean_length']:.2f}\n"""
            )
        reason_string = ", ".join(
            f"{key}={value}" for key, value in sorted(locs["iteration_done_reason_counts"].items())
        ) or "none"
        log_string += f"""{'Terminations this iteration:':>{pad}} {reason_string}\n"""
        curriculum = locs.get("iteration_segment_sampling")
        if curriculum is not None and curriculum.get("enabled"):
            ranked = sorted(
                zip(
                    curriculum["segments"],
                    curriculum["success_ema"],
                    curriculum["probabilities"],
                ),
                key=lambda item: item[2],
                reverse=True,
            )[:5]
            hard_string = ", ".join(
                f"s{segment}:ema={success:.2f}/p={probability:.3f}"
                for segment, success, probability in ranked
            )
            log_string += (
                f"{'Adaptive sampling:':>{pad}} warmup={not curriculum['warmup_complete']} "
                f"strategy={curriculum.get('strategy', 'difficulty')} "
                f"effective={curriculum['effective_segments']:.1f} {hard_string}\n"
            )
        entry_sampling = locs.get("iteration_entry_state_sampling")
        if entry_sampling is not None and entry_sampling.get("enabled"):
            log_string += (
                f"{'Entry-state resets:':>{pad}} "
                f"applied={entry_sampling['applied_total']} "
                f"missing_fallback={entry_sampling['missing_segment_fallbacks']} "
                f"coverage={len(entry_sampling['available_segments'])}\n"
            )
        reward_component_string = ", ".join(
            f"{key}={value / self.num_steps_per_env:.4f}"
            for key, value in sorted(locs["iteration_reward_component_sums"].items())
        ) or "none"
        log_string += f"""{'Reward components:':>{pad}} {reward_component_string}\n"""

        log_string += ep_string
        log_string += (
            f"""{'-' * width}\n"""
            f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
            f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
            f"""{'Total time:':>{pad}} {self.tot_time:.2f}s\n"""
            f"""{'ETA:':>{pad}} {self.tot_time / (locs['it'] - locs['start_iter'] + 1) * (locs['tot_iter'] - locs['it'] - 1):.1f}s\n"""
        )
        print(log_string)

    def save(self, path, infos=None):
        """Save the model checkpoint."""
        if infos is None:
            get_training_state = getattr(self.env, "get_training_state", None)
            if get_training_state is not None:
                infos = {"env_training_state": get_training_state()}
        if self.is_mdpo:
            saved_dict = {
                "model_state_dict": self.alg.actor_critic_1.state_dict(),
                "model_state_dict_2": self.alg.actor_critic_2.state_dict(),
                "optimizer_state_dict": self.alg.optimizer_1.state_dict(),
                "optimizer_state_dict_2": self.alg.optimizer_2.state_dict(),
                "iter": self.current_learning_iteration,
                "next_iter": self.current_learning_iteration + 1,
                "infos": infos,
                "algorithm_state_dict": self._algorithm_training_state_dict(),
                "rng_state_dict": self._rng_training_state_dict(),
                "runner_protocol": self._runner_protocol(),
            }
        else:
            saved_dict = {
                "model_state_dict": self.alg.actor_critic.state_dict(),
                "optimizer_state_dict": self.alg.optimizer.state_dict(),
                "iter": self.current_learning_iteration,
                "next_iter": self.current_learning_iteration + 1,
                "infos": infos,
                "algorithm_state_dict": self._algorithm_training_state_dict(),
                "rng_state_dict": self._rng_training_state_dict(),
                "runner_protocol": self._runner_protocol(),
            }
        if self.empirical_normalization:
            saved_dict["obs_norm_state_dict"] = self.obs_normalizer.state_dict()
            saved_dict["critic_obs_norm_state_dict"] = self.critic_obs_normalizer.state_dict()
        torch.save(saved_dict, path)

        if self.logger_type in ["neptune", "wandb"]:
            self.writer.save_model(path, self.current_learning_iteration)

    def load(self, path, load_optimizer=True):
        """Load a model checkpoint."""
        loaded_dict = torch.load(path, map_location=self.device, weights_only=True)
        if load_optimizer and loaded_dict.get("runner_protocol") != self._runner_protocol():
            raise ValueError(
                "checkpoint runner protocol does not match the current training config; "
                "use task-transfer initialization for a changed protocol"
            )
        if self.is_mdpo:
            self.alg.actor_critic_1.load_state_dict(loaded_dict["model_state_dict"], strict=True)
            self.alg.actor_critic_2.load_state_dict(
                loaded_dict.get("model_state_dict_2", loaded_dict["model_state_dict"]),
                strict=True,
            )
            if load_optimizer:
                self.alg.optimizer_1.load_state_dict(loaded_dict["optimizer_state_dict"])
                self.alg.optimizer_2.load_state_dict(
                    loaded_dict.get("optimizer_state_dict_2", loaded_dict["optimizer_state_dict"])
                )
        else:
            self.alg.actor_critic.load_state_dict(loaded_dict["model_state_dict"], strict=True)
            if load_optimizer:
                self.alg.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
        if load_optimizer:
            self._restore_algorithm_training_state(loaded_dict.get("algorithm_state_dict"))
        if self.empirical_normalization:
            self.obs_normalizer.load_state_dict(loaded_dict["obs_norm_state_dict"])
            self.critic_obs_normalizer.load_state_dict(loaded_dict["critic_obs_norm_state_dict"])
        # Legacy checkpoints only stored the last completed iteration. Start
        # from the following one so resume never repeats and overwrites it.
        self.current_learning_iteration = int(
            loaded_dict.get("next_iter", int(loaded_dict["iter"]) + 1)
        )
        if load_optimizer:
            self._restore_rng_training_state(loaded_dict.get("rng_state_dict"))
        return loaded_dict.get("infos")

    def _runner_protocol(self):
        """Behavioral optimizer/rollout settings required for exact resume."""

        protocol = {
            "version": 1,
            "algorithm_class": type(self.alg).__name__,
            "policy_class": (
                type(self.alg.actor_critic_1).__name__
                if self.is_mdpo else type(self.alg.actor_critic).__name__
            ),
            "seed": self.cfg.get("seed"),
            "num_steps_per_env": self.num_steps_per_env,
            "empirical_normalization": self.empirical_normalization,
            "reward_shifting_value": self.cfg.get("reward_shifting_value", 0.0),
            "policy": self.policy_cfg,
            "algorithm": self.alg_cfg,
        }
        if self.cfg.get("schedule_max_iterations") is not None:
            protocol["schedule_max_iterations"] = int(
                self.cfg["schedule_max_iterations"]
            )
        return protocol

    @staticmethod
    def _rng_training_state_dict():
        numpy_state = np.random.get_state()
        return {
            "python": random.getstate(),
            "numpy": {
                "bit_generator": numpy_state[0],
                # This PyTorch build cannot serialize uint32 storage. MT19937
                # keys are losslessly representable as signed int64.
                "keys": torch.from_numpy(numpy_state[1].astype(np.int64)),
                "position": int(numpy_state[2]),
                "has_gauss": int(numpy_state[3]),
                "cached_gaussian": float(numpy_state[4]),
            },
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        }

    @staticmethod
    def _restore_rng_training_state(state_dict):
        if not isinstance(state_dict, dict):
            raise ValueError(
                "checkpoint has no global RNG state; it cannot be used for exact resume"
            )
        random.setstate(state_dict["python"])
        numpy_state = state_dict["numpy"]
        np.random.set_state(
            (
                str(numpy_state["bit_generator"]),
                numpy_state["keys"].cpu().numpy().astype(np.uint32, copy=False),
                int(numpy_state["position"]),
                int(numpy_state["has_gauss"]),
                float(numpy_state["cached_gaussian"]),
            )
        )
        torch.set_rng_state(state_dict["torch_cpu"].cpu())
        cuda_states = state_dict.get("torch_cuda", [])
        if cuda_states:
            if not torch.cuda.is_available():
                raise ValueError(
                    "checkpoint contains CUDA RNG state but CUDA is unavailable"
                )
            if len(cuda_states) != torch.cuda.device_count():
                raise ValueError(
                    "checkpoint CUDA RNG state does not match visible CUDA devices"
                )
            torch.cuda.set_rng_state_all([value.cpu() for value in cuda_states])

    def _algorithm_training_state_dict(self):
        return {
            "learning_rate": float(self.alg.learning_rate),
            "schedule": str(self.alg.schedule),
        }

    def _restore_algorithm_training_state(self, state_dict):
        if state_dict is not None:
            self.alg.learning_rate = float(state_dict["learning_rate"])
            self.alg.schedule = str(state_dict["schedule"])
        else:
            optimizer = self.alg.optimizer_1 if self.is_mdpo else self.alg.optimizer
            self.alg.learning_rate = float(optimizer.param_groups[0]["lr"])

        optimizers = (
            (self.alg.optimizer_1, self.alg.optimizer_2)
            if self.is_mdpo
            else (self.alg.optimizer,)
        )
        for optimizer in optimizers:
            for param_group in optimizer.param_groups:
                param_group["lr"] = self.alg.learning_rate

    def initialize_from_checkpoint(self, path):
        """Load model weights without restoring iteration or optimizer state."""

        loaded_dict = torch.load(path, map_location=self.device, weights_only=True)
        state_dict = loaded_dict["model_state_dict"]
        if self.is_mdpo:
            self.alg.actor_critic_1.load_state_dict(state_dict, strict=True)
            self.alg.actor_critic_2.load_state_dict(
                loaded_dict.get("model_state_dict_2", state_dict), strict=True
            )
        else:
            self.alg.actor_critic.load_state_dict(state_dict, strict=True)
        return loaded_dict.get("infos")

    @staticmethod
    def _load_actor_parameters(actor_critic, state_dict):
        """Copy only actor parameters, leaving a newly initialized critic intact."""

        actor_parameters = list(actor_critic.get_actor_parameters())
        actor_parameter_ids = {id(parameter) for parameter in actor_parameters}
        named_actor_parameters = {
            name: parameter
            for name, parameter in actor_critic.named_parameters()
            if id(parameter) in actor_parameter_ids
        }
        if len(named_actor_parameters) != len(actor_parameters):
            raise RuntimeError("Actor parameter list contains duplicate or unnamed parameters")
        missing = sorted(set(named_actor_parameters) - set(state_dict))
        if missing:
            raise KeyError(f"Checkpoint is missing actor parameters: {missing}")
        with torch.no_grad():
            for name, parameter in named_actor_parameters.items():
                source = state_dict[name]
                if source.shape != parameter.shape:
                    raise ValueError(
                        f"Actor parameter shape mismatch for {name}: "
                        f"checkpoint={tuple(source.shape)} model={tuple(parameter.shape)}"
                    )
                parameter.copy_(source)
        return tuple(sorted(named_actor_parameters))

    def recover_actor_from_checkpoint(self, path):
        """Recover actor weights and iteration while resetting critic and optimizer."""

        loaded_dict = torch.load(path, map_location=self.device, weights_only=True)
        state_dict = loaded_dict["model_state_dict"]
        if self.is_mdpo:
            self._load_actor_parameters(self.alg.actor_critic_1, state_dict)
            self._load_actor_parameters(
                self.alg.actor_critic_2,
                loaded_dict.get("model_state_dict_2", state_dict),
            )
        else:
            self._load_actor_parameters(self.alg.actor_critic, state_dict)
        self.current_learning_iteration = int(
            loaded_dict.get("next_iter", int(loaded_dict["iter"]) + 1)
        )
        return loaded_dict.get("infos")

    def initialize_actor_from_checkpoint(self, path):
        """Initialize checkpoint actors while leaving critics and optimizers fresh."""

        loaded_dict = torch.load(path, map_location=self.device, weights_only=True)
        state_dict = loaded_dict["model_state_dict"]
        if self.is_mdpo:
            self._load_actor_parameters(self.alg.actor_critic_1, state_dict)
            self._load_actor_parameters(
                self.alg.actor_critic_2,
                loaded_dict.get("model_state_dict_2", state_dict),
            )
        else:
            self._load_actor_parameters(self.alg.actor_critic, state_dict)
        self.current_learning_iteration = 0
        return loaded_dict.get("infos")

    @staticmethod
    def _load_task_transfer_parameters(actor_critic, state_dict):
        """Load all parameters except the scalar value output layer."""

        critic_layers = [
            module
            for module in actor_critic.critic.modules()
            if isinstance(module, torch.nn.Linear)
        ]
        if not critic_layers:
            raise RuntimeError("Task transfer requires a linear critic value head")
        value_head = critic_layers[-1]
        value_head_ids = {id(parameter) for parameter in value_head.parameters()}
        value_head_names = {
            name
            for name, parameter in actor_critic.named_parameters()
            if id(parameter) in value_head_ids
        }
        merged = actor_critic.state_dict()
        missing = sorted(set(merged) - set(state_dict))
        if missing:
            raise KeyError(f"Checkpoint is missing task-transfer state: {missing}")
        for name in merged:
            if name not in value_head_names:
                merged[name] = state_dict[name]
        actor_critic.load_state_dict(merged, strict=True)
        return tuple(sorted(value_head_names))

    def initialize_task_transfer_from_checkpoint(
        self,
        path,
        policy_2_path=None,
        *,
        policy_1_state_key="model_state_dict",
        policy_2_state_key=None,
    ):
        """Keep actors and critic features, resetting only scalar value heads.

        MDPO may initialize either slot from an explicit state key in one or
        two checkpoints. This preserves complementary specialists instead of
        forcing both policies to inherit the same curriculum bias before
        mutual distillation.
        """

        loaded_dict = torch.load(path, map_location=self.device, weights_only=True)
        if policy_1_state_key not in loaded_dict:
            raise KeyError(f"{path} does not contain {policy_1_state_key}")
        state_dict = loaded_dict[policy_1_state_key]
        if self.is_mdpo:
            policy_2_dict = (
                torch.load(policy_2_path, map_location=self.device, weights_only=True)
                if policy_2_path is not None
                else loaded_dict
            )
            resolved_policy_2_key = policy_2_state_key
            if resolved_policy_2_key is None:
                resolved_policy_2_key = (
                    "model_state_dict"
                    if policy_2_path is not None
                    else (
                        "model_state_dict_2"
                        if "model_state_dict_2" in loaded_dict
                        else policy_1_state_key
                    )
                )
            if resolved_policy_2_key not in policy_2_dict:
                source = policy_2_path if policy_2_path is not None else path
                raise KeyError(f"{source} does not contain {resolved_policy_2_key}")
            heads_1 = self._load_task_transfer_parameters(
                self.alg.actor_critic_1, state_dict
            )
            heads_2 = self._load_task_transfer_parameters(
                self.alg.actor_critic_2,
                policy_2_dict[resolved_policy_2_key],
            )
            reset_heads = (heads_1, heads_2)
        else:
            reset_heads = (
                self._load_task_transfer_parameters(self.alg.actor_critic, state_dict),
            )
        self.current_learning_iteration = 0
        return loaded_dict.get("infos"), reset_heads

    def set_action_std(self, action_std):
        """Set action noise for all active policies and return their previous means."""

        action_std = float(action_std)
        if action_std <= 0.0:
            raise ValueError("action_std must be positive")
        actor_critics = (
            (self.alg.actor_critic_1, self.alg.actor_critic_2)
            if self.is_mdpo
            else (self.alg.actor_critic,)
        )
        previous_stds = []
        with torch.no_grad():
            for actor_critic in actor_critics:
                previous_stds.append(float(actor_critic.log_std.exp().mean().item()))
                actor_critic.log_std.fill_(math.log(action_std))
        return previous_stds

    def get_inference_policy(self, device=None):
        """Get the inference policy function."""
        self.eval_mode()
        if self.is_mdpo:
            actor_critic = self.alg.actor_critic_1
        else:
            actor_critic = self.alg.actor_critic

        if device is not None:
            actor_critic.to(device)
        policy = actor_critic.act_inference
        if self.cfg["empirical_normalization"]:
            if device is not None:
                self.obs_normalizer.to(device)
            policy = lambda x: actor_critic.act_inference(self.obs_normalizer(x))  # noqa: E731
        return policy

    def get_policy_reset(self, device=None):
        """Get the policy reset function."""
        self.eval_mode()
        if self.is_mdpo:
            actor_critic = self.alg.actor_critic_1
        else:
            actor_critic = self.alg.actor_critic

        if device is not None:
            actor_critic.to(device)
        return actor_critic.reset

    def train_mode(self):
        """Switch to training mode."""
        if self.is_mdpo:
            self.alg.train_mode()
        else:
            self.alg.actor_critic.train()
        if self.empirical_normalization:
            self.obs_normalizer.train()
            self.critic_obs_normalizer.train()

    def eval_mode(self):
        """Switch to evaluation mode."""
        if self.is_mdpo:
            self.alg.test_mode()
        else:
            self.alg.actor_critic.eval()
        if self.empirical_normalization:
            self.obs_normalizer.eval()
            self.critic_obs_normalizer.eval()

    def add_git_repo_to_log(self, repo_file_path):
        """Add a git repository to track for logging."""
        self.git_status_repos.append(repo_file_path)

    def set_video_recording(
        self, enable: bool, video_length: int = 200, video_interval: int = 2000, fps: int = 30, save_local: bool = True
    ):
        """Configure video recording during training.

        Video recording can upload to WandB and/or save locally as MP4 files.

        Requirements:
        1. Environment with render_mode="rgb_array"
        2. For WandB upload: logger="wandb" in agent config
        3. For local save: imageio package installed

        Args:
            enable: Whether to enable video recording.
            video_length: Number of environment steps per video. Defaults to 200.
            video_interval: Number of training iterations between video recordings. Defaults to 2000.
            fps: Frames per second for the recorded video. Defaults to 30.
            save_local: Whether to save videos locally as MP4 files. Defaults to True.
        """
        if enable:
            self.video_recorder = VideoRecorder(
                env=self.env,
                video_length=video_length,
                video_interval=video_interval,
                fps=fps,
                save_local=save_local,
                log_dir=self.log_dir,
            )
            self.video_recorder.enable()
        else:
            if self.video_recorder:
                self.video_recorder.disable()
            self.video_recorder = None
