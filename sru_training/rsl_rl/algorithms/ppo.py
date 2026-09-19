#  Copyright 2021 ETH Zurich, NVIDIA CORPORATION
#  Modified by Fan Yang, ETH Zurich 2025
#  SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.optim as optim

from rsl_rl.algorithms.optim import SingleDeviceMuonWithAuxAdam
from rsl_rl.modules import ActorCritic
from rsl_rl.storage import RolloutStorage

EPSILON = 1e-7

class PPO:
    actor_critic: ActorCritic

    def __init__(
        self,
        actor_critic,
        num_learning_epochs=1,
        num_mini_batches=1,
        clip_param=0.2,
        value_clip_param=0.5,
        gamma=0.998,
        lam=0.95,
        value_loss_coef=1.0,
        entropy_coef=0.0,
        learning_rate=1e-3,
        max_grad_norm=1.0,
        use_clipped_value_loss=True,
        schedule="fixed",
        desired_kl=0.01,
        min_learning_rate=1e-7,
        weight_decay=0.0,
        use_muon=False,
        verify_rollout_contract=False,
        device="cpu",
        **kwargs,
    ):
        self.device = device

        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate
        self.max_learning_rate = learning_rate
        self.min_learning_rate = min_learning_rate
        self.weight_decay = weight_decay

        # PPO components
        self.actor_critic = actor_critic
        self.actor_critic.to(self.device)
        self.storage = None  # initialized later
        self.optimizer = self._create_optimizer(learning_rate, use_muon)
        self.transition = RolloutStorage.Transition()
        self.last_kl_divergence = float("nan")
        self.last_clip_fraction = float("nan")

        # PPO parameters
        self.clip_param = clip_param
        self.value_clip_param = value_clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss
        self.verify_rollout_contract = verify_rollout_contract
        self._rollout_contract_verified = False

    def _verify_saved_rollout(self):
        """Compare stored rollout means with replay from every saved RNN state."""
        if not self.actor_critic.is_recurrent:
            return
        storage = self.storage
        if storage.saved_hidden_states_a is None:
            raise RuntimeError("Recurrent rollout has no saved actor hidden states")

        replayed_mu = []
        masks = torch.ones(1, storage.num_envs, dtype=torch.bool, device=self.device)
        dropout_mask = (
            storage.saved_dropout_masks_a
            if storage.saved_dropout_masks_a is not None
            else None
        )
        with torch.inference_mode():
            for step in range(storage.num_transitions_per_env):
                state = tuple(component[step] for component in storage.saved_hidden_states_a)
                self.actor_critic.act(
                    storage.observations[step : step + 1],
                    masks=masks,
                    hidden_states=state,
                    dropout_masks=dropout_mask,
                )
                replayed_mu.append(self.actor_critic.action_mean.clone())
        replayed_mu = torch.stack(replayed_mu)
        error = (replayed_mu - storage.mu).abs()
        per_step = error.reshape(error.shape[0], -1).amax(dim=1).cpu().tolist()
        print(
            "[PPO rollout contract] saved-state replay "
            f"replayed_shape={tuple(replayed_mu.shape)} stored_shape={tuple(storage.mu.shape)} "
            f"max_abs_error={error.max().item():.9g} "
            f"mean_abs_error={error.mean().item():.9g} "
            "per_step_max=" + ",".join(f"{value:.3g}" for value in per_step),
            flush=True,
        )
        if error.max().item() > 1.0e-4:
            raise RuntimeError(
                "PPO rollout observations, action means, and recurrent states are not aligned"
            )

    def _create_optimizer(self, learning_rate, use_muon):
        if not use_muon:
            print("[PPO] Using Adam optimizer")
            return optim.Adam(
                self.actor_critic.parameters(),
                lr=learning_rate,
                weight_decay=self.weight_decay,
            )

        last_actor_layer = [
            module for module in self.actor_critic.actor.modules() if isinstance(module, nn.Linear)
        ][-1]
        last_critic_layer = [
            module for module in self.actor_critic.critic.modules() if isinstance(module, nn.Linear)
        ][-1]
        hidden_weights = []
        nonhidden_params = []
        for parameter in self.actor_critic.parameters():
            is_output_weight = parameter is last_actor_layer.weight or parameter is last_critic_layer.weight
            if parameter.ndim == 2 and not is_output_weight:
                hidden_weights.append(parameter)
            else:
                nonhidden_params.append(parameter)
        print("[PPO] Using Muon optimizer for hidden weights and AdamW for other parameters")
        return SingleDeviceMuonWithAuxAdam(
            [
                {
                    "params": hidden_weights,
                    "use_muon": True,
                    "lr": learning_rate,
                    "weight_decay": self.weight_decay,
                },
                {
                    "params": nonhidden_params,
                    "use_muon": False,
                    "lr": learning_rate,
                    "betas": (0.9, 0.999),
                    "weight_decay": self.weight_decay,
                },
            ]
        )

    def _update_learning_rate(self, iteration, max_iterations):
        if self.schedule in {"fixed", "adaptive"}:
            return
        progress = iteration / max(max_iterations, 1)
        if self.schedule == "linear":
            progress = min(iteration / max(max_iterations * 0.33, 1.0), 1.0)
            self.learning_rate = self.max_learning_rate - progress * (
                self.max_learning_rate - self.min_learning_rate
            )
        elif self.schedule == "cosine":
            self.learning_rate = 0.5 * self.max_learning_rate * (
                1.0 + math.cos(math.pi * progress)
            )
        elif self.schedule == "exponential":
            self.learning_rate = self.min_learning_rate + (
                self.max_learning_rate - self.min_learning_rate
            ) * math.exp(-5.0 * progress)
        else:
            raise ValueError(f"Unsupported PPO learning-rate schedule: {self.schedule}")
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = self.learning_rate

    def init_storage(self, num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, action_shape):
        self.storage = RolloutStorage(
            num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, action_shape, self.device
        )

    def test_mode(self):
        self.actor_critic.eval()

    def train_mode(self):
        self.actor_critic.train()

    def act(self, obs, critic_obs):
        if self.actor_critic.is_recurrent:
            self.transition.hidden_states = self.actor_critic.get_hidden_states()
        # Compute the actions and values
        # print("obs", obs)
        # print("obs shape", obs.shape)
        self.transition.actions = self.actor_critic.act(obs).detach()
        self.transition.values = self.actor_critic.evaluate(critic_obs).detach()
        self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(self.transition.actions).detach()
        self.transition.action_mean = self.actor_critic.action_mean.detach()
        self.transition.action_sigma = self.actor_critic.action_std.detach()
        # The MuJoCo VecEnv reuses its observation buffers. Snapshot them now:
        # storage.add_transitions() runs after env.step(), by which point a
        # borrowed buffer already contains the next observation.
        self.transition.observations = obs.clone()
        self.transition.critic_observations = critic_obs.clone()
        return self.transition.actions

    def process_env_step(self, rewards, dones, infos):
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones
        # Bootstrapping on time outs
        if "time_outs" in infos:
            self.transition.rewards += self.gamma * torch.squeeze(
                self.transition.values * infos["time_outs"].unsqueeze(1).to(self.device), 1
            )

        # Record the transition
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.actor_critic.reset(dones)

    def compute_returns(self, last_critic_obs):
        last_values = self.actor_critic.evaluate(last_critic_obs).detach()
        self.storage.compute_returns(last_values, self.gamma, self.lam)
        
    def update_dropout_masks(self):
        # update the dropout masks
        self.storage.saved_dropout_masks_a, self.storage.saved_dropout_masks_c = self.actor_critic.get_dropout_masks()
        
    def reset_dropout_masks(self):
        self.actor_critic.reset_dropout_masks()

    def update(self, iter, max_iters):
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_kl_divergence = 0
        mean_clip_fraction = 0
        self._update_learning_rate(iter, max_iters)
        if self.verify_rollout_contract and not self._rollout_contract_verified:
            self._verify_saved_rollout()
        if self.actor_critic.is_recurrent:
            generator = self.storage.reccurent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        for (
            obs_batch,
            critic_obs_batch,
            actions_batch,
            target_values_batch,
            advantages_batch,
            returns_batch,
            old_actions_log_prob_batch,
            old_mu_batch,
            old_sigma_batch,
            hid_states_batch,
            masks_batch,
            dropout_masks_a,
            dropout_masks_c,
        ) in generator:
            self.actor_critic.act(obs_batch, masks=masks_batch, hidden_states=hid_states_batch[0], dropout_masks=dropout_masks_a)
            actions_log_prob_batch = self.actor_critic.get_actions_log_prob(actions_batch)
            value_batch = self.actor_critic.evaluate(
                critic_obs_batch, masks=masks_batch, hidden_states=hid_states_batch[1],
                dropout_masks=dropout_masks_c
            )
            mu_batch = self.actor_critic.action_mean
            sigma_batch = self.actor_critic.action_std
            entropy_batch = self.actor_critic.entropy

            if self.verify_rollout_contract and not self._rollout_contract_verified:
                sequence_error = (mu_batch - old_mu_batch).abs()
                print(
                    "[PPO rollout contract] sequence replay "
                    f"max_abs_error={sequence_error.max().item():.9g} "
                    f"mean_abs_error={sequence_error.mean().item():.9g}",
                    flush=True,
                )
                if sequence_error.max().item() > 1.0e-4:
                    raise RuntimeError(
                        "PPO recurrent sequence replay does not match rollout action means"
                    )
                self._rollout_contract_verified = True

            # Policy KL is useful both for adaptive scheduling and diagnostics.
            with torch.inference_mode():
                kl = torch.sum(
                    torch.log(sigma_batch / (old_sigma_batch + EPSILON) + EPSILON)
                    + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch))
                    / (2.0 * torch.square(sigma_batch))
                    - 0.5,
                    dim=-1,
                )
                kl_mean = torch.mean(kl)

                if self.desired_kl is not None and self.schedule == "adaptive":
                    if kl_mean > self.desired_kl * 2.0:
                        self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                    elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                        self.learning_rate = min(1e-3, self.learning_rate * 1.5)

                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate
                        
            # normalize advantages
            with torch.no_grad():
                advantages_batch = (advantages_batch - advantages_batch.mean()) / (advantages_batch.std() + EPSILON)

            # Surrogate loss
            ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()
            clip_fraction = torch.mean((torch.abs(ratio - 1.0) > self.clip_param).float())
            # print("PPO max ratio: ", ratio.max().item())

            # Value function loss
            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
                    -self.value_clip_param, self.value_clip_param
                )
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean() * 0.5
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean() * 0.5

            loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean()

            # Gradient step
            self.optimizer.zero_grad()
            loss.backward()
            
            # Clip the gradients for actor and critic separately
            nn.utils.clip_grad_norm_(self.actor_critic.get_actor_parameters(), self.max_grad_norm)
            nn.utils.clip_grad_norm_(self.actor_critic.get_critic_parameters(), self.max_grad_norm)
            
            self.optimizer.step()

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_kl_divergence += kl_mean.item()
            mean_clip_fraction += clip_fraction.item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        self.last_kl_divergence = mean_kl_divergence / num_updates
        self.last_clip_fraction = mean_clip_fraction / num_updates
        self.storage.clear()

        return mean_value_loss, mean_surrogate_loss
