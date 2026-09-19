"""HIMLoco network, PPO algorithm, and rollout storage.

This is an isolated S10 adaptation of the earlier TRON2 implementation.  The
actor consumes the newest deployable observation plus velocity/latent features
estimated from six observations.  The target branch and critic may use
privileged observations during training only.
"""

from __future__ import annotations

from collections.abc import Sequence
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


def _activation(name: str) -> nn.Module:
    activations = {
        "elu": nn.ELU,
        "relu": nn.ReLU,
        "selu": nn.SELU,
        "tanh": nn.Tanh,
    }
    try:
        return activations[name.lower()]()
    except KeyError as exc:
        raise ValueError(f"unsupported activation: {name}") from exc


def _mlp(input_dim: int, hidden_dims: Sequence[int], output_dim: int | None, activation: str) -> nn.Sequential:
    layers: list[nn.Module] = []
    current = input_dim
    for width in hidden_dims:
        layers.extend((nn.Linear(current, width), _activation(activation)))
        current = width
    if output_dim is not None:
        layers.append(nn.Linear(current, output_dim))
    return nn.Sequential(*layers)


@torch.no_grad()
def _sinkhorn(scores: torch.Tensor, eps: float = 0.05, iterations: int = 3) -> torch.Tensor:
    q = torch.exp(scores / eps).T
    q /= q.sum() + 1.0e-12
    prototypes, batch = q.shape
    for _ in range(iterations):
        q /= q.sum(dim=1, keepdim=True) + 1.0e-12
        q /= prototypes
        q /= q.sum(dim=0, keepdim=True) + 1.0e-12
        q /= batch
    return (q * batch).T


class PIMHIMCnnEstimator(nn.Module):
    def __init__(
        self,
        *,
        history_length: int,
        one_step_dim: int,
        proprio_dim: int,
        lidar_start: int,
        image_channels: int,
        image_height: int,
        image_width: int,
        target_slices: Sequence[tuple[int, int]],
        velocity_start: int,
        proprio_hidden: Sequence[int] = (128, 64),
        cnn_channels: Sequence[int] = (16, 32, 64),
        lidar_feature_dim: int = 64,
        fusion_hidden: Sequence[int] = (),
        target_hidden: Sequence[int] = (128, 64),
        velocity_dim: int = 3,
        latent_dim: int = 16,
        num_prototypes: int = 32,
        temperature: float = 3.0,
        activation: str = "elu",
        use_lidar: bool = True,
    ) -> None:
        super().__init__()
        self.history_length = history_length
        self.one_step_dim = one_step_dim
        self.proprio_dim = proprio_dim
        self.lidar_start = lidar_start
        self.image_channels = image_channels
        self.image_height = image_height
        self.image_width = image_width
        self.lidar_dim = image_channels * image_height * image_width
        self.velocity_start = velocity_start
        self.velocity_dim = velocity_dim
        self.latent_dim = latent_dim
        self.target_slices = tuple((int(start), int(end)) for start, end in target_slices)
        self.temperature = temperature
        self.use_lidar = bool(use_lidar)

        if self.use_lidar and lidar_start + self.lidar_dim != one_step_dim:
            raise ValueError(
                f"LiDAR must be the final one-step slice: {lidar_start}+{self.lidar_dim}!={one_step_dim}"
            )
        if not self.use_lidar and one_step_dim != proprio_dim:
            raise ValueError(
                f"blind HIM expects proprio-only observations: one_step={one_step_dim}, proprio={proprio_dim}"
            )
        self.proprio_encoder = _mlp(history_length * proprio_dim, proprio_hidden, None, activation)

        if self.use_lidar:
            convolution: list[nn.Module] = []
            channels_in = image_channels
            for index, channels_out in enumerate(cnn_channels):
                convolution.extend(
                    (
                        nn.Conv2d(
                            channels_in,
                            channels_out,
                            kernel_size=3,
                            stride=1 if index == 0 else 2,
                            padding=1,
                        ),
                        _activation(activation),
                    )
                )
                channels_in = channels_out
            conv = nn.Sequential(*convolution)
            with torch.no_grad():
                flat_dim = conv(torch.zeros(1, image_channels, image_height, image_width)).numel()
            self.lidar_encoder: nn.Module | None = nn.Sequential(
                conv,
                nn.Flatten(),
                nn.Linear(flat_dim, lidar_feature_dim),
                _activation(activation),
            )
            fusion_input_dim = proprio_hidden[-1] + lidar_feature_dim
        else:
            self.lidar_encoder = None
            fusion_input_dim = proprio_hidden[-1]
        self.fusion = _mlp(
            fusion_input_dim,
            fusion_hidden,
            velocity_dim + latent_dim,
            activation,
        )
        # Go2W target is critic_obs[:, 3:60]: 54 observable values followed
        # by the privileged 3-D base velocity, for 57 inputs total.
        target_dim = sum(end - start for start, end in self.target_slices)
        self.target = _mlp(target_dim, target_hidden, latent_dim, activation)
        self.prototypes = nn.Embedding(num_prototypes, latent_dim)

    def encode(self, history_flat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        history = history_flat.reshape(-1, self.history_length, self.one_step_dim)
        proprio = history[:, :, : self.proprio_dim].reshape(history.shape[0], -1)
        features = self.proprio_encoder(proprio)
        if self.use_lidar:
            lidar = history[:, 0, self.lidar_start :].reshape(
                history.shape[0], self.image_channels, self.image_height, self.image_width
            )
            assert self.lidar_encoder is not None
            features = torch.cat((features, self.lidar_encoder(lidar)), dim=-1)
        fused = self.fusion(features)
        velocity = fused[:, : self.velocity_dim]
        latent = F.normalize(fused[:, self.velocity_dim :], dim=-1, p=2)
        return velocity, latent

    def forward(self, history_flat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            return self.encode(history_flat)

    def compute_losses(
        self,
        history_flat: torch.Tensor,
        next_critic_obs: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        valid = valid_mask.reshape(-1).bool()
        if not torch.any(valid):
            return None, None
        history_flat = history_flat[valid]
        next_critic_obs = next_critic_obs[valid]
        velocity_target = next_critic_obs[
            :, self.velocity_start : self.velocity_start + self.velocity_dim
        ].detach()
        target_input = torch.cat(
            tuple(next_critic_obs[:, start:end] for start, end in self.target_slices), dim=-1
        ).detach()
        predicted_velocity, source_latent = self.encode(history_flat)
        target_latent = F.normalize(self.target(target_input), dim=-1, p=2)
        with torch.no_grad():
            self.prototypes.weight.copy_(F.normalize(self.prototypes.weight, dim=-1, p=2))
        source_scores = source_latent @ self.prototypes.weight.T
        target_scores = target_latent @ self.prototypes.weight.T
        with torch.no_grad():
            source_assignments = _sinkhorn(source_scores)
            target_assignments = _sinkhorn(target_scores)
        source_log_prob = F.log_softmax(source_scores / self.temperature, dim=-1)
        target_log_prob = F.log_softmax(target_scores / self.temperature, dim=-1)
        swap_loss = -0.5 * (
            source_assignments * target_log_prob + target_assignments * source_log_prob
        ).mean()
        velocity_loss = F.mse_loss(predicted_velocity, velocity_target)
        return velocity_loss, swap_loss


class PIMHIMActorCritic(nn.Module):
    is_recurrent = False
    distribution_type = "go2w_reference_normal"

    def __init__(
        self,
        *,
        history_dim: int,
        critic_dim: int,
        one_step_dim: int,
        action_dim: int,
        estimator: PIMHIMCnnEstimator,
        actor_hidden: Sequence[int] = (512, 256, 128),
        critic_hidden: Sequence[int] = (512, 256, 128),
        activation: str = "elu",
        initial_noise_std: float | Sequence[float] = 1.0,
        actor_observation_mode: str = "full",
    ) -> None:
        super().__init__()
        self.history_dim = history_dim
        self.critic_dim = critic_dim
        self.one_step_dim = one_step_dim
        self.action_dim = action_dim
        self.estimator = estimator
        if actor_observation_mode not in {"full", "proprio"}:
            raise ValueError(f"unsupported actor observation mode: {actor_observation_mode}")
        self.actor_observation_mode = actor_observation_mode
        direct_dim = one_step_dim if actor_observation_mode == "full" else estimator.proprio_dim
        actor_dim = direct_dim + estimator.velocity_dim + estimator.latent_dim
        self.actor = _mlp(actor_dim, actor_hidden, action_dim, activation)
        self.critic = _mlp(critic_dim, critic_hidden, 1, activation)
        # The reference explicitly uses PyTorch's default linear-layer
        # initialization and an unconstrained standard-deviation parameter.
        noise_std = torch.as_tensor(initial_noise_std, dtype=torch.float32)
        if noise_std.ndim == 0:
            noise_std = noise_std.repeat(action_dim)
        if noise_std.shape != (action_dim,) or torch.any(noise_std <= 0.0):
            raise ValueError(f"initial_noise_std must be positive scalar or length-{action_dim} sequence")
        self.std = nn.Parameter(noise_std.clone())
        self.distribution: Normal | None = None
        Normal.set_default_validate_args(False)

    def ppo_parameters(self) -> list[nn.Parameter]:
        return list(self.actor.parameters()) + list(self.critic.parameters()) + [self.std]

    def update_distribution(self, history: torch.Tensor) -> None:
        velocity, latent = self.estimator(history)
        actor_input = self._actor_input(history, velocity, latent)
        mean = self.actor(actor_input)
        # A tiny lower guard is the only intentional numerical safety
        # adaptation; it is inactive at the reference initialization of 1.0.
        std = self.std.clamp_min(1.0e-5)
        self.distribution = Normal(mean, mean * 0.0 + std)

    def act(self, history: torch.Tensor) -> torch.Tensor:
        self.update_distribution(history)
        assert self.distribution is not None
        return self.distribution.sample()

    def act_inference(self, history: torch.Tensor) -> torch.Tensor:
        velocity, latent = self.estimator(history)
        return self.actor(self._actor_input(history, velocity, latent))

    def _actor_input(
        self,
        history: torch.Tensor,
        velocity: torch.Tensor,
        latent: torch.Tensor,
    ) -> torch.Tensor:
        direct_dim = self.one_step_dim if self.actor_observation_mode == "full" else self.estimator.proprio_dim
        return torch.cat((history[:, :direct_dim], velocity, latent), dim=-1)

    def evaluate(self, critic_obs: torch.Tensor) -> torch.Tensor:
        return self.critic(critic_obs)

    @property
    def action_mean(self) -> torch.Tensor:
        assert self.distribution is not None
        return self.distribution.mean

    @property
    def action_std(self) -> torch.Tensor:
        assert self.distribution is not None
        return self.distribution.stddev

    @property
    def entropy(self) -> torch.Tensor:
        assert self.distribution is not None
        return self.distribution.entropy().sum(dim=-1)

    @torch.no_grad()
    def set_action_std(self, noise_std: float | Sequence[float]) -> None:
        """Replace exploration scale without changing policy weights."""

        value = torch.as_tensor(noise_std, dtype=self.std.dtype, device=self.std.device)
        if value.ndim == 0:
            value = value.repeat(self.action_dim)
        if value.shape != (self.action_dim,) or torch.any(value <= 0.0):
            raise ValueError(f"noise_std must be positive scalar or length-{self.action_dim} sequence")
        self.std.copy_(value)

    def log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        assert self.distribution is not None
        return self.distribution.log_prob(actions).sum(dim=-1)

    @torch.no_grad()
    def clamp_noise_std(self) -> None:
        self.std.clamp_(min=1.0e-5)


class PIMHIMRolloutStorage:
    class Transition:
        def __init__(self) -> None:
            self.clear()

        def clear(self) -> None:
            self.observations = None
            self.critic_observations = None
            self.next_critic_observations = None
            self.actions = None
            self.rewards = None
            self.dones = None
            self.values = None
            self.actions_log_prob = None
            self.action_mean = None
            self.action_sigma = None

    def __init__(self, num_envs: int, steps: int, history_dim: int, critic_dim: int, action_dim: int, device: str):
        shape = (steps, num_envs)
        self.observations = torch.zeros(*shape, history_dim, device=device)
        self.critic_observations = torch.zeros(*shape, critic_dim, device=device)
        self.next_critic_observations = torch.zeros(*shape, critic_dim, device=device)
        self.actions = torch.zeros(*shape, action_dim, device=device)
        self.rewards = torch.zeros(*shape, 1, device=device)
        self.dones = torch.zeros(*shape, 1, dtype=torch.bool, device=device)
        self.values = torch.zeros(*shape, 1, device=device)
        self.returns = torch.zeros(*shape, 1, device=device)
        self.advantages = torch.zeros(*shape, 1, device=device)
        self.actions_log_prob = torch.zeros(*shape, 1, device=device)
        self.mu = torch.zeros(*shape, action_dim, device=device)
        self.sigma = torch.zeros(*shape, action_dim, device=device)
        self.steps = steps
        self.num_envs = num_envs
        self.step = 0

    def add(self, transition: "PIMHIMRolloutStorage.Transition") -> None:
        if self.step >= self.steps:
            raise RuntimeError("rollout buffer overflow")
        index = self.step
        self.observations[index].copy_(transition.observations)
        self.critic_observations[index].copy_(transition.critic_observations)
        self.next_critic_observations[index].copy_(transition.next_critic_observations)
        self.actions[index].copy_(transition.actions)
        self.rewards[index].copy_(transition.rewards.view(-1, 1))
        self.dones[index].copy_(transition.dones.view(-1, 1).bool())
        self.values[index].copy_(transition.values)
        self.actions_log_prob[index].copy_(transition.actions_log_prob.view(-1, 1))
        self.mu[index].copy_(transition.action_mean)
        self.sigma[index].copy_(transition.action_sigma)
        self.step += 1

    def compute_returns(self, last_values: torch.Tensor, gamma: float, lam: float) -> None:
        advantage: torch.Tensor | float = 0.0
        for step in reversed(range(self.steps)):
            next_values = last_values if step == self.steps - 1 else self.values[step + 1]
            not_done = 1.0 - self.dones[step].float()
            delta = self.rewards[step] + not_done * gamma * next_values - self.values[step]
            advantage = delta + not_done * gamma * lam * advantage
            self.returns[step] = advantage + self.values[step]
        self.advantages = self.returns - self.values
        self.advantages = (self.advantages - self.advantages.mean()) / (self.advantages.std() + 1.0e-8)

    def batches(self, mini_batches: int, epochs: int):
        total = self.steps * self.num_envs
        flat = (
            self.observations.flatten(0, 1),
            self.critic_observations.flatten(0, 1),
            self.actions.flatten(0, 1),
            self.next_critic_observations.flatten(0, 1),
            self.dones.flatten(0, 1),
            self.values.flatten(0, 1),
            self.advantages.flatten(0, 1),
            self.returns.flatten(0, 1),
            self.actions_log_prob.flatten(0, 1),
            self.mu.flatten(0, 1),
            self.sigma.flatten(0, 1),
        )
        for _ in range(epochs):
            indices = torch.randperm(total, device=self.observations.device)
            # Cover the full rollout even when total is not divisible by the
            # requested number of mini-batches.
            for selected in torch.tensor_split(indices, mini_batches):
                if selected.numel() == 0:
                    continue
                yield tuple(tensor[selected] for tensor in flat)

    def clear(self) -> None:
        self.step = 0


class PIMHIMPPO:
    def __init__(
        self,
        actor_critic: PIMHIMActorCritic,
        storage: PIMHIMRolloutStorage,
        *,
        learning_rate: float = 3.0e-4,
        estimator_learning_rate: float = 1.0e-3,
        epochs: int = 5,
        mini_batches: int = 4,
        gamma: float = 0.99,
        lam: float = 0.95,
        clip: float = 0.2,
        value_coef: float = 1.0,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 1.0,
        estimator_max_grad_norm: float = 10.0,
        desired_kl: float = 0.01,
        min_learning_rate: float = 1.0e-5,
        max_learning_rate: float = 1.0e-2,
    ) -> None:
        self.actor_critic = actor_critic
        self.storage = storage
        self.transition = storage.Transition()
        self.optimizer = torch.optim.Adam(actor_critic.ppo_parameters(), lr=learning_rate)
        self.estimator_optimizer = torch.optim.Adam(
            actor_critic.estimator.parameters(), lr=estimator_learning_rate
        )
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.mini_batches = mini_batches
        self.gamma = gamma
        self.lam = lam
        self.clip = clip
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.estimator_max_grad_norm = estimator_max_grad_norm
        self.desired_kl = desired_kl
        self.min_learning_rate = min_learning_rate
        self.max_learning_rate = max_learning_rate

    def act(self, history: torch.Tensor, critic_obs: torch.Tensor) -> torch.Tensor:
        actions = self.actor_critic.act(history)
        self.transition.actions = actions.detach()
        self.transition.values = self.actor_critic.evaluate(critic_obs).detach()
        self.transition.actions_log_prob = self.actor_critic.log_prob(actions).detach()
        self.transition.action_mean = self.actor_critic.action_mean.detach()
        self.transition.action_sigma = self.actor_critic.action_std.detach()
        self.transition.observations = history
        self.transition.critic_observations = critic_obs
        return actions.detach()

    def process_step(
        self,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        infos: dict,
        next_critic_obs: torch.Tensor,
    ) -> None:
        self.transition.next_critic_observations = next_critic_obs
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones
        if "time_outs" in infos:
            self.transition.rewards += self.gamma * torch.squeeze(
                self.transition.values * infos["time_outs"].unsqueeze(1), dim=1
            )
        self.storage.add(self.transition)
        self.transition.clear()

    def compute_returns(self, critic_obs: torch.Tensor) -> None:
        self.storage.compute_returns(self.actor_critic.evaluate(critic_obs).detach(), self.gamma, self.lam)

    def update(self) -> dict[str, float]:
        totals = {"value": 0.0, "surrogate": 0.0, "velocity": 0.0, "swap": 0.0, "kl": 0.0}
        count = 0
        for (
            history,
            critic_obs,
            actions,
            next_critic_obs,
            dones,
            old_values,
            advantages,
            returns,
            old_log_prob,
            old_mu,
            old_sigma,
        ) in self.storage.batches(self.mini_batches, self.epochs):
            self.actor_critic.update_distribution(history)
            log_prob = self.actor_critic.log_prob(actions)
            values = self.actor_critic.evaluate(critic_obs)
            mu = self.actor_critic.action_mean
            sigma = self.actor_critic.action_std
            entropy = self.actor_critic.entropy
            with torch.no_grad():
                kl = torch.sum(
                    torch.log(sigma / old_sigma + 1.0e-5)
                    + (old_sigma.square() + (old_mu - mu).square()) / (2.0 * sigma.square())
                    - 0.5,
                    dim=-1,
                ).mean()
                if kl > self.desired_kl * 2.0:
                    self.learning_rate = max(self.min_learning_rate, self.learning_rate / 1.5)
                elif 0.0 < kl < self.desired_kl / 2.0:
                    self.learning_rate = min(self.max_learning_rate, self.learning_rate * 1.5)
                # The estimator has its own supervised/contrastive objective
                # and explicit learning rate. PPO's adaptive KL schedule must
                # only modify the policy/value optimizer.
                for group in self.optimizer.param_groups:
                    group["lr"] = self.learning_rate

            ratio = torch.exp(log_prob - old_log_prob.squeeze(-1))
            surrogate = -advantages.squeeze(-1) * ratio
            clipped_surrogate = -advantages.squeeze(-1) * ratio.clamp(1.0 - self.clip, 1.0 + self.clip)
            surrogate_loss = torch.maximum(surrogate, clipped_surrogate).mean()
            clipped_values = old_values + (values - old_values).clamp(-self.clip, self.clip)
            value_loss = torch.maximum((values - returns).square(), (clipped_values - returns).square()).mean()
            ppo_loss = surrogate_loss + self.value_coef * value_loss - self.entropy_coef * entropy.mean()
            self.optimizer.zero_grad()
            ppo_loss.backward()
            nn.utils.clip_grad_norm_(self.actor_critic.ppo_parameters(), self.max_grad_norm)
            self.optimizer.step()
            self.actor_critic.clamp_noise_std()

            velocity_loss, swap_loss = self.actor_critic.estimator.compute_losses(
                history, next_critic_obs, torch.ones_like(dones, dtype=torch.bool)
            )
            if velocity_loss is not None:
                # The reference drives the estimator optimizer with PPO's
                # adaptive learning rate rather than a fixed secondary rate.
                for group in self.estimator_optimizer.param_groups:
                    group["lr"] = self.learning_rate
                self.estimator_optimizer.zero_grad()
                (velocity_loss + swap_loss).backward()
                nn.utils.clip_grad_norm_(
                    self.actor_critic.estimator.parameters(), self.estimator_max_grad_norm
                )
                self.estimator_optimizer.step()
                totals["velocity"] += float(velocity_loss.item())
                totals["swap"] += float(swap_loss.item())
            totals["value"] += float(value_loss.item())
            totals["surrogate"] += float(surrogate_loss.item())
            totals["kl"] += float(kl.item())
            count += 1
        self.storage.clear()
        means = {name: value / max(count, 1) for name, value in totals.items()}
        means["ppo_learning_rate"] = float(self.optimizer.param_groups[0]["lr"])
        means["estimator_learning_rate"] = float(self.estimator_optimizer.param_groups[0]["lr"])
        return means
