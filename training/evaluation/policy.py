"""Shared SRU policy construction for current random-terrain evaluation."""

from __future__ import annotations

import torch

from sru_training.s10_policy_config import ppo_config, observation_spec_from_policy_state
from rsl_rl.modules import ActorCriticSRU


def build_policy(state_dict: dict[str, torch.Tensor], device: str) -> ActorCriticSRU:
    """Reconstruct a deterministic actor from a PPO/MDPO-compatible state dict."""

    policy_config = dict(ppo_config(smoke=False)["policy"])
    policy_config.pop("class_name")
    hidden_key = "memory_a.rnn.cells.0.transform_gate.weight"
    policy_config["rnn_hidden_size"] = int(state_dict[hidden_key].shape[0])
    spec = observation_spec_from_policy_state(state_dict)
    policy = ActorCriticSRU(
        spec.actor_obs_dim,
        spec.critic_obs_dim,
        2,
        **policy_config,
    ).to(device)
    policy.load_state_dict(state_dict, strict=True)
    policy.eval()
    return policy
