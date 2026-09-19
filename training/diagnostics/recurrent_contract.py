"""Check that recurrent rollout and training forwards are numerically equivalent."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from sru_training.s10_policy_config import S10ObservationSpec, ppo_config
from rsl_rl.modules import ActorCriticSRU


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--envs", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=123)
    return parser.parse_args()


def build_policy(device: torch.device) -> ActorCriticSRU:
    spec = S10ObservationSpec()
    cfg = ppo_config(smoke=False)["policy"]
    cfg.pop("class_name")
    policy = ActorCriticSRU(
        spec.actor_obs_dim,
        spec.critic_obs_dim,
        2,
        **cfg,
    ).to(device)
    return policy


def main() -> None:
    args = parse_args()
    if args.steps < 1 or args.envs < 1:
        raise ValueError("--steps and --envs must be positive")

    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    policy = build_policy(device)
    if args.checkpoint is not None:
        payload = torch.load(args.checkpoint.expanduser(), map_location=device, weights_only=False)
        policy.load_state_dict(payload["model_state_dict"])
    policy.eval()

    observations = torch.randn(
        args.steps,
        args.envs,
        policy.num_actor_obs,
        device=device,
    )
    rnn = policy.memory_a.rnn
    h0, c0 = rnn.init_state(args.envs, device)

    policy.memory_a.hidden_states = (h0.clone(), c0.clone())
    step_outputs = []
    with torch.inference_mode():
        for observation in observations:
            step_outputs.append(policy.act_inference(observation).clone())
        step_outputs = torch.stack(step_outputs)

        masks = torch.ones(args.steps, args.envs, dtype=torch.bool, device=device)
        sequence_outputs = policy.act_inference(
            observations,
            masks=masks,
            hidden_states=(h0.clone(), c0.clone()),
        )

    abs_error = (step_outputs - sequence_outputs).abs()
    per_step = abs_error.amax(dim=(1, 2)).cpu().tolist()
    print(f"shape={tuple(step_outputs.shape)}")
    print(f"max_abs_error={abs_error.max().item():.9g}")
    print(f"mean_abs_error={abs_error.mean().item():.9g}")
    print("per_step_max=" + ",".join(f"{value:.3g}" for value in per_step))
    if not torch.allclose(step_outputs, sequence_outputs, atol=1.0e-5, rtol=1.0e-5):
        raise SystemExit("FAILED: recurrent rollout and sequence forwards differ")
    print("PASS")


if __name__ == "__main__":
    main()
