"""Summarize S10 TensorBoard training metrics in fixed iteration windows."""

from __future__ import annotations

import argparse
import math
import statistics
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


TERMINATIONS = (
    "goal_complete",
    "large_angle",
    "base_contact",
    "terrain_fall",
    "nonfinite",
    "timeout",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("log_dir", type=Path)
    parser.add_argument("--window", type=int, default=25)
    args = parser.parse_args()
    if args.window < 1:
        raise ValueError("--window must be positive")
    event_files = sorted(args.log_dir.expanduser().resolve().glob("events.out.tfevents.*"))
    if not event_files:
        raise FileNotFoundError(f"No TensorBoard event file in {args.log_dir}")

    accumulator = EventAccumulator(str(event_files[-1]), size_guidance={"scalars": 0})
    accumulator.Reload()
    tags = accumulator.Tags()["scalars"]
    values = {
        tag: {event.step: float(event.value) for event in accumulator.Scalars(tag)}
        for tag in tags
    }
    canonical_tag = "Train/mean_step_reward"
    if canonical_tag not in values:
        raise RuntimeError(f"TensorBoard event file is missing {canonical_tag}")
    # Some runner tags use elapsed seconds as their x-axis. Restrict windows
    # to the canonical per-iteration metric so wall-clock steps are not mixed
    # with training iteration indices.
    steps = sorted(values[canonical_tag])
    if not steps:
        raise RuntimeError("TensorBoard event file contains no scalar metrics")

    def finite_mean(tag: str, selected: list[int]) -> float:
        samples = [values.get(tag, {}).get(step, math.nan) for step in selected]
        samples = [sample for sample in samples if math.isfinite(sample)]
        return statistics.mean(samples) if samples else math.nan

    def total(tag: str, selected: list[int]) -> float:
        return sum(values.get(tag, {}).get(step, 0.0) for step in selected)

    def last(tag: str, selected: list[int]) -> float:
        samples = [
            (step, values.get(tag, {}).get(step, math.nan)) for step in selected
        ]
        samples = [(step, value) for step, value in samples if math.isfinite(value)]
        return samples[-1][1] if samples else math.nan

    print(
        "RUN",
        {
            "event": str(event_files[-1]),
            "first_iteration": steps[0],
            "last_iteration": steps[-1],
            "logged_iterations": len(steps),
        },
    )
    for start in range(steps[0], steps[-1] + 1, args.window):
        selected = [step for step in steps if start <= step < start + args.window]
        if not selected:
            continue
        termination_counts = {
            reason: total(f"Termination/{reason}", selected) for reason in TERMINATIONS
        }
        completed = termination_counts["goal_complete"]
        terminal_count = sum(termination_counts.values())
        print(
            "WINDOW",
            {
                "iterations": f"{selected[0]}-{selected[-1]}",
                "mean_step_reward": round(finite_mean("Train/mean_step_reward", selected), 6),
                "mean_episode_reward": round(
                    finite_mean("Train/iteration_mean_reward", selected), 6
                ),
                "mean_episode_length": round(
                    finite_mean("Train/iteration_mean_episode_length", selected), 3
                ),
                "goal_complete": int(completed),
                "terminal_count": int(terminal_count),
                "success_per_terminal": round(completed / max(terminal_count, 1.0), 6),
                "large_angle": int(termination_counts["large_angle"]),
                "base_contact": int(termination_counts["base_contact"]),
                "timeout": int(termination_counts["timeout"]),
                "value_loss": round(finite_mean("Loss/value_function", selected), 6),
                "surrogate_loss": round(finite_mean("Loss/surrogate", selected), 6),
                "kl": round(
                    finite_mean(
                        "Loss/kl_divergence"
                        if "Loss/kl_divergence" in values
                        else "Loss/policy_kl",
                        selected,
                    ),
                    6,
                ),
                "distill_coef": round(
                    finite_mean("Loss/distill_coefficient", selected), 7
                ),
                "policy1_success": round(
                    finite_mean("PolicyGroup/policy_1_success_rate", selected), 6
                ),
                "policy2_success": round(
                    finite_mean("PolicyGroup/policy_2_success_rate", selected), 6
                ),
                "policy1_segment14_success": round(
                    finite_mean(
                        "PolicyGroup/policy_1_segment_14_success_rate", selected
                    ),
                    6,
                ),
                "policy2_segment14_success": round(
                    finite_mean(
                        "PolicyGroup/policy_2_segment_14_success_rate", selected
                    ),
                    6,
                ),
                "policy1_segment14_attempts": int(
                    total("PolicyGroup/policy_1_segment_14_attempts", selected)
                ),
                "policy2_segment14_attempts": int(
                    total("PolicyGroup/policy_2_segment_14_attempts", selected)
                ),
                "policy1_bridge_samples_total": round(
                    last("PolicyGroup/policy_1_bridge_samples_total", selected), 3
                ),
                "policy2_bridge_samples_total": round(
                    last("PolicyGroup/policy_2_bridge_samples_total", selected), 3
                ),
                "clip_fraction": round(finite_mean("Loss/clip_fraction", selected), 6),
                "learning_rate": round(finite_mean("Loss/learning_rate", selected), 9),
                "fps": round(finite_mean("Perf/total_fps", selected), 3),
            },
        )

    print("LAST_REWARD_COMPONENTS")
    for tag in sorted(tag for tag in tags if tag.startswith("Reward/")):
        events = accumulator.Scalars(tag)
        print(tag, events[-1].value)
    nonfinite = {
        tag: [(event.step, event.value) for event in accumulator.Scalars(tag) if not math.isfinite(event.value)]
        for tag in tags
    }
    nonfinite = {tag: samples for tag, samples in nonfinite.items() if samples}
    print("NONFINITE_METRICS", nonfinite)


if __name__ == "__main__":
    main()
