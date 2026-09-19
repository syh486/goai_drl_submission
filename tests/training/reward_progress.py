"""Contract checks for the optional Stage 1 dense navigation shaping."""

from __future__ import annotations

from sru_training.s10_mujoco_backend import (
    S10RewardConfig,
    compute_goal_progress_reward,
)


def main() -> None:
    assert S10RewardConfig().goal_progress == 0.0
    assert compute_goal_progress_reward(2.0, 1.0, 0.5) == 0.5
    assert compute_goal_progress_reward(1.0, 2.0, 0.5) == -0.5
    assert compute_goal_progress_reward(1.0, 1.0, 0.5) == 0.0
    try:
        compute_goal_progress_reward(1.0, 0.0, -0.1)
    except ValueError:
        pass
    else:
        raise AssertionError("negative goal-progress coefficients must be rejected")
    print("REWARD_PROGRESS_CONTRACT_OK")


if __name__ == "__main__":
    main()
