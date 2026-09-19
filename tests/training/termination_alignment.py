"""Check MuJoCo termination classification against the SRU task semantics."""

from __future__ import annotations

import argparse

import mujoco
import numpy as np
import torch

from sru_training.s10_mujoco_backend import (
    DONE_BASE_CONTACT,
    DONE_COMPLETE,
    DONE_LARGE_ANGLE,
    DONE_TIMEOUT,
    DONE_TERRAIN_FALL,
    S10NativeMujocoBackend,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-envs", type=int, default=4)
    args = parser.parse_args()
    backend = S10NativeMujocoBackend(
        num_envs=args.num_envs,
        task_mode="random_goal_sru",
        terrain_profile="stage4_full_no_pits",
        use_lidar=False,
        use_height=False,
        reset_mode="fixed",
        seed=123,
    )
    _, _, dones, info = backend.step(torch.zeros((args.num_envs, 3)))
    assert not bool(dones.any()), info
    assert all(reason == "none" for reason in info["done_reason"]), info

    backend._reset_one(0)
    backend.data[0].qpos[2] = backend.terrain_fall_height - 0.1
    backend.data[0].qvel[:] = 0.0
    mujoco.mj_forward(backend.model, backend.data[0])
    backend._last_illegal_contact_force[0] = 0.0
    backend._illegal_contact_steps[0] = 0
    backend.episode_steps[0] = 1
    backend._reward_done(np.zeros((args.num_envs, 3), dtype=np.float64))
    assert backend.last_done_reason[0] == DONE_TERRAIN_FALL

    backend._reset_one(0)
    backend.data[0].qpos[:3] = (backend.data[0].qpos[0], backend.data[0].qpos[1], 3.0)
    angle = np.deg2rad(50.0)
    backend.data[0].qpos[3:7] = (np.cos(angle / 2.0), np.sin(angle / 2.0), 0.0, 0.0)
    mujoco.mj_forward(backend.model, backend.data[0])
    backend._last_illegal_contact_force[0] = 0.0
    backend._illegal_contact_steps[0] = 0
    backend._reward_done(np.zeros((args.num_envs, 3), dtype=np.float64))
    assert backend.last_done_reason[0] == DONE_LARGE_ANGLE

    backend._reset_one(0)
    # A short impulse can disappear before the final 1000 Hz substep. The
    # 5 Hz termination must use the peak over the policy step, not this last
    # zero-force sample.
    backend._last_illegal_contact_force[0] = 0.0
    backend._max_illegal_contact_force[0] = backend.contact_threshold + 1.0
    backend._illegal_contact_steps[0] = 0
    for _ in range(backend.contact_persistence_steps - 1):
        backend._reward_done(np.zeros((args.num_envs, 3), dtype=np.float64))
    assert backend.last_done_reason[0] == "none"
    backend._reward_done(np.zeros((args.num_envs, 3), dtype=np.float64))
    assert backend.last_done_reason[0] == DONE_BASE_CONTACT

    # Timeout is a time-limit truncation, not a failure termination.
    backend.close()
    backend = S10NativeMujocoBackend(
        num_envs=1,
        task_mode="random_goal_sru",
        terrain_profile="stage4_full_no_pits",
        use_lidar=False,
        use_height=False,
        low_level="none",
        reset_mode="fixed",
        max_episode_length=1,
        seed=123,
    )
    _, _, timeout_done, timeout_info = backend.step(torch.zeros((1, 3)))
    assert bool(timeout_done[0]) and timeout_info["done_reason"][0] == DONE_TIMEOUT, timeout_info
    assert bool(timeout_info["time_outs"][0]), timeout_info
    backend.close()

    # Goal completion follows the original SRU latched-goal phase. First entry
    # starts goal rewards; completion occurs only after the configured hold.
    # It remains a true MDP terminal in the MuJoCo task and must not bootstrap.
    backend = S10NativeMujocoBackend(
        num_envs=1,
        task_mode="random_goal_sru",
        terrain_profile="stage4_full_no_pits",
        use_lidar=False,
        use_height=False,
        low_level="none",
        reset_mode="fixed",
        max_episode_length=100,
        seed=123,
    )
    backend._advance_all = lambda _commands: None  # type: ignore[method-assign]
    backend.random_goal_positions[0] = backend.data[0].qpos[:3]
    _, _, done, goal_info = backend.step(torch.zeros((1, 3)))
    assert not bool(done[0]), goal_info
    assert backend._goal_was_reached[0]
    assert backend._goal_hold_steps[0] == 1
    for _ in range(backend.required_goal_hold_steps - 1):
        _, _, done, goal_info = backend.step(torch.zeros((1, 3)))
        assert not bool(done[0]), goal_info
    _, _, done, goal_info = backend.step(torch.zeros((1, 3)))
    assert bool(done[0]) and goal_info["done_reason"][0] == DONE_COMPLETE, goal_info
    assert not bool(goal_info["time_outs"][0]), goal_info

    print(
        "TERMINATION_ALIGNMENT_OK",
        {
            "num_envs": args.num_envs,
            "contact_threshold": backend.contact_threshold,
            "contact_persistence_steps": backend.contact_persistence_steps,
            "terrain_fall_height": backend.terrain_fall_height,
        },
        flush=True,
    )
    backend.close()


if __name__ == "__main__":
    main()
