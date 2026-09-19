"""Longer multi-environment stability and isolation smoke for native MuJoCo."""

from __future__ import annotations

import argparse
import time
from collections import Counter

import numpy as np
import torch

from sru_training.s10_mujoco_backend import S10NativeMujocoBackend, quat_wxyz_to_rotmat


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--no-lidar", action="store_true")
    parser.add_argument("--no-height", action="store_true")
    args = parser.parse_args()

    backend = S10NativeMujocoBackend(
        num_envs=args.num_envs,
        task_mode="random_goal_sru",
        terrain_profile="stage4_full_no_pits",
        max_episode_length=max(args.steps + 5, 50),
        reset_mode="fixed",
        use_lidar=not args.no_lidar,
        use_height=not args.no_height,
    )
    min_z = np.full(args.num_envs, np.inf)
    max_tilt = np.zeros(args.num_envs)
    done_reasons: Counter[str] = Counter()
    start = time.perf_counter()
    for step in range(args.steps):
        # Distinct commands make accidental shared MjData/state visible.
        actions = torch.zeros((args.num_envs, 2))
        actions[:, 0] = torch.linspace(-0.2, 0.2, args.num_envs)
        actions[:, 1] = torch.linspace(-0.15, 0.15, args.num_envs)
        _, rewards, dones, info = backend.step(
            torch.stack((actions[:, 0] * 1.5, torch.zeros(args.num_envs), actions[:, 1]), dim=1)
        )
        assert torch.isfinite(rewards).all()
        for i, data in enumerate(backend.data):
            min_z[i] = min(min_z[i], float(data.qpos[2]))
            gravity = quat_wxyz_to_rotmat(data.qpos[3:7]).T @ np.asarray((0.0, 0.0, -1.0))
            max_tilt[i] = max(max_tilt[i], float(np.linalg.norm(gravity[:2])))
        for reason in info["done_reason"]:
            if reason != "none":
                done_reasons[reason] += 1
        if step % 10 == 0:
            print("STABILITY_STEP", step, info, flush=True)
    elapsed = time.perf_counter() - start
    print(
        "NATIVE_STABILITY_OK",
        {
            "num_envs": args.num_envs,
            "steps": args.steps,
            "elapsed_s": round(elapsed, 3),
            "min_z": np.round(min_z, 3).tolist(),
            "max_tilt_xy": np.round(max_tilt, 3).tolist(),
            "done_reasons": dict(done_reasons),
        },
        flush=True,
    )
    backend.close()


if __name__ == "__main__":
    main()
