"""Check the official S10 ONNX command-to-motion contract on flat MuJoCo."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from sru_training.s10_mujoco_backend import (
    REPO_ROOT,
    S10NativeMujocoBackend,
    quat_wxyz_to_rotmat,
)


CASES = (
    ("forward_0.30", (0.30, 0.0, 0.0)),
    ("forward_0.60", (0.60, 0.0, 0.0)),
    ("yaw_positive_1.0", (0.0, 0.0, 1.0)),
    ("yaw_negative_1.0", (0.0, 0.0, -1.0)),
)


def run_case(
    backend: S10NativeMujocoBackend,
    name: str,
    command: tuple[float, float, float],
    steps: int,
    window: int,
) -> dict[str, float | str | bool]:
    backend.reset()
    values: list[tuple[float, float, float]] = []
    for _ in range(steps):
        backend.step(torch.tensor([command], dtype=torch.float32))
        data = backend.data[0]
        rotation = quat_wxyz_to_rotmat(data.qpos[3:7])
        body_velocity = rotation.T @ data.qvel[:3]
        values.append((float(body_velocity[0]), float(body_velocity[1]), float(data.sensordata[9])))

    steady = np.asarray(values[-window:], dtype=np.float64)
    command_array = np.asarray(command, dtype=np.float64)
    return {
        "case": name,
        "command_vx": float(command_array[0]),
        "command_yaw": float(command_array[2]),
        "body_vx_mean": float(steady[:, 0].mean()),
        "body_vx_std": float(steady[:, 0].std()),
        "body_vy_mean": float(steady[:, 1].mean()),
        "gyro_z_mean": float(steady[:, 2].mean()),
        "gyro_z_std": float(steady[:, 2].std()),
        "response_sign_ok": bool(
            (command_array[0] == 0.0 or np.sign(steady[:, 0].mean()) == np.sign(command_array[0]))
            and (command_array[2] == 0.0 or np.sign(steady[:, 2].mean()) == np.sign(command_array[2]))
        ),
        "done_reason": str(backend.last_done_reason[0]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=REPO_ROOT / "src/S10_sdk_deploy/policy/policy_official_20260828.onnx",
    )
    parser.add_argument("--profile", choices=("legacy", "official_20260828"), default="official_20260828")
    parser.add_argument("--steps", type=int, default=50, help="high-level 5 Hz steps per case")
    parser.add_argument("--steady-window", type=int, default=20)
    args = parser.parse_args()
    if args.steps < 5 or not 1 <= args.steady_window <= args.steps:
        raise ValueError("require --steps >= 5 and 1 <= --steady-window <= --steps")

    backend = S10NativeMujocoBackend(
        num_envs=1,
        xml_path=REPO_ROOT / "src/S10_sdk_deploy/S10_description/s10_mjcf/mjcf/S10.xml",
        task_mode="waypoint",
        reset_mode="fixed",
        low_level="official_onnx",
        low_level_checkpoint=args.checkpoint,
        low_level_profile=args.profile,
        low_level_ready_after_reset=True,
        max_episode_length=args.steps + 10,
        use_lidar=False,
        use_height=False,
        reset_settle_physics_steps=40,
        seed=20260906,
    )
    try:
        print(
            "LOW_LEVEL_CONTRACT",
            {"checkpoint": str(args.checkpoint.resolve()), "profile": args.profile,
             "command_scale": tuple(float(x) for x in backend.onnx_controller.command_scale)},
            flush=True,
        )
        results = [run_case(backend, name, command, args.steps, args.steady_window) for name, command in CASES]
        for result in results:
            print("LOW_LEVEL_RESPONSE", result, flush=True)
        forward = results[:2]
        positive_yaw = results[2]
        negative_yaw = results[3]
        if not all(result["response_sign_ok"] and result["done_reason"] == "none" for result in results):
            raise RuntimeError("official ONNX response has a wrong sign or terminated on flat ground")
        if not (forward[0]["body_vx_mean"] > 0.10 and forward[1]["body_vx_mean"] > forward[0]["body_vx_mean"]):
            raise RuntimeError("forward speed does not track increasing vx command")
        if not (positive_yaw["gyro_z_mean"] > 0.10 and negative_yaw["gyro_z_mean"] < -0.10):
            raise RuntimeError("yaw command does not produce bidirectional yaw response")
        print("S10_LOW_LEVEL_COMMAND_RESPONSE_OK", flush=True)
    finally:
        backend.close()


if __name__ == "__main__":
    main()
