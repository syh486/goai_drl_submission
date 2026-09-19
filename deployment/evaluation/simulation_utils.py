"""Simulation-only helpers for evaluating onboard localization."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import torch

from deployment.common.math_utils import quat_wxyz_to_rotmat
from sru_training.s10_lidar_encoder import INVALID_RANGE_THRESHOLD_M, MIN_RANGE_M


def goal_body(pose: np.ndarray, target: np.ndarray) -> np.ndarray:
    pose = np.asarray(pose, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if pose.shape == (7,):
        delta = quat_wxyz_to_rotmat(pose[3:7]).T @ (target - pose[:3])
    elif pose.shape == (4,):
        yaw = float(pose[3])
        world_delta = target - pose[:3]
        cosine, sine = np.cos(yaw), np.sin(yaw)
        delta = np.asarray(
            (
                cosine * world_delta[0] + sine * world_delta[1],
                -sine * world_delta[0] + cosine * world_delta[1],
                world_delta[2],
            )
        )
    else:
        raise ValueError(
            "pose must be [x,y,z,yaw] or [x,y,z,qw,qx,qy,qz], "
            f"got {pose.shape}"
        )
    distance = max(float(np.linalg.norm(delta)), 1.0e-6)
    return np.concatenate((delta / distance, np.asarray((np.log1p(distance),))))


def replace_goal(state, goal: np.ndarray):
    return replace(
        state,
        goal_body=torch.as_tensor(goal, dtype=torch.float32).unsqueeze(0),
    )


def angular_goal_error(estimated: np.ndarray, truth: np.ndarray) -> float:
    estimated_direction = estimated[:3] / max(
        float(np.linalg.norm(estimated[:3])), 1.0e-9
    )
    truth_direction = truth[:3] / max(float(np.linalg.norm(truth[:3])), 1.0e-9)
    return float(
        np.degrees(
            np.arccos(np.clip(np.dot(estimated_direction, truth_direction), -1.0, 1.0))
        )
    )


def perturb_scans(
    scans: tuple[np.ndarray, np.ndarray],
    rng: np.random.Generator,
    *,
    range_noise_std: float,
    dropout_rate: float,
) -> tuple[np.ndarray, np.ndarray]:
    if range_noise_std <= 0.0 and dropout_rate <= 0.0:
        return scans
    perturbed = []
    for scan in scans:
        value = np.asarray(scan, dtype=np.float32).copy()
        valid = (value > MIN_RANGE_M) & (value < INVALID_RANGE_THRESHOLD_M)
        if range_noise_std > 0.0:
            value[valid] += rng.normal(0.0, range_noise_std, int(valid.sum()))
            value[valid] = np.clip(
                value[valid],
                MIN_RANGE_M + 1.0e-3,
                INVALID_RANGE_THRESHOLD_M - 1.0e-3,
            )
        if dropout_rate > 0.0:
            value[valid & (rng.random(value.shape) < dropout_rate)] = (
                INVALID_RANGE_THRESHOLD_M
            )
        perturbed.append(value)
    return perturbed[0], perturbed[1]


def perturb_history(
    history: dict[str, np.ndarray],
    rng: np.random.Generator,
    *,
    accel_bias: np.ndarray,
    gyro_bias: np.ndarray,
    accel_noise_std: float,
    gyro_noise_std: float,
    wheel_scale_noise_std: float,
) -> dict[str, np.ndarray]:
    value = {key: np.asarray(item).copy() for key, item in history.items()}
    value["accelerometer"] += accel_bias
    value["gyro"] += gyro_bias
    if accel_noise_std > 0.0:
        value["accelerometer"] += rng.normal(
            0.0, accel_noise_std, value["accelerometer"].shape
        )
    if gyro_noise_std > 0.0:
        value["gyro"] += rng.normal(0.0, gyro_noise_std, value["gyro"].shape)
    if wheel_scale_noise_std > 0.0:
        value["wheel_qvel"] *= 1.0 + rng.normal(
            0.0, wheel_scale_noise_std, value["wheel_qvel"].shape
        )
    return value
