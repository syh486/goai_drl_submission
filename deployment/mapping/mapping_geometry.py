"""Shared geometry primitives for the active GLIM/topometric pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, Sequence

import numpy as np


class LocalSubmapConfig(Protocol):
    submap_aggregate_radius_frames: int
    max_sensor_range_m: float
    submap_voxel_m: float


def load_points(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as payload:
        return np.asarray(payload["points_body_m"], dtype=np.float64)


def voxel_downsample(points: np.ndarray, voxel_m: float) -> np.ndarray:
    values = np.asarray(points, dtype=np.float64)
    values = values[np.isfinite(values).all(axis=1)]
    if not len(values):
        return values.reshape(0, 3)
    keys = np.floor(values / voxel_m).astype(np.int64)
    _, first = np.unique(keys, axis=0, return_index=True)
    return np.ascontiguousarray(values[np.sort(first)])


def transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    return points @ transform[:3, :3].T + transform[:3, 3]


def scan_context_descriptor(
    points: np.ndarray,
    rings: int,
    sectors: int,
    max_range_m: float = 30.0,
) -> np.ndarray:
    """Return an occupancy Scan Context normalized for cyclic comparison."""

    values = np.asarray(points)
    radius = np.linalg.norm(values[:, :2], axis=1)
    valid = (
        (radius >= 0.45)
        & (radius < max_range_m)
        & (values[:, 2] >= -1.5)
        & (values[:, 2] < 4.5)
    )
    ring = np.floor(radius[valid] / max_range_m * rings).astype(np.int64)
    azimuth = np.mod(
        np.arctan2(values[valid, 1], values[valid, 0]), 2.0 * np.pi
    )
    sector = np.floor(
        azimuth / (2.0 * np.pi) * sectors
    ).astype(np.int64)
    ring = np.clip(ring, 0, rings - 1)
    sector = np.clip(sector, 0, sectors - 1)
    descriptor = np.zeros((rings, sectors), dtype=np.float64)
    np.add.at(descriptor, (ring, sector), 1.0)
    descriptor = np.log1p(descriptor)
    norm = np.linalg.norm(descriptor)
    return descriptor / norm if norm > 0.0 else descriptor


def yaw_error_deg(reference: np.ndarray, estimate: np.ndarray) -> float:
    delta = reference[:3, :3].T @ estimate[:3, :3]
    return abs(float(np.degrees(np.arctan2(delta[1, 0], delta[0, 0]))))


def rotation_distance_deg(rotation: np.ndarray) -> float:
    cosine = np.clip((float(np.trace(rotation)) - 1.0) * 0.5, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def anchor_frames(poses: np.ndarray, spacing_m: float) -> list[int]:
    increments = np.linalg.norm(np.diff(poses[:, :3, 3], axis=0), axis=1)
    progress = np.concatenate((np.zeros(1), np.cumsum(increments)))
    targets = np.arange(0.0, progress[-1] + spacing_m * 0.5, spacing_m)
    frames = [int(np.argmin(np.abs(progress - target))) for target in targets]
    if frames[-1] != len(poses) - 1:
        frames.append(len(poses) - 1)
    return sorted(set(frames))


def local_keyframe_cloud(
    poses: np.ndarray,
    frame: int,
    config: LocalSubmapConfig,
    keyframe_files: Sequence[Path],
    *,
    voxel_m: float | None = None,
) -> np.ndarray:
    """Build a query-shaped local map from explicitly aligned scans."""

    files = list(keyframe_files)
    if len(files) != len(poses):
        raise ValueError(
            f"local cloud has {len(files)} keyframes for {len(poses)} poses"
        )
    anchor_inverse = np.linalg.inv(poses[frame])
    chunks = []
    begin = max(0, frame - config.submap_aggregate_radius_frames)
    end = min(len(files), frame + config.submap_aggregate_radius_frames + 1)
    for index in range(begin, end):
        points = load_points(files[index])
        distance = np.linalg.norm(points, axis=1)
        points = points[
            np.isfinite(points).all(axis=1)
            & (distance >= 0.45)
            & (distance <= config.max_sensor_range_m)
        ]
        chunks.append(transform_points(points, anchor_inverse @ poses[index]))
    return voxel_downsample(
        np.concatenate(chunks),
        config.submap_voxel_m if voxel_m is None else voxel_m,
    ).astype(np.float32)
