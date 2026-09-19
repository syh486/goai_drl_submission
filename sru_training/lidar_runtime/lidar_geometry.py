from __future__ import annotations

from dataclasses import dataclass
import importlib.util
from pathlib import Path
import sys

import numpy as np


UPSTREAM_LIDAR_CONFIG_PATH = Path(__file__).resolve().parent / "upstream_lidar_config.py"

FALLBACK_VERTICAL_RAY_ANGLES = (
    -0.07, 0.88, 1.81, 2.76, 3.69, 4.62, 5.54, 6.48, 7.41, 8.34, 9.27, 10.21, 11.15, 12.09, 13.03, 13.98,
    14.92, 15.87, 16.82, 17.77, 18.72, 19.67, 20.62, 21.57, 22.51, 23.45, 24.4, 25.33, 26.28, 27.21, 28.15,
    29.08, 30.02, 30.95, 31.88, 32.82, 33.74, 34.68, 35.62, 36.55, 37.5, 38.43, 39.37, 40.31, 41.25, 42.21,
    43.16, 44.09, 45.05, 46.0, 46.95, 47.9, 48.85, 49.8, 50.73, 51.69, 52.62, 53.56, 54.5, 55.45, 56.37,
    57.3, 58.24, 59.18, 60.12, 61.05, 61.99, 62.93, 63.86, 64.81, 65.76, 66.69, 67.65, 68.6, 69.56, 70.51,
    71.46, 72.42, 73.37, 74.33, 75.29, 76.24, 77.19, 78.14, 79.07, 80.02, 80.96, 81.9, 82.84, 83.78, 84.7,
    85.64, 86.57, 87.52, 88.46, 89.4,
)
FALLBACK_FRONT_POS = (0.38, 0.0, -0.035)
FALLBACK_FRONT_ROT = (0.00084463, 0.7071065, -0.00028154, 0.7071065)
FALLBACK_REAR_POS = (-0.38, 0.0, -0.035)
FALLBACK_REAR_ROT = (0.70703477, 0.0, -0.70717879, 0.0)
FLOAT16_CLIP_ABS = 65504.0
WORLD_Z_INFINITY_THRESHOLD = 9.9
WORLD_Z_METRIC_LIMIT = 3.0


@dataclass(frozen=True)
class LidarGeometry:
    vertical_ray_angles: tuple[float, ...]
    horizontal_fov: float
    horizontal_res: float
    min_range: float
    max_range: float
    front_pos: tuple[float, float, float]
    front_rot: tuple[float, float, float, float]
    rear_pos: tuple[float, float, float]
    rear_rot: tuple[float, float, float, float]
    horizontal_columns_start_at_zero_deg: bool = True

    @property
    def channels(self) -> int:
        return len(self.vertical_ray_angles)

    @property
    def horizontal_samples(self) -> int:
        return int(round(self.horizontal_fov / self.horizontal_res))

    @property
    def horizontal_roll_columns(self) -> int:
        if self.horizontal_columns_start_at_zero_deg and abs(self.horizontal_fov - 360.0) < 1.0e-6:
            return self.horizontal_samples // 2
        return 0


def load_lidar_geometry(upstream_config_path: Path = UPSTREAM_LIDAR_CONFIG_PATH) -> LidarGeometry:
    if upstream_config_path.exists():
        spec = importlib.util.spec_from_file_location("upstream_lidar_config", upstream_config_path)
        if spec is not None and spec.loader is not None:
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
            config = module.DEFAULT_LIDAR_CONFIG
            return LidarGeometry(
                vertical_ray_angles=tuple(float(v) for v in config.vertical_ray_angles),
                horizontal_fov=float(config.horizontal_fov),
                horizontal_res=float(config.horizontal_res),
                min_range=float(config.min_range),
                max_range=float(config.max_range),
                front_pos=tuple(float(v) for v in module.DEFAULT_FRONT_LIDAR_POS),
                front_rot=tuple(float(v) for v in module.DEFAULT_FRONT_LIDAR_ROT),
                rear_pos=tuple(float(v) for v in module.DEFAULT_REAR_LIDAR_POS),
                rear_rot=tuple(float(v) for v in module.DEFAULT_REAR_LIDAR_ROT),
                horizontal_columns_start_at_zero_deg=bool(
                    getattr(config, "horizontal_columns_start_at_zero_deg", True)
                ),
            )

    return LidarGeometry(
        vertical_ray_angles=FALLBACK_VERTICAL_RAY_ANGLES,
        horizontal_fov=360.0,
        horizontal_res=4.0,
        min_range=0.2,
        max_range=10.0,
        front_pos=FALLBACK_FRONT_POS,
        front_rot=FALLBACK_FRONT_ROT,
        rear_pos=FALLBACK_REAR_POS,
        rear_rot=FALLBACK_REAR_ROT,
        horizontal_columns_start_at_zero_deg=True,
    )


def quat_wxyz_to_rotmat(quat: np.ndarray) -> np.ndarray:
    quat = quat.astype(np.float32, copy=False)
    quat = quat / np.clip(np.linalg.norm(quat, axis=-1, keepdims=True), a_min=1.0e-8, a_max=None)
    w = quat[..., 0]
    x = quat[..., 1]
    y = quat[..., 2]
    z = quat[..., 3]

    xx = x * x
    yy = y * y
    zz = z * z
    xy = x * y
    xz = x * z
    yz = y * z
    wx = w * x
    wy = w * y
    wz = w * z

    rot = np.empty(quat.shape[:-1] + (3, 3), dtype=np.float32)
    rot[..., 0, 0] = 1.0 - 2.0 * (yy + zz)
    rot[..., 0, 1] = 2.0 * (xy - wz)
    rot[..., 0, 2] = 2.0 * (xz + wy)
    rot[..., 1, 0] = 2.0 * (xy + wz)
    rot[..., 1, 1] = 1.0 - 2.0 * (xx + zz)
    rot[..., 1, 2] = 2.0 * (yz - wx)
    rot[..., 2, 0] = 2.0 * (xz - wy)
    rot[..., 2, 1] = 2.0 * (yz + wx)
    rot[..., 2, 2] = 1.0 - 2.0 * (xx + yy)
    return rot


def build_sensor_frame_directions(geometry: LidarGeometry) -> np.ndarray:
    """Reproduce IsaacLab's Bpearl pattern and image-space roll exactly."""
    h = (
        np.arange(geometry.horizontal_samples, dtype=np.float32) * np.float32(geometry.horizontal_res)
        - np.float32(geometry.horizontal_fov / 2.0)
    )
    v = np.asarray(geometry.vertical_ray_angles, dtype=np.float32)
    pitch_deg, yaw_deg = np.meshgrid(v, h, indexing="xy")
    pitch = np.deg2rad(pitch_deg.reshape(-1)) + np.pi / 2.0
    yaw = np.deg2rad(yaw_deg.reshape(-1))

    x = np.sin(pitch) * np.cos(yaw)
    y = np.sin(pitch) * np.sin(yaw)
    z = np.cos(pitch)
    dirs = -np.stack([x, y, z], axis=1)

    image = dirs.reshape(geometry.horizontal_samples, geometry.channels, 3).transpose(1, 0, 2)
    if geometry.horizontal_roll_columns != 0:
        image = np.roll(image, shift=-geometry.horizontal_roll_columns, axis=1)
    return image.astype(np.float32, copy=False)


def compute_world_z_channel(
    noisy_d_metric: np.ndarray,
    raw_d_metric: np.ndarray,
    root_pos_w: np.ndarray,
    root_quat_w: np.ndarray,
    sensor_pos_b: tuple[float, float, float] | np.ndarray,
    sensor_rot_b: tuple[float, float, float, float] | np.ndarray,
    dirs_sensor: np.ndarray,
) -> np.ndarray:
    noisy_d_m = noisy_d_metric.astype(np.float32, copy=False)
    raw_d_m = raw_d_metric.astype(np.float32, copy=False)
    root_rot = quat_wxyz_to_rotmat(root_quat_w)
    sensor_pos_b = np.asarray(sensor_pos_b, dtype=np.float32)
    sensor_rot_b = np.asarray(sensor_rot_b, dtype=np.float32)
    sensor_pos_w = root_pos_w.astype(np.float32, copy=False) + np.einsum("nij,j->ni", root_rot, sensor_pos_b)
    sensor_rot = quat_wxyz_to_rotmat(sensor_rot_b)
    dirs_world = np.einsum("nij,jk,hwk->nhwi", root_rot, sensor_rot, dirs_sensor)
    finite_mask = (raw_d_m < WORLD_Z_INFINITY_THRESHOLD) & (noisy_d_m > 0.0)
    world_z = sensor_pos_w[:, None, None, 2] + noisy_d_m * dirs_world[:, :, :, 2]
    world_z = np.where(finite_mask, world_z, 0.0)
    world_z = np.clip(world_z, -WORLD_Z_METRIC_LIMIT, WORLD_Z_METRIC_LIMIT)
    world_z = np.clip(world_z, -FLOAT16_CLIP_ABS, FLOAT16_CLIP_ABS)
    return world_z.astype(np.float16)


def range_image_to_points_base(
    range_image: np.ndarray,
    sensor_dirs: np.ndarray,
    sensor_pos: tuple[float, float, float] | np.ndarray,
    sensor_rot: tuple[float, float, float, float] | np.ndarray,
    min_range: float,
    max_range: float,
    far_clip: float,
) -> np.ndarray:
    clipped = np.clip(range_image.astype(np.float32, copy=False), 0.0, max_range)
    effective_far = min(float(far_clip), float(max_range))
    valid = (clipped > min_range) & (clipped <= effective_far)
    if not np.any(valid):
        return np.zeros((0, 3), dtype=np.float32)

    points_sensor = sensor_dirs[valid] * clipped[valid, None]
    rotation = quat_wxyz_to_rotmat(np.asarray(sensor_rot, dtype=np.float32))
    translation = np.asarray(sensor_pos, dtype=np.float32)
    points_base = points_sensor @ rotation.T + translation[None, :]
    return points_base.astype(np.float32, copy=False)
