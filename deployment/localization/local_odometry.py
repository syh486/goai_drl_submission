"""Local dual-LiDAR, IMU, and wheel odometry without a global map."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
import time

import numpy as np

from deployment.common.math_utils import quat_wxyz_to_rotmat
from deployment.common.lidar_geometry import (
    INVALID_RANGE_THRESHOLD_M,
    MIN_RANGE_M,
    S10_FRONT_POS,
    S10_FRONT_ROT_WXYZ,
    S10_REAR_POS,
    S10_REAR_ROT_WXYZ,
    build_sensor_frame_directions,
)


GRAVITY = 9.81


def _skew(vector: np.ndarray) -> np.ndarray:
    x, y, z = np.asarray(vector, dtype=np.float64)
    return np.asarray(((0.0, -z, y), (z, 0.0, -x), (-y, x, 0.0)))


def _so3_exp(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float64)
    angle = float(np.linalg.norm(vector))
    if angle < 1.0e-8:
        return np.eye(3) + _skew(vector)
    axis_skew = _skew(vector / angle)
    return np.eye(3) + np.sin(angle) * axis_skew + (1.0 - np.cos(angle)) * (axis_skew @ axis_skew)


def _so3_log(rotation: np.ndarray) -> np.ndarray:
    rotation = np.asarray(rotation, dtype=np.float64)
    cosine = np.clip((float(np.trace(rotation)) - 1.0) * 0.5, -1.0, 1.0)
    angle = float(np.arccos(cosine))
    vector = np.asarray(
        (rotation[2, 1] - rotation[1, 2], rotation[0, 2] - rotation[2, 0], rotation[1, 0] - rotation[0, 1])
    )
    if angle < 1.0e-8:
        return vector * 0.5
    return vector * (0.5 * angle / max(np.sin(angle), 1.0e-8))


@dataclass(frozen=True)
class LocalOdometryConfig:
    wheel_radius: float = 0.0825
    voxel_size: float = 0.15
    vertical_stride: int = 2
    horizontal_stride: int = 6
    icp_threads: int = 4
    icp_max_iterations: int = 80
    accel_noise: float = 0.8
    gyro_noise: float = 0.04
    accel_bias_walk: float = 0.02
    gyro_bias_walk: float = 0.002
    orientation_measurement_sigma_deg: float = 0.5
    orientation_yaw_sigma_deg: float = 0.5
    orientation_nis_threshold: float = 0.0
    wheel_velocity_sigma: float = 0.12
    lidar_position_sigma: float = 0.10
    lidar_orientation_sigma_deg: float = 2.0
    max_icp_translation_correction: float = 0.75
    lidar_nis_threshold: float = 0.0
    enable_point_coupling: bool = False
    point_coupling_sigma: float = 0.08
    point_coupling_max_points: int = 1200
    point_coupling_neighbors: int = 8
    point_coupling_max_correspondence: float = 0.60
    point_coupling_min_planarity: float = 0.08
    point_coupling_effective_points: float = 120.0
    adaptive_wheel_weighting: bool = True
    enable_motion_constraints: bool = False
    lateral_velocity_sigma: float = 0.10
    vertical_velocity_sigma: float = 0.20
    enable_zero_velocity_update: bool = False
    zero_velocity_sigma: float = 0.03
    wheel_signs: tuple[float, float, float, float] = (-1.0, -1.0, -1.0, -1.0)
    enable_deskew: bool = False


class ImuWheelEskf:
    """Small error-state filter in the route-start body frame."""

    STATE_DIM = 16

    def __init__(self, gravity_initial: np.ndarray, config: LocalOdometryConfig):
        self.config = config
        self.gravity = np.asarray(gravity_initial, dtype=np.float64)
        self.position = np.zeros(3, dtype=np.float64)
        self.velocity = np.zeros(3, dtype=np.float64)
        self.rotation = np.eye(3, dtype=np.float64)
        self.accel_bias = np.zeros(3, dtype=np.float64)
        self.gyro_bias = np.zeros(3, dtype=np.float64)
        self.wheel_scale = 1.0
        standard_deviation = np.concatenate((
            np.full(3, 0.02),
            np.full(3, 0.15),
            np.full(3, np.deg2rad(1.0)),
            np.full(3, 0.5),
            np.full(3, 0.05),
            np.asarray((0.08,)),
        ))
        self.covariance = np.diag(standard_deviation ** 2)

    def _inject(self, delta: np.ndarray) -> None:
        self.position += delta[0:3]
        self.velocity += delta[3:6]
        self.rotation = self.rotation @ _so3_exp(delta[6:9])
        self.accel_bias += delta[9:12]
        self.gyro_bias += delta[12:15]
        self.wheel_scale = float(np.clip(self.wheel_scale + delta[15], 0.5, 1.5))

    def _update(
        self,
        residual: np.ndarray,
        jacobian: np.ndarray,
        noise: np.ndarray,
        *,
        nis_threshold: float = 0.0,
    ) -> tuple[float, float]:
        innovation_covariance = jacobian @ self.covariance @ jacobian.T + noise
        nis = float(residual @ np.linalg.solve(innovation_covariance, residual))
        noise_inflation = 1.0
        if nis_threshold > 0.0 and nis > nis_threshold:
            # Retain geometrically plausible ICP updates while preventing one
            # statistically inconsistent frame from dominating the filter.
            noise_inflation = float(np.clip(nis / nis_threshold, 1.0, 100.0))
            noise = noise * noise_inflation
            innovation_covariance = jacobian @ self.covariance @ jacobian.T + noise
        gain = np.linalg.solve(
            innovation_covariance,
            jacobian @ self.covariance,
        ).T
        delta = gain @ residual
        self._inject(delta)
        identity = np.eye(self.STATE_DIM)
        correction = identity - gain @ jacobian
        self.covariance = (
            correction @ self.covariance @ correction.T + gain @ noise @ gain.T
        )
        reset_jacobian = np.eye(self.STATE_DIM)
        reset_jacobian[6:9, 6:9] -= 0.5 * _skew(delta[6:9])
        self.covariance = reset_jacobian @ self.covariance @ reset_jacobian.T
        self.covariance = 0.5 * (self.covariance + self.covariance.T)
        return nis, noise_inflation

    def propagate(self, time_s: np.ndarray, accel: np.ndarray, gyro: np.ndarray) -> None:
        if len(time_s) < 2:
            return
        for index in range(1, len(time_s)):
            dt = float(time_s[index] - time_s[index - 1])
            if not 0.0 < dt <= 0.05:
                continue
            specific_force = 0.5 * (accel[index - 1] + accel[index]) - self.accel_bias
            angular_velocity = 0.5 * (gyro[index - 1] + gyro[index]) - self.gyro_bias
            rotation_mid = self.rotation @ _so3_exp(angular_velocity * dt * 0.5)
            acceleration_initial = rotation_mid @ specific_force + self.gravity
            self.position += self.velocity * dt + 0.5 * acceleration_initial * dt * dt
            self.velocity += acceleration_initial * dt
            self.rotation = self.rotation @ _so3_exp(angular_velocity * dt)

            impact = max(0.0, abs(float(np.linalg.norm(specific_force)) - GRAVITY) - 1.0)
            accel_noise = self.config.accel_noise * (1.0 + min(impact / 6.0, 3.0))
            transition = np.zeros((self.STATE_DIM, self.STATE_DIM), dtype=np.float64)
            transition[0:3, 3:6] = np.eye(3)
            transition[3:6, 6:9] = -rotation_mid @ _skew(specific_force)
            transition[3:6, 9:12] = -rotation_mid
            transition[6:9, 6:9] = -_skew(angular_velocity)
            transition[6:9, 12:15] = -np.eye(3)
            phi = np.eye(self.STATE_DIM) + transition * dt
            process = np.zeros((self.STATE_DIM, self.STATE_DIM), dtype=np.float64)
            process[3:6, 3:6] = np.eye(3) * accel_noise ** 2
            process[6:9, 6:9] = np.eye(3) * self.config.gyro_noise ** 2
            process[9:12, 9:12] = np.eye(3) * self.config.accel_bias_walk ** 2
            process[12:15, 12:15] = np.eye(3) * self.config.gyro_bias_walk ** 2
            process[15, 15] = 1.0e-6
            self.covariance = phi @ self.covariance @ phi.T + process * dt

    def update_orientation(
        self,
        measured_rotation: np.ndarray,
        sigma_deg: float,
        nis_threshold: float = 0.0,
        yaw_sigma_deg: float | None = None,
    ) -> tuple[float, float]:
        residual = _so3_log(self.rotation.T @ np.asarray(measured_rotation, dtype=np.float64))
        jacobian = np.zeros((3, self.STATE_DIM), dtype=np.float64)
        jacobian[:, 6:9] = np.eye(3)
        sigma = np.deg2rad(float(sigma_deg))
        yaw_sigma = sigma if yaw_sigma_deg is None else np.deg2rad(float(yaw_sigma_deg))
        # Rotation errors are represented in the current body frame. Deweight
        # rotation about map vertical when the hardware RPY yaw is not a
        # globally referenced heading, while retaining roll/pitch correction.
        vertical_body = self.rotation.T @ np.asarray((0.0, 0.0, 1.0))
        vertical_body /= max(float(np.linalg.norm(vertical_body)), 1.0e-9)
        yaw_projection = np.outer(vertical_body, vertical_body)
        noise = (
            np.eye(3) * sigma ** 2
            + yaw_projection * (yaw_sigma ** 2 - sigma ** 2)
        )
        return self._update(
            residual,
            jacobian,
            noise,
            nis_threshold=nis_threshold,
        )

    def update_wheel_velocity(self, wheel_speed: float, sigma: float) -> float:
        velocity_body = self.rotation.T @ self.velocity
        predicted = float(velocity_body[0] / self.wheel_scale)
        residual = np.asarray((float(wheel_speed) - predicted,))
        jacobian = np.zeros((1, self.STATE_DIM), dtype=np.float64)
        jacobian[0, 3:6] = self.rotation[:, 0] / self.wheel_scale
        jacobian[0, 6:9] = (_skew(velocity_body)[0]) / self.wheel_scale
        jacobian[0, 15] = -velocity_body[0] / (self.wheel_scale ** 2)
        self._update(residual, jacobian, np.asarray(((sigma ** 2,),)))
        return float(residual[0])

    def update_body_velocity_constraints(
        self,
        lateral_sigma: float,
        vertical_sigma: float,
    ) -> None:
        """Apply terrain-conditioned nonholonomic velocity pseudo-measurements."""
        velocity_body = self.rotation.T @ self.velocity
        residual = -velocity_body[1:3]
        jacobian = np.zeros((2, self.STATE_DIM), dtype=np.float64)
        jacobian[:, 3:6] = self.rotation[:, 1:3].T
        jacobian[:, 6:9] = _skew(velocity_body)[1:3]
        noise = np.diag((float(lateral_sigma) ** 2, float(vertical_sigma) ** 2))
        self._update(residual, jacobian, noise)

    def update_zero_velocity(self, sigma: float) -> None:
        residual = -self.velocity
        jacobian = np.zeros((3, self.STATE_DIM), dtype=np.float64)
        jacobian[:, 3:6] = np.eye(3)
        self._update(residual, jacobian, np.eye(3) * float(sigma) ** 2)

    def update_lidar_pose(
        self,
        position: np.ndarray,
        rotation: np.ndarray,
        position_sigma: float,
        orientation_sigma_deg: float,
        nis_threshold: float = 0.0,
    ) -> tuple[float, float]:
        residual = np.concatenate((
            np.asarray(position, dtype=np.float64) - self.position,
            _so3_log(self.rotation.T @ np.asarray(rotation, dtype=np.float64)),
        ))
        jacobian = np.zeros((6, self.STATE_DIM), dtype=np.float64)
        jacobian[0:3, 0:3] = np.eye(3)
        jacobian[3:6, 6:9] = np.eye(3)
        orientation_sigma = np.deg2rad(float(orientation_sigma_deg))
        noise = np.diag(np.concatenate((
            np.full(3, float(position_sigma) ** 2),
            np.full(3, orientation_sigma ** 2),
        )))
        return self._update(
            residual,
            jacobian,
            noise,
            nis_threshold=nis_threshold,
        )

    def update_lidar_point_planes(
        self,
        source_body: np.ndarray,
        targets_initial: np.ndarray,
        normals_initial: np.ndarray,
        weights: np.ndarray,
        sigma: float,
    ) -> tuple[float, float]:
        """Fuse scan-to-map point-plane residuals in information form."""
        source = np.asarray(source_body, dtype=np.float64)
        targets = np.asarray(targets_initial, dtype=np.float64)
        normals = np.asarray(normals_initial, dtype=np.float64)
        weights = np.asarray(weights, dtype=np.float64).reshape(-1)
        predicted = self.position + source @ self.rotation.T
        residual = np.einsum("ni,ni->n", normals, targets - predicted)

        jacobian = np.zeros((len(source), self.STATE_DIM), dtype=np.float64)
        jacobian[:, 0:3] = normals
        rotated_skew = np.einsum(
            "ij,njk->nik",
            self.rotation,
            np.stack([_skew(point) for point in source]),
        )
        jacobian[:, 6:9] = -np.einsum("ni,nij->nj", normals, rotated_skew)

        precision = np.maximum(weights, 0.0) / max(float(sigma) ** 2, 1.0e-9)
        prior_information = np.linalg.inv(self.covariance)
        measurement_information = jacobian.T @ (precision[:, None] * jacobian)
        posterior_information = prior_information + measurement_information
        posterior_covariance = np.linalg.inv(posterior_information)
        delta = posterior_covariance @ (jacobian.T @ (precision * residual))
        self._inject(delta)
        self.covariance = posterior_covariance
        reset_jacobian = np.eye(self.STATE_DIM)
        reset_jacobian[6:9, 6:9] -= 0.5 * _skew(delta[6:9])
        self.covariance = reset_jacobian @ self.covariance @ reset_jacobian.T
        self.covariance = 0.5 * (self.covariance + self.covariance.T)
        weighted_rmse = float(np.sqrt(
            np.sum(weights * residual ** 2) / max(np.sum(weights), 1.0e-9)
        ))
        pose_indices = np.asarray((0, 1, 2, 6, 7, 8))
        pose_information = measurement_information[np.ix_(pose_indices, pose_indices)]
        return weighted_rmse, float(np.linalg.cond(pose_information + np.eye(6) * 1.0e-9))


class DualLidarImuWheelEskfOdometry:
    """KISS local map with a high-rate IMU/wheel error-state front end."""

    def __init__(
        self,
        initial_pose: np.ndarray,
        initial_imu_quat: np.ndarray,
        *,
        horizontal_samples: int = 900,
        config: LocalOdometryConfig | None = None,
        kiss_runtime: Path | None = None,
    ) -> None:
        self.config = config or LocalOdometryConfig()
        if kiss_runtime is not None:
            runtime = kiss_runtime.expanduser().resolve()
            if str(runtime) not in sys.path:
                sys.path.append(str(runtime))
        from kiss_icp.config import KISSConfig
        from kiss_icp.kiss_icp import KissICP

        kiss_config = KISSConfig()
        kiss_config.data.min_range = float(MIN_RANGE_M)
        kiss_config.data.max_range = float(INVALID_RANGE_THRESHOLD_M)
        kiss_config.data.deskew = self.config.enable_deskew
        kiss_config.mapping.voxel_size = self.config.voxel_size
        kiss_config.registration.max_num_iterations = int(
            self.config.icp_max_iterations
        )
        kiss_config.registration.convergence_criterion = 1.0e-4
        kiss_config.registration.max_num_threads = self.config.icp_threads
        kiss_config.adaptive_threshold.initial_threshold = 1.0
        kiss_config.adaptive_threshold.min_motion_th = 0.05
        self.pipeline = KissICP(kiss_config)

        self.initial_position_w = np.asarray(initial_pose[:3], dtype=np.float64).copy()
        self.initial_rotation_wb = quat_wxyz_to_rotmat(initial_pose[3:7])
        self.initial_rotation_imu = quat_wxyz_to_rotmat(initial_imu_quat)
        gravity_initial = self.initial_rotation_wb.T @ np.asarray((0.0, 0.0, -GRAVITY))
        self.filter = ImuWheelEskf(gravity_initial, self.config)
        self.position_w = self.initial_position_w.copy()
        self.rotation_wb = self.initial_rotation_wb.copy()

        directions = build_sensor_frame_directions(horizontal_samples)
        self.directions = directions[
            ::self.config.vertical_stride, ::self.config.horizontal_stride
        ]
        self.sensors = (
            (np.asarray(S10_FRONT_POS, dtype=np.float64), quat_wxyz_to_rotmat(S10_FRONT_ROT_WXYZ)),
            (np.asarray(S10_REAR_POS, dtype=np.float64), quat_wxyz_to_rotmat(S10_REAR_ROT_WXYZ)),
        )
        self.registration_seconds = 0.0
        self.point_count = 0
        self.wheel_prior_distance = 0.0
        self.wheel_speed = 0.0
        self.wheel_velocity_sigma = self.config.wheel_velocity_sigma
        self.wheel_innovation = 0.0
        self.slip_score = 0.0
        self.accel_impact = 0.0
        self.icp_translation_correction = np.zeros(3, dtype=np.float64)
        self.icp_residual_rotation_deg = 0.0
        self.icp_update_accepted = True
        self.lidar_innovation_nis = 0.0
        self.lidar_noise_inflation = 1.0
        self.nonholonomic_lateral_sigma = 0.0
        self.nonholonomic_vertical_sigma = 0.0
        self.zero_velocity_applied = False
        self.orientation_innovation_nis = 0.0
        self.orientation_noise_inflation = 1.0
        self.point_coupling_used = False
        self.point_coupling_count = 0
        self.point_coupling_rmse = 0.0
        self.point_coupling_condition = 0.0
        self.point_coupling_seconds = 0.0

    def _point_cloud(self, scans: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
        points = []
        for scan, (sensor_position, sensor_rotation) in zip(scans, self.sensors):
            sampled = np.asarray(scan, dtype=np.float64)[
                ::self.config.vertical_stride, ::self.config.horizontal_stride
            ]
            valid = (sampled > MIN_RANGE_M) & (sampled < INVALID_RANGE_THRESHOLD_M)
            rays_b = np.einsum("ij,hwj->hwi", sensor_rotation, self.directions)
            hits_b = sensor_position + sampled[..., None] * rays_b
            points.append(hits_b[valid])
        return np.ascontiguousarray(np.concatenate(points, axis=0), dtype=np.float64)

    def _pose_matrix(self) -> np.ndarray:
        pose = np.eye(4)
        pose[:3, :3] = self.filter.rotation
        pose[:3, 3] = self.filter.position
        return pose

    def _register_without_map_update(
        self,
        cloud: np.ndarray,
        point_timestamps: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Run KISS alignment while deferring the map write until after fusion."""
        deskewed = self.pipeline.preprocessor.preprocess(
            cloud,
            point_timestamps,
            self.pipeline.last_delta,
        )
        source, frame_downsample = self.pipeline.voxelize(deskewed)
        sigma = self.pipeline.adaptive_threshold.get_threshold()
        initial_guess = self.pipeline.last_pose @ self.pipeline.last_delta
        measured_pose = self.pipeline.registration.align_points_to_map(
            points=source,
            voxel_map=self.pipeline.local_map,
            initial_guess=initial_guess,
            max_correspondance_distance=3.0 * sigma,
            kernel=sigma,
        )
        model_deviation = np.linalg.inv(initial_guess) @ measured_pose
        self.pipeline.adaptive_threshold.update_model_deviation(model_deviation)
        return deskewed, source, frame_downsample, np.asarray(measured_pose)

    def _point_plane_correspondences(
        self,
        source_body: np.ndarray,
        measured_pose: np.ndarray,
        map_points: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        from scipy.spatial import cKDTree

        max_points = max(int(self.config.point_coupling_max_points), 100)
        stride = max(1, int(np.ceil(len(source_body) / max_points)))
        source = np.asarray(source_body[::stride], dtype=np.float64)
        aligned = measured_pose[:3, 3] + source @ measured_pose[:3, :3].T
        neighbors = max(int(self.config.point_coupling_neighbors), 4)
        distances, indices = cKDTree(map_points).query(
            aligned,
            k=neighbors,
            workers=self.config.icp_threads,
        )
        neighborhoods = map_points[indices]
        centroids = neighborhoods.mean(axis=1)
        centered = neighborhoods - centroids[:, None, :]
        covariance = np.einsum("nki,nkj->nij", centered, centered) / neighbors
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        normals = eigenvectors[:, :, 0]
        planarity = (eigenvalues[:, 1] - eigenvalues[:, 0]) / np.maximum(
            eigenvalues[:, 2], 1.0e-9
        )
        valid = (
            np.isfinite(eigenvalues).all(axis=1)
            & (distances[:, 0] <= self.config.point_coupling_max_correspondence)
            & (planarity >= self.config.point_coupling_min_planarity)
        )
        source = source[valid]
        centroids = centroids[valid]
        normals = normals[valid]
        planarity = np.clip(planarity[valid], 0.0, 1.0)
        aligned = aligned[valid]
        plane_residual = np.abs(np.einsum("ni,ni->n", normals, centroids - aligned))
        robust = 1.0 / (1.0 + (plane_residual / max(self.config.point_coupling_sigma, 1.0e-3)) ** 2)
        weights = planarity * robust
        weight_sum = float(weights.sum())
        if weight_sum > self.config.point_coupling_effective_points:
            weights *= self.config.point_coupling_effective_points / weight_sum
        return source, centroids, normals, weights

    def initialize_scan(self, scans: tuple[np.ndarray, np.ndarray]) -> None:
        self.initialize_points(self._point_cloud(scans))

    def initialize_points(
        self,
        points_body: np.ndarray,
        timestamps: np.ndarray | None = None,
    ) -> None:
        cloud, stamps = self._validate_cloud(points_body, timestamps)
        self.pipeline.register_frame(cloud, stamps)
        self.point_count = len(cloud)

    @staticmethod
    def _validate_cloud(
        points_body: np.ndarray,
        timestamps: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        cloud = np.asarray(points_body, dtype=np.float64)
        if cloud.ndim != 2 or cloud.shape[1] != 3:
            raise ValueError(f"points_body must have shape [N,3], got {cloud.shape}")
        finite = np.isfinite(cloud).all(axis=1)
        if timestamps is None:
            stamps = np.empty(0, dtype=np.float64)
        else:
            stamps = np.asarray(timestamps, dtype=np.float64).reshape(-1)
            if len(stamps) == 0:
                stamps = np.empty(0, dtype=np.float64)
            elif len(stamps) != len(cloud):
                raise ValueError("point timestamps must match points_body")
            else:
                finite &= np.isfinite(stamps)
                stamps = stamps[finite]
        if bool(np.all(finite)):
            cloud = np.ascontiguousarray(cloud, dtype=np.float64)
        else:
            cloud = np.ascontiguousarray(cloud[finite], dtype=np.float64)
        if len(cloud) < 100:
            raise ValueError(f"point cloud has too few finite points: {len(cloud)}")
        if len(stamps) and float(np.ptp(stamps)) <= 1.0e-6:
            stamps = np.empty(0, dtype=np.float64)
        return cloud, stamps

    def _slip_metrics(self, history: dict[str, np.ndarray]) -> tuple[float, float, float]:
        wheel_signs = np.asarray(self.config.wheel_signs, dtype=np.float64)
        if wheel_signs.shape != (4,) or not np.isin(wheel_signs, (-1.0, 1.0)).all():
            raise ValueError("wheel_signs must contain four values in {-1,+1}")
        wheel_linear = np.asarray(history["wheel_qvel"]) * wheel_signs * self.config.wheel_radius
        wheel_speed_samples = np.mean(wheel_linear, axis=1)
        wheel_speed = float(np.median(wheel_speed_samples))
        wheel_spread = float(np.percentile(np.std(wheel_linear, axis=1), 90))
        accel_norm = np.linalg.norm(np.asarray(history["accelerometer"]), axis=1)
        self.accel_impact = float(np.percentile(np.abs(accel_norm - GRAVITY), 95))
        gyro_tilt = float(np.percentile(np.linalg.norm(np.asarray(history["gyro"])[:, :2], axis=1), 95))
        predicted_speed = float((self.filter.rotation.T @ self.filter.velocity)[0])
        speed_innovation = abs(wheel_speed * self.filter.wheel_scale - predicted_speed)
        components = np.asarray((
            wheel_spread / 0.5,
            max(0.0, self.accel_impact - 1.5) / 7.0,
            max(0.0, gyro_tilt - 0.5) / 3.0,
            speed_innovation / 1.0,
        ))
        self.slip_score = float(np.clip(np.max(components), 0.0, 1.0))
        sigma = self.config.wheel_velocity_sigma
        if self.config.adaptive_wheel_weighting:
            sigma *= 1.0 + 15.0 * self.slip_score ** 2
        return wheel_speed, sigma, float(history["time"][-1] - history["time"][0])

    def update(
        self,
        scans: tuple[np.ndarray, np.ndarray],
        history: dict[str, np.ndarray],
    ) -> None:
        cloud = self._point_cloud(scans)
        self.update_points(cloud, history)

    def update_points(
        self,
        points_body: np.ndarray,
        history: dict[str, np.ndarray],
        timestamps: np.ndarray | None = None,
    ) -> None:
        time_s = np.asarray(history["time"], dtype=np.float64)
        if len(time_s) < 2:
            raise RuntimeError("high-rate odometry requires at least two IMU samples")
        self.filter.propagate(
            time_s,
            np.asarray(history["accelerometer"], dtype=np.float64),
            np.asarray(history["gyro"], dtype=np.float64),
        )
        measured_rotation = (
            self.initial_rotation_imu.T
            @ quat_wxyz_to_rotmat(np.asarray(history["orientation_wxyz"][-1]))
        )
        wheel_speed, self.wheel_velocity_sigma, interval = self._slip_metrics(history)
        self.wheel_speed = wheel_speed
        orientation_sigma = self.config.orientation_measurement_sigma_deg * (1.0 + 3.0 * self.slip_score)
        (
            self.orientation_innovation_nis,
            self.orientation_noise_inflation,
        ) = self.filter.update_orientation(
            measured_rotation,
            orientation_sigma,
            self.config.orientation_nis_threshold,
            self.config.orientation_yaw_sigma_deg,
        )
        self.wheel_innovation = self.filter.update_wheel_velocity(
            wheel_speed, self.wheel_velocity_sigma
        )
        gyro = np.asarray(history["gyro"], dtype=np.float64)
        wheel_linear = (
            np.asarray(history["wheel_qvel"], dtype=np.float64)
            * np.asarray(self.config.wheel_signs, dtype=np.float64)
            * self.config.wheel_radius
        )
        wheel_spread = float(np.percentile(np.std(wheel_linear, axis=1), 90))
        gyro_norm = float(np.percentile(np.linalg.norm(gyro, axis=1), 90))
        if self.config.enable_motion_constraints:
            lateral_relaxation = 1.0 + 8.0 * min(wheel_spread / 0.5, 1.0) ** 2
            vertical_relaxation = 1.0 + 12.0 * min(self.accel_impact / 8.0, 1.0) ** 2
            self.nonholonomic_lateral_sigma = (
                self.config.lateral_velocity_sigma * lateral_relaxation
            )
            self.nonholonomic_vertical_sigma = (
                self.config.vertical_velocity_sigma * vertical_relaxation
            )
            self.filter.update_body_velocity_constraints(
                self.nonholonomic_lateral_sigma,
                self.nonholonomic_vertical_sigma,
            )
        else:
            self.nonholonomic_lateral_sigma = 0.0
            self.nonholonomic_vertical_sigma = 0.0

        accel_norm = np.linalg.norm(np.asarray(history["accelerometer"]), axis=1)
        stationary = bool(
            abs(wheel_speed) < 0.04
            and wheel_spread < 0.08
            and gyro_norm < 0.12
            and float(np.percentile(np.abs(accel_norm - GRAVITY), 90)) < 0.35
        )
        self.zero_velocity_applied = bool(
            self.config.enable_zero_velocity_update and stationary
        )
        if self.zero_velocity_applied:
            self.filter.update_zero_velocity(self.config.zero_velocity_sigma)
        self.wheel_prior_distance = wheel_speed * self.filter.wheel_scale * interval

        cloud, point_timestamps = self._validate_cloud(points_body, timestamps)
        predicted_pose = self._pose_matrix()
        self.pipeline.last_delta = np.linalg.inv(self.pipeline.last_pose) @ predicted_pose
        map_points = (
            np.asarray(self.pipeline.local_map.point_cloud(), dtype=np.float64).copy()
            if self.config.enable_point_coupling else np.empty((0, 3))
        )
        started = time.monotonic()
        if self.config.enable_point_coupling:
            deskewed, source, frame_downsample, measured_pose = self._register_without_map_update(
                cloud, point_timestamps
            )
        else:
            deskewed, source = self.pipeline.register_frame(cloud, point_timestamps)
            frame_downsample = np.empty((0, 3))
            measured_pose = np.asarray(self.pipeline.last_pose, dtype=np.float64).copy()
        self.registration_seconds = time.monotonic() - started
        self.icp_translation_correction = measured_pose[:3, 3] - predicted_pose[:3, 3]
        correction_norm = float(np.linalg.norm(self.icp_translation_correction))
        rotation_correction = _so3_log(predicted_pose[:3, :3].T @ measured_pose[:3, :3])
        self.icp_residual_rotation_deg = float(np.degrees(np.linalg.norm(rotation_correction)))
        self.icp_update_accepted = bool(
            correction_norm <= self.config.max_icp_translation_correction
            and self.icp_residual_rotation_deg <= 15.0
        )
        self.point_coupling_used = False
        self.point_coupling_count = 0
        self.point_coupling_rmse = 0.0
        self.point_coupling_condition = 0.0
        self.point_coupling_seconds = 0.0
        if self.icp_update_accepted:
            # Wheel slip is not evidence that an instantaneous simulated
            # LiDAR frame is bad. Inflate only from ICP's own departure from
            # the propagated pose; real deskew quality will become a separate
            # measurement-quality term in the hardware adapter.
            inflation = 1.0 + max(0.0, correction_norm - 0.20) * 2.0
            coupled = False
            if self.config.enable_point_coupling and len(map_points) >= 100:
                coupling_started = time.monotonic()
                source_c, targets_c, normals_c, weights_c = self._point_plane_correspondences(
                    source, measured_pose, map_points
                )
                self.point_coupling_seconds = time.monotonic() - coupling_started
                self.point_coupling_count = len(source_c)
                if len(source_c) >= 100 and float(weights_c.sum()) >= 10.0:
                    (
                        self.point_coupling_rmse,
                        self.point_coupling_condition,
                    ) = self.filter.update_lidar_point_planes(
                        source_c,
                        targets_c,
                        normals_c,
                        weights_c,
                        self.config.point_coupling_sigma * inflation,
                    )
                    self.point_coupling_used = True
                    coupled = True
                    self.lidar_innovation_nis = 0.0
                    self.lidar_noise_inflation = inflation
            if not coupled:
                self.lidar_innovation_nis, self.lidar_noise_inflation = self.filter.update_lidar_pose(
                    measured_pose[:3, 3],
                    measured_pose[:3, :3],
                    self.config.lidar_position_sigma * inflation,
                    self.config.lidar_orientation_sigma_deg * inflation,
                    self.config.lidar_nis_threshold,
                )
        else:
            self.lidar_innovation_nis = float("inf")
            self.lidar_noise_inflation = 1.0
            if not self.config.enable_point_coupling:
                # The default KISS path has already inserted this rejected frame.
                self.pipeline.local_map.clear()
                _, frame_downsample = self.pipeline.voxelize(deskewed)
                self.pipeline.local_map.update(frame_downsample, predicted_pose)

        posterior_pose = self._pose_matrix()
        if self.config.enable_point_coupling and self.icp_update_accepted:
            self.pipeline.local_map.update(frame_downsample, posterior_pose)
        self.pipeline.last_pose = posterior_pose
        self.pipeline.last_delta = np.eye(4)
        self.position_w = self.initial_position_w + self.initial_rotation_wb @ self.filter.position
        self.rotation_wb = self.initial_rotation_wb @ self.filter.rotation
        self.point_count = len(cloud)

    def goal_body(self, target_w: np.ndarray) -> np.ndarray:
        delta = self.rotation_wb.T @ (np.asarray(target_w) - self.position_w)
        distance = max(float(np.linalg.norm(delta)), 1.0e-6)
        return np.concatenate((delta / distance, np.asarray((np.log1p(distance),))))

    @property
    def covariance_trace(self) -> float:
        return float(np.trace(self.filter.covariance))
