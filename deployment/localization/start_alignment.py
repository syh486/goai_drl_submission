"""Automatic dual-LiDAR start-frame anchoring and relocalization."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
import tempfile
from typing import Any

import numpy as np
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class StartAlignmentConfig:
    capture_frames: int = 12
    voxel_size_m: float = 0.12
    min_range_m: float = 0.35
    max_range_m: float = 30.0
    structural_min_z_m: float = -0.20
    max_registration_points: int = 3500
    yaw_search_range_deg: float = 45.0
    yaw_search_step_deg: float = 2.0
    coarse_candidates: int = 4
    max_initial_translation_m: float = 1.5
    max_correspondence_m: float = 0.80
    trim_fraction: float = 0.70
    icp_iterations: int = 25
    stationary_wheel_speed_mps: float = 0.05
    stationary_gyro_rps: float = 0.12

    @classmethod
    def from_config(cls, config: dict[str, Any] | None) -> "StartAlignmentConfig":
        values = config or {}
        fields = cls.__dataclass_fields__
        result = cls(**{key: values[key] for key in fields if key in values})
        if result.capture_frames < 3:
            raise ValueError("start alignment needs at least three LiDAR frames")
        if not 0.2 <= result.trim_fraction <= 1.0:
            raise ValueError("start alignment trim_fraction must be in [0.2, 1.0]")
        if result.yaw_search_step_deg <= 0.0 or result.yaw_search_range_deg < 0.0:
            raise ValueError("start alignment yaw search values are invalid")
        return result


@dataclass(frozen=True)
class StartAnchor:
    points_reference_body_m: np.ndarray
    initial_imu_quaternion_wxyz: np.ndarray
    frame_count: int
    voxel_size_m: float


@dataclass(frozen=True)
class StartAlignmentResult:
    rotation_reference_live: np.ndarray
    translation_reference_live_m: np.ndarray
    yaw_deg: float
    rmse_m: float
    overlap_fraction: float
    correspondence_count: int
    candidate_margin_m: float

    @property
    def transform_reference_live(self) -> np.ndarray:
        transform = np.eye(4, dtype=np.float64)
        transform[:3, :3] = self.rotation_reference_live
        transform[:3, 3] = self.translation_reference_live_m
        return transform


def anchor_path_for_route(route_path: Path) -> Path:
    return route_path.with_suffix(".anchor.npz")


def _voxel_downsample(points: np.ndarray, voxel_size_m: float) -> np.ndarray:
    values = np.asarray(points, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError(f"anchor points must have shape [N,3], got {values.shape}")
    values = values[np.isfinite(values).all(axis=1)]
    if not len(values):
        return values
    keys = np.floor(values / float(voxel_size_m)).astype(np.int64)
    _, first = np.unique(keys, axis=0, return_index=True)
    return np.ascontiguousarray(values[np.sort(first)])


def _voxel_downsample_xy(points_xy: np.ndarray, voxel_size_m: float) -> np.ndarray:
    values = np.asarray(points_xy, dtype=np.float64)
    keys = np.floor(values / float(voxel_size_m)).astype(np.int64)
    _, first = np.unique(keys, axis=0, return_index=True)
    return np.ascontiguousarray(values[np.sort(first)])


def _filter_points(points: np.ndarray, config: StartAlignmentConfig) -> np.ndarray:
    values = np.asarray(points, dtype=np.float64)
    finite = np.isfinite(values).all(axis=1)
    distance = np.linalg.norm(values, axis=1)
    return values[
        finite
        & (distance >= config.min_range_m)
        & (distance <= config.max_range_m)
    ]


class StartAnchorAccumulator:
    """Accumulate a stationary multi-frame cloud in the current body frame."""

    def __init__(self, config: StartAlignmentConfig):
        self.config = config
        self.frames: list[np.ndarray] = []
        self.reset_count = 0

    @property
    def complete(self) -> bool:
        return len(self.frames) >= self.config.capture_frames

    def add(self, points_body: np.ndarray, *, stationary: bool) -> bool:
        if not stationary:
            if self.frames:
                self.frames.clear()
                self.reset_count += 1
            return False
        filtered = _filter_points(points_body, self.config)
        if len(filtered) < 100:
            return False
        self.frames.append(filtered)
        return self.complete

    def merged(self) -> np.ndarray:
        if not self.complete:
            raise RuntimeError(
                f"start anchor has {len(self.frames)}/{self.config.capture_frames} frames"
            )
        return _voxel_downsample(
            np.concatenate(self.frames, axis=0), self.config.voxel_size_m
        )


def save_start_anchor(
    path: Path,
    points_reference_body_m: np.ndarray,
    initial_imu_quaternion_wxyz: np.ndarray,
    *,
    frame_count: int,
    voxel_size_m: float,
) -> None:
    output = path.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    points = np.asarray(points_reference_body_m, dtype=np.float32)
    quaternion = np.asarray(initial_imu_quaternion_wxyz, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 100:
        raise ValueError("start anchor contains too few valid 3D points")
    if quaternion.shape != (4,) or not np.isfinite(quaternion).all():
        raise ValueError("start anchor IMU quaternion is invalid")
    with tempfile.NamedTemporaryFile(
        mode="wb", dir=output.parent, prefix=f".{output.name}.", delete=False
    ) as stream:
        temporary = Path(stream.name)
        np.savez_compressed(
            stream,
            schema_version=np.asarray(1, dtype=np.int64),
            points_reference_body_m=points,
            initial_imu_quaternion_wxyz=quaternion,
            frame_count=np.asarray(frame_count, dtype=np.int64),
            voxel_size_m=np.asarray(voxel_size_m, dtype=np.float64),
        )
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, output)


def load_start_anchor(path: Path) -> StartAnchor:
    source = path.expanduser().resolve()
    with np.load(source, allow_pickle=False) as payload:
        version = int(payload["schema_version"])
        if version != 1:
            raise ValueError(f"unsupported start anchor schema {version}: {source}")
        points = np.asarray(payload["points_reference_body_m"], dtype=np.float64)
        quaternion = np.asarray(payload["initial_imu_quaternion_wxyz"], dtype=np.float64)
        frame_count = int(payload["frame_count"])
        voxel_size = float(payload["voxel_size_m"])
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 100:
        raise ValueError(f"start anchor has invalid points: {source}")
    if quaternion.shape != (4,) or not np.isfinite(quaternion).all():
        raise ValueError(f"start anchor has invalid IMU orientation: {source}")
    return StartAnchor(points, quaternion, frame_count, voxel_size)


def _sample_points(points: np.ndarray, maximum: int) -> np.ndarray:
    if len(points) <= maximum:
        return points
    indices = np.linspace(0, len(points) - 1, maximum, dtype=np.int64)
    return points[indices]


def _yaw_rotation(yaw_rad: float) -> np.ndarray:
    cosine, sine = np.cos(yaw_rad), np.sin(yaw_rad)
    return np.asarray(((cosine, -sine), (sine, cosine)), dtype=np.float64)


def _fit_rigid(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    source_center = source.mean(axis=0)
    target_center = target.mean(axis=0)
    covariance = (source - source_center).T @ (target - target_center)
    left, _, right_t = np.linalg.svd(covariance)
    rotation = right_t.T @ left.T
    if np.linalg.det(rotation) < 0.0:
        right_t[-1] *= -1.0
        rotation = right_t.T @ left.T
    translation = target_center - rotation @ source_center
    return rotation, translation


def _correspondences(
    source: np.ndarray,
    target: np.ndarray,
    tree: cKDTree,
    rotation: np.ndarray,
    translation: np.ndarray,
    *,
    max_distance: float,
    trim_fraction: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    transformed = source @ rotation.T + translation
    distances, indices = tree.query(transformed, k=1, workers=-1)
    valid = np.isfinite(distances) & (distances <= max_distance)
    selected = np.flatnonzero(valid)
    if len(selected) < 20:
        return transformed[:0], target[:0], distances[:0]
    keep = max(20, int(np.ceil(len(selected) * trim_fraction)))
    selected = selected[np.argsort(distances[selected])[:keep]]
    return transformed[selected], target[indices[selected]], distances[selected]


def _refine_se2(
    source_xy: np.ndarray,
    target_xy: np.ndarray,
    initial_yaw: float,
    initial_translation: np.ndarray,
    config: StartAlignmentConfig,
) -> tuple[np.ndarray, np.ndarray, float, int]:
    rotation = _yaw_rotation(initial_yaw)
    translation = np.asarray(initial_translation, dtype=np.float64).copy()
    tree = cKDTree(target_xy)
    distances = np.empty(0)
    for _ in range(config.icp_iterations):
        matched_source, matched_target, distances = _correspondences(
            source_xy,
            target_xy,
            tree,
            rotation,
            translation,
            max_distance=config.max_correspondence_m,
            trim_fraction=config.trim_fraction,
        )
        if len(matched_source) < 20:
            break
        correction_rotation, correction_translation = _fit_rigid(
            matched_source, matched_target
        )
        rotation = correction_rotation @ rotation
        translation = correction_rotation @ translation + correction_translation
        if (
            np.linalg.norm(correction_translation) < 1.0e-4
            and abs(np.arctan2(correction_rotation[1, 0], correction_rotation[0, 0]))
            < 1.0e-5
        ):
            break
    rmse = float(np.sqrt(np.mean(np.square(distances)))) if len(distances) else float("inf")
    return rotation, translation, rmse, len(distances)


def _refine_point_to_line_se2(
    source_xy: np.ndarray,
    target_xy: np.ndarray,
    initial_rotation: np.ndarray,
    initial_translation: np.ndarray,
    config: StartAlignmentConfig,
) -> tuple[np.ndarray, np.ndarray, float, int]:
    """Refine yaw against structural lines without along-wall point bias."""

    rotation = np.asarray(initial_rotation, dtype=np.float64).copy()
    translation = np.asarray(initial_translation, dtype=np.float64).copy()
    tree = cKDTree(target_xy)
    neighbors = min(12, len(target_xy))
    _, normal_indices = tree.query(target_xy, k=neighbors, workers=-1)
    neighborhoods = target_xy[normal_indices]
    centered = neighborhoods - neighborhoods.mean(axis=1, keepdims=True)
    covariance = np.einsum("nki,nkj->nij", centered, centered) / neighbors
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    normals = eigenvectors[:, :, 0]
    linearity = (eigenvalues[:, 1] - eigenvalues[:, 0]) / np.maximum(
        eigenvalues[:, 1], 1.0e-9
    )
    count = 0
    residual = np.empty(0)
    for _ in range(config.icp_iterations):
        transformed = source_xy @ rotation.T + translation
        distances, indices = tree.query(transformed, k=1, workers=-1)
        valid = (
            np.isfinite(distances)
            & (distances <= config.max_correspondence_m)
            & (linearity[indices] >= 0.35)
        )
        selected = np.flatnonzero(valid)
        if len(selected) < 30:
            break
        keep = max(30, int(np.ceil(len(selected) * config.trim_fraction)))
        selected = selected[np.argsort(distances[selected])[:keep]]
        source = transformed[selected]
        target = target_xy[indices[selected]]
        normal = normals[indices[selected]]
        residual = np.einsum("ni,ni->n", normal, target - source)
        jacobian = np.column_stack((
            normal[:, 0],
            normal[:, 1],
            -normal[:, 0] * source[:, 1] + normal[:, 1] * source[:, 0],
        ))
        scale = max(float(np.median(np.abs(residual))) * 2.5, 0.02)
        weights = 1.0 / (1.0 + np.square(residual / scale))
        weighted = np.sqrt(weights)[:, None]
        delta, *_ = np.linalg.lstsq(weighted * jacobian, weighted[:, 0] * residual, rcond=None)
        delta[:2] = np.clip(delta[:2], -0.20, 0.20)
        delta[2] = float(np.clip(delta[2], -np.deg2rad(2.0), np.deg2rad(2.0)))
        correction = _yaw_rotation(float(delta[2]))
        rotation = correction @ rotation
        translation = correction @ translation + delta[:2]
        count = len(selected)
        if np.linalg.norm(delta[:2]) < 1.0e-4 and abs(delta[2]) < 1.0e-5:
            break
    rmse = float(np.sqrt(np.mean(np.square(residual)))) if len(residual) else float("inf")
    return rotation, translation, rmse, count


def _refine_se3(
    source: np.ndarray,
    target: np.ndarray,
    initial_rotation: np.ndarray,
    initial_translation: np.ndarray,
    config: StartAlignmentConfig,
) -> tuple[np.ndarray, np.ndarray, float, int]:
    rotation = np.asarray(initial_rotation, dtype=np.float64).copy()
    translation = np.asarray(initial_translation, dtype=np.float64).copy()
    tree = cKDTree(target)
    distances = np.empty(0)
    for _ in range(config.icp_iterations):
        matched_source, matched_target, distances = _correspondences(
            source,
            target,
            tree,
            rotation,
            translation,
            max_distance=config.max_correspondence_m,
            trim_fraction=config.trim_fraction,
        )
        if len(matched_source) < 30:
            break
        correction_rotation, correction_translation = _fit_rigid(
            matched_source, matched_target
        )
        rotation = correction_rotation @ rotation
        translation = correction_rotation @ translation + correction_translation
        angle = np.arccos(np.clip((np.trace(correction_rotation) - 1.0) * 0.5, -1.0, 1.0))
        if np.linalg.norm(correction_translation) < 1.0e-4 and angle < 1.0e-5:
            break
    rmse = float(np.sqrt(np.mean(np.square(distances)))) if len(distances) else float("inf")
    return rotation, translation, rmse, len(distances)


def align_start_anchor(
    anchor: StartAnchor,
    live_points_body_m: np.ndarray,
    config: StartAlignmentConfig,
) -> StartAlignmentResult:
    """Estimate the live body pose in the route's recorded start-body frame."""

    anchor_points = _voxel_downsample(
        _filter_points(anchor.points_reference_body_m, config), config.voxel_size_m
    )
    live_points = _voxel_downsample(
        _filter_points(live_points_body_m, config), config.voxel_size_m
    )
    anchor_structural = anchor_points[anchor_points[:, 2] >= config.structural_min_z_m]
    live_structural = live_points[live_points[:, 2] >= config.structural_min_z_m]
    anchor_structural = _sample_points(anchor_structural, config.max_registration_points)
    live_structural = _sample_points(live_structural, config.max_registration_points)
    if len(anchor_structural) < 100 or len(live_structural) < 100:
        raise RuntimeError("start alignment has too few non-ground structure points")

    target_xy = _voxel_downsample_xy(anchor_structural[:, :2], config.voxel_size_m)
    source_xy = _voxel_downsample_xy(live_structural[:, :2], config.voxel_size_m)
    tree = cKDTree(target_xy)
    yaw_values = np.deg2rad(np.arange(
        -config.yaw_search_range_deg,
        config.yaw_search_range_deg + 0.5 * config.yaw_search_step_deg,
        config.yaw_search_step_deg,
    ))
    coarse: list[tuple[float, float, np.ndarray]] = []
    for yaw in yaw_values:
        rotation = _yaw_rotation(float(yaw))
        transformed = source_xy @ rotation.T
        translation = np.zeros(2, dtype=np.float64)
        for _ in range(2):
            distances, indices = tree.query(transformed + translation, k=1, workers=-1)
            selected = np.argsort(distances)[:max(100, int(len(distances) * 0.5))]
            update = np.median(
                target_xy[indices[selected]] - (transformed[selected] + translation), axis=0
            )
            candidate = translation + update
            norm = float(np.linalg.norm(candidate))
            translation = (
                candidate
                if norm <= config.max_initial_translation_m
                else candidate * (config.max_initial_translation_m / max(norm, 1.0e-9))
            )
        distances, _ = tree.query(transformed + translation, k=1, workers=-1)
        keep = max(100, int(len(distances) * config.trim_fraction))
        score = float(np.sqrt(np.mean(np.square(np.partition(distances, keep - 1)[:keep]))))
        coarse.append((score, float(yaw), translation))

    refined: list[tuple[float, np.ndarray, np.ndarray, int]] = []
    for _, yaw, translation in sorted(coarse, key=lambda item: item[0])[:config.coarse_candidates]:
        rotation, translated, rmse, count = _refine_se2(
            source_xy, target_xy, yaw, translation, config
        )
        rotation, translated, line_rmse, line_count = _refine_point_to_line_se2(
            source_xy, target_xy, rotation, translated, config
        )
        refined.append((line_rmse, rotation, translated, line_count or count))
    refined.sort(key=lambda item: item[0])
    best_rmse_2d, rotation_2d, translation_2d, _ = refined[0]
    margin = (
        float(refined[1][0] - best_rmse_2d) if len(refined) > 1 else float("inf")
    )

    initial_rotation = np.eye(3, dtype=np.float64)
    initial_rotation[:2, :2] = rotation_2d
    initial_translation = np.asarray((translation_2d[0], translation_2d[1], 0.0))
    source_3d = _sample_points(live_points, config.max_registration_points)
    target_3d = _sample_points(anchor_points, config.max_registration_points * 2)
    rotation, translation, rmse, count = _refine_se3(
        source_3d, target_3d, initial_rotation, initial_translation, config
    )
    # The structural XY point-to-line solution is the better constrained yaw
    # estimate. Preserve it while retaining the 3D ICP roll/pitch correction.
    yaw_2d = float(np.arctan2(rotation_2d[1, 0], rotation_2d[0, 0]))
    yaw_3d = float(np.arctan2(rotation[1, 0], rotation[0, 0]))
    correction_yaw = np.eye(3)
    correction_yaw[:2, :2] = _yaw_rotation(yaw_2d - yaw_3d)
    rotation = correction_yaw @ rotation
    target_tree_3d = cKDTree(target_3d)
    for _ in range(3):
        matched_source, matched_target, _ = _correspondences(
            source_3d,
            target_3d,
            target_tree_3d,
            rotation,
            translation,
            max_distance=config.max_correspondence_m,
            trim_fraction=config.trim_fraction,
        )
        if len(matched_source) < 30:
            break
        translation += np.median(matched_target - matched_source, axis=0)
    _, _, final_distances = _correspondences(
        source_3d,
        target_3d,
        target_tree_3d,
        rotation,
        translation,
        max_distance=config.max_correspondence_m,
        trim_fraction=config.trim_fraction,
    )
    count = len(final_distances)
    rmse = (
        float(np.sqrt(np.mean(np.square(final_distances))))
        if count else float("inf")
    )
    yaw = float(np.degrees(np.arctan2(rotation[1, 0], rotation[0, 0])))
    overlap = float(count / max(len(source_3d), 1))
    return StartAlignmentResult(
        rotation_reference_live=rotation,
        translation_reference_live_m=translation,
        yaw_deg=yaw,
        rmse_m=rmse,
        overlap_fraction=overlap,
        correspondence_count=count,
        candidate_margin_m=margin,
    )


def compose_aligned_initial_pose(
    reference_pose_wxyz: np.ndarray,
    alignment: StartAlignmentResult,
) -> tuple[np.ndarray, np.ndarray]:
    """Return route-frame position and rotation for the live start body."""

    from deployment.common.math_utils import quat_wxyz_to_rotmat

    reference = np.asarray(reference_pose_wxyz, dtype=np.float64)
    rotation_route_reference = quat_wxyz_to_rotmat(reference[3:7])
    position = (
        reference[:3]
        + rotation_route_reference @ alignment.translation_reference_live_m
    )
    rotation = rotation_route_reference @ alignment.rotation_reference_live
    return position, rotation
