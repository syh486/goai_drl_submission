"""Prior-guided continuous localization against a frozen route map.

The local odometry pose remains untouched.  Accepted map observations update
``T_map_odom`` so control can always use ``T_map_base = T_map_odom @ T_odom_base``.
Multiple hypotheses are retained across ambiguous walls and route crossings.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import importlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from deployment.mapping.mapping_geometry import (
    rotation_distance_deg,
    scan_context_descriptor,
    voxel_downsample,
    yaw_error_deg,
)


@dataclass(frozen=True)
class ContinuousLocalizationConfig:
    canonical_reference_session: int = 0
    registration_target: str = "route_submap"
    registration_backend: str = "kiss_icp"
    beam_width: int = 2
    candidate_count: int = 4
    odometry_tracking_candidate_count: int = 1
    odometry_recovery_candidate_count: int = 2
    route_backtrack_submaps: int = 2
    route_lookahead_submaps: int = 6
    recovery_lookahead_submaps: int = 60
    recovery_growth_per_coast: int = 4
    recovery_odometry_radius_submaps: int = 12
    max_tracking_route_step: int = 2
    query_voxel_m: float = 0.22
    fine_query_voxel_m: float = 0.08
    fine_max_query_points: int = 16000
    query_submap_updates: int = 5
    cloud_cache_size: int = 24
    max_sensor_range_m: float = 22.0
    max_correspondence_m: float = 0.90
    max_icp_iterations: int = 25
    icp_threads: int = 2
    fine_max_icp_iterations: int = 18
    fine_max_correspondence_m: float = 0.40
    fine_min_fitness: float = 0.55
    fine_max_rmse_m: float = 0.18
    vgicp_target_voxel_m: float = 0.10
    vgicp_source_voxel_m: float = 0.10
    vgicp_voxel_resolution_m: float = 0.30
    vgicp_covariance_neighbors: int = 15
    vgicp_covariance_scale: float = 1.0
    vgicp_max_target_points: int = 24000
    vgicp_max_iterations: int = 15
    vgicp_num_threads: int = 1
    hybrid_max_translation_disagreement_m: float = 0.25
    hybrid_max_rotation_disagreement_deg: float = 2.0
    # KISS's voxel-map correspondences score cross-session route scans lower
    # than Open3D GICP. These gates are calibrated on independent S10 laps and
    # remain combined with descriptor, innovation and route-index gates.
    min_fitness: float = 0.50
    max_rmse_m: float = 0.48
    continuity_min_fitness: float = 0.40
    continuity_max_rmse_m: float = 0.50
    continuity_max_translation_m: float = 0.75
    continuity_max_yaw_deg: float = 5.0
    max_translation_innovation_m: float = 2.8
    max_yaw_innovation_deg: float = 28.0
    max_tracking_translation_correction_m: float = 1.8
    max_tracking_yaw_correction_deg: float = 12.0
    temporal_fusion_enabled: bool = False
    tracking_translation_gain: float = 0.20
    tracking_rotation_gain: float = 0.20
    tracking_max_translation_step_m: float = 0.25
    tracking_max_rotation_step_deg: float = 2.0
    relocalization_consistency_translation_m: float = 0.75
    relocalization_consistency_yaw_deg: float = 6.0
    relocalization_required_updates: int = 3
    max_automatic_relocalization_translation_m: float = 3.0
    max_automatic_relocalization_yaw_deg: float = 30.0
    max_route_consistent_relocalization_translation_m: float = 25.0
    max_route_consistent_relocalization_yaw_deg: float = 45.0
    max_stationary_relocalization_translation_m: float = 80.0
    max_stationary_relocalization_yaw_deg: float = 120.0
    max_descriptor_distance: float = 0.95
    route_step_cost: float = 0.015
    route_lag_cost: float = 0.05
    multisession_consensus_enabled: bool = False
    multisession_fitness_weight: float = 0.04
    multisession_descriptor_weight: float = 0.70
    multisession_translation_innovation_weight: float = 0.02
    multisession_yaw_innovation_weight: float = 0.002
    multisession_route_step_weight: float = 0.001
    multisession_odometry_route_weight: float = 0.05
    multisession_repeatability_bonus: float = 0.12
    multisession_max_translation_disagreement_m: float = 0.15
    multisession_max_yaw_disagreement_deg: float = 2.0
    multisession_history_decay: float = 0.0
    coast_base_cost: float = 0.09
    max_coast_updates: int = 3
    minimum_motion_before_matching_m: float = 1.0


@dataclass(frozen=True)
class LocalizationHypothesis:
    cost: float
    map_from_odom: np.ndarray
    coast_updates: int
    source: str
    route_index: int
    fitness: float | None = None
    rmse_m: float | None = None
    descriptor_distance: float | None = None
    trusted_tracking_candidate: bool = False


@dataclass(frozen=True)
class ContinuousLocalizationResult:
    map_from_body: np.ndarray
    map_from_odom: np.ndarray
    observation_accepted: bool
    selected_submap: int | None
    selected_route_index: int
    hypothesis_count: int
    score_margin: float | None
    correction_translation_m: float | None
    correction_yaw_deg: float | None
    mode: str
    relocalization_streak: int
    fitness: float | None
    rmse_m: float | None
    descriptor_distance: float | None


def _scan_context_alignment(
    reference: np.ndarray, query: np.ndarray
) -> tuple[float, float]:
    """Return yaw-invariant descriptor distance and signed query rotation."""

    sectors = reference.shape[1]
    correlation = np.fft.irfft(
        np.sum(
            np.conj(np.fft.rfft(reference, axis=1))
            * np.fft.rfft(query, axis=1),
            axis=0,
        ),
        n=sectors,
    )
    index = int(np.argmax(correlation))
    signed_index = index if index <= sectors // 2 else index - sectors
    similarity = float(np.clip(correlation[index], -1.0, 1.0))
    distance = float(np.sqrt(max(0.0, 2.0 - 2.0 * similarity)))
    return distance, float(signed_index * 360.0 / sectors)


def _route_prior_penalties(
    config: ContinuousLocalizationConfig,
    hypothesis_route_index: int,
    candidate_route_index: int,
    odometry_route_index: int,
) -> tuple[float, float]:
    route_step_penalty = config.route_step_cost * abs(
        candidate_route_index - hypothesis_route_index
    )
    route_lag_penalty = config.route_lag_cost * max(
        0,
        min(
            odometry_route_index,
            hypothesis_route_index + config.max_tracking_route_step,
        )
        - candidate_route_index,
    )
    return route_step_penalty, route_lag_penalty


def _multisession_support_is_reliable(
    config: ContinuousLocalizationConfig,
    canonical: dict[str, object],
    support: dict[str, object],
) -> bool:
    canonical_pose = np.asarray(canonical["observation"], dtype=np.float64)
    support_pose = np.asarray(support["observation"], dtype=np.float64)
    return bool(
        float(support["rmse_m"]) <= float(canonical["rmse_m"])
        and float(support["translation_innovation_m"])
        <= float(canonical["translation_innovation_m"])
        and np.linalg.norm(
            support_pose[:3, 3] - canonical_pose[:3, 3]
        ) <= config.multisession_max_translation_disagreement_m
        and yaw_error_deg(canonical_pose, support_pose)
        <= config.multisession_max_yaw_disagreement_deg
    )


def _multisession_route_cost(
    config: ContinuousLocalizationConfig,
    canonical: dict[str, object],
    *,
    hypothesis_route_index: int,
    odometry_route_index: int,
    reliable_support_count: int,
) -> float:
    route_index = int(canonical["route_index"])
    return float(
        float(canonical["rmse_m"])
        + config.multisession_fitness_weight
        * (1.0 - float(canonical["fitness"]))
        + config.multisession_descriptor_weight
        * float(canonical["descriptor_distance"])
        + config.multisession_translation_innovation_weight
        * float(canonical["translation_innovation_m"])
        + config.multisession_yaw_innovation_weight
        * float(canonical["yaw_innovation_deg"])
        + config.multisession_route_step_weight
        * abs(route_index - hypothesis_route_index)
        + config.multisession_odometry_route_weight
        * abs(route_index - odometry_route_index)
        - config.multisession_repeatability_bonus * reliable_support_count
    )


def _registration_rejection_reasons(
    config: ContinuousLocalizationConfig,
    *,
    fitness: float,
    rmse_m: float,
    descriptor_distance: float,
    translation_innovation_m: float,
    yaw_innovation_deg: float,
    continuity_candidate: bool,
    recovering_without_metric_prior: bool,
) -> list[str]:
    minimum_fitness = (
        config.continuity_min_fitness
        if continuity_candidate else config.min_fitness
    )
    maximum_rmse = (
        config.continuity_max_rmse_m
        if continuity_candidate else config.max_rmse_m
    )
    reasons = []
    if fitness < minimum_fitness:
        reasons.append("fitness")
    if rmse_m > maximum_rmse:
        reasons.append("rmse")
    # During recovery the predicted map pose is explicitly untrusted. Large
    # corrections are checked later by route consistency and repeated-pose
    # confirmation, not against the stale prediction used to rank candidates.
    if not recovering_without_metric_prior:
        if translation_innovation_m > config.max_translation_innovation_m:
            reasons.append("translation_innovation")
        if yaw_innovation_deg > config.max_yaw_innovation_deg:
            reasons.append("yaw_innovation")
    if descriptor_distance > config.max_descriptor_distance:
        reasons.append("descriptor")
    return reasons


def _hypothesis_selection_key(
    hypothesis: LocalizationHypothesis,
) -> tuple[int, float]:
    """Prefer a gated trusted-frame correction over an unobserved coast."""

    return (0 if hypothesis.trusted_tracking_candidate else 1, hypothesis.cost)


def _bounded_se3_update(
    current: np.ndarray,
    target: np.ndarray,
    *,
    translation_gain: float,
    rotation_gain: float,
    max_translation_step_m: float,
    max_rotation_step_deg: float,
) -> np.ndarray:
    """Move an SE(3) estimate toward an observation without a pose jump."""

    if not 0.0 <= translation_gain <= 1.0:
        raise ValueError("translation_gain must be within [0, 1]")
    if not 0.0 <= rotation_gain <= 1.0:
        raise ValueError("rotation_gain must be within [0, 1]")
    if max_translation_step_m <= 0.0:
        raise ValueError("max_translation_step_m must be positive")
    if max_rotation_step_deg <= 0.0:
        raise ValueError("max_rotation_step_deg must be positive")

    current_pose = np.asarray(current, dtype=np.float64)
    target_pose = np.asarray(target, dtype=np.float64)
    updated = current_pose.copy()

    translation_step = (
        translation_gain * (target_pose[:3, 3] - current_pose[:3, 3])
    )
    translation_norm = float(np.linalg.norm(translation_step))
    if translation_norm > max_translation_step_m:
        translation_step *= max_translation_step_m / translation_norm
    updated[:3, 3] += translation_step

    current_rotation = Rotation.from_matrix(current_pose[:3, :3])
    target_rotation = Rotation.from_matrix(target_pose[:3, :3])
    relative = current_rotation.inv() * target_rotation
    rotation_vector = relative.as_rotvec() * rotation_gain
    rotation_angle = float(np.linalg.norm(rotation_vector))
    max_rotation_rad = float(np.deg2rad(max_rotation_step_deg))
    if rotation_angle > max_rotation_rad:
        rotation_vector *= max_rotation_rad / rotation_angle
    updated[:3, :3] = (
        current_rotation * Rotation.from_rotvec(rotation_vector)
    ).as_matrix()
    updated[3] = (0.0, 0.0, 0.0, 1.0)
    return updated


class ContinuousMapLocalizer:
    """Track a frozen map while preserving high-rate local odometry motion."""

    def __init__(
        self,
        map_dir: Path,
        initial_map_from_body: np.ndarray | None,
        initial_odom_from_body: np.ndarray,
        *,
        config: ContinuousLocalizationConfig = ContinuousLocalizationConfig(),
    ):
        self.config = config
        if config.registration_backend in ("kiss_icp", "kiss_vgicp"):
            try:
                from kiss_icp.mapping import VoxelHashMap
                from kiss_icp.registration import Registration
            except ImportError as error:
                raise RuntimeError("map localization requires kiss-icp") from error
            self.voxel_map_type = VoxelHashMap
            self.registration = Registration(
                max_num_iterations=config.max_icp_iterations,
                convergence_criterion=1.0e-4,
                max_num_threads=max(1, config.icp_threads),
            )
            self.fine_registration = Registration(
                max_num_iterations=config.fine_max_icp_iterations,
                convergence_criterion=1.0e-4,
                max_num_threads=max(1, config.icp_threads),
            )
            self.o3d = None
            self.vgicp = None
            if config.registration_backend == "kiss_vgicp":
                native_dir = Path(__file__).resolve().parents[1] / "native" / "build"
                if str(native_dir) not in sys.path:
                    sys.path.insert(0, str(native_dir))
                try:
                    self.vgicp = importlib.import_module("s10_vgicp")
                except ImportError as error:
                    raise RuntimeError(
                        "kiss_vgicp backend is not built; run "
                        "deployment/scripts/mapping/build_multilap_optimizer.sh"
                    ) from error
        elif config.registration_backend == "gtsam_vgicp":
            native_dir = Path(__file__).resolve().parents[1] / "native" / "build"
            if str(native_dir) not in sys.path:
                sys.path.insert(0, str(native_dir))
            try:
                self.vgicp = importlib.import_module("s10_vgicp")
            except ImportError as error:
                raise RuntimeError(
                    "gtsam_vgicp backend is not built; run "
                    "deployment/scripts/mapping/build_multilap_optimizer.sh"
                ) from error
            self.voxel_map_type = None
            self.registration = None
            self.fine_registration = None
            self.o3d = None
        elif config.registration_backend == "open3d_gicp":
            try:
                import open3d as o3d
            except ImportError as error:
                raise RuntimeError("open3d_gicp backend requires Open3D") from error
            self.o3d = o3d
            self.vgicp = None
        else:
            raise ValueError(
                "registration_backend must be 'kiss_icp', 'kiss_vgicp', "
                "'gtsam_vgicp', or 'open3d_gicp'"
            )
        target_keys = {
            "route_submap": "points_anchor_m",
            "static_consensus": "registration_points_anchor_m",
        }
        if config.registration_target not in target_keys:
            raise ValueError(
                "registration_target must be 'route_submap' or 'static_consensus'"
            )
        self.target_key = target_keys[config.registration_target]
        self.root = map_dir.expanduser().resolve()
        manifest = json.loads(
            (self.root / "localization_map_manifest.json").read_text(encoding="utf-8")
        )
        self.submaps = []
        use_all_route_variants = (
            manifest.get("map_type") == "ordered_topometric_route_submaps"
        )
        for entry in manifest["submaps"]:
            if (
                not use_all_route_variants
                and int(entry["reference_session"])
                != config.canonical_reference_session
            ):
                continue
            with np.load(self.root / entry["file"], allow_pickle=False) as payload:
                if self.target_key not in payload.files:
                    raise ValueError(
                        f"submap {entry['file']} does not provide {self.target_key}"
                    )
                self.submaps.append({
                    "id": int(entry["submap_id"]),
                    "route_index": int(entry.get("route_index", len(self.submaps))),
                    "reference_session": int(entry["reference_session"]),
                    "tracking_eligible": bool(entry.get(
                        "tracking_eligible",
                        int(entry["reference_session"])
                        == config.canonical_reference_session,
                    )),
                    "progress": float(entry["anchor_progress_fraction"]),
                    "progress_m": float(entry["anchor_progress_m"]),
                    "anchor": np.asarray(payload["anchor_pose"], dtype=np.float64),
                    "descriptor": np.asarray(payload["descriptor"], dtype=np.float64),
                    "file": self.root / entry["file"],
                })
        if len(self.submaps) < 10:
            raise ValueError("localization map has too few canonical tracking submaps")
        self.route_progress_m = {}
        for submap in self.submaps:
            self.route_progress_m.setdefault(
                int(submap["route_index"]), float(submap["progress_m"])
            )
        self.max_route_index = max(self.route_progress_m)

        map_from_body = (
            np.asarray(self.submaps[0]["anchor"], dtype=np.float64)
            if initial_map_from_body is None
            else np.asarray(initial_map_from_body, dtype=np.float64)
        )
        initial_map_from_odom = (
            map_from_body
            @ np.linalg.inv(np.asarray(initial_odom_from_body, dtype=np.float64))
        )
        self.hypotheses = [LocalizationHypothesis(
            cost=0.0,
            map_from_odom=initial_map_from_odom,
            coast_updates=0,
            source="start_anchor",
            route_index=0,
        )]
        # The beam is allowed to search broadly, but navigation consumes only
        # this trusted transform.  A broad candidate must be corroborated over
        # several updates before it can replace the trusted transform.
        self.trusted_map_from_odom = initial_map_from_odom.copy()
        self.trusted_route_index = 0
        # Relate local odometry distance to the ordered route independently of
        # any held-out/global pose.  This lets the candidate window advance
        # through temporarily unmatchable vegetation or route deviations.
        self.route_progress_offset_m = 0.0
        self.pending_relocalization: np.ndarray | None = None
        self.pending_route_index: int | None = None
        self.relocalization_streak = 0
        self.query_history: list[tuple[np.ndarray, np.ndarray]] = []
        self.cloud_cache: OrderedDict[int, object] = OrderedDict()
        self.last_candidate_audits: list[dict[str, object]] = []
        self.last_odometry_route_index = 0

    def _set_trusted_route_index(
        self, route_index: int, traveled_distance_m: float
    ) -> None:
        previous_route_index = self.trusted_route_index
        next_route_index = int(route_index)
        if next_route_index <= previous_route_index:
            return
        self.trusted_route_index = next_route_index
        # Recalibrate the distance offset only after actual forward topological
        # progress. Rewriting it for every match to the same submap pins the
        # odometry prior to that index forever while the robot moves away.
        if self.trusted_route_index > previous_route_index:
            self.route_progress_offset_m = (
                traveled_distance_m
                - self.route_progress_m[self.trusted_route_index]
            )

    def _odometry_route_index(self, traveled_distance_m: float) -> int:
        expected_progress_m = max(
            0.0, traveled_distance_m - self.route_progress_offset_m
        )
        route_indices = np.asarray(sorted(self.route_progress_m), dtype=np.int64)
        progress_values = np.asarray(
            [self.route_progress_m[int(index)] for index in route_indices]
        )
        return int(route_indices[np.argmin(
            np.abs(progress_values - expected_progress_m)
        )])

    def _target_cloud(self, submap: dict[str, object]):
        submap_id = int(submap["id"])
        cached = self.cloud_cache.pop(submap_id, None)
        if cached is not None:
            self.cloud_cache[submap_id] = cached
            return cached
        with np.load(Path(submap["file"]), allow_pickle=False) as payload:
            points = voxel_downsample(
                np.asarray(payload[self.target_key], dtype=np.float64),
                self.config.query_voxel_m,
            )
            fine_points = voxel_downsample(
                np.asarray(
                    payload.get("fine_points_anchor_m", payload[self.target_key]),
                    dtype=np.float64,
                ),
                self.config.fine_query_voxel_m,
            )
        if self.config.registration_backend in ("kiss_icp", "kiss_vgicp"):
            cloud = self.voxel_map_type(
                voxel_size=self.config.query_voxel_m,
                max_distance=1000.0,
                max_points_per_voxel=20,
            )
            cloud.add_points(points)
            fine_cloud = self.voxel_map_type(
                voxel_size=self.config.fine_query_voxel_m,
                max_distance=1000.0,
                max_points_per_voxel=20,
            )
            fine_cloud.add_points(fine_points)
            cached_parts = [
                cloud,
                cKDTree(points),
                fine_cloud,
                cKDTree(fine_points),
            ]
            if self.config.registration_backend == "kiss_vgicp":
                target_points = voxel_downsample(
                    fine_points, self.config.vgicp_target_voxel_m
                )
                if len(target_points) > self.config.vgicp_max_target_points:
                    indices = np.linspace(
                        0,
                        len(target_points) - 1,
                        self.config.vgicp_max_target_points,
                        dtype=np.int64,
                    )
                    target_points = np.ascontiguousarray(target_points[indices])
                target_frame = self.vgicp.Frame(
                    np.ascontiguousarray(target_points, dtype=np.float64),
                    self.config.vgicp_covariance_neighbors,
                    self.config.vgicp_num_threads,
                    self.config.vgicp_covariance_scale,
                )
                cached_parts.append(self.vgicp.Target(
                    target_frame, self.config.vgicp_voxel_resolution_m
                ))
            cached_value = tuple(cached_parts)
        elif self.config.registration_backend == "gtsam_vgicp":
            target_points = voxel_downsample(
                fine_points, self.config.vgicp_target_voxel_m
            )
            if len(target_points) > self.config.vgicp_max_target_points:
                indices = np.linspace(
                    0,
                    len(target_points) - 1,
                    self.config.vgicp_max_target_points,
                    dtype=np.int64,
                )
                target_points = np.ascontiguousarray(target_points[indices])
            target_frame = self.vgicp.Frame(
                np.ascontiguousarray(target_points, dtype=np.float64),
                self.config.vgicp_covariance_neighbors,
                self.config.vgicp_num_threads,
                self.config.vgicp_covariance_scale,
            )
            cached_value = (
                self.vgicp.Target(
                    target_frame, self.config.vgicp_voxel_resolution_m
                ),
                cKDTree(fine_points),
            )
        else:
            cached_value = self._point_cloud(points)
        self.cloud_cache[submap_id] = cached_value
        while len(self.cloud_cache) > self.config.cloud_cache_size:
            self.cloud_cache.popitem(last=False)
        return cached_value

    def _register(
        self,
        query_points: np.ndarray,
        fine_query_points: np.ndarray,
        query_cloud,
        submap: dict[str, object],
        initial: np.ndarray,
    ) -> tuple[np.ndarray, float, float]:
        target = self._target_cloud(submap)
        if self.config.registration_backend in ("kiss_icp", "kiss_vgicp"):
            voxel_map, tree, fine_voxel_map, fine_tree = target[:4]
            transform = self.registration.align_points_to_map(
                points=query_points,
                voxel_map=voxel_map,
                initial_guess=initial,
                max_correspondance_distance=self.config.max_correspondence_m,
                kernel=self.config.max_correspondence_m / 3.0,
            )
            aligned = query_points @ transform[:3, :3].T + transform[:3, 3]
            distances, _ = tree.query(aligned, workers=-1)
            inliers = distances <= self.config.max_correspondence_m
            fitness = float(np.mean(inliers))
            rmse = (
                float(np.sqrt(np.mean(np.square(distances[inliers]))))
                if np.any(inliers) else float("inf")
            )
            fine_transform = self.fine_registration.align_points_to_map(
                points=fine_query_points,
                voxel_map=fine_voxel_map,
                initial_guess=transform,
                max_correspondance_distance=self.config.fine_max_correspondence_m,
                kernel=self.config.fine_max_correspondence_m / 3.0,
            )
            fine_aligned = (
                fine_query_points @ fine_transform[:3, :3].T
                + fine_transform[:3, 3]
            )
            fine_distances, _ = fine_tree.query(fine_aligned, workers=-1)
            fine_inliers = fine_distances <= self.config.fine_max_correspondence_m
            fine_fitness = float(np.mean(fine_inliers))
            fine_rmse = (
                float(np.sqrt(np.mean(np.square(fine_distances[fine_inliers]))))
                if np.any(fine_inliers) else float("inf")
            )
            if (
                fine_fitness >= self.config.fine_min_fitness
                and fine_rmse <= self.config.fine_max_rmse_m
            ):
                kiss_transform = np.asarray(fine_transform)
                kiss_fitness = fine_fitness
                kiss_rmse = fine_rmse
            else:
                kiss_transform = np.asarray(transform)
                kiss_fitness = fitness
                kiss_rmse = rmse
            if self.config.registration_backend == "kiss_icp":
                return kiss_transform, kiss_fitness, kiss_rmse

            vgicp_result = self.vgicp.align(
                target[4],
                query_cloud,
                np.ascontiguousarray(initial, dtype=np.float64),
                self.config.vgicp_max_iterations,
                self.config.vgicp_num_threads,
            )
            if not bool(vgicp_result["valid"]):
                return kiss_transform, kiss_fitness, kiss_rmse
            vgicp_transform = np.asarray(vgicp_result["pose"], dtype=np.float64)
            vgicp_aligned = (
                fine_query_points @ vgicp_transform[:3, :3].T
                + vgicp_transform[:3, 3]
            )
            vgicp_distances, _ = fine_tree.query(vgicp_aligned, workers=-1)
            vgicp_inliers = (
                vgicp_distances <= self.config.fine_max_correspondence_m
            )
            vgicp_fitness = float(np.mean(vgicp_inliers))
            vgicp_rmse = (
                float(np.sqrt(np.mean(np.square(vgicp_distances[vgicp_inliers]))))
                if np.any(vgicp_inliers) else float("inf")
            )
            disagreement = np.linalg.inv(kiss_transform) @ vgicp_transform
            disagreement_translation = float(np.linalg.norm(disagreement[:3, 3]))
            disagreement_rotation = rotation_distance_deg(disagreement[:3, :3])
            vgicp_is_consistent = bool(
                vgicp_fitness >= self.config.fine_min_fitness
                and vgicp_rmse <= self.config.fine_max_rmse_m
                and disagreement_translation
                <= self.config.hybrid_max_translation_disagreement_m
                and disagreement_rotation
                <= self.config.hybrid_max_rotation_disagreement_deg
            )
            if vgicp_is_consistent:
                return vgicp_transform, vgicp_fitness, vgicp_rmse
            return kiss_transform, kiss_fitness, kiss_rmse
        if self.config.registration_backend == "gtsam_vgicp":
            target_frame, fine_tree = target
            result = self.vgicp.align(
                target_frame,
                query_cloud,
                np.ascontiguousarray(initial, dtype=np.float64),
                self.config.vgicp_max_iterations,
                self.config.vgicp_num_threads,
            )
            if not bool(result["valid"]):
                return np.asarray(initial), 0.0, float("inf")
            transform = np.asarray(result["pose"], dtype=np.float64)
            aligned = (
                fine_query_points @ transform[:3, :3].T
                + transform[:3, 3]
            )
            distances, _ = fine_tree.query(aligned, workers=-1)
            inliers = distances <= self.config.fine_max_correspondence_m
            fitness = float(np.mean(inliers))
            rmse = (
                float(np.sqrt(np.mean(np.square(distances[inliers]))))
                if np.any(inliers) else float("inf")
            )
            return transform, fitness, rmse
        registration = self.o3d.pipelines.registration.registration_generalized_icp(
            query_cloud,
            target,
            self.config.max_correspondence_m,
            initial,
            self.o3d.pipelines.registration.TransformationEstimationForGeneralizedICP(),
            self.o3d.pipelines.registration.ICPConvergenceCriteria(
                max_iteration=self.config.max_icp_iterations
            ),
        )
        return (
            registration.transformation,
            float(registration.fitness),
            float(registration.inlier_rmse),
        )

    def _candidate_submaps(
        self,
        hypothesis: LocalizationHypothesis,
        traveled_distance_m: float,
    ) -> list[dict[str, object]]:
        config = self.config
        recovery_updates = max(
            0, hypothesis.coast_updates - config.max_coast_updates
        )
        lookahead = min(
            config.recovery_lookahead_submaps,
            config.route_lookahead_submaps
            + recovery_updates * config.recovery_growth_per_coast,
        )
        odometry_route_index = self._odometry_route_index(traveled_distance_m)
        self.last_odometry_route_index = odometry_route_index
        tracking_begin = max(
            0, hypothesis.route_index - config.route_backtrack_submaps
        )
        tracking_end = min(
            self.max_route_index + 1, hypothesis.route_index + lookahead + 1
        )
        allowed_indices = set(range(tracking_begin, tracking_end))
        if recovery_updates > 0:
            # Add only a compact odometry-predicted window. Including the
            # whole interval creates many false candidates on repeated walls.
            odometry_begin = max(
                0,
                odometry_route_index
                - config.recovery_odometry_radius_submaps,
            )
            odometry_end = min(
                self.max_route_index + 1,
                odometry_route_index
                + config.recovery_odometry_radius_submaps + 1,
            )
            allowed_indices.update(range(odometry_begin, odometry_end))
        return [
            submap for submap in self.submaps
            if int(submap["route_index"]) in allowed_indices
        ]

    def _point_cloud(self, points: np.ndarray):
        cloud = self.o3d.geometry.PointCloud()
        cloud.points = self.o3d.utility.Vector3dVector(points)
        return cloud

    def update(
        self,
        points_body_m: np.ndarray,
        odom_from_body: np.ndarray,
        *,
        traveled_distance_m: float,
        allow_large_relocalization: bool = False,
    ) -> ContinuousLocalizationResult:
        config = self.config
        odom_pose = np.asarray(odom_from_body, dtype=np.float64)
        if traveled_distance_m < config.minimum_motion_before_matching_m:
            # Route-map matching is intentionally disabled during the initial
            # metre. Avoid voxelization, temporal query fusion and descriptor
            # construction as well; on the AGX those unused operations were
            # enough to make the read-only odometry worker fall behind before
            # the robot had moved at all.
            self.last_candidate_audits = []
            self.pending_relocalization = None
            self.pending_route_index = None
            self.relocalization_streak = 0
            return ContinuousLocalizationResult(
                map_from_body=self.trusted_map_from_odom @ odom_pose,
                map_from_odom=self.trusted_map_from_odom.copy(),
                observation_accepted=False,
                selected_submap=None,
                selected_route_index=self.trusted_route_index,
                hypothesis_count=len(self.hypotheses),
                score_margin=None,
                correction_translation_m=None,
                correction_yaw_deg=None,
                mode="coasting",
                relocalization_streak=0,
                fitness=None,
                rmse_m=None,
                descriptor_distance=None,
            )
        raw_points = np.asarray(points_body_m, dtype=np.float64)
        ranges = np.linalg.norm(raw_points, axis=1)
        fine_points = voxel_downsample(
            raw_points[
                np.isfinite(raw_points).all(axis=1)
                & (ranges >= 0.45)
                & (ranges <= config.max_sensor_range_m)
            ],
            config.fine_query_voxel_m,
        )
        if len(fine_points) < 100:
            raise ValueError("localization query has fewer than 100 valid points")
        self.query_history.append((fine_points, odom_pose.copy()))
        self.query_history = self.query_history[-config.query_submap_updates :]
        body_from_odom = np.linalg.inv(odom_pose)
        query_parts = []
        for history_points, history_odom_pose in self.query_history:
            current_from_history = body_from_odom @ history_odom_pose
            query_parts.append(
                history_points @ current_from_history[:3, :3].T
                + current_from_history[:3, 3]
            )
        fine_query_points = voxel_downsample(
            np.concatenate(query_parts, axis=0), config.fine_query_voxel_m
        )
        if len(fine_query_points) > config.fine_max_query_points:
            indices = np.linspace(
                0,
                len(fine_query_points) - 1,
                config.fine_max_query_points,
                dtype=np.int64,
            )
            fine_query_points = np.ascontiguousarray(fine_query_points[indices])
        query_points = voxel_downsample(
            fine_query_points, config.query_voxel_m
        )
        if config.registration_backend == "open3d_gicp":
            query_cloud = self._point_cloud(query_points)
        elif config.registration_backend in ("gtsam_vgicp", "kiss_vgicp"):
            vgicp_query = voxel_downsample(
                fine_query_points, config.vgicp_source_voxel_m
            )
            query_cloud = self.vgicp.Frame(
                np.ascontiguousarray(vgicp_query, dtype=np.float64),
                config.vgicp_covariance_neighbors,
                config.vgicp_num_threads,
                config.vgicp_covariance_scale,
            )
        else:
            query_cloud = None
        query_descriptor = scan_context_descriptor(query_points, 20, 60)
        children: list[LocalizationHypothesis] = []
        candidate_audits: list[dict[str, object]] = []
        trusted_prediction = self.trusted_map_from_odom @ odom_pose

        for hypothesis in self.hypotheses:
            predicted = hypothesis.map_from_odom @ odom_pose
            if (
                traveled_distance_m < config.minimum_motion_before_matching_m
                or hypothesis.coast_updates < config.max_coast_updates
            ):
                coast_cost = 0.0 if traveled_distance_m < config.minimum_motion_before_matching_m else (
                    config.coast_base_cost + 0.03 * hypothesis.coast_updates
                )
                children.append(LocalizationHypothesis(
                    cost=hypothesis.cost + coast_cost,
                    map_from_odom=hypothesis.map_from_odom,
                    coast_updates=hypothesis.coast_updates + 1,
                    source="coast",
                    route_index=hypothesis.route_index,
                ))
            if traveled_distance_m < config.minimum_motion_before_matching_m:
                continue

            eligible_all = self._candidate_submaps(
                hypothesis, traveled_distance_m
            )
            eligible = (
                eligible_all
                if hypothesis.coast_updates > config.max_coast_updates
                else [
                    submap for submap in eligible_all
                    if bool(submap["tracking_eligible"])
                ]
            )
            scored = []
            for submap in eligible:
                descriptor_distance, _ = _scan_context_alignment(
                    submap["descriptor"], query_descriptor
                )
                predicted_distance = float(np.linalg.norm(
                    submap["anchor"][:2, 3] - predicted[:2, 3]
                ))
                # Route geometry can be globally distorted after hundreds of
                # metres.  Place appearance therefore ranks the ordered
                # window; predicted XY is only a tie-breaker.
                scored.append((descriptor_distance + 0.01 * predicted_distance,
                               descriptor_distance, submap))
            selection_scored = scored
            if config.multisession_consensus_enabled:
                canonical_scored = [
                    item for item in scored
                    if int(item[2]["reference_session"])
                    == config.canonical_reference_session
                ]
                if canonical_scored:
                    selection_scored = canonical_scored
            ranked_all = sorted(selection_scored, key=lambda item: item[0])
            local_limit = hypothesis.route_index + config.route_lookahead_submaps
            ranked_local = [
                item for item in ranked_all
                if int(item[2]["route_index"]) <= local_limit
            ]
            # Preserve the appearance-ranked candidates that work during
            # ordinary tracking.  Only after sustained coasting, append route
            # candidates predicted from local travel distance; never replace
            # the established candidates with this weaker prior.
            ranked_odometry = sorted(
                selection_scored,
                key=lambda item: (
                    abs(int(item[2]["route_index"]) - self.last_odometry_route_index),
                    item[1],
                ),
            )
            odometry_budget = min(
                config.odometry_tracking_candidate_count,
                max(0, config.candidate_count - 1),
            )
            appearance_budget = max(1, config.candidate_count - odometry_budget)
            local_budget = max(1, appearance_budget // 2)
            selected_candidates = ranked_local[:local_budget]
            selected_ids = {int(item[2]["id"]) for item in selected_candidates}
            selected_candidates.extend(
                item for item in ranked_all
                if int(item[2]["id"]) not in selected_ids
            )
            selected_candidates = selected_candidates[:appearance_budget]
            selected_ids = {int(item[2]["id"]) for item in selected_candidates}
            tracking_odometry_added = 0
            for item in ranked_odometry:
                submap_id = int(item[2]["id"])
                if submap_id in selected_ids:
                    continue
                selected_candidates.append(item)
                selected_ids.add(submap_id)
                tracking_odometry_added += 1
                if tracking_odometry_added >= odometry_budget:
                    break
            if hypothesis.coast_updates > config.max_coast_updates:
                recovery_added = 0
                for item in ranked_odometry:
                    submap_id = int(item[2]["id"])
                    if submap_id in selected_ids:
                        continue
                    selected_candidates.append(item)
                    selected_ids.add(submap_id)
                    recovery_added += 1
                    if recovery_added >= config.odometry_recovery_candidate_count:
                        break
            if config.multisession_consensus_enabled:
                selected_routes = {
                    int(item[2]["route_index"]) for item in selected_candidates
                }
                support_candidates = sorted(
                    (
                        item for item in scored
                        if int(item[2]["reference_session"])
                        != config.canonical_reference_session
                        and bool(item[2]["tracking_eligible"])
                        and int(item[2]["route_index"]) in selected_routes
                        and int(item[2]["id"]) not in selected_ids
                    ),
                    key=lambda item: (item[1], item[0]),
                )
                selected_candidates.extend(support_candidates)
                selected_ids.update(
                    int(item[2]["id"]) for item in support_candidates
                )
            registered_candidates: list[dict[str, object]] = []
            for _, descriptor_distance, submap in selected_candidates:
                # Continuous tracking already has a much stronger SE(3) prior
                # from T_odom_base.  Scan Context is rotation invariant and
                # sector-quantized, so replacing this yaw with its coarse yaw
                # estimate creates avoidable ICP jumps.
                recovering_without_metric_prior = bool(
                    hypothesis.coast_updates > config.max_coast_updates
                )
                # Route submaps are expressed in their anchor frame. After
                # sustained coasting, T_map_odom may be metres away from the
                # true pose and is a harmful ICP initializer. The ordered
                # route/distance prior has already selected a compact set of
                # nearby anchors, so restart those candidates at the local
                # anchor origin. Ordinary tracking retains the metric prior.
                initial = (
                    np.eye(4, dtype=np.float64)
                    if recovering_without_metric_prior
                    else np.linalg.inv(submap["anchor"]) @ predicted
                )
                registration_transform, registration_fitness, registration_rmse = self._register(
                    query_points, fine_query_points, query_cloud, submap, initial
                )
                observation = submap["anchor"] @ registration_transform
                correction_translation = float(np.linalg.norm(
                    observation[:3, 3] - predicted[:3, 3]
                ))
                correction_yaw = yaw_error_deg(predicted, observation)
                is_continuity_candidate = bool(
                    abs(int(submap["route_index"]) - hypothesis.route_index) <= 2
                    and hypothesis.coast_updates <= config.max_coast_updates
                    and correction_translation <= config.continuity_max_translation_m
                    and correction_yaw <= config.continuity_max_yaw_deg
                )
                rejection_reasons = _registration_rejection_reasons(
                    config,
                    fitness=registration_fitness,
                    rmse_m=registration_rmse,
                    descriptor_distance=descriptor_distance,
                    translation_innovation_m=correction_translation,
                    yaw_innovation_deg=correction_yaw,
                    continuity_candidate=is_continuity_candidate,
                    recovering_without_metric_prior=(
                        recovering_without_metric_prior
                    ),
                )
                candidate_audit = {
                    "candidate_submap_id": int(submap["id"]),
                    "hypothesis_route_index": hypothesis.route_index,
                    "candidate_route_index": int(submap["route_index"]),
                    "reference_session": int(submap["reference_session"]),
                    "observation_position_m": observation[:3, 3].tolist(),
                    "fitness": registration_fitness,
                    "rmse_m": registration_rmse,
                    "descriptor_distance": descriptor_distance,
                    "translation_innovation_m": correction_translation,
                    "yaw_innovation_deg": correction_yaw,
                    "continuity_gate": is_continuity_candidate,
                    "rejection_reasons": rejection_reasons,
                }
                candidate_audits.append(candidate_audit)
                if rejection_reasons:
                    continue
                if config.multisession_consensus_enabled:
                    registered_candidates.append({
                        "submap": submap,
                        "audit": candidate_audit,
                        "route_index": int(submap["route_index"]),
                        "reference_session": int(submap["reference_session"]),
                        "observation": observation,
                        "fitness": registration_fitness,
                        "rmse_m": registration_rmse,
                        "descriptor_distance": descriptor_distance,
                        "translation_innovation_m": correction_translation,
                        "yaw_innovation_deg": correction_yaw,
                        "continuity_candidate": is_continuity_candidate,
                    })
                    continue
                candidate_route_index = int(submap["route_index"])
                # Local wheel/IMU odometry is not trusted as a metric map
                # pose, but traveled distance is a useful weak topological
                # prior. Penalize hypotheses that remain behind that prior
                # after their scan registration has already passed all
                # geometric gates. This prevents an overlapping penultimate
                # submap from permanently masking a valid loop-end submap.
                route_step_penalty, route_lag_penalty = _route_prior_penalties(
                    config,
                    hypothesis.route_index,
                    candidate_route_index,
                    self.last_odometry_route_index,
                )
                incremental_cost = (
                    registration_rmse
                    + 0.30 * (1.0 - registration_fitness)
                    + 0.12 * descriptor_distance
                    + 0.035 * correction_translation
                    + 0.003 * correction_yaw
                    + route_step_penalty
                    + route_lag_penalty
                )
                candidate_audit["route_step_penalty"] = route_step_penalty
                candidate_audit["route_lag_penalty"] = route_lag_penalty
                candidate_audit["incremental_cost"] = incremental_cost
                source = f"submap:{submap['id']}"
                progress_route_index = max(
                    hypothesis.route_index, int(submap["route_index"])
                )
                observed_map_from_odom = observation @ np.linalg.inv(odom_pose)
                trusted_correction_translation = float(np.linalg.norm(
                    observation[:3, 3] - trusted_prediction[:3, 3]
                ))
                trusted_correction_yaw = yaw_error_deg(
                    trusted_prediction, observation
                )
                trusted_tracking_candidate = bool(
                    trusted_correction_translation
                    <= config.max_tracking_translation_correction_m
                    and trusted_correction_yaw
                    <= config.max_tracking_yaw_correction_deg
                    and abs(progress_route_index - self.trusted_route_index)
                    <= config.max_tracking_route_step
                )
                candidate_audit["trusted_tracking_candidate"] = (
                    trusted_tracking_candidate
                )
                candidate_audit["trusted_translation_correction_m"] = (
                    trusted_correction_translation
                )
                candidate_audit["trusted_yaw_correction_deg"] = (
                    trusted_correction_yaw
                )
                if config.temporal_fusion_enabled and is_continuity_candidate:
                    next_map_from_odom = _bounded_se3_update(
                        hypothesis.map_from_odom,
                        observed_map_from_odom,
                        translation_gain=config.tracking_translation_gain,
                        rotation_gain=config.tracking_rotation_gain,
                        max_translation_step_m=(
                            config.tracking_max_translation_step_m
                        ),
                        max_rotation_step_deg=(
                            config.tracking_max_rotation_step_deg
                        ),
                    )
                else:
                    next_map_from_odom = observed_map_from_odom
                children.append(LocalizationHypothesis(
                    cost=0.82 * hypothesis.cost + incremental_cost,
                    map_from_odom=next_map_from_odom,
                    coast_updates=0,
                    source=source,
                    route_index=progress_route_index,
                    fitness=registration_fitness,
                    rmse_m=registration_rmse,
                    descriptor_distance=float(descriptor_distance),
                    trusted_tracking_candidate=trusted_tracking_candidate,
                ))

            if config.multisession_consensus_enabled:
                candidates_by_route: dict[int, list[dict[str, object]]] = {}
                for candidate in registered_candidates:
                    candidates_by_route.setdefault(
                        int(candidate["route_index"]), []
                    ).append(candidate)
                for route_index, route_candidates in candidates_by_route.items():
                    canonical_candidates = [
                        candidate for candidate in route_candidates
                        if int(candidate["reference_session"])
                        == config.canonical_reference_session
                    ]
                    if not canonical_candidates:
                        continue
                    canonical = min(
                        canonical_candidates,
                        key=lambda candidate: (
                            float(candidate["rmse_m"]),
                            -float(candidate["fitness"]),
                        ),
                    )
                    reliable_supports = [
                        candidate for candidate in route_candidates
                        if int(candidate["reference_session"])
                        != config.canonical_reference_session
                        and _multisession_support_is_reliable(
                            config, canonical, candidate
                        )
                    ]
                    observation = np.asarray(
                        canonical["observation"], dtype=np.float64
                    ).copy()
                    for support_number, support in enumerate(
                        reliable_supports, start=1
                    ):
                        gain = 1.0 / float(support_number + 1)
                        observation = _bounded_se3_update(
                            observation,
                            np.asarray(support["observation"], dtype=np.float64),
                            translation_gain=gain,
                            rotation_gain=gain,
                            max_translation_step_m=float("inf"),
                            max_rotation_step_deg=float("inf"),
                        )
                    incremental_cost = _multisession_route_cost(
                        config,
                        canonical,
                        hypothesis_route_index=hypothesis.route_index,
                        odometry_route_index=self.last_odometry_route_index,
                        reliable_support_count=len(reliable_supports),
                    )
                    canonical_audit = canonical["audit"]
                    canonical_audit["multisession_route_cost"] = incremental_cost
                    canonical_audit["multisession_reliable_support_count"] = len(
                        reliable_supports
                    )
                    canonical_audit["multisession_fused_position_m"] = (
                        observation[:3, 3].tolist()
                    )
                    for support in reliable_supports:
                        support["audit"]["multisession_reliable_support"] = True
                    progress_route_index = max(
                        hypothesis.route_index, route_index
                    )
                    observed_map_from_odom = observation @ np.linalg.inv(odom_pose)
                    trusted_correction_translation = float(np.linalg.norm(
                        observation[:3, 3] - trusted_prediction[:3, 3]
                    ))
                    trusted_correction_yaw = yaw_error_deg(
                        trusted_prediction, observation
                    )
                    trusted_tracking_candidate = bool(
                        trusted_correction_translation
                        <= config.max_tracking_translation_correction_m
                        and trusted_correction_yaw
                        <= config.max_tracking_yaw_correction_deg
                        and abs(progress_route_index - self.trusted_route_index)
                        <= config.max_tracking_route_step
                    )
                    canonical_audit["trusted_tracking_candidate"] = (
                        trusted_tracking_candidate
                    )
                    canonical_audit["trusted_translation_correction_m"] = (
                        trusted_correction_translation
                    )
                    canonical_audit["trusted_yaw_correction_deg"] = (
                        trusted_correction_yaw
                    )
                    if (
                        config.temporal_fusion_enabled
                        and bool(canonical["continuity_candidate"])
                    ):
                        next_map_from_odom = _bounded_se3_update(
                            hypothesis.map_from_odom,
                            observed_map_from_odom,
                            translation_gain=config.tracking_translation_gain,
                            rotation_gain=config.tracking_rotation_gain,
                            max_translation_step_m=(
                                config.tracking_max_translation_step_m
                            ),
                            max_rotation_step_deg=(
                                config.tracking_max_rotation_step_deg
                            ),
                        )
                    else:
                        next_map_from_odom = observed_map_from_odom
                    children.append(LocalizationHypothesis(
                        cost=(
                            config.multisession_history_decay * hypothesis.cost
                            + incremental_cost
                        ),
                        map_from_odom=next_map_from_odom,
                        coast_updates=0,
                        source=f"submap:{canonical['submap']['id']}",
                        route_index=progress_route_index,
                        fitness=float(canonical["fitness"]),
                        rmse_m=float(canonical["rmse_m"]),
                        descriptor_distance=float(
                            canonical["descriptor_distance"]
                        ),
                        trusted_tracking_candidate=trusted_tracking_candidate,
                    ))

        if not children:
            previous = min(self.hypotheses, key=lambda item: item.cost)
            children.append(LocalizationHypothesis(
                cost=previous.cost + 0.20,
                map_from_odom=previous.map_from_odom,
                coast_updates=previous.coast_updates + 1,
                source="forced_coast",
                route_index=previous.route_index,
            ))
        children.sort(key=_hypothesis_selection_key)
        retained: list[LocalizationHypothesis] = []
        for candidate in children:
            pose = candidate.map_from_odom @ odom_pose
            duplicate = any(
                np.linalg.norm(
                    pose[:3, 3] - (other.map_from_odom @ odom_pose)[:3, 3]
                ) < 0.45
                and yaw_error_deg(pose, other.map_from_odom @ odom_pose) < 5.0
                for other in retained
            )
            if duplicate:
                continue
            retained.append(candidate)
            if len(retained) >= config.beam_width:
                break
        self.hypotheses = retained
        self.last_candidate_audits = candidate_audits
        selected = retained[0]
        trusted_prediction = self.trusted_map_from_odom @ odom_pose
        selected_submap = (
            int(selected.source.split(":", 1)[1])
            if selected.source.startswith("submap:") else None
        )
        score_margin = (
            retained[1].cost - retained[0].cost if len(retained) > 1 else None
        )
        mode = "coasting"
        observation_accepted = False
        selected_pose = selected.map_from_odom @ odom_pose
        trusted_correction_translation = float(np.linalg.norm(
            selected_pose[:3, 3] - trusted_prediction[:3, 3]
        ))
        trusted_correction_yaw = yaw_error_deg(
            trusted_prediction, selected_pose
        )

        if selected_submap is not None:
            is_tracking_correction = (
                trusted_correction_translation
                <= config.max_tracking_translation_correction_m
                and trusted_correction_yaw
                <= config.max_tracking_yaw_correction_deg
                and abs(selected.route_index - self.trusted_route_index)
                <= config.max_tracking_route_step
            )
            if is_tracking_correction:
                self.trusted_map_from_odom = selected.map_from_odom.copy()
                self._set_trusted_route_index(
                    selected.route_index, traveled_distance_m
                )
                self.pending_relocalization = None
                self.pending_route_index = None
                self.relocalization_streak = 0
                observation_accepted = True
                mode = "tracking"
            else:
                is_route_consistent_relocalization = bool(
                    selected.route_index >= self.trusted_route_index
                    and abs(
                        selected.route_index - self.last_odometry_route_index
                    ) <= config.recovery_odometry_radius_submaps
                )
                if allow_large_relocalization:
                    translation_limit = (
                        config.max_stationary_relocalization_translation_m
                    )
                    yaw_limit = config.max_stationary_relocalization_yaw_deg
                elif is_route_consistent_relocalization:
                    translation_limit = (
                        config.max_route_consistent_relocalization_translation_m
                    )
                    yaw_limit = (
                        config.max_route_consistent_relocalization_yaw_deg
                    )
                else:
                    translation_limit = (
                        config.max_automatic_relocalization_translation_m
                    )
                    yaw_limit = config.max_automatic_relocalization_yaw_deg
                is_safe_relocalization = (
                    selected.route_index
                    >= self.trusted_route_index - config.route_backtrack_submaps
                    and
                    trusted_correction_translation <= translation_limit
                    and trusted_correction_yaw <= yaw_limit
                )
                if not is_safe_relocalization:
                    self.pending_relocalization = None
                    self.pending_route_index = None
                    self.relocalization_streak = 0
                    mode = "relocalization_rejected"
                elif self.pending_relocalization is None:
                    consistent = False
                else:
                    pending_pose = self.pending_relocalization @ odom_pose
                    consistent = (
                        np.linalg.norm(
                            selected_pose[:3, 3] - pending_pose[:3, 3]
                        )
                        <= config.relocalization_consistency_translation_m
                        and yaw_error_deg(pending_pose, selected_pose)
                        <= config.relocalization_consistency_yaw_deg
                        and self.pending_route_index is not None
                        and abs(selected.route_index - self.pending_route_index) <= 1
                    )
                if is_safe_relocalization:
                    self.relocalization_streak = (
                        self.relocalization_streak + 1 if consistent else 1
                    )
                    self.pending_relocalization = selected.map_from_odom.copy()
                    self.pending_route_index = selected.route_index
                    mode = "relocalizing"
                    if (
                        self.relocalization_streak
                        >= config.relocalization_required_updates
                    ):
                        self.trusted_map_from_odom = selected.map_from_odom.copy()
                        self._set_trusted_route_index(
                            selected.route_index, traveled_distance_m
                        )
                        self.pending_relocalization = None
                        self.pending_route_index = None
                        self.relocalization_streak = 0
                        observation_accepted = True
                        mode = "relocalized"
        else:
            self.pending_relocalization = None
            self.pending_route_index = None
            self.relocalization_streak = 0

        return ContinuousLocalizationResult(
            map_from_body=self.trusted_map_from_odom @ odom_pose,
            map_from_odom=self.trusted_map_from_odom.copy(),
            observation_accepted=observation_accepted,
            selected_submap=selected_submap,
            selected_route_index=self.trusted_route_index,
            hypothesis_count=len(retained),
            score_margin=score_margin,
            correction_translation_m=(
                None if selected_submap is None
                else trusted_correction_translation
            ),
            correction_yaw_deg=(
                None if selected_submap is None else trusted_correction_yaw
            ),
            mode=mode,
            relocalization_streak=self.relocalization_streak,
            fitness=selected.fitness,
            rmse_m=selected.rmse_m,
            descriptor_distance=selected.descriptor_distance,
        )
