"""Refine one canonical route with verified cross-lap SE(3) constraints."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import subprocess
import tempfile

import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from scipy.spatial.transform import Rotation, Slerp

from deployment.localization.evaluate_metric_anchors import (
    MetricAnchorConfig,
    _aggregate_center_cloud,
    _register_local_anchor,
)
from deployment.localization.continuous_map_localization import _scan_context_alignment
from deployment.mapping.mapping_geometry import (
    rotation_distance_deg,
    scan_context_descriptor,
)
from deployment.common.trajectory_io import load_aligned_trajectory, prepare_trajectory_poses
from deployment.mapping.validate_multilap_refinement import (
    multilap_quality_failures,
    multilap_quality_measurements,
)


@dataclass(frozen=True)
class MultiLapRefinementConfig:
    anchor_stride_submaps: int = 4
    candidate_radius_frames: int = 8
    candidate_step_frames: int = 2
    aggregate_radius_frames: int = 4
    descriptor_candidate_count: int = 4
    candidates_to_verify: int = 3
    min_fitness: float = 0.60
    max_rmse_m: float = 0.20
    max_cycle_translation_m: float = 0.15
    max_cycle_rotation_deg: float = 3.5
    max_constraint_translation_m: float = 3.0
    max_constraint_rotation_deg: float = 35.0
    odometry_translation_sigma_m: float = 0.12
    odometry_rotation_sigma_deg: float = 2.0
    cross_translation_sigma_m: float = 0.06
    cross_rotation_sigma_deg: float = 1.5
    canonical_prior_translation_sigma_m: float = 0.30
    canonical_prior_rotation_sigma_deg: float = 3.0
    endpoint_translation_sigma_m: float = 0.03
    endpoint_rotation_sigma_deg: float = 1.0
    minimum_constraints: int = 20
    max_nfev: int = 400
    optimizer_backend: str = "gtsam"


def _path_progress(poses: np.ndarray) -> np.ndarray:
    return np.concatenate((
        np.zeros(1),
        np.cumsum(np.linalg.norm(np.diff(poses[:, :3, 3], axis=0), axis=1)),
    ))


def _pose_vector(pose: np.ndarray) -> np.ndarray:
    return np.concatenate((
        np.asarray(pose[:3, 3], dtype=np.float64),
        Rotation.from_matrix(pose[:3, :3]).as_rotvec(),
    ))


def _vector_pose(value: np.ndarray) -> np.ndarray:
    pose = np.eye(4, dtype=np.float64)
    pose[:3, 3] = value[:3]
    pose[:3, :3] = Rotation.from_rotvec(value[3:]).as_matrix()
    return pose


def _pose_error(measurement: np.ndarray, first: np.ndarray, second: np.ndarray) -> np.ndarray:
    residual = np.linalg.inv(measurement) @ np.linalg.inv(first) @ second
    return np.concatenate((
        residual[:3, 3],
        Rotation.from_matrix(residual[:3, :3]).as_rotvec(),
    ))


def _load_manifest(map_dir: Path) -> tuple[Path, dict[int, dict[str, object]]]:
    root = map_dir.expanduser().resolve()
    manifest = json.loads(
        (root / "localization_map_manifest.json").read_text(encoding="utf-8")
    )
    entries = {
        int(entry["route_index"]): entry
        for entry in manifest["submaps"]
        if int(entry["reference_session"]) == 0
    }
    if len(entries) < 20:
        raise ValueError("canonical map has too few route anchors")
    return root, entries


def _selected_route_indices(
    entries: dict[int, dict[str, object]], stride: int
) -> list[int]:
    if stride < 1:
        raise ValueError("anchor stride must be positive")
    route_indices = sorted(entries)
    selected = route_indices[::stride]
    if selected[-1] != route_indices[-1]:
        selected.append(route_indices[-1])
    return selected


def _registration_cost(fitness: float, rmse_m: float) -> float:
    return float(rmse_m + 0.30 * (1.0 - fitness))


def extract_cross_lap_constraints(
    map_dir: Path,
    support_session: Path,
    support_trajectory: Path,
    config: MultiLapRefinementConfig,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    root, entries = _load_manifest(map_dir)
    aligned = load_aligned_trajectory(support_session, support_trajectory)
    support_poses, support_pose_diagnostics = prepare_trajectory_poses(aligned, "lio")
    support_files = list(aligned.keyframe_files)
    support_progress = _path_progress(support_poses)
    support_fraction = support_progress / support_progress[-1]
    registration_config = MetricAnchorConfig(
        voxel_m=0.08,
        min_range_m=0.20,
        max_range_m=30.0,
        max_registration_points=12000,
        max_rmse_m=config.max_rmse_m,
        min_overlap=config.min_fitness,
    )
    constraints: list[dict[str, object]] = []
    rejected: list[dict[str, object]] = []
    previous_support_frame = -1
    previous_map_from_support: np.ndarray | None = None
    cloud_cache: dict[int, np.ndarray] = {}
    descriptor_cache: dict[int, np.ndarray] = {}

    def support_cloud(frame: int) -> np.ndarray:
        if frame not in cloud_cache:
            cloud_cache[frame] = _aggregate_center_cloud(
                support_poses,
                support_files,
                frame,
                config.aggregate_radius_frames,
                registration_config,
            )
        return cloud_cache[frame]

    def support_descriptor(frame: int) -> np.ndarray:
        if frame not in descriptor_cache:
            descriptor_cache[frame] = scan_context_descriptor(
                support_cloud(frame), 20, 60
            )
        return descriptor_cache[frame]

    selected_route_indices = _selected_route_indices(
        entries, config.anchor_stride_submaps
    )
    for anchor_number, route_index in enumerate(selected_route_indices, start=1):
        entry = entries[route_index]
        target_fraction = float(entry["anchor_progress_fraction"])
        center = int(np.argmin(np.abs(support_fraction - target_fraction)))
        begin = max(config.aggregate_radius_frames, center - config.candidate_radius_frames)
        end = min(
            len(support_files) - config.aggregate_radius_frames - 1,
            center + config.candidate_radius_frames,
        )
        candidate_frames = list(range(begin, end + 1, config.candidate_step_frames))
        if center not in candidate_frames and begin <= center <= end:
            candidate_frames.append(center)
        candidate_frames = sorted(
            frame for frame in set(candidate_frames)
            if frame > previous_support_frame
        )
        with np.load(root / str(entry["file"]), allow_pickle=False) as payload:
            canonical_points = np.asarray(
                payload.get("fine_points_anchor_m", payload["points_anchor_m"]),
                dtype=np.float64,
            )
            canonical_frame = int(payload["trajectory_frame"])
            canonical_anchor_pose = np.asarray(
                payload["anchor_pose"], dtype=np.float64
            )
        canonical_descriptor = scan_context_descriptor(canonical_points, 20, 60)

        descriptor_ranked = []
        for frame in candidate_frames:
            descriptor_distance, _ = _scan_context_alignment(
                canonical_descriptor, support_descriptor(frame)
            )
            descriptor_ranked.append((descriptor_distance, frame))
        descriptor_ranked.sort(key=lambda item: item[0])
        registration_frames = {
            frame for _, frame in descriptor_ranked[: config.descriptor_candidate_count]
        }
        if center in candidate_frames:
            registration_frames.add(center)

        ranked = []
        candidate_audit = []
        for descriptor_distance, frame in descriptor_ranked:
            if frame not in registration_frames:
                candidate_audit.append({
                    "support_frame": frame,
                    "descriptor_distance": descriptor_distance,
                    "registered": False,
                })
                continue
            points = support_cloud(frame)
            initializations = [("identity", np.eye(4, dtype=np.float64))]
            if previous_map_from_support is not None:
                support_motion = (
                    np.linalg.inv(support_poses[previous_support_frame])
                    @ support_poses[frame]
                )
                propagated_map_from_support = (
                    previous_map_from_support @ support_motion
                )
                initializations.append((
                    "ordered_local_odometry",
                    np.linalg.inv(canonical_anchor_pose)
                    @ propagated_map_from_support,
                ))
            frame_results = []
            for initialization_name, initialization in initializations:
                transform, fitness, rmse = _register_local_anchor(
                    canonical_points,
                    points,
                    initialization,
                    registration_config,
                )
                result = {
                    "initialization": initialization_name,
                    "fitness": fitness,
                    "rmse_m": rmse,
                    "cost": _registration_cost(fitness, rmse),
                }
                frame_results.append(result)
                if fitness >= config.min_fitness and rmse <= config.max_rmse_m:
                    ranked.append((
                        result["cost"],
                        frame,
                        points,
                        transform,
                        fitness,
                        rmse,
                        initialization_name,
                    ))
            candidate_audit.append({
                "support_frame": frame,
                "descriptor_distance": descriptor_distance,
                "registered": True,
                "initialization_results": frame_results,
            })
        ranked.sort(key=lambda item: item[0])

        verified_candidates: list[tuple[float, dict[str, object]]] = []
        for (
            _, frame, support_points, forward, forward_fitness, forward_rmse,
            initialization_name,
        ) in (
            ranked[: config.candidates_to_verify]
        ):
            reverse_transform, reverse_fitness, reverse_rmse = _register_local_anchor(
                support_points,
                canonical_points,
                np.eye(4, dtype=np.float64),
                registration_config,
            )
            cycle = forward @ reverse_transform
            cycle_translation = float(np.linalg.norm(cycle[:3, 3]))
            cycle_rotation = rotation_distance_deg(cycle[:3, :3])
            relative_translation = float(np.linalg.norm(forward[:3, 3]))
            relative_rotation = rotation_distance_deg(forward[:3, :3])
            accepted = bool(
                reverse_fitness >= config.min_fitness
                and reverse_rmse <= config.max_rmse_m
                and cycle_translation <= config.max_cycle_translation_m
                and cycle_rotation <= config.max_cycle_rotation_deg
                and relative_translation <= config.max_constraint_translation_m
                and relative_rotation <= config.max_constraint_rotation_deg
            )
            if accepted:
                bidirectional_cost = (
                    _registration_cost(forward_fitness, forward_rmse)
                    + _registration_cost(reverse_fitness, reverse_rmse)
                    + cycle_translation
                    + 0.01 * cycle_rotation
                )
                candidate = {
                    "route_index": route_index,
                    "anchor_progress_fraction": target_fraction,
                    "canonical_frame": canonical_frame,
                    "support_frame": frame,
                    "support_keyframe_index": int(aligned.keyframe_indices[frame]),
                    "canonical_from_support": forward.tolist(),
                    "forward_fitness": forward_fitness,
                    "forward_rmse_m": forward_rmse,
                    "reverse_fitness": reverse_fitness,
                    "reverse_rmse_m": reverse_rmse,
                    "cycle_translation_m": cycle_translation,
                    "cycle_rotation_deg": cycle_rotation,
                    "relative_translation_m": relative_translation,
                    "relative_rotation_deg": relative_rotation,
                    "candidate_center_frame": center,
                    "candidate_frame_offset": frame - center,
                    "initialization": initialization_name,
                    "bidirectional_cost": bidirectional_cost,
                }
                verified_candidates.append((bidirectional_cost, candidate))
        selected = (
            min(verified_candidates, key=lambda item: item[0])[1]
            if verified_candidates else None
        )
        if selected is None:
            rejected.append({
                "route_index": route_index,
                "anchor_progress_fraction": target_fraction,
                "candidate_center_frame": center,
                "candidate_audit": candidate_audit,
                "reason": "no bidirectionally consistent registration",
            })
            print(
                f"[multilap] {anchor_number}/{len(selected_route_indices)} "
                f"route={route_index} rejected",
                flush=True,
            )
            continue
        previous_support_frame = int(selected["support_frame"])
        previous_map_from_support = (
            canonical_anchor_pose
            @ np.asarray(selected["canonical_from_support"], dtype=np.float64)
        )
        constraints.append(selected)
        print(
            f"[multilap] {anchor_number}/{len(selected_route_indices)} "
            f"route={route_index} support_frame={selected['support_frame']} "
            f"fitness={float(selected['forward_fitness']):.3f} "
            f"rmse={float(selected['forward_rmse_m']):.3f} "
            f"cycle={float(selected['cycle_translation_m']):.3f}m",
            flush=True,
        )

    report = {
        "schema_version": 1,
        "support_session": str(support_session.expanduser().resolve()),
        "support_trajectory": str(support_trajectory.expanduser().resolve()),
        "config": asdict(config),
        "support_trajectory_alignment": aligned.report(),
        "support_pose_diagnostics": support_pose_diagnostics,
        "attempted_constraints": len(constraints) + len(rejected),
        "accepted_constraints": len(constraints),
        "rejected_constraints": len(rejected),
        "accepted_route_indices": [int(item["route_index"]) for item in constraints],
        "rejections": rejected,
    }
    if len(constraints) < config.minimum_constraints:
        raise ValueError(
            f"only {len(constraints)} cross-lap constraints passed; "
            f"at least {config.minimum_constraints} are required"
        )
    return constraints, report


def _factor_statistics(
    factors: list[tuple[str, int, int, np.ndarray]], poses: list[np.ndarray]
) -> dict[str, object]:
    grouped: dict[str, list[tuple[float, float]]] = {}
    for kind, first, second, measurement in factors:
        error = _pose_error(measurement, poses[first], poses[second])
        grouped.setdefault(kind, []).append((
            float(np.linalg.norm(error[:3])),
            float(np.degrees(np.linalg.norm(error[3:]))),
        ))
    result: dict[str, object] = {}
    for kind, values in grouped.items():
        array = np.asarray(values, dtype=np.float64)
        result[kind] = {
            "count": len(array),
            "translation_mean_m": float(np.mean(array[:, 0])),
            "translation_p95_m": float(np.percentile(array[:, 0], 95.0)),
            "translation_max_m": float(np.max(array[:, 0])),
            "rotation_mean_deg": float(np.mean(array[:, 1])),
            "rotation_p95_deg": float(np.percentile(array[:, 1], 95.0)),
            "rotation_max_deg": float(np.max(array[:, 1])),
        }
    return result


def _run_gtsam_optimizer(
    node_initial: list[np.ndarray],
    factors: list[tuple[str, int, int, np.ndarray]],
    factor_sigmas: list[tuple[float, float]],
    max_iterations: int,
) -> tuple[list[np.ndarray], dict[str, object]]:
    executable = (
        Path(__file__).resolve().parents[1]
        / "native"
        / "build"
        / "multilap_pose_graph"
    )
    if not executable.is_file():
        raise RuntimeError(
            "GTSAM multi-lap optimizer is not built; run "
            "deployment/scripts/mapping/build_multilap_optimizer.sh"
        )
    payload = {
        "fixed_node": 0,
        "max_iterations": max_iterations,
        "relative_error_tolerance": 1.0e-7,
        "absolute_error_tolerance": 1.0e-7,
        "nodes": [pose.tolist() for pose in node_initial],
        "factors": [
            {
                "kind": kind,
                "first": first,
                "second": second,
                "measurement": measurement.tolist(),
                "translation_sigma_m": sigmas[0],
                "rotation_sigma_rad": sigmas[1],
                "robust": kind == "cross_lap",
                "huber_width": 1.345,
            }
            for (kind, first, second, measurement), sigmas in zip(
                factors, factor_sigmas
            )
        ],
    }
    with tempfile.TemporaryDirectory(prefix="s10_multilap_gtsam_") as temporary:
        input_path = Path(temporary) / "input.json"
        output_path = Path(temporary) / "output.json"
        input_path.write_text(json.dumps(payload), encoding="utf-8")
        completed = subprocess.run(
            (str(executable), str(input_path), str(output_path)),
            text=True,
            capture_output=True,
            check=False,
        )
        if not output_path.is_file():
            raise RuntimeError(
                "GTSAM multi-lap optimizer produced no output: "
                f"{completed.stderr.strip()}"
            )
        result = json.loads(output_path.read_text(encoding="utf-8"))
        if completed.returncode not in (0, 1):
            raise RuntimeError(
                "GTSAM multi-lap optimizer failed: "
                f"{completed.stderr.strip()}"
            )
    optimized = [np.asarray(pose, dtype=np.float64) for pose in result["poses"]]
    if len(optimized) != len(node_initial):
        raise ValueError("GTSAM optimizer returned the wrong number of poses")
    return optimized, {
        "backend": "gtsam_lm",
        "success": bool(result["success"]),
        "status": 1 if result["success"] else 0,
        "message": "GTSAM Levenberg-Marquardt completed",
        "cost": float(result["final_error"]),
        "initial_linear_cost": float(result["initial_error"]),
        "cost_reduction_fraction": float(
            1.0
            - float(result["final_error"])
            / max(float(result["initial_error"]), 1.0e-12)
        ),
        "optimality": None,
        "function_evaluations": int(result["iterations"]),
    }


def optimize_two_lap_graph(
    canonical_poses: np.ndarray,
    support_poses: np.ndarray,
    constraints: list[dict[str, object]],
    config: MultiLapRefinementConfig,
) -> tuple[np.ndarray, dict[str, object]]:
    canonical_frames = np.asarray(
        [int(item["canonical_frame"]) for item in constraints], dtype=np.int64
    )
    support_frames = np.asarray(
        [int(item["support_frame"]) for item in constraints], dtype=np.int64
    )
    if np.any(np.diff(canonical_frames) <= 0) or np.any(np.diff(support_frames) <= 0):
        raise ValueError("cross-lap constraints must be strictly ordered")
    canonical_nodes = np.asarray(canonical_poses[canonical_frames], dtype=np.float64)
    support_raw_nodes = np.asarray(support_poses[support_frames], dtype=np.float64)
    # Every accepted registration directly observes the support body pose in
    # the canonical anchor frame. Initializing each node from that observation
    # avoids presenting the solver with the support lap's full accumulated
    # drift as a tens-of-metres cross-factor residual.
    support_nodes = np.asarray([
        canonical_nodes[index]
        @ np.asarray(item["canonical_from_support"], dtype=np.float64)
        for index, item in enumerate(constraints)
    ])
    node_initial = [pose.copy() for pose in canonical_nodes]
    node_initial.extend(pose.copy() for pose in support_nodes)
    canonical_count = len(canonical_nodes)

    factors: list[tuple[str, int, int, np.ndarray]] = []
    factor_sigmas: list[tuple[float, float]] = []
    for index in range(canonical_count - 1):
        factors.append((
            "canonical_odometry",
            index,
            index + 1,
            np.linalg.inv(canonical_nodes[index]) @ canonical_nodes[index + 1],
        ))
        factor_sigmas.append((
            config.odometry_translation_sigma_m,
            np.radians(config.odometry_rotation_sigma_deg),
        ))
        factors.append((
            "support_odometry",
            canonical_count + index,
            canonical_count + index + 1,
            np.linalg.inv(support_raw_nodes[index]) @ support_raw_nodes[index + 1],
        ))
        factor_sigmas.append((
            config.odometry_translation_sigma_m,
            np.radians(config.odometry_rotation_sigma_deg),
        ))
    for index in range(1, canonical_count):
        factors.append((
            "canonical_shape_prior",
            0,
            index,
            np.linalg.inv(canonical_nodes[0]) @ canonical_nodes[index],
        ))
        factor_sigmas.append((
            config.canonical_prior_translation_sigma_m,
            np.radians(config.canonical_prior_rotation_sigma_deg),
        ))
    for index, item in enumerate(constraints):
        cycle_scale = max(1.0, float(item["cycle_translation_m"]) / 0.03)
        factors.append((
            "cross_lap",
            index,
            canonical_count + index,
            np.asarray(item["canonical_from_support"], dtype=np.float64),
        ))
        factor_sigmas.append((
            config.cross_translation_sigma_m * cycle_scale,
            np.radians(config.cross_rotation_sigma_deg) * cycle_scale,
        ))
    factors.append((
        "canonical_endpoint",
        0,
        canonical_count - 1,
        np.linalg.inv(canonical_nodes[0]) @ canonical_nodes[-1],
    ))
    factor_sigmas.append((
        config.endpoint_translation_sigma_m,
        np.radians(config.endpoint_rotation_sigma_deg),
    ))

    initial_pose_list = [pose.copy() for pose in node_initial]
    if config.optimizer_backend == "gtsam":
        optimized, solver_report = _run_gtsam_optimizer(
            node_initial, factors, factor_sigmas, config.max_nfev
        )
    elif config.optimizer_backend == "scipy":
        variable_nodes = list(range(1, len(node_initial)))
        variable_slot = {node: slot for slot, node in enumerate(variable_nodes)}
        initial_vector = np.zeros(len(variable_nodes) * 6, dtype=np.float64)

        def unpack(values: np.ndarray) -> list[np.ndarray]:
            poses = [node_initial[0].copy()]
            for slot, node in enumerate(variable_nodes):
                correction = _vector_pose(values[slot * 6 : (slot + 1) * 6])
                poses.append(correction @ node_initial[node])
            return poses

        def residual(values: np.ndarray) -> np.ndarray:
            poses = unpack(values)
            chunks = []
            for factor, sigmas in zip(factors, factor_sigmas):
                _, first, second, measurement = factor
                error = _pose_error(measurement, poses[first], poses[second])
                error[:3] /= sigmas[0]
                error[3:] /= sigmas[1]
                chunks.append(error)
            return np.concatenate(chunks)

        sparsity = lil_matrix(
            (len(factors) * 6, len(variable_nodes) * 6), dtype=np.int8
        )
        for factor_index, (_, first, second, _) in enumerate(factors):
            rows = slice(factor_index * 6, (factor_index + 1) * 6)
            for node in (first, second):
                if node in variable_slot:
                    slot = variable_slot[node]
                    sparsity[rows, slot * 6 : (slot + 1) * 6] = 1
        initial_residual = residual(initial_vector)
        initial_cost = float(0.5 * np.dot(initial_residual, initial_residual))
        solution = least_squares(
            residual,
            initial_vector,
            jac_sparsity=sparsity.tocsr(),
            loss="huber",
            f_scale=1.0,
            x_scale="jac",
            ftol=1.0e-6,
            xtol=1.0e-6,
            gtol=1.0e-6,
            max_nfev=config.max_nfev,
            verbose=0,
        )
        optimized = unpack(solution.x)
        solver_report = {
            "backend": "scipy_sparse_trf",
            "success": bool(solution.success),
            "status": int(solution.status),
            "message": str(solution.message),
            "cost": float(solution.cost),
            "initial_linear_cost": initial_cost,
            "cost_reduction_fraction": float(
                1.0 - float(solution.cost) / max(initial_cost, 1.0e-12)
            ),
            "optimality": float(solution.optimality),
            "function_evaluations": int(solution.nfev),
        }
    else:
        raise ValueError(
            f"unsupported multi-lap optimizer backend: {config.optimizer_backend}"
        )
    optimized_canonical = np.asarray(optimized[:canonical_count])
    corrections = np.einsum(
        "nij,njk->nik", optimized_canonical, np.linalg.inv(canonical_nodes)
    )
    correction_translation = np.linalg.norm(corrections[:, :3, 3], axis=1)
    correction_rotation = np.asarray([
        rotation_distance_deg(pose[:3, :3]) for pose in corrections
    ])
    report = {
        "schema_version": 1,
        **solver_report,
        "canonical_node_count": canonical_count,
        "factor_count": len(factors),
        "factor_statistics_before": _factor_statistics(factors, initial_pose_list),
        "factor_statistics_after": _factor_statistics(factors, optimized),
        "canonical_correction_translation_p95_m": float(
            np.percentile(correction_translation, 95.0)
        ),
        "canonical_correction_translation_max_m": float(
            np.max(correction_translation)
        ),
        "canonical_correction_rotation_p95_deg": float(
            np.percentile(correction_rotation, 95.0)
        ),
        "canonical_correction_rotation_max_deg": float(
            np.max(correction_rotation)
        ),
    }
    return optimized_canonical, report


def interpolate_canonical_corrections(
    canonical_poses: np.ndarray,
    canonical_frames: np.ndarray,
    optimized_nodes: np.ndarray,
) -> np.ndarray:
    initial_nodes = canonical_poses[canonical_frames]
    corrections = np.einsum(
        "nij,njk->nik", optimized_nodes, np.linalg.inv(initial_nodes)
    )
    progress = _path_progress(canonical_poses)
    knot_progress = progress[canonical_frames]
    translation = np.column_stack([
        np.interp(progress, knot_progress, corrections[:, axis, 3])
        for axis in range(3)
    ])
    rotations = Rotation.from_matrix(corrections[:, :3, :3])
    interpolation = Slerp(knot_progress, rotations)(
        np.clip(progress, knot_progress[0], knot_progress[-1])
    ).as_matrix()
    interpolated = np.repeat(np.eye(4)[None], len(canonical_poses), axis=0)
    interpolated[:, :3, :3] = interpolation
    interpolated[:, :3, 3] = translation
    return np.einsum("nij,njk->nik", interpolated, canonical_poses)


def refine_route_multilap(
    map_dir: Path,
    canonical_session: Path,
    canonical_trajectory: Path,
    support_session: Path,
    support_trajectory: Path,
    output_dir: Path,
    config: MultiLapRefinementConfig = MultiLapRefinementConfig(),
) -> dict[str, object]:
    output = output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    constraints_path = output / "cross_lap_constraints.json"
    extraction_path = output / "constraint_extraction_report.json"
    if constraints_path.is_file() and extraction_path.is_file():
        constraints = json.loads(constraints_path.read_text(encoding="utf-8"))
        extraction_report = json.loads(
            extraction_path.read_text(encoding="utf-8")
        )
        print(
            f"[multilap] reusing {len(constraints)} saved constraints from "
            f"{constraints_path}",
            flush=True,
        )
    else:
        if any(output.iterdir()):
            raise ValueError(
                f"incomplete multi-lap output exists without reusable constraints: {output}"
            )
        constraints, extraction_report = extract_cross_lap_constraints(
            map_dir, support_session, support_trajectory, config
        )
        constraints_path.write_text(
            json.dumps(constraints, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        extraction_path.write_text(
            json.dumps(extraction_report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    canonical_aligned = load_aligned_trajectory(
        canonical_session, canonical_trajectory
    )
    canonical_poses, canonical_diagnostics = prepare_trajectory_poses(
        canonical_aligned, "lio"
    )
    support_aligned = load_aligned_trajectory(support_session, support_trajectory)
    support_poses, _ = prepare_trajectory_poses(support_aligned, "lio")
    optimized_nodes, optimization_report = optimize_two_lap_graph(
        canonical_poses, support_poses, constraints, config
    )
    canonical_frames = np.asarray([
        int(item["canonical_frame"]) for item in constraints
    ], dtype=np.int64)
    refined = interpolate_canonical_corrections(
        canonical_poses, canonical_frames, optimized_nodes
    )
    refined_path = output / "refined_route_trajectory.npz"
    np.savez_compressed(
        refined_path,
        schema_version=np.asarray(1, dtype=np.int64),
        poses=refined,
        timestamps_s=canonical_aligned.timestamps_s,
        keyframe_indices=canonical_aligned.keyframe_indices,
    )
    endpoint = np.linalg.inv(refined[0]) @ refined[-1]
    quality_measurements = multilap_quality_measurements(optimization_report)
    quality_failures = multilap_quality_failures(quality_measurements)
    refinement_qualified = bool(
        optimization_report["success"]
        and len(constraints) >= config.minimum_constraints
        and not quality_failures
    )
    report = {
        "schema_version": 1,
        "method": "ordered bidirectional cross-lap registration plus sparse SE3 graph",
        "full_route_metric_accuracy_qualified": False,
        "qualification_note": (
            "The support lap contributes to refinement. Qualification requires a "
            "third lap excluded from both factor generation and optimization."
        ),
        "map_dir": str(map_dir.expanduser().resolve()),
        "canonical_session": str(canonical_session.expanduser().resolve()),
        "canonical_trajectory": str(canonical_trajectory.expanduser().resolve()),
        "support_session": str(support_session.expanduser().resolve()),
        "support_trajectory": str(support_trajectory.expanduser().resolve()),
        "refined_trajectory": str(refined_path),
        "config": asdict(config),
        "canonical_trajectory_alignment": canonical_aligned.report(),
        "canonical_pose_diagnostics": canonical_diagnostics,
        "constraint_extraction": extraction_report,
        "optimization": optimization_report,
        "refinement_qualified": refinement_qualified,
        "refinement_quality_measurements": quality_measurements,
        "refinement_quality_failures": quality_failures,
        "refined_endpoint_translation_m": endpoint[:3, 3].tolist(),
        "refined_endpoint_rotation_rpy_deg": Rotation.from_matrix(
            endpoint[:3, :3]
        ).as_euler("xyz", degrees=True).tolist(),
        "constraints": constraints,
    }
    (output / "multilap_refinement_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map-dir", type=Path, required=True)
    parser.add_argument("--canonical-session", type=Path, required=True)
    parser.add_argument("--canonical-trajectory", type=Path, required=True)
    parser.add_argument("--support-session", type=Path, required=True)
    parser.add_argument("--support-trajectory", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--anchor-stride-submaps", type=int, default=4)
    parser.add_argument("--candidate-radius-frames", type=int, default=8)
    parser.add_argument("--candidate-step-frames", type=int, default=2)
    parser.add_argument("--minimum-constraints", type=int, default=20)
    parser.add_argument("--max-nfev", type=int, default=400)
    parser.add_argument(
        "--optimizer-backend", choices=("gtsam", "scipy"), default="gtsam"
    )
    args = parser.parse_args()
    report = refine_route_multilap(
        args.map_dir,
        args.canonical_session,
        args.canonical_trajectory,
        args.support_session,
        args.support_trajectory,
        args.output_dir,
        MultiLapRefinementConfig(
            anchor_stride_submaps=args.anchor_stride_submaps,
            candidate_radius_frames=args.candidate_radius_frames,
            candidate_step_frames=args.candidate_step_frames,
            minimum_constraints=args.minimum_constraints,
            max_nfev=args.max_nfev,
            optimizer_backend=args.optimizer_backend,
        ),
    )
    print(json.dumps({
        "refined_trajectory": report["refined_trajectory"],
        "accepted_constraints": report["constraint_extraction"]["accepted_constraints"],
        "optimization": report["optimization"],
        "full_route_metric_accuracy_qualified": report[
            "full_route_metric_accuracy_qualified"
        ],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
