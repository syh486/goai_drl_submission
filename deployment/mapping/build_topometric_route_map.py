"""Build an ordered route-submap map from one independent reference lap.

This representation intentionally does not require the several-hundred-metre
route to form one globally rigid point cloud.  Each anchor stores a short local
scan aggregate, while the ordered anchor index supplies the place prior used by
continuous localization.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from deployment.mapping.mapping_geometry import (
    anchor_frames,
    local_keyframe_cloud,
    scan_context_descriptor,
    voxel_downsample,
)
from deployment.common.trajectory_io import load_aligned_trajectory, prepare_trajectory_poses


@dataclass(frozen=True)
class TopometricMapConfig:
    anchor_spacing_m: float = 2.0
    submap_voxel_m: float = 0.20
    fine_submap_voxel_m: float = 0.08
    submap_aggregate_radius_frames: int = 12
    max_sensor_range_m: float = 25.0
    descriptor_rings: int = 20
    descriptor_sectors: int = 60
    trajectory_pose_mode: str = "lio"


def build_topometric_route_map(
    session_dir: Path,
    trajectory_path: Path,
    output_dir: Path,
    config: TopometricMapConfig = TopometricMapConfig(),
    support_laps: tuple[tuple[Path, Path], ...] = (),
    aligned_support_laps: tuple[tuple[Path, Path, Path, Path], ...] = (),
    anchor_manifest_path: Path | None = None,
    aligned_support_mode: str = "support",
) -> dict[str, object]:
    if aligned_support_mode not in ("support", "fused"):
        raise ValueError("aligned_support_mode must be 'support' or 'fused'")
    session = session_dir.expanduser().resolve()
    aligned = load_aligned_trajectory(session, trajectory_path)
    files = list(aligned.keyframe_files)
    if len(files) < 100:
        raise ValueError(f"reference lap has too few aligned frames: {len(files)}")
    poses, pose_diagnostics = prepare_trajectory_poses(
        aligned, config.trajectory_pose_mode
    )
    destination = output_dir.expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    submap_dir = destination / "submaps"
    submap_dir.mkdir(exist_ok=True)
    for stale in submap_dir.glob("submap_*.npz"):
        stale.unlink()
    for stale in (
        destination / "heldout_tracking_report.json",
        destination / "localization_map_manifest.json",
    ):
        stale.unlink(missing_ok=True)

    increments = np.linalg.norm(np.diff(poses[:, :3, 3], axis=0), axis=1)
    progress_m = np.concatenate((np.zeros(1), np.cumsum(increments)))
    if anchor_manifest_path is None:
        route_anchor_frames = anchor_frames(poses, config.anchor_spacing_m)
    else:
        anchor_manifest = json.loads(
            anchor_manifest_path.expanduser().resolve().read_text(encoding="utf-8")
        )
        canonical_entries = sorted(
            (
                entry for entry in anchor_manifest["submaps"]
                if int(entry["reference_session"]) == 0
            ),
            key=lambda entry: int(entry["route_index"]),
        )
        route_anchor_frames = [
            int(entry["canonical_trajectory_frame"])
            for entry in canonical_entries
        ]
        if not route_anchor_frames:
            raise ValueError("anchor manifest has no canonical route entries")
        if route_anchor_frames[-1] >= len(poses):
            raise ValueError("anchor manifest references a missing trajectory frame")
    entries: list[dict[str, object]] = []

    def write_variant(
        *,
        route_index: int,
        canonical_frame: int,
        source_session: Path,
        source_poses: np.ndarray,
        source_files: list[Path],
        source_keyframe_indices: np.ndarray,
        source_frame: int,
        reference_session: int,
        canonical_from_source_anchor: np.ndarray | None = None,
        tracking_eligible: bool = False,
        alignment_method: str = "canonical",
        fine_points_anchor_override: np.ndarray | None = None,
    ) -> None:
        if fine_points_anchor_override is None:
            fine_local = local_keyframe_cloud(
                source_poses,
                source_frame,
                config,  # compatible subset of LocalizationMapConfig fields
                source_files,
                voxel_m=config.fine_submap_voxel_m,
            )
        else:
            fine_local = np.asarray(
                fine_points_anchor_override, dtype=np.float32
            )
        if (
            canonical_from_source_anchor is not None
            and fine_points_anchor_override is None
        ):
            transform = np.asarray(
                canonical_from_source_anchor, dtype=np.float64
            )
            if transform.shape != (4, 4):
                raise ValueError("support alignment must be a 4x4 transform")
            fine_local = (
                fine_local @ transform[:3, :3].T + transform[:3, 3]
            ).astype(np.float32)
        local = voxel_downsample(
            fine_local, config.submap_voxel_m
        ).astype(np.float32)
        descriptor = scan_context_descriptor(
            local, config.descriptor_rings, config.descriptor_sectors
        ).astype(np.float32)
        suffix = "" if reference_session == 0 else f"_v{reference_session:02d}"
        path = submap_dir / f"submap_{route_index:04d}{suffix}.npz"
        submap_id = len(entries)
        np.savez_compressed(
            path,
            schema_version=np.asarray(4, dtype=np.int64),
            submap_id=np.asarray(submap_id, dtype=np.int64),
            route_index=np.asarray(route_index, dtype=np.int64),
            reference_session=np.asarray(reference_session, dtype=np.int64),
            anchor_frame=np.asarray(
                source_keyframe_indices[source_frame], dtype=np.int64
            ),
            canonical_anchor_frame=np.asarray(
                aligned.keyframe_indices[canonical_frame], dtype=np.int64
            ),
            trajectory_frame=np.asarray(source_frame, dtype=np.int64),
            canonical_trajectory_frame=np.asarray(canonical_frame, dtype=np.int64),
            anchor_pose=poses[canonical_frame],
            canonical_from_source_anchor=(
                np.eye(4, dtype=np.float64)
                if canonical_from_source_anchor is None
                else np.asarray(canonical_from_source_anchor, dtype=np.float64)
            ),
            points_anchor_m=local,
            fine_points_anchor_m=fine_local,
            descriptor=descriptor,
        )
        entries.append({
            "submap_id": submap_id,
            "route_index": route_index,
            "variant_index": reference_session,
            "reference_session": reference_session,
            "tracking_eligible": tracking_eligible,
            "alignment_method": alignment_method,
            "anchor_frame": int(source_keyframe_indices[source_frame]),
            "canonical_anchor_frame": int(aligned.keyframe_indices[canonical_frame]),
            "trajectory_frame": source_frame,
            "canonical_trajectory_frame": canonical_frame,
            "anchor_progress_m": float(progress_m[canonical_frame]),
            "anchor_progress_fraction": float(
                progress_m[canonical_frame] / progress_m[-1]
            ),
            "file": str(path.relative_to(destination)),
            "point_count": len(local),
            "fine_point_count": len(fine_local),
            "anchor_position_m": poses[canonical_frame, :3, 3].tolist(),
            "anchor_quaternion_xyzw": Rotation.from_matrix(
                poses[canonical_frame, :3, :3]
            ).as_quat().tolist(),
        })

    for route_index, frame in enumerate(route_anchor_frames):
        write_variant(
            route_index=route_index,
            canonical_frame=frame,
            source_session=session,
            source_poses=poses,
            source_files=files,
            source_keyframe_indices=aligned.keyframe_indices,
            source_frame=frame,
            reference_session=0,
            tracking_eligible=True,
        )

    reference_sessions = [str(session)]
    reference_trajectories = [str(trajectory_path.expanduser().resolve())]
    trajectory_alignments = [aligned.report()]
    session_frame_counts = [len(files)]
    for reference_session, (support_session_path, support_trajectory_path) in enumerate(
        support_laps, start=1
    ):
        support_session = support_session_path.expanduser().resolve()
        support_aligned = load_aligned_trajectory(
            support_session, support_trajectory_path
        )
        support_files = list(support_aligned.keyframe_files)
        if len(support_files) < 100:
            raise ValueError(
                f"support lap has too few aligned frames: {len(support_files)}"
            )
        support_poses, _ = prepare_trajectory_poses(
            support_aligned, config.trajectory_pose_mode
        )
        support_increments = np.linalg.norm(
            np.diff(support_poses[:, :3, 3], axis=0), axis=1
        )
        support_progress = np.concatenate((
            np.zeros(1), np.cumsum(support_increments)
        ))
        support_fraction = support_progress / support_progress[-1]
        for route_index, canonical_frame in enumerate(route_anchor_frames):
            canonical_fraction = progress_m[canonical_frame] / progress_m[-1]
            support_frame = int(np.argmin(
                np.abs(support_fraction - canonical_fraction)
            ))
            write_variant(
                route_index=route_index,
                canonical_frame=canonical_frame,
                source_session=support_session,
                source_poses=support_poses,
                source_files=support_files,
                source_keyframe_indices=support_aligned.keyframe_indices,
                source_frame=support_frame,
                reference_session=reference_session,
                tracking_eligible=False,
                alignment_method="route_progress_fraction_only",
            )
        reference_sessions.append(str(support_session))
        reference_trajectories.append(
            str(support_trajectory_path.expanduser().resolve())
        )
        trajectory_alignments.append(support_aligned.report())
        session_frame_counts.append(len(support_files))

    next_reference_session = len(reference_sessions)
    for (
        support_session_path,
        support_trajectory_path,
        constraints_path,
        constraint_canonical_trajectory_path,
    ) in aligned_support_laps:
        support_session = support_session_path.expanduser().resolve()
        support_aligned = load_aligned_trajectory(
            support_session, support_trajectory_path
        )
        support_files = list(support_aligned.keyframe_files)
        if len(support_files) < 100:
            raise ValueError(
                f"support lap has too few aligned frames: {len(support_files)}"
            )
        support_poses, _ = prepare_trajectory_poses(
            support_aligned, config.trajectory_pose_mode
        )
        constraint_canonical_aligned = load_aligned_trajectory(
            session, constraint_canonical_trajectory_path
        )
        constraint_canonical_poses, _ = prepare_trajectory_poses(
            constraint_canonical_aligned, config.trajectory_pose_mode
        )
        constraints = json.loads(
            constraints_path.expanduser().resolve().read_text(encoding="utf-8")
        )
        if not isinstance(constraints, list) or not constraints:
            raise ValueError("aligned support constraints must be a non-empty list")
        seen_route_indices: set[int] = set()
        for constraint in constraints:
            route_index = int(constraint["route_index"])
            if route_index in seen_route_indices:
                raise ValueError(
                    f"duplicate aligned support route index: {route_index}"
                )
            if route_index < 0 or route_index >= len(route_anchor_frames):
                raise ValueError(
                    f"aligned support route index is out of range: {route_index}"
                )
            canonical_frame = int(route_anchor_frames[route_index])
            recorded_canonical_frame = int(constraint["canonical_frame"])
            if (
                recorded_canonical_frame < 0
                or recorded_canonical_frame >= len(constraint_canonical_poses)
            ):
                raise ValueError(
                    "aligned support constraint references a missing canonical "
                    f"trajectory frame: {recorded_canonical_frame}"
                )
            support_frame = int(constraint["support_frame"])
            if support_frame < 0 or support_frame >= len(support_files):
                raise ValueError(
                    f"aligned support frame is out of range: {support_frame}"
                )
            support_to_final_anchor = (
                np.linalg.inv(poses[canonical_frame])
                @ constraint_canonical_poses[recorded_canonical_frame]
                @ np.asarray(
                    constraint["canonical_from_support"], dtype=np.float64
                )
            )
            fine_points_override = None
            alignment_method = "bidirectional_cross_lap_constraint_rebased"
            if aligned_support_mode == "fused":
                canonical_fine = local_keyframe_cloud(
                    poses,
                    canonical_frame,
                    config,
                    files,
                    voxel_m=config.fine_submap_voxel_m,
                )
                support_fine = local_keyframe_cloud(
                    support_poses,
                    support_frame,
                    config,
                    support_files,
                    voxel_m=config.fine_submap_voxel_m,
                )
                support_fine = (
                    support_fine @ support_to_final_anchor[:3, :3].T
                    + support_to_final_anchor[:3, 3]
                )
                fine_points_override = voxel_downsample(
                    np.concatenate((canonical_fine, support_fine), axis=0),
                    config.fine_submap_voxel_m,
                ).astype(np.float32)
                alignment_method = "verified_cross_lap_cloud_union"
            write_variant(
                route_index=route_index,
                canonical_frame=canonical_frame,
                source_session=support_session,
                source_poses=support_poses,
                source_files=support_files,
                source_keyframe_indices=support_aligned.keyframe_indices,
                source_frame=support_frame,
                reference_session=next_reference_session,
                canonical_from_source_anchor=support_to_final_anchor,
                tracking_eligible=True,
                alignment_method=alignment_method,
                fine_points_anchor_override=fine_points_override,
            )
            seen_route_indices.add(route_index)
        reference_sessions.append(str(support_session))
        reference_trajectories.append(
            str(support_trajectory_path.expanduser().resolve())
        )
        trajectory_alignments.append(support_aligned.report())
        session_frame_counts.append(len(support_files))
        next_reference_session += 1

    trajectory_path = destination / "canonical_route_poses.npy"
    np.save(trajectory_path, poses)
    timed_trajectory_path = destination / "canonical_route_trajectory.npz"
    np.savez_compressed(
        timed_trajectory_path,
        schema_version=np.asarray(1, dtype=np.int64),
        poses=poses,
        timestamps_s=aligned.timestamps_s,
        keyframe_indices=aligned.keyframe_indices,
    )
    endpoint_offset = np.linalg.inv(poses[0]) @ poses[-1]
    endpoint_rpy_deg = Rotation.from_matrix(
        endpoint_offset[:3, :3]
    ).as_euler("xyz", degrees=True)
    manifest = {
        "schema_version": 3,
        "map_type": "ordered_topometric_route_submaps",
        "coordinate_frame": "reference_lap_odom",
        "purpose": (
            "runtime localization against ordered query-shaped local submaps; "
            "global point-cloud visualization is deliberately not authoritative"
        ),
        "config": asdict(config),
        "reference_sessions": reference_sessions,
        "reference_trajectories": reference_trajectories,
        "trajectory_alignments": trajectory_alignments,
        "canonical_route_poses": str(trajectory_path.resolve()),
        "canonical_route_trajectory": str(timed_trajectory_path.resolve()),
        "reference_frames": len(files),
        "reference_session_frame_counts": session_frame_counts,
        "reference_path_length_m": float(progress_m[-1]),
        "reference_endpoint_offset": {
            "translation_xyz_m": endpoint_offset[:3, 3].tolist(),
            "translation_xy_m": float(np.linalg.norm(endpoint_offset[:2, 3])),
            "translation_norm_m": float(np.linalg.norm(endpoint_offset[:3, 3])),
            "rotation_rpy_deg": endpoint_rpy_deg.tolist(),
            "interpretation": (
                "diagnostic only; the terminal robot heading and elevation may differ, "
                "and no surveyed loop-closure truth is available"
            ),
        },
        "trajectory_pose_diagnostics": pose_diagnostics,
        "route_anchor_count": len(route_anchor_frames),
        "submap_count": len(entries),
        "submaps": entries,
    }
    (destination / "localization_map_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", type=Path, required=True)
    parser.add_argument(
        "--trajectory", type=Path,
        help="timed GLIM traj_lidar.txt or an exactly aligned NPZ/NPY trajectory",
        required=True,
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--anchor-spacing-m", type=float, default=2.0)
    parser.add_argument("--submap-aggregate-radius-frames", type=int, default=12)
    parser.add_argument(
        "--anchor-manifest", type=Path,
        help=(
            "reuse the exact canonical anchor frames from an existing route-map "
            "manifest for controlled map comparisons"
        ),
    )
    parser.add_argument(
        "--support-lap", nargs=2, action="append", default=[],
        metavar=("SESSION", "TRAJECTORY"),
        help="add another completed lap as an appearance variant",
    )
    parser.add_argument(
        "--aligned-support-lap", nargs=4, action="append", default=[],
        metavar=(
            "SESSION",
            "TRAJECTORY",
            "CONSTRAINTS",
            "CONSTRAINT_CANONICAL_TRAJECTORY",
        ),
        help=(
            "add only support submaps with verified canonical_from_support "
            "constraints; the final argument is the canonical trajectory against "
            "which those constraints were measured"
        ),
    )
    parser.add_argument(
        "--aligned-support-mode",
        choices=("support", "fused"),
        default="support",
        help=(
            "store the aligned support cloud alone or its fixed-anchor union "
            "with the canonical cloud"
        ),
    )
    args = parser.parse_args()
    result = build_topometric_route_map(
        args.session,
        args.trajectory,
        args.output_dir,
        TopometricMapConfig(
            anchor_spacing_m=args.anchor_spacing_m,
            submap_aggregate_radius_frames=args.submap_aggregate_radius_frames,
        ),
        support_laps=tuple(
            (Path(session), Path(poses)) for session, poses in args.support_lap
        ),
        aligned_support_laps=tuple(
            (
                Path(session),
                Path(poses),
                Path(constraints),
                Path(constraint_canonical_trajectory),
            )
            for (
                session,
                poses,
                constraints,
                constraint_canonical_trajectory,
            ) in args.aligned_support_lap
        ),
        anchor_manifest_path=args.anchor_manifest,
        aligned_support_mode=args.aligned_support_mode,
    )
    print(json.dumps({
        "output_dir": str(args.output_dir.expanduser().resolve()),
        "submap_count": result["submap_count"],
        "reference_path_length_m": result["reference_path_length_m"],
    }, indent=2))


if __name__ == "__main__":
    main()
