"""Load mapping trajectories with explicit keyframe timestamp alignment."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


@dataclass(frozen=True)
class AlignedTrajectory:
    poses: np.ndarray
    timestamps_s: np.ndarray
    keyframe_files: tuple[Path, ...]
    keyframe_indices: np.ndarray
    source: Path
    source_format: str
    maximum_timestamp_error_s: float
    skipped_keyframes_start: int
    skipped_keyframes_end: int

    def report(self) -> dict[str, object]:
        return {
            "source": str(self.source),
            "source_format": self.source_format,
            "pose_count": len(self.poses),
            "maximum_timestamp_error_s": self.maximum_timestamp_error_s,
            "first_keyframe_index": int(self.keyframe_indices[0]),
            "last_keyframe_index": int(self.keyframe_indices[-1]),
            "skipped_keyframes_start": self.skipped_keyframes_start,
            "skipped_keyframes_end": self.skipped_keyframes_end,
        }


def _mapping_keyframes(session_dir: Path) -> tuple[tuple[Path, ...], np.ndarray]:
    root = session_dir.expanduser().resolve()
    files = tuple(sorted((root / "keyframes").glob("*.npz")))
    if len(files) < 2:
        raise ValueError(f"mapping session has too few keyframes: {root}")
    stamps = []
    for path in files:
        with np.load(path, allow_pickle=False) as payload:
            stamps.append(float(payload["stamp_s"]))
    timestamps = np.asarray(stamps, dtype=np.float64)
    if not np.isfinite(timestamps).all() or np.any(np.diff(timestamps) <= 0.0):
        raise ValueError("mapping keyframe timestamps must be finite and increasing")
    return files, timestamps


def _validate_poses(poses: np.ndarray) -> np.ndarray:
    result = np.asarray(poses, dtype=np.float64)
    if result.ndim != 3 or result.shape[1:] != (4, 4):
        raise ValueError("trajectory poses must have shape [N,4,4]")
    if len(result) < 2 or not np.isfinite(result).all():
        raise ValueError("trajectory poses must be finite and contain at least two frames")
    if not np.allclose(result[:, 3], (0.0, 0.0, 0.0, 1.0), atol=1.0e-8):
        raise ValueError("trajectory poses contain invalid homogeneous rows")
    return result


def _load_tum(path: Path) -> tuple[np.ndarray, np.ndarray]:
    trajectory = np.loadtxt(path, dtype=np.float64)
    if trajectory.ndim == 1:
        trajectory = trajectory[None]
    if trajectory.ndim != 2 or trajectory.shape[1] != 8:
        raise ValueError("TUM trajectory must contain timestamp xyz qx qy qz qw")
    timestamps = trajectory[:, 0]
    if not np.isfinite(trajectory).all() or np.any(np.diff(timestamps) <= 0.0):
        raise ValueError("TUM trajectory values must be finite and timestamps increasing")
    quaternions = trajectory[:, 4:8]
    norms = np.linalg.norm(quaternions, axis=1)
    if np.any(norms < 1.0e-8):
        raise ValueError("TUM trajectory contains a zero quaternion")
    poses = np.repeat(np.eye(4, dtype=np.float64)[None], len(trajectory), axis=0)
    poses[:, :3, :3] = Rotation.from_quat(quaternions / norms[:, None]).as_matrix()
    poses[:, :3, 3] = trajectory[:, 1:4]
    return _validate_poses(poses), timestamps


def _nearest_keyframe_indices(
    keyframe_timestamps: np.ndarray,
    trajectory_timestamps: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    upper = np.searchsorted(keyframe_timestamps, trajectory_timestamps, side="left")
    upper = np.clip(upper, 0, len(keyframe_timestamps) - 1)
    lower = np.maximum(0, upper - 1)
    upper_error = np.abs(keyframe_timestamps[upper] - trajectory_timestamps)
    lower_error = np.abs(keyframe_timestamps[lower] - trajectory_timestamps)
    use_lower = lower_error <= upper_error
    indices = np.where(use_lower, lower, upper).astype(np.int64)
    errors = np.abs(keyframe_timestamps[indices] - trajectory_timestamps)
    return indices, errors


def load_aligned_trajectory(
    session_dir: Path,
    trajectory_path: Path,
    *,
    maximum_timestamp_error_s: float = 1.0e-3,
) -> AlignedTrajectory:
    """Load a trajectory and align every pose to its recorded LiDAR keyframe.

    GLIM intentionally omits frames used for IMU initialization and may omit a
    final frame while flushing. A TUM trajectory must therefore be aligned by
    sensor timestamp, never by slicing the first ``N`` keyframes.
    """

    if maximum_timestamp_error_s <= 0.0:
        raise ValueError("maximum timestamp error must be positive")
    files, keyframe_timestamps = _mapping_keyframes(session_dir)
    source = trajectory_path.expanduser().resolve()
    if source.suffix.lower() in {".txt", ".tum"}:
        poses, timestamps = _load_tum(source)
        source_format = "tum_xyz_quaternion_xyzw"
    elif source.suffix.lower() == ".npz":
        with np.load(source, allow_pickle=False) as payload:
            poses = _validate_poses(payload["poses"])
            if "timestamps_s" not in payload.files:
                if len(poses) != len(files):
                    raise ValueError(
                        "timestamp-free NPZ trajectory must match every mapping keyframe"
                    )
                timestamps = keyframe_timestamps.copy()
            else:
                timestamps = np.asarray(payload["timestamps_s"], dtype=np.float64)
        source_format = "npz_homogeneous_poses"
    elif source.suffix.lower() == ".npy":
        poses = _validate_poses(np.load(source))
        if len(poses) != len(files):
            raise ValueError(
                f"timestamp-free NPY trajectory has {len(poses)} poses for "
                f"{len(files)} keyframes; use the GLIM traj_lidar.txt or a timed NPZ"
            )
        timestamps = keyframe_timestamps.copy()
        source_format = "npy_homogeneous_poses_exact_length"
    else:
        raise ValueError(f"unsupported trajectory format: {source}")

    timestamps = np.asarray(timestamps, dtype=np.float64)
    if timestamps.shape != (len(poses),) or not np.isfinite(timestamps).all():
        raise ValueError("trajectory timestamps must have shape [N] with finite values")
    if np.any(np.diff(timestamps) <= 0.0):
        raise ValueError("trajectory timestamps must be strictly increasing")
    indices, errors = _nearest_keyframe_indices(keyframe_timestamps, timestamps)
    if np.any(np.diff(indices) <= 0):
        raise ValueError("trajectory timestamps do not map to unique increasing keyframes")
    maximum_error = float(np.max(errors))
    if maximum_error > maximum_timestamp_error_s:
        raise ValueError(
            f"trajectory/keyframe timestamp error {maximum_error:.6f}s exceeds "
            f"{maximum_timestamp_error_s:.6f}s"
        )
    aligned_files = tuple(files[int(index)] for index in indices)
    return AlignedTrajectory(
        poses=poses,
        timestamps_s=timestamps,
        keyframe_files=aligned_files,
        keyframe_indices=indices,
        source=source,
        source_format=source_format,
        maximum_timestamp_error_s=maximum_error,
        skipped_keyframes_start=int(indices[0]),
        skipped_keyframes_end=int(len(files) - 1 - indices[-1]),
    )


def prepare_trajectory_poses(
    aligned: AlignedTrajectory,
    pose_mode: str,
) -> tuple[np.ndarray, dict[str, object]]:
    """Prepare poses according to the frontend that produced the trajectory."""

    raw = aligned.poses
    raw_euler = Rotation.from_matrix(raw[:, :3, :3]).as_euler("xyz", degrees=True)
    if pose_mode != "lio":
        raise ValueError("the active mapping pipeline accepts only GLIM LIO poses")
    origin_inverse = np.linalg.inv(raw[0])
    prepared = np.einsum("ij,njk->nik", origin_inverse, raw)
    prepared_euler = Rotation.from_matrix(
        prepared[:, :3, :3]
    ).as_euler("xyz", degrees=True)
    diagnostics = {
        "pose_mode": pose_mode,
        "raw_roll_pitch_p95_deg": np.percentile(
            np.abs(raw_euler[:, :2]), 95.0, axis=0
        ).tolist(),
        "prepared_roll_pitch_p95_deg": np.percentile(
            np.abs(prepared_euler[:, :2]), 95.0, axis=0
        ).tolist(),
    }
    return _validate_poses(prepared), diagnostics
