"""Contract checks for timestamp-aligned mapping trajectories."""

from __future__ import annotations

from pathlib import Path
import tempfile

import numpy as np

from deployment.common.trajectory_io import load_aligned_trajectory, prepare_trajectory_poses


def main() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        keyframes = root / "session" / "keyframes"
        keyframes.mkdir(parents=True)
        for index in range(8):
            np.savez_compressed(
                keyframes / f"{index:06d}.npz",
                stamp_s=np.asarray(100.0 + 0.2 * index),
                rotation_odom_body=np.eye(3),
                points_body_m=np.zeros((4, 3), dtype=np.float32),
            )

        trajectory = np.zeros((5, 8), dtype=np.float64)
        trajectory[:, 0] = 100.0 + 0.2 * np.arange(2, 7)
        trajectory[:, 1] = np.arange(5)
        trajectory[:, 7] = 1.0
        tum_path = root / "traj_lidar.txt"
        np.savetxt(tum_path, trajectory, fmt="%.9f")
        aligned = load_aligned_trajectory(root / "session", tum_path)
        np.testing.assert_array_equal(aligned.keyframe_indices, np.arange(2, 7))
        np.testing.assert_allclose(aligned.poses[:, 0, 3], np.arange(5))
        assert aligned.skipped_keyframes_start == 2
        assert aligned.skipped_keyframes_end == 1
        assert aligned.maximum_timestamp_error_s < 1.0e-8
        prepared, diagnostics = prepare_trajectory_poses(aligned, "lio")
        np.testing.assert_allclose(prepared[0], np.eye(4), atol=1.0e-12)
        np.testing.assert_allclose(prepared[:, 0, 3], np.arange(5))
        assert diagnostics["pose_mode"] == "lio"

        timed_path = root / "trajectory.npz"
        np.savez_compressed(
            timed_path,
            poses=aligned.poses,
            timestamps_s=aligned.timestamps_s,
        )
        timed = load_aligned_trajectory(root / "session", timed_path)
        np.testing.assert_array_equal(timed.keyframe_indices, np.arange(2, 7))

        npy_path = root / "poses.npy"
        np.save(npy_path, aligned.poses)
        try:
            load_aligned_trajectory(root / "session", npy_path)
        except ValueError as error:
            assert "use the GLIM traj_lidar.txt" in str(error)
        else:
            raise AssertionError("timestamp-free partial NPY trajectory was accepted")
    print("TRAJECTORY_IO_OK")


if __name__ == "__main__":
    main()
