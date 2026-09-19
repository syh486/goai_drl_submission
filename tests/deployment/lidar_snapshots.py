"""Contract test for raw front/rear waypoint observations."""

from __future__ import annotations

import hashlib
from pathlib import Path
import tempfile

import numpy as np

from deployment.waypoints.lidar_snapshots import LidarSnapshotFrame, WaypointLidarSnapshotStore
from deployment.localization.start_alignment import save_start_anchor
from deployment.waypoints.validate_route import validate_route
from deployment.waypoints.collection import PoseSample, WaypointCollectionSession


def main() -> None:
    with tempfile.TemporaryDirectory() as directory:
        route = Path(directory) / "route.yaml"
        anchor = route.with_suffix(".anchor.npz")
        angles = np.linspace(0.0, 2.0 * np.pi, 256, endpoint=False)
        anchor_points = np.column_stack((
            3.0 * np.cos(angles),
            2.0 * np.sin(angles),
            np.linspace(-0.2, 1.2, len(angles)),
        ))
        save_start_anchor(
            anchor,
            anchor_points,
            np.asarray((1.0, 0.0, 0.0, 0.0)),
            frame_count=12,
            voxel_size_m=0.12,
        )
        store = WaypointLidarSnapshotStore(route, pairs_per_waypoint=3)
        for pair_index in range(3):
            for side, skew in (("front", 0.0), ("rear", 0.01)):
                count = 120 + pair_index
                store.append(
                    side,
                    LidarSnapshotFrame(
                        points_xyz=np.full((count, 3), pair_index, dtype=np.float64),
                        point_timestamps=np.linspace(10.0, 10.1, count),
                        rings=np.arange(count) % 96,
                        stamp_s=10.0 + pair_index * 0.1 + skew,
                        receipt_s=20.0 + pair_index * 0.1 + skew,
                        frame_id="lidar_link",
                    ),
                )
        pairs = store.prepare(20.25)
        metadata = store.save(0, pairs)
        output = Path(directory) / metadata["file"]
        assert output.is_file()
        assert hashlib.sha256(output.read_bytes()).hexdigest() == metadata["sha256"]
        with np.load(output, allow_pickle=False) as payload:
            assert int(payload["schema_version"]) == 1
            assert int(payload["pair_count"]) == 3
            assert payload["pair_02_front_xyz"].shape == (122, 3)
            assert payload["pair_00_rear_ring"].shape == (120,)
            assert str(payload["pair_00_front_frame_id"]) == "lidar_link"
        assert metadata["ring_available"] is True
        assert metadata["point_timestamps_available"] is True

        xyz_only_route = Path(directory) / "xyz_only.yaml"
        xyz_only_store = WaypointLidarSnapshotStore(
            xyz_only_route, pairs_per_waypoint=1
        )
        for side, skew in (("front", 0.0), ("rear", 0.005)):
            xyz_only_store.append(
                side,
                LidarSnapshotFrame(
                    points_xyz=np.ones((120, 3), dtype=np.float64),
                    point_timestamps=np.empty(0, dtype=np.float64),
                    rings=None,
                    stamp_s=20.0 + skew,
                    receipt_s=30.0 + skew,
                    frame_id="lidar_link",
                ),
            )
        xyz_only_metadata = xyz_only_store.save(
            0, xyz_only_store.prepare(30.1)
        )
        assert xyz_only_metadata["fields"] == ["x", "y", "z"]
        assert xyz_only_metadata["ring_available"] is False
        assert xyz_only_metadata["point_timestamps_available"] is False
        with np.load(Path(directory) / xyz_only_metadata["file"], allow_pickle=False) as payload:
            assert payload["pair_00_front_timestamp"].shape == (0,)
            assert payload["pair_00_front_ring"].shape == (0,)
        session = WaypointCollectionSession(route)
        for offset in range(5):
            session.append(PoseSample(
                stamp_s=30.0 + offset * 0.2,
                receipt_s=30.0 + offset * 0.2,
                position=np.asarray((offset * 0.001, 0.0, 0.425)),
                quaternion_wxyz=np.asarray((1.0, 0.0, 0.0, 0.0)),
                linear_velocity=np.zeros(3),
                angular_velocity=np.zeros(3),
                healthy=True,
                quality_state="GOOD",
                diagnostics={"covariance_trace": 0.1, "support_height_map_m": 0.0},
            ))
        first = session.mark(now_s=30.8)
        session.attach_lidar_observation(first["index"], metadata)
        for offset in range(5):
            session.append(PoseSample(
                stamp_s=32.0 + offset * 0.2,
                receipt_s=32.0 + offset * 0.2,
                position=np.asarray((1.0, 0.0, 0.425)),
                quaternion_wxyz=np.asarray((1.0, 0.0, 0.0, 0.0)),
                linear_velocity=np.zeros(3),
                angular_velocity=np.zeros(3),
                healthy=True,
                quality_state="GOOD",
                diagnostics={"covariance_trace": 0.1, "support_height_map_m": 0.0},
            ))
        session.mark(now_s=32.8)
        summary = validate_route(route)
        assert summary["lidar_observations"] == {"waypoints": 1, "pairs": 3}
        assert summary["warnings"] == ["LiDAR observations cover only 1/2 waypoints"]
        print("LIDAR_SNAPSHOT_OK", metadata, flush=True)


if __name__ == "__main__":
    main()
