"""CPU smoke for bounded route-mapping keyframe output."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile

import numpy as np

from deployment.mapping.mapping_recording import MappingKeyframeRecorder, MappingRecordingConfig
from deployment.mapping.qualify_mapping_recording import qualify
from deployment.mapping.validate_mapping_recording import validate


def main() -> None:
    rng = np.random.default_rng(7)
    points = rng.uniform((-8.0, -8.0, -1.0), (8.0, 8.0, 3.0), size=(20000, 3))
    identity = np.eye(3)
    yaw = np.deg2rad(12.0)
    rotated = np.asarray((
        (np.cos(yaw), -np.sin(yaw), 0.0),
        (np.sin(yaw), np.cos(yaw), 0.0),
        (0.0, 0.0, 1.0),
    ))
    with tempfile.TemporaryDirectory() as temporary:
        output = Path(temporary) / "map"
        recorder = MappingKeyframeRecorder(
            output,
            MappingRecordingConfig(max_points=5000, reserve_free_gib=0.25),
        )
        recorder.record_imu(
            sensor_stamp_s=0.9,
            receipt_monotonic_s=10.0,
            rpy_deg=np.asarray((1.0, 2.0, 3.0)),
            acceleration=np.asarray((0.0, 0.0, 9.81)),
            angular_velocity=np.asarray((0.1, 0.2, 0.3)),
        )
        timestamps = np.linspace(0.0, 0.1, len(points))
        rings = np.arange(len(points), dtype=np.int32) % 96
        assert recorder.consider(
            stamp_s=1.0, position_odom_m=np.zeros(3),
            rotation_odom_body=identity, points_body_m=points,
            point_timestamps_s=timestamps, rings=rings,
        )
        assert not recorder.consider(
            stamp_s=1.3, position_odom_m=np.asarray((0.1, 0.0, 0.0)),
            rotation_odom_body=identity, points_body_m=points,
        )
        assert recorder.consider(
            stamp_s=1.6, position_odom_m=np.asarray((0.6, 0.0, 0.0)),
            rotation_odom_body=identity, points_body_m=points,
        )
        assert recorder.consider(
            stamp_s=1.9, position_odom_m=np.asarray((0.6, 0.0, 0.0)),
            rotation_odom_body=rotated, points_body_m=points,
        )
        assert recorder.consider(
            stamp_s=2.2, position_odom_m=np.asarray((0.6, 0.0, 0.0)),
            rotation_odom_body=rotated, points_body_m=points, force=True,
        )
        assert not recorder.consider(
            stamp_s=2.2, position_odom_m=np.asarray((0.6, 0.0, 0.0)),
            rotation_odom_body=rotated, points_body_m=points, force=True,
        )
        recorder.close()

        session = json.loads((output / "session.json").read_text())
        assert session["state"] == "complete"
        assert session["written_keyframes"] == 4
        assert session["imu_samples"] == 1
        assert np.fromfile(output / "imu_samples.f64", dtype=np.float64).shape == (11,)
        files = sorted((output / "keyframes").glob("*.npz"))
        assert len(files) == 4
        for index, path in enumerate(files):
            with np.load(path, allow_pickle=False) as payload:
                assert int(payload["index"]) == index
                assert payload["points_body_m"].shape[1] == 3
                assert len(payload["points_body_m"]) <= 5000
                assert payload["rotation_odom_body"].shape == (3, 3)
                if index == 0:
                    assert len(payload["point_timestamps_s"]) == len(payload["points_body_m"])
                    assert len(payload["rings"]) == len(payload["points_body_m"])
        summary = validate(output)
        assert summary["keyframes"] == 4
        assert summary["local_path_length_m"] == 0.6
        assert summary["dropped_keyframes"] == 0
        assert summary["imu_stream"]["samples"] == 1
        assert np.isclose(summary["imu_stream"]["lidar_start_minus_imu_start_s"], 0.1)
        assert np.isclose(summary["imu_stream"]["imu_end_minus_lidar_end_s"], -1.3)
        assert summary["imu_stream"]["max_gap_s"] == 0.0
        qualification = qualify(output, min_keyframes=1, min_duration_s=0.0, min_imu_hz=0.0)
        assert "imu_does_not_cover_lidar_end" in qualification["failures"]

        dense_output = Path(temporary) / "dense_map"
        dense = MappingKeyframeRecorder(
            dense_output,
            MappingRecordingConfig(
                min_interval_s=0.1,
                record_every_interval=True,
                max_points=5000,
                reserve_free_gib=0.25,
            ),
        )
        assert dense.consider(
            stamp_s=1.0, position_odom_m=np.zeros(3),
            rotation_odom_body=identity, points_body_m=points,
        )
        assert not dense.consider(
            stamp_s=1.05, position_odom_m=np.zeros(3),
            rotation_odom_body=identity, points_body_m=points,
        )
        assert dense.consider(
            stamp_s=1.10, position_odom_m=np.zeros(3),
            rotation_odom_body=identity, points_body_m=points,
        )
        dense.close()
        dense_session = json.loads((dense_output / "session.json").read_text())
        assert dense_session["written_keyframes"] == 2
        assert dense_session["config"]["record_every_interval"] is True
        assert dense_session["config"]["sync_interval_frames"] == 0
    print("mapping recording smoke passed")


if __name__ == "__main__":
    main()
