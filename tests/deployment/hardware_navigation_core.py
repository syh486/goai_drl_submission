"""CPU smoke for the hardware navigation path without ROS2 publishers."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from deployment.navigation.core import (
    AiryPointCloudAdapter,
    CloudFrame,
    DualCloudSynchronizer,
    ImuWheelBuffer,
    LocalizationQualityGate,
    RouteManager,
    SensorExtrinsic,
    estimate_support_height_map,
    load_hardware_config,
)
from deployment.common.lidar_geometry import build_sensor_frame_directions
from deployment.navigation.ros2_node import _map_corrected_pose, _pool_native_raster


REPO_ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    local_pose = np.eye(4)
    local_pose[:3, 3] = (2.0, -1.0, 0.5)
    map_from_odom = np.eye(4)
    map_from_odom[:3, 3] = (10.0, 3.0, -0.2)
    result = type("MapResult", (), {"map_from_odom": map_from_odom})()
    np.testing.assert_allclose(
        _map_corrected_pose(local_pose, result), map_from_odom @ local_pose
    )
    np.testing.assert_allclose(_map_corrected_pose(local_pose, None), local_pose)

    config = load_hardware_config(REPO_ROOT / "deployment/config/hardware_navigation.yaml")
    localization_config = load_hardware_config(
        REPO_ROOT / "deployment/config/hardware_localization.yaml"
    )
    assert config["topics"]["steer"] == "/s10/navigation/steer"
    assert localization_config["topics"]["steer"] == "/s10/navigation/steer"
    assert localization_config["sensors"] == config["sensors"]
    assert localization_config["localization"] == config["localization"]
    assert localization_config["map_localization"] == config["map_localization"]
    assert localization_config["start_alignment"] == config["start_alignment"]
    for key in ("wheel_radius_m", "wheel_signs", "wheel_indices"):
        assert localization_config["runtime"][key] == config["runtime"][key]
    assert localization_config["runtime"]["publish_legacy_cmd_vel"] is False
    assert localization_config["runtime"]["publish_dds_steer"] is False
    assert config["localization"]["enable_point_coupling"] is False
    assert config["localization"]["point_coupling_effective_points"] == 120.0
    assert config["map_localization"]["icp_threads"] == 2
    route = RouteManager(config["route"])
    assert route.active_index == 1
    assert len(route.waypoints) == 2
    initial_pose = route.initial_pose()
    assert initial_pose.shape == (7,) and np.isfinite(initial_pose).all()

    sequence = RouteManager({
        "waypoints_map_m": [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.5, 0.0],
            [3.0, 0.5, 0.2],
        ],
        "start_waypoint": 0,
        "final_waypoint": 3,
        "reach_xy_m": 0.5,
        "reach_z_m": 0.55,
        "hold_s": 0.0,
    })
    assert sequence.active_index == 1
    assert sequence.update(np.asarray((1.0, 0.0, 0.0)), 1.0)
    assert sequence.active_index == 2 and not sequence.complete
    assert sequence.update(np.asarray((2.0, 0.5, 0.0)), 2.0)
    assert sequence.active_index == 3 and not sequence.complete
    assert sequence.update(np.asarray((3.0, 0.5, 0.2)), 3.0)
    assert sequence.complete

    directions = build_sensor_frame_directions()
    columns = np.arange(0, 900, 5)
    rows = np.repeat(np.arange(96), len(columns))
    selected_directions = directions[:, columns].reshape(-1, 3)
    ranges = 2.0 + 0.2 * np.sin(np.arange(len(rows)) * 0.01)
    points = selected_directions * ranges[:, None]
    timestamps = 100.0 + np.tile(columns / 9000.0, 96)
    frame = CloudFrame(points, timestamps, rows, 100.0, 10.0)
    extrinsic = SensorExtrinsic(np.zeros(3), np.eye(3))
    adapter = AiryPointCloudAdapter(extrinsic)
    adapted = adapter.convert(frame, np.zeros(3), np.eye(3))
    assert adapted.points_body.shape == points.shape
    assert adapted.has_point_timestamps
    assert np.count_nonzero(adapted.distance_native < 9.9) == len(points)
    pooled_distance, pooled_z = _pool_native_raster(
        adapted.distance_native, adapted.world_z_native
    )
    grouped_distance = adapted.distance_native.reshape(96, 90, 10)
    grouped_z = adapted.world_z_native.reshape(96, 90, 10)
    valid = (grouped_distance > 0.05) & (grouped_distance < 9.9)
    safe = np.where(valid, grouped_distance, np.float32(10.0))
    expected_indices = np.argmin(safe, axis=-1)
    expected_distance = np.take_along_axis(
        safe, expected_indices[..., None], axis=-1
    )[..., 0]
    expected_z = np.take_along_axis(
        grouped_z, expected_indices[..., None], axis=-1
    )[..., 0]
    np.testing.assert_array_equal(pooled_distance, expected_distance)
    np.testing.assert_array_equal(pooled_z, expected_z)
    localization_only = adapter.convert(
        frame, np.zeros(3), np.eye(3), build_raster=False
    )
    np.testing.assert_allclose(localization_only.points_body, adapted.points_body)
    assert localization_only.distance_native.shape == (0, 0)
    assert localization_only.world_z_native.shape == (0, 0)

    front_extrinsic = SensorExtrinsic.from_config(config["sensors"]["front"])
    raw_adapter = AiryPointCloudAdapter(front_extrinsic)
    raw_adapted = raw_adapter.convert(frame, np.zeros(3), np.eye(3))
    points_body = (
        front_extrinsic.position_body
        + points @ front_extrinsic.rotation_body_sensor.T
    )
    body_frame = CloudFrame(points_body, timestamps, rows, 100.0, 10.0)
    transformed_adapter = AiryPointCloudAdapter(
        front_extrinsic,
        points_in_body_frame=True,
    )
    transformed_adapted = transformed_adapter.convert(
        body_frame, np.zeros(3), np.eye(3)
    )
    np.testing.assert_allclose(
        transformed_adapted.distance_native,
        raw_adapted.distance_native,
        atol=1.0e-6,
    )
    np.testing.assert_allclose(
        transformed_adapted.points_body,
        raw_adapted.points_body,
        atol=1.0e-9,
    )

    synchronizer = DualCloudSynchronizer(max_skew_s=0.03)
    assert synchronizer.push("front", frame) is None
    paired = synchronizer.push(
        "rear", CloudFrame(points.copy(), timestamps.copy(), rows.copy(), 100.01, 10.01)
    )
    assert paired is not None

    buffer = ImuWheelBuffer()
    buffer.update_wheels(np.ones(4), np.zeros(4))
    for index in range(41):
        stamp = index * 0.005
        buffer.append_imu(
            stamp,
            20.0 + stamp,
            np.asarray((1.0, 0.0, 0.0, 0.0)),
            np.asarray((0.0, 0.0, 9.81)),
            np.zeros(3),
        )
    history = buffer.history(0.0, 0.2)
    assert len(history["time"]) == 41

    delayed = ImuWheelBuffer()
    for index in range(5):
        stamp = index * 0.05
        delayed.append_imu(
            stamp,
            50.0 + (4 - index) * 0.01,
            np.asarray((1.0, 0.0, 0.0, 0.0)),
            np.asarray((0.0, 0.0, 9.81)),
            np.zeros(3),
        )
    delayed_history = delayed.history(0.05, 0.20)
    assert np.allclose(delayed_history["time"], (0.05, 0.10, 0.15, 0.20))

    receipt_history = buffer.history_by_receipt(20.05, 20.20)
    assert np.allclose(receipt_history["time"], np.arange(10, 41) * 0.005)

    gate = LocalizationQualityGate(warmup_frames=3)
    quality = None
    for _ in range(3):
        quality = gate.evaluate(
            pair_skew_s=0.01,
            points_per_lidar=(len(points), len(points)),
            point_timestamps_present=True,
            imu_samples=41,
            imu_span_s=0.2,
            covariance_trace=1.0,
            icp_accepted=True,
        )
    assert quality is not None and quality.healthy

    hardware_gate = LocalizationQualityGate(
        warmup_frames=1, min_imu_samples=3, min_imu_span_s=0.01
    )
    assert hardware_gate.evaluate(
        pair_skew_s=0.001,
        points_per_lidar=(len(points), len(points)),
        point_timestamps_present=True,
        imu_samples=3,
        imu_span_s=0.01,
        covariance_trace=1.0,
        icp_accepted=True,
    ).healthy

    xy = np.stack(np.meshgrid(np.linspace(-0.8, 0.8, 30), np.linspace(-0.8, 0.8, 30)), axis=-1).reshape(-1, 2)
    support_points = np.column_stack((xy, -0.425 + 0.1 * xy[:, 0]))
    support_height, support_count, support_rmse = estimate_support_height_map(
        support_points, np.asarray((2.0, 3.0, 1.425)), np.eye(3)
    )
    assert support_count >= 50
    assert np.isclose(support_height, 1.0, atol=1.0e-6)
    assert support_rmse < 1.0e-8

    print("HARDWARE_NAVIGATION_CORE_OK", {
        "points": len(points),
        "imu_samples": len(history["time"]),
        "pooled_shape": list(pooled_distance.shape),
    })


if __name__ == "__main__":
    main()
