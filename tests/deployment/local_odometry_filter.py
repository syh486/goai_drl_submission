"""CPU smoke for high-rate onboard samples and the local ESKF contract."""

from __future__ import annotations

import numpy as np

from deployment.localization.local_odometry import ImuWheelEskf, LocalOdometryConfig


def main() -> None:
    config = LocalOdometryConfig()
    stationary = ImuWheelEskf(np.asarray((0.0, 0.0, -9.81)), config)
    time_s = np.linspace(0.0, 0.2, 41)
    accel = np.tile(np.asarray((0.0, 0.0, 9.81)), (41, 1))
    gyro = np.zeros((41, 3), dtype=np.float64)
    stationary.propagate(time_s, accel, gyro)
    stationary.update_orientation(np.eye(3), config.orientation_measurement_sigma_deg)
    stationary.update_wheel_velocity(0.0, config.wheel_velocity_sigma)
    assert np.linalg.norm(stationary.position) < 1.0e-10
    assert np.linalg.norm(stationary.velocity) < 1.0e-10
    assert np.isfinite(stationary.covariance).all()

    tilt_only = ImuWheelEskf(np.asarray((0.0, 0.0, -9.81)), config)
    tilt_only.rotation = np.asarray(((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)))
    yaw_before = tilt_only.rotation.copy()
    tilt_only.update_orientation(np.eye(3), 0.5, yaw_sigma_deg=180.0)
    assert np.linalg.norm(tilt_only.rotation - yaw_before) < 0.01

    constrained = ImuWheelEskf(np.asarray((0.0, 0.0, -9.81)), config)
    constrained.velocity[:] = (1.0, 0.5, -0.4)
    constrained.update_body_velocity_constraints(0.03, 0.03)
    assert abs(constrained.velocity[1]) < 0.1
    assert abs(constrained.velocity[2]) < 0.1
    constrained.update_zero_velocity(0.01)
    assert np.linalg.norm(constrained.velocity) < 0.1

    robust = ImuWheelEskf(np.asarray((0.0, 0.0, -9.81)), config)
    nis, inflation = robust.update_lidar_pose(
        np.asarray((5.0, 0.0, 0.0)),
        np.eye(3),
        0.1,
        2.0,
        nis_threshold=22.46,
    )
    assert nis > 22.46
    assert inflation > 1.0

    point_coupled = ImuWheelEskf(np.asarray((0.0, 0.0, -9.81)), config)
    source = np.column_stack((
        np.zeros(100),
        np.linspace(-1.0, 1.0, 100),
        np.linspace(-0.5, 0.5, 100),
    ))
    targets = source + np.asarray((0.20, 0.0, 0.0))
    normals = np.tile(np.asarray((1.0, 0.0, 0.0)), (len(source), 1))
    rmse, condition = point_coupled.update_lidar_point_planes(
        source, targets, normals, np.ones(len(source)) * 0.2, 0.08
    )
    assert point_coupled.position[0] > 0.05
    assert np.isclose(rmse, 0.20)
    assert np.isfinite(condition)

    print(
        "LOCAL_ODOMETRY_FILTER_OK",
        {"samples": len(time_s), "covariance_trace": float(np.trace(robust.covariance))},
        flush=True,
    )


if __name__ == "__main__":
    main()
