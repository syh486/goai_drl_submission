"""Regression check for bounded runtime PointCloud2 decoding."""

from __future__ import annotations

import numpy as np

from deployment.navigation.ros2_node import _read_cloud


def main() -> None:
    from sensor_msgs.msg import PointCloud2, PointField
    from sensor_msgs_py.point_cloud2 import create_cloud
    from std_msgs.msg import Header

    count = 10_003
    points = np.empty(count, dtype=np.dtype([
        ("x", "<f4"),
        ("y", "<f4"),
        ("z", "<f4"),
        ("ring", "<u2"),
        ("timestamp", "<f4"),
    ]))
    points["x"] = np.linspace(-5.0, 5.0, count, dtype=np.float32)
    points["y"] = np.linspace(2.0, 8.0, count, dtype=np.float32)
    points["z"] = np.linspace(-1.0, 1.0, count, dtype=np.float32)
    points["ring"] = np.arange(count, dtype=np.uint16) % 96
    points["timestamp"] = np.linspace(0.0, 0.1, count, dtype=np.float32)
    fields = [
        PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
        PointField(name="ring", offset=12, datatype=PointField.UINT16, count=1),
        PointField(name="timestamp", offset=14, datatype=PointField.FLOAT32, count=1),
    ]
    message: PointCloud2 = create_cloud(Header(frame_id="airy"), fields, points)

    full_points, full_stamps, full_rings = _read_cloud(message)
    sampled_points, sampled_stamps, sampled_rings = _read_cloud(
        message, max_points=4000
    )

    assert len(full_points) == count
    assert len(sampled_points) == 4000
    assert len(full_stamps) == count and len(sampled_stamps) == 4000
    assert full_rings is not None and sampled_rings is not None
    np.testing.assert_allclose(sampled_points[0], full_points[0])
    np.testing.assert_allclose(sampled_points[-1], full_points[-1])

    try:
        _read_cloud(message, max_points=-1)
    except ValueError:
        pass
    else:
        raise AssertionError("negative PointCloud2 sampling limit was accepted")

    print("POINTCLOUD_EARLY_SAMPLING_OK", len(sampled_points), len(full_points))


if __name__ == "__main__":
    main()
