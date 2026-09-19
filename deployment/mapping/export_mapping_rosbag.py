"""Export an S10 mapping capture as a standard ROS 2 PointCloud2/Imu bag."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from deployment.mapping.validate_mapping_recording import validate


POINT_DTYPE = np.dtype([
    ("x", "<f4"),
    ("y", "<f4"),
    ("z", "<f4"),
    ("intensity", "<f4"),
    ("ring", "<u2"),
    ("_padding", "<u2"),
    ("time", "<f4"),
])


def _stamp_parts(stamp_s: float) -> tuple[int, int]:
    seconds = int(np.floor(stamp_s))
    nanoseconds = int(round((stamp_s - seconds) * 1_000_000_000))
    if nanoseconds >= 1_000_000_000:
        seconds += 1
        nanoseconds -= 1_000_000_000
    return seconds, nanoseconds


def _relative_point_times(values: np.ndarray, frame_stamp_s: float) -> np.ndarray:
    """Return finite per-point seconds relative to the cloud header stamp."""

    timestamps = np.asarray(values, dtype=np.float64).reshape(-1)
    if not len(timestamps):
        return np.empty(0, dtype=np.float32)
    if not np.isfinite(timestamps).all():
        raise ValueError("point timestamps contain non-finite values")
    candidates: list[tuple[float, np.ndarray]] = []
    # Airy/Robosense drivers in the field have exposed seconds, milliseconds,
    # microseconds, and nanoseconds. Prefer an absolute clock when it agrees
    # with the PointCloud2 header; otherwise infer a relative scan clock from
    # its span. A normal scan is roughly 0.1 s, which resolves the units.
    for scale in (1.0, 1.0e-3, 1.0e-6, 1.0e-9):
        scaled = timestamps * scale
        header_error = abs(float(np.median(scaled)) - frame_stamp_s)
        span = float(np.ptp(scaled))
        if header_error <= 2.0 and 0.0 < span <= 1.0:
            score = header_error + abs(np.log10(max(span, 1.0e-9) / 0.1)) * 0.01
            candidates.append((score, scaled - frame_stamp_s))
    if candidates:
        relative = min(candidates, key=lambda item: item[0])[1]
    else:
        relative_candidates: list[tuple[float, np.ndarray]] = []
        for scale in (1.0, 1.0e-3, 1.0e-6, 1.0e-9):
            scaled = timestamps * scale
            span = float(np.ptp(scaled))
            if 1.0e-6 < span <= 1.0:
                score = abs(np.log10(span / 0.1))
                relative_candidates.append((score, scaled))
        if not relative_candidates:
            raise ValueError("cannot infer per-point timestamp units")
        relative = min(relative_candidates, key=lambda item: item[0])[1]
        # Normalize relative clocks to scan end. This also makes front/rear
        # scans with independent zero points compatible after concatenation.
        relative = relative - float(np.max(relative))
    if np.ptp(relative) > 1.0 or np.max(np.abs(relative)) > 2.0:
        raise ValueError("point timestamps are inconsistent with the cloud header")
    return np.ascontiguousarray(relative, dtype=np.float32)


def _load_imu_rows(session_dir: Path) -> tuple[np.ndarray, list[str]]:
    metadata = json.loads((session_dir / "session.json").read_text(encoding="utf-8"))
    stream = metadata.get("imu_stream")
    if not stream:
        return np.empty((0, 11), dtype=np.float64), []
    columns = list(stream.get("columns", ()))
    required = [
        "sensor_stamp_s", "receipt_monotonic_s",
        "roll_deg", "pitch_deg", "yaw_deg",
        "acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z",
    ]
    if columns != required:
        raise ValueError(f"unsupported IMU columns: {columns}")
    values = np.fromfile(session_dir / str(stream["file"]), dtype=np.float64)
    if len(values) % len(columns):
        raise ValueError("IMU stream has a truncated row")
    rows = values.reshape(-1, len(columns))
    rows = rows[np.isfinite(rows).all(axis=1)]
    if len(rows) < 10:
        raise ValueError("capture has too few finite IMU samples")
    order = np.argsort(rows[:, 0], kind="stable")
    rows = rows[order]
    keep = np.r_[True, np.diff(rows[:, 0]) > 0.0]
    return np.ascontiguousarray(rows[keep]), columns


def _quaternion_xyzw_from_rotation(rotation: np.ndarray) -> tuple[float, float, float, float]:
    matrix = np.asarray(rotation, dtype=np.float64)
    if matrix.shape != (3, 3):
        raise ValueError("rotation must have shape [3,3]")
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = np.sqrt(trace + 1.0) * 2.0
        values = (
            (matrix[2, 1] - matrix[1, 2]) / scale,
            (matrix[0, 2] - matrix[2, 0]) / scale,
            (matrix[1, 0] - matrix[0, 1]) / scale,
            0.25 * scale,
        )
    else:
        index = int(np.argmax(np.diag(matrix)))
        if index == 0:
            scale = np.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
            values = (
                0.25 * scale,
                (matrix[0, 1] + matrix[1, 0]) / scale,
                (matrix[0, 2] + matrix[2, 0]) / scale,
                (matrix[2, 1] - matrix[1, 2]) / scale,
            )
        elif index == 1:
            scale = np.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
            values = (
                (matrix[0, 1] + matrix[1, 0]) / scale,
                0.25 * scale,
                (matrix[1, 2] + matrix[2, 1]) / scale,
                (matrix[0, 2] - matrix[2, 0]) / scale,
            )
        else:
            scale = np.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
            values = (
                (matrix[0, 2] + matrix[2, 0]) / scale,
                (matrix[1, 2] + matrix[2, 1]) / scale,
                0.25 * scale,
                (matrix[1, 0] - matrix[0, 1]) / scale,
            )
    quaternion = np.asarray(values, dtype=np.float64)
    quaternion /= np.linalg.norm(quaternion)
    return tuple(float(value) for value in quaternion)


def _quaternion_xyzw_from_rpy_deg(rpy_deg: np.ndarray) -> tuple[float, float, float, float]:
    roll, pitch, yaw = np.deg2rad(np.asarray(rpy_deg, dtype=np.float64)) * 0.5
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    return (
        float(sr * cp * cy - cr * sp * sy),
        float(cr * sp * cy + sr * cp * sy),
        float(cr * cp * sy - sr * sp * cy),
        float(cr * cp * cy + sr * sp * sy),
    )


def export_mapping_rosbag(
    session_dir: Path,
    output_dir: Path,
    *,
    max_frames: int | None = None,
) -> dict[str, object]:
    try:
        import rosbag2_py
        from rclpy.serialization import serialize_message
        from nav_msgs.msg import Odometry
        from sensor_msgs.msg import Imu, PointCloud2, PointField
    except ImportError as error:
        raise RuntimeError("ROS 2 Python packages are required for bag export") from error

    session = session_dir.expanduser().resolve()
    files = sorted((session / "keyframes").glob("*.npz"))
    if max_frames is not None:
        if max_frames < 1:
            raise ValueError("max_frames must be positive")
        files = files[:max_frames]
    if not files:
        raise ValueError("mapping capture has no keyframes")

    cloud_stamps = np.empty(len(files), dtype=np.float64)
    odom_positions = np.empty((len(files), 3), dtype=np.float64)
    odom_rotations = np.empty((len(files), 3, 3), dtype=np.float64)
    for index, path in enumerate(files):
        with np.load(path, allow_pickle=False) as payload:
            cloud_stamps[index] = float(payload["stamp_s"])
            odom_positions[index] = np.asarray(payload["position_odom_m"])
            odom_rotations[index] = np.asarray(payload["rotation_odom_body"])
    # Legacy captures could force-save the final already-recorded cloud during
    # shutdown. Drop only trailing duplicates; reordered data remains invalid.
    while len(files) > 1 and cloud_stamps[-1] <= cloud_stamps[-2]:
        if cloud_stamps[-1] < cloud_stamps[-2]:
            raise ValueError("mapping keyframe timestamps are reordered")
        files.pop()
        cloud_stamps = cloud_stamps[:-1]
        odom_positions = odom_positions[:-1]
        odom_rotations = odom_rotations[:-1]
    validation = validate(session, end_frame=len(files))
    if validation["state"] != "complete":
        raise ValueError("mapping capture is not complete")
    imu_rows, _ = _load_imu_rows(session)
    if not len(imu_rows):
        raise ValueError("IMU stream is required for the GLIM LIO mapping bag")
    time_margin_s = 0.5
    imu_rows = imu_rows[
        (imu_rows[:, 0] >= cloud_stamps[0] - time_margin_s)
        & (imu_rows[:, 0] <= cloud_stamps[-1] + time_margin_s)
    ]
    if len(imu_rows) < 10:
        raise ValueError("too few IMU samples overlap the exported point clouds")

    destination = output_dir.expanduser().resolve()
    if destination.exists():
        if any(destination.iterdir()):
            raise FileExistsError(f"ROS bag output is not empty: {destination}")
        destination.rmdir()
    destination.parent.mkdir(parents=True, exist_ok=True)
    writer = rosbag2_py.SequentialWriter()
    writer.open(
        rosbag2_py.StorageOptions(uri=str(destination), storage_id="sqlite3"),
        rosbag2_py.ConverterOptions("", ""),
    )
    for name, message_type in (
        ("/s10/points_body", "sensor_msgs/msg/PointCloud2"),
        ("/s10/imu", "sensor_msgs/msg/Imu"),
        ("/s10/keyframe_odom", "nav_msgs/msg/Odometry"),
    ):
        writer.create_topic(rosbag2_py.TopicMetadata(
            name=name,
            type=message_type,
            serialization_format="cdr",
            offered_qos_profiles="",
        ))

    def write_imu(row: np.ndarray) -> None:
        message = Imu()
        stamp_s = float(row[0])
        message.header.stamp.sec, message.header.stamp.nanosec = _stamp_parts(stamp_s)
        message.header.frame_id = "base_link"
        qx, qy, qz, qw = _quaternion_xyzw_from_rpy_deg(row[2:5])
        message.orientation.x, message.orientation.y = qx, qy
        message.orientation.z, message.orientation.w = qz, qw
        message.linear_acceleration.x = float(row[5])
        message.linear_acceleration.y = float(row[6])
        message.linear_acceleration.z = float(row[7])
        message.angular_velocity.x = float(row[8])
        message.angular_velocity.y = float(row[9])
        message.angular_velocity.z = float(row[10])
        writer.write("/s10/imu", serialize_message(message), int(round(stamp_s * 1.0e9)))

    def write_odom(index: int) -> None:
        message = Odometry()
        stamp_s = float(cloud_stamps[index])
        message.header.stamp.sec, message.header.stamp.nanosec = _stamp_parts(stamp_s)
        message.header.frame_id = "odom"
        message.child_frame_id = "base_link"
        position = odom_positions[index]
        rotation = odom_rotations[index]
        message.pose.pose.position.x = float(position[0])
        message.pose.pose.position.y = float(position[1])
        message.pose.pose.position.z = float(position[2])
        qx, qy, qz, qw = _quaternion_xyzw_from_rotation(rotation)
        message.pose.pose.orientation.x = qx
        message.pose.pose.orientation.y = qy
        message.pose.pose.orientation.z = qz
        message.pose.pose.orientation.w = qw
        if index:
            elapsed = stamp_s - float(cloud_stamps[index - 1])
            if elapsed > 0.0:
                velocity_world = (position - odom_positions[index - 1]) / elapsed
                velocity_body = rotation.T @ velocity_world
                message.twist.twist.linear.x = float(velocity_body[0])
                message.twist.twist.linear.y = float(velocity_body[1])
                message.twist.twist.linear.z = float(velocity_body[2])
                delta = odom_rotations[index - 1].T @ rotation
                angular_body = np.asarray((
                    delta[2, 1] - delta[1, 2],
                    delta[0, 2] - delta[2, 0],
                    delta[1, 0] - delta[0, 1],
                )) / (2.0 * elapsed)
                message.twist.twist.angular.x = float(angular_body[0])
                message.twist.twist.angular.y = float(angular_body[1])
                message.twist.twist.angular.z = float(angular_body[2])
        writer.write(
            "/s10/keyframe_odom", serialize_message(message), int(round(stamp_s * 1.0e9))
        )

    def write_cloud(path: Path) -> tuple[bool, bool]:
        with np.load(path, allow_pickle=False) as payload:
            stamp_s = float(payload["stamp_s"])
            points = np.asarray(payload["points_body_m"], dtype=np.float32)
            raw_times = np.asarray(
                payload["point_timestamps_s"]
                if "point_timestamps_s" in payload.files else np.empty(0)
            )
            raw_rings = np.asarray(
                payload["rings"] if "rings" in payload.files else np.empty(0)
            )
        point_times = _relative_point_times(raw_times, stamp_s)
        has_times = len(point_times) == len(points)
        has_rings = len(raw_rings) == len(points)
        cloud = np.zeros(len(points), dtype=POINT_DTYPE)
        cloud["x"], cloud["y"], cloud["z"] = points.T
        if has_times:
            cloud["time"] = point_times
        if has_rings:
            cloud["ring"] = np.clip(raw_rings, 0, 65535).astype(np.uint16)

        message = PointCloud2()
        message.header.stamp.sec, message.header.stamp.nanosec = _stamp_parts(stamp_s)
        message.header.frame_id = "base_link"
        message.height = 1
        message.width = len(cloud)
        message.fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name="intensity", offset=12, datatype=PointField.FLOAT32, count=1),
            PointField(name="ring", offset=16, datatype=PointField.UINT16, count=1),
            PointField(name="time", offset=20, datatype=PointField.FLOAT32, count=1),
        ]
        message.is_bigendian = False
        message.point_step = POINT_DTYPE.itemsize
        message.row_step = message.point_step * len(cloud)
        message.data = cloud.tobytes()
        message.is_dense = bool(np.isfinite(points).all())
        writer.write(
            "/s10/points_body", serialize_message(message), int(round(stamp_s * 1.0e9))
        )
        return has_times, has_rings

    imu_index = 0
    timestamped_clouds = 0
    ring_clouds = 0
    for cloud_index, (path, cloud_stamp) in enumerate(zip(files, cloud_stamps)):
        while imu_index < len(imu_rows) and imu_rows[imu_index, 0] <= cloud_stamp:
            write_imu(imu_rows[imu_index])
            imu_index += 1
        write_odom(cloud_index)
        has_times, has_rings = write_cloud(path)
        timestamped_clouds += int(has_times)
        ring_clouds += int(has_rings)
        if cloud_index % 250 == 0:
            print(f"exported {cloud_index + 1}/{len(files)} clouds", flush=True)
    while imu_index < len(imu_rows):
        write_imu(imu_rows[imu_index])
        imu_index += 1
    del writer

    report = {
        "schema_version": 1,
        "source_session": str(session),
        "output_bag": str(destination),
        "pointcloud_messages": len(files),
        "odometry_messages": len(files),
        "imu_messages": len(imu_rows),
        "pointclouds_with_point_time": timestamped_clouds,
        "pointclouds_with_ring": ring_clouds,
        "duration_s": float(cloud_stamps[-1] - cloud_stamps[0]),
        "output_bytes": sum(path.stat().st_size for path in destination.rglob("*") if path.is_file()),
    }
    (destination / "s10_export_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--max-frames", type=int)
    args = parser.parse_args()
    report = export_mapping_rosbag(
        args.session_dir, args.output_dir, max_frames=args.max_frames
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
