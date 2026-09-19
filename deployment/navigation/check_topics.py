"""Fail-fast ROS2 topic and Airy field audit before enabling robot motion."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

from deployment.navigation.ros2_node import REPO_ROOT, _stamp_seconds
from deployment.navigation.core import load_hardware_config


class TopicAudit:
    def __init__(self, node, config: dict, *, min_joint_hz: float = 20.0):
        from drdds.msg import ImuData, JointsData
        from rclpy.qos import qos_profile_sensor_data
        from sensor_msgs.msg import PointCloud2

        self.node = node
        self.receipts = {name: [] for name in ("front", "rear", "imu", "joints")}
        self.header_stamps = {name: [] for name in ("front", "rear", "imu")}
        self.stamp_lags = {name: [] for name in ("front", "rear", "imu")}
        self.cloud_fields = {name: set() for name in ("front", "rear")}
        self.accel_norms = []
        self.min_joint_hz = float(min_joint_hz)
        self.require_point_timestamps = bool(
            config.get("runtime", {}).get("require_point_timestamps", True)
        )
        topics = config["topics"]
        node.create_subscription(PointCloud2, topics["front_cloud"], lambda msg: self._cloud("front", msg), qos_profile_sensor_data)
        node.create_subscription(PointCloud2, topics["rear_cloud"], lambda msg: self._cloud("rear", msg), qos_profile_sensor_data)
        node.create_subscription(ImuData, topics["robot_imu"], self._imu, qos_profile_sensor_data)
        node.create_subscription(JointsData, topics["robot_joints"], lambda msg: self._mark("joints"), qos_profile_sensor_data)

    def _mark(self, name: str) -> None:
        self.receipts[name].append(time.monotonic())

    def _cloud(self, name: str, message) -> None:
        self._mark(name)
        stamp_s = _stamp_seconds(message.header.stamp)
        self.header_stamps[name].append(stamp_s)
        self.stamp_lags[name].append(time.time() - stamp_s)
        self.cloud_fields[name] = {field.name for field in message.fields}

    def _imu(self, message) -> None:
        self._mark("imu")
        stamp_s = _stamp_seconds(message.header.stamp)
        self.header_stamps["imu"].append(stamp_s)
        self.stamp_lags["imu"].append(time.time() - stamp_s)
        data = message.data
        self.accel_norms.append(float(np.linalg.norm((data.acc_x, data.acc_y, data.acc_z))))

    @staticmethod
    def _stamp_stats(stamps: list[float], lags: list[float]) -> dict:
        if not stamps:
            return {}
        values = np.asarray(stamps, dtype=np.float64)
        increments = np.diff(values)
        positive = increments[increments > 0.0]
        lag_values = np.asarray(lags, dtype=np.float64)
        return {
            "samples": len(values),
            "unique": int(len(np.unique(values))),
            "nonpositive_increments": int(np.count_nonzero(increments <= 0.0)),
            "positive_increment_median_s": (
                float(np.median(positive)) if len(positive) else None
            ),
            "positive_increment_p95_s": (
                float(np.percentile(positive, 95)) if len(positive) else None
            ),
            "receipt_minus_header_median_s": float(np.median(lag_values)),
            "receipt_minus_header_p95_s": float(np.percentile(lag_values, 95)),
        }

    @staticmethod
    def _rate(receipts: list[float]) -> float:
        return (len(receipts) - 1) / (receipts[-1] - receipts[0]) if len(receipts) >= 2 else 0.0

    def result(self) -> tuple[dict, list[str]]:
        rates = {name: self._rate(value) for name, value in self.receipts.items()}
        failures = []
        warnings = []
        for side in ("front", "rear"):
            fields = self.cloud_fields[side]
            if not {"x", "y", "z"}.issubset(fields):
                failures.append(f"{side}_cloud_missing_XYZ")
            timestamp_present = bool({"timestamp", "time", "t"} & fields)
            if self.require_point_timestamps and not timestamp_present:
                failures.append(f"{side}_cloud_missing_point_timestamps")
            elif not timestamp_present:
                warnings.append(f"{side}_cloud_has_no_deskew_timestamps")
            if "ring" not in fields:
                warnings.append(f"{side}_cloud_ring_inferred_from_elevation")
            if rates[side] < 5.0:
                failures.append(f"{side}_cloud_rate_below_5Hz")
        if rates["imu"] < 100.0:
            failures.append("robot_imu_rate_below_100Hz")
        if rates["joints"] < self.min_joint_hz:
            failures.append(f"joint_rate_below_{self.min_joint_hz:g}Hz")
        if self.accel_norms and not 7.0 <= float(np.median(self.accel_norms)) <= 12.5:
            failures.append("imu_acceleration_units_or_gravity_invalid")
        pair_skews = []
        if self.header_stamps["front"] and self.header_stamps["rear"]:
            rear = np.asarray(self.header_stamps["rear"])
            pair_skews = [float(np.min(np.abs(rear - stamp))) for stamp in self.header_stamps["front"]]
            if np.percentile(pair_skews, 95) > 0.03:
                failures.append("dual_lidar_clock_skew_above_30ms")
        result = {
            "message_counts": {name: len(value) for name, value in self.receipts.items()},
            "rates_hz": rates,
            "cloud_fields": {name: sorted(value) for name, value in self.cloud_fields.items()},
            "dual_lidar_skew_s": {
                "median": float(np.median(pair_skews)) if pair_skews else None,
                "p95": float(np.percentile(pair_skews, 95)) if pair_skews else None,
            },
            "header_stamp_stats": {
                name: self._stamp_stats(self.header_stamps[name], self.stamp_lags[name])
                for name in ("front", "rear", "imu")
            },
            "acceleration_norm_median": float(np.median(self.accel_norms)) if self.accel_norms else None,
            "warnings": warnings,
            "failures": failures,
        }
        return result, failures


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path,
        default=REPO_ROOT / "deployment/config/hardware_navigation.yaml",
    )
    parser.add_argument("--duration", type=float, default=8.0)
    parser.add_argument("--min-joint-hz", type=float, default=20.0)
    args, ros_args = parser.parse_known_args()
    if args.duration <= 1.0:
        raise ValueError("audit duration must exceed one second")
    if args.min_joint_hz < 0.0:
        raise ValueError("minimum joint rate must be non-negative")

    import rclpy

    rclpy.init(args=ros_args)
    node = rclpy.create_node("s10_hardware_topic_audit")
    audit = TopicAudit(
        node,
        load_hardware_config(args.config.expanduser().resolve()),
        min_joint_hz=args.min_joint_hz,
    )
    deadline = time.monotonic() + args.duration
    while rclpy.ok() and time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.1)
    result, failures = audit.result()
    print("S10_HARDWARE_TOPIC_AUDIT", json.dumps(result, ensure_ascii=False), flush=True)
    node.destroy_node()
    rclpy.shutdown()
    if failures:
        raise SystemExit(2)
    print("S10_HARDWARE_TOPICS_OK", flush=True)


if __name__ == "__main__":
    main()
