"""Record the native S10 IMU stream in a process isolated from LiDAR decoding."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile
import time

import numpy as np


IMU_COLUMNS = (
    "sensor_stamp_s", "receipt_monotonic_s",
    "roll_deg", "pitch_deg", "yaw_deg",
    "acc_x", "acc_y", "acc_z",
    "gyro_x", "gyro_y", "gyro_z",
)


def _stamp_seconds(stamp) -> float:
    return float(stamp.sec) + float(stamp.nanosec) * 1.0e-9


class HardwareImuRecorder:
    def __init__(self, node, output: Path, topic: str, flush_samples: int = 200) -> None:
        from drdds.msg import ImuData
        from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy

        self.node = node
        self.output = output.expanduser().resolve()
        self.output.parent.mkdir(parents=True, exist_ok=True)
        if self.output.exists():
            raise FileExistsError(f"refusing to overwrite IMU recording: {self.output}")
        self.stream = self.output.open("xb", buffering=1024 * 1024)
        self.flush_samples = int(flush_samples)
        self.samples = 0
        self.first_stamp_s: float | None = None
        self.last_stamp_s: float | None = None
        self.max_gap_s = 0.0
        qos = QoSProfile(
            depth=400,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.subscription = node.create_subscription(ImuData, topic, self._imu, qos)

    def _imu(self, message) -> None:
        stamp_s = _stamp_seconds(message.header.stamp)
        data = message.data
        row = np.asarray((
            stamp_s,
            time.monotonic(),
            data.roll, data.pitch, data.yaw,
            data.acc_x, data.acc_y, data.acc_z,
            data.omega_x, data.omega_y, data.omega_z,
        ), dtype=np.float64)
        if not np.isfinite(row).all():
            return
        if self.last_stamp_s is not None:
            if stamp_s < self.last_stamp_s:
                self.node.get_logger().error("IMU sensor timestamps moved backwards")
                return
            self.max_gap_s = max(self.max_gap_s, stamp_s - self.last_stamp_s)
        else:
            self.first_stamp_s = stamp_s
        self.last_stamp_s = stamp_s
        self.stream.write(row.tobytes())
        self.samples += 1
        if self.samples % self.flush_samples == 0:
            self.stream.flush()

    def close(self) -> None:
        if not self.stream.closed:
            self.stream.flush()
            os.fsync(self.stream.fileno())
            self.stream.close()
        duration_s = (
            self.last_stamp_s - self.first_stamp_s
            if self.first_stamp_s is not None and self.last_stamp_s is not None else 0.0
        )
        report = {
            "schema_version": 1,
            "file": str(self.output),
            "columns": list(IMU_COLUMNS),
            "samples": self.samples,
            "duration_s": duration_s,
            "rate_hz": ((self.samples - 1) / duration_s if self.samples > 1 and duration_s > 0 else 0.0),
            "first_stamp_s": self.first_stamp_s,
            "last_stamp_s": self.last_stamp_s,
            "max_gap_s": self.max_gap_s,
        }
        report_path = self.output.with_suffix(self.output.suffix + ".json")
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=report_path.parent,
            prefix=f".{report_path.name}.", delete=False,
        ) as stream:
            temporary = Path(stream.name)
            json.dump(report, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, report_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--topic", default="/IMU_DATA")
    args, ros_args = parser.parse_known_args()

    import rclpy
    from rclpy.executors import ExternalShutdownException

    rclpy.init(args=ros_args)
    node = rclpy.create_node("s10_mapping_imu_recorder")
    recorder = HardwareImuRecorder(node, args.output, args.topic)
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        recorder.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
