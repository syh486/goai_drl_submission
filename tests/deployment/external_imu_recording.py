"""Regression test for attaching a process-isolated IMU recording."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile

import numpy as np

from deployment.mapping.attach_external_imu import attach_external_imu
from deployment.mapping.hardware_imu_recorder import IMU_COLUMNS


def main() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        session = root / "mapping"
        keyframes = session / "keyframes"
        keyframes.mkdir(parents=True)
        (session / "session.json").write_text(json.dumps({
            "state": "complete",
            "imu_samples": 0,
            "imu_stream": None,
            "config": {"record_imu_stream": False},
        }), encoding="utf-8")
        np.savez(keyframes / "000000.npz", stamp_s=np.asarray(1.0))
        np.savez(keyframes / "000001.npz", stamp_s=np.asarray(2.0))

        stamps = np.arange(0.5, 2.501, 0.005)
        rows = np.zeros((len(stamps), len(IMU_COLUMNS)), dtype=np.float64)
        rows[:, 0] = stamps
        external = root / "external.f64"
        rows.tofile(external)
        report = attach_external_imu(session, external)

        assert report["rate_hz"] > 199.0
        assert report["max_gap_s"] < 0.006
        assert report["lidar_start_minus_imu_start_s"] == 0.5
        assert report["imu_end_minus_lidar_end_s"] >= 0.49
        assert not external.exists()
        assert (session / "imu_samples.f64").is_file()
        metadata = json.loads((session / "session.json").read_text(encoding="utf-8"))
        assert metadata["imu_samples"] == len(stamps)
        assert metadata["imu_stream"]["capture_process"] == "isolated_rclpy"
        assert metadata["config"]["record_imu_stream"] is True
    print("EXTERNAL_IMU_RECORDING_OK")


if __name__ == "__main__":
    main()
