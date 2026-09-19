"""Attach an independently recorded S10 IMU stream to a mapping session."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile

import numpy as np

from deployment.mapping.hardware_imu_recorder import IMU_COLUMNS


def attach_external_imu(session_dir: Path, external_file: Path) -> dict[str, object]:
    session = session_dir.expanduser().resolve()
    source = external_file.expanduser().resolve()
    metadata_path = session / "session.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("state") != "complete":
        raise ValueError("mapping session must be complete before attaching IMU")
    values = np.fromfile(source, dtype=np.float64)
    if not len(values) or len(values) % len(IMU_COLUMNS):
        raise ValueError("external IMU file is empty or has an invalid row size")
    rows = values.reshape(-1, len(IMU_COLUMNS))
    stamps = rows[:, 0]
    if not np.isfinite(rows).all() or np.any(np.diff(stamps) < 0.0):
        raise ValueError("external IMU rows are non-finite or reordered")

    keyframes = sorted((session / "keyframes").glob("*.npz"))
    if not keyframes:
        raise ValueError("mapping session has no keyframes")
    with np.load(keyframes[0], allow_pickle=False) as payload:
        lidar_start_s = float(payload["stamp_s"])
    with np.load(keyframes[-1], allow_pickle=False) as payload:
        lidar_end_s = float(payload["stamp_s"])
    duration_s = float(stamps[-1] - stamps[0])
    gaps = np.diff(stamps)
    report = {
        "samples": len(rows),
        "duration_s": duration_s,
        "rate_hz": ((len(rows) - 1) / duration_s if duration_s > 0.0 else 0.0),
        "max_gap_s": float(gaps.max(initial=0.0)),
        "lidar_start_minus_imu_start_s": float(lidar_start_s - stamps[0]),
        "imu_end_minus_lidar_end_s": float(stamps[-1] - lidar_end_s),
    }
    if report["rate_hz"] < 100.0:
        raise ValueError(f"external IMU rate is too low: {report['rate_hz']:.3f} Hz")
    if report["max_gap_s"] > 0.10:
        raise ValueError(f"external IMU has a {report['max_gap_s']:.3f} s gap")
    if report["lidar_start_minus_imu_start_s"] < -0.10:
        raise ValueError("external IMU does not cover LiDAR start")
    if report["imu_end_minus_lidar_end_s"] < -0.10:
        raise ValueError("external IMU does not cover LiDAR end")

    destination = session / "imu_samples.f64"
    if destination.exists():
        raise FileExistsError(f"mapping session already contains IMU: {destination}")
    os.replace(source, destination)
    metadata["imu_samples"] = len(rows)
    metadata["imu_stream"] = {
        "file": destination.name,
        "dtype": "float64",
        "columns": list(IMU_COLUMNS),
        "capture_process": "isolated_rclpy",
    }
    metadata.setdefault("config", {})["record_imu_stream"] = True
    metadata["external_imu_attachment"] = report
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=session,
        prefix=".session.external_imu.", delete=False,
    ) as stream:
        temporary = Path(stream.name)
        json.dump(metadata, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, metadata_path)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session_dir", type=Path)
    parser.add_argument("external_file", type=Path)
    args = parser.parse_args()
    print(json.dumps(attach_external_imu(args.session_dir, args.external_file), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
