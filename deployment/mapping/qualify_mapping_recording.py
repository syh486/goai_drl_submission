"""Apply deployment-oriented quality gates to an S10 mapping capture."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from deployment.mapping.validate_mapping_recording import validate


def qualify(
    session_dir: Path,
    *,
    min_keyframes: int = 100,
    min_duration_s: float = 20.0,
    min_imu_hz: float = 100.0,
) -> dict[str, object]:
    summary = validate(session_dir)
    keyframes = int(summary["keyframes"])
    duration_s = float(summary["duration_s"])
    imu = summary.get("imu_stream")
    has_imu_samples = bool(imu and int(imu["samples"]) > 0)
    imu_hz = (
        float(imu["samples"]) / float(imu["duration_s"])
        if has_imu_samples and float(imu["duration_s"]) > 0.0 else 0.0
    )
    timestamp_coverage = float(summary["point_timestamp_keyframes"]) / keyframes
    ring_coverage = float(summary["ring_keyframes"]) / keyframes
    mapping_failures = []
    if summary["state"] != "complete":
        mapping_failures.append("capture_not_complete")
    if keyframes < min_keyframes:
        mapping_failures.append("too_few_keyframes")
    if duration_s < min_duration_s:
        mapping_failures.append("capture_too_short")
    if int(summary["dropped_keyframes"]) > max(5, int(keyframes * 0.005)):
        mapping_failures.append("too_many_dropped_keyframes")
    if imu_hz < min_imu_hz:
        mapping_failures.append("imu_stream_too_slow_or_missing")
    if has_imu_samples:
        if float(imu["lidar_start_minus_imu_start_s"]) < -0.10:
            mapping_failures.append("imu_does_not_cover_lidar_start")
        if float(imu["imu_end_minus_lidar_end_s"]) < -0.10:
            mapping_failures.append("imu_does_not_cover_lidar_end")
        if float(imu["max_gap_s"]) > 0.10:
            mapping_failures.append("imu_stream_has_gap_over_100ms")

    capabilities = {
        "fixed_map_mapping_ready": not mapping_failures,
        "imu_lio_ready": not mapping_failures,
        "per_point_deskew_ready": (
            not mapping_failures and timestamp_coverage >= 0.95
        ),
        "ring_aware_frontend_ready": ring_coverage >= 0.95,
    }
    warnings = []
    if timestamp_coverage < 0.95:
        warnings.append(
            "per-point timestamps unavailable; use global-shutter LiDAR mode and do not claim point-level deskew"
        )
    if ring_coverage < 0.95:
        warnings.append("ring IDs unavailable; use geometry-only preprocessing")
    return {
        "schema_version": 1,
        "session": str(session_dir.expanduser().resolve()),
        "qualified": not mapping_failures,
        "failures": mapping_failures,
        "warnings": warnings,
        "capabilities": capabilities,
        "metrics": {
            "keyframes": keyframes,
            "duration_s": duration_s,
            "imu_hz": imu_hz,
            "imu_lidar_start_margin_s": (
                float(imu["lidar_start_minus_imu_start_s"]) if has_imu_samples else None
            ),
            "imu_lidar_end_margin_s": (
                float(imu["imu_end_minus_lidar_end_s"]) if has_imu_samples else None
            ),
            "imu_max_gap_s": float(imu["max_gap_s"]) if has_imu_samples else None,
            "point_timestamp_coverage": timestamp_coverage,
            "ring_coverage": ring_coverage,
            "dropped_keyframes": int(summary["dropped_keyframes"]),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session_dir", type=Path)
    parser.add_argument("--min-keyframes", type=int, default=100)
    parser.add_argument("--min-duration-s", type=float, default=20.0)
    parser.add_argument("--min-imu-hz", type=float, default=100.0)
    parser.add_argument(
        "--require-point-deskew",
        action="store_true",
        help="fail unless at least 95%% of clouds have per-point timestamps",
    )
    args = parser.parse_args()
    report = qualify(
        args.session_dir,
        min_keyframes=args.min_keyframes,
        min_duration_s=args.min_duration_s,
        min_imu_hz=args.min_imu_hz,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    passed = bool(report["qualified"])
    if args.require_point_deskew:
        passed &= bool(report["capabilities"]["per_point_deskew_ready"])
    raise SystemExit(0 if passed else 2)


if __name__ == "__main__":
    main()
