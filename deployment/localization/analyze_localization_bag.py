"""Summarize a read-only S10 localization validation rosbag."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path

import numpy as np


def _as_float(values: dict[str, str], key: str) -> float | None:
    value = values.get(key)
    if value is None or value in {"None", "nan"}:
        return None
    return float(value)


def _as_bool(values: dict[str, str], key: str) -> bool | None:
    value = values.get(key)
    if value is None or value == "None":
        return None
    return value.lower() == "true"


def _percentiles(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"median": None, "p95": None, "maximum": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "median": float(np.median(array)),
        "p95": float(np.percentile(array, 95)),
        "maximum": float(np.max(array)),
    }


def _yaw_from_quaternion(message) -> float:
    q = message.pose.pose.orientation
    return math.atan2(
        2.0 * (q.w * q.z + q.x * q.y),
        1.0 - 2.0 * (q.y * q.y + q.z * q.z),
    )


def _circular_mean(values: np.ndarray) -> float:
    return float(math.atan2(np.sin(values).mean(), np.cos(values).mean()))


def analyze_bag(bag_dir: Path) -> dict[str, object]:
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message

    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=str(bag_dir), storage_id="mcap"),
        rosbag2_py.ConverterOptions("", ""),
    )
    topic_types = {
        item.name: get_message(item.type)
        for item in reader.get_all_topics_and_types()
    }
    odometry = []
    diagnostics = []
    while reader.has_next():
        topic, payload, stamp_ns = reader.read_next()
        if topic not in topic_types:
            continue
        message = deserialize_message(payload, topic_types[topic])
        if topic == "/s10/local_odometry":
            position = message.pose.pose.position
            odometry.append((
                stamp_ns * 1.0e-9,
                np.asarray((position.x, position.y, position.z), dtype=np.float64),
                _yaw_from_quaternion(message),
            ))
        elif topic == "/s10/localization/diagnostics":
            decoded = json.loads(message.data)
            diagnostics.append((stamp_ns * 1.0e-9, decoded))

    if len(odometry) < 2 or not diagnostics:
        raise ValueError("bag lacks localization odometry or diagnostics")

    odom_time = np.asarray([item[0] for item in odometry])
    positions = np.stack([item[1] for item in odometry])
    yaws = np.asarray([item[2] for item in odometry])
    intervals = np.diff(odom_time)
    positive = intervals[intervals > 0.0]
    window = min(10, len(positions) // 2)
    start_position = positions[:window].mean(axis=0)
    end_position = positions[-window:].mean(axis=0)
    start_yaw = _circular_mean(yaws[:window])
    end_yaw = _circular_mean(yaws[-window:])
    yaw_delta = math.atan2(
        math.sin(end_yaw - start_yaw), math.cos(end_yaw - start_yaw)
    )

    records = []
    for stamp, diagnostic in diagnostics:
        values = diagnostic.get("values", {})
        records.append({
            "stamp_s": stamp,
            "state": diagnostic.get("state"),
            "values": values,
        })

    metric_keys = (
        "pair_total_seconds",
        "pair_queue_age_seconds",
        "cloud_decode_front_seconds",
        "cloud_decode_rear_seconds",
        "adapter_seconds",
        "odometry_seconds",
        "registration_seconds",
    )
    timings = {
        key: _percentiles([
            value for item in records
            if (value := _as_float(item["values"], key)) is not None
        ])
        for key in metric_keys
    }
    route_indices = [
        int(value) for item in records
        if (value := _as_float(item["values"], "map_route_index")) is not None
    ]
    route_regressions = sum(
        current < previous for previous, current in zip(route_indices, route_indices[1:])
    )
    map_started = [
        item for item in records
        if (_as_float(item["values"], "local_traveled_distance_m") or 0.0) >= 1.0
    ]
    accepted_samples = [
        value for item in map_started
        if (value := _as_bool(item["values"], "map_observation_accepted")) is not None
    ]
    modes = Counter(
        item["values"].get("map_localization_mode", "unavailable")
        for item in map_started
    )
    rejection_reasons = Counter()
    candidate_route_indices = Counter()
    candidate_samples = []
    for item in map_started:
        values = item["values"]
        reasons = item["values"].get("map_best_candidate_rejections", "")
        for reason in reasons.split(","):
            if reason and reason != "accepted":
                rejection_reasons[reason] += 1
        candidate_route = _as_float(values, "map_best_candidate_route_index")
        if candidate_route is not None:
            candidate_route_indices[int(candidate_route)] += 1
        candidate_samples.append({
            "stamp_s": item["stamp_s"],
            "traveled_distance_m": _as_float(
                values, "local_traveled_distance_m"
            ),
            "mode": values.get("map_localization_mode"),
            "trusted_route_index": _as_float(values, "map_route_index"),
            "candidate_route_index": candidate_route,
            "candidate_fitness": _as_float(
                values, "map_best_candidate_fitness"
            ),
            "candidate_rmse_m": _as_float(
                values, "map_best_candidate_rmse_m"
            ),
            "candidate_translation_innovation_m": _as_float(
                values, "map_best_candidate_translation_innovation_m"
            ),
            "candidate_yaw_innovation_deg": _as_float(
                values, "map_best_candidate_yaw_innovation_deg"
            ),
            "candidate_rejections": reasons,
        })

    dropped = [
        int(value) for item in records
        if (value := _as_float(item["values"], "dropped_lidar_pairs")) is not None
    ]
    traveled = [
        value for item in records
        if (value := _as_float(item["values"], "local_traveled_distance_m")) is not None
    ]
    map_updates = [
        int(value) for item in records
        if (value := _as_float(item["values"], "map_localization_updates")) is not None
    ]
    fitness = [
        value for item in map_started
        if (value := _as_float(item["values"], "map_fitness")) is not None
    ]
    rmse = [
        value for item in map_started
        if (value := _as_float(item["values"], "map_rmse_m")) is not None
    ]

    return {
        "schema_version": 1,
        "bag_dir": str(bag_dir),
        "duration_s": float(odom_time[-1] - odom_time[0]),
        "odometry_messages": len(odometry),
        "diagnostic_messages": len(diagnostics),
        "odometry_rate_hz": float(1.0 / np.median(positive)),
        "odometry_mean_rate_hz": float((len(odometry) - 1) / (odom_time[-1] - odom_time[0])),
        "odometry_interval_s": _percentiles(positive.tolist()),
        "closure_translation_xyz_m": (end_position - start_position).tolist(),
        "closure_xy_m": float(np.linalg.norm((end_position - start_position)[:2])),
        "closure_z_m": float(end_position[2] - start_position[2]),
        "closure_yaw_deg": float(math.degrees(yaw_delta)),
        "reported_path_length_m": float(np.linalg.norm(np.diff(positions, axis=0), axis=1).sum()),
        "local_traveled_distance_m": traveled[-1] - traveled[0] if traveled else None,
        "dropped_lidar_pairs_delta": dropped[-1] - dropped[0] if dropped else None,
        "map_localization_updates_delta": map_updates[-1] - map_updates[0] if map_updates else None,
        "map_started_diagnostic_samples": len(map_started),
        "map_observation_accept_fraction": (
            float(np.mean(accepted_samples)) if accepted_samples else None
        ),
        "map_modes": dict(modes),
        "map_rejection_reasons": dict(rejection_reasons),
        "map_candidate_route_indices": {
            str(key): value for key, value in sorted(candidate_route_indices.items())
        },
        "map_candidate_samples": candidate_samples,
        "map_route_index_initial": route_indices[0] if route_indices else None,
        "map_route_index_final": route_indices[-1] if route_indices else None,
        "map_route_index_max": max(route_indices) if route_indices else None,
        "map_route_index_regressions": route_regressions,
        "map_fitness": _percentiles(fitness),
        "map_rmse_m": _percentiles(rmse),
        "timings_s": timings,
        "state_counts": dict(Counter(item["state"] for item in records)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bag_dir", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = analyze_bag(args.bag_dir.expanduser().resolve())
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.expanduser().resolve().write_text(encoded, encoding="utf-8")
    print(encoded, end="")


if __name__ == "__main__":
    main()
