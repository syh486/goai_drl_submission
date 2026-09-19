"""Validate a simultaneously recorded waypoint and mapping collection bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from deployment.waypoints.rebind import _active_mark_events, _mapping_stamps
from deployment.mapping.validate_mapping_recording import validate as validate_mapping
from deployment.waypoints.validate_route import validate_route


def validate_bundle(session_dir: Path, max_timestamp_offset_s: float = 0.50) -> dict:
    root = session_dir.expanduser().resolve()
    route = root / "route.yaml"
    mapping = root / "mapping"
    route_summary = validate_route(route)
    mapping_summary = validate_mapping(mapping)
    audit = json.loads(route.with_suffix(".quality.json").read_text(encoding="utf-8"))
    active = _active_mark_events(list(audit.get("events", [])))
    stamps = _mapping_stamps(mapping, int(mapping_summary["keyframes"]))
    offsets = []
    outside = []
    for index in range(int(route_summary["waypoint_count"])):
        if index not in active:
            raise ValueError(f"waypoint {index} has no active mark event")
        stamp = float(active[index]["sensor_stamp_s"])
        offset = float(np.min(np.abs(stamps - stamp)))
        offsets.append(offset)
        if stamp < stamps[0] or stamp > stamps[-1] or offset > max_timestamp_offset_s:
            outside.append(index)
    qualified = bool(
        mapping_summary["state"] == "complete"
        and mapping_summary["dropped_keyframes"] == 0
        and mapping_summary["max_keyframe_interval_s"] <= 1.0
        and not outside
        and route_summary["lidar_observations"]["waypoints"]
        == route_summary["waypoint_count"]
    )
    return {
        "schema_version": 1,
        "qualified": qualified,
        "session_dir": str(root),
        "route": route_summary,
        "mapping": mapping_summary,
        "timestamp_binding": {
            "maximum_nearest_frame_offset_s": max(offsets),
            "maximum_allowed_s": max_timestamp_offset_s,
            "invalid_waypoint_indices": outside,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session_dir", type=Path)
    parser.add_argument("--max-timestamp-offset-s", type=float, default=0.50)
    args = parser.parse_args()
    report = validate_bundle(args.session_dir, args.max_timestamp_offset_s)
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["qualified"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
