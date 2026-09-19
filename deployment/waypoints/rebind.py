"""Rebind collected waypoints to an offline-optimized mapping trajectory."""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation, Slerp
import yaml

from deployment.common.trajectory_io import load_aligned_trajectory


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _atomic_write(path: Path, payload: dict[str, Any], *, json_format: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as stream:
        temporary = Path(stream.name)
        if json_format:
            json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
        else:
            yaml.safe_dump(payload, stream, sort_keys=False, allow_unicode=True)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _mapping_stamps(session_dir: Path, pose_count: int) -> np.ndarray:
    files = sorted((session_dir / "keyframes").glob("*.npz"))
    if len(files) != pose_count:
        raise ValueError(
            f"mapping frame/pose mismatch: {len(files)} keyframes, {pose_count} poses"
        )
    stamps = []
    for path in files:
        with np.load(path, allow_pickle=False) as payload:
            stamps.append(float(payload["stamp_s"]))
    result = np.asarray(stamps, dtype=np.float64)
    if len(result) < 2 or np.any(np.diff(result) <= 0.0):
        raise ValueError("mapping timestamps must be strictly increasing")
    return result


def _interpolate_pose(stamps: np.ndarray, poses: np.ndarray, stamp_s: float) -> tuple[np.ndarray, float]:
    upper = int(np.searchsorted(stamps, stamp_s, side="left"))
    if upper <= 0:
        return poses[0].copy(), abs(float(stamp_s - stamps[0]))
    if upper >= len(stamps):
        return poses[-1].copy(), abs(float(stamp_s - stamps[-1]))
    lower = upper - 1
    span = float(stamps[upper] - stamps[lower])
    alpha = float((stamp_s - stamps[lower]) / span)
    result = np.eye(4, dtype=np.float64)
    result[:3, 3] = poses[lower, :3, 3] * (1.0 - alpha) + poses[upper, :3, 3] * alpha
    result[:3, :3] = Slerp(
        stamps[[lower, upper]], Rotation.from_matrix(poses[[lower, upper], :3, :3])
    )([stamp_s]).as_matrix()[0]
    return result, min(abs(float(stamp_s - stamps[lower])), abs(float(stamps[upper] - stamp_s)))


def _active_mark_events(events: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    active: dict[int, dict[str, Any]] = {}
    for event in events:
        if event.get("event") == "mark":
            active[int(event["index"])] = event
        elif event.get("event") == "undo":
            active.pop(int(event["removed_index"]), None)
    return active


def _rewrite_relative_file(
    metadata: dict[str, Any], source_route: Path, output_route: Path
) -> None:
    value = metadata.get("file")
    if not value:
        return
    source = Path(str(value)).expanduser()
    if not source.is_absolute():
        source = source_route.parent / source
    metadata["file"] = os.path.relpath(source.resolve(), output_route.parent.resolve())


def rebind_waypoints(
    route_path: Path,
    mapping_session: Path,
    optimized_poses_path: Path,
    output_path: Path,
    *,
    max_timestamp_offset_s: float = 0.50,
    topometric_map: Path | None = None,
) -> dict[str, Any]:
    source_route = route_path.expanduser().resolve()
    destination = output_path.expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"rebound route already exists: {destination}")
    route = yaml.safe_load(source_route.read_text(encoding="utf-8"))
    if not isinstance(route, dict) or route.get("coordinate_frame") != "s10_route_map":
        raise ValueError("input is not an s10_route_map route")
    nodes = route.get("nodes")
    if not isinstance(nodes, list) or len(nodes) < 2:
        raise ValueError("input route needs at least two waypoints")

    audit_path = source_route.with_suffix(".quality.json")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    active = _active_mark_events(list(audit.get("events", [])))
    if set(active) != set(range(len(nodes))):
        raise ValueError("quality audit does not contain one active mark for every waypoint")

    mapping_root = mapping_session.expanduser().resolve()
    aligned = load_aligned_trajectory(mapping_root, optimized_poses_path)
    poses = aligned.poses
    stamps = aligned.timestamps_s
    topometric_root = None
    topometric_entries: list[dict[str, Any]] = []
    if topometric_map is not None:
        topometric_root = topometric_map.expanduser().resolve()
        manifest = json.loads(
            (topometric_root / "localization_map_manifest.json").read_text(
                encoding="utf-8"
            )
        )
        references = {
            str(Path(value).expanduser().resolve())
            for value in manifest.get("reference_sessions", ())
        }
        if str(mapping_root) not in references:
            raise ValueError(
                "topometric map was not built from the waypoint mapping session"
            )
        topometric_entries = sorted(
            manifest["submaps"], key=lambda item: int(item.get("route_index", item["submap_id"]))
        )
        if not topometric_entries:
            raise ValueError("topometric map contains no route submaps")

    rebound = deepcopy(route)
    rebound_nodes = rebound["nodes"]
    records = []
    rebound_poses = []
    for index, node in enumerate(rebound_nodes):
        event = active[index]
        sensor_stamp = float(event["sensor_stamp_s"])
        pose, nearest_offset = _interpolate_pose(stamps, poses, sensor_stamp)
        if nearest_offset > max_timestamp_offset_s:
            raise ValueError(
                f"waypoint {index} is {nearest_offset:.3f}s from the nearest mapping frame"
            )
        online_base = np.asarray(event["position_base_median_m"], dtype=np.float64)
        online_terrain = np.asarray(event["position_terrain_m"], dtype=np.float64)
        if online_base.shape != (3,) or online_terrain.shape != (3,):
            raise ValueError(f"waypoint {index} audit positions are invalid")
        terrain = pose[:3, 3].copy()
        terrain[2] += float(online_terrain[2] - online_base[2])
        quaternion_xyzw = Rotation.from_matrix(pose[:3, :3]).as_quat()
        quaternion_wxyz = quaternion_xyzw[[3, 0, 1, 2]]
        original_position = np.asarray(node["position"], dtype=np.float64)
        node["position"] = [round(float(value), 4) for value in terrain]
        node["orientation_wxyz"] = [round(float(value), 7) for value in quaternion_wxyz]
        node["source"] = "offline_optimized_mapping_trajectory"
        node["online_position_original_m"] = original_position.tolist()
        if topometric_root is not None:
            nearest_frame = int(np.argmin(np.abs(stamps - sensor_stamp)))
            entry = min(
                topometric_entries,
                key=lambda item: abs(int(item["anchor_frame"]) - nearest_frame),
            )
            with np.load(topometric_root / entry["file"], allow_pickle=False) as payload:
                anchor_pose = np.asarray(payload["anchor_pose"], dtype=np.float64)
            terrain_anchor = np.linalg.inv(anchor_pose) @ np.append(terrain, 1.0)
            node["topometric_binding"] = {
                "route_index": int(entry.get("route_index", entry["submap_id"])),
                "submap_id": int(entry["submap_id"]),
                "position_anchor_m": [
                    round(float(value), 4) for value in terrain_anchor[:3]
                ],
            }
        if isinstance(node.get("lidar_observation"), dict):
            _rewrite_relative_file(node["lidar_observation"], source_route, destination)
        rebound_poses.append(pose)
        records.append({
            "index": index,
            "name": node.get("name"),
            "sensor_stamp_s": sensor_stamp,
            "nearest_mapping_frame_offset_s": nearest_offset,
            "online_position_m": original_position.tolist(),
            "rebound_position_m": terrain.tolist(),
            "online_to_rebound_distance_m": float(np.linalg.norm(terrain - original_position)),
        })

    first_rotation = rebound_poses[0][:3, :3]
    rebound["initial_map_yaw_deg"] = float(
        np.degrees(np.arctan2(first_rotation[1, 0], first_rotation[0, 0]))
    )
    rebound.setdefault("collection", {}).update({
        "offline_rebound": True,
        "offline_rebound_at": _utc_now(),
        "mapping_session": str(mapping_session.expanduser().resolve()),
        "optimized_trajectory": str(optimized_poses_path.expanduser().resolve()),
        "online_coordinates_retained_for_audit": True,
    })
    if topometric_root is not None:
        rebound["topometric_map"] = os.path.relpath(
            topometric_root, destination.parent.resolve()
        )
    alignment = rebound.get("initial_alignment")
    if isinstance(alignment, dict) and alignment.get("anchor_file"):
        anchor = Path(str(alignment["anchor_file"])).expanduser()
        if not anchor.is_absolute():
            anchor = source_route.parent / anchor
        alignment["anchor_file"] = os.path.relpath(anchor.resolve(), destination.parent.resolve())

    rebound_audit = deepcopy(audit)
    rebound_audit["route_file"] = str(destination)
    rebound_audit["offline_rebinding"] = records
    rebound_audit["source_quality_audit"] = str(audit_path)
    report = {
        "schema_version": 1,
        "qualified": True,
        "source_route": str(source_route),
        "output_route": str(destination),
        "waypoint_count": len(nodes),
        "mapping_time_range_s": [float(stamps[0]), float(stamps[-1])],
        "trajectory_alignment": aligned.report(),
        "max_nearest_frame_offset_s": max(item["nearest_mapping_frame_offset_s"] for item in records),
        "max_online_to_rebound_distance_m": max(item["online_to_rebound_distance_m"] for item in records),
        "topometric_map": str(topometric_root) if topometric_root is not None else None,
        "records": records,
    }
    _atomic_write(destination, rebound, json_format=False)
    _atomic_write(destination.with_suffix(".quality.json"), rebound_audit, json_format=True)
    _atomic_write(destination.with_suffix(".rebind.json"), report, json_format=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--route", type=Path, required=True)
    parser.add_argument("--mapping-session", type=Path, required=True)
    parser.add_argument("--optimized-poses", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-timestamp-offset-s", type=float, default=0.50)
    parser.add_argument(
        "--topometric-map",
        type=Path,
        help="also bind each waypoint to an ordered route submap",
    )
    args = parser.parse_args()
    report = rebind_waypoints(
        args.route,
        args.mapping_session,
        args.optimized_poses,
        args.output,
        max_timestamp_offset_s=args.max_timestamp_offset_s,
        topometric_map=args.topometric_map,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
