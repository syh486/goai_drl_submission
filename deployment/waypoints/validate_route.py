"""Validate a collected route before enabling autonomous motion."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml

from deployment.localization.start_alignment import load_start_anchor


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_route(path: Path) -> dict:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") not in (1, 2):
        raise ValueError("route must use schema_version 1 or 2")
    if payload.get("coordinate_frame") != "s10_route_map":
        raise ValueError("route coordinate_frame must be s10_route_map")
    nodes = payload.get("nodes")
    if not isinstance(nodes, list) or len(nodes) < 2:
        raise ValueError("route needs at least two waypoints")
    names = [str(node.get("name", "")) for node in nodes]
    if any(not name for name in names) or len(set(names)) != len(names):
        raise ValueError("waypoint names must be non-empty and unique")
    positions = np.asarray([node.get("position") for node in nodes], dtype=np.float64)
    if positions.shape != (len(nodes), 3) or not np.isfinite(positions).all():
        raise ValueError("every waypoint position must contain three finite values")
    spacing = np.linalg.norm(np.diff(positions[:, :2], axis=0), axis=1)
    distances = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    warnings = []
    anchor_summary = None
    observation_count = 0
    observation_pairs = 0
    topometric_summary = None
    if payload.get("schema_version") == 2:
        alignment = payload.get("initial_alignment")
        if not isinstance(alignment, dict) or not alignment.get("anchor_file"):
            raise ValueError("schema_version 2 route has no initial LiDAR anchor")
        anchor_path = Path(str(alignment["anchor_file"])).expanduser()
        if not anchor_path.is_absolute():
            anchor_path = path.parent / anchor_path
        if not anchor_path.is_file():
            raise ValueError(f"route start anchor is missing: {anchor_path}")
        expected_sha256 = alignment.get("anchor_sha256")
        if not expected_sha256:
            raise ValueError("schema_version 2 route has no start anchor checksum")
        digest = _sha256(anchor_path)
        if digest != str(expected_sha256):
            raise ValueError(f"route start anchor checksum mismatch: {anchor_path}")
        anchor = load_start_anchor(anchor_path)
        anchor_summary = {
            "file": str(anchor_path.resolve()),
            "points": len(anchor.points_reference_body_m),
            "frames": anchor.frame_count,
            "voxel_size_m": anchor.voxel_size_m,
        }
    for index, node in enumerate(nodes):
        observation = node.get("lidar_observation")
        if observation is None:
            continue
        if not isinstance(observation, dict) or not observation.get("file"):
            raise ValueError(f"waypoint {index} has invalid LiDAR observation metadata")
        observation_path = Path(str(observation["file"])).expanduser()
        if not observation_path.is_absolute():
            observation_path = path.parent / observation_path
        if not observation_path.is_file():
            raise ValueError(f"waypoint {index} LiDAR observation is missing: {observation_path}")
        if _sha256(observation_path) != str(observation.get("sha256", "")):
            raise ValueError(f"waypoint {index} LiDAR observation checksum mismatch")
        with np.load(observation_path, allow_pickle=False) as snapshot:
            if int(snapshot["schema_version"]) != 1:
                raise ValueError(f"waypoint {index} has unsupported LiDAR snapshot schema")
            pair_count = int(snapshot["pair_count"])
            if pair_count < 1 or int(snapshot["waypoint_index"]) != index:
                raise ValueError(f"waypoint {index} LiDAR snapshot index/pair count is invalid")
            for pair_index in range(pair_count):
                for side in ("front", "rear"):
                    prefix = f"pair_{pair_index:02d}_{side}"
                    points = np.asarray(snapshot[f"{prefix}_xyz"])
                    if points.ndim != 2 or points.shape[1] != 3 or not len(points):
                        raise ValueError(f"waypoint {index} {prefix} points are invalid")
                    timestamps = np.asarray(snapshot[f"{prefix}_timestamp"])
                    rings = np.asarray(snapshot[f"{prefix}_ring"])
                    if timestamps.shape not in {(0,), (len(points),)} or not np.isfinite(timestamps).all():
                        raise ValueError(f"waypoint {index} {prefix} timestamps are invalid")
                    if rings.shape not in {(0,), (len(points),)}:
                        raise ValueError(f"waypoint {index} {prefix} rings are invalid")
        observation_count += 1
        observation_pairs += pair_count
    if observation_count and observation_count != len(nodes):
        warnings.append(
            f"LiDAR observations cover only {observation_count}/{len(nodes)} waypoints"
        )
    bindings = [node.get("topometric_binding") for node in nodes]
    if any(binding is not None for binding in bindings):
        if not all(isinstance(binding, dict) for binding in bindings):
            raise ValueError("topometric bindings must cover every waypoint")
        map_value = payload.get("topometric_map")
        if not map_value:
            raise ValueError("bound route does not name its topometric map")
        map_path = Path(str(map_value)).expanduser()
        if not map_path.is_absolute():
            map_path = path.parent / map_path
        manifest_path = map_path / "localization_map_manifest.json"
        if not manifest_path.is_file():
            raise ValueError(f"topometric map manifest is missing: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("map_type") != "ordered_topometric_route_submaps":
            raise ValueError("route references an unsupported localization map type")
        available = {
            (int(entry.get("route_index", entry["submap_id"])), int(entry["submap_id"]))
            for entry in manifest.get("submaps", ())
        }
        route_indices = []
        for index, binding in enumerate(bindings):
            key = (int(binding["route_index"]), int(binding["submap_id"]))
            if key not in available:
                raise ValueError(f"waypoint {index} references an unknown route submap")
            local_position = np.asarray(binding.get("position_anchor_m"), dtype=np.float64)
            if local_position.shape != (3,) or not np.isfinite(local_position).all():
                raise ValueError(f"waypoint {index} has an invalid anchor-local position")
            route_indices.append(key[0])
        if np.any(np.diff(route_indices) < 0):
            raise ValueError("waypoint topometric route indices must be nondecreasing")
        topometric_summary = {
            "map": str(map_path.resolve()),
            "submaps": int(manifest["submap_count"]),
            "first_waypoint_route_index": route_indices[0],
            "last_waypoint_route_index": route_indices[-1],
        }
    if np.linalg.norm(positions[0, :2]) > 0.25:
        warnings.append("first waypoint is not near route_map origin")
    close = np.flatnonzero(spacing < 0.25)
    if len(close):
        warnings.append(f"close XY waypoint pairs: {(close + 1).tolist()}")
    long = np.flatnonzero(distances > 8.0)
    if len(long):
        warnings.append(f"long 3D waypoint gaps: {(long + 1).tolist()}")
    return {
        "route": str(path),
        "waypoint_count": len(nodes),
        "total_length_m": float(np.sum(distances)),
        "min_spacing_m": float(np.min(distances)),
        "max_spacing_m": float(np.max(distances)),
        "height_range_m": [float(np.min(positions[:, 2])), float(np.max(positions[:, 2]))],
        "start_anchor": anchor_summary,
        "lidar_observations": {
            "waypoints": observation_count,
            "pairs": observation_pairs,
        },
        "topometric_map": topometric_summary,
        "warnings": warnings,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("route", type=Path)
    args = parser.parse_args()
    result = validate_route(args.route.expanduser().resolve())
    print("S10_ROUTE_VALIDATION", json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
