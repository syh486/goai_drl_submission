"""CPU contract test for timestamp-based offline waypoint rebinding."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile

import numpy as np
import yaml

from deployment.waypoints.rebind import rebind_waypoints
from deployment.navigation.core import apply_route_file


def _pose(x: float, yaw_deg: float) -> np.ndarray:
    yaw = np.deg2rad(yaw_deg)
    result = np.eye(4)
    result[:2, :2] = ((np.cos(yaw), -np.sin(yaw)), (np.sin(yaw), np.cos(yaw)))
    result[:3, 3] = (x, 0.0, 0.425)
    return result


def main() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        route = root / "route.yaml"
        route.write_text(yaml.safe_dump({
            "schema_version": 2,
            "coordinate_frame": "s10_route_map",
            "initial_map_yaw_deg": 0.0,
            "initial_base_height_m": 0.425,
            "collection": {},
            "initial_alignment": {},
            "nodes": [
                {"name": "start", "position": [0.0, 0.0, 0.0]},
                {"name": "goal", "position": [10.0, 0.0, 0.1]},
            ],
        }, sort_keys=False), encoding="utf-8")
        events = [{
            "event": "mark",
            "index": index,
            "name": name,
            "sensor_stamp_s": stamp,
            "position_base_median_m": online_base,
            "position_terrain_m": online_terrain,
        } for index, name, stamp, online_base, online_terrain in (
            (0, "start", 100.0, [0.0, 0.0, 0.425], [0.0, 0.0, 0.0]),
            (1, "goal", 101.0, [10.0, 0.0, 0.525], [10.0, 0.0, 0.1]),
        )]
        route.with_suffix(".quality.json").write_text(json.dumps({
            "schema_version": 1, "route_file": str(route), "events": events,
        }), encoding="utf-8")
        session = root / "mapping"
        (session / "keyframes").mkdir(parents=True)
        for index, stamp in enumerate((100.0, 102.0)):
            np.savez_compressed(session / "keyframes" / f"{index:06d}.npz", stamp_s=stamp)
        poses = np.asarray((_pose(0.0, 10.0), _pose(4.0, 30.0)))
        poses_path = root / "poses.npy"
        np.save(poses_path, poses)
        topometric = root / "topometric"
        (topometric / "submaps").mkdir(parents=True)
        np.savez_compressed(
            topometric / "submaps" / "submap_0000.npz",
            anchor_pose=poses[0],
        )
        np.savez_compressed(
            topometric / "submaps" / "submap_0001.npz",
            anchor_pose=poses[1],
        )
        (topometric / "localization_map_manifest.json").write_text(json.dumps({
            "map_type": "ordered_topometric_route_submaps",
            "submap_count": 2,
            "reference_sessions": [str(session.resolve())],
            "submaps": [
                {"submap_id": 0, "route_index": 0, "anchor_frame": 0,
                 "file": "submaps/submap_0000.npz"},
                {"submap_id": 1, "route_index": 1, "anchor_frame": 1,
                 "file": "submaps/submap_0001.npz"},
            ],
        }), encoding="utf-8")
        output = root / "route_rebound.yaml"
        report = rebind_waypoints(
            route, session, poses_path, output, max_timestamp_offset_s=1.1,
            topometric_map=topometric,
        )
        payload = yaml.safe_load(output.read_text(encoding="utf-8"))
        np.testing.assert_allclose(payload["nodes"][1]["position"], (2.0, 0.0, 0.0), atol=1e-4)
        assert abs(payload["initial_map_yaw_deg"] - 10.0) < 1e-6
        assert report["waypoint_count"] == 2
        assert payload["nodes"][0]["topometric_binding"]["route_index"] == 0
        assert payload["nodes"][1]["topometric_binding"]["route_index"] == 0
        np.testing.assert_allclose(
            payload["nodes"][0]["topometric_binding"]["position_anchor_m"],
            (0.0, 0.0, -0.425), atol=1e-4,
        )
        assert output.with_suffix(".quality.json").is_file()
        assert output.with_suffix(".rebind.json").is_file()
        hardware = {
            "route": {"waypoints_map_m": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]},
            "map_localization": {"enabled": False},
        }
        apply_route_file(hardware, output)
        assert hardware["map_localization"]["enabled"] is True
        assert Path(hardware["map_localization"]["map_dir"]) == topometric.resolve()
        assert len(hardware["route"]["waypoint_topometric_bindings"]) == 2
    print("WAYPOINT_REBIND_OK")


if __name__ == "__main__":
    main()
