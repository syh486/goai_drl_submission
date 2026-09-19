"""CPU contract for ordered route-submap map construction."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile

import numpy as np

from deployment.mapping.build_topometric_route_map import (
    TopometricMapConfig,
    build_topometric_route_map,
)
from deployment.localization.evaluate_topometric_replay import evaluate_topometric_replay


def main() -> None:
    rng = np.random.default_rng(19)
    world = rng.uniform((-5.0, -4.0, -1.0), (12.0, 4.0, 3.0), (1800, 3))
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        session = root / "session"
        keyframes = session / "keyframes"
        keyframes.mkdir(parents=True)
        poses = np.tile(np.eye(4), (120, 1, 1))
        poses[:, 0, 3] = np.linspace(0.0, 6.0, len(poses))
        for index, pose in enumerate(poses):
            points = world - pose[:3, 3]
            np.savez_compressed(
                keyframes / f"{index:06d}.npz",
                stamp_s=np.asarray(index * 0.2),
                rotation_odom_body=np.eye(3),
                points_body_m=points.astype(np.float32),
            )
        poses_path = root / "poses.npy"
        np.save(poses_path, poses)
        output = root / "map"
        manifest = build_topometric_route_map(
            session,
            poses_path,
            output,
            TopometricMapConfig(
                anchor_spacing_m=1.0,
                submap_voxel_m=0.25,
                submap_aggregate_radius_frames=2,
            ),
        )
        assert manifest["map_type"] == "ordered_topometric_route_submaps"
        assert manifest["submap_count"] >= 7
        assert Path(manifest["canonical_route_poses"]).is_file()
        assert Path(manifest["canonical_route_trajectory"]).is_file()
        assert manifest["trajectory_alignments"][0]["skipped_keyframes_start"] == 0
        loaded = json.loads(
            (output / "localization_map_manifest.json").read_text(encoding="utf-8")
        )
        assert [entry["route_index"] for entry in loaded["submaps"]] == list(
            range(loaded["submap_count"])
        )
        with np.load(output / loaded["submaps"][2]["file"], allow_pickle=False) as data:
            assert data["points_anchor_m"].shape[1] == 3
            assert data["descriptor"].shape == (20, 60)
            assert "registration_points_anchor_m" not in data.files
            assert "trajectory_frame" in data.files

        multi_output = root / "multi_map"
        multi = build_topometric_route_map(
            session,
            poses_path,
            multi_output,
            TopometricMapConfig(
                anchor_spacing_m=1.0,
                submap_voxel_m=0.25,
                submap_aggregate_radius_frames=2,
            ),
            support_laps=((session, poses_path),),
        )
        assert multi["submap_count"] == 2 * multi["route_anchor_count"]
        assert len(multi["reference_sessions"]) == 2
        route_indices = [entry["route_index"] for entry in multi["submaps"]]
        assert route_indices[: multi["route_anchor_count"]] == list(
            range(multi["route_anchor_count"])
        )
        assert route_indices[multi["route_anchor_count"] :] == list(
            range(multi["route_anchor_count"])
        )
        assert not any(
            entry["tracking_eligible"]
            for entry in multi["submaps"][multi["route_anchor_count"] :]
        )

        constrained_route = 2
        constrained_entry = manifest["submaps"][constrained_route]
        constraints_path = root / "aligned_constraints.json"
        constraints_path.write_text(json.dumps([{
            "route_index": constrained_route,
            "canonical_frame": constrained_entry["trajectory_frame"],
            "support_frame": constrained_entry["trajectory_frame"],
            "canonical_from_support": np.eye(4).tolist(),
        }]), encoding="utf-8")
        aligned_output = root / "aligned_multi_map"
        aligned_multi = build_topometric_route_map(
            session,
            poses_path,
            aligned_output,
            TopometricMapConfig(
                anchor_spacing_m=1.0,
                submap_voxel_m=0.25,
                submap_aggregate_radius_frames=2,
            ),
            aligned_support_laps=((
                session,
                poses_path,
                constraints_path,
                poses_path,
            ),),
        )
        aligned_variant = aligned_multi["submaps"][-1]
        assert aligned_multi["submap_count"] == (
            aligned_multi["route_anchor_count"] + 1
        )
        assert aligned_variant["route_index"] == constrained_route
        assert aligned_variant["tracking_eligible"]
        assert aligned_variant["alignment_method"] == (
            "bidirectional_cross_lap_constraint_rebased"
        )
        with np.load(
            aligned_output / aligned_variant["file"], allow_pickle=False
        ) as payload:
            assert np.allclose(payload["canonical_from_source_anchor"], np.eye(4))
        try:
            evaluate_topometric_replay(multi_output, session, poses_path)
        except ValueError as error:
            assert "used to build" in str(error)
        else:
            raise AssertionError("map reference session was accepted as held-out data")
    print("TOPOMETRIC_ROUTE_MAP_OK")


if __name__ == "__main__":
    main()
