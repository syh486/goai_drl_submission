"""CPU contract test for quality-gated waypoint collection."""

from __future__ import annotations

from pathlib import Path
import tempfile

import numpy as np
import yaml

from deployment.waypoints.validate_route import validate_route
from deployment.waypoints.collection import (
    CollectionLimits,
    CollectionRejected,
    PoseSample,
    WaypointCollectionSession,
)
from deployment.localization.start_alignment import save_start_anchor


def sample(index: int, *, position=(0.0, 0.0, 0.425), speed=0.0, healthy=True):
    return PoseSample(
        stamp_s=100.0 + index * 0.1,
        receipt_s=10.0 + index * 0.1,
        position=np.asarray(position, dtype=np.float64),
        quaternion_wxyz=np.asarray((1.0, 0.0, 0.0, 0.0)),
        linear_velocity=np.asarray((speed, 0.0, 0.0)),
        angular_velocity=np.zeros(3),
        healthy=healthy,
        quality_state="GOOD" if healthy else "DEGRADED",
        diagnostics={"covariance_trace": 0.25, "icp_accepted": True},
    )


def main() -> None:
    with tempfile.TemporaryDirectory() as directory:
        route = Path(directory) / "route.yaml"
        limits = CollectionLimits(window_s=0.8, min_samples=5)
        session = WaypointCollectionSession(route, limits=limits)
        for index in range(8):
            session.append(sample(index, position=(index * 0.001, 0.0, 0.425)))
        first = session.mark(name="start", tags=["start"], now_s=10.7)
        assert first["index"] == 0

        for index in range(8, 18):
            session.append(sample(index, position=(1.0 + index * 0.001, 0.0, 0.525)))
        second = session.mark(name="turn_entry", tags=["turn"], now_s=11.7)
        assert second["index"] == 1
        save_start_anchor(
            session.anchor_path,
            np.column_stack((
                np.linspace(0.5, 2.0, 120),
                np.sin(np.linspace(0.0, 4.0, 120)),
                np.linspace(-0.1, 1.0, 120),
            )),
            np.asarray((1.0, 0.0, 0.0, 0.0)),
            frame_count=12,
            voxel_size_m=0.12,
        )
        session.save()
        payload = yaml.safe_load(route.read_text(encoding="utf-8"))
        assert payload["schema_version"] == 2
        assert payload["initial_alignment"]["anchor_file"] == "route.anchor.npz"
        assert len(payload["initial_alignment"]["anchor_sha256"]) == 64
        assert payload["nodes"][0]["position"][2] == 0.0
        assert payload["nodes"][1]["position"][2] == 0.1
        assert session.audit_path.exists()
        result = validate_route(route)
        assert result["waypoint_count"] == 2
        assert result["start_anchor"]["frames"] == 12

        resumed = WaypointCollectionSession(route, limits=limits, resume=True)
        assert len(resumed.nodes) == 2
        resumed.undo()
        assert len(resumed.nodes) == 1

        moving_route = Path(directory) / "moving.yaml"
        moving = WaypointCollectionSession(moving_route, limits=limits)
        for index in range(8):
            moving.append(sample(index, speed=0.2))
        try:
            moving.mark(now_s=10.7)
        except CollectionRejected as error:
            assert "translating" in str(error)
        else:
            raise AssertionError("moving robot was allowed to record a waypoint")

        start_route = Path(directory) / "single_sample_start.yaml"
        start_session = WaypointCollectionSession(
            start_route,
            limits=CollectionLimits(
                window_s=5.0,
                min_samples=3,
                start_min_samples=1,
                max_pose_age_s=2.0,
            ),
        )
        start_session.append(sample(0))
        event = start_session.mark(
            name="start",
            tags=["start"],
            min_samples=start_session.limits.start_min_samples,
            now_s=10.0,
        )
        assert event["window"]["samples"] == 1

        print("WAYPOINT_COLLECTION_OK", result, flush=True)


if __name__ == "__main__":
    main()
