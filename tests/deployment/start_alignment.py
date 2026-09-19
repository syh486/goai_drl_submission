"""CPU contract for automatic LiDAR start-frame relocalization."""

from __future__ import annotations

from pathlib import Path
import tempfile

import numpy as np

from deployment.localization.start_alignment import (
    StartAlignmentConfig,
    StartAnchor,
    StartAnchorAccumulator,
    align_start_anchor,
    anchor_path_for_route,
    compose_aligned_initial_pose,
    load_start_anchor,
    save_start_anchor,
)


def _asymmetric_scene() -> np.ndarray:
    points = []
    for x, y_start, width, height in ((5.0, -3.0, 6.0, 3.0), (-4.0, 6.0, 4.0, 4.0)):
        y, z = np.meshgrid(
            np.linspace(y_start, y_start + width, 100),
            np.linspace(-0.2, height, 30),
        )
        points.append(np.column_stack((np.full(y.size, x), y.ravel(), z.ravel())))
    for center_x, center_y, radius, height in (
        (2.0, 4.0, 0.25, 4.0),
        (-5.0, -2.0, 0.35, 5.0),
        (7.0, -6.0, 0.30, 3.0),
    ):
        angle, z = np.meshgrid(
            np.linspace(0.0, 2.0 * np.pi, 80, endpoint=False),
            np.linspace(-0.3, height, 35),
        )
        points.append(np.column_stack((
            center_x + radius * np.cos(angle.ravel()),
            center_y + radius * np.sin(angle.ravel()),
            z.ravel(),
        )))
    return np.concatenate(points, axis=0)


def _rotation_z(yaw_deg: float) -> np.ndarray:
    yaw = np.deg2rad(yaw_deg)
    cosine, sine = np.cos(yaw), np.sin(yaw)
    return np.asarray(((cosine, -sine, 0.0), (sine, cosine, 0.0), (0.0, 0.0, 1.0)))


def main() -> None:
    rng = np.random.default_rng(20260912)
    reference = _asymmetric_scene()
    config = StartAlignmentConfig(
        capture_frames=3,
        voxel_size_m=0.10,
        yaw_search_step_deg=2.0,
        max_registration_points=3500,
    )

    accumulator = StartAnchorAccumulator(config)
    assert not accumulator.add(reference, stationary=True)
    assert not accumulator.add(reference, stationary=False)
    assert accumulator.reset_count == 1 and not accumulator.frames
    for _ in range(3):
        complete = accumulator.add(
            reference + rng.normal(0.0, 0.003, reference.shape), stationary=True
        )
    assert complete
    merged = accumulator.merged()

    with tempfile.TemporaryDirectory() as directory:
        route_path = Path(directory) / "route.yaml"
        anchor_path = anchor_path_for_route(route_path)
        save_start_anchor(
            anchor_path,
            merged,
            np.asarray((1.0, 0.0, 0.0, 0.0)),
            frame_count=3,
            voxel_size_m=config.voxel_size_m,
        )
        anchor = load_start_anchor(anchor_path)
        assert anchor.frame_count == 3 and len(anchor.points_reference_body_m) == len(merged)

        expected_yaw_deg = 5.0
        expected_rotation = _rotation_z(expected_yaw_deg)
        expected_translation = np.asarray((0.23, -0.17, 0.04))
        live = (reference - expected_translation) @ expected_rotation
        live = live[rng.random(len(live)) > 0.25]
        live += rng.normal(0.0, 0.006, live.shape)
        result = align_start_anchor(
            StartAnchor(reference, anchor.initial_imu_quaternion_wxyz, 3, 0.10),
            live,
            config,
        )
        assert abs(result.yaw_deg - expected_yaw_deg) < 0.20, result
        np.testing.assert_allclose(
            result.translation_reference_live_m, expected_translation, atol=0.035
        )
        assert result.rmse_m < 0.06 and result.overlap_fraction > 0.45

        reference_pose = np.asarray((10.0, 20.0, 0.425, 1.0, 0.0, 0.0, 0.0))
        position, rotation = compose_aligned_initial_pose(reference_pose, result)
        np.testing.assert_allclose(position, reference_pose[:3] + expected_translation, atol=0.035)
        assert abs(np.degrees(np.arctan2(rotation[1, 0], rotation[0, 0])) - 5.0) < 0.20

    print("START_ALIGNMENT_OK", {
        "yaw_deg": result.yaw_deg,
        "translation_m": result.translation_reference_live_m.tolist(),
        "rmse_m": result.rmse_m,
        "overlap": result.overlap_fraction,
    })


if __name__ == "__main__":
    main()
