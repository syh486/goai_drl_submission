"""Smoke test for deterministic anchor-frame prior-map export."""

from __future__ import annotations

import json
from pathlib import Path
import struct
import tempfile

import numpy as np

from deployment.mapping.export_topometric_prior_map import export_prior_map


def _read_xyz_ply(path: Path) -> np.ndarray:
    with path.open("rb") as stream:
        count = None
        while True:
            line = stream.readline().decode("ascii").strip()
            if line.startswith("element vertex "):
                count = int(line.rsplit(" ", 1)[1])
            if line == "end_header":
                break
        assert count is not None
        return np.frombuffer(stream.read(), dtype="<f4").reshape(count, 3)


def main() -> None:
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        (root / "submaps").mkdir()
        pose0 = np.eye(4)
        pose1 = np.eye(4)
        pose1[0, 3] = 1.0
        np.savez_compressed(
            root / "submaps/submap_0000.npz",
            anchor_pose=pose0,
            fine_points_anchor_m=np.asarray([[0, 0, 0], [0.02, 0, 0]], np.float32),
        )
        np.savez_compressed(
            root / "submaps/submap_0001.npz",
            anchor_pose=pose1,
            fine_points_anchor_m=np.asarray([[0, 0, 0], [1, 0, 0]], np.float32),
        )
        manifest = {
            "coordinate_frame": "test_map",
            "submaps": [
                {"file": "submaps/submap_0000.npz"},
                {"file": "submaps/submap_0001.npz"},
            ],
        }
        manifest_path = root / "localization_map_manifest.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        output = root / "prior.ply"
        report = export_prior_map(manifest_path, output, voxel_size_m=0.1, batch_submaps=1)
        points = _read_xyz_ply(output)
        assert report["voxel_points"] == 3
        assert np.allclose(points[:, 0], [0.0, 1.0, 2.0])
        assert report["coordinate_frame"] == "test_map"


if __name__ == "__main__":
    main()
