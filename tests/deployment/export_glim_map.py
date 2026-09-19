"""Regression checks for GLIM map export primitives."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from deployment.mapping.export_glim_map import _read_world_origin, _voxel_downsample, _write_xyzi_pcd


def main() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        data = root / "data.txt"
        data.write_text(
            "id: 0\nT_world_origin: \n"
            "1 0 0 1\n0 1 0 2\n0 0 1 3\n0 0 0 1\n",
            encoding="utf-8",
        )
        transform = _read_world_origin(data)
        np.testing.assert_allclose(transform[:3, 3], (1.0, 2.0, 3.0))
        points = _voxel_downsample(
            np.asarray(((0.01, 0.01, 0.01), (0.02, 0.02, 0.02), (1.0, 0.0, 0.0))),
            0.1,
        )
        assert points.shape == (2, 3)
        pcd = root / "map.pcd"
        _write_xyzi_pcd(pcd, points)
        payload = pcd.read_bytes()
        assert b"FIELDS x y z intensity" in payload
        assert payload.endswith(np.zeros(1, dtype=np.float32).tobytes())
    print("EXPORT_GLIM_MAP_OK")


if __name__ == "__main__":
    main()
