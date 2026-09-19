"""ROS-independent regression checks for the standard mapping bag exporter."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from deployment.mapping.export_mapping_rosbag import (
    _load_imu_rows,
    _quaternion_xyzw_from_rotation,
    _relative_point_times,
    _stamp_parts,
)


def main() -> None:
    assert _stamp_parts(10.25) == (10, 250_000_000)
    assert _stamp_parts(10.9999999996) == (11, 0)

    expected = np.linspace(-0.1, 0.0, 8, dtype=np.float64)
    for values in (
        expected,
        (expected + 1234.5) * 1.0e9,
        (expected + 1234.5) * 1.0e6,
        (expected - expected.min()) * 1.0e3,
        (expected - expected.min()) * 1.0e9,
    ):
        frame_stamp = 1234.5 if float(np.median(np.abs(values))) > 1.0e8 else 1234.5
        actual = _relative_point_times(values, frame_stamp)
        np.testing.assert_allclose(np.ptp(actual), 0.1, atol=1.0e-6)
        assert float(actual.max()) <= 1.0e-5

    qx, qy, qz, qw = _quaternion_xyzw_from_rotation(np.eye(3))
    np.testing.assert_allclose((qx, qy, qz, qw), (0.0, 0.0, 0.0, 1.0))

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        (root / "session.json").write_text('{"imu_stream": null}\n', encoding="utf-8")
        rows, columns = _load_imu_rows(root)
        assert rows.shape == (0, 11)
        assert columns == []

    print("EXPORT_MAPPING_ROSBAG_OK")


if __name__ == "__main__":
    main()
