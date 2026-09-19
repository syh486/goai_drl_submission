"""Contract checks for the controlled low-level capability courses."""

from __future__ import annotations

import numpy as np

from training.terrains.capability_course import build_capability_course


def main() -> None:
    _, platform = build_capability_course("platform", 0.28)
    assert platform.height_xy.shape == (121, 61)
    assert set(np.unique(platform.height_xy)) == {0.0, 0.28}
    assert np.isclose(platform.total_height, 0.28)

    _, stairs = build_capability_course("stairs", 0.14)
    expected = np.asarray((0.0, 0.14, 0.28, 0.42, 0.56, 0.70))
    np.testing.assert_allclose(np.unique(stairs.height_xy), expected, atol=1.0e-12)
    assert np.isclose(stairs.total_height, 0.70)
    assert stairs.stable_check_x > stairs.top_entry_x
    print("CAPABILITY_COURSE_CONTRACT_OK")


if __name__ == "__main__":
    main()
