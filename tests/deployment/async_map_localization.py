"""Regression checks for non-blocking latest-only map localization."""

from __future__ import annotations

import threading
import time

import numpy as np

from deployment.navigation.ros2_node import _AsyncMapLocalizationWorker


class _Logger:
    def error(self, message: str) -> None:
        raise AssertionError(message)

    def warning(self, _message: str) -> None:
        pass


class _Result:
    observation_accepted = True


class _SlowLocalizer:
    def __init__(self) -> None:
        self.started = threading.Event()
        self.calls: list[float] = []
        self.last_candidate_audits = []

    def update(
        self,
        _points: np.ndarray,
        _pose: np.ndarray,
        *,
        traveled_distance_m: float,
        allow_large_relocalization: bool,
    ) -> _Result:
        del allow_large_relocalization
        self.calls.append(traveled_distance_m)
        self.started.set()
        time.sleep(0.04)
        return _Result()


def main() -> None:
    localizer = _SlowLocalizer()
    worker = _AsyncMapLocalizationWorker(_Logger())
    worker.start(localizer)
    cloud = np.zeros((10, 3), dtype=np.float64)
    pose = np.eye(4)
    worker.submit(cloud, pose, traveled_distance_m=1.0, allow_large_relocalization=False)
    assert localizer.started.wait(timeout=1.0)
    worker.submit(cloud, pose, traveled_distance_m=2.0, allow_large_relocalization=False)
    worker.submit(cloud, pose, traveled_distance_m=3.0, allow_large_relocalization=False)

    deadline = time.monotonic() + 2.0
    while worker.snapshot().completed < 2 and time.monotonic() < deadline:
        time.sleep(0.01)
    snapshot = worker.snapshot()
    worker.close()

    assert localizer.calls == [1.0, 3.0]
    assert snapshot.submitted == 3
    assert snapshot.completed == 2
    assert snapshot.dropped_requests == 1
    assert snapshot.last_accepted_monotonic_s > 0.0
    assert snapshot.last_update_seconds >= 0.035
    assert snapshot.last_error is None
    print("ASYNC_MAP_LOCALIZATION_OK")


if __name__ == "__main__":
    main()
