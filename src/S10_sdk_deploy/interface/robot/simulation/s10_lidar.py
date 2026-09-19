"""S10 Airy-style dual LiDAR ray casting and replay chunk writer."""

from __future__ import annotations

import os
from pathlib import Path
import queue
import threading

import mujoco
import numpy as np


AIRY_VERTICAL_RAY_ANGLES = np.asarray(
    (
        -0.07, 0.88, 1.81, 2.76, 3.69, 4.62, 5.54, 6.48, 7.41, 8.34, 9.27, 10.21, 11.15, 12.09, 13.03, 13.98,
        14.92, 15.87, 16.82, 17.77, 18.72, 19.67, 20.62, 21.57, 22.51, 23.45, 24.4, 25.33, 26.28, 27.21, 28.15,
        29.08, 30.02, 30.95, 31.88, 32.82, 33.74, 34.68, 35.62, 36.55, 37.5, 38.43, 39.37, 40.31, 41.25, 42.21,
        43.16, 44.09, 45.05, 46.0, 46.95, 47.9, 48.85, 49.8, 50.73, 51.69, 52.62, 53.56, 54.5, 55.45, 56.37,
        57.3, 58.24, 59.18, 60.12, 61.05, 61.99, 62.93, 63.86, 64.81, 65.76, 66.69, 67.65, 68.6, 69.56, 70.51,
        71.46, 72.42, 73.37, 74.33, 75.29, 76.24, 77.19, 78.14, 79.07, 80.02, 80.96, 81.9, 82.84, 83.78, 84.7,
        85.64, 86.57, 87.52, 88.46, 89.4,
    ),
    dtype=np.float64,
)

FRONT_LIDAR_POS = np.asarray((0.22341, 0.0, -0.0001), dtype=np.float64)
REAR_LIDAR_POS = np.asarray((-0.22341, 0.0, -0.0001), dtype=np.float64)
# These are the established SRU front/rear sensor-frame orientations.
FRONT_LIDAR_ROT_WXYZ = np.asarray((0.00084463, 0.7071065, -0.00028154, 0.7071065), dtype=np.float64)
REAR_LIDAR_ROT_WXYZ = np.asarray((0.70703477, 0.0, -0.70717879, 0.0), dtype=np.float64)


def quat_wxyz_to_rotmat(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float64)
    quat = quat / max(float(np.linalg.norm(quat)), 1.0e-12)
    w, x, y, z = quat
    return np.asarray(
        (
            (1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z), 2.0 * (x * z + w * y)),
            (2.0 * (x * y + w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x)),
            (2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y)),
        ),
        dtype=np.float64,
    )


def build_sensor_directions(horizontal_samples: int = 900) -> np.ndarray:
    horizontal_deg = np.arange(horizontal_samples, dtype=np.float64) * (360.0 / horizontal_samples) - 180.0
    pitch_deg, yaw_deg = np.meshgrid(AIRY_VERTICAL_RAY_ANGLES, horizontal_deg, indexing="xy")
    pitch = np.deg2rad(pitch_deg.reshape(-1)) + np.pi / 2.0
    yaw = np.deg2rad(yaw_deg.reshape(-1))
    x = np.sin(pitch) * np.cos(yaw)
    y = np.sin(pitch) * np.sin(yaw)
    z = np.cos(pitch)
    # Match the Bpearl pattern convention used by the previous SRU pipeline.
    directions = -np.stack((x, y, z), axis=1)
    return np.ascontiguousarray(
        np.roll(directions.reshape(horizontal_samples, 96, 3).transpose(1, 0, 2), -horizontal_samples // 2, axis=1),
        dtype=np.float64,
    )


class S10LidarSampler:
    """Capture clean native scans and write atomic replay chunks."""

    def __init__(
        self,
        model: mujoco.MjModel,
        replay_dir: str | Path | None,
        *,
        chunk_size: int = 64,
        max_range_m: float = 10.0,
        max_samples: int | None = None,
        horizontal_samples: int = 900,
    ) -> None:
        self.model = model
        self.replay_dir = None if replay_dir is None else Path(replay_dir).expanduser().resolve()
        if self.replay_dir is not None:
            self.replay_dir.mkdir(parents=True, exist_ok=True)
        self.chunk_size = max(1, int(chunk_size))
        self.max_range_m = float(max_range_m)
        self.horizontal_samples = int(horizontal_samples)
        if self.horizontal_samples < 90 or self.horizontal_samples % 90 != 0:
            raise ValueError("horizontal_samples must be a multiple of 90")
        self.max_samples = None if max_samples is None or int(max_samples) <= 0 else int(max_samples)
        self.directions = build_sensor_directions(self.horizontal_samples)
        self.origins = np.empty((2, 3), dtype=np.float64)
        self.world_directions = np.empty((2, 96, self.horizontal_samples, 3), dtype=np.float64)
        self.geom_ids = np.empty((96 * self.horizontal_samples,), dtype=np.int32)
        self.distances = np.empty((96 * self.horizontal_samples,), dtype=np.float64)
        # S10_track.xml puts the terrain mesh in group 0 and robot collision
        # geoms in group 1. bodyexclude removes the robot subtree, so group 0
        # is the equivalent of IsaacLab's mesh_prim_paths=["/World/ground"].
        self.geom_group = np.asarray((1, 0, 0, 0, 0, 0), dtype=np.uint8)
        self.base_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
        if self.base_body_id < 0:
            raise RuntimeError("S10 LiDAR sampler cannot find base_link.")
        self.sensor_positions_b = (FRONT_LIDAR_POS, REAR_LIDAR_POS)
        self.sensor_rotations_b = (
            quat_wxyz_to_rotmat(FRONT_LIDAR_ROT_WXYZ),
            quat_wxyz_to_rotmat(REAR_LIDAR_ROT_WXYZ),
        )
        self.chunk_index = self._next_chunk_index()
        self.total_samples = 0
        self._records: list[dict[str, np.ndarray | int | float]] = []

    def _next_chunk_index(self) -> int:
        if self.replay_dir is None:
            return 0
        existing = sorted(self.replay_dir.glob("chunk_*.npz"))
        if not existing:
            return 0
        return max(int(path.stem.split("_")[-1]) for path in existing) + 1

    @property
    def reached_limit(self) -> bool:
        return self.max_samples is not None and self.total_samples >= self.max_samples

    def capture_pose(
        self,
        data: mujoco.MjData,
        root_pos: np.ndarray,
        root_quat_wxyz: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        root_pos = np.asarray(root_pos, dtype=np.float64)
        root_rot = quat_wxyz_to_rotmat(np.asarray(root_quat_wxyz, dtype=np.float64))
        clean_scans = []
        for sensor_pos_b, sensor_rot_b in zip(self.sensor_positions_b, self.sensor_rotations_b):
            origin = root_pos + root_rot @ sensor_pos_b
            world_rot = root_rot @ sensor_rot_b
            world_dirs = np.einsum("ij,hwj->hwi", world_rot, self.directions)
            self.origins[len(clean_scans)] = origin
            self.world_directions[len(clean_scans)] = world_dirs
            mujoco.mj_multiRay(
                self.model,
                data,
                origin,
                np.ascontiguousarray(world_dirs.reshape(-1), dtype=np.float64),
                self.geom_group,
                1,
                self.base_body_id,
                self.geom_ids,
                self.distances,
                None,
                96 * self.horizontal_samples,
                self.max_range_m,
            )
            scan = self.distances.reshape(96, self.horizontal_samples).copy()
            scan[(scan <= 0.0) | (scan > self.max_range_m)] = self.max_range_m
            clean_scans.append(scan.astype(np.float16))
        return clean_scans[0], clean_scans[1]

    def capture(self, data: mujoco.MjData) -> tuple[np.ndarray, np.ndarray]:
        return self.capture_pose(data, data.qpos[:3], data.qpos[3:7])

    def append(
        self,
        data: mujoco.MjData,
        *,
        episode_id: int,
        segment_index: int,
        sim_time: float,
    ) -> bool:
        if self.reached_limit:
            return False
        front, rear = self.capture(data)
        self._records.append(
            {
                "front_raw_d": front,
                "rear_raw_d": rear,
                "root_pos_w": np.asarray(data.qpos[:3], dtype=np.float32).copy(),
                "root_quat_w": np.asarray(data.qpos[3:7], dtype=np.float32).copy(),
                "episode_id": int(episode_id),
                "segment_index": int(segment_index),
                "sim_time": float(sim_time),
            }
        )
        self.total_samples += 1
        if len(self._records) >= self.chunk_size or self.reached_limit:
            self.flush()
        return True

    def append_pose(
        self,
        data: mujoco.MjData,
        root_pos: np.ndarray,
        root_quat_wxyz: np.ndarray,
        *,
        episode_id: int,
        segment_index: int,
        sim_time: float,
    ) -> bool:
        if self.reached_limit:
            return False
        front, rear = self.capture_pose(data, root_pos, root_quat_wxyz)
        self._records.append(
            {
                "front_raw_d": front,
                "rear_raw_d": rear,
                "root_pos_w": np.asarray(root_pos, dtype=np.float32).copy(),
                "root_quat_w": np.asarray(root_quat_wxyz, dtype=np.float32).copy(),
                "episode_id": int(episode_id),
                "segment_index": int(segment_index),
                "sim_time": float(sim_time),
            }
        )
        self.total_samples += 1
        if len(self._records) >= self.chunk_size or self.reached_limit:
            self.flush()
        return True

    def flush(self) -> None:
        if not self._records:
            return
        records = self._records
        self._records = []
        if self.replay_dir is None:
            return
        part_path = self.replay_dir / f"chunk_{self.chunk_index:06d}.npz.part"
        final_path = self.replay_dir / f"chunk_{self.chunk_index:06d}.npz"
        with part_path.open("wb") as handle:
            np.savez(
                handle,
                front_raw_d=np.stack([record["front_raw_d"] for record in records]),
                rear_raw_d=np.stack([record["rear_raw_d"] for record in records]),
                root_pos_w=np.stack([record["root_pos_w"] for record in records]),
                root_quat_w=np.stack([record["root_quat_w"] for record in records]),
                episode_id=np.asarray([record["episode_id"] for record in records], dtype=np.int64),
                segment_index=np.asarray([record["segment_index"] for record in records], dtype=np.int64),
                sim_time=np.asarray([record["sim_time"] for record in records], dtype=np.float64),
                horizontal_resolution_deg=np.asarray(
                    360.0 / self.horizontal_samples, dtype=np.float32
                ),
                vertical_ray_angles=AIRY_VERTICAL_RAY_ANGLES.astype(np.float32),
            )
        os.replace(part_path, final_path)
        self.chunk_index += 1


class S10LidarReplayWorker:
    """Run expensive native raycasts away from the simulation control loop."""

    def __init__(
        self,
        xml_path: str | Path,
        replay_dir: str | Path,
        *,
        chunk_size: int = 64,
        max_range_m: float = 10.0,
        max_samples: int | None = None,
    ) -> None:
        self.model = mujoco.MjModel.from_xml_path(str(Path(xml_path).expanduser().resolve()))
        self.data = mujoco.MjData(self.model)
        self.sampler = S10LidarSampler(
            self.model,
            replay_dir,
            chunk_size=chunk_size,
            max_range_m=max_range_m,
            max_samples=max_samples,
        )
        self.queue: queue.Queue[tuple[np.ndarray, np.ndarray, np.ndarray, int, int, float] | None] = queue.Queue(maxsize=2)
        self.dropped_requests = 0
        self.error: BaseException | None = None
        self._thread = threading.Thread(target=self._run, name="s10-lidar-replay", daemon=True)
        self._thread.start()

    @property
    def reached_limit(self) -> bool:
        return self.sampler.reached_limit

    @property
    def total_samples(self) -> int:
        return self.sampler.total_samples

    def submit(
        self,
        model_qpos: np.ndarray,
        root_pos: np.ndarray,
        root_quat_wxyz: np.ndarray,
        *,
        episode_id: int,
        segment_index: int,
        sim_time: float,
    ) -> bool:
        if self.reached_limit:
            return False
        request = (
            np.asarray(model_qpos, dtype=np.float64).copy(),
            np.asarray(root_pos, dtype=np.float64).copy(),
            np.asarray(root_quat_wxyz, dtype=np.float64).copy(),
            int(episode_id),
            int(segment_index),
            float(sim_time),
        )
        try:
            self.queue.put_nowait(request)
            return True
        except queue.Full:
            self.dropped_requests += 1
            return False

    def _run(self) -> None:
        while True:
            request = self.queue.get()
            if request is None:
                self.queue.task_done()
                break
            try:
                model_qpos, root_pos, root_quat, episode_id, segment_index, sim_time = request
                self.data.qpos[:] = model_qpos
                self.data.qvel[:] = 0.0
                mujoco.mj_forward(self.model, self.data)
                self.sampler.append_pose(
                    self.data,
                    root_pos,
                    root_quat,
                    episode_id=episode_id,
                    segment_index=segment_index,
                    sim_time=sim_time,
                )
            except BaseException as exc:  # Surface worker failure on close instead of silently losing data.
                self.error = exc
            finally:
                self.queue.task_done()

    def close(self) -> None:
        self.queue.join()
        self.queue.put(None)
        self._thread.join()
        self.sampler.flush()
        if self.error is not None:
            raise RuntimeError("S10 LiDAR replay worker failed") from self.error


class S10LidarLiveWorker:
    """Asynchronous native scanner that retains only the newest in-memory frame."""

    def __init__(self, xml_path: str | Path, *, max_range_m: float = 10.0):
        self.model = mujoco.MjModel.from_xml_path(str(Path(xml_path).expanduser().resolve()))
        self.data = mujoco.MjData(self.model)
        self.sampler = S10LidarSampler(self.model, None, max_range_m=max_range_m)
        self.queue: queue.Queue[np.ndarray | None] = queue.Queue(maxsize=1)
        self.latest: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
        self.latest_lock = threading.Lock()
        self.dropped_requests = 0
        self.error: BaseException | None = None
        self._thread = threading.Thread(target=self._run, name="s10-lidar-live", daemon=True)
        self._thread.start()

    def submit(self, model_qpos: np.ndarray) -> bool:
        request = np.asarray(model_qpos, dtype=np.float64).copy()
        try:
            self.queue.put_nowait(request)
            return True
        except queue.Full:
            self.dropped_requests += 1
            return False

    def get_latest(self) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        with self.latest_lock:
            if self.latest is None:
                return None
            return self.latest[0].copy(), self.latest[1].copy(), self.latest[2].copy()

    def _run(self) -> None:
        while True:
            model_qpos = self.queue.get()
            if model_qpos is None:
                self.queue.task_done()
                break
            try:
                self.data.qpos[:] = model_qpos
                self.data.qvel[:] = 0.0
                mujoco.mj_forward(self.model, self.data)
                frame = self.sampler.capture(self.data)
                with self.latest_lock:
                    # Publish only the free-root pose. The remaining joint qpos
                    # is an internal raycast state and is not part of the SRU
                    # sensor/world-height contract.
                    self.latest = (frame[0], frame[1], model_qpos[:7].copy())
            except BaseException as exc:
                self.error = exc
            finally:
                self.queue.task_done()

    def close(self) -> None:
        self.queue.join()
        self.queue.put(None)
        self._thread.join()
        if self.error is not None:
            raise RuntimeError("S10 live LiDAR worker failed") from self.error
