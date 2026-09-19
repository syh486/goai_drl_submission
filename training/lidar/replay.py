"""Compact native LiDAR replay format used by the MuJoCo encoder trainer."""

from __future__ import annotations

from dataclasses import dataclass
import json
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch

from sru_training.s10_lidar_encoder import (
    ENCODER_DISTANCE_SCALE_M,
    ENCODER_WORLD_Z_SCALE_M,
    INVALID_RANGE_THRESHOLD_M,
    MAX_RANGE_M,
    MIN_RANGE_M,
    S10_FRONT_POS,
    S10_FRONT_ROT_WXYZ,
    S10_REAR_POS,
    S10_REAR_ROT_WXYZ,
    build_sensor_frame_directions,
    gather_aux_at_min_distance,
    native_to_90,
    world_z_native,
)


NATIVE_SHAPE = (96, 900)
ENCODER_SHAPE = (96, 90)


def add_sensor_noise(
    raw_distance: np.ndarray,
    *,
    rng: np.random.Generator,
    noise_std_m: float,
    dropout_probability: float,
) -> np.ndarray:
    raw_distance = np.asarray(raw_distance, dtype=np.float32)
    noisy = raw_distance + rng.normal(0.0, noise_std_m, raw_distance.shape).astype(np.float32)
    noisy = np.clip(noisy, MIN_RANGE_M, MAX_RANGE_M)
    if dropout_probability > 0.0:
        keep = rng.random(raw_distance.shape) >= dropout_probability
        noisy = np.where(keep, noisy, MAX_RANGE_M)
    return noisy.astype(np.float32, copy=False)


def _metric_input(
    raw_distance_native: np.ndarray,
    root_qpos: np.ndarray,
    *,
    front: bool,
    rng: np.random.Generator,
    sensor_dirs: np.ndarray,
    noise_std_m: float,
    dropout_probability: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    raw_distance_90 = native_to_90(raw_distance_native)[0][0].numpy()
    noisy_native = add_sensor_noise(
        raw_distance_native,
        rng=rng,
        noise_std_m=noise_std_m,
        dropout_probability=dropout_probability,
    )
    noisy_distance_90 = native_to_90(noisy_native)[0][0].numpy()
    noisy_world_z_native = world_z_native(
        noisy_native, root_qpos, front=front, sensor_dirs=sensor_dirs
    )
    noisy_world_z_90 = gather_aux_at_min_distance(
        noisy_world_z_native, raw_distance_native
    )[0].numpy()
    normalized = np.stack(
        (
            noisy_distance_90 / ENCODER_DISTANCE_SCALE_M,
            noisy_world_z_90 / ENCODER_WORLD_Z_SCALE_M,
        ),
        axis=0,
    ).astype(np.float32, copy=False)
    return normalized, raw_distance_90[None].astype(np.float32, copy=False), noisy_distance_90


def _native_to_90_numpy(native: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    value = np.asarray(native, dtype=np.float32)
    if value.ndim == 2:
        value = value[None]
    if value.ndim != 3 or value.shape[1:] != NATIVE_SHAPE:
        raise ValueError(f"Expected [N,96,900] or [96,900], got {value.shape}")
    grouped = value.reshape(value.shape[0], 96, 90, 10)
    valid = (grouped > MIN_RANGE_M) & (grouped < INVALID_RANGE_THRESHOLD_M)
    safe = np.where(valid, grouped, MAX_RANGE_M)
    indices = np.argmin(safe, axis=-1)
    pooled = np.take_along_axis(safe, indices[..., None], axis=-1)[..., 0]
    pooled = np.where(valid.any(axis=-1), pooled, MAX_RANGE_M)
    return pooled.astype(np.float32, copy=False), indices.astype(np.int64, copy=False)


def _quat_batch_to_rotmat(quaternions: np.ndarray) -> np.ndarray:
    quat = np.asarray(quaternions, dtype=np.float32)
    quat = quat / np.clip(np.linalg.norm(quat, axis=-1, keepdims=True), 1.0e-8, None)
    w, x, y, z = [quat[:, index] for index in range(4)]
    rotation = np.empty((len(quat), 3, 3), dtype=np.float32)
    rotation[:, 0, 0] = 1.0 - 2.0 * (y * y + z * z)
    rotation[:, 0, 1] = 2.0 * (x * y - w * z)
    rotation[:, 0, 2] = 2.0 * (x * z + w * y)
    rotation[:, 1, 0] = 2.0 * (x * y + w * z)
    rotation[:, 1, 1] = 1.0 - 2.0 * (x * x + z * z)
    rotation[:, 1, 2] = 2.0 * (y * z - w * x)
    rotation[:, 2, 0] = 2.0 * (x * z - w * y)
    rotation[:, 2, 1] = 2.0 * (y * z + w * x)
    rotation[:, 2, 2] = 1.0 - 2.0 * (x * x + y * y)
    return rotation


def _world_z_batch(
    distance_native: np.ndarray,
    root_qpos: np.ndarray,
    *,
    front: bool,
    sensor_dirs: np.ndarray,
) -> np.ndarray:
    root_qpos = np.asarray(root_qpos, dtype=np.float32)
    root_rot = _quat_batch_to_rotmat(root_qpos[:, 3:7])
    sensor_pos = S10_FRONT_POS if front else S10_REAR_POS
    sensor_rot = _quat_batch_to_rotmat(
        np.asarray((S10_FRONT_ROT_WXYZ if front else S10_REAR_ROT_WXYZ,), dtype=np.float32)
    )[0]
    sensor_tf = np.einsum("bij,jk->bik", root_rot, sensor_rot)
    directions_world = np.einsum("bij,hwj->bhwi", sensor_tf, sensor_dirs)
    sensor_origins = root_qpos[:, :3] + np.einsum("bij,j->bi", root_rot, sensor_pos)
    values = sensor_origins[:, None, None, 2] + np.asarray(distance_native, dtype=np.float32) * directions_world[..., 2]
    valid = (np.asarray(distance_native) < INVALID_RANGE_THRESHOLD_M) & (np.asarray(distance_native) > MIN_RANGE_M)
    return np.where(valid, np.clip(values, -3.0, 3.0), 0.0).astype(np.float32, copy=False)


def build_training_batch(
    front_raw_native: np.ndarray,
    rear_raw_native: np.ndarray,
    root_qpos: np.ndarray,
    *,
    rng: np.random.Generator,
    sensor_dirs: np.ndarray,
    noise_std_m: float = 0.03,
    dropout_probability: float = 0.0,
) -> dict[str, np.ndarray]:
    """Build a batch without Python loops over samples."""

    front_raw_native = np.asarray(front_raw_native, dtype=np.float32)
    rear_raw_native = np.asarray(rear_raw_native, dtype=np.float32)
    root_qpos = np.asarray(root_qpos, dtype=np.float32)
    if front_raw_native.ndim == 2:
        front_raw_native = front_raw_native[None]
        rear_raw_native = rear_raw_native[None]
        root_qpos = root_qpos[None]
    front_raw, front_indices = _native_to_90_numpy(front_raw_native)
    rear_raw, rear_indices = _native_to_90_numpy(rear_raw_native)
    front_noisy_native = add_sensor_noise(front_raw_native, rng=rng, noise_std_m=noise_std_m, dropout_probability=dropout_probability)
    rear_noisy_native = add_sensor_noise(rear_raw_native, rng=rng, noise_std_m=noise_std_m, dropout_probability=dropout_probability)
    front_noisy, _ = _native_to_90_numpy(front_noisy_native)
    rear_noisy, _ = _native_to_90_numpy(rear_noisy_native)
    front_z_native = _world_z_batch(front_noisy_native, root_qpos, front=True, sensor_dirs=sensor_dirs)
    rear_z_native = _world_z_batch(rear_noisy_native, root_qpos, front=False, sensor_dirs=sensor_dirs)
    front_z_grouped = front_z_native.reshape(len(root_qpos), 96, 90, 10)
    rear_z_grouped = rear_z_native.reshape(len(root_qpos), 96, 90, 10)
    front_z = np.take_along_axis(front_z_grouped, front_indices[..., None], axis=-1)[..., 0]
    rear_z = np.take_along_axis(rear_z_grouped, rear_indices[..., None], axis=-1)[..., 0]
    return {
        "front_input": np.stack((front_noisy / ENCODER_DISTANCE_SCALE_M, front_z / ENCODER_WORLD_Z_SCALE_M), axis=1).astype(np.float32, copy=False),
        "rear_input": np.stack((rear_noisy / ENCODER_DISTANCE_SCALE_M, rear_z / ENCODER_WORLD_Z_SCALE_M), axis=1).astype(np.float32, copy=False),
        "front_target": front_raw[:, None].astype(np.float32, copy=False),
        "rear_target": rear_raw[:, None].astype(np.float32, copy=False),
    }


def _quat_batch_to_rotmat_torch(quaternions: torch.Tensor) -> torch.Tensor:
    quat = quaternions / torch.linalg.vector_norm(quaternions, dim=-1, keepdim=True).clamp_min(1.0e-8)
    w, x, y, z = quat.unbind(dim=-1)
    rotation = torch.empty((len(quat), 3, 3), dtype=quat.dtype, device=quat.device)
    rotation[:, 0, 0] = 1.0 - 2.0 * (y * y + z * z)
    rotation[:, 0, 1] = 2.0 * (x * y - w * z)
    rotation[:, 0, 2] = 2.0 * (x * z + w * y)
    rotation[:, 1, 0] = 2.0 * (x * y + w * z)
    rotation[:, 1, 1] = 1.0 - 2.0 * (x * x + z * z)
    rotation[:, 1, 2] = 2.0 * (y * z - w * x)
    rotation[:, 2, 0] = 2.0 * (x * z - w * y)
    rotation[:, 2, 1] = 2.0 * (y * z + w * x)
    rotation[:, 2, 2] = 1.0 - 2.0 * (x * x + y * y)
    return rotation


def build_training_batch_torch(
    raw_batch: dict[str, np.ndarray],
    *,
    device: torch.device,
    noise_std_m: float = 0.03,
    dropout_probability: float = 0.0,
) -> dict[str, torch.Tensor]:
    """Build augmented encoder tensors on the learner device."""

    front_raw_native = torch.as_tensor(raw_batch["front_raw_d"], dtype=torch.float32, device=device)
    rear_raw_native = torch.as_tensor(raw_batch["rear_raw_d"], dtype=torch.float32, device=device)
    root_qpos = torch.as_tensor(raw_batch["root_qpos"], dtype=torch.float32, device=device)
    dirs = torch.as_tensor(build_sensor_frame_directions(), dtype=torch.float32, device=device)

    def augment(native: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        noisy = (native + torch.randn_like(native) * noise_std_m).clamp(MIN_RANGE_M, MAX_RANGE_M)
        if dropout_probability > 0.0:
            noisy = torch.where(torch.rand_like(noisy) >= dropout_probability, noisy, MAX_RANGE_M)
        raw_90, raw_indices = native_to_90(native)
        noisy_90, _ = native_to_90(noisy)
        return noisy, raw_90, raw_indices, noisy_90

    def world_z(native: torch.Tensor, *, front: bool) -> torch.Tensor:
        root_rot = _quat_batch_to_rotmat_torch(root_qpos[:, 3:7])
        sensor_pos = torch.as_tensor(S10_FRONT_POS if front else S10_REAR_POS, dtype=torch.float32, device=device)
        sensor_quat = torch.as_tensor(S10_FRONT_ROT_WXYZ if front else S10_REAR_ROT_WXYZ, dtype=torch.float32, device=device).expand(len(native), -1)
        sensor_tf = torch.bmm(root_rot, _quat_batch_to_rotmat_torch(sensor_quat))
        directions_world = torch.einsum("bij,hwj->bhwi", sensor_tf, dirs)
        origins = root_qpos[:, :3] + torch.bmm(root_rot, sensor_pos.view(1, 3, 1).expand(len(native), -1, -1)).squeeze(-1)
        values = origins[:, None, None, 2] + native * directions_world[..., 2]
        valid = (native > MIN_RANGE_M) & (native < INVALID_RANGE_THRESHOLD_M)
        return torch.where(valid, values.clamp(-3.0, 3.0), torch.zeros_like(values))

    front_noisy, front_raw, front_indices, front_noisy_90 = augment(front_raw_native)
    rear_noisy, rear_raw, rear_indices, rear_noisy_90 = augment(rear_raw_native)
    front_z_native = world_z(front_noisy, front=True)
    rear_z_native = world_z(rear_noisy, front=False)
    front_z = front_z_native.reshape(len(root_qpos), 96, 90, 10).gather(-1, front_indices[..., None]).squeeze(-1)
    rear_z = rear_z_native.reshape(len(root_qpos), 96, 90, 10).gather(-1, rear_indices[..., None]).squeeze(-1)
    return {
        "front_input": torch.stack((front_noisy_90 / ENCODER_DISTANCE_SCALE_M, front_z / ENCODER_WORLD_Z_SCALE_M), dim=1),
        "rear_input": torch.stack((rear_noisy_90 / ENCODER_DISTANCE_SCALE_M, rear_z / ENCODER_WORLD_Z_SCALE_M), dim=1),
        "front_target": front_raw.unsqueeze(1),
        "rear_target": rear_raw.unsqueeze(1),
    }


def build_training_sample(
    front_raw_native: np.ndarray,
    rear_raw_native: np.ndarray,
    root_qpos: np.ndarray,
    *,
    rng: np.random.Generator,
    sensor_dirs: np.ndarray,
    noise_std_m: float = 0.03,
    dropout_probability: float = 0.0,
) -> dict[str, np.ndarray]:
    front_input, front_target, _ = _metric_input(
        front_raw_native, root_qpos, front=True, rng=rng, sensor_dirs=sensor_dirs,
        noise_std_m=noise_std_m, dropout_probability=dropout_probability,
    )
    rear_input, rear_target, _ = _metric_input(
        rear_raw_native, root_qpos, front=False, rng=rng, sensor_dirs=sensor_dirs,
        noise_std_m=noise_std_m, dropout_probability=dropout_probability,
    )
    return {
        "front_input": front_input,
        "rear_input": rear_input,
        "front_target": front_target,
        "rear_target": rear_target,
    }


@dataclass(frozen=True)
class ReplayRecord:
    path: Path
    row: int
    episode_id: int
    terrain_tile: int


class ReplayIndex:
    """Incrementally index atomic ``chunk_*.npz`` files."""

    def __init__(self, replay_dir: str | Path):
        self.replay_dir = Path(replay_dir).expanduser().resolve()
        self.records: list[ReplayRecord] = []
        self._known: set[Path] = set()
        self._indexed_rows: dict[str, int] = {}
        self._chunk_record_indices: list[np.ndarray] = []
        self._chunk_cache: OrderedDict[Path, dict[str, np.ndarray]] = OrderedDict()
        self._max_cached_chunks = 4
        self.refresh()

    def refresh(self) -> int:
        index_path = self.replay_dir / "replay_index.json"
        if not self._indexed_rows and index_path.is_file():
            payload = json.loads(index_path.read_text(encoding="utf-8"))
            self._indexed_rows = {
                str(item["file"]): int(item["rows"])
                for item in payload.get("chunks", ())
                if isinstance(item, dict) and int(item.get("rows", 0)) > 0
            }
        for path in sorted(self.replay_dir.glob("chunk_*.npz")):
            if path in self._known:
                continue
            indexed_rows = self._indexed_rows.get(path.name)
            if indexed_rows is None:
                with np.load(path, allow_pickle=False) as data:
                    rows = int(data["front_raw_d"].shape[0])
                    episodes = data.get("episode_id", np.zeros(rows, dtype=np.int64))
                    tiles = data.get("terrain_tile", np.zeros(rows, dtype=np.int64))
                self._indexed_rows[path.name] = rows
            else:
                rows = indexed_rows
                # Validation is chunk-separated for the compact index path;
                # this avoids reopening every compressed file just for IDs.
                episodes = np.full(rows, len(self._known), dtype=np.int64)
                tiles = np.zeros(rows, dtype=np.int64)
            start = len(self.records)
            self.records.extend(
                ReplayRecord(path, row, int(episodes[row]), int(tiles[row]))
                for row in range(rows)
            )
            self._chunk_record_indices.append(np.arange(start, start + rows, dtype=np.int64))
            self._known.add(path)
        if self._indexed_rows and not any(self.replay_dir.glob("chunk_*.npz.part")):
            index_path.write_text(json.dumps({
                "format": "s10_replay_index_v1",
                "chunks": [
                    {"file": path.name, "rows": rows}
                    for path, rows in sorted(
                        ((self.replay_dir / name, rows) for name, rows in self._indexed_rows.items()),
                        key=lambda item: item[0].name,
                    )
                ],
            }, indent=2), encoding="utf-8")
        return len(self.records)

    def __len__(self) -> int:
        return len(self.records)

    def sample_indices(
        self, eligible_indices: np.ndarray, size: int, rng: np.random.Generator
    ) -> np.ndarray:
        """Sample uniformly while keeping a batch local to one chunk.

        Chunks have equal size except the final one. Choosing a chunk in
        proportion to its eligible rows preserves record-level uniformity,
        while avoiding a full decompression for every sample in a batch.
        """

        eligible = np.asarray(eligible_indices, dtype=np.int64)
        if len(eligible) == 0 or size < 1:
            raise ValueError("eligible replay indices and batch size must be non-empty")
        mask = np.zeros(len(self.records), dtype=bool)
        mask[eligible] = True
        candidates = [indices[mask[indices]] for indices in self._chunk_record_indices]
        candidates = [indices for indices in candidates if len(indices)]
        weights = np.asarray([len(indices) for indices in candidates], dtype=np.float64)
        chunk = candidates[int(rng.choice(len(candidates), p=weights / weights.sum()))]
        return rng.choice(chunk, size=size, replace=len(chunk) < size).astype(np.int64, copy=False)

    def choose_chunk_indices(
        self, eligible_indices: np.ndarray, rng: np.random.Generator
    ) -> np.ndarray:
        eligible = np.asarray(eligible_indices, dtype=np.int64)
        mask = np.zeros(len(self.records), dtype=bool)
        mask[eligible] = True
        candidates = [indices[mask[indices]] for indices in self._chunk_record_indices]
        candidates = [indices for indices in candidates if len(indices)]
        if not candidates:
            raise ValueError("eligible replay indices do not contain a complete chunk")
        weights = np.asarray([len(indices) for indices in candidates], dtype=np.float64)
        return candidates[int(rng.choice(len(candidates), p=weights / weights.sum()))]

    def _load_chunk(self, path: Path) -> dict[str, np.ndarray]:
        cached = self._chunk_cache.get(path)
        if cached is not None:
            self._chunk_cache.move_to_end(path)
            return cached
        with np.load(path, allow_pickle=False) as data:
            cached = {
                "front_raw_d": np.asarray(data["front_raw_d"], dtype=np.float32).copy(),
                "rear_raw_d": np.asarray(data["rear_raw_d"], dtype=np.float32).copy(),
                "root_qpos": np.asarray(data["root_qpos"], dtype=np.float32).copy(),
            }
        self._chunk_cache[path] = cached
        self._chunk_cache.move_to_end(path)
        while len(self._chunk_cache) > self._max_cached_chunks:
            self._chunk_cache.popitem(last=False)
        return cached

    def load_raw_batch(self, indices: np.ndarray) -> dict[str, np.ndarray]:
        """Load raw arrays only; augmentation is performed on the learner GPU."""

        grouped: dict[Path, list[tuple[int, int]]] = {}
        for output_index, index in enumerate(np.asarray(indices, dtype=np.int64)):
            record = self.records[int(index)]
            grouped.setdefault(record.path, []).append((record.row, output_index))
        if not grouped:
            raise ValueError("cannot load an empty replay batch")
        output: dict[str, np.ndarray] = {}
        for path, rows in grouped.items():
            data = self._load_chunk(path)
            for key in ("front_raw_d", "rear_raw_d", "root_qpos"):
                values = np.stack([data[key][row] for row, _ in rows])
                if key not in output:
                    output[key] = np.empty((len(indices), *values.shape[1:]), dtype=values.dtype)
                for local_index, (_, output_index) in enumerate(rows):
                    output[key][output_index] = values[local_index]
        return output

    def load_batch(
        self,
        indices: np.ndarray,
        *,
        rng: np.random.Generator,
        noise_std_m: float,
        dropout_probability: float,
    ) -> dict[str, np.ndarray]:
        dirs = build_sensor_frame_directions()
        grouped: dict[Path, list[tuple[int, int]]] = {}
        for output_index, index in enumerate(np.asarray(indices, dtype=np.int64)):
            record = self.records[int(index)]
            grouped.setdefault(record.path, []).append((record.row, output_index))
        samples: list[dict[str, np.ndarray] | None] = [None] * len(indices)
        for path, rows in grouped.items():
            data = self._load_chunk(path)
            row_indices = np.asarray([row for row, _ in rows], dtype=np.int64)
            built = build_training_batch(
                data["front_raw_d"][row_indices], data["rear_raw_d"][row_indices],
                data["root_qpos"][row_indices], rng=rng, sensor_dirs=dirs,
                noise_std_m=noise_std_m, dropout_probability=dropout_probability,
            )
            for local_index, (_, output_index) in enumerate(rows):
                samples[output_index] = {key: value[local_index] for key, value in built.items()}
        if not samples or samples[0] is None:
            raise ValueError("cannot load an empty replay batch")
        return {key: np.stack([sample[key] for sample in samples if sample is not None]) for key in samples[0]}


def write_metadata(replay_dir: str | Path, *, terrain_seeds: list[int], num_envs: int) -> None:
    path = Path(replay_dir).expanduser().resolve()
    path.mkdir(parents=True, exist_ok=True)
    (path / "metadata.json").write_text(json.dumps({
        "format": "s10_mujoco_native_lidar_replay_v1",
        "native_shape_per_view": list(NATIVE_SHAPE),
        "encoder_shape_per_view": list(ENCODER_SHAPE),
        "distance_input_scale_m": ENCODER_DISTANCE_SCALE_M,
        "world_z_input_scale_m": ENCODER_WORLD_Z_SCALE_M,
        "horizontal_resolution_deg": 0.4,
        "front_rear_mode": True,
        "terrain_seeds": [int(seed) for seed in terrain_seeds],
        "num_envs": int(num_envs),
        "stores": ["front_raw_d", "rear_raw_d", "root_qpos", "episode_id", "terrain_tile"],
    }, indent=2), encoding="utf-8")
