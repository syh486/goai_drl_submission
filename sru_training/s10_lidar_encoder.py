"""In-memory S10 LiDAR adapter for the legacy 96x90 encoder contract.

The native official raycast is [96, 900] per view.  This adapter applies the
same deterministic min-pool used by ``lidar_project/s10_replay.py`` and then
calls the legacy model's ``extract_latents``.  It never writes a replay file.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

from deployment.common.lidar_geometry import (
    AIRY_VERTICAL_RAY_ANGLES,
    INVALID_RANGE_THRESHOLD_M,
    MAX_RANGE_M,
    MIN_RANGE_M,
    NATIVE_HEIGHT,
    S10_FRONT_POS,
    S10_FRONT_ROT_WXYZ,
    S10_REAR_POS,
    S10_REAR_ROT_WXYZ,
    build_sensor_frame_directions,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
LIDAR_RUNTIME_ROOT = Path(__file__).resolve().parent / "lidar_runtime"
DEFAULT_ENCODER_CHECKPOINT = (
    REPO_ROOT / "checkpoints/lidar_encoder_random_terrain_ft/best.pt"
)

ENCODER_SHAPE = (96, 90)
WORLD_Z_LIMIT_M = 3.0
# These are part of the original encoder training contract.  Keep them in
# one place so MuJoCo, deployment, and replay training cannot silently drift.
ENCODER_DISTANCE_SCALE_M = 10.0
ENCODER_WORLD_Z_SCALE_M = 3.0

def native_to_90(native: np.ndarray | torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Min-pool native 0.4 degree columns into 4 degree columns."""

    value = torch.as_tensor(native, dtype=torch.float32)
    if value.ndim == 2:
        value = value.unsqueeze(0)
    width = int(value.shape[-1])
    if value.shape[-2] != NATIVE_HEIGHT or width < 90 or width % 90 != 0:
        raise ValueError(f"Expected [N,96,90*K], got {tuple(value.shape)}")
    grouped = value.reshape(value.shape[:-1] + (90, width // 90))
    valid = (grouped > MIN_RANGE_M) & (grouped < INVALID_RANGE_THRESHOLD_M)
    safe = torch.where(valid, grouped, torch.full_like(grouped, MAX_RANGE_M))
    pooled, indices = safe.min(dim=-1)
    pooled = torch.where(valid.any(dim=-1), pooled, torch.full_like(pooled, MAX_RANGE_M))
    return pooled, indices


def gather_aux_at_min_distance(aux_native: np.ndarray | torch.Tensor, distance_native: np.ndarray | torch.Tensor) -> torch.Tensor:
    aux = torch.as_tensor(aux_native, dtype=torch.float32)
    distances = torch.as_tensor(distance_native, dtype=torch.float32)
    if aux.ndim == 2:
        aux = aux.unsqueeze(0)
    if distances.ndim == 2:
        distances = distances.unsqueeze(0)
    pooled, indices = native_to_90(distances)
    if tuple(aux.shape) != tuple(distances.shape):
        raise ValueError("auxiliary map and distance map must have equal shape")
    grouped = aux.reshape(aux.shape[:-1] + (90, aux.shape[-1] // 90))
    return torch.gather(grouped, -1, indices.unsqueeze(-1)).squeeze(-1)


def _quat_wxyz_to_rotmat(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float32)
    quat = quat / max(float(np.linalg.norm(quat)), 1.0e-8)
    w, x, y, z = quat
    return np.asarray(
        ((1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)),
         (2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)),
         (2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y))),
        dtype=np.float32,
    )


def world_z_native(
    distance_native: np.ndarray,
    root_qpos: np.ndarray,
    *,
    front: bool,
    sensor_dirs: np.ndarray | None = None,
) -> np.ndarray:
    root_qpos = np.asarray(root_qpos, dtype=np.float32)
    root_rot = _quat_wxyz_to_rotmat(root_qpos[3:7])
    sensor_pos = S10_FRONT_POS if front else S10_REAR_POS
    sensor_rot = S10_FRONT_ROT_WXYZ if front else S10_REAR_ROT_WXYZ
    dirs = build_sensor_frame_directions() if sensor_dirs is None else sensor_dirs
    dirs_world = np.einsum("ij,hwj->hwi", root_rot @ _quat_wxyz_to_rotmat(sensor_rot), dirs)
    sensor_pos_w = root_qpos[:3] + root_rot @ sensor_pos
    value = sensor_pos_w[2] + np.asarray(distance_native, dtype=np.float32) * dirs_world[..., 2]
    valid = (np.asarray(distance_native) < INVALID_RANGE_THRESHOLD_M) & (np.asarray(distance_native) > MIN_RANGE_M)
    return np.where(valid, np.clip(value, -WORLD_Z_LIMIT_M, WORLD_Z_LIMIT_M), 0.0).astype(np.float32)


class S10LegacyLidarEncoder:
    """Load the already-trained stage2 model for inference only."""

    def __init__(
        self,
        checkpoint: str | Path = DEFAULT_ENCODER_CHECKPOINT,
        device: str | torch.device = "cpu",
    ):
        checkpoint = Path(checkpoint).expanduser().resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"LiDAR encoder checkpoint not found: {checkpoint}")
        if str(LIDAR_RUNTIME_ROOT) not in sys.path:
            sys.path.insert(0, str(LIDAR_RUNTIME_ROOT))
        from inference_utils import load_checkpoint_model

        self.device = torch.device(device)
        self.checkpoint_path = checkpoint
        self.model, self.model_name, _ = load_checkpoint_model(checkpoint, device=self.device)
        self.model.eval()

    @torch.inference_mode()
    def encode_native(
        self,
        front_native: np.ndarray | torch.Tensor,
        rear_native: np.ndarray | torch.Tensor,
    ) -> torch.Tensor:
        front_d, _ = native_to_90(front_native)
        rear_d, _ = native_to_90(rear_native)
        front_d = front_d.to(self.device)
        rear_d = rear_d.to(self.device)
        # The legacy two-channel input is distance plus world-z.  For the first
        # real closed-loop smoke, derive the second channel from caller-provided
        # geometry in a later backend; zeros are rejected by the explicit flag in
        # encode_distance_only so it cannot be mistaken for final training input.
        raise RuntimeError(
            "encode_native requires world-z maps; call encode_maps(front_d, rear_d, front_z, rear_z)."
        )

    @torch.inference_mode()
    def encode_maps(
        self,
        front_distance: np.ndarray | torch.Tensor,
        rear_distance: np.ndarray | torch.Tensor,
        front_world_z: np.ndarray | torch.Tensor,
        rear_world_z: np.ndarray | torch.Tensor,
    ) -> torch.Tensor:
        def batch(value: np.ndarray | torch.Tensor, name: str) -> torch.Tensor:
            result = torch.as_tensor(value, dtype=torch.float32, device=self.device)
            if result.ndim == 2:
                result = result.unsqueeze(0)
            if result.ndim != 3 or tuple(result.shape[-2:]) != ENCODER_SHAPE:
                raise ValueError(f"{name} must have shape [N,96,90] or [96,90], got {tuple(result.shape)}")
            return result

        front_d = batch(front_distance, "front_distance")
        rear_d = batch(rear_distance, "rear_distance")
        front_z = batch(front_world_z, "front_world_z")
        rear_z = batch(rear_world_z, "rear_world_z")
        if not (front_d.shape == rear_d.shape == front_z.shape == rear_z.shape):
            raise ValueError(
                "front/rear distance and world-z maps must have identical batched shapes: "
                f"{tuple(front_d.shape)}, {tuple(rear_d.shape)}, "
                f"{tuple(front_z.shape)}, {tuple(rear_z.shape)}"
            )

        # The checkpoint was trained with normalized inputs but metric-space
        # reconstructions.  The previous MuJoCo path omitted this conversion.
        front_input = torch.stack(
            (front_d / ENCODER_DISTANCE_SCALE_M, front_z / ENCODER_WORLD_Z_SCALE_M), dim=1
        )
        rear_input = torch.stack(
            (rear_d / ENCODER_DISTANCE_SCALE_M, rear_z / ENCODER_WORLD_Z_SCALE_M), dim=1
        )
        output = self.model.extract_latents(front_input=front_input, rear_input=rear_input)
        latent = output["fused_latent"]
        if tuple(latent.shape[1:]) != (64, 5, 8):
            raise RuntimeError(f"legacy encoder returned {tuple(latent.shape)}, expected [N,64,5,8]")
        return latent
