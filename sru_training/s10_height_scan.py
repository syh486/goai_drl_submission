"""IsaacLab-compatible MuJoCo height scan and critic feature encoder.

IsaacLab reference contract:

* sensor origin is ``base`` + ``(0, 0, 20)``;
* rays are yaw-aligned, vertical downward, in a 10 m x 10 m grid;
* grid spacing is 0.2 m, therefore 51 x 51 samples;
* raw value is ``sensor_z - hit_z - 0.5`` and is clipped to [-5, 5];
* the fixed height VAE maps [51, 51] to [64, 7, 7].

The caller owns the MuJoCo stepping and must sample this feature at the same
policy boundary as the LiDAR/goal observations.  No disk replay is involved.
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
import torch


HEIGHT_GRID_SIZE = 51
HEIGHT_GRID_RESOLUTION = 0.2
HEIGHT_GRID_EXTENT = 10.0
HEIGHT_SENSOR_Z_OFFSET = 20.0
HEIGHT_VALUE_OFFSET = 0.5
HEIGHT_CLIP = (-5.0, 5.0)
HEIGHT_FEATURE_SHAPE = (64, 7, 7)


def build_yaw_aligned_downward_grid() -> np.ndarray:
    """Return body-frame ray origins/directions in IsaacLab row-major order."""

    coords = np.linspace(
        -HEIGHT_GRID_EXTENT / 2.0,
        HEIGHT_GRID_EXTENT / 2.0,
        HEIGHT_GRID_SIZE,
        dtype=np.float64,
    )
    xx, yy = np.meshgrid(coords, coords, indexing="ij")
    rays = np.zeros((HEIGHT_GRID_SIZE * HEIGHT_GRID_SIZE, 3), dtype=np.float64)
    rays[:, 0] = xx.reshape(-1)
    rays[:, 1] = yy.reshape(-1)
    rays[:, 2] = -1.0
    return rays


class MuJoCoHeightScan:
    """Raycast the equivalent privileged height image from an MjData state."""

    def __init__(
        self,
        model: mujoco.MjModel,
        *,
        mesh_geom_group: int = 0,
        max_range_m: float = 40.0,
    ):
        self.model = model
        self.rays_b = build_yaw_aligned_downward_grid()
        self.origins_w = np.empty((HEIGHT_GRID_SIZE * HEIGHT_GRID_SIZE, 3), dtype=np.float64)
        self.directions_w = np.empty_like(self.origins_w)
        self.geom_ids = np.empty((self.rays_b.shape[0],), dtype=np.int32)
        self.distances = np.empty((self.rays_b.shape[0],), dtype=np.float64)
        self.geom_group = np.zeros((6,), dtype=np.uint8)
        self.geom_group[mesh_geom_group] = 1
        self.max_range_m = float(max_range_m)
        self.base_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
        if self.base_body_id < 0:
            raise RuntimeError("MuJoCo height scan cannot find body 'base_link'")

    @staticmethod
    def _quat_wxyz_to_mat(quat: np.ndarray) -> np.ndarray:
        quat = np.asarray(quat, dtype=np.float64)
        quat = quat / max(float(np.linalg.norm(quat)), 1.0e-12)
        w, x, y, z = quat
        return np.asarray(
            (
                (1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)),
                (2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)),
                (2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)),
            ),
            dtype=np.float64,
        )

    def raw_scan_pose(self, data: mujoco.MjData, root_qpos: np.ndarray) -> np.ndarray:
        """Return the preprocessed [51, 51] scan consumed by the VAE."""

        root_qpos = np.asarray(root_qpos, dtype=np.float64)
        root_pos = root_qpos[:3]
        root_rot = self._quat_wxyz_to_mat(root_qpos[3:7])
        # IsaacLab's ray_alignment='yaw' means discard roll and pitch.
        yaw = np.arctan2(root_rot[1, 0], root_rot[0, 0])
        cy, sy = np.cos(yaw), np.sin(yaw)
        yaw_rot = np.asarray(((cy, -sy, 0.0), (sy, cy, 0.0), (0.0, 0.0, 1.0)))
        sensor_origin = root_pos + np.asarray((0.0, 0.0, HEIGHT_SENSOR_Z_OFFSET))
        self.origins_w[:] = sensor_origin
        self.directions_w[:] = self.rays_b @ yaw_rot.T
        mujoco.mj_multiRay(
            self.model,
            data,
            sensor_origin,
            np.ascontiguousarray(self.directions_w.reshape(-1)),
            self.geom_group,
            1,
            self.base_body_id,
            self.geom_ids,
            self.distances,
            None,
            self.rays_b.shape[0],
            self.max_range_m,
        )
        # IsaacLab stores ``data.pos_w`` at the attached body origin, while the
        # +20 m offset is applied only to ray starts. Therefore the equivalent
        # height is base_z - hit_z - 0.5, not sensor_z - hit_z - 0.5.
        distances = self.distances.copy()
        distances[(distances <= 0.0) | (distances > self.max_range_m)] = self.max_range_m
        hit_z = sensor_origin[2] - distances
        scan = root_pos[2] - hit_z.reshape(HEIGHT_GRID_SIZE, HEIGHT_GRID_SIZE) - HEIGHT_VALUE_OFFSET
        return np.clip(scan, HEIGHT_CLIP[0], HEIGHT_CLIP[1]).astype(np.float32)

    def raw_scan(self, data: mujoco.MjData) -> np.ndarray:
        return self.raw_scan_pose(data, data.qpos)


class HeightFeatureEncoder:
    """Load the original fixed height VAE and emit [64, 7, 7] features."""

    def __init__(self, checkpoint: str | Path | None = None, device: str | torch.device = "cpu"):
        from .heightscan_encoder import HeightScanFeatEncoder

        self.device = torch.device(device)
        checkpoint = Path(checkpoint or Path(__file__).parent / "assets" / "vae_heightscan3.pth")
        if not checkpoint.is_file():
            raise FileNotFoundError(f"height encoder checkpoint not found: {checkpoint}")
        self.checkpoint_path = checkpoint.expanduser().resolve()
        # The upstream class resolves its checkpoint relative to this package.
        self.model = HeightScanFeatEncoder(feature_dim=64).to(self.device).eval()
        loaded = torch.load(checkpoint, map_location=self.device, weights_only=True)
        self.model.encoder.load_state_dict(loaded, strict=True)

    @torch.inference_mode()
    def encode(self, scan: np.ndarray | torch.Tensor) -> torch.Tensor:
        value = torch.as_tensor(scan, dtype=torch.float32, device=self.device)
        if value.ndim == 2:
            value = value.unsqueeze(0)
        if value.ndim != 3 or tuple(value.shape[-2:]) != (HEIGHT_GRID_SIZE, HEIGHT_GRID_SIZE):
            raise ValueError(f"height scan must be [N,51,51], got {tuple(value.shape)}")
        output = self.model(value)
        if tuple(output.shape[1:]) != HEIGHT_FEATURE_SHAPE:
            raise RuntimeError(f"height encoder returned {tuple(output.shape)}, expected [N,64,7,7]")
        return output


def scan_and_encode(
    scanner: MuJoCoHeightScan,
    encoder: HeightFeatureEncoder,
    data: mujoco.MjData,
) -> torch.Tensor:
    return encoder.encode(scanner.raw_scan(data))
