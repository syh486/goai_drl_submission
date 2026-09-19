"""CUDA raycasts against static MuJoCo group-zero terrain using Warp."""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np
import torch
import warp as wp

from .s10_lidar_encoder import build_sensor_frame_directions


@wp.func
def _mesh_and_plane_distance(
    mesh: wp.uint64,
    origin: wp.vec3,
    direction: wp.vec3,
    max_range: float,
    use_plane: int,
    plane_position: wp.array(dtype=wp.vec3),
    plane_axis_x: wp.array(dtype=wp.vec3),
    plane_axis_y: wp.array(dtype=wp.vec3),
    plane_normal: wp.array(dtype=wp.vec3),
    plane_size: wp.array(dtype=wp.vec2),
) -> float:
    query = wp.mesh_query_ray(mesh, origin, direction, max_range)
    distance = max_range
    if query.result:
        distance = query.t

    if use_plane != 0:
        normal = plane_normal[0]
        denominator = wp.dot(direction, normal)
        # MuJoCo planes are one-sided: rays only hit their front face.
        if denominator < -1.0e-8:
            plane_distance = wp.dot(plane_position[0] - origin, normal) / denominator
            if plane_distance >= 0.0 and plane_distance <= distance:
                hit = origin + direction * plane_distance
                relative = hit - plane_position[0]
                size = plane_size[0]
                within_x = size[0] <= 0.0 or wp.abs(wp.dot(relative, plane_axis_x[0])) <= size[0]
                within_y = size[1] <= 0.0 or wp.abs(wp.dot(relative, plane_axis_y[0])) <= size[1]
                if within_x and within_y:
                    distance = plane_distance
    return distance


@wp.kernel
def _lidar_raycast_kernel(
    mesh: wp.uint64,
    root_positions: wp.array(dtype=wp.vec3),
    root_quaternions: wp.array(dtype=wp.quat),
    sensor_positions: wp.array(dtype=wp.vec3),
    sensor_quaternions: wp.array(dtype=wp.quat),
    sensor_directions: wp.array(dtype=wp.vec3),
    rays_per_view: int,
    max_range: float,
    use_plane: int,
    plane_position: wp.array(dtype=wp.vec3),
    plane_axis_x: wp.array(dtype=wp.vec3),
    plane_axis_y: wp.array(dtype=wp.vec3),
    plane_normal: wp.array(dtype=wp.vec3),
    plane_size: wp.array(dtype=wp.vec2),
    distances: wp.array(dtype=float),
    world_z: wp.array(dtype=float),
):
    tid = wp.tid()
    ray_index = tid % rays_per_view
    view_index = (tid // rays_per_view) % 2
    env_index = tid // (2 * rays_per_view)

    root_q = root_quaternions[env_index]
    origin = root_positions[env_index] + wp.quat_rotate(
        root_q, sensor_positions[view_index]
    )
    direction = wp.quat_rotate(
        root_q,
        wp.quat_rotate(sensor_quaternions[view_index], sensor_directions[ray_index]),
    )
    distance = _mesh_and_plane_distance(
        mesh,
        origin,
        direction,
        max_range,
        use_plane,
        plane_position,
        plane_axis_x,
        plane_axis_y,
        plane_normal,
        plane_size,
    )
    distances[tid] = distance
    if distance < max_range:
        world_z[tid] = origin[2] + distance * direction[2]
    else:
        world_z[tid] = 0.0


@wp.kernel
def _world_raycast_kernel(
    mesh: wp.uint64,
    origins: wp.array(dtype=wp.vec3),
    directions: wp.array(dtype=wp.vec3),
    max_range: float,
    use_plane: int,
    plane_position: wp.array(dtype=wp.vec3),
    plane_axis_x: wp.array(dtype=wp.vec3),
    plane_axis_y: wp.array(dtype=wp.vec3),
    plane_normal: wp.array(dtype=wp.vec3),
    plane_size: wp.array(dtype=wp.vec2),
    distances: wp.array(dtype=float),
):
    tid = wp.tid()
    distances[tid] = _mesh_and_plane_distance(
        mesh,
        origins[tid],
        directions[tid],
        max_range,
        use_plane,
        plane_position,
        plane_axis_x,
        plane_axis_y,
        plane_normal,
        plane_size,
    )


@dataclass(frozen=True)
class StaticTerrainGeometry:
    vertices: np.ndarray
    faces: np.ndarray
    plane_position: np.ndarray
    plane_axes: np.ndarray
    plane_size: np.ndarray
    geom_counts: dict[str, int]
    hfield_patch_count: int

    @property
    def has_plane(self) -> bool:
        return self.geom_counts.get("plane", 0) == 1


def _quat_wxyz_to_xyzw(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float32)
    return quat[..., (1, 2, 3, 0)]


def _hfield_top_mesh(
    heights: np.ndarray,
    size_x: float,
    size_y: float,
    *,
    block_cells: int = 20,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Triangulate an hfield exactly while merging only constant regions."""

    heights = np.asarray(heights, dtype=np.float32)
    nrow, ncol = heights.shape
    x = np.linspace(-size_x, size_x, ncol, dtype=np.float32)
    y = np.linspace(-size_y, size_y, nrow, dtype=np.float32)
    vertices: list[np.ndarray] = []
    faces: list[np.ndarray] = []
    vertex_offset = 0
    patch_count = 0

    def append_patch(r0: int, r1: int, c0: int, c1: int) -> None:
        nonlocal vertex_offset, patch_count
        region = heights[r0:r1 + 1, c0:c1 + 1]
        is_flat = bool(np.min(region) == np.max(region))
        if not is_flat and (r1 - r0 > 1 or c1 - c0 > 1):
            rows = r1 - r0
            cols = c1 - c0
            if rows >= cols and rows > 1:
                middle = r0 + rows // 2
                append_patch(r0, middle, c0, c1)
                append_patch(middle, r1, c0, c1)
            else:
                middle = c0 + cols // 2
                append_patch(r0, r1, c0, middle)
                append_patch(r0, r1, middle, c1)
            return

        corners = np.asarray(
            (
                (x[c0], y[r0], heights[r0, c0]),
                (x[c1], y[r0], heights[r0, c1]),
                (x[c1], y[r1], heights[r1, c1]),
                (x[c0], y[r1], heights[r1, c0]),
            ),
            dtype=np.float32,
        )
        vertices.append(corners)
        faces.append(
            np.asarray(((0, 1, 2), (0, 2, 3)), dtype=np.int32) + vertex_offset
        )
        vertex_offset += 4
        patch_count += 1

    for r0 in range(0, nrow - 1, block_cells):
        r1 = min(r0 + block_cells, nrow - 1)
        for c0 in range(0, ncol - 1, block_cells):
            c1 = min(c0 + block_cells, ncol - 1)
            append_patch(r0, r1, c0, c1)

    return np.concatenate(vertices), np.concatenate(faces), patch_count


def _hfield_closed_mesh(
    heights: np.ndarray,
    size: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Create MuJoCo's top surface, bottom and four hfield side walls."""

    size_x, size_y, _, size_bottom = map(float, size)
    top_vertices, top_faces, patch_count = _hfield_top_mesh(
        heights, size_x, size_y
    )
    vertices: list[np.ndarray] = [top_vertices]
    faces: list[np.ndarray] = [top_faces]
    vertex_offset = top_vertices.shape[0]
    bottom_z = -size_bottom

    bottom = np.asarray(
        (
            (-size_x, -size_y, bottom_z),
            (-size_x, size_y, bottom_z),
            (size_x, size_y, bottom_z),
            (size_x, -size_y, bottom_z),
        ),
        dtype=np.float32,
    )
    vertices.append(bottom)
    faces.append(
        np.asarray(((0, 1, 2), (0, 2, 3)), dtype=np.int32) + vertex_offset
    )
    vertex_offset += 4

    def append_side(points: np.ndarray) -> None:
        nonlocal vertex_offset
        for start in range(points.shape[0] - 1):
            p0 = points[start]
            p1 = points[start + 1]
            corners = np.asarray(
                (p0, p1, (p1[0], p1[1], bottom_z), (p0[0], p0[1], bottom_z)),
                dtype=np.float32,
            )
            vertices.append(corners)
            faces.append(
                np.asarray(((0, 1, 2), (0, 2, 3)), dtype=np.int32) + vertex_offset
            )
            vertex_offset += 4

    nrow, ncol = heights.shape
    x = np.linspace(-size_x, size_x, ncol, dtype=np.float32)
    y = np.linspace(-size_y, size_y, nrow, dtype=np.float32)
    append_side(np.column_stack((x, np.full_like(x, -size_y), heights[0])))
    append_side(np.column_stack((x[::-1], np.full_like(x, size_y), heights[-1, ::-1])))
    append_side(np.column_stack((np.full_like(y, -size_x), y[::-1], heights[::-1, 0])))
    append_side(np.column_stack((np.full_like(y, size_x), y, heights[:, -1])))

    return np.concatenate(vertices), np.concatenate(faces), patch_count


def _box_mesh(size: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    sx, sy, sz = map(float, size)
    vertices = np.asarray(
        (
            (-sx, -sy, -sz), (sx, -sy, -sz), (sx, sy, -sz), (-sx, sy, -sz),
            (-sx, -sy, sz), (sx, -sy, sz), (sx, sy, sz), (-sx, sy, sz),
        ),
        dtype=np.float32,
    )
    faces = np.asarray(
        (
            (0, 2, 1), (0, 3, 2), (4, 5, 6), (4, 6, 7),
            (0, 1, 5), (0, 5, 4), (1, 2, 6), (1, 6, 5),
            (2, 3, 7), (2, 7, 6), (3, 0, 4), (3, 4, 7),
        ),
        dtype=np.int32,
    )
    return vertices, faces


def _transform_vertices(
    local_vertices: np.ndarray,
    rotation: np.ndarray,
    position: np.ndarray,
) -> np.ndarray:
    return np.asarray(local_vertices @ rotation.T + position, dtype=np.float32)


def _build_static_terrain(model: mujoco.MjModel) -> StaticTerrainGeometry:
    """Merge supported static group-zero MuJoCo geoms in world coordinates."""

    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    vertices: list[np.ndarray] = []
    faces: list[np.ndarray] = []
    vertex_offset = 0
    geom_counts = {"mesh": 0, "hfield": 0, "box": 0, "plane": 0}
    plane_position = np.zeros(3, dtype=np.float32)
    plane_axes = np.eye(3, dtype=np.float32)
    plane_size = np.zeros(2, dtype=np.float32)
    hfield_patch_count = 0

    for geom_id in range(model.ngeom):
        if int(model.geom_group[geom_id]) != 0:
            continue
        geom_type = int(model.geom_type[geom_id])
        rotation = np.asarray(data.geom_xmat[geom_id], dtype=np.float32).reshape(3, 3)
        position = np.asarray(data.geom_xpos[geom_id], dtype=np.float32)

        if geom_type == int(mujoco.mjtGeom.mjGEOM_PLANE):
            geom_counts["plane"] += 1
            if geom_counts["plane"] > 1:
                raise NotImplementedError("Warp terrain supports at most one group-zero plane")
            plane_position = position.copy()
            plane_axes = rotation.copy()
            plane_size = np.asarray(model.geom_size[geom_id, :2], dtype=np.float32).copy()
            continue

        if geom_type == int(mujoco.mjtGeom.mjGEOM_MESH):
            geom_counts["mesh"] += 1
            mesh_id = int(model.geom_dataid[geom_id])
            vert_adr = int(model.mesh_vertadr[mesh_id])
            vert_num = int(model.mesh_vertnum[mesh_id])
            face_adr = int(model.mesh_faceadr[mesh_id])
            face_num = int(model.mesh_facenum[mesh_id])
            local_vertices = np.asarray(
                model.mesh_vert[vert_adr:vert_adr + vert_num], dtype=np.float32
            )
            local_faces = np.asarray(
                model.mesh_face[face_adr:face_adr + face_num], dtype=np.int32
            )
        elif geom_type == int(mujoco.mjtGeom.mjGEOM_HFIELD):
            geom_counts["hfield"] += 1
            hfield_id = int(model.geom_dataid[geom_id])
            nrow = int(model.hfield_nrow[hfield_id])
            ncol = int(model.hfield_ncol[hfield_id])
            address = int(model.hfield_adr[hfield_id])
            normalized = np.asarray(
                model.hfield_data[address:address + nrow * ncol], dtype=np.float32
            ).reshape(nrow, ncol)
            hfield_size = np.asarray(model.hfield_size[hfield_id], dtype=np.float32)
            heights = normalized * hfield_size[2]
            local_vertices, local_faces, patches = _hfield_closed_mesh(
                heights, hfield_size
            )
            hfield_patch_count += patches
        elif geom_type == int(mujoco.mjtGeom.mjGEOM_BOX):
            geom_counts["box"] += 1
            local_vertices, local_faces = _box_mesh(model.geom_size[geom_id])
        else:
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
            raise NotImplementedError(
                f"unsupported group-zero terrain geom {name!r} of type {geom_type}"
            )

        vertices.append(_transform_vertices(local_vertices, rotation, position))
        faces.append(local_faces + vertex_offset)
        vertex_offset += local_vertices.shape[0]

    if not vertices:
        raise RuntimeError("Warp terrain needs at least one group-zero mesh, hfield, or box")

    return StaticTerrainGeometry(
        vertices=np.ascontiguousarray(np.concatenate(vertices), dtype=np.float32),
        faces=np.ascontiguousarray(np.concatenate(faces), dtype=np.int32),
        plane_position=plane_position,
        plane_axes=plane_axes,
        plane_size=plane_size,
        geom_counts=geom_counts,
        hfield_patch_count=hfield_patch_count,
    )


class WarpStaticTerrainLidar:
    """Batch front/rear LiDAR raycasts on a static CUDA terrain BVH."""

    def __init__(
        self,
        model: mujoco.MjModel,
        *,
        horizontal_samples: int = 270,
        device: str = "cuda:0",
        max_range_m: float = 10.0,
    ) -> None:
        if horizontal_samples < 90 or horizontal_samples % 90 != 0:
            raise ValueError("horizontal_samples must be a multiple of 90")
        if not wp.is_cuda_available():
            raise RuntimeError("Warp CUDA is unavailable")
        self.device = device
        self.horizontal_samples = int(horizontal_samples)
        self.rays_per_view = 96 * self.horizontal_samples
        self.max_range_m = float(max_range_m)

        self.geometry = _build_static_terrain(model)
        with wp.ScopedDevice(self.device):
            self.mesh_points = wp.array(
                self.geometry.vertices, dtype=wp.vec3, device=self.device
            )
            self.mesh_indices = wp.array(
                self.geometry.faces.reshape(-1), dtype=int, device=self.device
            )
            self.mesh = wp.Mesh(
                points=self.mesh_points,
                indices=self.mesh_indices,
                bvh_constructor="sah",
            )
            self.plane_position = wp.array(
                self.geometry.plane_position[None], dtype=wp.vec3, device=self.device
            )
            self.plane_axis_x = wp.array(
                self.geometry.plane_axes[:, 0][None], dtype=wp.vec3, device=self.device
            )
            self.plane_axis_y = wp.array(
                self.geometry.plane_axes[:, 1][None], dtype=wp.vec3, device=self.device
            )
            self.plane_normal = wp.array(
                self.geometry.plane_axes[:, 2][None], dtype=wp.vec3, device=self.device
            )
            self.plane_size = wp.array(
                self.geometry.plane_size[None], dtype=wp.vec2, device=self.device
            )
            directions = build_sensor_frame_directions(self.horizontal_samples).reshape(-1, 3)
            self.directions = wp.array(
                directions, dtype=wp.vec3, device=self.device
            )
            self.sensor_positions = wp.array(
                ((0.22341, 0.0, -0.0001), (-0.22341, 0.0, -0.0001)),
                dtype=wp.vec3,
                device=self.device,
            )
            self.sensor_quaternions = wp.array(
                _quat_wxyz_to_xyzw(
                    np.asarray(
                        ((0.00084463, 0.7071065, -0.00028154, 0.7071065),
                         (0.70703477, 0.0, -0.70717879, 0.0)),
                        dtype=np.float32,
                    )
                ),
                dtype=wp.quat,
                device=self.device,
            )

    def _plane_inputs(self) -> tuple[object, ...]:
        return (
            int(self.geometry.has_plane),
            self.plane_position,
            self.plane_axis_x,
            self.plane_axis_y,
            self.plane_normal,
            self.plane_size,
        )

    def raycast_world(
        self,
        origins: torch.Tensor,
        directions: torch.Tensor,
        *,
        max_range_m: float | None = None,
    ) -> torch.Tensor:
        """Cast arbitrary normalized world-frame rays for equivalence tests."""

        origins = torch.as_tensor(origins, dtype=torch.float32, device=self.device).contiguous()
        directions = torch.as_tensor(directions, dtype=torch.float32, device=self.device).contiguous()
        if origins.ndim != 2 or origins.shape[1] != 3 or origins.shape != directions.shape:
            raise ValueError("origins and directions must both have shape [N,3]")
        distances = torch.empty(origins.shape[0], dtype=torch.float32, device=self.device)
        wp.launch(
            _world_raycast_kernel,
            dim=origins.shape[0],
            inputs=(
                self.mesh.id,
                wp.from_torch(origins, dtype=wp.vec3),
                wp.from_torch(directions, dtype=wp.vec3),
                self.max_range_m if max_range_m is None else float(max_range_m),
                *self._plane_inputs(),
                wp.from_torch(distances),
            ),
            device=self.device,
        )
        return distances

    def capture_with_world_z(
        self, root_qpos: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        root_qpos = torch.as_tensor(
            root_qpos, dtype=torch.float32, device=self.device
        ).contiguous()
        if root_qpos.ndim != 2 or root_qpos.shape[1] != 7:
            raise ValueError(f"root_qpos must be [N,7], got {tuple(root_qpos.shape)}")
        positions = root_qpos[:, :3].contiguous()
        quaternions = root_qpos[:, (4, 5, 6, 3)].contiguous()
        distances = torch.empty(
            (root_qpos.shape[0], 2, self.rays_per_view),
            dtype=torch.float32,
            device=self.device,
        )
        world_z = torch.empty_like(distances)
        wp.launch(
            _lidar_raycast_kernel,
            dim=distances.numel(),
            inputs=(
                self.mesh.id,
                wp.from_torch(positions, dtype=wp.vec3),
                wp.from_torch(quaternions, dtype=wp.quat),
                self.sensor_positions,
                self.sensor_quaternions,
                self.directions,
                self.rays_per_view,
                self.max_range_m,
                *self._plane_inputs(),
                wp.from_torch(distances.reshape(-1)),
                wp.from_torch(world_z.reshape(-1)),
            ),
            device=self.device,
        )
        shaped = distances.reshape(
            root_qpos.shape[0], 2, 96, self.horizontal_samples
        )
        shaped_z = world_z.reshape(
            root_qpos.shape[0], 2, 96, self.horizontal_samples
        )
        valid = (shaped > 0.2) & (shaped < 9.9)
        shaped_z = torch.where(valid, shaped_z.clamp(-3.0, 3.0), 0.0)
        return shaped[:, 0], shaped[:, 1], shaped_z[:, 0], shaped_z[:, 1]

    def capture(self, root_qpos: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        front, rear, _, _ = self.capture_with_world_z(root_qpos)
        return front, rear
