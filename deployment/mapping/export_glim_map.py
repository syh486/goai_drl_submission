"""Export a GLIM dump as a voxelized XYZ/I PCD and summarize its closure."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _read_world_origin(data_path: Path) -> np.ndarray:
    lines = data_path.read_text(encoding="utf-8").splitlines()
    try:
        start = lines.index("T_world_origin: ") + 1
    except ValueError as error:
        raise ValueError(f"T_world_origin missing from {data_path}") from error
    matrix = np.asarray([
        [float(value) for value in line.split()] for line in lines[start:start + 4]
    ])
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError(f"invalid T_world_origin in {data_path}")
    return matrix


def _voxel_downsample(points: np.ndarray, voxel_size_m: float) -> np.ndarray:
    if voxel_size_m <= 0.0:
        raise ValueError("voxel size must be positive")
    cloud = np.asarray(points, dtype=np.float64)
    keys = np.floor(cloud / voxel_size_m).astype(np.int64)
    _, indices = np.unique(keys, axis=0, return_index=True)
    indices.sort()
    return np.ascontiguousarray(cloud[indices], dtype=np.float32)


def _write_xyzi_pcd(path: Path, points: np.ndarray) -> None:
    cloud = np.zeros(
        len(points),
        dtype=np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("intensity", "<f4")]),
    )
    cloud["x"], cloud["y"], cloud["z"] = np.asarray(points, dtype=np.float32).T
    header = (
        "# .PCD v0.7 - Point Cloud Data file format\n"
        "VERSION 0.7\n"
        "FIELDS x y z intensity\n"
        "SIZE 4 4 4 4\n"
        "TYPE F F F F\n"
        "COUNT 1 1 1 1\n"
        f"WIDTH {len(cloud)}\n"
        "HEIGHT 1\n"
        "VIEWPOINT 0 0 0 1 0 0 0\n"
        f"POINTS {len(cloud)}\n"
        "DATA binary\n"
    ).encode("ascii")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        stream.write(header)
        stream.write(cloud.tobytes())


def export_glim_map(dump_dir: Path, output_pcd: Path, voxel_size_m: float) -> dict[str, object]:
    root = dump_dir.expanduser().resolve()
    graph_lines = (root / "graph.txt").read_text(encoding="utf-8").splitlines()
    if not graph_lines or not graph_lines[0].startswith("num_submaps:"):
        raise ValueError("invalid GLIM graph.txt")
    submap_count = int(graph_lines[0].split(":", 1)[1])
    transformed = []
    raw_points = 0
    for index in range(submap_count):
        submap = root / f"{index:06d}"
        # gtsam_points compact storage is Eigen::Vector3f without a header.
        points = np.fromfile(submap / "points_compact.bin", dtype=np.float32)
        if len(points) % 3:
            raise ValueError(f"invalid compact point file: {submap}")
        points = points.reshape(-1, 3)
        points = points[np.isfinite(points).all(axis=1)]
        transform = _read_world_origin(submap / "data.txt")
        transformed.append(points @ transform[:3, :3].T + transform[:3, 3])
        raw_points += len(points)
    map_points = _voxel_downsample(np.concatenate(transformed), voxel_size_m)
    output = output_pcd.expanduser().resolve()
    _write_xyzi_pcd(output, map_points)

    trajectory = np.loadtxt(root / "traj_lidar.txt")
    closure_xyz_m = float(np.linalg.norm(trajectory[-1, 1:4] - trajectory[0, 1:4]))
    report = {
        "schema_version": 1,
        "glim_dump": str(root),
        "output_pcd": str(output),
        "submaps": submap_count,
        "raw_points": raw_points,
        "voxel_points": len(map_points),
        "voxel_size_m": voxel_size_m,
        "trajectory_frames": len(trajectory),
        "trajectory_closure_xyz_m": closure_xyz_m,
        "closure_qualified": closure_xyz_m <= 2.0,
    }
    report_path = output.with_suffix(output.suffix + ".report.json")
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dump_dir", type=Path)
    parser.add_argument("output_pcd", type=Path)
    parser.add_argument("--voxel-size-m", type=float, default=0.20)
    args = parser.parse_args()
    report = export_glim_map(args.dump_dir, args.output_pcd, args.voxel_size_m)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
