#!/usr/bin/env bash
set -eo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
source "${ROOT}/deployment/scripts/hardware/hardware_env.sh"
cd "${ROOT}"

PYTHON="${S10_PYTHON:-python3}"
MAP_DIR="${S10_ROUTE_MAP_DIR:-${ROOT}/deployment/maps/current}"
CONFIG="${ROOT}/deployment/config/hardware_localization.yaml"

[[ -f "${CONFIG}" ]] || { echo "Missing localization config: ${CONFIG}" >&2; exit 2; }
[[ -f "${MAP_DIR}/localization_map_manifest.json" ]] || {
  echo "Missing route-map manifest: ${MAP_DIR}/localization_map_manifest.json" >&2
  exit 2
}

export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
"${PYTHON}" -m py_compile \
  deployment/localization/continuous_map_localization.py \
  deployment/navigation/ros2_node.py

S10_PREFLIGHT_MAP_DIR="${MAP_DIR}" "${PYTHON}" - <<'PY'
import json
import os
from pathlib import Path

import numpy as np
import yaml

import rclpy
from drdds.msg import ImuData, JointsData, Steer
from sensor_msgs.msg import PointCloud2

from deployment.localization.continuous_map_localization import (
    ContinuousLocalizationConfig,
    ContinuousMapLocalizer,
)

root = Path.cwd()
config = yaml.safe_load(
    (root / "deployment/config/hardware_localization.yaml").read_text()
)
runtime = config["runtime"]
localization = config["localization"]
map_config = config["map_localization"]
map_dir = Path(os.environ["S10_PREFLIGHT_MAP_DIR"]).expanduser().resolve()

assert runtime["enable_motion"] is False
assert runtime["publish_legacy_cmd_vel"] is False
assert runtime["publish_dds_steer"] is False
assert localization["max_cloud_points_per_lidar"] == 4000
assert localization["max_registration_points"] == 6000

manifest = json.loads(
    (map_dir / "localization_map_manifest.json").read_text()
)
assert manifest["map_type"] == "ordered_topometric_route_submaps"
assert int(manifest["submap_count"]) == len(manifest["submaps"])

profile = ContinuousLocalizationConfig(
    registration_target=str(map_config["registration_target"]),
    registration_backend=str(map_config["registration_backend"]),
    beam_width=int(map_config["beam_width"]),
    candidate_count=int(map_config["candidate_count"]),
    odometry_recovery_candidate_count=int(
        map_config["odometry_recovery_candidate_count"]
    ),
    query_voxel_m=float(map_config["query_voxel_m"]),
    fine_query_voxel_m=float(map_config["fine_query_voxel_m"]),
    fine_max_query_points=int(map_config["fine_max_query_points"]),
    query_submap_updates=int(map_config["query_submap_updates"]),
    max_icp_iterations=int(map_config["max_icp_iterations"]),
    icp_threads=int(map_config["icp_threads"]),
    fine_max_icp_iterations=int(map_config["fine_max_icp_iterations"]),
    fine_max_correspondence_m=float(map_config["fine_max_correspondence_m"]),
    fine_min_fitness=float(map_config["fine_min_fitness"]),
    fine_max_rmse_m=float(map_config["fine_max_rmse_m"]),
    odometry_tracking_candidate_count=int(
        map_config["odometry_tracking_candidate_count"]
    ),
    recovery_odometry_radius_submaps=int(
        map_config["recovery_odometry_radius_submaps"]
    ),
    route_lag_cost=float(map_config["route_lag_cost"]),
    min_fitness=float(map_config["min_fitness"]),
    max_rmse_m=float(map_config["max_rmse_m"]),
    continuity_min_fitness=float(map_config["continuity_min_fitness"]),
    continuity_max_rmse_m=float(map_config["continuity_max_rmse_m"]),
    temporal_fusion_enabled=bool(map_config["temporal_fusion_enabled"]),
    max_route_consistent_relocalization_translation_m=float(
        map_config["max_route_consistent_relocalization_translation_m"]
    ),
    max_route_consistent_relocalization_yaw_deg=float(
        map_config["max_route_consistent_relocalization_yaw_deg"]
    ),
)
localizer = ContinuousMapLocalizer(
    map_dir, initial_map_from_body=None, initial_odom_from_body=np.eye(4),
    config=profile,
)
first = localizer.submaps[0]
with np.load(first["file"], allow_pickle=False) as payload:
    scan = np.asarray(payload["points_anchor_m"], dtype=np.float64)
result = localizer.update(
    scan, np.eye(4), traveled_distance_m=1.1,
    allow_large_relocalization=True,
)
assert result.observation_accepted
assert result.selected_route_index == 0
print(
    "ROUTE_MAP_RUNTIME_PREFLIGHT_OK",
    f"submaps={len(localizer.submaps)}",
    f"fitness={result.fitness:.6f}",
    f"rmse_m={result.rmse_m:.3e}",
)
print("MOTION_PUBLISHERS_DISABLED_BY_CONFIG")
PY

if [[ -f tests/deployment/pointcloud_early_sampling.py ]]; then
  "${PYTHON}" tests/deployment/pointcloud_early_sampling.py
fi
if [[ -f tests/deployment/async_map_localization.py ]]; then
  "${PYTHON}" tests/deployment/async_map_localization.py
fi

echo "ROUTE_MAP_LOCALIZATION_RUNTIME_PREFLIGHT_OK"
