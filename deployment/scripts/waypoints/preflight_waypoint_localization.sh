#!/usr/bin/env bash
set -eo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
source "${ROOT}/deployment/scripts/hardware/hardware_env.sh"
cd "${ROOT}"

PYTHON="${S10_PYTHON:-python3}"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

run_check() {
  local name="$1"
  shift
  printf '\n[%s]\n' "${name}"
  "$@"
}

run_check "Repository models and frozen map" \
  "${PYTHON}" scripts/verify_install.py

for test_file in \
  tests/deployment/waypoint_collection.py \
  tests/deployment/lidar_snapshots.py \
  tests/deployment/start_alignment.py \
  tests/deployment/rebind_waypoints.py \
  tests/deployment/local_odometry_filter.py \
  tests/deployment/continuous_map_progression.py \
  tests/deployment/hardware_navigation_core.py \
  tests/deployment/pointcloud_early_sampling.py \
  tests/deployment/topometric_route_map.py; do
  run_check "${test_file}" "${PYTHON}" "${test_file}"
done

run_check "ROS2 Python imports" "${PYTHON}" -c \
  'import rclpy; import deployment.navigation.ros2_node; import deployment.waypoints.ros2_collector'

if [[ ! -f "${ROOT}/install/setup.bash" ]]; then
  echo "ROS2 workspace overlay is missing: ${ROOT}/install/setup.bash" >&2
  echo "Build it before field use with:" >&2
  echo "  colcon build --symlink-install --packages-select drdds rslidar_msg dual_airy_merger s10_sdk_deploy" >&2
  exit 2
fi
source "${ROOT}/install/setup.bash"

run_check "Generated S10 ROS2 interfaces" "${PYTHON}" -c \
  'from drdds.msg import ImuData, JointsData, Steer; print("drdds interfaces OK")'

printf '\n[Waypoint collector ROS transport]\n'
env -u FASTRTPS_DEFAULT_PROFILES_FILE ROS_LOCALHOST_ONLY=1 \
  "${PYTHON}" tests/deployment/waypoint_collector_ros.py

for script in \
  deployment/scripts/hardware/hardware_env.sh \
  deployment/scripts/waypoints/run_waypoint_collection.sh \
  deployment/scripts/waypoints/start_waypoint_collection_detached.sh \
  deployment/scripts/waypoints/stop_waypoint_collection_detached.sh \
  deployment/scripts/waypoints/start_route_collection.sh \
  deployment/scripts/waypoints/stop_route_collection.sh \
  deployment/scripts/localization/run_route_map_localization.sh; do
  bash -n "${script}"
done

"${PYTHON}" - <<'PY'
import json
from pathlib import Path

import numpy as np
import yaml

from deployment.localization.continuous_map_localization import (
    ContinuousLocalizationConfig,
    ContinuousMapLocalizer,
)

root = Path.cwd()
waypoint = yaml.safe_load((root / "deployment/config/waypoint_collection.yaml").read_text())
localization = yaml.safe_load((root / "deployment/config/hardware_localization.yaml").read_text())
manifest = json.loads(
    (root / "deployment/maps/current/localization_map_manifest.json").read_text()
)

assert waypoint["collection"]["gamepad_controls"] is True
assert waypoint["collection"]["gamepad_start_key"] == "G12_KEY_A"
assert waypoint["collection"]["gamepad_mark_key"] == "G12_KEY_B"
assert waypoint["collection"]["save_lidar_observations"] is True
assert waypoint["topics"]["odometry"] == localization["topics"]["odometry"]
assert waypoint["topics"]["diagnostics"] == localization["topics"]["diagnostics"]
assert localization["runtime"]["enable_motion"] is False
assert localization["runtime"]["publish_legacy_cmd_vel"] is False
assert localization["runtime"]["publish_dds_steer"] is False
assert manifest["map_type"] == "ordered_topometric_route_submaps"
assert manifest["submap_count"] == len(manifest["submaps"]) == 163

map_root = root / "deployment/maps/current"
localizer = ContinuousMapLocalizer(
    map_root,
    initial_map_from_body=None,
    initial_odom_from_body=np.eye(4),
    config=ContinuousLocalizationConfig(),
)
first = localizer.submaps[0]
with np.load(first["file"], allow_pickle=False) as payload:
    scan = np.asarray(payload["points_anchor_m"], dtype=np.float64)
result = localizer.update(
    scan,
    np.eye(4),
    traveled_distance_m=1.1,
    allow_large_relocalization=True,
)
assert result.observation_accepted
assert result.selected_route_index == 0
assert result.fitness is not None and result.fitness > 0.99
assert result.rmse_m is not None and result.rmse_m < 1.0e-6
print("WAYPOINT_LOCALIZATION_CONFIG_OK")
print(
    "FROZEN_MAP_RUNTIME_OK",
    f"submaps={len(localizer.submaps)}",
    f"fitness={result.fitness:.6f}",
    f"rmse_m={result.rmse_m:.3e}",
)
PY

echo
echo "S10_WAYPOINT_LOCALIZATION_PREFLIGHT_OK"
echo "No robot command publisher was started."
