#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
if (( $# != 2 )); then
  echo "Usage: $0 CANONICAL_SESSION_DIR SUPPORT_SESSION_DIR" >&2
  exit 2
fi

CANONICAL_DIR="$(realpath "$1")"
SUPPORT_DIR="$(realpath "$2")"
CANONICAL_MAPPING="${CANONICAL_DIR}/mapping"
SUPPORT_MAPPING="${SUPPORT_DIR}/mapping"
ROUTE="${CANONICAL_DIR}/route.yaml"
CANONICAL_ROSBAG="${CANONICAL_DIR}/mapping_rosbag"
SUPPORT_ROSBAG="${SUPPORT_DIR}/mapping_rosbag"
CANONICAL_GLIM="${CANONICAL_DIR}/glim_lio"
SUPPORT_GLIM="${SUPPORT_DIR}/glim_lio"
MULTILAP_ROOT="${CANONICAL_DIR}/multilap_build"
TOPO_MAP="${MULTILAP_ROOT}/localization_map"
TOPO_MAP_LINK="${CANONICAL_DIR}/topometric_map"
PYTHON="${S10_SLAM_PYTHON:-python3}"

for path in "${CANONICAL_MAPPING}" "${SUPPORT_MAPPING}" "${ROUTE}"; do
  [[ -e "${path}" ]] || { echo "Required input is missing: ${path}" >&2; exit 2; }
done
[[ ! -e "${CANONICAL_DIR}/route_rebound.yaml" ]] || {
  echo "Refusing to overwrite route_rebound.yaml in ${CANONICAL_DIR}" >&2
  exit 2
}

cd "${ROOT}"
PYTHONPATH=. "${PYTHON}" -m deployment.waypoints.validate_bundle \
  "${CANONICAL_DIR}"
if [[ ! -f "${CANONICAL_ROSBAG}/metadata.yaml" ]]; then
  deployment/scripts/mapping/export_mapping_rosbag.sh \
    "${CANONICAL_MAPPING}" "${CANONICAL_ROSBAG}"
fi
if [[ ! -f "${SUPPORT_ROSBAG}/metadata.yaml" ]]; then
  deployment/scripts/mapping/export_mapping_rosbag.sh \
    "${SUPPORT_MAPPING}" "${SUPPORT_ROSBAG}"
fi
if [[ ! -f "${CANONICAL_GLIM}/traj_lidar.txt" ]]; then
  deployment/scripts/mapping/run_glim_mapping.sh \
    "${CANONICAL_ROSBAG}" "${CANONICAL_GLIM}"
fi
if [[ ! -f "${SUPPORT_GLIM}/traj_lidar.txt" ]]; then
  deployment/scripts/mapping/run_glim_mapping.sh \
    "${SUPPORT_ROSBAG}" "${SUPPORT_GLIM}"
fi

deployment/scripts/mapping/build_multilap_route_map.sh \
  "${CANONICAL_MAPPING}" "${CANONICAL_GLIM}" \
  "${SUPPORT_MAPPING}" "${SUPPORT_GLIM}" \
  "${MULTILAP_ROOT}"

if [[ -L "${TOPO_MAP_LINK}" ]]; then
  [[ "$(realpath "${TOPO_MAP_LINK}")" == "$(realpath "${TOPO_MAP}")" ]] || {
    echo "topometric_map points to a different map: ${TOPO_MAP_LINK}" >&2
    exit 2
  }
elif [[ -e "${TOPO_MAP_LINK}" ]]; then
  echo "Refusing to replace existing topometric_map: ${TOPO_MAP_LINK}" >&2
  exit 2
else
  ln -s "multilap_build/localization_map" "${TOPO_MAP_LINK}"
fi

OPTIMIZED="${MULTILAP_ROOT}/multilap_refinement/refined_route_trajectory.npz"
PYTHONPATH=. "${PYTHON}" -m deployment.waypoints.rebind \
  --route "${ROUTE}" \
  --mapping-session "${CANONICAL_MAPPING}" \
  --optimized-poses "${OPTIMIZED}" \
  --output "${CANONICAL_DIR}/route_rebound.yaml" \
  --topometric-map "${TOPO_MAP}"
PYTHONPATH=. "${PYTHON}" -m deployment.waypoints.validate_route \
  "${CANONICAL_DIR}/route_rebound.yaml" \
  | tee "${CANONICAL_DIR}/route_rebound_validation.json"
echo "ROUTE_MAP_READY=${CANONICAL_DIR}/route_rebound.yaml"
