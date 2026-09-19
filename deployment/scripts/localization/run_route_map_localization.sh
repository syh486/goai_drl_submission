#!/usr/bin/env bash
set -eo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
source "${ROOT}/deployment/scripts/hardware/hardware_env.sh"
cd "${ROOT}"

MAP_DIR="${S10_ROUTE_MAP_DIR:-${ROOT}/deployment/maps/current}"
if [[ ! -f "${MAP_DIR}/localization_map_manifest.json" ]]; then
  echo "Frozen route map not found: ${MAP_DIR}" >&2
  echo "Set S10_ROUTE_MAP_DIR or install a map at deployment/maps/current." >&2
  exit 2
fi

exec "${S10_PYTHON:-python3}" -m deployment.navigation.ros2_node \
  --config "${ROOT}/deployment/config/hardware_localization.yaml" \
  --localization-only \
  --map-dir "${MAP_DIR}" \
  "$@"
