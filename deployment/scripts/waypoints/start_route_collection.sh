#!/usr/bin/env bash
set -eo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
SESSION_NAME="${1:-route_$(date +%Y%m%d_%H%M%S)}"
COLLECTION_ROOT="${S10_COLLECTION_ROOT:-${HOME}/s10_route_collection}"
SESSION_DIR="${COLLECTION_ROOT}/${SESSION_NAME}"
RUNTIME_DIR="${S10_WAYPOINT_RUNTIME_DIR:-/tmp/s10_waypoint_collection}"

if [[ -e "${SESSION_DIR}" ]]; then
  echo "Refusing to overwrite collection session: ${SESSION_DIR}" >&2
  exit 2
fi
FREE_KIB="$(df -Pk "${COLLECTION_ROOT%/*}" | awk 'NR==2 {print $4}')"
if [[ -z "${FREE_KIB}" || "${FREE_KIB}" -lt 2097152 ]]; then
  echo "At least 2 GiB free space is required for route collection." >&2
  exit 2
fi
mkdir -p "${SESSION_DIR}" "${RUNTIME_DIR}"
printf '%s\n' "${SESSION_DIR}" >"${RUNTIME_DIR}/session_dir"

"${ROOT}/deployment/scripts/waypoints/start_waypoint_collection_detached.sh" \
  --output "${SESSION_DIR}/route.yaml" \
  --map-session "${SESSION_DIR}/mapping"

echo "ROUTE_COLLECTION_SESSION=${SESSION_DIR}"
echo "A: record start once; B: record each later waypoint after standing still."
echo "Return to the start and stand still before running stop_route_collection.sh."
