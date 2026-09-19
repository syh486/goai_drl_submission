#!/usr/bin/env bash
set -eo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
RUNTIME_DIR="${S10_WAYPOINT_RUNTIME_DIR:-/tmp/s10_waypoint_collection}"
SESSION_FILE="${RUNTIME_DIR}/session_dir"
if [[ ! -f "${SESSION_FILE}" ]]; then
  echo "No active route collection session is recorded." >&2
  exit 2
fi
SESSION_DIR="$(cat "${SESSION_FILE}")"
"${ROOT}/deployment/scripts/waypoints/stop_waypoint_collection_detached.sh"
rm -f "${SESSION_FILE}"

source "${ROOT}/deployment/scripts/hardware/hardware_env.sh"
"${S10_PYTHON:-python3}" -m deployment.waypoints.validate_bundle "${SESSION_DIR}" \
  | tee "${SESSION_DIR}/collection_validation.json"
echo "ROUTE_COLLECTION_COMPLETE=${SESSION_DIR}"
