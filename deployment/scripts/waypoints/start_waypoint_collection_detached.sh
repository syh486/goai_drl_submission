#!/usr/bin/env bash
set -eo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
RUNTIME_DIR="${S10_WAYPOINT_RUNTIME_DIR:-/tmp/s10_waypoint_collection}"
PID_FILE="${RUNTIME_DIR}/pid"
LOG_FILE="${RUNTIME_DIR}/current.log"
mkdir -p "${RUNTIME_DIR}"

if [[ -f "${PID_FILE}" ]]; then
  PID="$(cat "${PID_FILE}")"
  if kill -0 "${PID}" 2>/dev/null; then
    echo "Waypoint collection is already running: pid=${PID}" >&2
    echo "Log: ${LOG_FILE}" >&2
    exit 2
  fi
  rm -f "${PID_FILE}"
fi

nohup setsid "${ROOT}/deployment/scripts/waypoints/run_waypoint_collection.sh" \
  --no-interactive "$@" >"${LOG_FILE}" 2>&1 </dev/null &
PID=$!
printf '%s\n' "${PID}" >"${PID_FILE}"
sleep 2

if ! kill -0 "${PID}" 2>/dev/null; then
  echo "Waypoint collection failed to start:" >&2
  tail -n 40 "${LOG_FILE}" >&2 || true
  rm -f "${PID_FILE}"
  exit 2
fi

echo "WAYPOINT_COLLECTION_STARTED pid=${PID}"
echo "The SSH connection may now disconnect without stopping collection."
echo "Log: ${LOG_FILE}"
echo "Monitor: tail -f ${LOG_FILE}"
echo "Stop: ${ROOT}/deployment/scripts/waypoints/stop_waypoint_collection_detached.sh"
