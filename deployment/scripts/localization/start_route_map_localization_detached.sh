#!/usr/bin/env bash
set -eo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
RUNTIME_DIR="${S10_LOCALIZATION_RUNTIME_DIR:-/tmp/s10_route_map_localization}"
PID_FILE="${RUNTIME_DIR}/pid"
LOG_FILE="${RUNTIME_DIR}/current.log"
mkdir -p "${RUNTIME_DIR}"

if [[ -f "${PID_FILE}" ]]; then
  PID="$(cat "${PID_FILE}")"
  if kill -0 "${PID}" 2>/dev/null; then
    echo "Route-map localization is already running: pid=${PID}" >&2
    echo "Log: ${LOG_FILE}" >&2
    exit 2
  fi
  rm -f "${PID_FILE}"
fi

if pgrep -f 'deployment\.ros2_navigation.*--localization-only.*--map-dir' >/dev/null 2>&1; then
  echo "Another route-map localization process is already running." >&2
  echo "Stop it before starting a detached instance." >&2
  exit 2
fi

nohup setsid "${ROOT}/deployment/scripts/localization/run_route_map_localization.sh" \
  "$@" >"${LOG_FILE}" 2>&1 </dev/null &
PID=$!
printf '%s\n' "${PID}" >"${PID_FILE}"
sleep 2

if ! kill -0 "${PID}" 2>/dev/null; then
  echo "Route-map localization failed to start:" >&2
  tail -n 60 "${LOG_FILE}" >&2 || true
  rm -f "${PID_FILE}"
  exit 2
fi

echo "ROUTE_MAP_LOCALIZATION_STARTED pid=${PID}"
echo "This process is read-only and does not publish robot commands."
echo "Log: ${LOG_FILE}"
echo "Status: ${ROOT}/deployment/scripts/localization/status_route_map_localization.sh"
echo "Stop: ${ROOT}/deployment/scripts/localization/stop_route_map_localization_detached.sh"
