#!/usr/bin/env bash
set -eo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
RUNTIME_DIR="${S10_GAMEPAD_MAPPING_RUNTIME_DIR:-/tmp/s10_gamepad_mapping}"
PID_FILE="${RUNTIME_DIR}/pid"
LOG_FILE="${RUNTIME_DIR}/controller.log"
mkdir -p "${RUNTIME_DIR}"

if [[ -f "${PID_FILE}" ]] && kill -0 "$(cat "${PID_FILE}")" 2>/dev/null; then
  echo "Gamepad mapping controller is already running: pid=$(cat "${PID_FILE}")"
  echo "Log: ${LOG_FILE}"
  exit 0
fi

nohup setsid "${ROOT}/deployment/scripts/mapping/run_gamepad_mapping_controller.sh" \
  "$@" \
  >"${LOG_FILE}" 2>&1 </dev/null &
PID=$!
printf '%s\n' "${PID}" >"${PID_FILE}"
sleep 2
if ! kill -0 "${PID}" 2>/dev/null; then
  tail -n 80 "${LOG_FILE}" >&2 || true
  rm -f "${PID_FILE}"
  exit 2
fi
grep -q '"state": "READY"' "${LOG_FILE}" || {
  tail -n 80 "${LOG_FILE}" >&2 || true
  echo "Controller did not report READY." >&2
  exit 2
}
echo "S10_GAMEPAD_MAPPING_CONTROLLER_READY pid=${PID}"
echo "A=start canonical_lap/support_lap; B=stop and qualify"
echo "The controller survives SSH disconnects. Log: ${LOG_FILE}"
