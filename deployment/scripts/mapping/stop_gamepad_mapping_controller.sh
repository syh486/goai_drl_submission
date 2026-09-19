#!/usr/bin/env bash
set -eo pipefail

RUNTIME_DIR="${S10_GAMEPAD_MAPPING_RUNTIME_DIR:-/tmp/s10_gamepad_mapping}"
PID_FILE="${RUNTIME_DIR}/pid"
[[ -f "${PID_FILE}" ]] || { echo "No gamepad mapping controller."; exit 0; }
PID="$(cat "${PID_FILE}")"
if kill -0 "${PID}" 2>/dev/null; then
  kill -INT -- "-${PID}" 2>/dev/null || kill -INT "${PID}" 2>/dev/null || true
  for _ in $(seq 1 40); do
    kill -0 "${PID}" 2>/dev/null || break
    sleep 0.25
  done
fi
rm -f "${PID_FILE}"
echo "S10_GAMEPAD_MAPPING_CONTROLLER_STOPPED"
