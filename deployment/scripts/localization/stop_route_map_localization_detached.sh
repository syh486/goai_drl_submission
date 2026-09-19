#!/usr/bin/env bash
set -eo pipefail

RUNTIME_DIR="${S10_LOCALIZATION_RUNTIME_DIR:-/tmp/s10_route_map_localization}"
PID_FILE="${RUNTIME_DIR}/pid"
LOG_FILE="${RUNTIME_DIR}/current.log"

if [[ ! -f "${PID_FILE}" ]]; then
  echo "No detached route-map localization PID file exists."
  exit 0
fi

PID="$(cat "${PID_FILE}")"
if kill -0 "${PID}" 2>/dev/null; then
  kill -INT -- "-${PID}" 2>/dev/null || kill -INT "${PID}" 2>/dev/null || true
  for _ in $(seq 1 80); do
    kill -0 "${PID}" 2>/dev/null || break
    sleep 0.25
  done
  if kill -0 "${PID}" 2>/dev/null; then
    kill -TERM -- "-${PID}" 2>/dev/null || kill -TERM "${PID}" 2>/dev/null || true
  fi
fi
rm -f "${PID_FILE}"

echo "ROUTE_MAP_LOCALIZATION_STOPPED"
echo "Log: ${LOG_FILE}"
