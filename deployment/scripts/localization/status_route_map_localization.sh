#!/usr/bin/env bash
set -eo pipefail

RUNTIME_DIR="${S10_LOCALIZATION_RUNTIME_DIR:-/tmp/s10_route_map_localization}"
PID_FILE="${RUNTIME_DIR}/pid"
LOG_FILE="${RUNTIME_DIR}/current.log"

if [[ ! -f "${PID_FILE}" ]]; then
  echo "ROUTE_MAP_LOCALIZATION_NOT_RUNNING"
  [[ -f "${LOG_FILE}" ]] && tail -n 40 "${LOG_FILE}"
  exit 1
fi

PID="$(cat "${PID_FILE}")"
if ! kill -0 "${PID}" 2>/dev/null; then
  echo "ROUTE_MAP_LOCALIZATION_STALE_PID pid=${PID}"
  tail -n 60 "${LOG_FILE}" 2>/dev/null || true
  exit 2
fi

echo "ROUTE_MAP_LOCALIZATION_RUNNING pid=${PID}"
echo "Log: ${LOG_FILE}"
tail -n 40 "${LOG_FILE}" 2>/dev/null || true
