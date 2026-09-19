#!/usr/bin/env bash
set -eo pipefail

RUNTIME_DIR="${S10_WAYPOINT_RUNTIME_DIR:-/tmp/s10_waypoint_collection}"
PID_FILE="${RUNTIME_DIR}/pid"
LOG_FILE="${RUNTIME_DIR}/current.log"

if [[ ! -f "${PID_FILE}" ]]; then
  echo "No detached waypoint collection PID file exists."
  exit 0
fi

PID="$(cat "${PID_FILE}")"
if kill -0 "${PID}" 2>/dev/null; then
  kill -INT -- "-${PID}" 2>/dev/null || kill -INT "${PID}" 2>/dev/null || true
  # A full mapping lap may still have compressed keyframes in the writer
  # queue. Give the localizer up to 30 seconds to flush and mark the session
  # complete before escalating to TERM.
  for _ in $(seq 1 120); do
    kill -0 "${PID}" 2>/dev/null || break
    sleep 0.25
  done
  if kill -0 "${PID}" 2>/dev/null; then
    kill -TERM -- "-${PID}" 2>/dev/null || kill -TERM "${PID}" 2>/dev/null || true
  fi
fi
rm -f "${PID_FILE}"

echo "WAYPOINT_COLLECTION_STOPPED"
echo "Log: ${LOG_FILE}"
