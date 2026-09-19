#!/usr/bin/env bash
set -eo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
RUNTIME_DIR="${S10_MAPPING_RUNTIME_DIR:-/tmp/s10_support_mapping}"
PID_FILE="${RUNTIME_DIR}/pid"
IMU_PID_FILE="${RUNTIME_DIR}/imu_pid"
SESSION_FILE="${RUNTIME_DIR}/session_dir"
[[ -f "${PID_FILE}" && -f "${IMU_PID_FILE}" && -f "${SESSION_FILE}" ]] || {
  echo "No active support mapping session." >&2
  exit 2
}
PID="$(cat "${PID_FILE}")"
IMU_PID="$(cat "${IMU_PID_FILE}")"
SESSION_DIR="$(cat "${SESSION_FILE}")"

stop_group() {
  local pid="$1"
  local ticks="$2"
  if kill -0 "${pid}" 2>/dev/null; then
    kill -INT -- "-${pid}" 2>/dev/null || kill -INT "${pid}" 2>/dev/null || true
    for _ in $(seq 1 "${ticks}"); do
      kill -0 "${pid}" 2>/dev/null || break
      sleep 0.25
    done
    if kill -0 "${pid}" 2>/dev/null; then
      kill -TERM -- "-${pid}" 2>/dev/null || kill -TERM "${pid}" 2>/dev/null || true
      for _ in $(seq 1 20); do
        kill -0 "${pid}" 2>/dev/null || break
        sleep 0.25
      done
    fi
    if kill -0 "${pid}" 2>/dev/null; then
      echo "Process did not stop cleanly: ${pid}" >&2
      return 2
    fi
  fi
}

# Stop LiDAR first so the independent IMU file necessarily covers its tail.
stop_group "${PID}" 120
stop_group "${IMU_PID}" 40
rm -f "${PID_FILE}" "${IMU_PID_FILE}" "${SESSION_FILE}"
source "${ROOT}/deployment/scripts/hardware/hardware_env.sh"
"${S10_PYTHON:-python3}" -m deployment.mapping.attach_external_imu \
  "${SESSION_DIR}/mapping" "${SESSION_DIR}/imu_samples_external.f64" \
  | tee "${SESSION_DIR}/imu_attachment.json"
"${S10_PYTHON:-python3}" -m deployment.mapping.validate_mapping_recording \
  "${SESSION_DIR}/mapping" | tee "${SESSION_DIR}/mapping_validation.json"
set +e
"${S10_PYTHON:-python3}" -m deployment.mapping.qualify_mapping_recording \
  "${SESSION_DIR}/mapping" | tee "${SESSION_DIR}/mapping_qualification.json"
QUALIFICATION_STATUS=${PIPESTATUS[0]}
set -e
if [[ ${QUALIFICATION_STATUS} -ne 0 ]]; then
  echo "S10_SUPPORT_MAPPING_NOT_QUALIFIED=${SESSION_DIR}" >&2
  exit "${QUALIFICATION_STATUS}"
fi
echo "S10_SUPPORT_MAPPING_COMPLETE=${SESSION_DIR}"
