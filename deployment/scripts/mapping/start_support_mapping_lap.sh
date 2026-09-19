#!/usr/bin/env bash
set -eo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
SESSION_NAME="${1:-support_$(date +%Y%m%d_%H%M%S)}"
COLLECTION_ROOT="${S10_COLLECTION_ROOT:-${HOME}/s10_route_collection}"
SESSION_DIR="${COLLECTION_ROOT}/${SESSION_NAME}"
RUNTIME_DIR="${S10_MAPPING_RUNTIME_DIR:-/tmp/s10_support_mapping}"
PID_FILE="${RUNTIME_DIR}/pid"
IMU_PID_FILE="${RUNTIME_DIR}/imu_pid"
LOG_FILE="${RUNTIME_DIR}/current.log"
IMU_LOG_FILE="${RUNTIME_DIR}/imu.log"
CONFIG="${ROOT}/deployment/config/hardware_localization.yaml"

if [[ -f "${PID_FILE}" ]] && kill -0 "$(cat "${PID_FILE}")" 2>/dev/null; then
  echo "Support mapping is already running: pid=$(cat "${PID_FILE}")" >&2
  exit 2
fi
if [[ -f "${IMU_PID_FILE}" ]] && kill -0 "$(cat "${IMU_PID_FILE}")" 2>/dev/null; then
  echo "Support mapping IMU recorder is already running: pid=$(cat "${IMU_PID_FILE}")" >&2
  exit 2
fi
if [[ -e "${SESSION_DIR}" ]]; then
  echo "Refusing to overwrite support session: ${SESSION_DIR}" >&2
  exit 2
fi
FREE_KIB="$(df -Pk "${COLLECTION_ROOT%/*}" | awk 'NR==2 {print $4}')"
if [[ -z "${FREE_KIB}" || "${FREE_KIB}" -lt 2097152 ]]; then
  echo "At least 2 GiB free space is required for a support mapping lap." >&2
  exit 2
fi

mkdir -p "${SESSION_DIR}" "${RUNTIME_DIR}"
printf '%s\n' "${SESSION_DIR}" >"${RUNTIME_DIR}/session_dir"

EXTERNAL_IMU="${SESSION_DIR}/imu_samples_external.f64"
nohup setsid "${ROOT}/deployment/scripts/hardware/run_hardware_imu_recorder.sh" \
  "${EXTERNAL_IMU}" >"${IMU_LOG_FILE}" 2>&1 </dev/null &
IMU_PID=$!
printf '%s\n' "${IMU_PID}" >"${IMU_PID_FILE}"
sleep 1
if ! kill -0 "${IMU_PID}" 2>/dev/null; then
  tail -n 50 "${IMU_LOG_FILE}" >&2 || true
  rm -f "${IMU_PID_FILE}" "${RUNTIME_DIR}/session_dir"
  exit 2
fi

nohup setsid "${ROOT}/deployment/scripts/localization/run_hardware_localization.sh" \
  --record-map "${SESSION_DIR}/mapping" \
  --record-start-anchor "${SESSION_DIR}/start.anchor.npz" \
  >"${LOG_FILE}" 2>&1 </dev/null &
PID=$!
printf '%s\n' "${PID}" >"${PID_FILE}"
sleep 2
if ! kill -0 "${PID}" 2>/dev/null; then
  kill -INT -- "-${IMU_PID}" 2>/dev/null || kill -INT "${IMU_PID}" 2>/dev/null || true
  tail -n 50 "${LOG_FILE}" >&2 || true
  rm -f "${PID_FILE}" "${IMU_PID_FILE}" "${RUNTIME_DIR}/session_dir"
  exit 2
fi

# Record first, then audit.  The operator has no visual acknowledgement on the
# factory controller, so a preflight here used to leave the first several
# seconds after A completely unrecorded and could miss the stationary anchor.
echo "Auditing live dual-LiDAR, IMU and joint topics while recording..."
if ! "${ROOT}/deployment/scripts/hardware/check_hardware_topics.sh" \
  --config "${CONFIG}" --duration 5.0 --min-joint-hz 0.5; then
  kill -INT -- "-${PID}" 2>/dev/null || kill -INT "${PID}" 2>/dev/null || true
  kill -INT -- "-${IMU_PID}" 2>/dev/null || kill -INT "${IMU_PID}" 2>/dev/null || true
  sleep 1
  kill -TERM -- "-${PID}" 2>/dev/null || kill -TERM "${PID}" 2>/dev/null || true
  kill -TERM -- "-${IMU_PID}" 2>/dev/null || kill -TERM "${IMU_PID}" 2>/dev/null || true
  rm -f "${PID_FILE}" "${IMU_PID_FILE}" "${RUNTIME_DIR}/session_dir"
  echo "Recording startup failed: hardware topic audit failed; do not move." >&2
  exit 2
fi

# Do not tell the operator to move until both data paths have reached disk.
# The isolated recorder flushes every 200 rows (11 float64 columns per row).
DATA_READY=0
for _ in $(seq 1 240); do
  if ! kill -0 "${PID}" 2>/dev/null || ! kill -0 "${IMU_PID}" 2>/dev/null; then
    break
  fi
  KEYFRAMES="$(find "${SESSION_DIR}/mapping/keyframes" -maxdepth 1 -name '*.npz' -type f 2>/dev/null | wc -l)"
  IMU_BYTES="$(stat -c '%s' "${EXTERNAL_IMU}" 2>/dev/null || printf '0')"
  IMU_SAMPLES=$((IMU_BYTES / 88))
  if [[ "${KEYFRAMES}" -ge 3 && "${IMU_SAMPLES}" -ge 200 ]]; then
    DATA_READY=1
    break
  fi
  sleep 0.25
done
if [[ "${DATA_READY}" -ne 1 ]]; then
  kill -INT -- "-${PID}" 2>/dev/null || kill -INT "${PID}" 2>/dev/null || true
  kill -INT -- "-${IMU_PID}" 2>/dev/null || kill -INT "${IMU_PID}" 2>/dev/null || true
  sleep 1
  kill -TERM -- "-${PID}" 2>/dev/null || kill -TERM "${PID}" 2>/dev/null || true
  kill -TERM -- "-${IMU_PID}" 2>/dev/null || kill -TERM "${IMU_PID}" 2>/dev/null || true
  rm -f "${PID_FILE}" "${IMU_PID_FILE}" "${RUNTIME_DIR}/session_dir"
  tail -n 80 "${LOG_FILE}" >&2 || true
  tail -n 80 "${IMU_LOG_FILE}" >&2 || true
  echo "Recording startup failed: no durable IMU + dual-LiDAR growth; do not move." >&2
  exit 2
fi
echo "S10_SUPPORT_MAPPING_STARTED session=${SESSION_DIR} mapping_pid=${PID} imu_pid=${IMU_PID}"
echo "S10_SUPPORT_MAPPING_DATA_READY keyframes=${KEYFRAMES} imu_samples=${IMU_SAMPLES}"
echo "Monitor: tail -f ${LOG_FILE}"
