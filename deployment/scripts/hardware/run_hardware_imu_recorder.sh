#!/usr/bin/env bash
set -eo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
OUTPUT="${1:?usage: run_hardware_imu_recorder.sh OUTPUT_FILE}"
source "${ROOT}/deployment/scripts/hardware/hardware_env.sh"
cd "${ROOT}"
exec "${S10_PYTHON:-python3}" -m deployment.mapping.hardware_imu_recorder \
  --output "${OUTPUT}" --topic /IMU_DATA
