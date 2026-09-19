#!/usr/bin/env bash
set -eo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
source "${ROOT}/deployment/scripts/hardware/hardware_env.sh"

ENABLE_OUTPUT=false
if [[ "${1:-}" == "--enable-command-output" ]]; then
  ENABLE_OUTPUT=true
  shift
fi
if (( $# > 0 )); then
  echo "usage: $0 [--enable-command-output]" >&2
  exit 2
fi

ENCODER="${ROOT}/deployment/models/s10_lidar_encoder.onnx"
POLICY="${ROOT}/deployment/models/sru_policy1_model2750.onnx"
for model in "${ENCODER}" "${POLICY}"; do
  if [[ ! -f "${model}" ]]; then
    echo "missing ONNX model: ${model}" >&2
    exit 2
  fi
done

if [[ "${ENABLE_OUTPUT}" == "true" ]]; then
  echo "S10_ONNX_COMMAND_OUTPUT_ENABLED topic=/s10/navigation/steer" >&2
else
  echo "S10_ONNX_DRY_RUN_ONLY topic=/s10/navigation/candidate_cmd" >&2
fi

exec ros2 run s10_sdk_deploy sru_high_level_dry_run --ros-args \
  -p encoder_model:="${ENCODER}" \
  -p policy_model:="${POLICY}" \
  -p enable_command_output:="${ENABLE_OUTPUT}"
