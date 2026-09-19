#!/usr/bin/env bash
set -eo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
if (( $# < 1 || $# > 2 )); then
  echo "usage: $0 ROUTE_YAML [--enable-command-output]" >&2
  exit 2
fi
ROUTE="$1"
MODE="${2:-}"
if [[ -n "${MODE}" && "${MODE}" != "--enable-command-output" ]]; then
  echo "unknown option: ${MODE}" >&2
  exit 2
fi
if [[ ! -f "${ROUTE}" ]]; then
  echo "route file not found: ${ROUTE}" >&2
  exit 2
fi

producer_pid=""
runner_pid=""
stop_stack() {
  [[ -z "${producer_pid}" ]] || kill "${producer_pid}" 2>/dev/null || true
  [[ -z "${runner_pid}" ]] || kill "${runner_pid}" 2>/dev/null || true
  wait 2>/dev/null || true
}
trap stop_stack EXIT INT TERM

"${ROOT}/deployment/scripts/navigation/run_hardware_navigation.sh" \
  --route-file "${ROUTE}" --onnx-dry-run &
producer_pid=$!

if [[ "${MODE}" == "--enable-command-output" ]]; then
  "${ROOT}/deployment/scripts/navigation/run_high_level_onnx.sh" \
    --enable-command-output &
else
  "${ROOT}/deployment/scripts/navigation/run_high_level_onnx.sh" &
fi
runner_pid=$!

echo "S10_ONNX_STACK_STARTED producer=${producer_pid} runner=${runner_pid} mode=${MODE:-dry_run}"
wait -n "${producer_pid}" "${runner_pid}"
echo "S10_ONNX_STACK_COMPONENT_EXITED; stopping the other component" >&2
exit 1
