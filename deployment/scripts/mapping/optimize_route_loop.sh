#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
if (( $# != 3 )); then
  echo "Usage: $0 MAPPING_SESSION GLIM_DUMP OUTPUT_DIR" >&2
  exit 2
fi

SESSION="$(realpath "$1")"
GLIM_DUMP="$(realpath "$2")"
OUTPUT_DIR="$3"
TRAJECTORY="${GLIM_DUMP}/traj_lidar.txt"
PYTHON="${S10_SLAM_PYTHON:-python3}"

[[ -f "${SESSION}/session.json" ]] || {
  echo "Mapping session is incomplete: ${SESSION}" >&2
  exit 2
}
[[ -f "${TRAJECTORY}" ]] || {
  echo "GLIM trajectory is missing: ${TRAJECTORY}" >&2
  exit 2
}
[[ ! -e "${OUTPUT_DIR}" ]] || {
  echo "Refusing to overwrite loop output: ${OUTPUT_DIR}" >&2
  exit 2
}

cd "${ROOT}"
PYTHONPATH=. "${PYTHON}" -m deployment.mapping.optimize_route_loop \
  --session "${SESSION}" \
  --trajectory "${TRAJECTORY}" \
  --output-dir "${OUTPUT_DIR}"

echo "S10_LOOP_TRAJECTORY=$(realpath "${OUTPUT_DIR}/optimized_route_trajectory.npz")"
