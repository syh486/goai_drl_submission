#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
if (( $# != 4 )); then
  echo "Usage: $0 MAP_DIR HELDOUT_SESSION HELDOUT_GLIM OUTPUT_ROOT" >&2
  exit 2
fi

MAP_DIR="$(realpath "$1")"
HELDOUT_SESSION="$(realpath "$2")"
HELDOUT_GLIM="$(realpath "$3")"
OUTPUT_ROOT="$4"
PYTHON="${S10_SLAM_PYTHON:-python3}"
MANIFEST="${MAP_DIR}/localization_map_manifest.json"
HELDOUT_TRAJECTORY="${HELDOUT_GLIM}/traj_lidar.txt"

[[ -f "${MANIFEST}" ]] || { echo "Map manifest is missing: ${MANIFEST}" >&2; exit 2; }
[[ -f "${HELDOUT_TRAJECTORY}" ]] || {
  echo "Held-out GLIM trajectory is missing: ${HELDOUT_TRAJECTORY}" >&2
  exit 2
}
[[ ! -e "${OUTPUT_ROOT}" ]] || {
  echo "Refusing to overwrite held-out evaluation: ${OUTPUT_ROOT}" >&2
  exit 2
}
mkdir -p "${OUTPUT_ROOT}"
OUTPUT_ROOT="$(realpath "${OUTPUT_ROOT}")"

REFERENCE_SESSION="$(jq -r '.reference_sessions[0]' "${MANIFEST}")"
REFERENCE_TRAJECTORY="$(jq -r '.reference_trajectories[0]' "${MANIFEST}")"
REPLAY="${OUTPUT_ROOT}/heldout_replay.json"
cd "${ROOT}"
PYTHONPATH=. "${PYTHON}" -m deployment.localization.evaluate_topometric_replay \
  --map-dir "${MAP_DIR}" \
  --heldout-session "${HELDOUT_SESSION}" \
  --heldout-trajectory "${HELDOUT_TRAJECTORY}" \
  --localization-stride 1 \
  --output-report "${REPLAY}"
PYTHONPATH=. "${PYTHON}" -m deployment.localization.evaluate_metric_endpoint \
  --map-dir "${MAP_DIR}" \
  --reference-session "${REFERENCE_SESSION}" \
  --reference-trajectory "${REFERENCE_TRAJECTORY}" \
  --heldout-session "${HELDOUT_SESSION}" \
  --heldout-trajectory "${HELDOUT_TRAJECTORY}" \
  --replay-report "${REPLAY}" \
  --output-report "${OUTPUT_ROOT}/endpoint_metric_report.json"
PYTHONPATH=. "${PYTHON}" -m deployment.localization.evaluate_multilap_holdout \
  --map-dir "${MAP_DIR}" \
  --heldout-session "${HELDOUT_SESSION}" \
  --heldout-trajectory "${HELDOUT_TRAJECTORY}" \
  --replay-report "${REPLAY}" \
  --output-dir "${OUTPUT_ROOT}/causal_route_metric" \
  --anchor-stride-submaps 4 \
  --candidate-radius-frames 8 \
  --candidate-step-frames 2 \
  --minimum-constraints 20

echo "S10_MULTILAP_EVALUATION=$(realpath "${OUTPUT_ROOT}")"
