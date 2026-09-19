#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
if (( $# != 5 )); then
  echo "Usage: $0 CANONICAL_SESSION CANONICAL_GLIM SUPPORT_SESSION SUPPORT_GLIM OUTPUT_ROOT" >&2
  exit 2
fi

CANONICAL_SESSION="$(realpath "$1")"
CANONICAL_GLIM="$(realpath "$2")"
SUPPORT_SESSION="$(realpath "$3")"
SUPPORT_GLIM="$(realpath "$4")"
mkdir -p "$5"
OUTPUT_ROOT="$(realpath "$5")"
PYTHON="${S10_SLAM_PYTHON:-python3}"
CANONICAL_TRAJECTORY="${CANONICAL_GLIM}/traj_lidar.txt"
SUPPORT_TRAJECTORY="${SUPPORT_GLIM}/traj_lidar.txt"
BOOTSTRAP_LOOP="${OUTPUT_ROOT}/bootstrap_endpoint_loop"
BOOTSTRAP_MAP="${OUTPUT_ROOT}/bootstrap_map"
REFINEMENT="${OUTPUT_ROOT}/multilap_refinement"
FINAL_MAP="${OUTPUT_ROOT}/localization_map"

for path in \
  "${CANONICAL_SESSION}/session.json" \
  "${SUPPORT_SESSION}/session.json" \
  "${CANONICAL_TRAJECTORY}" \
  "${SUPPORT_TRAJECTORY}"; do
  [[ -f "${path}" ]] || { echo "Required input is missing: ${path}" >&2; exit 2; }
done
cd "${ROOT}"

if [[ ! -f "${BOOTSTRAP_LOOP}/optimized_route_trajectory.npz" ]]; then
  [[ ! -e "${BOOTSTRAP_LOOP}" ]] || {
    echo "Incomplete endpoint bootstrap exists: ${BOOTSTRAP_LOOP}" >&2
    exit 2
  }
  deployment/scripts/mapping/optimize_route_loop.sh \
    "${CANONICAL_SESSION}" "${CANONICAL_GLIM}" "${BOOTSTRAP_LOOP}"
fi
PYTHONPATH=. "${PYTHON}" -m deployment.mapping.validate_loop_optimization \
  "${BOOTSTRAP_LOOP}"

if [[ ! -f "${BOOTSTRAP_MAP}/localization_map_manifest.json" ]]; then
  [[ ! -e "${BOOTSTRAP_MAP}" ]] || {
    echo "Incomplete bootstrap map exists: ${BOOTSTRAP_MAP}" >&2
    exit 2
  }
  PYTHONPATH=. "${PYTHON}" -m deployment.mapping.build_topometric_route_map \
    --session "${CANONICAL_SESSION}" \
    --trajectory "${BOOTSTRAP_LOOP}/optimized_route_trajectory.npz" \
    --output-dir "${BOOTSTRAP_MAP}" \
    --anchor-spacing-m 2.0 \
    --submap-aggregate-radius-frames 4
fi

deployment/scripts/mapping/build_multilap_optimizer.sh >/dev/null
if [[ ! -f "${REFINEMENT}/multilap_refinement_report.json" ]]; then
  PYTHONPATH=. "${PYTHON}" -m deployment.mapping.refine_route_multilap \
    --map-dir "${BOOTSTRAP_MAP}" \
    --canonical-session "${CANONICAL_SESSION}" \
    --canonical-trajectory "${BOOTSTRAP_LOOP}/optimized_route_trajectory.npz" \
    --support-session "${SUPPORT_SESSION}" \
    --support-trajectory "${SUPPORT_TRAJECTORY}" \
    --output-dir "${REFINEMENT}" \
    --anchor-stride-submaps 4 \
    --candidate-radius-frames 8 \
    --candidate-step-frames 2 \
    --minimum-constraints 20
fi
PYTHONPATH=. "${PYTHON}" -m deployment.mapping.validate_multilap_refinement "${REFINEMENT}"

if [[ ! -f "${FINAL_MAP}/localization_map_manifest.json" ]]; then
  [[ ! -e "${FINAL_MAP}" ]] || {
    echo "Incomplete final localization map exists: ${FINAL_MAP}" >&2
    exit 2
  }
  PYTHONPATH=. "${PYTHON}" -m deployment.mapping.build_topometric_route_map \
    --session "${CANONICAL_SESSION}" \
    --trajectory "${REFINEMENT}/refined_route_trajectory.npz" \
    --output-dir "${FINAL_MAP}" \
    --anchor-spacing-m 2.0 \
    --submap-aggregate-radius-frames 4
fi

echo "S10_MULTILAP_MAP=$(realpath "${FINAL_MAP}")"
