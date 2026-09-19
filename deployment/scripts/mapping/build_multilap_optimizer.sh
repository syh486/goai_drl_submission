#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
SOURCE_DIR="${ROOT}/deployment/native"
BUILD_DIR="${S10_MULTILAP_BUILD_DIR:-${SOURCE_DIR}/build}"

cmake -S "${SOURCE_DIR}" -B "${BUILD_DIR}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DPython3_EXECUTABLE="${S10_SLAM_PYTHON:-$(command -v python3)}"
TARGETS=(multilap_pose_graph)
TARGET_HELP="$(cmake --build "${BUILD_DIR}" --target help 2>/dev/null || true)"
if grep -q '^... s10_vgicp$' <<<"${TARGET_HELP}"; then
  TARGETS+=(s10_vgicp)
fi
cmake --build "${BUILD_DIR}" --target "${TARGETS[@]}" --parallel
echo "S10_MULTILAP_OPTIMIZER=${BUILD_DIR}/multilap_pose_graph"
if [[ -f "${BUILD_DIR}/s10_vgicp.so" ]]; then
  echo "S10_VGICP_PYTHON=${BUILD_DIR}/s10_vgicp.so"
fi
