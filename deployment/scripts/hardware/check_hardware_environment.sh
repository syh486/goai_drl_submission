#!/usr/bin/env bash
set -o pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
source "${ROOT}/deployment/scripts/hardware/hardware_env.sh" || exit $?
failures=0

check() {
  local description="$1"
  shift
  if "$@" >/dev/null 2>&1; then
    printf '[OK]   %s\n' "${description}"
  else
    printf '[FAIL] %s\n' "${description}"
    failures=$((failures + 1))
  fi
}

check_sha256() {
  local description="$1"
  local expected="$2"
  local path="$3"
  if [[ -f "${path}" ]] && [[ "$(sha256sum "${path}" | awk '{print $1}')" == "${expected}" ]]; then
    printf '[OK]   %s\n' "${description}"
  else
    printf '[FAIL] %s\n' "${description}"
    failures=$((failures + 1))
  fi
}

check "ARM64 target" test "$(uname -m)" = "aarch64"
check "ROS2 Jazzy" test "${ROS_DISTRO:-}" = "jazzy"
check "workspace overlay" test -f "${ROOT}/install/setup.bash"
check "Python localization imports" "${S10_PYTHON:-python3}" -c \
  'import numpy, scipy, yaml, kiss_icp; import deployment.navigation.ros2_node'
check "KISS map-registration API" "${S10_PYTHON:-python3}" -c \
  'from kiss_icp.mapping import VoxelHashMap; from kiss_icp.registration import Registration'
check_sha256 "ONNX high-level policy model_2750" \
  77ba167f52861e471cacb6499dcaddf5355c1bc9e363d87ab860469fd8a5f446 \
  "${ROOT}/deployment/models/sru_policy1_model2750.onnx"
check_sha256 "ONNX random-terrain LiDAR encoder" \
  011eb34e5093ba6481dfffdcc63354076ad09d3dd1ee475b9193b07938e7706c \
  "${ROOT}/deployment/models/s10_lidar_encoder.onnx"
check_sha256 "official 20260828 low-level ONNX" \
  92db62c118c4ebad3da8bfedb89691f8caae8803966b472b4ce541e1a32f2d0d \
  "${ROOT}/src/S10_sdk_deploy/policy/policy.onnx"
check "runtime route map" test -f \
  "${ROOT}/deployment/maps/current/localization_map_manifest.json"
check "S10 reachable" ping -c 1 -W 1 "${S10_ROBOT_ADDRESS:-10.21.33.103}"
check "front Airy reachable" ping -c 1 -W 1 "${S10_FRONT_LIDAR_ADDRESS:-10.21.33.201}"
check "rear Airy reachable" ping -c 1 -W 1 "${S10_REAR_LIDAR_ADDRESS:-10.21.33.202}"
check "Airy multicast route" bash -c "ip route | grep -q '224.10.10.0/24'"

if (( failures > 0 )); then
  echo "S10_HARDWARE_ENVIRONMENT_FAILED count=${failures}" >&2
  exit 2
fi
echo "S10_HARDWARE_ENVIRONMENT_OK"
