#!/usr/bin/env bash
set -eo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
HARDWARE_ENV="${S10_HARDWARE_ENV:-${ROOT}/deployment/config/hardware.env}"
if [[ -f "${HARDWARE_ENV}" ]]; then
  set -a
  source "${HARDWARE_ENV}"
  set +a
fi
SDK_DIR="${ROOT}/external/rsLiDAR_sdk"
ENV_NAME="${S10_CONDA_ENV:-}"
RACE_PREFIX="${HOME}/miniconda3/envs/${ENV_NAME}"
SDK_COMMIT="8b4b4b7ff910799260347821084c59e1c73d50d5"
if [[ "$(uname -m)" == "aarch64" ]]; then
  BUILD_PLATFORM="${S10_BUILD_PLATFORM:-arm}"
else
  BUILD_PLATFORM="${S10_BUILD_PLATFORM:-x86}"
fi

ROS_SETUP="${S10_ROS_SETUP:-/opt/ros/jazzy/setup.bash}"
if [[ ! -f "${ROS_SETUP}" && -f /opt/ros/humble/setup.bash ]]; then
  ROS_SETUP=/opt/ros/humble/setup.bash
fi
source "${ROS_SETUP}"
if [[ -f /usr/include/pcap.h ]]; then
  :
elif [[ -n "${ENV_NAME}" && -f "${RACE_PREFIX}/include/pcap.h" ]]; then
  export CPLUS_INCLUDE_PATH="${RACE_PREFIX}/include${CPLUS_INCLUDE_PATH:+:${CPLUS_INCLUDE_PATH}}"
  export LIBRARY_PATH="${RACE_PREFIX}/lib${LIBRARY_PATH:+:${LIBRARY_PATH}}"
else
  echo "Missing libpcap headers. Run: conda install -n race -c conda-forge libpcap" >&2
  exit 2
fi
mkdir -p "${ROOT}/external"
if [[ ! -d "${SDK_DIR}/.git" ]]; then
  git clone --recurse-submodules --branch v1.5.20 --depth 1 \
    https://github.com/RoboSense-LiDAR/rsLiDAR_sdk.git "${SDK_DIR}"
fi

# A field reinstall must work without Internet access. Only fetch when the
# pinned commit is absent or an explicit refresh was requested.
if [[ "${S10_UPDATE_AIRY_SDK:-0}" == "1" ]]; then
  git -C "${SDK_DIR}" fetch --depth 1 origin tag v1.5.20
fi
if ! git -C "${SDK_DIR}" cat-file -e "${SDK_COMMIT}^{commit}" 2>/dev/null; then
  git -C "${SDK_DIR}" fetch --depth 1 origin tag v1.5.20
fi
if [[ "$(git -C "${SDK_DIR}" rev-parse HEAD)" != "${SDK_COMMIT}" ]]; then
  git -C "${SDK_DIR}" checkout --detach "${SDK_COMMIT}"
fi
if [[ ! -e "${SDK_DIR}/src/rs_driver/.git" ]]; then
  git -C "${SDK_DIR}" submodule update --init --recursive
fi

# Upstream defaults to XYZI, which drops the per-point timestamps required
# for deskew. Keep this explicit and fail if their CMake layout changes.
if grep -q '^set(POINT_TYPE XYZI)' "${SDK_DIR}/CMakeLists.txt"; then
  sed -i 's/^set(POINT_TYPE XYZI)$/set(POINT_TYPE XYZIRT)/' "${SDK_DIR}/CMakeLists.txt"
fi
grep -q '^set(POINT_TYPE XYZIRT)' "${SDK_DIR}/CMakeLists.txt"

cd "${ROOT}"
/usr/bin/colcon build --symlink-install \
  --cmake-clean-cache \
  --cmake-clean-first \
  --cmake-force-configure \
  --packages-select rslidar_msg drdds rslidar_sdk dual_airy_merger s10_sdk_deploy \
  --cmake-args \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_PLATFORM="${BUILD_PLATFORM}" \
    -DS10_COMMAND_INTERFACE="${S10_COMMAND_INTERFACE:-dds}" \
    -DPython3_EXECUTABLE=/usr/bin/python3 \
    -DPYTHON_EXECUTABLE=/usr/bin/python3

echo "HARDWARE_ROS2_SETUP_OK"
