#!/usr/bin/env bash
set -eo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
BAG="${1:?usage: run_glim_mapping.sh BAG OUTPUT_DIR}"
OUTPUT="${2:?usage: run_glim_mapping.sh BAG OUTPUT_DIR}"
PREFIX="${S10_SLAM_PREFIX:-${HOME}/.local/s10_slam}"
CONFIG="${ROOT}/deployment/config/glim_s10_lio"
[[ ! -e "${OUTPUT}" ]] || {
  echo "Refusing to overwrite GLIM output: ${OUTPUT}" >&2
  exit 2
}

source /opt/ros/humble/setup.bash
source "${PREFIX}/glim/share/glim/local_setup.bash"
source "${PREFIX}/glim_ros/share/glim_ros/local_setup.bash"
if [[ -f "${PREFIX}/glim_ext/share/glim_ext/local_setup.bash" ]]; then
  source "${PREFIX}/glim_ext/share/glim_ext/local_setup.bash"
fi
export LD_LIBRARY_PATH="${PREFIX}/glim/lib:${PREFIX}/glim_ros/lib:${PREFIX}/glim_ext/lib:${LD_LIBRARY_PATH:-}"

exec "${PREFIX}/glim_ros/lib/glim_ros/glim_rosbag" "${BAG}" --ros-args \
  -p config_path:="${CONFIG}" \
  -p auto_quit:=true \
  -p dump_path:="${OUTPUT}"
