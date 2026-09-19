#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
BAG="${1:?usage: run_glim_tightly_coupled_replay.sh BAG OUTPUT_DIR [PRIOR_MAP]}"
OUTPUT="${2:?usage: run_glim_tightly_coupled_replay.sh BAG OUTPUT_DIR [PRIOR_MAP]}"
PRIOR_MAP="${3:-${S10_PRIOR_MAP:-}}"
PREFIX="${S10_SLAM_PREFIX:-${HOME}/.local/s10_slam}"
CONFIG="${ROOT}/deployment/config/glim_s10_tightly_coupled"

[[ -n "${PRIOR_MAP}" ]] || {
  echo "A prior-map PLY is required as argument 3 or S10_PRIOR_MAP" >&2
  exit 2
}
[[ -f "${PRIOR_MAP}" ]] || {
  echo "Prior map does not exist: ${PRIOR_MAP}" >&2
  exit 2
}
[[ ! -e "${OUTPUT}" ]] || {
  echo "Refusing to overwrite GLIM output: ${OUTPUT}" >&2
  exit 2
}

export PATH=/usr/bin:/bin:/usr/sbin:/sbin
unset CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_PROMPT_MODIFIER CONDA_EXE CONDA_PYTHON_EXE
set +u
source /opt/ros/humble/setup.bash
set -u

export LD_LIBRARY_PATH="${PREFIX}/glim_prior_map_localizer_tightly_coupled/lib:${PREFIX}/glim_ros2_tightly_coupled/lib:${PREFIX}/glil_tightly_coupled/lib:${PREFIX}/kiss_matcher/lib:/usr/local/lib:/opt/ros/humble/lib:/opt/ros/humble/lib/x86_64-linux-gnu"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export GLIM_PRIOR_MAP_PATH="$(realpath "${PRIOR_MAP}")"
export GLIM_PRIOR_MAP_TIGHTLY_COUPLED=1
export GLIM_PRIOR_MAP_BOOTSTRAP_CENTER_X=0.0
export GLIM_PRIOR_MAP_BOOTSTRAP_CENTER_Y=0.0
export GLIM_PRIOR_MAP_BOOTSTRAP_CENTER_Z=0.0
export GLIM_PRIOR_MAP_BOOTSTRAP_YAW_DEG=0.0
export GLIM_PRIOR_MAP_FACTOR_NUM_THREADS=1
export GLIM_PRIOR_MAP_FACTOR_FRAME_STRIDE=2
export GLIM_PRIOR_MAP_FACTOR_OVERLAP_STRIDE=10

exec "${PREFIX}/glim_ros2_tightly_coupled/lib/glim_ros/glim_rosbag" \
  "$(realpath "${BAG}")" --ros-args \
  -p config_path:="${CONFIG}" \
  -p auto_quit:=true \
  -p dump_path:="$(realpath -m "${OUTPUT}")"
