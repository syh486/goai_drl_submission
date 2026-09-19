#!/usr/bin/env bash

# This file is sourced by deployment entry points. Do not enable `set -u`: ROS
# setup scripts may reference unset variables.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
HARDWARE_ENV="${S10_HARDWARE_ENV:-${ROOT}/deployment/config/hardware.env}"
if [[ -f "${HARDWARE_ENV}" ]]; then
  set -a
  source "${HARDWARE_ENV}"
  set +a
fi

if [[ -n "${S10_CONDA_ENV:-}" ]]; then
  source "${HOME}/miniconda3/etc/profile.d/conda.sh"
  conda activate "${S10_CONDA_ENV}"
fi

ROS_SETUP="${S10_ROS_SETUP:-/opt/ros/jazzy/setup.bash}"
if [[ ! -f "${ROS_SETUP}" ]]; then
  echo "ROS2 setup not found: ${ROS_SETUP}" >&2
  return 2 2>/dev/null || exit 2
fi
source "${ROS_SETUP}"
if [[ -f "${ROOT}/install/setup.bash" ]]; then
  source "${ROOT}/install/setup.bash"
fi

export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-0}"
DEFAULT_FASTDDS_PROFILE="${ROOT}/src/dual_airy_merger/config/fastdds_ethernet.xml"
if [[ -n "${FASTRTPS_DEFAULT_PROFILES_FILE:-}" ]]; then
  export FASTRTPS_DEFAULT_PROFILES_FILE
elif [[ -f "${DEFAULT_FASTDDS_PROFILE}" ]]; then
  export FASTRTPS_DEFAULT_PROFILES_FILE="${DEFAULT_FASTDDS_PROFILE}"
else
  unset FASTRTPS_DEFAULT_PROFILES_FILE
fi
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
