#!/usr/bin/env bash
set -eo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
source "${ROOT}/deployment/scripts/hardware/hardware_env.sh"
if ! ldconfig -p 2>/dev/null | grep -q 'libpcap'; then
  export LD_LIBRARY_PATH="${HOME}/miniconda3/envs/race/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi

exec ros2 run rslidar_sdk rslidar_sdk_node --ros-args \
  -p config_path:="${ROOT}/src/dual_airy_merger/config/airy_dual.yaml"
