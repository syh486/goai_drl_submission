#!/usr/bin/env bash
set -eo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
SESSION_DIR="${1:?usage: export_mapping_rosbag.sh SESSION_DIR [OUTPUT_BAG]}"
OUTPUT_BAG="${2:-${SESSION_DIR%/}/rosbag2}"

source /opt/ros/humble/setup.bash
cd "${ROOT}"
exec /usr/bin/python3 -m deployment.mapping.export_mapping_rosbag \
  "${SESSION_DIR}" "${OUTPUT_BAG}"
