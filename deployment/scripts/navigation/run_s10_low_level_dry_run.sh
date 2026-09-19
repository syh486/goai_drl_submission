#!/usr/bin/env bash
set -eo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
source "${ROOT}/deployment/scripts/hardware/hardware_env.sh"
cd "${ROOT}"

# This target is structurally read-only: it does not link any JointsDataCmd
# code and creates no /JOINTS_CMD publisher.
exec ros2 run s10_sdk_deploy s10_low_level_dry_run "$@"
