#!/usr/bin/env bash
set -eo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
source "${ROOT}/deployment/scripts/hardware/hardware_env.sh"
cd "${ROOT}"

if pgrep -f '(^|/)(rl_deploy)( |$)|ros2 run s10_sdk_deploy rl_deploy' >/dev/null 2>&1; then
  echo "Refusing gamepad collection while the repository low-level runner is active." >&2
  echo "Its DDS interface maps G12 A/B to robot motion states; stop it and use factory teleop." >&2
  exit 2
fi

COLLECTOR_ARGS=()
MAP_SESSION=""
while (( "$#" )); do
  case "$1" in
    --map-session)
      if (( "$#" < 2 )); then
        echo "--map-session requires a directory" >&2
        exit 2
      fi
      MAP_SESSION="$2"
      shift 2
      ;;
    --map-session=*)
      MAP_SESSION="${1#--map-session=}"
      shift
      ;;
    *)
      COLLECTOR_ARGS+=("$1")
      shift
      ;;
  esac
done

for argument in "${COLLECTOR_ARGS[@]}"; do
  if [[ "${argument}" == "--resume" ]]; then
    echo "Do not use --resume with the combined stack: restarting localization changes route_map." >&2
    echo "Keep the existing localizer running and use run_waypoint_collector.sh --resume instead." >&2
    exit 2
  fi
done

OUTPUT=""
EXPECT_OUTPUT=0
for argument in "${COLLECTOR_ARGS[@]}"; do
  if [[ "${EXPECT_OUTPUT}" == "1" ]]; then
    OUTPUT="${argument}"
    EXPECT_OUTPUT=0
  elif [[ "${argument}" == "--output" ]]; then
    EXPECT_OUTPUT=1
  elif [[ "${argument}" == --output=* ]]; then
    OUTPUT="${argument#--output=}"
  fi
done
if [[ -z "${OUTPUT}" ]]; then
  OUTPUT="${ROOT}/deployment/routes/route.yaml"
elif [[ "${OUTPUT}" != /* ]]; then
  OUTPUT="${ROOT}/${OUTPUT}"
fi
ANCHOR="${OUTPUT%.*}.anchor.npz"
if [[ -e "${OUTPUT}" || -e "${ANCHOR}" ]]; then
  echo "Collection output already exists; choose a new --output path:" >&2
  echo "  route=${OUTPUT}" >&2
  echo "  anchor=${ANCHOR}" >&2
  exit 2
fi
if [[ -n "${MAP_SESSION}" ]]; then
  if [[ "${MAP_SESSION}" != /* ]]; then
    MAP_SESSION="${ROOT}/${MAP_SESSION}"
  fi
  if [[ -d "${MAP_SESSION}" && -n "$(find "${MAP_SESSION}" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    echo "Mapping output is not empty: ${MAP_SESSION}" >&2
    exit 2
  fi
fi

LOCALIZER_ARGS=(
  --config "${ROOT}/deployment/config/hardware_localization.yaml"
  --localization-only
  --record-start-anchor "${ANCHOR}"
)
if [[ -n "${MAP_SESSION}" ]]; then
  LOCALIZER_ARGS+=(--record-map "${MAP_SESSION}")
fi
"${S10_PYTHON:-python3}" -m deployment.navigation.ros2_node "${LOCALIZER_ARGS[@]}" &
LOCALIZER_PID=$!
cleanup() {
  kill "${LOCALIZER_PID}" 2>/dev/null || true
  wait "${LOCALIZER_PID}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

"${S10_PYTHON:-python3}" -m deployment.waypoints.ros2_collector \
  "${COLLECTOR_ARGS[@]}" --anchor-file "${ANCHOR}"
