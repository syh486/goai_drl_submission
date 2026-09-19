#!/usr/bin/env bash
set -eo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
source "${ROOT}/deployment/scripts/hardware/hardware_env.sh"
cd "${ROOT}"

POLICY="${ROOT}/src/S10_sdk_deploy/policy/policy.onnx"
OFFICIAL_POLICY="${ROOT}/src/S10_sdk_deploy/policy/policy_official_20260828.onnx"
EXPECTED_SHA256="92db62c118c4ebad3da8bfedb89691f8caae8803966b472b4ce541e1a32f2d0d"

for model in "${POLICY}" "${OFFICIAL_POLICY}"; do
  if [[ ! -f "${model}" ]]; then
    echo "S10_LOW_LEVEL_REFUSED missing_model=${model}" >&2
    exit 2
  fi
  actual_sha256="$(sha256sum "${model}" | awk '{print $1}')"
  if [[ "${actual_sha256}" != "${EXPECTED_SHA256}" ]]; then
    echo "S10_LOW_LEVEL_REFUSED model_sha256_mismatch=${model}" >&2
    echo "expected=${EXPECTED_SHA256} actual=${actual_sha256}" >&2
    exit 2
  fi
done

if ! cmp -s "${POLICY}" "${OFFICIAL_POLICY}"; then
  echo "S10_LOW_LEVEL_REFUSED policy.onnx_differs_from_official_archive" >&2
  exit 2
fi

echo "S10_LOW_LEVEL_OFFICIAL_POLICY_OK sha256=${EXPECTED_SHA256}"

exec ros2 run s10_sdk_deploy rl_deploy "$@"
