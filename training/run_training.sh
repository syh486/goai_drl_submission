#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 CONFIG_YAML [TRAINING_OVERRIDES...]" >&2
  exit 2
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="${S10_CONDA_ENV:-race}"
CONFIG="$1"
shift
if [[ "${CONFIG}" != /* ]]; then
  CONFIG="${ROOT}/${CONFIG}"
fi

cd "${ROOT}"
exec conda run --no-capture-output -n "${ENV_NAME}" \
  python -m training.train --config "${CONFIG}" "$@"
