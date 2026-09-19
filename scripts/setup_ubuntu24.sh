#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="${S10_CONDA_ENV:-s10_sru_demo}"

if ! command -v conda >/dev/null 2>&1; then
  echo "Conda is required for the MuJoCo/Warp training environment." >&2
  exit 1
fi
if ! conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  conda create -y -n "${ENV_NAME}" python=3.10 pip
fi

conda run --no-capture-output -n "${ENV_NAME}" \
  python -m pip install torch==2.5.1 torchvision==0.20.1 \
  --index-url https://download.pytorch.org/whl/cu121
conda run --no-capture-output -n "${ENV_NAME}" \
  python -m pip install -r "${ROOT}/requirements.txt"

echo "Training environment '${ENV_NAME}' is ready."
echo "Run: S10_CONDA_ENV=${ENV_NAME} ${ROOT}/verify_install.sh"
