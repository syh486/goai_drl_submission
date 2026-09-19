#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT}"

if [[ -n "${S10_PYTHON:-}" ]]; then
  exec "${S10_PYTHON}" scripts/verify_install.py
fi
if python3 -c 'import torch, mujoco, warp, onnxruntime, kiss_icp' >/dev/null 2>&1; then
  exec python3 scripts/verify_install.py
fi
if command -v conda >/dev/null 2>&1; then
  if [[ -n "${S10_CONDA_ENV:-}" ]]; then
    exec conda run --no-capture-output -n "${S10_CONDA_ENV}" \
      python scripts/verify_install.py
  fi
  for env_name in race s10_sru_demo; do
    if conda env list | awk '{print $1}' | grep -qx "${env_name}"; then
      exec conda run --no-capture-output -n "${env_name}" \
        python scripts/verify_install.py
    fi
  done
fi
echo "Training runtime not found. Run ./scripts/setup_ubuntu24.sh first." >&2
exit 1
