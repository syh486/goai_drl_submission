#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 <a|b|c|official|hybrid> <physical-gpu-index>" >&2
  exit 2
fi

variant="$1"
physical_gpu="$2"
case "$variant" in
  a|b|c|official|hybrid) ;;
  *) echo "invalid variant: $variant" >&2; exit 2 ;;
esac

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
remote_root="${S10_REMOTE_ROOT:-$(dirname "$repo_root")}"
isaaclab_root="${S10_ISAACLAB_ROOT:-/home/et25-hebl/sru_project_4.5_remote/IsaacLab}"
isaacsim_python="${S10_ISAACSIM_PYTHON:-/home/et25-hebl/isaacsim-4.5.0-recovery/python.sh}"
log_dir="$remote_root/run_logs"
log_file="$log_dir/s10_strict_him_${variant}_gpu${physical_gpu}.log"

num_envs="${S10_NUM_ENVS:-4096}"
max_iterations="${S10_MAX_ITERATIONS:-20000}"
seed="${S10_SEED:-42}"
initial_noise_std="${S10_INITIAL_NOISE_STD:-1.0}"
case "$variant" in
  official)
    default_entropy_coef="0.003"
    default_learning_rate="0.001"
    default_save_interval="100"
    ;;
  hybrid)
    default_entropy_coef="0.005"
    default_learning_rate="0.0005"
    default_save_interval="100"
    ;;
  *)
    default_entropy_coef="0.01"
    default_learning_rate="0.0003"
    default_save_interval="250"
    ;;
esac
entropy_coef="${S10_ENTROPY_COEF:-$default_entropy_coef}"
learning_rate="${S10_LEARNING_RATE:-$default_learning_rate}"
save_interval="${S10_SAVE_INTERVAL:-$default_save_interval}"

mkdir -p "$log_dir"
export PYTHONPATH="$repo_root:/home/et25-hebl/sru_project_4.5_remote/sru-navigation-learning:$isaaclab_root/source/isaaclab:$isaaclab_root/source/isaaclab_assets:$isaaclab_root/source/isaaclab_rl:$isaaclab_root/source/isaaclab_tasks${PYTHONPATH:+:$PYTHONPATH}"
export OMNI_KIT_ACCEPT_EULA=Y
export CUDA_VISIBLE_DEVICES="$physical_gpu"

{
  echo "[LAUNCH] date=$(date --iso-8601=seconds) m20_reference_commit=6d317dfb33060226139e38600510e1751372eb5d variant=$variant physical_gpu=$physical_gpu"
  echo "[LAUNCH] envs=$num_envs max_iterations=$max_iterations save_interval=$save_interval seed=$seed"
  echo "[LAUNCH] initial_noise_std=$initial_noise_std entropy_coef=$entropy_coef learning_rate=$learning_rate resume=none model=blind_him"
  "$isaacsim_python" -u "$repo_root/scripts/s10/train_him.py" \
    --variant "$variant" \
    --num-envs "$num_envs" \
    --max-iterations "$max_iterations" \
    --save-interval "$save_interval" \
    --seed "$seed" \
    --initial-noise-std "$initial_noise_std" \
    --entropy-coef "$entropy_coef" \
    --learning-rate "$learning_rate" \
    --run-name "remote_strict_gpu${physical_gpu}" \
    --headless \
    --device cuda:0
} 2>&1 | tee "$log_file"
