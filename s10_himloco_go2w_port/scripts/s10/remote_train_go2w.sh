#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 <physical-gpu-index>" >&2
  exit 2
fi

physical_gpu="$1"
if [[ "$physical_gpu" == "1" ]]; then
  echo "GPU1 is excluded because it is known faulty" >&2
  exit 2
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
isaaclab_root="${S10_ISAACLAB_ROOT:-/home/et25-hebl/sru_project_4.5_remote/IsaacLab}"
isaacsim_python="${S10_ISAACSIM_PYTHON:-/home/et25-hebl/isaacsim-4.5.0-recovery/python.sh}"
rsl_rl_root="${S10_RSL_RL_ROOT:-/home/et25-hebl/sru_project_4.5_remote/sru-navigation-learning}"
num_envs="${S10_NUM_ENVS:-4096}"
max_iterations="${S10_MAX_ITERATIONS:-20000}"
save_interval="${S10_SAVE_INTERVAL:-1000}"
seed="${S10_SEED:-1}"
log_dir="$repo_root/run_logs"
log_file="$log_dir/s10_go2w_him_gpu${physical_gpu}.log"

mkdir -p "$log_dir"
if [[ ! -f "$rsl_rl_root/rsl_rl/env/vec_env.py" ]]; then
  echo "rsl_rl VecEnv source not found: $rsl_rl_root/rsl_rl/env/vec_env.py" >&2
  exit 2
fi

export PYTHONPATH="$repo_root:$rsl_rl_root:$isaaclab_root/source/isaaclab:$isaaclab_root/source/isaaclab_assets:$isaaclab_root/source/isaaclab_rl:$isaaclab_root/source/isaaclab_tasks${PYTHONPATH:+:$PYTHONPATH}"
export OMNI_KIT_ACCEPT_EULA=Y
export CUDA_VISIBLE_DEVICES="$physical_gpu"

{
  echo "[LAUNCH] date=$(date --iso-8601=seconds) reference=TrackinBIT/HIMLoco-for-Go2W commit=011693738c61603c3f22f2bce755098dd36fa7eb"
  echo "[LAUNCH] physical_gpu=$physical_gpu envs=$num_envs iterations=$max_iterations save_interval=$save_interval seed=$seed"
  echo "[LAUNCH] fresh_start=true history=6x57 critic=262 rollout=48 entropy=0.005 lr=0.001"
  "$isaacsim_python" -u "$repo_root/scripts/s10/train_go2w_him.py" \
    --num-envs "$num_envs" \
    --max-iterations "$max_iterations" \
    --save-interval "$save_interval" \
    --seed "$seed" \
    --run-name "remote_gpu${physical_gpu}" \
    --headless \
    --device cuda:0
} 2>&1 | tee "$log_file"
