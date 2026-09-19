#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 <log-file> <physical-gpu-index> <label>" >&2
  exit 2
fi

log_file="$1"
physical_gpu="$2"
label="$3"

while true; do
  clear
  echo "[$label] $(date --iso-8601=seconds)"
  nvidia-smi -i "$physical_gpu" --query-gpu=index,memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv,noheader
  echo
  if [[ -f "$log_file" ]]; then
    tail -n 80 "$log_file"
  else
    echo "waiting for $log_file"
  fi
  sleep 10
done
