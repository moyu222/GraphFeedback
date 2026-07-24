#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
run_id="random60v1"
log_dir="$project_root/experiments/outputs/$run_id"
mkdir -p "$log_dir"

ps_script="$(cygpath -w "$project_root/experiments/run_local.ps1")"
config_path="$(cygpath -w "$project_root/experiments/config.random60.yaml")"

powershell.exe -NoProfile -ExecutionPolicy Bypass \
  -File "$ps_script" \
  -Mode Confirmatory \
  -RunId "$run_id" \
  -SampleSize 60 \
  -ConfigPath "$config_path" 2>&1 |
  tee -a "$log_dir/run_console.log"
