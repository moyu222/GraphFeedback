#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
run_root="$project_root/experiments/outputs/random60v1"
data_dir="$run_root/citeseer"

count_lines() {
  local path="$1"
  if [[ -f "$path" ]]; then
    wc -l < "$path" | tr -d ' '
  else
    printf '0'
  fi
}

printf 'selection=%s/60\n' "$(count_lines "$data_dir/selection.jsonl")"
printf 'base_evaluated=%s/180\n' "$(count_lines "$data_dir/evaluated.jsonl")"
printf 'feedback_evaluated=%s/120\n' "$(count_lines "$data_dir/feedback_evaluated.jsonl")"

if [[ -f "$data_dir/summary.csv" ]]; then
  printf '\nLatest summary:\n'
  cat "$data_dir/summary.csv"
fi

if [[ -f "$run_root/run_console.log" ]]; then
  printf '\nLast 20 log lines:\n'
  tail -n 20 "$run_root/run_console.log"
fi
