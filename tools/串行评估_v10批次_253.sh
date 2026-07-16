#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/weiaokang/fishing/blind_nav_rl}
BATCH_ROOT=${BATCH_ROOT:-}
WATCH_INTERVAL=${WATCH_INTERVAL:-45}
CHECKPOINT_INTERVAL=${CHECKPOINT_INTERVAL:-250}
MAX_CHECKPOINT=${MAX_CHECKPOINT:-250}
RUN_GLOB=${RUN_GLOB:-*_ep*}
PYTHON_BIN=${PYTHON_BIN:-/home/weiaokang/ML/bin/python}
LOG_NAME=${LOG_NAME:-手动串行评估.log}

if test -z "$BATCH_ROOT"; then
  echo "missing_required_env BATCH_ROOT" >&2
  exit 1
fi

cd "$ROOT"

if test ! -d "$BATCH_ROOT"; then
  echo "batch_root_not_found path=$BATCH_ROOT" >&2
  exit 1
fi

shopt -s nullglob
runs=("$BATCH_ROOT"/$RUN_GLOB)
shopt -u nullglob

if test ${#runs[@]} -eq 0; then
  echo "no_runs_found batch_root=$BATCH_ROOT glob=$RUN_GLOB" >&2
  exit 1
fi

for run_dir in "${runs[@]}"; do
  if test ! -d "$run_dir"; then
    continue
  fi
  case "$(basename "$run_dir")" in
    *_checkpoint评估)
      continue
      ;;
  esac

  watch_dir="${run_dir}_checkpoint评估"
  mkdir -p "$watch_dir"
  log_path="$watch_dir/$LOG_NAME"
  summary_path="$watch_dir/checkpoint_eval_summary.csv"

  if test -f "$summary_path"; then
    echo "skip_existing_summary run_dir=$run_dir summary=$summary_path"
    continue
  fi

  echo "serial_eval_start run_dir=$run_dir watch_dir=$watch_dir"
  bash -lc "
    cd '$ROOT' &&
    source /home/weiaokang/ML/bin/activate &&
    '$PYTHON_BIN' watch_v10_checkpoints.py \
      --run-dir '$run_dir' \
      --out-dir '$watch_dir' \
      --curriculum \
      --interval '$WATCH_INTERVAL' \
      --checkpoint-step '$CHECKPOINT_INTERVAL' \
      --max-checkpoint '$MAX_CHECKPOINT'
  " >"$log_path" 2>&1
  echo "serial_eval_done run_dir=$run_dir summary=$summary_path"
done
