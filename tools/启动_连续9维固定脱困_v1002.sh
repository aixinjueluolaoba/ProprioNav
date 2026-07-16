#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/diana/fishing/blind_nav_rl}
RUN_DIR=${RUN_DIR:-$ROOT/runs/连续9维固定脱困_v1002_10k}
EXPERIMENT=${EXPERIMENT:-rppo_small_lstm128x1_observable9_target8_continuous_fixed_recovery_v1}
EPISODES=${EPISODES:-10000}
N_ENVS=${N_ENVS:-128}
THREADS=${THREADS:-2}
DEVICE=${DEVICE:-cpu}
EVAL_EPISODES=${EVAL_EPISODES:-20}
VIDEO_EPISODES=${VIDEO_EPISODES:-3}
SEED_BASE=${SEED_BASE:-9970000}
EVAL_SEED_BASE=${EVAL_SEED_BASE:-9980000}
CONDA_SH=${CONDA_SH:-/home/diana/miniconda3/etc/profile.d/conda.sh}
CONDA_ENV=${CONDA_ENV:-ML}

cd "$ROOT"
mkdir -p "$RUN_DIR"
export MPLCONFIGDIR=${MPLCONFIGDIR:-"$RUN_DIR/图形缓存"}
mkdir -p "$MPLCONFIGDIR"

if test -f "$RUN_DIR/训练.pid" && ps -p "$(cat "$RUN_DIR/训练.pid")" >/dev/null 2>&1; then
  echo "train_already_running pid=$(cat "$RUN_DIR/训练.pid")"
  exit 0
fi

train_cmd="
  cd $ROOT &&
  source $CONDA_SH &&
  conda activate $CONDA_ENV &&
  python benchmark_state_dims_10k.py \
    --output-dir $RUN_DIR \
    --episodes $EPISODES \
    --n-envs $N_ENVS \
    --eval-episodes $EVAL_EPISODES \
    --video-episodes $VIDEO_EPISODES \
    --only $EXPERIMENT \
    --device $DEVICE \
    --torch-threads $THREADS \
    --seed-base $SEED_BASE \
    --eval-seed-base $EVAL_SEED_BASE
"

setsid bash -lc "$train_cmd" > "$RUN_DIR/训练.log" 2>&1 &
echo $! > "$RUN_DIR/训练.pid"
echo "train_started pid=$(cat "$RUN_DIR/训练.pid")"
echo "run_dir=$RUN_DIR"
