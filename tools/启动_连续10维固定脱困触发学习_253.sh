#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/weiaokang/fishing/blind_nav_rl}
RUN_DIR=${RUN_DIR:-$ROOT/runs/连续10维固定脱困降误触_253_10k}
EXPERIMENT=${EXPERIMENT:-rppo_small_lstm128x1_observable10_target8_continuous_fixed_recovery_v3}
EPISODES=${EPISODES:-10000}
N_ENVS=${N_ENVS:-128}
THREADS=${THREADS:-4}
DEVICE=${DEVICE:-cpu}
EVAL_EPISODES=${EVAL_EPISODES:-20}
VIDEO_EPISODES=${VIDEO_EPISODES:-3}
SEED_BASE=${SEED_BASE:-9990000}
EVAL_SEED_BASE=${EVAL_SEED_BASE:-9991000}

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
  ulimit -n 4096 &&
  source /home/weiaokang/ML/bin/activate &&
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
