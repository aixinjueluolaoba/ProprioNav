#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/weiaokang/fishing/blind_nav_rl}
RUN_DIR=${RUN_DIR:-$ROOT/runs/目标8显式脱困v10阶段3高压脱困_253}
WATCH_DIR=${WATCH_DIR:-$ROOT/runs/目标8显式脱困v10阶段3高压脱困_checkpoint评估_253}
INIT_POLICY_STATE_DICT=${INIT_POLICY_STATE_DICT:-$ROOT/runs/目标8显式脱困v10阶段3方案A保留线_253_阶段2起点/stage2_ckpt3000_policy_state_dict.pt}
ENV_OVERRIDES_FILE=${ENV_OVERRIDES_FILE:-$ROOT/configs/v10_stage3/方案D_高压脱困.json}
START_STAGE=${START_STAGE:-rppo_medium_lstm128x2_observable12_target8_discrete_recovery_v10_stage3}
END_STAGE=${END_STAGE:-rppo_medium_lstm128x2_observable12_target8_discrete_recovery_v10_stage3}
STAGE3_EPISODES=${STAGE3_EPISODES:-2000}
N_ENVS=${N_ENVS:-96}
THREADS=${THREADS:-1}
DEVICE=${DEVICE:-cpu}
EVAL_EPISODES=${EVAL_EPISODES:-20}
VIDEO_EPISODES=${VIDEO_EPISODES:-0}
SEED_BASE=${SEED_BASE:-9825000}
EVAL_SEED_BASE=${EVAL_SEED_BASE:-9925000}
CHECKPOINT_INTERVAL=${CHECKPOINT_INTERVAL:-500}
PROGRESS_INTERVAL=${PROGRESS_INTERVAL:-250}
WATCH_INTERVAL=${WATCH_INTERVAL:-60}

cd "$ROOT"
mkdir -p "$RUN_DIR" "$WATCH_DIR"
export MPLCONFIGDIR=${MPLCONFIGDIR:-"$RUN_DIR/图形缓存"}
mkdir -p "$MPLCONFIGDIR"

if test ! -f "$INIT_POLICY_STATE_DICT"; then
  echo "init_policy_state_dict_not_found path=$INIT_POLICY_STATE_DICT" >&2
  exit 1
fi

if test -f "$RUN_DIR/训练.pid" && ps -p "$(cat "$RUN_DIR/训练.pid")" >/dev/null 2>&1; then
  echo "train_already_running pid=$(cat "$RUN_DIR/训练.pid")"
else
  train_cmd="
    cd $ROOT &&
    ulimit -n 4096 &&
    source /home/weiaokang/ML/bin/activate &&
    ROOT=$ROOT \
    RUN_DIR=$RUN_DIR \
    INIT_POLICY_STATE_DICT=$INIT_POLICY_STATE_DICT \
    START_STAGE=$START_STAGE \
    END_STAGE=$END_STAGE \
    STAGE3_EPISODES=$STAGE3_EPISODES \
    N_ENVS=$N_ENVS \
    THREADS=$THREADS \
    DEVICE=$DEVICE \
    EVAL_EPISODES=$EVAL_EPISODES \
    VIDEO_EPISODES=$VIDEO_EPISODES \
    SEED_BASE=$SEED_BASE \
    EVAL_SEED_BASE=$EVAL_SEED_BASE \
    CHECKPOINT_INTERVAL=$CHECKPOINT_INTERVAL \
    PROGRESS_INTERVAL=$PROGRESS_INTERVAL \
    ENV_OVERRIDES_FILE=$ENV_OVERRIDES_FILE \
    MPLCONFIGDIR=$RUN_DIR/图形缓存 \
    bash launch_target8_v10_stage2_resume.sh
  "
  setsid bash -lc "$train_cmd" > "$RUN_DIR/阶段3训练.log" 2>&1 &
  echo $! > "$RUN_DIR/训练.pid"
  echo "train_started pid=$(cat "$RUN_DIR/训练.pid")"
fi

if test -f "$WATCH_DIR/watcher.pid" && ps -p "$(cat "$WATCH_DIR/watcher.pid")" >/dev/null 2>&1; then
  echo "watcher_already_running pid=$(cat "$WATCH_DIR/watcher.pid")"
else
  watch_cmd="
    cd $ROOT &&
    source /home/weiaokang/ML/bin/activate &&
    python watch_v10_checkpoints.py \
      --run-dir $RUN_DIR \
      --out-dir $WATCH_DIR \
      --curriculum \
      --interval $WATCH_INTERVAL \
      --checkpoint-step $CHECKPOINT_INTERVAL \
      --max-checkpoint $STAGE3_EPISODES
  "
  setsid bash -lc "$watch_cmd" > "$WATCH_DIR/watcher.log" 2>&1 &
  echo $! > "$WATCH_DIR/watcher.pid"
  echo "watcher_started pid=$(cat "$WATCH_DIR/watcher.pid")"
fi

echo "run_dir=$RUN_DIR"
echo "watch_dir=$WATCH_DIR"
