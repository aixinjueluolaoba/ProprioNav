#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/diana/fishing/blind_nav_rl_reward_code_20260521
TRAIN_DIR=/home/diana/fishing/blind_nav_rl/runs/目标8显式脱困v10b9短训_v1002
WATCH_DIR=/home/diana/fishing/blind_nav_rl/runs/目标8显式脱困v10b9短训_checkpoint评估_v1002

mkdir -p "$TRAIN_DIR" "$WATCH_DIR"

if test -f "$TRAIN_DIR/训练.pid" && ps -p "$(cat "$TRAIN_DIR/训练.pid")" >/dev/null 2>&1; then
  echo "train_already_running pid=$(cat "$TRAIN_DIR/训练.pid")"
else
  TRAIN_CMD="
    cd $ROOT &&
    source ~/miniconda3/etc/profile.d/conda.sh &&
    conda activate ML &&
    python train_target8_v10b9_curriculum.py \
      --output-dir $TRAIN_DIR \
      --stage1-episodes 2000 \
      --stage2a-episodes 1000 \
      --stage2-episodes 2000 \
      --stage3-episodes 1000 \
      --n-envs 96 \
      --eval-episodes 20 \
      --video-episodes 0 \
      --device cpu \
      --torch-threads 1 \
      --seed-base 10021000 \
      --eval-seed-base 10031000 \
      --checkpoint-interval 500 \
      --progress-interval 250
  "
  setsid bash -lc "$TRAIN_CMD" > "$TRAIN_DIR/训练.log" 2>&1 &
  echo $! > "$TRAIN_DIR/训练.pid"
  echo "train_started pid=$(cat "$TRAIN_DIR/训练.pid")"
fi

if test -f "$WATCH_DIR/watcher.pid" && ps -p "$(cat "$WATCH_DIR/watcher.pid")" >/dev/null 2>&1; then
  echo "watcher_already_running pid=$(cat "$WATCH_DIR/watcher.pid")"
else
  WATCH_CMD="
    cd $ROOT &&
    source ~/miniconda3/etc/profile.d/conda.sh &&
    conda activate ML &&
    python watch_v10b9_checkpoints.py \
      --run-dir $TRAIN_DIR \
      --out-dir $WATCH_DIR \
      --curriculum \
      --checkpoint-step 500 \
      --stage1-max-checkpoint 2000 \
      --stage2a-max-checkpoint 1000 \
      --stage2-max-checkpoint 2000 \
      --stage3-max-checkpoint 1000
  "
  setsid bash -lc "$WATCH_CMD" > "$WATCH_DIR/watcher.log" 2>&1 &
  echo $! > "$WATCH_DIR/watcher.pid"
  echo "watcher_started pid=$(cat "$WATCH_DIR/watcher.pid")"
fi

echo "train_dir=$TRAIN_DIR"
echo "watch_dir=$WATCH_DIR"
