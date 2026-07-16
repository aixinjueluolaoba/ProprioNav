#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/diana/fishing/blind_nav_rl
RUN_DIR=$ROOT/runs/目标8宏动作库v11b卡住探索短训_v1002
WATCH_DIR=$ROOT/runs/目标8宏动作库v11b_checkpoint评估_v1002

mkdir -p "$WATCH_DIR"

if test -f "$WATCH_DIR/watcher.pid" && ps -p "$(cat "$WATCH_DIR/watcher.pid")" >/dev/null 2>&1; then
  echo "watcher_already_running pid=$(cat "$WATCH_DIR/watcher.pid")"
else
  WATCH_CMD="
    cd $ROOT &&
    source ~/miniconda3/etc/profile.d/conda.sh &&
    conda activate ML &&
    python watch_v11_checkpoints.py \
      --run-dir $RUN_DIR \
      --out-dir $WATCH_DIR \
      --curriculum \
      --max-checkpoint 2000 \
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

echo "run_dir=$RUN_DIR"
echo "watch_dir=$WATCH_DIR"
