#!/usr/bin/env bash
set -euo pipefail

LOCAL_ROOT=/home/diana/fishing/blind_nav_rl
REMOTE_ROOT=/home/diana/fishing/blind_nav_rl
TRAIN_DIR=$REMOTE_ROOT/runs/目标8宏动作库v11d压步数探针_v1002
WATCH_DIR=$REMOTE_ROOT/runs/目标8宏动作库v11d压步数探针_v1002_watch

copy_file() {
  local rel="$1"
  ssh v1002 "mkdir -p '$REMOTE_ROOT/$(dirname "$rel")'"
  scp "$LOCAL_ROOT/$rel" "v1002:$REMOTE_ROOT/$rel"
}

copy_file watch_v11_checkpoints.py
copy_file eval_target8_sweep.py
copy_file eval_target8_pressure.py
copy_file benchmark_state_dims_10k.py
copy_file blind_nav_rl/env.py

ssh v1002 "mkdir -p '$WATCH_DIR'"

WATCH_CMD="
  cd $REMOTE_ROOT &&
  source ~/miniconda3/etc/profile.d/conda.sh &&
  conda activate ML &&
  python watch_v11_checkpoints.py \
    --run-dir '$TRAIN_DIR' \
    --out-dir '$WATCH_DIR' \
    --curriculum \
    --checkpoint-step 500 \
    --stage1-max-checkpoint 2000 \
    --stage2a-max-checkpoint 1000 \
    --stage2-max-checkpoint 2000 \
    --stage3-max-checkpoint 1000 \
    --interval 60
"

ssh v1002 "setsid bash -lc \"$WATCH_CMD\" > '$WATCH_DIR/watcher.log' 2>&1 & echo \$! > '$WATCH_DIR/watcher.pid'"
echo "watcher_started pid=$(ssh v1002 "cat '$WATCH_DIR/watcher.pid'")"
echo "watch_dir=$WATCH_DIR"
