#!/usr/bin/env bash
set -euo pipefail

LOCAL_ROOT=/home/diana/fishing/blind_nav_rl
REMOTE_ROOT=/home/diana/fishing/blind_nav_rl
TRAIN_DIR=$REMOTE_ROOT/runs/目标8宏动作库v11d压步数探针_v1002
OVERRIDES_LOCAL=$LOCAL_ROOT/configs/v11/方案D_主线小幅压步数.json
OVERRIDES_REMOTE=$REMOTE_ROOT/configs/v11/方案D_主线小幅压步数.json

copy_file() {
  local rel="$1"
  ssh v1002 "mkdir -p '$REMOTE_ROOT/$(dirname "$rel")'"
  scp "$LOCAL_ROOT/$rel" "v1002:$REMOTE_ROOT/$rel"
}

copy_file train_target8_v11_macro_library_curriculum.py
copy_file train_target8_v10_curriculum.py
copy_file benchmark_state_dims_10k.py
copy_file blind_nav_rl/env.py
copy_file "configs/v11/方案D_主线小幅压步数.json"

ssh v1002 "mkdir -p '$TRAIN_DIR'"

if ssh v1002 "test -f '$TRAIN_DIR/训练.pid' && ps -p \"\$(cat '$TRAIN_DIR/训练.pid')\" >/dev/null 2>&1"; then
  echo "train_already_running pid=$(ssh v1002 "cat '$TRAIN_DIR/训练.pid'")"
else
  TRAIN_CMD="
    cd $REMOTE_ROOT &&
    source ~/miniconda3/etc/profile.d/conda.sh &&
    conda activate ML &&
    python train_target8_v11_macro_library_curriculum.py \
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
      --seed-base 10241000 \
      --eval-seed-base 10251000 \
      --checkpoint-interval 500 \
      --progress-interval 250 \
      --env-overrides-file '$OVERRIDES_REMOTE'
  "
  ssh v1002 "setsid bash -lc \"$TRAIN_CMD\" > '$TRAIN_DIR/训练.log' 2>&1 & echo \$! > '$TRAIN_DIR/训练.pid'"
  echo "train_started pid=$(ssh v1002 "cat '$TRAIN_DIR/训练.pid'")"
fi

echo "train_dir=$TRAIN_DIR"
