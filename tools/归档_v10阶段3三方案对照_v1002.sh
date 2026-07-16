#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/diana/fishing/blind_nav_rl}
BATCH_ROOT=${BATCH_ROOT:-$ROOT/runs/目标8显式脱困v10阶段3三方案对照_v1002}
MANIFEST_PATH=${MANIFEST_PATH:-$BATCH_ROOT/批次参数.csv}
ARCHIVE_ROOT=${ARCHIVE_ROOT:-$ROOT/runs/目标8显式脱困v10阶段3三方案对照_最终归档_v1002}
MIN_CHECKPOINT=${MIN_CHECKPOINT:-500}
EPISODES=${EPISODES:-20}
VIDEO_EPISODES=${VIDEO_EPISODES:-20}
DEVICE=${DEVICE:-cpu}
SCORE_PROFILE=${SCORE_PROFILE:-pressure_priority}

cd "$ROOT"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ML

mkdir -p "$ARCHIVE_ROOT"

while IFS=, read -r run_id variant_name env_overrides_file run_name train_dir watch_dir seed_base eval_seed_base stage3_episodes checkpoint_interval resume_model <&3; do
  if test "$run_id" = "run_id"; then
    continue
  fi
  out_dir="$ARCHIVE_ROOT/$run_name"
  if test -f "$out_dir/最终选择说明.md"; then
    echo "skip_finalized run=$run_name"
    continue
  fi
  ROOT="$ROOT" \
  RUN_DIR="$train_dir" \
  CHECKPOINT_EVAL_DIR="$watch_dir" \
  OUT_DIR="$out_dir" \
  MIN_CHECKPOINT="$MIN_CHECKPOINT" \
  EPISODES="$EPISODES" \
  VIDEO_EPISODES="$VIDEO_EPISODES" \
  DEVICE="$DEVICE" \
  SCORE_PROFILE="$SCORE_PROFILE" \
  bash launch_v10_finalizer.sh </dev/null
done 3< "$MANIFEST_PATH"

python summarize_v10_variant_batch.py \
  --manifest "$MANIFEST_PATH" \
  --archive-root "$ARCHIVE_ROOT" \
  --out-dir "$ARCHIVE_ROOT" \
  --min-checkpoint "$MIN_CHECKPOINT" \
  --score-profile "$SCORE_PROFILE"

echo "archive_root=$ARCHIVE_ROOT"
