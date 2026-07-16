#!/usr/bin/env bash
set -euo pipefail

cd /home/diana/fishing/blind_nav_rl

TRAIN_DIR=/home/diana/fishing/blind_nav_rl/runs/目标8显式脱困v10阶段3优化版_v1002
WATCH_DIR=/home/diana/fishing/blind_nav_rl/runs/目标8显式脱困v10阶段3优化版_checkpoint评估_v1002
FINAL_DIR=/home/diana/fishing/blind_nav_rl/runs/目标8显式脱困v10阶段3优化版_最终归档_v1002
RESUME_MODEL=/home/diana/fishing/blind_nav_rl/runs/目标8显式脱困v10四阶段_v1002_110k/rppo_medium_lstm128x2_observable12_target8_discrete_recovery_v10_stage2/models/checkpoint_ep_03000.zip

mkdir -p "$TRAIN_DIR" "$WATCH_DIR" "$FINAL_DIR"

if test -f "$TRAIN_DIR/训练.pid" && ps -p "$(cat "$TRAIN_DIR/训练.pid")" >/dev/null 2>&1; then
  echo "train_already_running pid=$(cat "$TRAIN_DIR/训练.pid")"
else
  TRAIN_CMD="
    cd /home/diana/fishing/blind_nav_rl &&
    source ~/miniconda3/etc/profile.d/conda.sh &&
    conda activate ML &&
    ROOT=/home/diana/fishing/blind_nav_rl \
    RUN_DIR=$TRAIN_DIR \
    RESUME_MODEL=$RESUME_MODEL \
    START_STAGE=rppo_medium_lstm128x2_observable12_target8_discrete_recovery_v10_stage3 \
    END_STAGE=rppo_medium_lstm128x2_observable12_target8_discrete_recovery_v10_stage3 \
    STAGE3_EPISODES=1500 \
    N_ENVS=256 \
    THREADS=1 \
    DEVICE=cpu \
    EVAL_EPISODES=20 \
    VIDEO_EPISODES=0 \
    SEED_BASE=9800000 \
    EVAL_SEED_BASE=9900000 \
    bash launch_target8_v10_stage2_resume.sh
  "
  setsid bash -lc "$TRAIN_CMD" > "$TRAIN_DIR/阶段3优化训练.log" 2>&1 &
  echo $! > "$TRAIN_DIR/训练.pid"
  echo "train_started pid=$(cat "$TRAIN_DIR/训练.pid")"
fi

if test -f "$WATCH_DIR/watcher.pid" && ps -p "$(cat "$WATCH_DIR/watcher.pid")" >/dev/null 2>&1; then
  echo "watcher_already_running pid=$(cat "$WATCH_DIR/watcher.pid")"
else
  WATCH_CMD="
    cd /home/diana/fishing/blind_nav_rl &&
    source ~/miniconda3/etc/profile.d/conda.sh &&
    conda activate ML &&
    python watch_v10_checkpoints.py \
      --run-dir $TRAIN_DIR \
      --out-dir $WATCH_DIR \
      --curriculum \
      --checkpoint-step 1000
  "
  setsid bash -lc "$WATCH_CMD" > "$WATCH_DIR/watcher.log" 2>&1 &
  echo $! > "$WATCH_DIR/watcher.pid"
  echo "watcher_started pid=$(cat "$WATCH_DIR/watcher.pid")"
fi

echo "train_dir=$TRAIN_DIR"
echo "watch_dir=$WATCH_DIR"
echo "final_dir=$FINAL_DIR"
