#!/usr/bin/env bash
set -euo pipefail

cd /home/diana/fishing/blind_nav_rl

TRAIN_DIR=/home/diana/fishing/blind_nav_rl/runs/目标8显式脱困v10四阶段_v1002_110k
WATCH_DIR=/home/diana/fishing/blind_nav_rl/runs/目标8显式脱困v10四阶段_checkpoint评估_v1002
FINAL_DIR=/home/diana/fishing/blind_nav_rl/runs/目标8显式脱困v10四阶段_最终归档_v1002
RESUME_MODEL=/home/diana/fishing/blind_nav_rl/runs/目标8显式脱困v10课程_v1002_100k/rppo_medium_lstm128x2_observable12_target8_discrete_recovery_v10_stage1/models/checkpoint_ep_07000.zip

mkdir -p "$TRAIN_DIR" "$WATCH_DIR" "$FINAL_DIR"

if test -f "$TRAIN_DIR/训练.pid" && ps -p "$(cat "$TRAIN_DIR/训练.pid")" >/dev/null 2>&1; then
  echo "train_already_running pid=$(cat "$TRAIN_DIR/训练.pid")"
else
  TRAIN_CMD="
    cd /home/diana/fishing/blind_nav_rl &&
    source ~/miniconda3/etc/profile.d/conda.sh &&
    conda activate ML &&
    ROOT=/home/diana/fishing/blind_nav_rl
    RUN_DIR=/home/diana/fishing/blind_nav_rl/runs/目标8显式脱困v10四阶段_v1002_110k
    RESUME_MODEL=$RESUME_MODEL
    STAGE2A_EPISODES=10000
    STAGE2_EPISODES=30000
    STAGE3_EPISODES=50000
    N_ENVS=256
    THREADS=1
    DEVICE=cpu
    EVAL_EPISODES=20
    VIDEO_EPISODES=0
    SEED_BASE=9600000
    EVAL_SEED_BASE=9700000
    bash launch_target8_v10_stage2_resume.sh
  "
  setsid bash -lc "$TRAIN_CMD" > "$TRAIN_DIR/训练.log" 2>&1 &
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
    python watch_v10_checkpoints.py
      --run-dir /home/diana/fishing/blind_nav_rl/runs/目标8显式脱困v10四阶段_v1002_110k
      --out-dir /home/diana/fishing/blind_nav_rl/runs/目标8显式脱困v10四阶段_checkpoint评估_v1002
      --curriculum
      --checkpoint-step 1000
  "
  setsid bash -lc "$WATCH_CMD" > "$WATCH_DIR/watcher.log" 2>&1 &
  echo $! > "$WATCH_DIR/watcher.pid"
  echo "watcher_started pid=$(cat "$WATCH_DIR/watcher.pid")"
fi

if test -f "$FINAL_DIR/finalizer.pid" && ps -p "$(cat "$FINAL_DIR/finalizer.pid")" >/dev/null 2>&1; then
  echo "finalizer_already_running pid=$(cat "$FINAL_DIR/finalizer.pid")"
else
  FINAL_CMD="
    cd /home/diana/fishing/blind_nav_rl &&
    source ~/miniconda3/etc/profile.d/conda.sh &&
    conda activate ML &&
    ROOT=/home/diana/fishing/blind_nav_rl
    RUN_DIR=/home/diana/fishing/blind_nav_rl/runs/目标8显式脱困v10四阶段_v1002_110k
    CHECKPOINT_EVAL_DIR=/home/diana/fishing/blind_nav_rl/runs/目标8显式脱困v10四阶段_checkpoint评估_v1002
    OUT_DIR=/home/diana/fishing/blind_nav_rl/runs/目标8显式脱困v10四阶段_最终归档_v1002
    MIN_CHECKPOINT=50000
    EPISODES=20
    VIDEO_EPISODES=20
    DEVICE=cpu
    bash launch_v10_finalizer.sh
  "
  setsid bash -lc "$FINAL_CMD" > "$FINAL_DIR/finalizer.log" 2>&1 &
  echo $! > "$FINAL_DIR/finalizer.pid"
  echo "finalizer_started pid=$(cat "$FINAL_DIR/finalizer.pid")"
fi
