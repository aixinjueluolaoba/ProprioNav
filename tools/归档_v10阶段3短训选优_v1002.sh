#!/usr/bin/env bash
set -euo pipefail

cd /home/diana/fishing/blind_nav_rl
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ML

ROOT=/home/diana/fishing/blind_nav_rl \
RUN_DIR=/home/diana/fishing/blind_nav_rl/runs/目标8显式脱困v10阶段3短训选优_v1002 \
CHECKPOINT_EVAL_DIR=/home/diana/fishing/blind_nav_rl/runs/目标8显式脱困v10阶段3短训选优_checkpoint评估_v1002 \
OUT_DIR=/home/diana/fishing/blind_nav_rl/runs/目标8显式脱困v10阶段3短训选优_最终归档_v1002 \
MIN_CHECKPOINT=500 \
EPISODES=20 \
VIDEO_EPISODES=20 \
DEVICE=cpu \
SCORE_PROFILE=pressure_priority \
bash launch_v10_finalizer.sh
