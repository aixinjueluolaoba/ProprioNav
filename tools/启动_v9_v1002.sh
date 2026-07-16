#!/usr/bin/env bash
set -euo pipefail

cd /home/diana/fishing/blind_nav_rl
mkdir -p \
  runs/目标8离散触发脱困v9课程_v1002_100k \
  runs/目标8离散触发脱困v9_checkpoint评估_v1002 \
  runs/目标8离散触发脱困v9_最终归档_v1002

if test -f runs/目标8离散触发脱困v9课程_v1002_100k/训练.pid && ps -p "$(cat runs/目标8离散触发脱困v9课程_v1002_100k/训练.pid)" >/dev/null 2>&1; then
  echo "train_already_running pid=$(cat runs/目标8离散触发脱困v9课程_v1002_100k/训练.pid)"
else
  setsid bash -lc 'cd /home/diana/fishing/blind_nav_rl && source ~/miniconda3/etc/profile.d/conda.sh && conda activate ML && ROOT=/home/diana/fishing/blind_nav_rl RUN_DIR=/home/diana/fishing/blind_nav_rl/runs/目标8离散触发脱困v9课程_v1002_100k STAGE1_EPISODES=30000 STAGE2_EPISODES=70000 N_ENVS=256 THREADS=1 DEVICE=cpu EVAL_EPISODES=20 VIDEO_EPISODES=0 SEED_BASE=8600000 EVAL_SEED_BASE=8700000 bash launch_target8_v9_curriculum.sh' > runs/目标8离散触发脱困v9课程_v1002_100k/训练.log 2>&1 &
  echo $! > runs/目标8离散触发脱困v9课程_v1002_100k/训练.pid
  echo "train_started pid=$(cat runs/目标8离散触发脱困v9课程_v1002_100k/训练.pid)"
fi

if test -f runs/目标8离散触发脱困v9_checkpoint评估_v1002/watcher.pid && ps -p "$(cat runs/目标8离散触发脱困v9_checkpoint评估_v1002/watcher.pid)" >/dev/null 2>&1; then
  echo "watcher_already_running pid=$(cat runs/目标8离散触发脱困v9_checkpoint评估_v1002/watcher.pid)"
else
  setsid bash -lc 'cd /home/diana/fishing/blind_nav_rl && source ~/miniconda3/etc/profile.d/conda.sh && conda activate ML && python watch_v9_checkpoints.py --run-dir /home/diana/fishing/blind_nav_rl/runs/目标8离散触发脱困v9课程_v1002_100k --out-dir /home/diana/fishing/blind_nav_rl/runs/目标8离散触发脱困v9_checkpoint评估_v1002 --max-checkpoint 70000 --checkpoint-step 10000' > runs/目标8离散触发脱困v9_checkpoint评估_v1002/watcher.stdout.log 2>&1 &
  echo $! > runs/目标8离散触发脱困v9_checkpoint评估_v1002/watcher.pid
  echo "watcher_started pid=$(cat runs/目标8离散触发脱困v9_checkpoint评估_v1002/watcher.pid)"
fi

if test -f runs/目标8离散触发脱困v9_最终归档_v1002/finalizer.pid && ps -p "$(cat runs/目标8离散触发脱困v9_最终归档_v1002/finalizer.pid)" >/dev/null 2>&1; then
  echo "finalizer_already_running pid=$(cat runs/目标8离散触发脱困v9_最终归档_v1002/finalizer.pid)"
else
  setsid bash -lc 'cd /home/diana/fishing/blind_nav_rl && source ~/miniconda3/etc/profile.d/conda.sh && conda activate ML && ROOT=/home/diana/fishing/blind_nav_rl RUN_DIR=/home/diana/fishing/blind_nav_rl/runs/目标8离散触发脱困v9课程_v1002_100k CHECKPOINT_EVAL_DIR=/home/diana/fishing/blind_nav_rl/runs/目标8离散触发脱困v9_checkpoint评估_v1002 OUT_DIR=/home/diana/fishing/blind_nav_rl/runs/目标8离散触发脱困v9_最终归档_v1002 MIN_CHECKPOINT=70000 bash launch_v9_finalizer.sh' > runs/目标8离散触发脱困v9_最终归档_v1002/finalizer.stdout.log 2>&1 &
  echo $! > runs/目标8离散触发脱困v9_最终归档_v1002/finalizer.pid
  echo "finalizer_started pid=$(cat runs/目标8离散触发脱困v9_最终归档_v1002/finalizer.pid)"
fi
