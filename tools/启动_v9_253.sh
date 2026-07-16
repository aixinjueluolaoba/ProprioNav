#!/usr/bin/env bash
set -euo pipefail

cd /home/weiaokang/fishing/blind_nav_rl
mkdir -p \
  runs/目标8离散触发脱困v9课程_253_100k \
  runs/目标8离散触发脱困v9_checkpoint评估_253 \
  runs/目标8离散触发脱困v9_最终归档_253

if test -f runs/目标8离散触发脱困v9课程_253_100k/训练.pid && ps -p "$(cat runs/目标8离散触发脱困v9课程_253_100k/训练.pid)" >/dev/null 2>&1; then
  echo "train_already_running pid=$(cat runs/目标8离散触发脱困v9课程_253_100k/训练.pid)"
else
  setsid bash -lc 'cd /home/weiaokang/fishing/blind_nav_rl && source /home/weiaokang/ML/bin/activate && ROOT=/home/weiaokang/fishing/blind_nav_rl RUN_DIR=/home/weiaokang/fishing/blind_nav_rl/runs/目标8离散触发脱困v9课程_253_100k STAGE1_EPISODES=30000 STAGE2_EPISODES=70000 N_ENVS=256 THREADS=1 DEVICE=cpu EVAL_EPISODES=20 VIDEO_EPISODES=0 SEED_BASE=8400000 EVAL_SEED_BASE=8500000 bash launch_target8_v9_curriculum.sh' > runs/目标8离散触发脱困v9课程_253_100k/训练.log 2>&1 &
  echo $! > runs/目标8离散触发脱困v9课程_253_100k/训练.pid
  echo "train_started pid=$(cat runs/目标8离散触发脱困v9课程_253_100k/训练.pid)"
fi

if test -f runs/目标8离散触发脱困v9_checkpoint评估_253/watcher.pid && ps -p "$(cat runs/目标8离散触发脱困v9_checkpoint评估_253/watcher.pid)" >/dev/null 2>&1; then
  echo "watcher_already_running pid=$(cat runs/目标8离散触发脱困v9_checkpoint评估_253/watcher.pid)"
else
  setsid bash -lc 'cd /home/weiaokang/fishing/blind_nav_rl && source /home/weiaokang/ML/bin/activate && python watch_v9_checkpoints.py --run-dir /home/weiaokang/fishing/blind_nav_rl/runs/目标8离散触发脱困v9课程_253_100k --out-dir /home/weiaokang/fishing/blind_nav_rl/runs/目标8离散触发脱困v9_checkpoint评估_253 --max-checkpoint 70000 --checkpoint-step 10000' > runs/目标8离散触发脱困v9_checkpoint评估_253/watcher.stdout.log 2>&1 &
  echo $! > runs/目标8离散触发脱困v9_checkpoint评估_253/watcher.pid
  echo "watcher_started pid=$(cat runs/目标8离散触发脱困v9_checkpoint评估_253/watcher.pid)"
fi

if test -f runs/目标8离散触发脱困v9_最终归档_253/finalizer.pid && ps -p "$(cat runs/目标8离散触发脱困v9_最终归档_253/finalizer.pid)" >/dev/null 2>&1; then
  echo "finalizer_already_running pid=$(cat runs/目标8离散触发脱困v9_最终归档_253/finalizer.pid)"
else
  setsid bash -lc 'cd /home/weiaokang/fishing/blind_nav_rl && source /home/weiaokang/ML/bin/activate && ROOT=/home/weiaokang/fishing/blind_nav_rl RUN_DIR=/home/weiaokang/fishing/blind_nav_rl/runs/目标8离散触发脱困v9课程_253_100k CHECKPOINT_EVAL_DIR=/home/weiaokang/fishing/blind_nav_rl/runs/目标8离散触发脱困v9_checkpoint评估_253 OUT_DIR=/home/weiaokang/fishing/blind_nav_rl/runs/目标8离散触发脱困v9_最终归档_253 MIN_CHECKPOINT=70000 bash launch_v9_finalizer.sh' > runs/目标8离散触发脱困v9_最终归档_253/finalizer.stdout.log 2>&1 &
  echo $! > runs/目标8离散触发脱困v9_最终归档_253/finalizer.pid
  echo "finalizer_started pid=$(cat runs/目标8离散触发脱困v9_最终归档_253/finalizer.pid)"
fi
