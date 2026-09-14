#!/usr/bin/env bash
# Live progress for the background maze training run.
# Usage: bash scripts/watch_maze_train.sh [log_path]
LOG="${1:-Logs/maze_train.log}"
while true; do
  clear
  echo "=== $(date '+%F %T') ==="
  nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu \
    --format=csv,noheader 2>/dev/null | sed 's/^/GPU /'
  echo
  echo "=== validation ==="
  grep -E "独立仿真验证" "$LOG" | tail -n 3
  echo
  echo "=== last training iterations ==="
  grep -E "迭代次数" "$LOG" | tail -n 8
  echo
  echo "=== log tail ==="
  tail -n 6 "$LOG"
  sleep 5
done
