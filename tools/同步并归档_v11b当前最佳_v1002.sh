#!/usr/bin/env bash
set -euo pipefail

LOCAL_ROOT=/home/diana/fishing/blind_nav_rl
REMOTE_ROOT=/home/diana/fishing/blind_nav_rl

copy_file() {
  local rel="$1"
  local remote_dir
  remote_dir="$(dirname "$rel")"
  ssh v1002 "mkdir -p '$REMOTE_ROOT/$remote_dir'"
  scp "$LOCAL_ROOT/$rel" "v1002:$REMOTE_ROOT/$rel"
}

copy_file benchmark_state_dims_10k.py
copy_file blind_nav_rl/env.py
copy_file eval_target8_sweep.py
copy_file eval_target8_pressure.py
copy_file eval_target8_diagonal_dense_video.py
copy_file tools/归档_v11b当前最佳_v1002.sh

ssh v1002 "bash '$REMOTE_ROOT/tools/归档_v11b当前最佳_v1002.sh'"
