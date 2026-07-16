#!/usr/bin/env bash
set -euo pipefail

LOCAL_ROOT=/home/diana/fishing/blind_nav_rl
REMOTE_ROOT=/home/diana/fishing/blind_nav_rl
RUN_NAME=${RUN_NAME:-v11re_hybrid_anchor_micro}
RUN_DIR=${RUN_DIR:-$REMOTE_ROOT/runs/$RUN_NAME}
OUT_DIR=${OUT_DIR:-$REMOTE_ROOT/runs/${RUN_NAME}_watch}
WATCH_EXTRA_ARGS=${WATCH_EXTRA_ARGS:---micro --max-checkpoint 128 --checkpoint-step 32}

copy_file() {
  local rel="$1"
  ssh v1002 "mkdir -p '$REMOTE_ROOT/$(dirname "$rel")'"
  scp "$LOCAL_ROOT/$rel" "v1002:$REMOTE_ROOT/$rel"
}

copy_file benchmark_state_dims_10k.py
copy_file blind_nav_rl/env.py
copy_file eval_target8_sweep.py
copy_file eval_target8_pressure.py
copy_file watch_v11_checkpoints.py
copy_file watch_v11re_checkpoints.py

ssh v1002 "mkdir -p '$OUT_DIR'"

if ssh v1002 "test -f '$OUT_DIR/评估.pid' && ps -p \"\$(cat '$OUT_DIR/评估.pid')\" >/dev/null 2>&1"; then
  echo "watch_already_running pid=$(ssh v1002 "cat '$OUT_DIR/评估.pid'")"
else
  WATCH_CMD="
    cd $REMOTE_ROOT &&
    source ~/miniconda3/etc/profile.d/conda.sh &&
    conda activate ML &&
    python watch_v11re_checkpoints.py \
      $WATCH_EXTRA_ARGS \
      --run-dir '$RUN_DIR' \
      --out-dir '$OUT_DIR' \
      --interval 60 \
      --experiment rppo_medium_lstm128x2_observable12_target8_macro_library_v11re_stage2 \
      --state-mode observable12_target8_macro_library_v11re_stage2
  "
  ssh v1002 "setsid bash -lc \"$WATCH_CMD\" > '$OUT_DIR/评估.log' 2>&1 & echo \$! > '$OUT_DIR/评估.pid'"
  echo "watch_started pid=$(ssh v1002 "cat '$OUT_DIR/评估.pid'")"
fi

echo "out_dir=$OUT_DIR"
