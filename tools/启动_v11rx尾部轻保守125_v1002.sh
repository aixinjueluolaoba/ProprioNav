#!/usr/bin/env bash
set -euo pipefail

LOCAL_ROOT=/home/diana/fishing/blind_nav_rl
REMOTE_ROOT=/home/diana/fishing/blind_nav_rl
RUN_NAME=${RUN_NAME:-v11rx_tail2_conservative_lite_125}
RUN_DIR=${RUN_DIR:-$REMOTE_ROOT/runs/$RUN_NAME}
RESUME_MODEL=${RESUME_MODEL:-$REMOTE_ROOT/runs/v11rx_conflict4_micro/rppo_medium_lstm128x2_observable12_target8_macro_library_v11rx_stage2/models/checkpoint_ep_00250.zip}
OVERRIDES_REMOTE=${OVERRIDES_REMOTE:-$REMOTE_ROOT/configs/v11/方案Nlite_v11rx尾部轻保守探针.json}
SEED_POOL=${SEED_POOL:-6300058,6300006}

copy_file() {
  local rel="$1"
  ssh v1002 "mkdir -p '$REMOTE_ROOT/$(dirname "$rel")'"
  scp "$LOCAL_ROOT/$rel" "v1002:$REMOTE_ROOT/$rel"
}

copy_file benchmark_state_dims_10k.py
copy_file blind_nav_rl/env.py
copy_file train_target8_v11re_reward_exit_curriculum.py
copy_file train_target8_v11rx_bad_seed_curriculum.py
copy_file "configs/v11/方案Nlite_v11rx尾部轻保守探针.json"

ssh v1002 "mkdir -p '$RUN_DIR'"

if ! ssh v1002 "test -f '$RESUME_MODEL'"; then
  echo "missing_resume_model path=$RESUME_MODEL" >&2
  exit 1
fi

echo "resume_model=$RESUME_MODEL"
echo "seed_pool=$SEED_POOL"
echo "overrides=$OVERRIDES_REMOTE"

if ssh v1002 "test -f '$RUN_DIR/训练.pid' && ps -p \"\$(cat '$RUN_DIR/训练.pid')\" >/dev/null 2>&1"; then
  echo "train_already_running pid=$(ssh v1002 "cat '$RUN_DIR/训练.pid'")"
else
  TRAIN_CMD="
    cd $REMOTE_ROOT &&
    source ~/miniconda3/etc/profile.d/conda.sh &&
    conda activate ML &&
    python train_target8_v11rx_bad_seed_curriculum.py \
      --output-dir '$RUN_DIR' \
      --resume-model '$RESUME_MODEL' \
      --seed-pool '$SEED_POOL' \
      --stage-episodes 125 \
      --eval-episodes 4 \
      --checkpoint-interval 125 \
      --progress-interval 125 \
      --video-episodes 0 \
      --device cpu \
      --torch-threads 1 \
      --env-overrides-file '$OVERRIDES_REMOTE'
  "
  ssh v1002 "setsid bash -lc \"$TRAIN_CMD\" > '$RUN_DIR/训练.log' 2>&1 & echo \$! > '$RUN_DIR/训练.pid'"
  echo "train_started pid=$(ssh v1002 "cat '$RUN_DIR/训练.pid'")"
fi

echo "run_dir=$RUN_DIR"
