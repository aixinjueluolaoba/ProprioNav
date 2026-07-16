#!/usr/bin/env bash
set -euo pipefail

LOCAL_ROOT=/home/diana/fishing/blind_nav_rl
REMOTE_ROOT=/home/diana/fishing/blind_nav_rl
RUN_NAME=${RUN_NAME:-v11re_hybrid_anchor_micro}
RUN_DIR=${RUN_DIR:-$REMOTE_ROOT/runs/$RUN_NAME}
DATA_DIR=$RUN_DIR/teacher_datasets
RESUME_MODEL=${RESUME_MODEL:-$REMOTE_ROOT/runs/v11re_reward_exit_probe/rppo_medium_lstm128x2_observable12_target8_macro_library_v11re_stage2/models/checkpoint_ep_01000.zip}
NORMAL_MODEL=${NORMAL_MODEL:-$RESUME_MODEL}
PRESSURE_MODEL=${PRESSURE_MODEL:-$RESUME_MODEL}
DIAGONAL_MODEL=${DIAGONAL_MODEL:-$REMOTE_ROOT/runs/v11re_bad_seed_micro/rppo_medium_lstm128x2_observable12_target8_macro_library_v11re_stage2/models/checkpoint_ep_00125.zip}
OVERRIDES_REMOTE=${OVERRIDES_REMOTE:-$REMOTE_ROOT/configs/v11/方案H_脱困退出与收角强化.json}
NORMAL_DATASET=$DATA_DIR/普通锚点轨迹.npz
PRESSURE_DATASET=$DATA_DIR/压力锚点轨迹.npz
DIAGONAL_DATASET=$DATA_DIR/坏种子脱困锚点轨迹.npz
NORMAL_SEED_START=${NORMAL_SEED_START:-1810000}
NORMAL_EPISODES=${NORMAL_EPISODES:-24}
PRESSURE_SEED_START=${PRESSURE_SEED_START:-10651000}
PRESSURE_EPISODES=${PRESSURE_EPISODES:-24}
PRESSURE_SCAN=${PRESSURE_SCAN:-4000}
TRAIN_EXTRA_ARGS=${TRAIN_EXTRA_ARGS:---micro}
ANCHOR_DATASET_ROLES=${ANCHOR_DATASET_ROLES:-base,base,specialist}
ANCHOR_DATASET_WEIGHTS=${ANCHOR_DATASET_WEIGHTS:-1.0,1.2,1.8}
PHASE2_BAD_SEED_RATIO=${PHASE2_BAD_SEED_RATIO:-0.25}
RECOVERY_WEIGHT=${RECOVERY_WEIGHT:-1.4}
STUCK_WEIGHT=${STUCK_WEIGHT:-1.3}
COLLISION_WEIGHT=${COLLISION_WEIGHT:-1.2}
PPO_LEARNING_RATE=${PPO_LEARNING_RATE:-3e-5}
ANCHOR_LEARNING_RATE=${ANCHOR_LEARNING_RATE:-1e-4}

copy_file() {
  local rel="$1"
  ssh v1002 "mkdir -p '$REMOTE_ROOT/$(dirname "$rel")'"
  scp "$LOCAL_ROOT/$rel" "v1002:$REMOTE_ROOT/$rel"
}

copy_file benchmark_state_dims_10k.py
copy_file blind_nav_rl/env.py
copy_file eval_target8_bad_seeds.py
copy_file eval_target8_pressure.py
copy_file eval_target8_sweep.py
copy_file export_v11_teacher_dataset.py
copy_file train_target8_v11re_reward_exit_curriculum.py
copy_file train_target8_v11re_bad_seed_curriculum.py
copy_file train_target8_v11re_bad_seed_mergeback_curriculum.py
copy_file train_target8_v11re_hybrid_anchor_finetune.py
copy_file watch_v11_checkpoints.py
copy_file watch_v11re_checkpoints.py
copy_file "configs/v11/方案H_脱困退出与收角强化.json"
copy_file tools/启动_v11re_hybrid_anchor评估_v1002.sh

ssh v1002 "mkdir -p '$RUN_DIR' '$DATA_DIR'"

if ! ssh v1002 "test -f '$RESUME_MODEL'"; then
  echo "missing_resume_model path=$RESUME_MODEL" >&2
  exit 1
fi
if ! ssh v1002 "test -f '$NORMAL_MODEL'"; then
  echo "missing_normal_model path=$NORMAL_MODEL" >&2
  exit 1
fi
if ! ssh v1002 "test -f '$PRESSURE_MODEL'"; then
  echo "missing_pressure_model path=$PRESSURE_MODEL" >&2
  exit 1
fi
if ! ssh v1002 "test -f '$DIAGONAL_MODEL'"; then
  echo "fallback_specialist_model path=$RESUME_MODEL"
  DIAGONAL_MODEL=$RESUME_MODEL
fi

echo "resume_model=$RESUME_MODEL"
echo "normal_model=$NORMAL_MODEL"
echo "pressure_model=$PRESSURE_MODEL"
echo "diagonal_model=$DIAGONAL_MODEL"

maybe_export_dataset() {
  local dataset_path="$1"
  local export_cmd="$2"
  if ssh v1002 "test -f '$dataset_path'"; then
    echo "dataset_exists path=$dataset_path"
  else
    ssh v1002 "bash -lc \"$export_cmd\""
    echo "dataset_exported path=$dataset_path"
  fi
}

NORMAL_EXPORT_CMD="
  cd $REMOTE_ROOT &&
  source ~/miniconda3/etc/profile.d/conda.sh &&
  conda activate ML &&
  python export_v11_teacher_dataset.py \
    --model '$NORMAL_MODEL' \
    --out '$NORMAL_DATASET' \
    --scenario normal \
    --state-mode observable12_target8_macro_library_v11re_stage2 \
    --seed-start $NORMAL_SEED_START \
    --episodes $NORMAL_EPISODES \
    --device cpu \
    --env-overrides-file '$OVERRIDES_REMOTE'
"
maybe_export_dataset "$NORMAL_DATASET" "$NORMAL_EXPORT_CMD"

PRESSURE_EXPORT_CMD="
  cd $REMOTE_ROOT &&
  source ~/miniconda3/etc/profile.d/conda.sh &&
  conda activate ML &&
  python export_v11_teacher_dataset.py \
    --model '$PRESSURE_MODEL' \
    --out '$PRESSURE_DATASET' \
    --scenario pressure \
    --state-mode observable12_target8_macro_library_v11re_stage2 \
    --seed-start $PRESSURE_SEED_START \
    --episodes $PRESSURE_EPISODES \
    --pressure-scan $PRESSURE_SCAN \
    --device cpu \
    --env-overrides-file '$OVERRIDES_REMOTE'
"
maybe_export_dataset "$PRESSURE_DATASET" "$PRESSURE_EXPORT_CMD"

DIAGONAL_EXPORT_CMD="
  cd $REMOTE_ROOT &&
  source ~/miniconda3/etc/profile.d/conda.sh &&
  conda activate ML &&
  python export_v11_teacher_dataset.py \
    --model '$DIAGONAL_MODEL' \
    --out '$DIAGONAL_DATASET' \
    --scenario diagonal \
    --state-mode observable12_target8_macro_library_v11re_stage2 \
    --device cpu \
    --env-overrides-file '$OVERRIDES_REMOTE'
"
maybe_export_dataset "$DIAGONAL_DATASET" "$DIAGONAL_EXPORT_CMD"

if ssh v1002 "test -f '$RUN_DIR/训练.pid' && ps -p \"\$(cat '$RUN_DIR/训练.pid')\" >/dev/null 2>&1"; then
  echo "train_already_running pid=$(ssh v1002 "cat '$RUN_DIR/训练.pid'")"
else
  TRAIN_CMD="
    cd $REMOTE_ROOT &&
    source ~/miniconda3/etc/profile.d/conda.sh &&
    conda activate ML &&
    python train_target8_v11re_hybrid_anchor_finetune.py \
      $TRAIN_EXTRA_ARGS \
      --output-dir '$RUN_DIR' \
      --resume-model '$RESUME_MODEL' \
      --anchor-datasets '$NORMAL_DATASET' '$PRESSURE_DATASET' '$DIAGONAL_DATASET' \
      --anchor-dataset-roles '$ANCHOR_DATASET_ROLES' \
      --anchor-dataset-weights '$ANCHOR_DATASET_WEIGHTS' \
      --anchor-success-only \
      --recovery-weight $RECOVERY_WEIGHT \
      --stuck-weight $STUCK_WEIGHT \
      --collision-weight $COLLISION_WEIGHT \
      --phase2-bad-seed-ratio $PHASE2_BAD_SEED_RATIO \
      --ppo-learning-rate $PPO_LEARNING_RATE \
      --anchor-learning-rate $ANCHOR_LEARNING_RATE \
      --video-episodes 0 \
      --device cpu \
      --torch-threads 1 \
      --env-overrides-file '$OVERRIDES_REMOTE'
  "
  ssh v1002 "setsid bash -lc \"$TRAIN_CMD\" > '$RUN_DIR/训练.log' 2>&1 & echo \$! > '$RUN_DIR/训练.pid'"
  echo "train_started pid=$(ssh v1002 "cat '$RUN_DIR/训练.pid'")"
fi

echo "run_dir=$RUN_DIR"
echo "watch_cmd=bash $LOCAL_ROOT/tools/启动_v11re_hybrid_anchor评估_v1002.sh"
