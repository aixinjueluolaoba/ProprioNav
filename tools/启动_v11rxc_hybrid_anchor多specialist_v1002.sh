#!/usr/bin/env bash
set -euo pipefail

LOCAL_ROOT=/home/diana/fishing/blind_nav_rl
REMOTE_ROOT=/home/diana/fishing/blind_nav_rl
RUN_NAME=${RUN_NAME:-v11rxc_hybrid_anchor_multi6300031_tiny}
RUN_DIR=${RUN_DIR:-$REMOTE_ROOT/runs/$RUN_NAME}
DATA_DIR=$RUN_DIR/teacher_datasets
RESUME_MODEL=${RESUME_MODEL:-$REMOTE_ROOT/runs/v11rxc_two_stage_6300031push112_tail2_N2stabilize32/rppo_medium_lstm128x2_observable12_target8_macro_library_v11rxc_stage2/rppo_medium_lstm128x2_observable12_target8_macro_library_v11rxc_stage2.zip}
NORMAL_MODEL=${NORMAL_MODEL:-$REMOTE_ROOT/runs/v11rxc_nlite125_fix6300031_targeted_far6_stuck10_tiny64/rppo_medium_lstm128x2_observable12_target8_macro_library_v11rxc_stage2/rppo_medium_lstm128x2_observable12_target8_macro_library_v11rxc_stage2.zip}
PRESSURE_MODEL=${PRESSURE_MODEL:-$NORMAL_MODEL}
SPECIALIST_MODEL_A=${SPECIALIST_MODEL_A:-$REMOTE_ROOT/runs/v11rxc_nlite125_fix6300031_targeted_far6_stuck10_search112/rppo_medium_lstm128x2_observable12_target8_macro_library_v11rxc_stage2/models/checkpoint_ep_00112.zip}
SPECIALIST_MODEL_B=${SPECIALIST_MODEL_B:-$REMOTE_ROOT/runs/v11rxc_nlite125_fix6300031_targeted_far6_stuck10_rl125/rppo_medium_lstm128x2_observable12_target8_macro_library_v11rxc_stage2/rppo_medium_lstm128x2_observable12_target8_macro_library_v11rxc_stage2.zip}
SPECIALIST_MODEL_C=${SPECIALIST_MODEL_C:-$RESUME_MODEL}
OVERRIDES_REMOTE=${OVERRIDES_REMOTE:-$REMOTE_ROOT/configs/v11/方案Nlite_v11rxc尾部轻保守探针.json}
NORMAL_DATASET=$DATA_DIR/普通锚点轨迹_v11rxc.npz
PRESSURE_DATASET=$DATA_DIR/压力锚点轨迹_v11rxc.npz
SPECIALIST_DATASET_A=$DATA_DIR/坏种子6300031修复轨迹_ep112_v11rxc.npz
SPECIALIST_DATASET_B=$DATA_DIR/坏种子6300031修复轨迹_rl125_v11rxc.npz
SPECIALIST_DATASET_C=$DATA_DIR/坏种子6300031修复轨迹_n2final_v11rxc.npz
NORMAL_SEED_START=${NORMAL_SEED_START:-1810000}
NORMAL_EPISODES=${NORMAL_EPISODES:-24}
PRESSURE_SEED_START=${PRESSURE_SEED_START:-10652000}
PRESSURE_EPISODES=${PRESSURE_EPISODES:-24}
PRESSURE_SCAN=${PRESSURE_SCAN:-5000}
TRAIN_EXTRA_ARGS=${TRAIN_EXTRA_ARGS:---tiny}
ANCHOR_DATASET_ROLES=${ANCHOR_DATASET_ROLES:-base,base,specialist,specialist,specialist}
ANCHOR_DATASET_WEIGHTS=${ANCHOR_DATASET_WEIGHTS:-1.1,1.3,1.2,1.4,1.8}
PHASE2_BAD_SEED_RATIO=${PHASE2_BAD_SEED_RATIO:-0.125}
RECOVERY_WEIGHT=${RECOVERY_WEIGHT:-1.35}
STUCK_WEIGHT=${STUCK_WEIGHT:-1.2}
COLLISION_WEIGHT=${COLLISION_WEIGHT:-1.1}
PPO_LEARNING_RATE=${PPO_LEARNING_RATE:-2e-5}
ANCHOR_LEARNING_RATE=${ANCHOR_LEARNING_RATE:-8e-5}
SEED_POOL=${SEED_POOL:-6300031,6300058,6300069,6300006}
ONLINE_BAD_SEED_POOL=${ONLINE_BAD_SEED_POOL:-6300058,6300006}
SPECIALIST_SEED_POOL=${SPECIALIST_SEED_POOL:-6300031}

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
copy_file train_target8_v11re_bad_seed_mergeback_curriculum.py
copy_file train_target8_v11rxc_bad_seed_curriculum.py
copy_file train_target8_v11rxc_hybrid_anchor_finetune.py
copy_file "configs/v11/方案Nlite_v11rxc尾部轻保守探针.json"

ssh v1002 "mkdir -p '$RUN_DIR' '$DATA_DIR'"

for required in \
  "$RESUME_MODEL" \
  "$NORMAL_MODEL" \
  "$PRESSURE_MODEL" \
  "$SPECIALIST_MODEL_A" \
  "$SPECIALIST_MODEL_B" \
  "$SPECIALIST_MODEL_C"; do
  if ! ssh v1002 "test -f '$required'"; then
    echo "missing_required path=$required" >&2
    exit 1
  fi
done

echo "resume_model=$RESUME_MODEL"
echo "normal_model=$NORMAL_MODEL"
echo "pressure_model=$PRESSURE_MODEL"
echo "specialist_model_a=$SPECIALIST_MODEL_A"
echo "specialist_model_b=$SPECIALIST_MODEL_B"
echo "specialist_model_c=$SPECIALIST_MODEL_C"
echo "seed_pool=$SEED_POOL"
echo "online_bad_seed_pool=$ONLINE_BAD_SEED_POOL"
echo "specialist_seed_pool=$SPECIALIST_SEED_POOL"

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
    --state-mode observable12_target8_macro_library_v11rxc_stage2 \
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
    --state-mode observable12_target8_macro_library_v11rxc_stage2 \
    --seed-start $PRESSURE_SEED_START \
    --episodes $PRESSURE_EPISODES \
    --pressure-scan $PRESSURE_SCAN \
    --device cpu \
    --env-overrides-file '$OVERRIDES_REMOTE'
"
maybe_export_dataset "$PRESSURE_DATASET" "$PRESSURE_EXPORT_CMD"

SPECIALIST_EXPORT_CMD_A="
  cd $REMOTE_ROOT &&
  source ~/miniconda3/etc/profile.d/conda.sh &&
  conda activate ML &&
  python export_v11_teacher_dataset.py \
    --model '$SPECIALIST_MODEL_A' \
    --out '$SPECIALIST_DATASET_A' \
    --scenario diagonal \
    --seed-list '$SPECIALIST_SEED_POOL' \
    --state-mode observable12_target8_macro_library_v11rxc_stage2 \
    --device cpu \
    --env-overrides-file '$OVERRIDES_REMOTE'
"
maybe_export_dataset "$SPECIALIST_DATASET_A" "$SPECIALIST_EXPORT_CMD_A"

SPECIALIST_EXPORT_CMD_B="
  cd $REMOTE_ROOT &&
  source ~/miniconda3/etc/profile.d/conda.sh &&
  conda activate ML &&
  python export_v11_teacher_dataset.py \
    --model '$SPECIALIST_MODEL_B' \
    --out '$SPECIALIST_DATASET_B' \
    --scenario diagonal \
    --seed-list '$SPECIALIST_SEED_POOL' \
    --state-mode observable12_target8_macro_library_v11rxc_stage2 \
    --device cpu \
    --env-overrides-file '$OVERRIDES_REMOTE'
"
maybe_export_dataset "$SPECIALIST_DATASET_B" "$SPECIALIST_EXPORT_CMD_B"

SPECIALIST_EXPORT_CMD_C="
  cd $REMOTE_ROOT &&
  source ~/miniconda3/etc/profile.d/conda.sh &&
  conda activate ML &&
  python export_v11_teacher_dataset.py \
    --model '$SPECIALIST_MODEL_C' \
    --out '$SPECIALIST_DATASET_C' \
    --scenario diagonal \
    --seed-list '$SPECIALIST_SEED_POOL' \
    --state-mode observable12_target8_macro_library_v11rxc_stage2 \
    --device cpu \
    --env-overrides-file '$OVERRIDES_REMOTE'
"
maybe_export_dataset "$SPECIALIST_DATASET_C" "$SPECIALIST_EXPORT_CMD_C"

ssh v1002 "cd '$REMOTE_ROOT' && source ~/miniconda3/etc/profile.d/conda.sh && conda activate ML && python train_target8_v11rxc_hybrid_anchor_finetune.py \
  $TRAIN_EXTRA_ARGS \
  --output-dir '$RUN_DIR' \
  --resume-model '$RESUME_MODEL' \
  --anchor-datasets '$NORMAL_DATASET' '$PRESSURE_DATASET' '$SPECIALIST_DATASET_A' '$SPECIALIST_DATASET_B' '$SPECIALIST_DATASET_C' \
  --anchor-dataset-roles '$ANCHOR_DATASET_ROLES' \
  --anchor-dataset-weights '$ANCHOR_DATASET_WEIGHTS' \
  --anchor-success-only \
  --recovery-weight $RECOVERY_WEIGHT \
  --stuck-weight $STUCK_WEIGHT \
  --collision-weight $COLLISION_WEIGHT \
  --phase2-bad-seed-ratio $PHASE2_BAD_SEED_RATIO \
  --ppo-learning-rate $PPO_LEARNING_RATE \
  --anchor-learning-rate $ANCHOR_LEARNING_RATE \
  --seed-pool '$SEED_POOL' \
  --online-bad-seed-pool '$ONLINE_BAD_SEED_POOL' \
  --video-episodes 0 \
  --device cpu \
  --torch-threads 1 \
  --env-overrides-file '$OVERRIDES_REMOTE'"

MODEL_BASE="$RUN_DIR/rppo_medium_lstm128x2_observable12_target8_macro_library_v11rxc_stage2/rppo_medium_lstm128x2_observable12_target8_macro_library_v11rxc_stage2"
MODEL_ZIP="${MODEL_BASE}.zip"

ssh v1002 "cd '$REMOTE_ROOT' && source ~/miniconda3/etc/profile.d/conda.sh && conda activate ML && python eval_target8_bad_seeds.py \
  --model '$MODEL_BASE' \
  --out-dir '${RUN_DIR}_badseed_gate' \
  --state-mode observable12_target8_macro_library_v11rxc_stage2 \
  --env-overrides-file '$OVERRIDES_REMOTE' \
  --device cpu \
  --seed-list 6300031,6300058,6300069,6300006"

ssh v1002 "cd '$REMOTE_ROOT' && source ~/miniconda3/etc/profile.d/conda.sh && conda activate ML && python eval_target8_pressure.py \
  --model '$MODEL_BASE' \
  --out-dir '${RUN_DIR}_pressure20' \
  --action-mode v11 \
  --state-mode observable12_target8_macro_library_v11rxc_stage2 \
  --episodes 20 \
  --video-episodes 0 \
  --env-overrides-file '$OVERRIDES_REMOTE' \
  --device cpu"

ssh v1002 "cd '$REMOTE_ROOT' && source ~/miniconda3/etc/profile.d/conda.sh && conda activate ML && python eval_target8_sweep.py \
  --train-output-dir '$RUN_DIR' \
  --model-path '$MODEL_ZIP' \
  --out-dir '${RUN_DIR}_normal20' \
  --experiments rppo_medium_lstm128x2_observable12_target8_macro_library_v11rxc_stage2 \
  --episodes 20 \
  --video-episodes 0 \
  --env-overrides-file '$OVERRIDES_REMOTE' \
  --device cpu"

echo "run_dir=$RUN_DIR"
