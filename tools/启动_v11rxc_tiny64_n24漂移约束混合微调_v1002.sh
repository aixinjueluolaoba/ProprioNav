#!/usr/bin/env bash
set -euo pipefail

LOCAL_ROOT=/home/diana/fishing/blind_nav_rl
REMOTE_ROOT=/home/diana/fishing/blind_nav_rl
RUN_NAME=${RUN_NAME:-v11rxc_tiny64_n24_driftctl_policyhead}
RUN_DIR=${RUN_DIR:-$REMOTE_ROOT/runs/$RUN_NAME}
DATA_DIR=$RUN_DIR/teacher_datasets
RESUME_MODEL=${RESUME_MODEL:-$REMOTE_ROOT/runs/v11rxc_nlite125_fix6300031_targeted_far6_stuck10_tiny64/rppo_medium_lstm128x2_observable12_target8_macro_library_v11rxc_stage2/rppo_medium_lstm128x2_observable12_target8_macro_library_v11rxc_stage2.zip}
NORMAL_MODEL=${NORMAL_MODEL:-$RESUME_MODEL}
PRESSURE_MODEL=${PRESSURE_MODEL:-$RESUME_MODEL}
TRAIN_OVERRIDES_REMOTE=${TRAIN_OVERRIDES_REMOTE:-$REMOTE_ROOT/configs/v11/方案N24_v11rxc近N9加远距targeted_far6评估探针.json}
EVAL_OVERRIDES_REMOTE=${EVAL_OVERRIDES_REMOTE:-$REMOTE_ROOT/configs/v11/方案N9_v11rxc近N6但wrapper不注入macro6评估探针.json}
NORMAL_DATASET=$DATA_DIR/普通锚点轨迹_policy_tiny64_n9_v11rxc.npz
PRESSURE_DATASET=$DATA_DIR/压力锚点轨迹_policy_tiny64_n9_v11rxc.npz
NORMAL_SEED_START=${NORMAL_SEED_START:-1810000}
NORMAL_EPISODES=${NORMAL_EPISODES:-24}
PRESSURE_SEED_START=${PRESSURE_SEED_START:-5200000}
PRESSURE_EPISODES=${PRESSURE_EPISODES:-24}
PRESSURE_SCAN=${PRESSURE_SCAN:-5000}
SEED_POOL=${SEED_POOL:-6300031,6300058,6300069,6300006}
ONLINE_BAD_SEED_POOL=${ONLINE_BAD_SEED_POOL:-6300031,6300058,6300069,6300006}
ANCHOR_DATASET_ROLES=${ANCHOR_DATASET_ROLES:-base,base}
ANCHOR_DATASET_WEIGHTS=${ANCHOR_DATASET_WEIGHTS:-1.0,1.25}
ONLINE_TRAINABLE_PARAM_MODE=${ONLINE_TRAINABLE_PARAM_MODE:-policy_head_only}
ANCHOR_TRAINABLE_PARAM_MODE=${ANCHOR_TRAINABLE_PARAM_MODE:-policy_head_only}
HYBRID_ROUNDS=${HYBRID_ROUNDS:-4}
PPO_EPISODES=${PPO_EPISODES:-8}
ANCHOR_EPOCHS=${ANCHOR_EPOCHS:-4}
PHASE2_BAD_SEED_RATIO=${PHASE2_BAD_SEED_RATIO:-1.0}
RECOVERY_WEIGHT=${RECOVERY_WEIGHT:-1.05}
STUCK_WEIGHT=${STUCK_WEIGHT:-1.0}
COLLISION_WEIGHT=${COLLISION_WEIGHT:-1.0}
PPO_LEARNING_RATE=${PPO_LEARNING_RATE:-1e-5}
ANCHOR_LEARNING_RATE=${ANCHOR_LEARNING_RATE:-2e-5}

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
copy_file "configs/v11/方案N24_v11rxc近N9加远距targeted_far6评估探针.json"
copy_file "configs/v11/方案N9_v11rxc近N6但wrapper不注入macro6评估探针.json"

ssh v1002 "mkdir -p '$RUN_DIR' '$DATA_DIR'"

for required in "$RESUME_MODEL" "$NORMAL_MODEL" "$PRESSURE_MODEL"; do
  if ! ssh v1002 "test -f '$required'"; then
    echo "missing_required path=$required" >&2
    exit 1
  fi
done

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
    --action-label-source policy \
    --device cpu \
    --env-overrides-file '$EVAL_OVERRIDES_REMOTE'
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
    --action-label-source policy \
    --device cpu \
    --env-overrides-file '$EVAL_OVERRIDES_REMOTE'
"
maybe_export_dataset "$PRESSURE_DATASET" "$PRESSURE_EXPORT_CMD"

ssh v1002 "cd '$REMOTE_ROOT' && source ~/miniconda3/etc/profile.d/conda.sh && conda activate ML && python train_target8_v11rxc_hybrid_anchor_finetune.py \
  --output-dir '$RUN_DIR' \
  --resume-model '$RESUME_MODEL' \
  --anchor-datasets '$NORMAL_DATASET' '$PRESSURE_DATASET' \
  --hybrid-rounds $HYBRID_ROUNDS \
  --ppo-episodes $PPO_EPISODES \
  --anchor-epochs $ANCHOR_EPOCHS \
  --anchor-dataset-roles '$ANCHOR_DATASET_ROLES' \
  --anchor-dataset-weights '$ANCHOR_DATASET_WEIGHTS' \
  --anchor-success-only \
  --recovery-weight $RECOVERY_WEIGHT \
  --stuck-weight $STUCK_WEIGHT \
  --collision-weight $COLLISION_WEIGHT \
  --phase2-bad-seed-ratio $PHASE2_BAD_SEED_RATIO \
  --ppo-learning-rate $PPO_LEARNING_RATE \
  --anchor-learning-rate $ANCHOR_LEARNING_RATE \
  --online-trainable-param-mode '$ONLINE_TRAINABLE_PARAM_MODE' \
  --anchor-trainable-param-mode '$ANCHOR_TRAINABLE_PARAM_MODE' \
  --seed-pool '$SEED_POOL' \
  --online-bad-seed-pool '$ONLINE_BAD_SEED_POOL' \
  --video-episodes 0 \
  --device cpu \
  --torch-threads 1 \
  --env-overrides-file '$TRAIN_OVERRIDES_REMOTE'"

MODEL_BASE="$RUN_DIR/rppo_medium_lstm128x2_observable12_target8_macro_library_v11rxc_stage2/rppo_medium_lstm128x2_observable12_target8_macro_library_v11rxc_stage2"
MODEL_ZIP="${MODEL_BASE}.zip"

ssh v1002 "cd '$REMOTE_ROOT' && source ~/miniconda3/etc/profile.d/conda.sh && conda activate ML && python eval_target8_bad_seeds.py \
  --model '$MODEL_BASE' \
  --out-dir '${RUN_DIR}_badseed_gate' \
  --state-mode observable12_target8_macro_library_v11rxc_stage2 \
  --env-overrides-file '$EVAL_OVERRIDES_REMOTE' \
  --device cpu \
  --seed-list 6300031,6300058,6300069,6300006"

ssh v1002 "cd '$REMOTE_ROOT' && source ~/miniconda3/etc/profile.d/conda.sh && conda activate ML && python eval_target8_pressure.py \
  --model '$MODEL_BASE' \
  --out-dir '${RUN_DIR}_pressure20' \
  --action-mode v11 \
  --state-mode observable12_target8_macro_library_v11rxc_stage2 \
  --episodes 20 \
  --video-episodes 0 \
  --env-overrides-file '$EVAL_OVERRIDES_REMOTE' \
  --device cpu"

ssh v1002 "cd '$REMOTE_ROOT' && source ~/miniconda3/etc/profile.d/conda.sh && conda activate ML && python eval_target8_sweep.py \
  --train-output-dir '$RUN_DIR' \
  --model-path '$MODEL_ZIP' \
  --out-dir '${RUN_DIR}_normal20' \
  --experiments rppo_medium_lstm128x2_observable12_target8_macro_library_v11rxc_stage2 \
  --episodes 20 \
  --video-episodes 0 \
  --env-overrides-file '$EVAL_OVERRIDES_REMOTE' \
  --device cpu"

echo "run_dir=$RUN_DIR"
