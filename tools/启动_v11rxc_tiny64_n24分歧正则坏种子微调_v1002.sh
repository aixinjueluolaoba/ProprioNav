#!/usr/bin/env bash
set -euo pipefail

LOCAL_ROOT=/home/diana/fishing/blind_nav_rl
REMOTE_ROOT=/home/diana/fishing/blind_nav_rl
RUN_NAME=${RUN_NAME:-v11rxc_tiny64_n24_disagree_reg_all}
RUN_DIR=${RUN_DIR:-$REMOTE_ROOT/runs/$RUN_NAME}
DATA_DIR=$RUN_DIR/teacher_datasets
RESUME_MODEL=${RESUME_MODEL:-$REMOTE_ROOT/runs/v11rxc_nlite125_fix6300031_targeted_far6_stuck10_tiny64/rppo_medium_lstm128x2_observable12_target8_macro_library_v11rxc_stage2/rppo_medium_lstm128x2_observable12_target8_macro_library_v11rxc_stage2.zip}
PRESSURE_MODEL=${PRESSURE_MODEL:-$RESUME_MODEL}
SPECIALIST_MODEL=${SPECIALIST_MODEL:-$RESUME_MODEL}
TRAIN_OVERRIDES_REMOTE=${TRAIN_OVERRIDES_REMOTE:-$REMOTE_ROOT/configs/v11/方案N24_v11rxc近N9加远距targeted_far6评估探针.json}
EVAL_OVERRIDES_REMOTE=${EVAL_OVERRIDES_REMOTE:-$REMOTE_ROOT/configs/v11/方案N9_v11rxc近N6但wrapper不注入macro6评估探针.json}
PRESSURE_DATASET=$DATA_DIR/压力锚点轨迹_policy_tiny64_n9_v11rxc.npz
SPECIALIST_DATASET=$DATA_DIR/坏种子N24_effective_tiny64_v11rxc.npz
PRESSURE_DATASET_OVERRIDE=${PRESSURE_DATASET_OVERRIDE:-}
SPECIALIST_DATASET_OVERRIDE=${SPECIALIST_DATASET_OVERRIDE:-}
if [[ -n "$PRESSURE_DATASET_OVERRIDE" ]]; then
  PRESSURE_DATASET="$PRESSURE_DATASET_OVERRIDE"
fi
if [[ -n "$SPECIALIST_DATASET_OVERRIDE" ]]; then
  SPECIALIST_DATASET="$SPECIALIST_DATASET_OVERRIDE"
fi
PRESSURE_SEED_START=${PRESSURE_SEED_START:-5200000}
PRESSURE_EPISODES=${PRESSURE_EPISODES:-24}
PRESSURE_SCAN=${PRESSURE_SCAN:-5000}
PRESSURE_SEED_LIST=${PRESSURE_SEED_LIST:-}
SPECIALIST_SEED_POOL=${SPECIALIST_SEED_POOL:-6300031,6300058,6300069,6300006}
SEED_POOL=${SEED_POOL:-6300031,6300058,6300069,6300006}
TRAIN_EXTRA_ARGS=${TRAIN_EXTRA_ARGS:---tiny}
TRAINABLE_PARAM_MODE=${TRAINABLE_PARAM_MODE:-all}
LEARNING_RATE=${LEARNING_RATE:-1e-5}
PRESSURE_LOSS_WEIGHT=${PRESSURE_LOSS_WEIGHT:-0.45}
PRESSURE_LOSS_WEIGHT_LATE=${PRESSURE_LOSS_WEIGHT_LATE:-}
PRESSURE_BATCH_SIZE=${PRESSURE_BATCH_SIZE:-128}
PRESSURE_LOSS_HEADS=${PRESSURE_LOSS_HEADS:-all}
SPECIALIST_LOSS_WEIGHT=${SPECIALIST_LOSS_WEIGHT:-1.8}
SPECIALIST_LOSS_WEIGHT_LATE=${SPECIALIST_LOSS_WEIGHT_LATE:-}
SPECIALIST_BATCH_SIZE=${SPECIALIST_BATCH_SIZE:-128}
SPECIALIST_LOSS_HEADS=${SPECIALIST_LOSS_HEADS:-macro_only}
SPECIALIST_DISAGREEMENT_FILTER=${SPECIALIST_DISAGREEMENT_FILTER:-macro}
SPECIALIST_SEED_WHITELIST=${SPECIALIST_SEED_WHITELIST:-}
SPECIALIST_SEED_MODE_WHITELIST=${SPECIALIST_SEED_MODE_WHITELIST:-}
SPECIALIST_MIN_STUCK_TIME=${SPECIALIST_MIN_STUCK_TIME:-}
SPECIALIST_MIN_NO_PROGRESS_TIME=${SPECIALIST_MIN_NO_PROGRESS_TIME:-}
SPECIALIST_RECOVERY_MODES=${SPECIALIST_RECOVERY_MODES:-}
LOSS_WEIGHT_SWITCH_FRAC=${LOSS_WEIGHT_SWITCH_FRAC:-}

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
copy_file train_target8_v11rxc_disagreement_regularized_bad_seed.py
copy_file "configs/v11/方案N24_v11rxc近N9加远距targeted_far6评估探针.json"
copy_file "configs/v11/方案N9_v11rxc近N6但wrapper不注入macro6评估探针.json"

ssh v1002 "mkdir -p '$RUN_DIR' '$DATA_DIR'"

for required in "$RESUME_MODEL" "$PRESSURE_MODEL" "$SPECIALIST_MODEL"; do
  if ! ssh v1002 "test -f '$required'"; then
    echo "missing_required path=$required" >&2
    exit 1
  fi
done

PRESSURE_EXPORT_CMD="
  cd $REMOTE_ROOT &&
  source ~/miniconda3/etc/profile.d/conda.sh &&
  conda activate ML &&
  python export_v11_teacher_dataset.py \
    --model '$PRESSURE_MODEL' \
    --out '$PRESSURE_DATASET' \
    --scenario pressure \
    --state-mode observable12_target8_macro_library_v11rxc_stage2 \
    --action-label-source policy \
    --device cpu \
    --env-overrides-file '$EVAL_OVERRIDES_REMOTE'"

if [[ -n "$PRESSURE_SEED_LIST" ]]; then
  PRESSURE_EXPORT_CMD="$PRESSURE_EXPORT_CMD \"--seed-list\" \"$PRESSURE_SEED_LIST\""
else
  PRESSURE_EXPORT_CMD="$PRESSURE_EXPORT_CMD \"--seed-start\" \"$PRESSURE_SEED_START\" \"--episodes\" \"$PRESSURE_EPISODES\" \"--pressure-scan\" \"$PRESSURE_SCAN\""
fi

SPECIALIST_EXPORT_CMD="
  cd $REMOTE_ROOT &&
  source ~/miniconda3/etc/profile.d/conda.sh &&
  conda activate ML &&
  python export_v11_teacher_dataset.py \
    --model '$SPECIALIST_MODEL' \
    --out '$SPECIALIST_DATASET' \
    --scenario diagonal \
    --seed-list '$SPECIALIST_SEED_POOL' \
    --state-mode observable12_target8_macro_library_v11rxc_stage2 \
    --action-label-source effective_macro \
    --device cpu \
    --env-overrides-file '$TRAIN_OVERRIDES_REMOTE'
"

if ssh v1002 "test -f '$PRESSURE_DATASET'"; then
  echo "dataset_exists path=$PRESSURE_DATASET"
else
  ssh v1002 "bash -lc \"$PRESSURE_EXPORT_CMD\""
  echo "dataset_exported path=$PRESSURE_DATASET"
fi

if ssh v1002 "test -f '$SPECIALIST_DATASET'"; then
  echo "dataset_exists path=$SPECIALIST_DATASET"
else
  ssh v1002 "bash -lc \"$SPECIALIST_EXPORT_CMD\""
  echo "dataset_exported path=$SPECIALIST_DATASET"
fi

ssh v1002 "cd '$REMOTE_ROOT' && source ~/miniconda3/etc/profile.d/conda.sh && conda activate ML && python train_target8_v11rxc_disagreement_regularized_bad_seed.py \
  $TRAIN_EXTRA_ARGS \
  --output-dir '$RUN_DIR' \
  --resume-model '$RESUME_MODEL' \
  --seed-pool '$SEED_POOL' \
  --env-overrides-file '$TRAIN_OVERRIDES_REMOTE' \
  --pressure-dataset '$PRESSURE_DATASET' \
  --pressure-loss-weight $PRESSURE_LOSS_WEIGHT \
  ${PRESSURE_LOSS_WEIGHT_LATE:+--pressure-loss-weight-late "$PRESSURE_LOSS_WEIGHT_LATE"} \
  --pressure-batch-size $PRESSURE_BATCH_SIZE \
  --pressure-loss-heads '$PRESSURE_LOSS_HEADS' \
  --specialist-dataset '$SPECIALIST_DATASET' \
  --specialist-loss-weight $SPECIALIST_LOSS_WEIGHT \
  ${SPECIALIST_LOSS_WEIGHT_LATE:+--specialist-loss-weight-late "$SPECIALIST_LOSS_WEIGHT_LATE"} \
  --specialist-batch-size $SPECIALIST_BATCH_SIZE \
  --specialist-loss-heads '$SPECIALIST_LOSS_HEADS' \
  --specialist-disagreement-filter '$SPECIALIST_DISAGREEMENT_FILTER' \
  --trainable-param-mode '$TRAINABLE_PARAM_MODE' \
  --learning-rate $LEARNING_RATE \
  --video-episodes 0 \
  --device cpu \
  --torch-threads 1 \
  ${SPECIALIST_SEED_WHITELIST:+--specialist-seed-whitelist "$SPECIALIST_SEED_WHITELIST"} \
  ${SPECIALIST_SEED_MODE_WHITELIST:+--specialist-seed-mode-whitelist "$SPECIALIST_SEED_MODE_WHITELIST"} \
  ${SPECIALIST_MIN_STUCK_TIME:+--specialist-min-stuck-time "$SPECIALIST_MIN_STUCK_TIME"} \
  ${SPECIALIST_MIN_NO_PROGRESS_TIME:+--specialist-min-no-progress-time "$SPECIALIST_MIN_NO_PROGRESS_TIME"} \
  ${SPECIALIST_RECOVERY_MODES:+--specialist-recovery-modes "$SPECIALIST_RECOVERY_MODES"} \
  ${LOSS_WEIGHT_SWITCH_FRAC:+--loss-weight-switch-frac "$LOSS_WEIGHT_SWITCH_FRAC"}"

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
