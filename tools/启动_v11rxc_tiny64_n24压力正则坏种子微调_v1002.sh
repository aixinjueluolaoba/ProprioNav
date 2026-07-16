#!/usr/bin/env bash
set -euo pipefail

LOCAL_ROOT=/home/diana/fishing/blind_nav_rl
REMOTE_ROOT=/home/diana/fishing/blind_nav_rl
RUN_NAME=${RUN_NAME:-v11rxc_tiny64_n24_pressure_reg_all}
RUN_DIR=${RUN_DIR:-$REMOTE_ROOT/runs/$RUN_NAME}
DATA_DIR=$RUN_DIR/teacher_datasets
RESUME_MODEL=${RESUME_MODEL:-$REMOTE_ROOT/runs/v11rxc_nlite125_fix6300031_targeted_far6_stuck10_tiny64/rppo_medium_lstm128x2_observable12_target8_macro_library_v11rxc_stage2/rppo_medium_lstm128x2_observable12_target8_macro_library_v11rxc_stage2.zip}
PRESSURE_MODEL=${PRESSURE_MODEL:-$RESUME_MODEL}
TRAIN_OVERRIDES_REMOTE=${TRAIN_OVERRIDES_REMOTE:-$REMOTE_ROOT/configs/v11/方案N24_v11rxc近N9加远距targeted_far6评估探针.json}
EVAL_OVERRIDES_REMOTE=${EVAL_OVERRIDES_REMOTE:-$REMOTE_ROOT/configs/v11/方案N9_v11rxc近N6但wrapper不注入macro6评估探针.json}
PRESSURE_DATASET=$DATA_DIR/压力锚点轨迹_policy_tiny64_n9_v11rxc.npz
PRESSURE_SEED_START=${PRESSURE_SEED_START:-5200000}
PRESSURE_EPISODES=${PRESSURE_EPISODES:-24}
PRESSURE_SCAN=${PRESSURE_SCAN:-5000}
SEED_POOL=${SEED_POOL:-6300031,6300058,6300069,6300006}
TRAIN_EXTRA_ARGS=${TRAIN_EXTRA_ARGS:---tiny}
TRAINABLE_PARAM_MODE=${TRAINABLE_PARAM_MODE:-all}
LEARNING_RATE=${LEARNING_RATE:-1e-5}
PRESSURE_LOSS_WEIGHT=${PRESSURE_LOSS_WEIGHT:-0.6}
PRESSURE_BATCH_SIZE=${PRESSURE_BATCH_SIZE:-128}
PRESSURE_LOSS_HEADS=${PRESSURE_LOSS_HEADS:-all}

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
copy_file train_target8_v11rxc_pressure_regularized_bad_seed.py
copy_file "configs/v11/方案N24_v11rxc近N9加远距targeted_far6评估探针.json"
copy_file "configs/v11/方案N9_v11rxc近N6但wrapper不注入macro6评估探针.json"

ssh v1002 "mkdir -p '$RUN_DIR' '$DATA_DIR'"

for required in "$RESUME_MODEL" "$PRESSURE_MODEL"; do
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
    --seed-start $PRESSURE_SEED_START \
    --episodes $PRESSURE_EPISODES \
    --pressure-scan $PRESSURE_SCAN \
    --action-label-source policy \
    --device cpu \
    --env-overrides-file '$EVAL_OVERRIDES_REMOTE'
"

if ssh v1002 "test -f '$PRESSURE_DATASET'"; then
  echo "dataset_exists path=$PRESSURE_DATASET"
else
  ssh v1002 "bash -lc \"$PRESSURE_EXPORT_CMD\""
  echo "dataset_exported path=$PRESSURE_DATASET"
fi

ssh v1002 "cd '$REMOTE_ROOT' && source ~/miniconda3/etc/profile.d/conda.sh && conda activate ML && python train_target8_v11rxc_pressure_regularized_bad_seed.py \
  $TRAIN_EXTRA_ARGS \
  --output-dir '$RUN_DIR' \
  --resume-model '$RESUME_MODEL' \
  --seed-pool '$SEED_POOL' \
  --env-overrides-file '$TRAIN_OVERRIDES_REMOTE' \
  --pressure-dataset '$PRESSURE_DATASET' \
  --pressure-loss-weight $PRESSURE_LOSS_WEIGHT \
  --pressure-batch-size $PRESSURE_BATCH_SIZE \
  --pressure-loss-heads '$PRESSURE_LOSS_HEADS' \
  --trainable-param-mode '$TRAINABLE_PARAM_MODE' \
  --learning-rate $LEARNING_RATE \
  --video-episodes 0 \
  --device cpu \
  --torch-threads 1"

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
