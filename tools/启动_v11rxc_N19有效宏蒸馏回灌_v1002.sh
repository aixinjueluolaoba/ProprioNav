#!/usr/bin/env bash
set -euo pipefail

LOCAL_ROOT=/home/diana/fishing/blind_nav_rl
REMOTE_ROOT=/home/diana/fishing/blind_nav_rl
RUN_NAME=${RUN_NAME:-v11rxc_n19_effective_macro_distill_tiny}
RUN_DIR=${RUN_DIR:-$REMOTE_ROOT/runs/$RUN_NAME}
DATA_DIR=$RUN_DIR/teacher_datasets
BASE_MODEL=${BASE_MODEL:-$REMOTE_ROOT/runs/v11rxc_nlite125_fix6300031_targeted_far6_stuck10_search112/rppo_medium_lstm128x2_observable12_target8_macro_library_v11rxc_stage2/models/checkpoint_ep_00112.zip}
RESUME_MODEL=${RESUME_MODEL:-$BASE_MODEL}
NORMAL_MODEL=${NORMAL_MODEL:-$BASE_MODEL}
PRESSURE_MODEL=${PRESSURE_MODEL:-$BASE_MODEL}
SPECIALIST_MODEL=${SPECIALIST_MODEL:-$BASE_MODEL}
OVERRIDES_REMOTE=${OVERRIDES_REMOTE:-$REMOTE_ROOT/configs/v11/方案N19_v11rxc中距抑制宏长链覆写评估探针.json}
NORMAL_DATASET=$DATA_DIR/普通锚点轨迹_policy_v11rxc_N19.npz
PRESSURE_DATASET=$DATA_DIR/压力锚点轨迹_effective_v11rxc_N19.npz
SPECIALIST_DATASET=$DATA_DIR/压力专科轨迹_effective_v11rxc_N19.npz
OUTPUT_MODEL=$RUN_DIR/rppo_medium_lstm128x2_observable12_target8_macro_library_v11rxc_stage2.zip
NORMAL_SEED_START=${NORMAL_SEED_START:-1810000}
NORMAL_EPISODES=${NORMAL_EPISODES:-24}
PRESSURE_SEED_START=${PRESSURE_SEED_START:-5200000}
PRESSURE_EPISODES=${PRESSURE_EPISODES:-24}
PRESSURE_SCAN=${PRESSURE_SCAN:-5000}
SPECIALIST_SEED_POOL=${SPECIALIST_SEED_POOL:-5200015,5200015,5200015,5200015,5200008,5200022}
NORMAL_ACTION_LABEL_SOURCE=${NORMAL_ACTION_LABEL_SOURCE:-policy}
PRESSURE_ACTION_LABEL_SOURCE=${PRESSURE_ACTION_LABEL_SOURCE:-effective_macro_speed}
SPECIALIST_ACTION_LABEL_SOURCE=${SPECIALIST_ACTION_LABEL_SOURCE:-effective_macro_speed}
DATASET_WEIGHTS=${DATASET_WEIGHTS:-1.0,1.4,2.2}
DATASET_ROLES=${DATASET_ROLES:-base,generic,specialist}
DATASET_LOSS_HEADS=${DATASET_LOSS_HEADS:-all,speed_macro,speed_macro}
TRAINABLE_PARAM_MODE=${TRAINABLE_PARAM_MODE:-action_net_only}
EPOCHS=${EPOCHS:-8}
LEARNING_RATE=${LEARNING_RATE:-3e-5}
RECOVERY_WEIGHT=${RECOVERY_WEIGHT:-1.2}
STUCK_WEIGHT=${STUCK_WEIGHT:-1.15}
COLLISION_WEIGHT=${COLLISION_WEIGHT:-1.1}

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
copy_file distill_v11rxc_policy_bc.py
copy_file "configs/v11/方案N19_v11rxc中距抑制宏长链覆写评估探针.json"

ssh v1002 "mkdir -p '$RUN_DIR' '$DATA_DIR'"

for required in "$RESUME_MODEL" "$NORMAL_MODEL" "$PRESSURE_MODEL" "$SPECIALIST_MODEL"; do
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
    --action-label-source '$NORMAL_ACTION_LABEL_SOURCE' \
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
    --action-label-source '$PRESSURE_ACTION_LABEL_SOURCE' \
    --device cpu \
    --env-overrides-file '$OVERRIDES_REMOTE'
"
maybe_export_dataset "$PRESSURE_DATASET" "$PRESSURE_EXPORT_CMD"

SPECIALIST_EXPORT_CMD="
  cd $REMOTE_ROOT &&
  source ~/miniconda3/etc/profile.d/conda.sh &&
  conda activate ML &&
  python export_v11_teacher_dataset.py \
    --model '$SPECIALIST_MODEL' \
    --out '$SPECIALIST_DATASET' \
    --scenario pressure \
    --seed-list '$SPECIALIST_SEED_POOL' \
    --state-mode observable12_target8_macro_library_v11rxc_stage2 \
    --action-label-source '$SPECIALIST_ACTION_LABEL_SOURCE' \
    --device cpu \
    --env-overrides-file '$OVERRIDES_REMOTE'
"
maybe_export_dataset "$SPECIALIST_DATASET" "$SPECIALIST_EXPORT_CMD"

ssh v1002 "cd '$REMOTE_ROOT' && source ~/miniconda3/etc/profile.d/conda.sh && conda activate ML && python distill_v11rxc_policy_bc.py \
  --teacher-datasets '$NORMAL_DATASET' '$PRESSURE_DATASET' '$SPECIALIST_DATASET' \
  --resume-model '$RESUME_MODEL' \
  --output-model '$OUTPUT_MODEL' \
  --epochs $EPOCHS \
  --learning-rate $LEARNING_RATE \
  --dataset-roles '$DATASET_ROLES' \
  --dataset-weights '$DATASET_WEIGHTS' \
  --dataset-action-formats v11rxc,v11rxc,v11rxc \
  --dataset-loss-heads '$DATASET_LOSS_HEADS' \
  --trainable-param-mode '$TRAINABLE_PARAM_MODE' \
  --dataset-remap-macro4-to-7 0,0,0 \
  --success-only \
  --recovery-weight $RECOVERY_WEIGHT \
  --stuck-weight $STUCK_WEIGHT \
  --collision-weight $COLLISION_WEIGHT \
  --device cpu"

ssh v1002 "cd '$REMOTE_ROOT' && source ~/miniconda3/etc/profile.d/conda.sh && conda activate ML && python eval_target8_bad_seeds.py \
  --model '$OUTPUT_MODEL' \
  --out-dir '${RUN_DIR}_badseed' \
  --state-mode observable12_target8_macro_library_v11rxc_stage2 \
  --env-overrides-file '$OVERRIDES_REMOTE' \
  --device cpu \
  --seed-list 6300031,6300058,6300069,6300006"

ssh v1002 "cd '$REMOTE_ROOT' && source ~/miniconda3/etc/profile.d/conda.sh && conda activate ML && python eval_target8_pressure.py \
  --model '$OUTPUT_MODEL' \
  --out-dir '${RUN_DIR}_pressure20' \
  --action-mode v11 \
  --state-mode observable12_target8_macro_library_v11rxc_stage2 \
  --episodes 20 \
  --video-episodes 0 \
  --env-overrides-file '$OVERRIDES_REMOTE' \
  --device cpu"

ssh v1002 "cd '$REMOTE_ROOT' && source ~/miniconda3/etc/profile.d/conda.sh && conda activate ML && python eval_target8_sweep.py \
  --train-output-dir '$RUN_DIR' \
  --model-path '$OUTPUT_MODEL' \
  --out-dir '${RUN_DIR}_normal20' \
  --experiments rppo_medium_lstm128x2_observable12_target8_macro_library_v11rxc_stage2 \
  --episodes 20 \
  --video-episodes 0 \
  --env-overrides-file '$OVERRIDES_REMOTE' \
  --device cpu"

echo "run_dir=$RUN_DIR"
