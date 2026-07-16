#!/usr/bin/env bash
set -euo pipefail

LOCAL_ROOT=/home/diana/fishing/blind_nav_rl
REMOTE_ROOT=/home/diana/fishing/blind_nav_rl
RUN_NAME=${RUN_NAME:-v11rx_conflict4_merge_nlite125_macrohead}
RUN_DIR=${RUN_DIR:-$REMOTE_ROOT/runs/$RUN_NAME}
DATA_DIR=$RUN_DIR/teacher_datasets
BASE_MODEL=${BASE_MODEL:-$REMOTE_ROOT/runs/v11rx_conflict4_micro/rppo_medium_lstm128x2_observable12_target8_macro_library_v11rx_stage2/models/checkpoint_ep_00250.zip}
NORMAL_DATASET=${NORMAL_DATASET:-$REMOTE_ROOT/runs/v11re_hybrid_anchor_bridge_micro/teacher_datasets/普通锚点轨迹.npz}
PRESSURE_DATASET=${PRESSURE_DATASET:-$REMOTE_ROOT/runs/v11re_hybrid_anchor_bridge_micro/teacher_datasets/压力锚点轨迹.npz}
SPECIALIST_MODEL=${SPECIALIST_MODEL:-$REMOTE_ROOT/runs/v11rx_tail2_conservative_lite_125/rppo_medium_lstm128x2_observable12_target8_macro_library_v11rx_stage2/models/checkpoint_ep_00125.zip}
SPECIALIST_DATASET=${SPECIALIST_DATASET:-$DATA_DIR/坏种子Nlite125轨迹.npz}
SPECIALIST_SEED_POOL=${SPECIALIST_SEED_POOL:-}
OVERRIDES_REMOTE=${OVERRIDES_REMOTE:-$REMOTE_ROOT/configs/v11/方案Nlite_v11rx尾部轻保守探针.json}
OUTPUT_MODEL=$RUN_DIR/rppo_medium_lstm128x2_observable12_target8_macro_library_v11rx_stage2.zip
DATASET_WEIGHTS=${DATASET_WEIGHTS:-1.0,1.2,1.6}
DATASET_LOSS_HEADS=${DATASET_LOSS_HEADS:-all,all,macro_only}
TRAINABLE_PARAM_MODE=${TRAINABLE_PARAM_MODE:-all}
EPOCHS=${EPOCHS:-10}
LEARNING_RATE=${LEARNING_RATE:-3e-5}
RECOVERY_WEIGHT=${RECOVERY_WEIGHT:-1.5}
STUCK_WEIGHT=${STUCK_WEIGHT:-1.3}
COLLISION_WEIGHT=${COLLISION_WEIGHT:-1.2}

copy_file() {
  local rel="$1"
  ssh v1002 "mkdir -p '$REMOTE_ROOT/$(dirname "$rel")'"
  scp "$LOCAL_ROOT/$rel" "v1002:$REMOTE_ROOT/$rel"
}

copy_file benchmark_state_dims_10k.py
copy_file blind_nav_rl/env.py
copy_file eval_target8_sweep.py
copy_file eval_target8_pressure.py
copy_file eval_target8_bad_seeds.py
copy_file export_v11_teacher_dataset.py
copy_file distill_v11rx_policy_bc.py
copy_file "configs/v11/方案Nlite_v11rx尾部轻保守探针.json"

ssh v1002 "mkdir -p '$RUN_DIR' '$DATA_DIR'"

for required in "$BASE_MODEL" "$NORMAL_DATASET" "$PRESSURE_DATASET" "$SPECIALIST_MODEL"; do
  if ! ssh v1002 "test -f '$required'"; then
    echo "missing_required path=$required" >&2
    exit 1
  fi
done

if ! ssh v1002 "test -f '$SPECIALIST_DATASET'"; then
  ssh v1002 "cd '$REMOTE_ROOT' && source ~/miniconda3/etc/profile.d/conda.sh && conda activate ML && python export_v11_teacher_dataset.py \
    --model '$SPECIALIST_MODEL' \
    --out '$SPECIALIST_DATASET' \
    --scenario diagonal \
    ${SPECIALIST_SEED_POOL:+--seed-list '$SPECIALIST_SEED_POOL'} \
    --state-mode observable12_target8_macro_library_v11rx_stage2 \
    --device cpu \
    --env-overrides-file '$OVERRIDES_REMOTE'"
fi

ssh v1002 "cd '$REMOTE_ROOT' && source ~/miniconda3/etc/profile.d/conda.sh && conda activate ML && python distill_v11rx_policy_bc.py \
  --teacher-datasets '$NORMAL_DATASET' '$PRESSURE_DATASET' '$SPECIALIST_DATASET' \
  --resume-model '$BASE_MODEL' \
  --output-model '$OUTPUT_MODEL' \
  --epochs $EPOCHS \
  --learning-rate $LEARNING_RATE \
  --dataset-roles base,base,specialist \
  --dataset-weights '$DATASET_WEIGHTS' \
  --dataset-action-formats v11re,v11re,v11rx \
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
  --state-mode observable12_target8_macro_library_v11rx_stage2 \
  --env-overrides-file '$OVERRIDES_REMOTE' \
  --device cpu"

ssh v1002 "cd '$REMOTE_ROOT' && source ~/miniconda3/etc/profile.d/conda.sh && conda activate ML && python eval_target8_pressure.py \
  --model '$OUTPUT_MODEL' \
  --out-dir '${RUN_DIR}_pressure20' \
  --action-mode v11 \
  --state-mode observable12_target8_macro_library_v11rx_stage2 \
  --episodes 20 \
  --video-episodes 0 \
  --env-overrides-file '$OVERRIDES_REMOTE' \
  --device cpu"

ssh v1002 "cd '$REMOTE_ROOT' && source ~/miniconda3/etc/profile.d/conda.sh && conda activate ML && python eval_target8_sweep.py \
  --train-output-dir '$RUN_DIR' \
  --model-path '$RUN_DIR/rppo_medium_lstm128x2_observable12_target8_macro_library_v11rx_stage2' \
  --out-dir '${RUN_DIR}_normal20' \
  --experiments rppo_medium_lstm128x2_observable12_target8_macro_library_v11rx_stage2 \
  --episodes 20 \
  --video-episodes 0 \
  --env-overrides-file '$OVERRIDES_REMOTE' \
  --device cpu"

echo "run_dir=$RUN_DIR"
