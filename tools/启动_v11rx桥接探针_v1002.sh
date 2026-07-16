#!/usr/bin/env bash
set -euo pipefail

LOCAL_ROOT=/home/diana/fishing/blind_nav_rl
REMOTE_ROOT=/home/diana/fishing/blind_nav_rl
RUN_DIR=$REMOTE_ROOT/runs/v11rx_bridge_probe
MODEL_PATH=$RUN_DIR/rppo_medium_lstm128x2_observable12_target8_macro_library_v11rx_stage2.zip
BASE_MODEL=$REMOTE_ROOT/runs/v11re_reward_exit_probe/rppo_medium_lstm128x2_observable12_target8_macro_library_v11re_stage2/models/checkpoint_ep_01000.zip
NORMAL_DATASET=$REMOTE_ROOT/runs/v11re_hybrid_anchor_bridge_micro/teacher_datasets/普通锚点轨迹.npz
PRESSURE_DATASET=$REMOTE_ROOT/runs/v11re_hybrid_anchor_bridge_micro/teacher_datasets/压力锚点轨迹.npz
SPECIALIST_DATASET=$REMOTE_ROOT/runs/v11re_bridge_bc_probe/坏种子拼接锚点轨迹.npz
OVERRIDES_REMOTE=$REMOTE_ROOT/configs/v11/方案M_v11rx短左回正探针.json

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
copy_file distill_v11rx_policy_bc.py
copy_file "configs/v11/方案M_v11rx短左回正探针.json"

ssh v1002 "mkdir -p '$RUN_DIR'"

for required in "$BASE_MODEL" "$NORMAL_DATASET" "$PRESSURE_DATASET" "$SPECIALIST_DATASET"; do
  if ! ssh v1002 "test -f '$required'"; then
    echo "missing_required path=$required" >&2
    exit 1
  fi
done

ssh v1002 "cd '$REMOTE_ROOT' && source ~/miniconda3/etc/profile.d/conda.sh && conda activate ML && python distill_v11rx_policy_bc.py \
  --teacher-datasets '$NORMAL_DATASET' '$PRESSURE_DATASET' '$SPECIALIST_DATASET' \
  --resume-model '$BASE_MODEL' \
  --output-model '$MODEL_PATH' \
  --epochs 12 \
  --learning-rate 5e-5 \
  --dataset-roles base,base,specialist \
  --dataset-weights 1.0,1.2,3.0 \
  --dataset-remap-macro4-to-7 0,0,1 \
  --success-only \
  --recovery-weight 1.8 \
  --stuck-weight 1.5 \
  --collision-weight 1.3 \
  --device cpu"

ssh v1002 "cd '$REMOTE_ROOT' && source ~/miniconda3/etc/profile.d/conda.sh && conda activate ML && python eval_target8_sweep.py \
  --train-output-dir '$RUN_DIR' \
  --model-path '$RUN_DIR/rppo_medium_lstm128x2_observable12_target8_macro_library_v11rx_stage2' \
  --out-dir '$RUN_DIR/normal20' \
  --experiments rppo_medium_lstm128x2_observable12_target8_macro_library_v11rx_stage2 \
  --episodes 20 \
  --video-episodes 0 \
  --env-overrides-file '$OVERRIDES_REMOTE'"

ssh v1002 "cd '$REMOTE_ROOT' && source ~/miniconda3/etc/profile.d/conda.sh && conda activate ML && python eval_target8_pressure.py \
  --model '$MODEL_PATH' \
  --out-dir '$RUN_DIR/pressure20' \
  --action-mode v11 \
  --state-mode observable12_target8_macro_library_v11rx_stage2 \
  --episodes 20 \
  --video-episodes 0 \
  --env-overrides-file '$OVERRIDES_REMOTE' \
  --device cpu"

ssh v1002 "cd '$REMOTE_ROOT' && source ~/miniconda3/etc/profile.d/conda.sh && conda activate ML && python eval_target8_bad_seeds.py \
  --model '$MODEL_PATH' \
  --out-dir '$RUN_DIR/badseed' \
  --state-mode observable12_target8_macro_library_v11rx_stage2 \
  --env-overrides-file '$OVERRIDES_REMOTE' \
  --device cpu"

echo "run_dir=$RUN_DIR"
