#!/usr/bin/env bash
set -euo pipefail

LOCAL_ROOT=/home/diana/fishing/blind_nav_rl
REMOTE_ROOT=/home/diana/fishing/blind_nav_rl
RUN_NAME=${RUN_NAME:-v11rxc_two_stage_6300031push112_tail2_N3stabilize32}
RUN_DIR=${RUN_DIR:-$REMOTE_ROOT/runs/$RUN_NAME}
RESUME_MODEL=${RESUME_MODEL:-$REMOTE_ROOT/runs/v11rx_tail2_conservative_lite_125/rppo_medium_lstm128x2_observable12_target8_macro_library_v11rx_stage2/models/checkpoint_ep_00125.zip}
PHASE1_OVERRIDES=${PHASE1_OVERRIDES:-$REMOTE_ROOT/configs/v11/方案Nlite_v11rxc修6300031远距定向macro6探针.json}
PHASE2_OVERRIDES=${PHASE2_OVERRIDES:-$REMOTE_ROOT/configs/v11/方案N3_v11rxc尾部更强抑激进稳定探针.json}
PHASE1_SEED_POOL=${PHASE1_SEED_POOL:-6300031}
PHASE2_SEED_POOL=${PHASE2_SEED_POOL:-6300058,6300006}
PHASE1_EPISODES=${PHASE1_EPISODES:-112}
PHASE2_EPISODES=${PHASE2_EPISODES:-32}
PHASE2_BAD_SEED_RATIO=${PHASE2_BAD_SEED_RATIO:-1.0}
PHASE1_LEARNING_RATE=${PHASE1_LEARNING_RATE:-}
PHASE2_LEARNING_RATE=${PHASE2_LEARNING_RATE:-}

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
copy_file train_target8_v11re_reward_exit_curriculum.py
copy_file train_target8_v11re_bad_seed_mergeback_curriculum.py
copy_file train_target8_v11rxc_bad_seed_curriculum.py
copy_file train_target8_v11rxc_bad_seed_mergeback_curriculum.py
copy_file "configs/v11/方案Nlite_v11rxc修6300031远距定向macro6探针.json"
copy_file "configs/v11/方案N3_v11rxc尾部更强抑激进稳定探针.json"
copy_file "configs/v11/方案Nlite_v11rxc尾部轻保守探针.json"

ssh v1002 "mkdir -p '$RUN_DIR'"

if ! ssh v1002 "test -f '$RESUME_MODEL'"; then
  echo "missing_required path=$RESUME_MODEL" >&2
  exit 1
fi

ssh v1002 "cd '$REMOTE_ROOT' && source ~/miniconda3/etc/profile.d/conda.sh && conda activate ML && python train_target8_v11rxc_bad_seed_mergeback_curriculum.py \
  --output-dir '$RUN_DIR' \
  --resume-model '$RESUME_MODEL' \
  --phase1-episodes $PHASE1_EPISODES \
  --phase2-episodes $PHASE2_EPISODES \
  --phase1-seed-pool '$PHASE1_SEED_POOL' \
  --phase2-seed-pool '$PHASE2_SEED_POOL' \
  --phase2-bad-seed-ratio $PHASE2_BAD_SEED_RATIO \
  ${PHASE1_LEARNING_RATE:+--phase1-learning-rate $PHASE1_LEARNING_RATE} \
  ${PHASE2_LEARNING_RATE:+--phase2-learning-rate $PHASE2_LEARNING_RATE} \
  --bad-seed-env-overrides-file '$PHASE1_OVERRIDES' \
  --mergeback-env-overrides-file '$PHASE2_OVERRIDES' \
  --video-episodes 0 \
  --device cpu \
  --torch-threads 1"

MODEL_BASE="$RUN_DIR/rppo_medium_lstm128x2_observable12_target8_macro_library_v11rxc_stage2/rppo_medium_lstm128x2_observable12_target8_macro_library_v11rxc_stage2"
MODEL_ZIP="${MODEL_BASE}.zip"
EVAL_OVERRIDES="$REMOTE_ROOT/configs/v11/方案Nlite_v11rxc尾部轻保守探针.json"

ssh v1002 "cd '$REMOTE_ROOT' && source ~/miniconda3/etc/profile.d/conda.sh && conda activate ML && python eval_target8_bad_seeds.py \
  --model '$MODEL_BASE' \
  --out-dir '${RUN_DIR}_badseed_gate' \
  --state-mode observable12_target8_macro_library_v11rxc_stage2 \
  --env-overrides-file '$EVAL_OVERRIDES' \
  --device cpu \
  --seed-list 6300031,6300058,6300069,6300006"

ssh v1002 "cd '$REMOTE_ROOT' && source ~/miniconda3/etc/profile.d/conda.sh && conda activate ML && python eval_target8_pressure.py \
  --model '$MODEL_BASE' \
  --out-dir '${RUN_DIR}_pressure20' \
  --action-mode v11 \
  --state-mode observable12_target8_macro_library_v11rxc_stage2 \
  --episodes 20 \
  --video-episodes 0 \
  --env-overrides-file '$EVAL_OVERRIDES' \
  --device cpu"

ssh v1002 "cd '$REMOTE_ROOT' && source ~/miniconda3/etc/profile.d/conda.sh && conda activate ML && python eval_target8_sweep.py \
  --train-output-dir '$RUN_DIR' \
  --model-path '$MODEL_ZIP' \
  --out-dir '${RUN_DIR}_normal20' \
  --experiments rppo_medium_lstm128x2_observable12_target8_macro_library_v11rxc_stage2 \
  --episodes 20 \
  --video-episodes 0 \
  --env-overrides-file '$EVAL_OVERRIDES' \
  --device cpu"

echo "run_dir=$RUN_DIR"
