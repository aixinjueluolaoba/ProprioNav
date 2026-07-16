#!/usr/bin/env bash
set -euo pipefail

LOCAL_ROOT=/home/diana/fishing/blind_nav_rl
REMOTE_ROOT=/home/diana/fishing/blind_nav_rl
RUN_NAME=${RUN_NAME:-v11rxc_tiny64_targeted_far6_fix6300031_tiny}
RUN_DIR=${RUN_DIR:-$REMOTE_ROOT/runs/$RUN_NAME}
RESUME_MODEL=${RESUME_MODEL:-$REMOTE_ROOT/runs/v11rxc_nlite125_fix6300031_targeted_far6_stuck10_tiny64/rppo_medium_lstm128x2_observable12_target8_macro_library_v11rxc_stage2/rppo_medium_lstm128x2_observable12_target8_macro_library_v11rxc_stage2.zip}
TRAIN_OVERRIDES=${TRAIN_OVERRIDES:-$REMOTE_ROOT/configs/v11/方案Nlite_v11rxc修6300031远距定向macro6探针.json}
EVAL_OVERRIDES=${EVAL_OVERRIDES:-$REMOTE_ROOT/configs/v11/方案N9_v11rxc近N6但wrapper不注入macro6评估探针.json}
SEED_POOL=${SEED_POOL:-6300031}
EPISODES=${EPISODES:-64}

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
copy_file train_target8_v11rxc_bad_seed_curriculum.py
copy_file "configs/v11/方案Nlite_v11rxc修6300031远距定向macro6探针.json"
copy_file "configs/v11/方案N9_v11rxc近N6但wrapper不注入macro6评估探针.json"

ssh v1002 "mkdir -p '$RUN_DIR'"

if ! ssh v1002 "test -f '$RESUME_MODEL'"; then
  echo "missing_required path=$RESUME_MODEL" >&2
  exit 1
fi

ssh v1002 "cd '$REMOTE_ROOT' && source ~/miniconda3/etc/profile.d/conda.sh && conda activate ML && python train_target8_v11rxc_bad_seed_curriculum.py \
  --output-dir '$RUN_DIR' \
  --resume-model '$RESUME_MODEL' \
  --stage-episodes '$EPISODES' \
  --seed-pool '$SEED_POOL' \
  --env-overrides-file '$TRAIN_OVERRIDES' \
  --video-episodes 0 \
  --device cpu \
  --torch-threads 1"

MODEL_BASE="$RUN_DIR/rppo_medium_lstm128x2_observable12_target8_macro_library_v11rxc_stage2/rppo_medium_lstm128x2_observable12_target8_macro_library_v11rxc_stage2"
MODEL_ZIP="${MODEL_BASE}.zip"

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
