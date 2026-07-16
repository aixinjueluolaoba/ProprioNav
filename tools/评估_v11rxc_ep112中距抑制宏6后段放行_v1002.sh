#!/usr/bin/env bash
set -euo pipefail

LOCAL_ROOT=/home/diana/fishing/blind_nav_rl
REMOTE_ROOT=/home/diana/fishing/blind_nav_rl
MODEL_PATH=${MODEL_PATH:-$REMOTE_ROOT/runs/v11rxc_nlite125_fix6300031_targeted_far6_stuck10_search112/rppo_medium_lstm128x2_observable12_target8_macro_library_v11rxc_stage2/models/checkpoint_ep_00112.zip}
STATE_MODE=${STATE_MODE:-observable12_target8_macro_library_v11rxc_stage2}
EVAL_OVERRIDES=${EVAL_OVERRIDES:-$REMOTE_ROOT/configs/v11/方案N22_v11rxc中距抑制宏6后段放行评估探针.json}
OUT_PREFIX=${OUT_PREFIX:-$REMOTE_ROOT/runs/v11rxc_ep112_mid_macro6_late_passthrough_eval}

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
copy_file "configs/v11/方案N22_v11rxc中距抑制宏6后段放行评估探针.json"

MODEL_BASE="${MODEL_PATH%.zip}"

ssh v1002 "cd '$REMOTE_ROOT' && source ~/miniconda3/etc/profile.d/conda.sh && conda activate ML && python eval_target8_bad_seeds.py \
  --model '$MODEL_BASE' \
  --out-dir '${OUT_PREFIX}_badseed' \
  --state-mode '$STATE_MODE' \
  --env-overrides-file '$EVAL_OVERRIDES' \
  --device cpu \
  --seed-list 6300031,6300058,6300069,6300006"

ssh v1002 "cd '$REMOTE_ROOT' && source ~/miniconda3/etc/profile.d/conda.sh && conda activate ML && python eval_target8_pressure.py \
  --model '$MODEL_BASE' \
  --out-dir '${OUT_PREFIX}_pressure20' \
  --action-mode v11 \
  --state-mode '$STATE_MODE' \
  --episodes 20 \
  --video-episodes 0 \
  --env-overrides-file '$EVAL_OVERRIDES' \
  --device cpu"

ssh v1002 "cd '$REMOTE_ROOT' && source ~/miniconda3/etc/profile.d/conda.sh && conda activate ML && python eval_target8_sweep.py \
  --train-output-dir '${MODEL_PATH%/models/*}' \
  --model-path '$MODEL_PATH' \
  --out-dir '${OUT_PREFIX}_normal20' \
  --experiments rppo_medium_lstm128x2_observable12_target8_macro_library_v11rxc_stage2 \
  --episodes 20 \
  --video-episodes 0 \
  --env-overrides-file '$EVAL_OVERRIDES' \
  --device cpu"

echo "out_prefix=$OUT_PREFIX"
