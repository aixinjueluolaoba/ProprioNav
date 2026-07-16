#!/usr/bin/env bash
set -euo pipefail

LOCAL_ROOT=/home/diana/fishing/blind_nav_rl
REMOTE_ROOT=/home/diana/fishing/blind_nav_rl
OUT_DIR=${OUT_DIR:-$REMOTE_ROOT/runs/v11rxc_tail_trace_compare}
STATE_MODE=${STATE_MODE:-observable12_target8_macro_library_v11rxc_stage2}
ENV_OVERRIDES=${ENV_OVERRIDES:-$REMOTE_ROOT/configs/v11/方案Nlite_v11rxc尾部轻保守探针.json}
SEEDS=${SEEDS:-6300058 6300006}

MODEL_NLITE125=${MODEL_NLITE125:-$REMOTE_ROOT/runs/v11rx_tail2_conservative_lite_125/rppo_medium_lstm128x2_observable12_target8_macro_library_v11rx_stage2/models/checkpoint_ep_00125.zip}
MODEL_EP112=${MODEL_EP112:-$REMOTE_ROOT/runs/v11rxc_nlite125_fix6300031_targeted_far6_stuck10_search112/rppo_medium_lstm128x2_observable12_target8_macro_library_v11rxc_stage2/models/checkpoint_ep_00112.zip}
MODEL_N2=${MODEL_N2:-$REMOTE_ROOT/runs/v11rxc_two_stage_6300031push112_tail2_N2stabilize32/rppo_medium_lstm128x2_observable12_target8_macro_library_v11rxc_stage2/rppo_medium_lstm128x2_observable12_target8_macro_library_v11rxc_stage2.zip}

copy_file() {
  local rel="$1"
  ssh v1002 "mkdir -p '$REMOTE_ROOT/$(dirname "$rel")'"
  scp "$LOCAL_ROOT/$rel" "v1002:$REMOTE_ROOT/$rel"
}

copy_file benchmark_state_dims_10k.py
copy_file blind_nav_rl/env.py
copy_file eval_target8_bad_seeds.py
copy_file tools/debug_v11rx_single_seed_trace.py
copy_file "configs/v11/方案Nlite_v11rxc尾部轻保守探针.json"

ssh v1002 "mkdir -p '$OUT_DIR'"

run_one() {
  local label="$1"
  local model="$2"
  for seed in $SEEDS; do
    ssh v1002 "cd '$REMOTE_ROOT' && source ~/miniconda3/etc/profile.d/conda.sh && conda activate ML && python tools/debug_v11rx_single_seed_trace.py \
      --model '$model' \
      --seed '$seed' \
      --state-mode '$STATE_MODE' \
      --env-overrides-file '$ENV_OVERRIDES' \
      --out '$OUT_DIR/${label}_seed${seed}_trace.csv' \
      --summary-out '$OUT_DIR/${label}_seed${seed}_summary.csv' \
      --device cpu"
  done
}

run_one nlite125 "$MODEL_NLITE125"
run_one ep112 "$MODEL_EP112"
run_one n2final "$MODEL_N2"

echo "out_dir=$OUT_DIR"
