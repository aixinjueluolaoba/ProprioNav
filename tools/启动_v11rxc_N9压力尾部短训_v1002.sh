#!/usr/bin/env bash
set -euo pipefail

LOCAL_ROOT=/home/diana/fishing/blind_nav_rl
REMOTE_ROOT=/home/diana/fishing/blind_nav_rl
RUN_NAME=${RUN_NAME:-v11rxc_n9_pressure_tail_tiny}
RUN_DIR=${RUN_DIR:-$REMOTE_ROOT/runs/$RUN_NAME}
RESUME_MODEL=${RESUME_MODEL:-$REMOTE_ROOT/runs/v11rxc_nlite125_fix6300031_targeted_far6_stuck10_search112/rppo_medium_lstm128x2_observable12_target8_macro_library_v11rxc_stage2/models/checkpoint_ep_00112.zip}
OVERRIDES_REMOTE=${OVERRIDES_REMOTE:-$REMOTE_ROOT/configs/v11/方案N9_v11rxc近N6但wrapper不注入macro6评估探针.json}
TRAIN_EXTRA_ARGS=${TRAIN_EXTRA_ARGS:---tiny}
SEED_POOL=${SEED_POOL:-5200015,5200015,5200015,5200015,5200008,5200022}

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
copy_file train_target8_v11rxc_pressure_seed_curriculum.py
copy_file "configs/v11/方案N9_v11rxc近N6但wrapper不注入macro6评估探针.json"

ssh v1002 "mkdir -p '$RUN_DIR'"

if ! ssh v1002 "test -f '$RESUME_MODEL'"; then
  echo "missing_required path=$RESUME_MODEL" >&2
  exit 1
fi

ssh v1002 "cd '$REMOTE_ROOT' && source ~/miniconda3/etc/profile.d/conda.sh && conda activate ML && python train_target8_v11rxc_pressure_seed_curriculum.py \
  $TRAIN_EXTRA_ARGS \
  --output-dir '$RUN_DIR' \
  --resume-model '$RESUME_MODEL' \
  --seed-pool '$SEED_POOL' \
  --video-episodes 0 \
  --device cpu \
  --torch-threads 1 \
  --env-overrides-file '$OVERRIDES_REMOTE'"

MODEL_BASE="$RUN_DIR/rppo_medium_lstm128x2_observable12_target8_macro_library_v11rxc_stage2/rppo_medium_lstm128x2_observable12_target8_macro_library_v11rxc_stage2"
MODEL_ZIP="${MODEL_BASE}.zip"

ssh v1002 "cd '$REMOTE_ROOT' && source ~/miniconda3/etc/profile.d/conda.sh && conda activate ML && python eval_target8_bad_seeds.py \
  --model '$MODEL_BASE' \
  --out-dir '${RUN_DIR}_badseed_gate' \
  --state-mode observable12_target8_macro_library_v11rxc_stage2 \
  --env-overrides-file '$OVERRIDES_REMOTE' \
  --device cpu \
  --seed-list 6300031,6300058,6300069,6300006"

ssh v1002 "cd '$REMOTE_ROOT' && source ~/miniconda3/etc/profile.d/conda.sh && conda activate ML && python eval_target8_pressure.py \
  --model '$MODEL_BASE' \
  --out-dir '${RUN_DIR}_pressure20' \
  --action-mode v11 \
  --state-mode observable12_target8_macro_library_v11rxc_stage2 \
  --episodes 20 \
  --video-episodes 0 \
  --env-overrides-file '$OVERRIDES_REMOTE' \
  --device cpu"

ssh v1002 "cd '$REMOTE_ROOT' && source ~/miniconda3/etc/profile.d/conda.sh && conda activate ML && python eval_target8_sweep.py \
  --train-output-dir '$RUN_DIR' \
  --model-path '$MODEL_ZIP' \
  --out-dir '${RUN_DIR}_normal20' \
  --experiments rppo_medium_lstm128x2_observable12_target8_macro_library_v11rxc_stage2 \
  --episodes 20 \
  --video-episodes 0 \
  --env-overrides-file '$OVERRIDES_REMOTE' \
  --device cpu"

echo "run_dir=$RUN_DIR"
