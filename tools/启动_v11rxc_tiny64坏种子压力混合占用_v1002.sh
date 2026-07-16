#!/usr/bin/env bash
set -euo pipefail

LOCAL_ROOT=/home/diana/fishing/blind_nav_rl
REMOTE_ROOT=/home/diana/fishing/blind_nav_rl
RUN_NAME=${RUN_NAME:-v11rxc_tiny64_mix50_from_40rows_w14875_tiny}
RUN_DIR=${RUN_DIR:-$REMOTE_ROOT/runs/$RUN_NAME}
RESUME_MODEL=${RESUME_MODEL:-$REMOTE_ROOT/runs/v11rxc_tiny64_n24_disagree_reg_40rows_w14875_pressure3/rppo_medium_lstm128x2_observable12_target8_macro_library_v11rxc_stage2/rppo_medium_lstm128x2_observable12_target8_macro_library_v11rxc_stage2.zip}
TRAIN_OVERRIDES_REMOTE=${TRAIN_OVERRIDES_REMOTE:-$REMOTE_ROOT/configs/v11/方案N9_v11rxc近N6但wrapper不注入macro6评估探针.json}
EVAL_OVERRIDES_REMOTE=${EVAL_OVERRIDES_REMOTE:-$REMOTE_ROOT/configs/v11/方案N9_v11rxc近N6但wrapper不注入macro6评估探针.json}
BAD_SEED_POOL=${BAD_SEED_POOL:-6300031,6300058,6300069,6300006}
PRESSURE_SEED_POOL=${PRESSURE_SEED_POOL:-5200015,5200015,5200015,5200015,5200008,5200022}
BAD_SEED_RATIO=${BAD_SEED_RATIO:-0.5}
N_ENVS=${N_ENVS:-24}
TRAIN_EXTRA_ARGS=${TRAIN_EXTRA_ARGS:---tiny}
LEARNING_RATE=${LEARNING_RATE:-2e-5}
TRAINABLE_PARAM_MODE=${TRAINABLE_PARAM_MODE:-all}
SPECIALIST_DATASET=${SPECIALIST_DATASET:-}
SPECIALIST_LOSS_WEIGHT=${SPECIALIST_LOSS_WEIGHT:-0}
SPECIALIST_BATCH_SIZE=${SPECIALIST_BATCH_SIZE:-128}
SPECIALIST_LOSS_HEADS=${SPECIALIST_LOSS_HEADS:-macro_only}
SPECIALIST_MAX_SAMPLES=${SPECIALIST_MAX_SAMPLES:-}
SPECIALIST_DISAGREEMENT_FILTER=${SPECIALIST_DISAGREEMENT_FILTER:-macro}
SPECIALIST_SEED_WHITELIST=${SPECIALIST_SEED_WHITELIST:-}
SPECIALIST_SEED_MODE_WHITELIST=${SPECIALIST_SEED_MODE_WHITELIST:-}
SPECIALIST_MIN_STUCK_TIME=${SPECIALIST_MIN_STUCK_TIME:-}
SPECIALIST_MIN_NO_PROGRESS_TIME=${SPECIALIST_MIN_NO_PROGRESS_TIME:-}
SPECIALIST_RECOVERY_MODES=${SPECIALIST_RECOVERY_MODES:-}
SPECIALIST_TRIGGER_MIN_PRESSURE_ACTIVE_FRAC=${SPECIALIST_TRIGGER_MIN_PRESSURE_ACTIVE_FRAC:-}
SPECIALIST_TRIGGER_SCALE_BY_PRESSURE_ACTIVE_FRAC=${SPECIALIST_TRIGGER_SCALE_BY_PRESSURE_ACTIVE_FRAC:-0}
SPECIALIST_TRIGGER_MIN_NO_PROGRESS_TIME=${SPECIALIST_TRIGGER_MIN_NO_PROGRESS_TIME:-}
SPECIALIST_TRIGGER_MIN_STUCK_TIME=${SPECIALIST_TRIGGER_MIN_STUCK_TIME:-}
SPECIALIST_TRIGGER_RECOVERY_MODES=${SPECIALIST_TRIGGER_RECOVERY_MODES:-}
SPECIALIST_TRIGGER_MIN_RECENT_COLLISION_NORM=${SPECIALIST_TRIGGER_MIN_RECENT_COLLISION_NORM:-}
SPECIALIST_TRIGGER_REQUIRE_STUCK_MACRO_CONTEXT=${SPECIALIST_TRIGGER_REQUIRE_STUCK_MACRO_CONTEXT:-0}
SPECIALIST_TRIGGER_REQUIRE_TARGETED_STUCK_MACRO_CONTEXT=${SPECIALIST_TRIGGER_REQUIRE_TARGETED_STUCK_MACRO_CONTEXT:-0}

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
copy_file train_target8_v11re_bad_seed_mergeback_curriculum.py
copy_file train_target8_v11re_reward_exit_curriculum.py
copy_file train_target8_v11rxc_bad_seed_curriculum.py
copy_file train_target8_v11rxc_badseed_pressure_mix_curriculum.py
copy_file train_target8_v11rxc_disagreement_regularized_bad_seed.py
copy_file "configs/v11/方案N9_v11rxc近N6但wrapper不注入macro6评估探针.json"

ssh v1002 "mkdir -p '$RUN_DIR'"

if ! ssh v1002 "test -f '$RESUME_MODEL'"; then
  echo "missing_required path=$RESUME_MODEL" >&2
  exit 1
fi

ssh v1002 "cd '$REMOTE_ROOT' && source ~/miniconda3/etc/profile.d/conda.sh && conda activate ML && python train_target8_v11rxc_badseed_pressure_mix_curriculum.py \
  $TRAIN_EXTRA_ARGS \
  --output-dir '$RUN_DIR' \
  --resume-model '$RESUME_MODEL' \
  --env-overrides-file '$TRAIN_OVERRIDES_REMOTE' \
  --bad-seed-pool '$BAD_SEED_POOL' \
  --pressure-seed-pool '$PRESSURE_SEED_POOL' \
  --bad-seed-ratio $BAD_SEED_RATIO \
  --n-envs $N_ENVS \
  --learning-rate $LEARNING_RATE \
  --trainable-param-mode '$TRAINABLE_PARAM_MODE' \
  ${SPECIALIST_DATASET:+--specialist-dataset "$SPECIALIST_DATASET"} \
  --specialist-loss-weight $SPECIALIST_LOSS_WEIGHT \
  --specialist-batch-size $SPECIALIST_BATCH_SIZE \
  --specialist-loss-heads '$SPECIALIST_LOSS_HEADS' \
  ${SPECIALIST_MAX_SAMPLES:+--specialist-max-samples "$SPECIALIST_MAX_SAMPLES"} \
  --specialist-disagreement-filter '$SPECIALIST_DISAGREEMENT_FILTER' \
  ${SPECIALIST_SEED_WHITELIST:+--specialist-seed-whitelist "$SPECIALIST_SEED_WHITELIST"} \
  ${SPECIALIST_SEED_MODE_WHITELIST:+--specialist-seed-mode-whitelist "$SPECIALIST_SEED_MODE_WHITELIST"} \
  ${SPECIALIST_MIN_STUCK_TIME:+--specialist-min-stuck-time "$SPECIALIST_MIN_STUCK_TIME"} \
  ${SPECIALIST_MIN_NO_PROGRESS_TIME:+--specialist-min-no-progress-time "$SPECIALIST_MIN_NO_PROGRESS_TIME"} \
  ${SPECIALIST_RECOVERY_MODES:+--specialist-recovery-modes "$SPECIALIST_RECOVERY_MODES"} \
  ${SPECIALIST_TRIGGER_MIN_PRESSURE_ACTIVE_FRAC:+--specialist-trigger-min-pressure-active-frac "$SPECIALIST_TRIGGER_MIN_PRESSURE_ACTIVE_FRAC"} \
  $(if [[ "$SPECIALIST_TRIGGER_SCALE_BY_PRESSURE_ACTIVE_FRAC" == "1" ]]; then printf %s "--specialist-trigger-scale-by-pressure-active-frac"; fi) \
  ${SPECIALIST_TRIGGER_MIN_NO_PROGRESS_TIME:+--specialist-trigger-min-no-progress-time "$SPECIALIST_TRIGGER_MIN_NO_PROGRESS_TIME"} \
  ${SPECIALIST_TRIGGER_MIN_STUCK_TIME:+--specialist-trigger-min-stuck-time "$SPECIALIST_TRIGGER_MIN_STUCK_TIME"} \
  ${SPECIALIST_TRIGGER_RECOVERY_MODES:+--specialist-trigger-recovery-modes "$SPECIALIST_TRIGGER_RECOVERY_MODES"} \
  ${SPECIALIST_TRIGGER_MIN_RECENT_COLLISION_NORM:+--specialist-trigger-min-recent-collision-norm "$SPECIALIST_TRIGGER_MIN_RECENT_COLLISION_NORM"} \
  $(if [[ "$SPECIALIST_TRIGGER_REQUIRE_STUCK_MACRO_CONTEXT" == "1" ]]; then printf %s "--specialist-trigger-require-stuck-macro-context"; fi) \
  $(if [[ "$SPECIALIST_TRIGGER_REQUIRE_TARGETED_STUCK_MACRO_CONTEXT" == "1" ]]; then printf %s "--specialist-trigger-require-targeted-stuck-macro-context"; fi) \
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
