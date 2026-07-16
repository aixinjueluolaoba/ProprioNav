#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/weiaokang/fishing/blind_nav_rl}
RUN_GROUP_NAME=${RUN_GROUP_NAME:-目标8显式脱困v10阶段3方案K收角微调对照_253}
BATCH_ROOT=${BATCH_ROOT:-$ROOT/runs/$RUN_GROUP_NAME}
INIT_POLICY_STATE_DICT=${INIT_POLICY_STATE_DICT:-$ROOT/runs/目标8显式脱困v10阶段3方案A保留线_253_阶段2起点/stage2_ckpt3000_policy_state_dict.pt}
RUN_COUNT=${RUN_COUNT:-3}
VARIANT_NAMES=${VARIANT_NAMES:-方案L_K轻收角|方案M_K保脱困收角|方案N_K轻压收角}
ENV_OVERRIDES_FILES=${ENV_OVERRIDES_FILES:-$ROOT/configs/v10_stage3/方案L_K轻收角.json|$ROOT/configs/v10_stage3/方案M_K保脱困收角.json|$ROOT/configs/v10_stage3/方案N_K轻压收角.json}
SEED_BASE_START=${SEED_BASE_START:-12755000}
SEED_STRIDE=${SEED_STRIDE:-200000}
EVAL_SEED_OFFSET=${EVAL_SEED_OFFSET:-100000}
STAGE3_EPISODES=${STAGE3_EPISODES:-250}
CHECKPOINT_INTERVAL=${CHECKPOINT_INTERVAL:-250}
PROGRESS_INTERVAL=${PROGRESS_INTERVAL:-125}
WATCH_INTERVAL=${WATCH_INTERVAL:-45}
N_ENVS=${N_ENVS:-64}
THREADS=${THREADS:-1}
DEVICE=${DEVICE:-cpu}
EVAL_EPISODES=${EVAL_EPISODES:-20}
VIDEO_EPISODES=${VIDEO_EPISODES:-0}
MANIFEST_PATH=${MANIFEST_PATH:-$BATCH_ROOT/批次参数.csv}

if test ! -f "$INIT_POLICY_STATE_DICT"; then
  echo "init_policy_state_dict_not_found path=$INIT_POLICY_STATE_DICT" >&2
  exit 1
fi

cd "$ROOT"
mkdir -p "$BATCH_ROOT"

IFS='|' read -r -a variant_names <<< "$VARIANT_NAMES"
IFS='|' read -r -a env_override_files <<< "$ENV_OVERRIDES_FILES"

printf '%s\n' \
  'run_id,variant_name,env_overrides_file,run_name,train_dir,watch_dir,seed_base,eval_seed_base,stage3_episodes,checkpoint_interval,init_policy_state_dict' \
  > "$MANIFEST_PATH"

for ((index = 1; index <= RUN_COUNT; index++)); do
  run_id=$(printf '%02d' "$index")
  seed_base=$((SEED_BASE_START + (index - 1) * SEED_STRIDE))
  eval_seed_base=$((seed_base + EVAL_SEED_OFFSET))
  variant_name="${variant_names[$((index - 1))]}"
  env_overrides_file="${env_override_files[$((index - 1))]}"
  run_label="seed${seed_base}_ep${STAGE3_EPISODES}"
  run_name="${run_id}_${variant_name}_${run_label}"
  train_dir="$BATCH_ROOT/$run_name"
  watch_dir="$BATCH_ROOT/${run_name}_checkpoint评估"

  mkdir -p "$train_dir" "$watch_dir"
  printf '%s\n' \
    "$run_id,$variant_name,$env_overrides_file,$run_name,$train_dir,$watch_dir,$seed_base,$eval_seed_base,$STAGE3_EPISODES,$CHECKPOINT_INTERVAL,$INIT_POLICY_STATE_DICT" \
    >> "$MANIFEST_PATH"

  if test -f "$train_dir/训练.pid" && ps -p "$(cat "$train_dir/训练.pid")" >/dev/null 2>&1; then
    echo "train_already_running run=$run_name pid=$(cat "$train_dir/训练.pid")"
  else
    train_cmd="
      cd $ROOT &&
      ulimit -n 4096 &&
      source /home/weiaokang/ML/bin/activate &&
      ROOT=$ROOT \
      RUN_DIR=$train_dir \
      INIT_POLICY_STATE_DICT=$INIT_POLICY_STATE_DICT \
      START_STAGE=rppo_medium_lstm128x2_observable12_target8_discrete_recovery_v10_stage3 \
      END_STAGE=rppo_medium_lstm128x2_observable12_target8_discrete_recovery_v10_stage3 \
      STAGE3_EPISODES=$STAGE3_EPISODES \
      N_ENVS=$N_ENVS \
      THREADS=$THREADS \
      DEVICE=$DEVICE \
      EVAL_EPISODES=$EVAL_EPISODES \
      VIDEO_EPISODES=$VIDEO_EPISODES \
      SEED_BASE=$seed_base \
      EVAL_SEED_BASE=$eval_seed_base \
      CHECKPOINT_INTERVAL=$CHECKPOINT_INTERVAL \
      PROGRESS_INTERVAL=$PROGRESS_INTERVAL \
      ENV_OVERRIDES_FILE=$env_overrides_file \
      MPLCONFIGDIR=$train_dir/图形缓存 \
      bash launch_target8_v10_stage2_resume.sh
    "
    setsid bash -lc "$train_cmd" > "$train_dir/阶段3训练.log" 2>&1 &
    echo $! > "$train_dir/训练.pid"
    echo "train_started run=$run_name pid=$(cat "$train_dir/训练.pid")"
  fi

  if test -f "$watch_dir/watcher.pid" && ps -p "$(cat "$watch_dir/watcher.pid")" >/dev/null 2>&1; then
    echo "watcher_already_running run=$run_name pid=$(cat "$watch_dir/watcher.pid")"
  else
    watch_cmd="
      cd $ROOT &&
      source /home/weiaokang/ML/bin/activate &&
      python watch_v10_checkpoints.py \
        --run-dir $train_dir \
        --out-dir $watch_dir \
        --curriculum \
        --interval $WATCH_INTERVAL \
        --checkpoint-step $CHECKPOINT_INTERVAL \
        --max-checkpoint $STAGE3_EPISODES
    "
    setsid bash -lc "$watch_cmd" > "$watch_dir/watcher.log" 2>&1 &
    echo $! > "$watch_dir/watcher.pid"
    echo "watcher_started run=$run_name pid=$(cat "$watch_dir/watcher.pid")"
  fi

  echo "train_dir[$run_id]=$train_dir"
  echo "watch_dir[$run_id]=$watch_dir"
done

echo "batch_root=$BATCH_ROOT"
echo "manifest=$MANIFEST_PATH"
