#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/diana/fishing/blind_nav_rl}
RUN_GROUP_NAME=${RUN_GROUP_NAME:-目标8显式脱困v10阶段3微调对照_v1002}
BATCH_ROOT=${BATCH_ROOT:-$ROOT/runs/$RUN_GROUP_NAME}
RESUME_MODEL=${RESUME_MODEL:-$ROOT/runs/目标8显式脱困v10四阶段_v1002_110k/rppo_medium_lstm128x2_observable12_target8_discrete_recovery_v10_stage2/models/checkpoint_ep_03000.zip}
START_STAGE=${START_STAGE:-rppo_medium_lstm128x2_observable12_target8_discrete_recovery_v10_stage3}
END_STAGE=${END_STAGE:-rppo_medium_lstm128x2_observable12_target8_discrete_recovery_v10_stage3}
RUN_COUNT=${RUN_COUNT:-3}
VARIANT_NAMES=${VARIANT_NAMES:-}
ENV_OVERRIDES_FILES=${ENV_OVERRIDES_FILES:-}
SEED_BASE_START=${SEED_BASE_START:-9815000}
SEED_STRIDE=${SEED_STRIDE:-200000}
EVAL_SEED_OFFSET=${EVAL_SEED_OFFSET:-100000}
STAGE3_EPISODES=${STAGE3_EPISODES:-1500}
CHECKPOINT_INTERVAL=${CHECKPOINT_INTERVAL:-500}
PROGRESS_INTERVAL=${PROGRESS_INTERVAL:-250}
WATCH_INTERVAL=${WATCH_INTERVAL:-60}
N_ENVS=${N_ENVS:-256}
THREADS=${THREADS:-1}
DEVICE=${DEVICE:-cpu}
EVAL_EPISODES=${EVAL_EPISODES:-20}
VIDEO_EPISODES=${VIDEO_EPISODES:-0}
CONDA_SH=${CONDA_SH:-/home/diana/miniconda3/etc/profile.d/conda.sh}
CONDA_ENV=${CONDA_ENV:-ML}
MANIFEST_PATH=${MANIFEST_PATH:-$BATCH_ROOT/批次参数.csv}

if test "$RUN_COUNT" -lt 1; then
  echo "invalid_run_count value=$RUN_COUNT" >&2
  exit 1
fi

if test ! -f "$RESUME_MODEL"; then
  echo "resume_model_not_found path=$RESUME_MODEL" >&2
  exit 1
fi

cd "$ROOT"
mkdir -p "$BATCH_ROOT"

declare -a variant_names=()
declare -a env_override_files=()
if test -n "$VARIANT_NAMES"; then
  IFS='|' read -r -a variant_names <<< "$VARIANT_NAMES"
fi
if test -n "$ENV_OVERRIDES_FILES"; then
  IFS='|' read -r -a env_override_files <<< "$ENV_OVERRIDES_FILES"
fi
if test "${#variant_names[@]}" -gt 0 && test "${#variant_names[@]}" -ne "$RUN_COUNT"; then
  echo "variant_name_count_mismatch expected=$RUN_COUNT actual=${#variant_names[@]}" >&2
  exit 1
fi
if test "${#env_override_files[@]}" -gt 0 && test "${#env_override_files[@]}" -ne "$RUN_COUNT"; then
  echo "env_override_count_mismatch expected=$RUN_COUNT actual=${#env_override_files[@]}" >&2
  exit 1
fi

printf '%s\n' \
  'run_id,variant_name,env_overrides_file,run_name,train_dir,watch_dir,seed_base,eval_seed_base,stage3_episodes,checkpoint_interval,resume_model' \
  > "$MANIFEST_PATH"

for ((index = 1; index <= RUN_COUNT; index++)); do
  run_id=$(printf '%02d' "$index")
  seed_base=$((SEED_BASE_START + (index - 1) * SEED_STRIDE))
  eval_seed_base=$((seed_base + EVAL_SEED_OFFSET))
  variant_name=""
  env_overrides_file=""
  if test "${#variant_names[@]}" -gt 0; then
    variant_name="${variant_names[$((index - 1))]}"
  fi
  if test "${#env_override_files[@]}" -gt 0; then
    env_overrides_file="${env_override_files[$((index - 1))]}"
  fi
  if test -n "$env_overrides_file" && test ! -f "$env_overrides_file"; then
    echo "env_overrides_not_found run=$run_id path=$env_overrides_file" >&2
    exit 1
  fi
  run_label="seed${seed_base}_ep${STAGE3_EPISODES}"
  if test -n "$variant_name"; then
    run_name="${run_id}_${variant_name}_${run_label}"
  else
    run_name="${run_id}_${run_label}"
  fi
  train_dir="$BATCH_ROOT/$run_name"
  watch_dir="$BATCH_ROOT/${run_name}_checkpoint评估"

  mkdir -p "$train_dir" "$watch_dir"

  printf '%s\n' \
    "$run_id,$variant_name,$env_overrides_file,$run_name,$train_dir,$watch_dir,$seed_base,$eval_seed_base,$STAGE3_EPISODES,$CHECKPOINT_INTERVAL,$RESUME_MODEL" \
    >> "$MANIFEST_PATH"

  if test -f "$train_dir/训练.pid" && ps -p "$(cat "$train_dir/训练.pid")" >/dev/null 2>&1; then
    echo "train_already_running run=$run_name pid=$(cat "$train_dir/训练.pid")"
  else
    train_cmd="
      cd $ROOT &&
      source $CONDA_SH &&
      conda activate $CONDA_ENV &&
      ROOT=$ROOT \
      RUN_DIR=$train_dir \
      RESUME_MODEL=$RESUME_MODEL \
      START_STAGE=$START_STAGE \
      END_STAGE=$END_STAGE \
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
    setsid bash -lc "$train_cmd" > "$train_dir/阶段3微调训练.log" 2>&1 &
    echo $! > "$train_dir/训练.pid"
    echo "train_started run=$run_name pid=$(cat "$train_dir/训练.pid")"
  fi

  if test -f "$watch_dir/watcher.pid" && ps -p "$(cat "$watch_dir/watcher.pid")" >/dev/null 2>&1; then
    echo "watcher_already_running run=$run_name pid=$(cat "$watch_dir/watcher.pid")"
  else
    watch_cmd="
      cd $ROOT &&
      source $CONDA_SH &&
      conda activate $CONDA_ENV &&
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
