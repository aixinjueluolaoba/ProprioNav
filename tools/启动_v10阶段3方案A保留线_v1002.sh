#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/diana/fishing/blind_nav_rl}
export RUN_GROUP_NAME=${RUN_GROUP_NAME:-目标8显式脱困v10阶段3方案A保留线_v1002}
export RUN_COUNT=1
export N_ENVS=${N_ENVS:-96}
export STAGE3_EPISODES=${STAGE3_EPISODES:-1500}
export CHECKPOINT_INTERVAL=${CHECKPOINT_INTERVAL:-500}
export PROGRESS_INTERVAL=${PROGRESS_INTERVAL:-250}
export WATCH_INTERVAL=${WATCH_INTERVAL:-60}
export SEED_BASE_START=${SEED_BASE_START:-9815000}
export VARIANT_NAMES='方案A收紧gate'
export ENV_OVERRIDES_FILES="${ROOT}/configs/v10_stage3/方案A_收紧gate.json"

cd "$ROOT"
bash tools/启动_v10阶段3微调对照_v1002.sh
