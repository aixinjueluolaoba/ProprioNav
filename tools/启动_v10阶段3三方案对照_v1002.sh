#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/diana/fishing/blind_nav_rl}
export RUN_GROUP_NAME=${RUN_GROUP_NAME:-目标8显式脱困v10阶段3三方案对照_v1002}
export RUN_COUNT=3
export N_ENVS=${N_ENVS:-96}
export STAGE3_EPISODES=${STAGE3_EPISODES:-1500}
export CHECKPOINT_INTERVAL=${CHECKPOINT_INTERVAL:-500}
export PROGRESS_INTERVAL=${PROGRESS_INTERVAL:-250}
export WATCH_INTERVAL=${WATCH_INTERVAL:-60}
export VARIANT_NAMES='方案A收紧gate|方案C降trapped|方案B降深凹'
export ENV_OVERRIDES_FILES="${ROOT}/configs/v10_stage3/方案A_收紧gate.json|${ROOT}/configs/v10_stage3/方案C_降低trapped与反向朝向.json|${ROOT}/configs/v10_stage3/方案B_降低深凹与陷阱密度.json"

cd "$ROOT"
bash tools/启动_v10阶段3微调对照_v1002.sh
