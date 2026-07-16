#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/diana/fishing/blind_nav_rl}
export RUN_GROUP_NAME=${RUN_GROUP_NAME:-目标8显式脱困v10阶段3方案VWX角度对照_v1002}
export RUN_COUNT=3
export VARIANT_NAMES='方案V_A收紧gate轻压角|方案W_A收紧gate45度收角|方案X_A收紧gate45度轻压角'
export ENV_OVERRIDES_FILES="${ROOT}/configs/v10_stage3/方案V_A收紧gate轻压角.json|${ROOT}/configs/v10_stage3/方案W_A收紧gate45度收角.json|${ROOT}/configs/v10_stage3/方案X_A收紧gate45度轻压角.json"
export SEED_BASE_START=${SEED_BASE_START:-17155000}
export STAGE3_EPISODES=${STAGE3_EPISODES:-500}
export CHECKPOINT_INTERVAL=${CHECKPOINT_INTERVAL:-250}
export PROGRESS_INTERVAL=${PROGRESS_INTERVAL:-125}
export N_ENVS=${N_ENVS:-96}
export THREADS=${THREADS:-1}
export EVAL_EPISODES=${EVAL_EPISODES:-20}
export VIDEO_EPISODES=${VIDEO_EPISODES:-0}

cd "$ROOT"
bash tools/启动_v10阶段3微调对照_v1002.sh
