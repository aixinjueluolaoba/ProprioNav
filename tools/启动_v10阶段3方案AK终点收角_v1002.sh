#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/diana/fishing/blind_nav_rl_reward_code_20260521}
export ROOT
export RUN_GROUP_NAME=${RUN_GROUP_NAME:-目标8显式脱困v10阶段3方案AK终点收角_v1002}
export BATCH_ROOT=${BATCH_ROOT:-/home/diana/fishing/blind_nav_rl/runs/$RUN_GROUP_NAME}
export RESUME_MODEL=${RESUME_MODEL:-/home/diana/fishing/blind_nav_rl/runs/目标8显式脱困v10阶段3超轻压角_codecopy_v1002/01_方案AE_超轻非recovery压角_seed23155000_ep1000/rppo_medium_lstm128x2_observable12_target8_discrete_recovery_v10_stage3/models/checkpoint_ep_00500.zip}
export RUN_COUNT=${RUN_COUNT:-1}
export VARIANT_NAMES=${VARIANT_NAMES:-方案AK_AE500终点收角保脱困}
export ENV_OVERRIDES_FILES=${ENV_OVERRIDES_FILES:-$ROOT/configs/v10_stage3/方案AK_A终点收角保脱困.json}
export SEED_BASE_START=${SEED_BASE_START:-26155000}
export STAGE3_EPISODES=${STAGE3_EPISODES:-500}
export CHECKPOINT_INTERVAL=${CHECKPOINT_INTERVAL:-250}
export PROGRESS_INTERVAL=${PROGRESS_INTERVAL:-125}
export N_ENVS=${N_ENVS:-96}
export EVAL_EPISODES=${EVAL_EPISODES:-20}
export VIDEO_EPISODES=${VIDEO_EPISODES:-0}
export WATCH_INTERVAL=${WATCH_INTERVAL:-20}

bash "$ROOT/tools/启动_v10阶段3微调对照_v1002.sh"
