#!/usr/bin/env bash
set -euo pipefail

REMOTE_ROOT=/home/diana/fishing/blind_nav_rl
RUN_NAME=${RUN_NAME:-v11rx_tail2_conservative_lite_125}
RUN_DIR=${RUN_DIR:-$REMOTE_ROOT/runs/$RUN_NAME}
MODEL_PATH=${MODEL_PATH:-$RUN_DIR/rppo_medium_lstm128x2_observable12_target8_macro_library_v11rx_stage2/models/checkpoint_ep_00125.zip}
OVERRIDES_REMOTE=${OVERRIDES_REMOTE:-$REMOTE_ROOT/configs/v11/方案Nlite_v11rx尾部轻保守探针.json}

ssh v1002 "cd '$REMOTE_ROOT' && source ~/miniconda3/etc/profile.d/conda.sh && conda activate ML && \
  python eval_target8_bad_seeds.py \
    --model '$MODEL_PATH' \
    --out-dir '${RUN_DIR}_badseed_00125' \
    --state-mode observable12_target8_macro_library_v11rx_stage2 \
    --env-overrides-file '$OVERRIDES_REMOTE' \
    --device cpu"

ssh v1002 "cd '$REMOTE_ROOT' && source ~/miniconda3/etc/profile.d/conda.sh && conda activate ML && \
  python eval_target8_pressure.py \
    --model '$MODEL_PATH' \
    --out-dir '${RUN_DIR}_pressure20_00125' \
    --action-mode v11 \
    --state-mode observable12_target8_macro_library_v11rx_stage2 \
    --episodes 20 \
    --video-episodes 0 \
    --env-overrides-file '$OVERRIDES_REMOTE' \
    --device cpu"

ssh v1002 "cd '$REMOTE_ROOT' && source ~/miniconda3/etc/profile.d/conda.sh && conda activate ML && \
  python eval_target8_sweep.py \
    --train-output-dir '$RUN_DIR' \
    --out-dir '${RUN_DIR}_normal20_00125' \
    --model-path '$MODEL_PATH' \
    --experiments rppo_medium_lstm128x2_observable12_target8_macro_library_v11rx_stage2 \
    --episodes 20 \
    --video-episodes 0 \
    --env-overrides-file '$OVERRIDES_REMOTE' \
    --device cpu"

echo "run_dir=$RUN_DIR"
echo "model_path=$MODEL_PATH"
