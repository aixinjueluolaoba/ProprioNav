#!/usr/bin/env bash
set -euo pipefail

# 从253拉取训练结果到本地

REMOTE_RUN_DIR=${1:-/home/weiaokang/fishing/blind_nav_rl/runs/连续9维固定脱困_253_10k_256env}
LOCAL_OUTPUT_DIR=${2:-/home/diana/screencap/file/blind_nav_253_10k_256env}
EXPERIMENT_NAME="rppo_small_lstm128x1_observable9_target8_continuous_fixed_recovery_v1"

echo "=== 从253拉取训练结果 ==="
echo "远程目录: $REMOTE_RUN_DIR"
echo "本地目录: $LOCAL_OUTPUT_DIR"
echo ""

mkdir -p "$LOCAL_OUTPUT_DIR"

# 拉取结果CSV和MD
echo "拉取结果文件..."
rsync -avz --progress 253:"$REMOTE_RUN_DIR/state_dim_results.csv" "$LOCAL_OUTPUT_DIR/" 2>/dev/null || echo "CSV文件不可用"
rsync -avz --progress 253:"$REMOTE_RUN_DIR/state_dim_results.md" "$LOCAL_OUTPUT_DIR/" 2>/dev/null || echo "MD文件不可用"

# 拉取最终模型
echo "拉取最终模型..."
rsync -avz --progress 253:"$REMOTE_RUN_DIR/$EXPERIMENT_NAME/$EXPERIMENT_NAME.zip" "$LOCAL_OUTPUT_DIR/" 2>/dev/null || echo "模型文件不可用"

# 拉取视频
echo "拉取评估视频..."
mkdir -p "$LOCAL_OUTPUT_DIR/videos"
rsync -avz --progress 253:"$REMOTE_RUN_DIR/$EXPERIMENT_NAME/eval_*.mp4" "$LOCAL_OUTPUT_DIR/videos/" 2>/dev/null || echo "视频文件不可用"

# 拉取拼接视频
echo "拉取拼接视频..."
rsync -avz --progress 253:"$REMOTE_RUN_DIR/$EXPERIMENT_NAME/eval_concat_compressed.mp4" "$LOCAL_OUTPUT_DIR/" 2>/dev/null || echo "拼接视频不可用"

# 拉取日志
echo "拉取训练日志..."
rsync -avz --progress 253:"$REMOTE_RUN_DIR/训练.log" "$LOCAL_OUTPUT_DIR/" 2>/dev/null || echo "日志文件不可用"

echo ""
echo "=== 拉取完成 ==="
echo "本地结果位置: $LOCAL_OUTPUT_DIR"
echo ""
echo "文件列表:"
ls -lh "$LOCAL_OUTPUT_DIR"

echo ""
echo "=== 清理远程文件 ==="
read -p "是否删除远程文件以节省空间? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
  echo "删除远程文件..."
  ssh 253 "rm -rf $REMOTE_RUN_DIR" && echo "✓ 远程文件已删除"
else
  echo "保留远程文件"
fi
