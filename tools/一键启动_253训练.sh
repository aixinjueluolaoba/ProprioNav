#!/usr/bin/env bash
set -euo pipefail

# 一键启动253训练 + 监控 + 拉取结果

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║  盲人寻路 RL 训练启动脚本 (253机器)                            ║"
echo "║  配置: 10k轮 + 256并行环境 + 20次评估视频                      ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

# 参数
REMOTE_RUN_DIR="/home/weiaokang/fishing/blind_nav_rl/runs/连续9维固定脱困_253_10k_256env"
LOCAL_OUTPUT_DIR="/home/diana/screencap/file/blind_nav_253_10k_256env"
EXPERIMENT="rppo_small_lstm128x1_observable9_target8_continuous_fixed_recovery_v1"

echo "📋 配置信息:"
echo "  远程运行目录: $REMOTE_RUN_DIR"
echo "  本地输出目录: $LOCAL_OUTPUT_DIR"
echo "  实验名称: $EXPERIMENT"
echo ""

# 第一步：启动训练
echo "🚀 第一步: 启动253上的训练..."
echo ""
ssh 253 'bash /home/weiaokang/fishing/blind_nav_rl/tools/训练_253_10k_256env.sh'
echo ""

# 获取PID
TRAIN_PID=$(ssh 253 "cat $REMOTE_RUN_DIR/训练.pid 2>/dev/null" || echo "")
if [ -z "$TRAIN_PID" ]; then
  echo "❌ 无法获取训练进程PID，请检查远程执行"
  exit 1
fi

echo "✓ 训练已启动 (PID: $TRAIN_PID)"
echo ""

# 第二步：实时监控
echo "📊 第二步: 启动实时监控 (每30秒更新一次)"
echo "   按 Ctrl+C 停止监控（训练会继续运行）"
echo ""

MONITOR_COUNT=0
while true; do
  MONITOR_COUNT=$((MONITOR_COUNT + 1))

  # 检查进程是否还在运行
  if ! ssh 253 "ps -p $TRAIN_PID >/dev/null 2>&1"; then
    echo ""
    echo "✓ 训练进程已完成"
    break
  fi

  # 显示进度
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "监控周期 #$MONITOR_COUNT - $(date '+%Y-%m-%d %H:%M:%S')"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

  # 最近日志
  echo "📝 最近日志:"
  ssh 253 "tail -5 $REMOTE_RUN_DIR/训练.log 2>/dev/null" | sed 's/^/  /'

  # 资源使用
  echo ""
  echo "💻 资源使用:"
  ssh 253 "ps aux | grep benchmark_state_dims | grep -v grep | awk '{printf \"  CPU: %.1f%% | MEM: %.1f%%\n\", \$3, \$4}'" 2>/dev/null || echo "  (无法获取)"

  # 检查点统计
  echo ""
  echo "📦 训练进度:"
  CKPT_COUNT=$(ssh 253 "ls -1 $REMOTE_RUN_DIR/$EXPERIMENT/models/checkpoint_ep_* 2>/dev/null | wc -l" || echo "0")
  VIDEO_COUNT=$(ssh 253 "ls -1 $REMOTE_RUN_DIR/$EXPERIMENT/eval_*.mp4 2>/dev/null | wc -l" || echo "0")
  echo "  检查点: $CKPT_COUNT 个"
  echo "  评估视频: $VIDEO_COUNT 个"

  # 等待30秒
  sleep 30
done

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# 第三步：拉取结果
echo "📥 第三步: 拉取结果到本地..."
echo ""

mkdir -p "$LOCAL_OUTPUT_DIR"

echo "  拉取结果CSV..."
rsync -avz --progress 253:"$REMOTE_RUN_DIR/state_dim_results.csv" "$LOCAL_OUTPUT_DIR/" 2>/dev/null || echo "  (CSV不可用)"

echo "  拉取最终模型..."
rsync -avz --progress 253:"$REMOTE_RUN_DIR/$EXPERIMENT/$EXPERIMENT.zip" "$LOCAL_OUTPUT_DIR/" 2>/dev/null || echo "  (模型不可用)"

echo "  拉取评估视频..."
mkdir -p "$LOCAL_OUTPUT_DIR/videos"
rsync -avz --progress 253:"$REMOTE_RUN_DIR/$EXPERIMENT/eval_*.mp4" "$LOCAL_OUTPUT_DIR/videos/" 2>/dev/null || echo "  (视频不可用)"

echo "  拉取拼接视频..."
rsync -avz --progress 253:"$REMOTE_RUN_DIR/$EXPERIMENT/eval_concat_compressed.mp4" "$LOCAL_OUTPUT_DIR/" 2>/dev/null || echo "  (拼接视频不可用)"

echo "  拉取训练日志..."
rsync -avz --progress 253:"$REMOTE_RUN_DIR/训练.log" "$LOCAL_OUTPUT_DIR/" 2>/dev/null || echo "  (日志不可用)"

echo ""
echo "✓ 拉取完成"
echo ""

# 第四步：显示结果
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📊 最终结果"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "本地输出目录: $LOCAL_OUTPUT_DIR"
echo ""
echo "文件列表:"
ls -lh "$LOCAL_OUTPUT_DIR" | tail -10
echo ""

if [ -f "$LOCAL_OUTPUT_DIR/state_dim_results.csv" ]; then
  echo "训练结果摘要:"
  tail -1 "$LOCAL_OUTPUT_DIR/state_dim_results.csv" | awk -F',' '{
    print "  成功率: " $9
    print "  平均步数: " $10
    print "  平均奖励: " $11
    print "  平均最终距离: " $12
    print "  平均碰撞数: " $13
  }'
fi

echo ""
echo "✓ 训练流程完成！"
echo ""

# 可选：清理远程文件
echo "🧹 清理远程文件?"
read -p "是否删除远程文件以节省空间? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
  echo "删除远程文件..."
  ssh 253 "rm -rf $REMOTE_RUN_DIR" && echo "✓ 远程文件已删除"
else
  echo "保留远程文件"
fi

echo ""
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║  完成！                                                        ║"
echo "╚════════════════════════════════════════════════════════════════╝"
