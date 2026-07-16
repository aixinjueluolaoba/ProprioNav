#!/usr/bin/env bash
# 实时监控253上的训练进度

RUN_DIR=${1:-/home/weiaokang/fishing/blind_nav_rl/runs/连续9维固定脱困_253_10k_256env}

echo "=== 监控训练进度 ==="
echo "运行目录: $RUN_DIR"
echo ""

# 检查进程
if [ -f "$RUN_DIR/训练.pid" ]; then
  PID=$(cat "$RUN_DIR/训练.pid")
  if ssh 253 "ps -p $PID >/dev/null 2>&1"; then
    echo "✓ 训练进程运行中 (PID: $PID)"
  else
    echo "✗ 训练进程已停止"
  fi
else
  echo "✗ 未找到训练进程"
fi

echo ""
echo "=== 最近日志 ==="
ssh 253 "tail -30 $RUN_DIR/训练.log" 2>/dev/null || echo "日志文件不可用"

echo ""
echo "=== 资源使用 ==="
ssh 253 "ps aux | grep benchmark_state_dims | grep -v grep | awk '{print \"CPU: \" \$3 \"% MEM: \" \$4 \"%\"}'" 2>/dev/null || echo "无法获取资源信息"

echo ""
echo "=== 输出文件统计 ==="
ssh 253 "ls -lh $RUN_DIR/rppo_small_lstm128x1_observable9_target8_continuous_fixed_recovery_v1/ 2>/dev/null | tail -20" || echo "输出目录不可用"

echo ""
echo "=== 检查点统计 ==="
ssh 253 "ls -1 $RUN_DIR/rppo_small_lstm128x1_observable9_target8_continuous_fixed_recovery_v1/models/checkpoint_ep_* 2>/dev/null | wc -l" | xargs echo "已保存检查点数:"

echo ""
echo "=== 评估视频统计 ==="
ssh 253 "ls -1 $RUN_DIR/rppo_small_lstm128x1_observable9_target8_continuous_fixed_recovery_v1/eval_*.mp4 2>/dev/null | wc -l" | xargs echo "已生成评估视频数:"

echo ""
echo "提示: 使用 'tail -f' 查看实时日志:"
echo "  ssh 253 'tail -f $RUN_DIR/训练.log'"
