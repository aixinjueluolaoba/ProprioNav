# 迷宫 / 通用寻路环境 (map_env)

在 `run_pipeline.py` 的向量化 PPO 管线里，用「栅格迷宫」或「迷宫+开放世界混合」训练
**纯坐标**的盲导航策略，并支持导出 NCNN + C ABI 部署。

## 组成

| 文件 | 作用 |
|---|---|
| `generate_maze.py` | 程序化生成干净迷宫（BSP 房间+走廊+障碍），输出 `maze_grid.npz`（含测地距离场） |
| `build_maze_grid.py` | 从等距截图提取栅格（描图版，输出 `maze_grid_traced.npz`） |
| `maze_env.py` | `GPUImageMazeNavEnv`：继承 V4 环境，栅格碰撞、随机目标、课程、减速奖励 |
| `mixed_env.py` | `MixedNavEnv`：一半迷宫 + 一半开放世界，共用一套纯坐标策略 |
| `render_maze.py` / `render_policy_rollout.py` | 迷宫轨迹渲染（PNG/MP4） |
| `smoke_test.py` | 随机走 200 步 + 穿墙检查 |

## 两种模式

- **地图引导**（`--maze-map-guidance`）：观测含预计算测地场的「罗盘方向」。
  成功率高但**需要完整先验地图**（非通用）。
- **纯坐标**（默认）：只用自身坐标 + 目标坐标 + 定位年龄派生的信息（速度、卡顿、碰撞）。
  可跨游戏，是本目录主推的通用模式。

## 观测 / 动作（两个子环境一致）

- 观测 13 维：目标相对位置、目标方位 sin/cos、速度 xy、目标距离、卡顿、碰撞触觉、
  低障碍信号、定位年龄、朝向置信度、上一步转向。
- 动作：7 档相对转向（±45°/步）+ 2 档速度（慢/快，可用 `--speed-bins 3` 加“停”）+ 跳跃位。
- 网络：LSTM(hidden 192) + actor_fc(96)（旧 V4 是 96/48）。

## 定位延迟与减速

- 定位年龄 200–800ms（`--position-age-min-ms/max-ms`），偶发长时延尖峰。
- **环境强制降速**（默认开启）：物理速度 = `策略速度 × freshness(年龄)`，
  200ms 内 1.0，500ms→0.5，stale(1200ms)→0。策略本身无需学。
- **接近目标限速**（可选，`--terminal-slow-radius`）：进接近段后物理速度封顶为
  `cap_frac × 快档`。可靠但实测对成功率无增益（默认关）。
- **过冲惩罚**（可选，`--maze-overshoot-coef`）：只惩罚接近段“目标距离变大”，
  能让策略**学会**接近减速且不坍缩成恒慢（见 `policy_weights_overshoot.pth`）。

## 训练

```bash
# 1) 生成迷宫（已提交 maze_grid.npz，可跳过）
python -m map_env.generate_maze --seed 0

# 2) 通用混合策略（迷宫 + 开放世界，纯坐标，hidden 192）
python run_pipeline.py --train-only --no-training-video \
  --maze-grid map_env/maze_grid.npz --maze-mix-open-world \
  --hidden-dim 192 --actor-width 96 --rollout-steps 256 --num-envs 2048 \
  --maze-target-range 0.25 \
  --position-age-min-ms 200 --position-age-max-ms 800 --position-stale-ms 1200 \
  --position-age-extreme-prob 0.3 --episodes 1000000 \
  --weights-name policy_weights_mixed.pth

# 3) 迷宫 + 学会接近减速（过冲惩罚）
python run_pipeline.py --train-only --no-training-video \
  --maze-grid map_env/maze_grid.npz --learn-deceleration \
  --maze-overshoot-coef 0.4 --maze-approach-radius 60 \
  --hidden-dim 192 --actor-width 96 --rollout-steps 256 --num-envs 2048 \
  --episodes 600000 --weights-name policy_weights_overshoot.pth
```

## 结果

| 权重 | 场景 | 成功率 |
|---|---|---|
| `pipeline_out/policy_weights_mixed.pth` | 开放世界 / 迷宫(≤1/4 距离) | **~94% / ~30-40%** |
| `pipeline_out/policy_weights_overshoot.pth` | 迷宫(≤1/4)，学到接近减速 | **~40%** |

评估（10 路并发 + 拼接）：

```bash
MPLBACKEND=Agg python run_pipeline.py --eval-only --no-play \
  --maze-grid map_env/maze_grid.npz --maze-target-range 0.25 \
  --weights-name policy_weights_mixed.pth \
  --hidden-dim 192 --actor-width 96 --eval-episodes 10 --eval-sample \
  --position-age-min-ms 200 --position-age-max-ms 800 \
  --position-stale-ms 1200 --position-age-extreme-prob 0.3
```

## 部署（NCNN + C ABI）

导出与编译：

```bash
# 通用模型 -> TorchScript -> NCNN
python pipeline_out/export_mixed_ncnn.py \
  --weights pipeline_out/policy_weights_mixed.pth --output pipeline_out/policy_mixed.pt
pnnx pipeline_out/policy_mixed.pt 'inputshape=[1,13],[1,192],[1,192]' fp16=0
# 产物: pipeline_out/policy_mixed.ncnn.param / .bin

# Linux 动态库
RUSTFLAGS="-C linker=/usr/bin/gcc" \
NCNN_LIB_DIR=<ncnn ubuntu lib dir> \
cargo build --manifest-path ncnn_rust/Cargo.toml --release

# Android arm64（见 scripts/build_android_arm64.sh）
```

**调用方只需给 3 个输入**（`nav_step`）：自身坐标 `(x,y)`、目标/waypoint 坐标 `(x,y)`、
定位年龄 `position_age_ms`；库内部完成朝向/速度估计、LSTM 状态、freshness 限速。
输出：`turn_delta`(弧度 ±45°)、`speed`(0–100)、`jump`。

**注意（部署前必读）**：`ncnn_rust/src/lib.rs` 顶部的常量必须与该模型的训练环境一致：

```rust
const WORLD_SIZE: f32 = 512.0;           // 迷宫栅格边长（开放世界 2250）
const MAX_POSITION_AGE_MS: f32 = 800.0;  // 定位最大年龄
const STALE_POSITION_AGE_MS: f32 = 1200.0;
const MAX_REASONABLE_SPEED: f32 = 30.0;  // 速度归一化尺度（迷宫）
```

hidden 已改为 192；换其它游戏时按该游戏的地图尺度/速度上限改这些常量并重编。

## 已知边界

- 纯坐标在复杂迷宫里只能做**小段**（目标约 1/4 地图内）试错到达；长程需上层拆 waypoint。
- 混合策略在两域间共享容量：开放世界很强，迷宫较难；两者指标会互相牵制。
- 「按定位年龄的条件减速」用奖励/共享速度头**学不出来**（会退化恒慢或与年龄无关），
  可靠做法是环境强制（freshness）或对“过冲”这种真实后果做惩罚。
