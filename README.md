# ProprioNav

仅依赖自身位置、目标位置和人物朝向的盲区导航策略。当前 V3 使用 10 维本体感知观测和
LSTM，空旷区贴近目标直行，碰撞或停滞后切换为朝向相对绕行。跳跃不再使用不稳定的神经
网络动作头，而由共享库根据连续位置反馈执行一次试跳、失败冷却和绕行。

1024 局 hard 仿真评估：成功率 96.7%，成功局中位 52 步，中位路径比 1.09。

## 目录

```text
.
├── run_pipeline.py                    # GPU 向量化训练与视频评估
├── render_eval10_concat.py            # 当前环境的视频渲染工具
├── ncnn_rust/
│   ├── include/proprionav.h           # 稳定 C ABI
│   ├── src/lib.rs                     # NCNN 推理与内部导航状态
│   └── build.rs
├── pipeline_out/
│   ├── policy_weights_v3_hybrid_sharp.pth
│   ├── policy.param / policy.bin      # 当前 FP32 NCNN 模型
│   ├── export_v3_ncnn.py
│   ├── eval_2d_metrics.py
│   ├── test_inference.py
│   └── test_nav_api.py
└── examples/
    └── fastapi_server.py
```

`libncnn_rust.so` 是本地构建产物，不提交到 Git。

## SO 接口

推荐只使用 `ncnn_rust/include/proprionav.h` 中的高层接口：

```c
void* nav_init(const char* param_path, const char* bin_path);

int nav_step(
    void* nav,
    float pos_x, float pos_y,
    float target_x, float target_y,
    float heading,
    float* direction,
    float* speed,
    int* jump
);

void nav_free(void* nav);
```

调用方每个决策周期反馈最新位置和朝向即可。库内部维护 LSTM、碰撞/停滞推断、hybrid
恢复与跳跃冷却；输出方向为弧度，速度为 50 或 100，跳跃为 0/1。当前策略按 `dt=0.3`
训练，`direction` 已应用每步最大 45° 的转向限制。

## 训练和快速评估

依赖 Python、PyTorch、NumPy、Matplotlib 和 FFmpeg。默认配置已对齐 V3：
`hidden=96`、`actor=48`、`hard collision`、`hybrid h10`、`jump_probe`。

```bash
# 从头训练时使用新的权重名，避免续训正式权重
python run_pipeline.py --train-only \
  --episodes 1000000 \
  --weights-name policy_weights_candidate.pth

# 快速批量指标评估
python pipeline_out/eval_2d_metrics.py \
  --weights pipeline_out/policy_weights_v3_hybrid_sharp.pth \
  --action-mode hybrid \
  --hybrid-free-max-deg 10 \
  --obstacle-signal-mode jump_probe \
  --jump-controller probe \
  --episodes 1024

# 生成训练仿真环境评估视频
python run_pipeline.py --eval-only --no-play
```

## 导出 NCNN

使用 FP32。FP16 的连续误差虽小，但可能改变离散动作的 argmax。

```bash
python pipeline_out/export_v3_ncnn.py
pnnx pipeline_out/policy_v3.pt \
  'inputshape=[1,10],[1,96],[1,96]' fp16=0
cp pipeline_out/policy_v3.ncnn.param pipeline_out/policy.param
cp pipeline_out/policy_v3.ncnn.bin pipeline_out/policy.bin
```

## 构建共享库

`build.rs` 期望静态库位于 `pipeline_out/ncnn_source/build/src/libncnn.a`。准备 NCNN 后执行：

```bash
RUSTFLAGS="-C linker=/usr/bin/gcc" cargo build \
  --manifest-path ncnn_rust/Cargo.toml --release
cp ncnn_rust/target/release/libncnn_rust.so pipeline_out/
```

共享库只依赖标准 C/C++、数学库和 GNU OpenMP，不依赖 protobuf 运行时。

## 验证

```bash
python pipeline_out/test_inference.py
python pipeline_out/test_nav_api.py
```

前者对齐 PyTorch 与 NCNN 的 logits/LSTM 状态，后者逐步对齐高层 `nav_step` 的方向、
速度和跳跃控制。HTTP 接入示例见 `examples/README.md`。
