# 🌲 2D Gymnasium 盲人导航强化学习 NCNN 部署与 Rust 推理库工程

本工程是针对 2D Gymnasium 盲人导航强化学习（RL）模型的轻量级部署与推理系统。
我们通过将带有 LSTM 隐状态的 Recurrent PPO 策略网络重构并转换为 NCNN 模型格式（FP16 半精度优化），使用 Rust 编写高性能 FFI 桥接接口并打包为动态链接库 `.so`，最后通过 Python 实现了高精度的 ctypes 自动化回归测试。

---

## 🗺️ 部署目录结构

```text
/home/diana/盲人寻路/
├── ncnn_rust/                 # Rust 编译工程
│   ├── Cargo.toml             # Rust 依赖与 cdylib（动态库）配置
│   ├── build.rs               # 静态链接 libncnn.a 并关联系统动态库
│   └── src/
│       └── lib.rs             # C-API FFI 导出与 NCNN 零拷贝推理逻辑
│
├── pipeline_out/              # 模型资产与部署产物输出目录
│   ├── policy_weights.pth     # 原始 PyTorch 训练权重 (PPO 策略模型)
│   ├── export_onnx.py         # PyTorch 重构导出 ONNX & TorchScript 脚本
│   ├── policy.pt              # TorchScript 追踪模型
│   ├── policy.param           # PNNX 转换输出的 NCNN 网络描述文件 (已包含 FP16 参数)
│   ├── policy.bin             # PNNX 转换输出的 NCNN 权重文件 (已包含 FP16 参数)
│   ├── libncnn_rust.so        # 最终编译生成的高性能 Rust 动态链接库 (16MB，已打包 AVX-512)
│   └── test_inference.py      # Python 自动化精度比对验证脚本
│
└── README.md                  # 本说明文档
```

---

## 🔄 模型转换流水线 (Model Conversion Pipeline)

### 1. PyTorch 模型重构
由于 PyTorch 的 `nn.LSTM` 默认导出的 ONNX 会生成复杂的 runtime 转置和切片，导致标准的 `onnx2ncnn` 无法正确识别其常量权重，在 `.bin` 中导出为空权重（2=0）。
我们在 [export_onnx.py](file:///home/diana/盲人寻路/pipeline_out/export_onnx.py) 中，在数学上等价地将单步 LSTM 推理展开为标准的矩阵乘法和基本逻辑：
* 在构造函数 `__init__` 中提前对权重进行转置：
  ```python
  self.W_ih_t = nn.Parameter(model.lstm.weight_ih_l0.clone().t())
  self.W_hh_t = nn.Parameter(model.lstm.weight_hh_l0.clone().t())
  ```
* 在 `forward` 阶段直接进行 `torch.matmul`：
  ```python
  gates = torch.matmul(x, self.W_ih_t) + self.b_ih + torch.matmul(h, self.W_hh_t) + self.b_hh
  ```
消除了任何动态 `Transpose` 节点后，导出的模型能够被完全静态解析。

### 2. 使用 PNNX 转换 (FP16 优化)
我们直接将模型导出为 TorchScript 格式（`policy.pt`），并使用 NCNN 官方推荐的最先进的 **PNNX** 转换器一键生成 NCNN 参数与模型：
```bash
/home/diana/miniconda3/envs/ML/bin/pnnx \
  /home/diana/盲人寻路/pipeline_out/policy.pt \
  inputshape=[1,12],[1,64],[1,64]
```
PNNX 默认启用了 **FP16** 精度的网络优化，成功将浮点模型权重无损地压缩至 `45KB` 左右。之后复制为标准名称：
* [policy.param](file:///home/diana/盲人寻路/pipeline_out/policy.param)
* [policy.bin](file:///home/diana/盲人寻路/pipeline_out/policy.bin)

---

## 🦀 Rust 推理共享库 (ncnn_rust)

### 1. FFI C-API 接口定义
Rust 库 [lib.rs](file:///home/diana/盲人寻路/ncnn_rust/src/lib.rs) 通过 FFI 声明并封装了 NCNN 内部的 C-API 符号，实现了全管道零拷贝：
* **`init_net`**
  ```rust
  #[no_mangle]
  pub unsafe extern "C" fn init_net(param_path: *const c_char, bin_path: *const c_char) -> *mut c_void
  ```
  输入 `.param` 和 `.bin` 路径，实例化 `ncnn::Net` 并完成参数及模型装载。成功返回句柄指针，失败返回 NULL。

* **`free_net`**
  ```rust
  #[no_mangle]
  pub unsafe extern "C" fn free_net(net_ptr: *mut c_void)
  ```
  释放 `ncnn::Net` 实例，安全清理 C 堆内存。

* **`run_inference`**
  ```rust
  #[no_mangle]
  pub unsafe extern "C" fn run_inference(
      net_ptr: *mut c_void,
      x: *const f32,             // 观测向量 (float[12])
      h_in: *const f32,          // 输入隐藏状态 (float[64])
      c_in: *const f32,          // 输入细胞状态 (float[64])
      steer_logits: *mut f32,    // 舵角输出 logits (float[5])
      speed_logits: *mut f32,    // 速度输出 logits (float[2])
      macro_logits: *mut f32,    // 宏动作输出 logits (float[8])
      h_out: *mut f32,           // 更新后的隐藏状态写入区 (float[64])
      c_out: *mut f32,           // 更新后的细胞状态写入区 (float[64])
  ) -> i32
  ```
  执行前向计算。内部通过 `ncnn_mat_create_external_1d` 实现传入内存数据的**零拷贝绑定**，并在推理完毕后安全销毁提取器和中间 Mat 临时变量。成功返回 `0`。

### 2. 编译指南
为了避免开发环境中的 linker 符号重定向报错（例如 `unknown option -m64`），请使用真实的系统 GCC 编译器作为后端链接器进行编译：
```bash
cd /home/diana/盲人寻路/ncnn_rust
RUSTFLAGS="-C linker=/usr/bin/gcc" cargo build --release
```
编译产物会生成在 `target/release/libncnn_rust.so`，由于静态打包了 NCNN 内部的全部向量化计算库（支持系统 AVX-512 SIMD 并行加速），其大小约为 16MB。编译完成后可直接拷贝至 [pipeline_out](file:///home/diana/盲人寻路/pipeline_out) 目录。

---

## 🐍 Python 自动化精度比对 (Python test code)

测试验证脚本 [test_inference.py](file:///home/diana/盲人寻路/pipeline_out/test_inference.py) 使用 `ctypes` 装载 Rust 动态库并和原生的 PyTorch 执行对齐校验：

### 运行方式
```bash
/home/diana/miniconda3/envs/ML/bin/python /home/diana/盲人寻路/pipeline_out/test_inference.py
```

### 验证精度报告示例
在相同的随机数输入下，PyTorch 浮点精度与经过 PNNX FP16 浮点优化转换后的 NCNN 推理结果对齐精度偏差如下（最大绝对差限制在 `1e-3` 以下）：
```text
================== 🌲 PyTorch VS NCNN 推理一致性验证 🌲 ==================

--- 1. steer_logits 比对 ---
PyTorch: [-0.25599557  0.3966241   0.03859585  0.07449278 -0.11496811]
NCNN   : [-0.255881    0.39641744  0.03866953  0.07448566 -0.11482652]
Max Abs Diff: 0.000207

--- 2. speed_logits 比对 ---
PyTorch: [-0.1282038   0.10113877]
NCNN   : [-0.1281742   0.10108508]
Max Abs Diff: 0.000054

--- 3. macro_logits 比对 ---
PyTorch: [-0.05721664 -0.12095418 -0.09497169  0.06450429  0.16704914 -0.17184229
  0.20102794 -0.15634519]
NCNN   : [-0.05725737 -0.12096433 -0.0949143   0.06440697  0.16704965 -0.17185226
  0.20095643 -0.15621693]
Max Abs Diff: 0.000128

--- 4. h_next 比对 (部分前 5 维数据展示) ---
PyTorch: [-0.15412176 -0.24694636 -0.14697811 -0.60128635 -0.3198939 ]
NCNN   : [-0.15404864 -0.24690773 -0.14692682 -0.60118216 -0.31991163]
Max Abs Diff: 0.000134

--- 5. c_next 比对 (部分前 5 维数据展示) ---
PyTorch: [-0.18451998 -0.47837177 -0.2880792  -1.2733505  -0.56230444]
NCNN   : [-0.1844478  -0.47834173 -0.2879503  -1.2731442  -0.56231123]
Max Abs Diff: 0.000354

==============================================================
🎉 验证通过！PyTorch 与 Rust-NCNN 推理结果高度一致！最大偏差: 0.000354
==============================================================
```

数据表明，PNNX 生成的压缩版 FP16 模型在节省一倍显存的同时，输出偏差（~0.0003）完全处于高精度安全部署范围内，可无缝平替原本的 PyTorch 在线寻路组件。
