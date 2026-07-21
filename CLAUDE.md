# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**ProprioNav** (基于本体感知反馈的盲区导航系统) is a reinforcement learning environment and inference deployment engine for training agents to navigate toward a target without visual obstacle information. The agent receives only relative target offset, heading error, motion history, and collision feedback—obstacles are hidden and discovered dynamically.

**Current recommended version:**
- Algorithm: `RecurrentPPO` with LSTM (hidden size 64)
- State Mode: `observable12_target8_macro_library_v11rxc_stage2` (12 dimensions)
- Action Space: MultiDiscrete `[5, 2, 8]`
  - Steering: 5 bins (`[-45°, -15°, 0°, +15°, +45°]` steering offset)
  - Speed: 2 bins (`[50, 100]` units/step)
  - Macro: 8 bins (0: normal navigation, 1-7: macro recovery behaviors)
- Model: ~42k parameters

## Directory Structure

```text
ProprioNav/
├── blind_nav_rl/               # Gym Environment
│   ├── __init__.py
│   └── env.py                  # Gym environment definitions
├── ncnn_rust/                  # Rust Inference Library
│   ├── Cargo.toml              # Rust library dependencies
│   ├── build.rs                # Link script for libncnn
│   └── src/lib.rs              # Rust-FFI C-API bridging and zero-copy inference
├── pipeline_out/               # Models & Verification Scripts
│   ├── policy_weights.pth      # PyTorch policy weights
│   ├── export_onnx.py          # PyTorch LSTM equivalent math rewrite & ONNX/TorchScript export
│   ├── policy.param / bin      # FP16 NCNN network params and weights
│   ├── libncnn_rust.so         # Compiled Rust-NCNN inference library
│   ├── generate_svg.py         # System architecture diagram generator
│   ├── proprio_nav_architecture.svg # Architecture diagram
│   └── test_inference.py       # Python-Rust consistency tester (ctypes)
├── run_pipeline.py             # End-to-end training and evaluation script
├── render_eval10_concat.py     # Evaluation & video generation script
├── README.md
└── CLAUDE.md
```

## Common Commands

### Local Pipeline Execution
To execute end-to-end GPU vectorized training, CPU parallel evaluation, and video rendering:
```bash
python run_pipeline.py
```

### Compile Rust FFI Library
To build the high-performance Rust FFI library (AVX-512 accelerated NCNN binding):
```bash
cd ncnn_rust
RUSTFLAGS="-C linker=/usr/bin/gcc" cargo build --release
cp target/release/libncnn_rust.so ../pipeline_out/
```

### Run Consistency Verification
To run the ctypes-based comparison between PyTorch and Rust-NCNN outputs:
```bash
python pipeline_out/test_inference.py
```

## Response & Style Guidelines

- **Always provide full absolute paths** for all files mentioned in responses to the user (e.g., `[/home/diana/盲人寻路/run_pipeline.py](file:///home/diana/盲人寻路/run_pipeline.py)`).
- **Whenever a video is generated or requested to be displayed**, automatically run `mpv` locally to play the video file immediately.
