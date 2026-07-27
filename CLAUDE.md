# CLAUDE.md

## Project

ProprioNav V3 is a recurrent PPO blind-navigation policy with a stateful Rust/NCNN
deployment library.

- Observation: 10 proprioceptive values maintained inside the SO.
- LSTM hidden size: 96; actor width: 48.
- Neural actions: 7 steering bins and 2 speed bins.
- Steering: goal-relative offsets in free space, heading-relative offsets during recovery.
- Jump: deterministic collision probe in the SO; there is no deployed jump or macro head.
- External step input: current position, target position, heading.
- External output: direction, speed, jump.

## Important files

- `run_pipeline.py`: GPU-vectorized training and evaluation.
- `pipeline_out/policy_weights_v3_hybrid_sharp.pth`: canonical V3 PyTorch weights.
- `pipeline_out/policy.param` and `policy.bin`: canonical FP32 NCNN model.
- `pipeline_out/export_v3_ncnn.py`: V3 TorchScript exporter.
- `ncnn_rust/src/lib.rs`: low-level inference and high-level stateful navigation ABI.
- `ncnn_rust/include/proprionav.h`: public C header.
- `pipeline_out/test_inference.py`: low-level parity test.
- `pipeline_out/test_nav_api.py`: high-level parity test.

## Commands

```bash
python pipeline_out/eval_2d_metrics.py \
  --weights pipeline_out/policy_weights_v3_hybrid_sharp.pth \
  --action-mode hybrid --hybrid-free-max-deg 10 \
  --obstacle-signal-mode jump_probe --jump-controller probe

RUSTFLAGS="-C linker=/usr/bin/gcc" cargo build \
  --manifest-path ncnn_rust/Cargo.toml --release

python pipeline_out/test_inference.py
python pipeline_out/test_nav_api.py
```

`libncnn_rust.so`, videos, logs, PNNX intermediates, NCNN source and Rust build
outputs are generated locally and intentionally ignored.
