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
- `scripts/build_android_arm64.sh`: NCNN/Rust Android cross-build.
- `android/proprionav`: arm64-v8a AAR module with JNI/Kotlin wrapper.
- `android/sample`: minimal Android integration app.
- `android/dist`: checked release AAR and sample APK.
- `pipeline_out/test_inference.py`: low-level parity test.
- `pipeline_out/test_nav_api.py`: high-level parity test.

## Maze / general policy (map_env)

- `map_env/README.md`: full documentation (env, training, results, deployment).
- `map_env/generate_maze.py`: procedural maze -> `map_env/maze_grid.npz`.
- `map_env/maze_env.py`: `GPUImageMazeNavEnv` (grid collision, random goals, curriculum).
- `map_env/mixed_env.py`: `MixedNavEnv` (half maze + half open world, one coordinate-only policy).
- `pipeline_out/export_mixed_ncnn.py`: exports the hidden-192 general policy to NCNN.
- Flags: `--maze-grid`, `--maze-mix-open-world`, `--maze-target-range`,
  `--maze-curriculum-max`, `--maze-map-guidance`, `--learn-deceleration`,
  `--maze-overshoot-coef`, `--terminal-slow-radius`, `--position-age-min-ms`.
- Weights: `policy_weights_mixed.pth` (general), `policy_weights_overshoot.pth`
  (maze, learned approach slowdown). NCNN: `policy_mixed.ncnn.param/bin`.
- Coordinate-only ABI input stays pos/target/position_age_ms. The lib detects
  the LSTM hidden size from the model `.param` automatically and takes the game
  scale at runtime via `nav_configure(world_size, age_max_ms, stale_ms, max_speed)`
  (defaults = V5; the maze AAR configures 512/800/1200/30).
- Anti-spin is built into the lib: scale-relative motion/stuck/collision
  thresholds plus turn cooldown / no-turn-while-stationary.

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

ANDROID_NDK_HOME="$ANDROID_HOME/ndk/26.1.10909125" \
  bash scripts/build_android_arm64.sh
cd android && ./gradlew :proprionav:assembleRelease :sample:assembleDebug
```

The Linux `pipeline_out/libncnn_rust.so`, videos, logs, PNNX intermediates,
NCNN source and build directories are generated locally and ignored. The checked
Android arm64 SO and release artifacts are intentional deployment assets.
