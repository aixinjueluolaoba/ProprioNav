# CLAUDE.md

## Project

ProprioNav is a recurrent PPO blind-navigation policy with a stateful Rust/NCNN
deployment library.

- Observation: 13 delayed-position values assembled inside the SO.
- LSTM hidden size: auto-detected from the model `.param` (96 for V5, 192 for the
  maze/general policy); actor width 48 (V5) / 96 (general).
- Neural actions: 7 steering bins and 2 speed bins.
- Steering: goal-relative offsets in free space, heading-relative offsets during recovery.
- Jump: collision probe in the SO; the V5 macro model also has a recovery macro head.
- External step input: current position, target position, position_age_ms.
  Collision is always inferred inside the library; there is no caller-supplied
  collision flag.
- External output: relative turn, speed, jump, and the library-maintained
  absolute joystick angle `abs_angle`.

## Important files

- `run_pipeline.py`: GPU-vectorized training and evaluation.
- `pipeline_out/policy_weights_mixed.pth` + `pipeline_out/policy_mixed.ncnn.param/bin`:
  canonical maze/general PyTorch weights and FP32 NCNN model.
- `pipeline_out/policy_weights_overshoot.pth`: maze weights with learned approach slowdown.
- `pipeline_out/export_mixed_ncnn.py`: maze/general TorchScript + PNNX exporter.
- `pipeline_out/delayed_belief.py`: scalar delayed-position belief (reused by the viewer).
- `ncnn_rust/src/lib.rs`: low-level inference and high-level stateful navigation ABI.
- `ncnn_rust/include/proprionav.h`: public C header.
- `scripts/build_android_arm64.sh`: NCNN/Rust Android cross-build.
- `android/proprionav`: arm64-v8a AAR module with JNI/Kotlin wrapper.
- `android/sample`: minimal Android integration app.
- `android/dist`: checked release AAR and sample APK.
- `pipeline_out/test_nav_api.py`: high-level ABI smoke test (incl. `abs_angle`).

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
- `nav_step` / `nav_step_feedback` also return `abs_angle`: the absolute
  joystick heading in radians (y-up, 0 = +x/right), maintained inside the lib.
  It is seeded with the bearing to the target on the first call and then
  accumulates the policy's relative turns, so callers feed it straight to the
  joystick and must NOT also accumulate `turn_delta`. Pass NULL to skip it.

## Commands

```bash
# Maze / general policy eval (10 concurrent episodes + concat video)
MPLBACKEND=Agg python run_pipeline.py --eval-only --no-play \
  --maze-grid map_env/maze_grid.npz --maze-target-range 0.25 \
  --weights-name policy_weights_mixed.pth \
  --hidden-dim 192 --actor-width 96 --eval-episodes 10 --eval-sample \
  --position-age-min-ms 200 --position-age-max-ms 800 \
  --position-stale-ms 1200 --position-age-extreme-prob 0.3

RUSTFLAGS="-C linker=/usr/bin/gcc" cargo build \
  --manifest-path ncnn_rust/Cargo.toml --release
cp ncnn_rust/target/release/libncnn_rust.so pipeline_out/

python pipeline_out/test_nav_api.py

ANDROID_NDK_HOME="$ANDROID_HOME/ndk/26.1.10909125" \
  bash scripts/build_android_arm64.sh
cd android && ./gradlew :proprionav:assembleRelease :sample:assembleDebug
```

The Linux `pipeline_out/libncnn_rust.so`, videos, logs, PNNX intermediates,
NCNN source and build directories are generated locally and ignored. The checked
Android arm64 SO and release artifacts are intentional deployment assets.
