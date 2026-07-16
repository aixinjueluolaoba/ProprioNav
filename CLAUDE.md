# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Blind Navigation RL** is a reinforcement learning environment for training agents to navigate toward a target without visual obstacle information. The agent receives only relative target position, heading error, and motion history—obstacles (trees, mountains) are hidden and discovered through collision feedback.

**Current recommended version:**
- Algorithm: `RecurrentPPO` with `MlpLstmPolicy` (LSTM hidden size 64)
- State: `observable8_delta45` (8 dimensions)
- Action: 3D continuous (delta_stick_angle ±45°, speed, jump)
- Training: 256 parallel environments, 10,000 episodes
- Model: ~42k parameters, 0.53 MB

## Architecture

### Core Environment (`blind_nav_rl/env.py`)

The `BlindNavEnv` class implements a Gymnasium environment with:

- **World**: 2400×2400 unit area with random obstacles
- **Obstacles**: Trees (circular, radius ~28) and mountains (irregular polygons)
- **Physics**: dt=0.3s per step, max turn rate 160°/s, tangential sliding on collision
- **Observation**: 8-17 dimensional state (depending on mode), no obstacle/ray data
- **Action**: 3D continuous [delta_stick_angle, speed_norm, jump]
- **Reward**: Progress reward, time cost, collision penalty, stuck penalty, jump incentives

### State Representations

The project supports multiple state modes (see `benchmark_state_dims_10k.py`):

- `observable8_delta45`: 8D, recommended for RecurrentPPO
  - Target relative position (2D), angle error, angle error delta, distance progress, movement distance, last action speed, last action jump
- `observable10`: 10D, earlier version (not recommended)
- `observable6`: 6D, minimal state (lower performance)
- `full17`: All available observations

### Action Space

3D continuous actions:
1. `delta_stick_angle`: [-1,1] → [-45°, +45°] relative to current heading
2. `speed_norm`: [0,1] → [50, 100] units/step
3. `jump`: > 0.5 triggers jump/unstuck behavior

## Common Commands

### Local Development

```bash
# Activate environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ML

# Smoke test (verify environment works)
python smoke_test.py

# Random policy demo
python show_random_policy.py
```

### Training on Remote (ssh 253)

**Current recommended training:**
```bash
rtk ssh 253 'cd /home/weiaokang/fishing/blind_nav_rl && source /home/weiaokang/ML/bin/activate && python benchmark_state_dims_10k.py --output-dir /home/weiaokang/screencap/file/blind_nav_rppo_observable8_delta45_10k_vec256 --episodes 10000 --n-envs 256 --eval-episodes 20 --video-episodes 3 --only rppo_lstm64_observable8_delta45'
```

**For 384 parallel environments (requires higher ulimit):**
```bash
rtk ssh 253 'ulimit -n 8192 && cd /home/weiaokang/fishing/blind_nav_rl && source /home/weiaokang/ML/bin/activate && python benchmark_state_dims_10k.py --output-dir /home/weiaokang/screencap/file/blind_nav_rppo_observable8_delta45_100k_vec384 --episodes 100000 --n-envs 384 --eval-episodes 20 --video-episodes 3 --only rppo_lstm64_observable8_delta45'
```

### Benchmarking

`benchmark_state_dims_10k.py` is the main training script. It:
- Trains multiple algorithm/state combinations
- Evaluates on standard test episodes
- Generates evaluation videos
- Outputs results CSV with success rate, avg steps, collisions, etc.

Key parameters:
- `--episodes`: Total training episodes (default 10000)
- `--n-envs`: Parallel environments (default 128, recommended 256)
- `--eval-episodes`: Evaluation runs (default 20)
- `--video-episodes`: Video recording runs (default 3)
- `--only`: Train specific experiment only (e.g., `rppo_lstm64_observable8_delta45`)

## Key Concepts

### Reward Function

Per-step rewards:
- Progress: `progress * 0.08` (positive when moving closer to target)
- Time cost: `-0.2 * dt`
- Collision: `-0.25`
- Stuck penalty: `-0.05 * min(no_progress_time, 5.0)` (when stuck > 1s)
- Jump penalty: `-0.08` (when not stuck)
- Jump reward: `+0.5` (when stuck and jumping)
- Recovery reward: `+1.0` (escaping stuck state with forward progress)
- Goal reward: `+20.0` (reached target)
- Timeout penalty: `-5.0` (exceeded max steps without reaching target)

### State Dimensions (observable8_delta45)

1. `delta_x / world_size` - target X offset, normalized
2. `delta_y / world_size` - target Y offset, normalized
3. `angle_error / pi` - heading error to target, normalized [-1,1]
4. `delta_angle_error / pi` - change in heading error (improving/worsening)
5. `distance_progress / 30` - distance reduced this step
6. `movement_dist / 40` - actual movement distance this step
7. `last_action_speed` - previous speed command, normalized
8. `last_action_jump` - previous jump command

### Why RecurrentPPO + LSTM?

- **Recurrent**: LSTM maintains hidden state across steps, enabling temporal reasoning about stuck/collision patterns
- **PPO**: Stable on-policy learning with clipped objectives
- **LSTM hidden size 64**: Balances expressiveness vs. model size (42k params total)
- **observable8_delta45**: Provides error-correction feedback (angle_error, delta_angle_error) suitable for incremental joystick adjustments

## Important Notes

- **Not all parallel is better**: 384 environments showed worse results than 256 (higher final distance, more collisions). Diminishing returns exist.
- **Training stability**: Current route focuses on stabilizing RecurrentPPO long-term training, not increasing parallelism further.
- **Deployment timing**: Real environment decision cycle should match training dt (~0.3s). Large timing variance may require adding delta_t to state.
- **Jump statistics**: Current models show high jump attempts (~63) but lower triggers (~21). This is acceptable if real environment supports multi-touch without interrupting movement.
- **File handles**: 384 parallel environments may hit system ulimit (default 1024). Increase with `ulimit -n 8192` before training.

## Output Locations

**Current recommended model (256 parallel, 10k episodes):**
- Model: `/home/diana/screencap/file/blind_nav_rppo_observable8_delta45_10k_vec256/rppo_lstm64_observable8_delta45/rppo_lstm64_observable8_delta45.zip`
- Results CSV: `http://43.242.75.250:12346/file/blind_nav_rppo_observable8_delta45_10k_vec256/state_dim_results.csv`
- Eval video: `http://43.242.75.250:12346/file/blind_nav_rppo_observable8_delta45_10k_vec256/rppo_lstm64_observable8_delta45/eval_concat_compressed.mp4`

## Verified but Not Recommended

- **observable6 + delta90**: 95% success, 81.2 avg steps, 43.38 final distance (worse than observable8_delta45)
- **observable8_delta45 + 384 parallel + 100k episodes**: 95% success, 71.05 avg steps, 91.72 final distance (regression, slower training)

## Testing

- `smoke_test.py`: Validates environment with scripted policy (straight toward target)
- `show_random_policy.py`: Visualizes random policy behavior
- `render_eval10_concat.py`: Generates evaluation videos from trained models
