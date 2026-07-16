from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces
from stable_baselines3 import A2C, PPO, SAC
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv

from blind_nav_rl import BlindNavEnv
from render_eval10_concat import concat_and_compress, render_episode_video

try:
    from sb3_contrib import RecurrentPPO
except Exception:  # pragma: no cover - optional dependency
    RecurrentPPO = None


N_ENVS = 128
TARGET_EPISODES = 10_000
MAX_STEPS = 240
TREE_COUNT = 90
MOUNTAIN_COUNT = 16
TREE_RADIUS = 28.0
EVAL_EPISODES = 20
VIDEO_EPISODES = 3
CHECKPOINT_INTERVAL = 1_000
PROGRESS_INTERVAL = 500


STATE_MODES: dict[str, list[int]] = {
    "full17": list(range(17)),
    # Keep target vector, heading, speed, progress/stuck/collision timers,
    # stuck flag, jump cooldown, and jump history.
    "reduced12": [0, 1, 5, 6, 7, 8, 9, 10, 11, 12, 13, 16],
    # Deploy-oriented state: no measured speed and no jump cooldown. Use the
    # previous speed command instead because it is available to the script.
    "deploy11": [0, 1, 5, 6, 8, 9, 10, 11, 12, 15, 16],
    # Minimal blind state: target vector, heading, recent progress, stuck
    # timers, collision recency, and jump history.
    "minimal9": [0, 1, 5, 6, 8, 9, 10, 11, 16],
    "observable12_target8_macro_library_v11ds_stage2": [0, 1, 5, 6, 8, 9, 10, 11, 12, 13, 14, 16],
    "observable12_target8_macro_library_v11dsl_stage2": [0, 1, 5, 6, 8, 9, 10, 11, 12, 13, 14, 16],
    "observable12_target8_macro_library_v11tm_stage2": [0, 1, 5, 6, 8, 9, 10, 11, 12, 13, 14, 16],
    "observable12_target8_macro_library_v11re_stage2": [0, 1, 5, 6, 8, 9, 10, 11, 12, 13, 14, 16],
    "observable12_target8_macro_library_v11rx_stage2": [0, 1, 5, 6, 8, 9, 10, 11, 12, 13, 14, 16],
    "observable12_target8_macro_library_v11rxc_stage2": [0, 1, 5, 6, 8, 9, 10, 11, 12, 13, 14, 16],
}

OBSERVABLE10_DIM = 10
OBSERVABLE8_DIM = 8
OBSERVABLE8_TARGET_DIM = 8
OBSERVABLE12_TARGET_DIM = 12
OBSERVABLE9_TARGET_DIM = 9
OBSERVABLE10_TARGET_DIM = 10
OBSERVABLE11_DIM = 11
OBSERVABLE7_DIM = 7
OBSERVABLE6_DIM = 6

OBSERVABLE7_STOP_MODE_LIST = [
    "observable7_stop_delta45",
    "observable7_stop_quality_delta60",
    "observable7_stop_line_delta45",
    "observable7_stop_curriculum_delta45",
    "observable7_stop_stage1_delta45",
    "observable7_stop_stage2_delta45",
    "observable7_stop_stage3_delta45",
    "observable7_stop_stage3_heading_delta45",
    "observable7_stop_final_warm1_delta45",
    "observable7_stop_final_warm2_delta45",
]
OBSERVABLE7_STOP_MODES = set(OBSERVABLE7_STOP_MODE_LIST)
OBSERVABLE11_STOP_MODE_LIST = [
    mode.replace("observable7", "observable11", 1) for mode in OBSERVABLE7_STOP_MODE_LIST
]
OBSERVABLE11_STOP_MODES = set(OBSERVABLE11_STOP_MODE_LIST)

OBSERVABLE8_TARGET_MODE_LIST = [
    "observable8_target8_line_delta90",
    "observable8_target8_v2_offset60",
    "observable8_target8_recovery_angle",
    "observable8_target8_recovery_pressure",
    "observable8_target8_full180_pressure",
    "observable8_target8_macro_recovery_dense_halfmountain",
    "observable8_target8_macro_trigger_dense_halfmountain",
]
OBSERVABLE8_TARGET_MODES = set(OBSERVABLE8_TARGET_MODE_LIST)

OBSERVABLE12_TARGET_MODE_LIST = [
    "observable12_target8_recovery_v4",
    "observable12_target8_discrete_macro_v5",
    "observable12_target8_discrete_macro_v6",
    "observable12_target8_discrete_macro_v7",
    "observable12_target8_discrete_macro_v8",
    "observable12_target8_discrete_trigger_v9_stage1",
    "observable12_target8_discrete_trigger_v9_stage2",
    "observable12_target8_discrete_recovery_v10_stage1",
    "observable12_target8_discrete_recovery_v10_stage2a",
    "observable12_target8_discrete_recovery_v10_stage2",
    "observable12_target8_discrete_recovery_v10_stage3",
    "observable12_target8_discrete_recovery_v10b9_stage1",
    "observable12_target8_discrete_recovery_v10b9_stage2a",
    "observable12_target8_discrete_recovery_v10b9_stage2",
    "observable12_target8_discrete_recovery_v10b9_stage3",
    "observable12_target8_macro_library_v11_stage1",
    "observable12_target8_macro_library_v11_stage2a",
    "observable12_target8_macro_library_v11_stage2",
    "observable12_target8_macro_library_v11_stage3",
    "observable12_target8_macro_library_v11ds_stage2",
    "observable12_target8_macro_library_v11dsl_stage2",
    "observable12_target8_macro_library_v11tm_stage2",
    "observable12_target8_macro_library_v11re_stage2",
    "observable12_target8_macro_library_v11rx_stage2",
    "observable12_target8_macro_library_v11rxc_stage2",
]
OBSERVABLE12_TARGET_MODES = set(OBSERVABLE12_TARGET_MODE_LIST)
OBSERVABLE9_TARGET_MODE_LIST = [
    "observable9_target8_continuous_fixed_recovery_v1",
]
OBSERVABLE9_TARGET_MODES = set(OBSERVABLE9_TARGET_MODE_LIST)
OBSERVABLE10_TARGET_MODE_LIST = [
    "observable10_target8_continuous_fixed_recovery_v2",
    "observable10_target8_continuous_fixed_recovery_v3",
    "observable10_target8_continuous_fixed_recovery_v4",
    "observable10_target8_continuous_fixed_recovery_v5",
    "observable10_target8_continuous_fixed_recovery_v6",
]
OBSERVABLE10_TARGET_MODES = set(OBSERVABLE10_TARGET_MODE_LIST)
NO_POLICY_JUMP_STATE_MODES = {
    "observable12_target8_discrete_macro_v5",
    "observable12_target8_discrete_macro_v6",
    "observable12_target8_discrete_trigger_v9_stage1",
    "observable12_target8_discrete_trigger_v9_stage2",
    "observable12_target8_discrete_recovery_v10_stage1",
    "observable12_target8_discrete_recovery_v10_stage2a",
    "observable12_target8_discrete_recovery_v10_stage2",
    "observable12_target8_discrete_recovery_v10_stage3",
    "observable12_target8_discrete_recovery_v10b9_stage1",
    "observable12_target8_discrete_recovery_v10b9_stage2a",
    "observable12_target8_discrete_recovery_v10b9_stage2",
    "observable12_target8_discrete_recovery_v10b9_stage3",
    "observable12_target8_macro_library_v11_stage1",
    "observable12_target8_macro_library_v11_stage2a",
    "observable12_target8_macro_library_v11_stage2",
    "observable12_target8_macro_library_v11_stage3",
    "observable12_target8_macro_library_v11ds_stage2",
    "observable12_target8_macro_library_v11dsl_stage2",
    "observable12_target8_macro_library_v11tm_stage2",
    "observable12_target8_macro_library_v11re_stage2",
    "observable12_target8_macro_library_v11rx_stage2",
    "observable12_target8_macro_library_v11rxc_stage2",
    "observable9_target8_continuous_fixed_recovery_v1",
    "observable10_target8_continuous_fixed_recovery_v2",
    "observable10_target8_continuous_fixed_recovery_v3",
    "observable10_target8_continuous_fixed_recovery_v4",
    "observable10_target8_continuous_fixed_recovery_v5",
    "observable10_target8_continuous_fixed_recovery_v6",
}


@dataclass(frozen=True)
class Experiment:
    name: str
    algo: str
    policy: str
    net_arch: Any
    state_mode: str
    lstm_hidden_size: int = 64
    n_lstm_layers: int = 1


BASE_EXPERIMENTS = [
    ("a2c_mlp_64x64", "A2C", "MlpPolicy", [64, 64]),
    ("ppo_mlp_64x64", "PPO", "MlpPolicy", [64, 64]),
    ("sac_mlp_32x32", "SAC", "MlpPolicy", [32, 32]),
]

EXPERIMENTS = [
    Experiment(
        name=f"{base}_{state_mode}",
        algo=algo,
        policy=policy,
        net_arch=net_arch,
        state_mode=state_mode,
    )
    for state_mode in ("full17", "reduced12", "deploy11", "observable10", "minimal9")
    for base, algo, policy, net_arch in BASE_EXPERIMENTS
]

EXPERIMENTS.append(
    Experiment(
        name="rppo_lstm64_observable10",
        algo="RecurrentPPO",
        policy="MlpLstmPolicy",
        net_arch={"pi": [32], "vf": [32]},
        state_mode="observable10",
    )
)

EXPERIMENTS.append(
    Experiment(
        name="rppo_lstm64_observable8",
        algo="RecurrentPPO",
        policy="MlpLstmPolicy",
        net_arch={"pi": [32], "vf": [32]},
        state_mode="observable8",
    )
)

EXPERIMENTS.append(
    Experiment(
        name="rppo_lstm64_observable6_delta90",
        algo="RecurrentPPO",
        policy="MlpLstmPolicy",
        net_arch={"pi": [32], "vf": [32]},
        state_mode="observable6",
    )
)

EXPERIMENTS.append(
    Experiment(
        name="rppo_lstm64_observable8_delta45",
        algo="RecurrentPPO",
        policy="MlpLstmPolicy",
        net_arch={"pi": [32], "vf": [32]},
        state_mode="observable8_delta45",
    )
)

EXPERIMENTS.append(
    Experiment(
        name="rppo_lstm256x2_observable8_target8_line_delta90",
        algo="RecurrentPPO",
        policy="MlpLstmPolicy",
        net_arch={"pi": [128, 128], "vf": [128, 128]},
        state_mode="observable8_target8_line_delta90",
        lstm_hidden_size=256,
        n_lstm_layers=2,
    )
)

EXPERIMENTS.append(
    Experiment(
        name="rppo_small_lstm128x1_observable9_target8_continuous_fixed_recovery_v1",
        algo="RecurrentPPO",
        policy="MlpLstmPolicy",
        net_arch={"pi": [64, 64], "vf": [64, 64]},
        state_mode="observable9_target8_continuous_fixed_recovery_v1",
        lstm_hidden_size=128,
        n_lstm_layers=1,
    )
)

EXPERIMENTS.append(
    Experiment(
        name="rppo_small_lstm128x1_observable10_target8_continuous_fixed_recovery_v2",
        algo="RecurrentPPO",
        policy="MlpLstmPolicy",
        net_arch={"pi": [64, 64], "vf": [64, 64]},
        state_mode="observable10_target8_continuous_fixed_recovery_v2",
        lstm_hidden_size=128,
        n_lstm_layers=1,
    )
)

EXPERIMENTS.append(
    Experiment(
        name="rppo_small_lstm128x1_observable10_target8_continuous_fixed_recovery_v3",
        algo="RecurrentPPO",
        policy="MlpLstmPolicy",
        net_arch={"pi": [64, 64], "vf": [64, 64]},
        state_mode="observable10_target8_continuous_fixed_recovery_v3",
        lstm_hidden_size=128,
        n_lstm_layers=1,
    )
)

EXPERIMENTS.append(
    Experiment(
        name="rppo_small_lstm128x1_observable10_target8_continuous_fixed_recovery_v4",
        algo="RecurrentPPO",
        policy="MlpLstmPolicy",
        net_arch={"pi": [64, 64], "vf": [64, 64]},
        state_mode="observable10_target8_continuous_fixed_recovery_v4",
        lstm_hidden_size=128,
        n_lstm_layers=1,
    )
)

EXPERIMENTS.append(
    Experiment(
        name="rppo_small_lstm128x1_observable10_target8_continuous_fixed_recovery_v5",
        algo="RecurrentPPO",
        policy="MlpLstmPolicy",
        net_arch={"pi": [64, 64], "vf": [64, 64]},
        state_mode="observable10_target8_continuous_fixed_recovery_v5",
        lstm_hidden_size=128,
        n_lstm_layers=1,
    )
)

EXPERIMENTS.append(
    Experiment(
        name="rppo_small_lstm128x1_observable10_target8_continuous_fixed_recovery_v6",
        algo="RecurrentPPO",
        policy="MlpLstmPolicy",
        net_arch={"pi": [64, 64], "vf": [64, 64]},
        state_mode="observable10_target8_continuous_fixed_recovery_v6",
        lstm_hidden_size=128,
        n_lstm_layers=1,
    )
)

TARGET8_V2_EXPERIMENTS = [
    Experiment(
        name="rppo_tiny_lstm64x1_observable8_target8_v2_offset60",
        algo="RecurrentPPO",
        policy="MlpLstmPolicy",
        net_arch={"pi": [64], "vf": [64]},
        state_mode="observable8_target8_v2_offset60",
        lstm_hidden_size=64,
        n_lstm_layers=1,
    ),
    Experiment(
        name="rppo_small_lstm128x1_observable8_target8_v2_offset60",
        algo="RecurrentPPO",
        policy="MlpLstmPolicy",
        net_arch={"pi": [64, 64], "vf": [64, 64]},
        state_mode="observable8_target8_v2_offset60",
        lstm_hidden_size=128,
        n_lstm_layers=1,
    ),
    Experiment(
        name="rppo_medium_lstm128x2_observable8_target8_v2_offset60",
        algo="RecurrentPPO",
        policy="MlpLstmPolicy",
        net_arch={"pi": [128, 128], "vf": [128, 128]},
        state_mode="observable8_target8_v2_offset60",
        lstm_hidden_size=128,
        n_lstm_layers=2,
    ),
    Experiment(
        name="rppo_current_lstm256x2_observable8_target8_v2_offset60",
        algo="RecurrentPPO",
        policy="MlpLstmPolicy",
        net_arch={"pi": [128, 128], "vf": [128, 128]},
        state_mode="observable8_target8_v2_offset60",
        lstm_hidden_size=256,
        n_lstm_layers=2,
    ),
    Experiment(
        name="rppo_large_lstm384x2_observable8_target8_v2_offset60",
        algo="RecurrentPPO",
        policy="MlpLstmPolicy",
        net_arch={"pi": [192, 192], "vf": [192, 192]},
        state_mode="observable8_target8_v2_offset60",
        lstm_hidden_size=384,
        n_lstm_layers=2,
    ),
]

EXPERIMENTS.extend(TARGET8_V2_EXPERIMENTS)

EXPERIMENTS.append(
    Experiment(
        name="rppo_tiny_lstm64x1_observable8_target8_recovery_angle",
        algo="RecurrentPPO",
        policy="MlpLstmPolicy",
        net_arch={"pi": [64], "vf": [64]},
        state_mode="observable8_target8_recovery_angle",
        lstm_hidden_size=64,
        n_lstm_layers=1,
    )
)

EXPERIMENTS.append(
    Experiment(
        name="rppo_tiny_lstm64x1_observable8_target8_recovery_pressure",
        algo="RecurrentPPO",
        policy="MlpLstmPolicy",
        net_arch={"pi": [64], "vf": [64]},
        state_mode="observable8_target8_recovery_pressure",
        lstm_hidden_size=64,
        n_lstm_layers=1,
    )
)

EXPERIMENTS.append(
    Experiment(
        name="rppo_tiny_lstm64x1_observable8_target8_full180_pressure",
        algo="RecurrentPPO",
        policy="MlpLstmPolicy",
        net_arch={"pi": [64], "vf": [64]},
        state_mode="observable8_target8_full180_pressure",
        lstm_hidden_size=64,
        n_lstm_layers=1,
    )
)

EXPERIMENTS.append(
    Experiment(
        name="rppo_tiny_lstm64x1_observable8_target8_macro_recovery_dense_halfmountain",
        algo="RecurrentPPO",
        policy="MlpLstmPolicy",
        net_arch={"pi": [64], "vf": [64]},
        state_mode="observable8_target8_macro_recovery_dense_halfmountain",
        lstm_hidden_size=64,
        n_lstm_layers=1,
    )
)

EXPERIMENTS.append(
    Experiment(
        name="rppo_current_lstm256x2_observable8_target8_macro_recovery_dense_halfmountain",
        algo="RecurrentPPO",
        policy="MlpLstmPolicy",
        net_arch={"pi": [128, 128], "vf": [128, 128]},
        state_mode="observable8_target8_macro_recovery_dense_halfmountain",
        lstm_hidden_size=256,
        n_lstm_layers=2,
    )
)

EXPERIMENTS.append(
    Experiment(
        name="rppo_tiny_lstm64x1_observable8_target8_macro_trigger_dense_halfmountain",
        algo="RecurrentPPO",
        policy="MlpLstmPolicy",
        net_arch={"pi": [64], "vf": [64]},
        state_mode="observable8_target8_macro_trigger_dense_halfmountain",
        lstm_hidden_size=64,
        n_lstm_layers=1,
    )
)

EXPERIMENTS.append(
    Experiment(
        name="rppo_medium_lstm128x2_observable8_target8_macro_trigger_dense_halfmountain",
        algo="RecurrentPPO",
        policy="MlpLstmPolicy",
        net_arch={"pi": [128, 128], "vf": [128, 128]},
        state_mode="observable8_target8_macro_trigger_dense_halfmountain",
        lstm_hidden_size=128,
        n_lstm_layers=2,
    )
)

EXPERIMENTS.append(
    Experiment(
        name="rppo_current_lstm256x2_observable8_target8_macro_trigger_dense_halfmountain",
        algo="RecurrentPPO",
        policy="MlpLstmPolicy",
        net_arch={"pi": [128, 128], "vf": [128, 128]},
        state_mode="observable8_target8_macro_trigger_dense_halfmountain",
        lstm_hidden_size=256,
        n_lstm_layers=2,
    )
)

EXPERIMENTS.append(
    Experiment(
        name="rppo_medium_lstm128x2_observable12_target8_recovery_v4",
        algo="RecurrentPPO",
        policy="MlpLstmPolicy",
        net_arch={"pi": [128, 128], "vf": [128, 128]},
        state_mode="observable12_target8_recovery_v4",
        lstm_hidden_size=128,
        n_lstm_layers=2,
    )
)

EXPERIMENTS.append(
    Experiment(
        name="rppo_medium_lstm128x2_observable12_target8_discrete_macro_v5",
        algo="RecurrentPPO",
        policy="MlpLstmPolicy",
        net_arch={"pi": [128, 128], "vf": [128, 128]},
        state_mode="observable12_target8_discrete_macro_v5",
        lstm_hidden_size=128,
        n_lstm_layers=2,
    )
)

EXPERIMENTS.append(
    Experiment(
        name="rppo_medium_lstm128x2_observable12_target8_discrete_macro_v6",
        algo="RecurrentPPO",
        policy="MlpLstmPolicy",
        net_arch={"pi": [128, 128], "vf": [128, 128]},
        state_mode="observable12_target8_discrete_macro_v6",
        lstm_hidden_size=128,
        n_lstm_layers=2,
    )
)

EXPERIMENTS.append(
    Experiment(
        name="rppo_medium_lstm128x2_observable12_target8_discrete_macro_v7",
        algo="RecurrentPPO",
        policy="MlpLstmPolicy",
        net_arch={"pi": [128, 128], "vf": [128, 128]},
        state_mode="observable12_target8_discrete_macro_v7",
        lstm_hidden_size=128,
        n_lstm_layers=2,
    )
)

EXPERIMENTS.append(
    Experiment(
        name="rppo_medium_lstm128x2_observable12_target8_discrete_macro_v8",
        algo="RecurrentPPO",
        policy="MlpLstmPolicy",
        net_arch={"pi": [128, 128], "vf": [128, 128]},
        state_mode="observable12_target8_discrete_macro_v8",
        lstm_hidden_size=128,
        n_lstm_layers=2,
    )
)

EXPERIMENTS.append(
    Experiment(
        name="rppo_medium_lstm128x2_observable12_target8_discrete_trigger_v9_stage1",
        algo="RecurrentPPO",
        policy="MlpLstmPolicy",
        net_arch={"pi": [128, 128], "vf": [128, 128]},
        state_mode="observable12_target8_discrete_trigger_v9_stage1",
        lstm_hidden_size=128,
        n_lstm_layers=2,
    )
)

EXPERIMENTS.append(
    Experiment(
        name="rppo_medium_lstm128x2_observable12_target8_discrete_trigger_v9_stage2",
        algo="RecurrentPPO",
        policy="MlpLstmPolicy",
        net_arch={"pi": [128, 128], "vf": [128, 128]},
        state_mode="observable12_target8_discrete_trigger_v9_stage2",
        lstm_hidden_size=128,
        n_lstm_layers=2,
    )
)

EXPERIMENTS.append(
    Experiment(
        name="rppo_medium_lstm128x2_observable12_target8_discrete_recovery_v10_stage1",
        algo="RecurrentPPO",
        policy="MlpLstmPolicy",
        net_arch={"pi": [128, 128], "vf": [128, 128]},
        state_mode="observable12_target8_discrete_recovery_v10_stage1",
        lstm_hidden_size=128,
        n_lstm_layers=2,
    )
)

EXPERIMENTS.append(
    Experiment(
        name="rppo_medium_lstm128x2_observable12_target8_discrete_recovery_v10_stage2a",
        algo="RecurrentPPO",
        policy="MlpLstmPolicy",
        net_arch={"pi": [128, 128], "vf": [128, 128]},
        state_mode="observable12_target8_discrete_recovery_v10_stage2a",
        lstm_hidden_size=128,
        n_lstm_layers=2,
    )
)

EXPERIMENTS.append(
    Experiment(
        name="rppo_medium_lstm128x2_observable12_target8_discrete_recovery_v10_stage2",
        algo="RecurrentPPO",
        policy="MlpLstmPolicy",
        net_arch={"pi": [128, 128], "vf": [128, 128]},
        state_mode="observable12_target8_discrete_recovery_v10_stage2",
        lstm_hidden_size=128,
        n_lstm_layers=2,
    )
)

EXPERIMENTS.append(
    Experiment(
        name="rppo_medium_lstm128x2_observable12_target8_discrete_recovery_v10_stage3",
        algo="RecurrentPPO",
        policy="MlpLstmPolicy",
        net_arch={"pi": [128, 128], "vf": [128, 128]},
        state_mode="observable12_target8_discrete_recovery_v10_stage3",
        lstm_hidden_size=128,
        n_lstm_layers=2,
    )
)

EXPERIMENTS.append(
    Experiment(
        name="rppo_medium_lstm128x2_observable12_target8_discrete_recovery_v10b9_stage1",
        algo="RecurrentPPO",
        policy="MlpLstmPolicy",
        net_arch={"pi": [128, 128], "vf": [128, 128]},
        state_mode="observable12_target8_discrete_recovery_v10b9_stage1",
        lstm_hidden_size=128,
        n_lstm_layers=2,
    )
)

EXPERIMENTS.append(
    Experiment(
        name="rppo_medium_lstm128x2_observable12_target8_discrete_recovery_v10b9_stage2a",
        algo="RecurrentPPO",
        policy="MlpLstmPolicy",
        net_arch={"pi": [128, 128], "vf": [128, 128]},
        state_mode="observable12_target8_discrete_recovery_v10b9_stage2a",
        lstm_hidden_size=128,
        n_lstm_layers=2,
    )
)

EXPERIMENTS.append(
    Experiment(
        name="rppo_medium_lstm128x2_observable12_target8_discrete_recovery_v10b9_stage2",
        algo="RecurrentPPO",
        policy="MlpLstmPolicy",
        net_arch={"pi": [128, 128], "vf": [128, 128]},
        state_mode="observable12_target8_discrete_recovery_v10b9_stage2",
        lstm_hidden_size=128,
        n_lstm_layers=2,
    )
)

EXPERIMENTS.append(
    Experiment(
        name="rppo_medium_lstm128x2_observable12_target8_discrete_recovery_v10b9_stage3",
        algo="RecurrentPPO",
        policy="MlpLstmPolicy",
        net_arch={"pi": [128, 128], "vf": [128, 128]},
        state_mode="observable12_target8_discrete_recovery_v10b9_stage3",
        lstm_hidden_size=128,
        n_lstm_layers=2,
    )
)

EXPERIMENTS.append(
    Experiment(
        name="rppo_medium_lstm128x2_observable12_target8_macro_library_v11_stage1",
        algo="RecurrentPPO",
        policy="MlpLstmPolicy",
        net_arch={"pi": [128, 128], "vf": [128, 128]},
        state_mode="observable12_target8_macro_library_v11_stage1",
        lstm_hidden_size=128,
        n_lstm_layers=2,
    )
)

EXPERIMENTS.append(
    Experiment(
        name="rppo_medium_lstm128x2_observable12_target8_macro_library_v11_stage2a",
        algo="RecurrentPPO",
        policy="MlpLstmPolicy",
        net_arch={"pi": [128, 128], "vf": [128, 128]},
        state_mode="observable12_target8_macro_library_v11_stage2a",
        lstm_hidden_size=128,
        n_lstm_layers=2,
    )
)

EXPERIMENTS.append(
    Experiment(
        name="rppo_medium_lstm128x2_observable12_target8_macro_library_v11_stage2",
        algo="RecurrentPPO",
        policy="MlpLstmPolicy",
        net_arch={"pi": [128, 128], "vf": [128, 128]},
        state_mode="observable12_target8_macro_library_v11_stage2",
        lstm_hidden_size=128,
        n_lstm_layers=2,
    )
)

EXPERIMENTS.append(
    Experiment(
        name="rppo_medium_lstm128x2_observable12_target8_macro_library_v11_stage3",
        algo="RecurrentPPO",
        policy="MlpLstmPolicy",
        net_arch={"pi": [128, 128], "vf": [128, 128]},
        state_mode="observable12_target8_macro_library_v11_stage3",
        lstm_hidden_size=128,
        n_lstm_layers=2,
    )
)

EXPERIMENTS.append(
    Experiment(
        name="rppo_medium_lstm128x2_observable12_target8_macro_library_v11ds_stage2",
        algo="RecurrentPPO",
        policy="MlpLstmPolicy",
        net_arch={"pi": [128, 128], "vf": [128, 128]},
        state_mode="observable12_target8_macro_library_v11ds_stage2",
        lstm_hidden_size=128,
        n_lstm_layers=2,
    )
)

EXPERIMENTS.append(
    Experiment(
        name="rppo_medium_lstm128x2_observable12_target8_macro_library_v11dsl_stage2",
        algo="RecurrentPPO",
        policy="MlpLstmPolicy",
        net_arch={"pi": [128, 128], "vf": [128, 128]},
        state_mode="observable12_target8_macro_library_v11dsl_stage2",
        lstm_hidden_size=128,
        n_lstm_layers=2,
    )
)

EXPERIMENTS.append(
    Experiment(
        name="rppo_medium_lstm128x2_observable12_target8_macro_library_v11tm_stage2",
        algo="RecurrentPPO",
        policy="MlpLstmPolicy",
        net_arch={"pi": [128, 128], "vf": [128, 128]},
        state_mode="observable12_target8_macro_library_v11tm_stage2",
        lstm_hidden_size=128,
        n_lstm_layers=2,
    )
)

EXPERIMENTS.append(
    Experiment(
        name="rppo_medium_lstm128x2_observable12_target8_macro_library_v11re_stage2",
        algo="RecurrentPPO",
        policy="MlpLstmPolicy",
        net_arch={"pi": [128, 128], "vf": [128, 128]},
        state_mode="observable12_target8_macro_library_v11re_stage2",
        lstm_hidden_size=128,
        n_lstm_layers=2,
    )
)

EXPERIMENTS.append(
    Experiment(
        name="rppo_medium_lstm128x2_observable12_target8_macro_library_v11rx_stage2",
        algo="RecurrentPPO",
        policy="MlpLstmPolicy",
        net_arch={"pi": [128, 128], "vf": [128, 128]},
        state_mode="observable12_target8_macro_library_v11rx_stage2",
        lstm_hidden_size=128,
        n_lstm_layers=2,
    )
)

EXPERIMENTS.append(
    Experiment(
        name="rppo_medium_lstm128x2_observable12_target8_macro_library_v11rxc_stage2",
        algo="RecurrentPPO",
        policy="MlpLstmPolicy",
        net_arch={"pi": [128, 128], "vf": [128, 128]},
        state_mode="observable12_target8_macro_library_v11rxc_stage2",
        lstm_hidden_size=128,
        n_lstm_layers=2,
    )
)

EXPERIMENTS.append(
    Experiment(
        name="rppo_lstm64_observable7_stop_delta45",
        algo="RecurrentPPO",
        policy="MlpLstmPolicy",
        net_arch={"pi": [32], "vf": [32]},
        state_mode="observable7_stop_delta45",
    )
)

EXPERIMENTS.append(
    Experiment(
        name="rppo_lstm256x2_observable7_stop_delta45",
        algo="RecurrentPPO",
        policy="MlpLstmPolicy",
        net_arch={"pi": [128, 128], "vf": [128, 128]},
        state_mode="observable7_stop_delta45",
        lstm_hidden_size=256,
        n_lstm_layers=2,
    )
)

EXPERIMENTS.append(
    Experiment(
        name="rppo_lstm256x2_observable7_stop_curriculum_delta45",
        algo="RecurrentPPO",
        policy="MlpLstmPolicy",
        net_arch={"pi": [128, 128], "vf": [128, 128]},
        state_mode="observable7_stop_curriculum_delta45",
        lstm_hidden_size=256,
        n_lstm_layers=2,
    )
)

EXPERIMENTS.append(
    Experiment(
        name="rppo_lstm256x2_observable7_stop_stage1_delta45",
        algo="RecurrentPPO",
        policy="MlpLstmPolicy",
        net_arch={"pi": [128, 128], "vf": [128, 128]},
        state_mode="observable7_stop_stage1_delta45",
        lstm_hidden_size=256,
        n_lstm_layers=2,
    )
)

EXPERIMENTS.append(
    Experiment(
        name="rppo_lstm256x2_observable7_stop_stage2_delta45",
        algo="RecurrentPPO",
        policy="MlpLstmPolicy",
        net_arch={"pi": [128, 128], "vf": [128, 128]},
        state_mode="observable7_stop_stage2_delta45",
        lstm_hidden_size=256,
        n_lstm_layers=2,
    )
)

EXPERIMENTS.append(
    Experiment(
        name="rppo_lstm256x2_observable7_stop_stage3_delta45",
        algo="RecurrentPPO",
        policy="MlpLstmPolicy",
        net_arch={"pi": [128, 128], "vf": [128, 128]},
        state_mode="observable7_stop_stage3_delta45",
        lstm_hidden_size=256,
        n_lstm_layers=2,
    )
)

for observable11_state_mode in OBSERVABLE11_STOP_MODE_LIST:
    EXPERIMENTS.append(
        Experiment(
            name=f"rppo_lstm256x2_{observable11_state_mode}",
            algo="RecurrentPPO",
            policy="MlpLstmPolicy",
            net_arch={"pi": [128, 128], "vf": [128, 128]},
            state_mode=observable11_state_mode,
            lstm_hidden_size=256,
            n_lstm_layers=2,
        )
    )


class SelectObsWrapper(gym.ObservationWrapper):
    def __init__(self, env: gym.Env, indices: list[int]) -> None:
        super().__init__(env)
        self.indices = np.asarray(indices, dtype=np.int64)
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(len(indices),),
            dtype=np.float32,
        )

    def observation(self, observation: np.ndarray) -> np.ndarray:
        return np.asarray(observation, dtype=np.float32)[self.indices]


class Observable10Wrapper(gym.Wrapper):
    """Deployment-style observation built from coordinates and past commands."""

    def __init__(self, env: BlindNavEnv) -> None:
        super().__init__(env)
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(OBSERVABLE10_DIM,), dtype=np.float32)
        self.prev_pos = np.zeros(2, dtype=np.float32)
        self.prev_distance = 0.0
        self.last_cmd_angle = 0.0
        self.last_action_speed = -1.0
        self.last_action_jump = 0.0
        self.last_distance_progress = 0.0
        self.last_movement = 0.0
        self.no_progress_time_est = 0.0
        self.low_movement_time_est = 0.0

    @property
    def base_env(self) -> BlindNavEnv:
        return self.env.unwrapped

    def reset(self, **kwargs):
        _, info = self.env.reset(**kwargs)
        base = self.base_env
        self.prev_pos = base.pos.copy()
        self.prev_distance = float(np.linalg.norm(base.target - base.pos))
        delta = base.target - base.pos
        self.last_cmd_angle = float(np.arctan2(delta[1], delta[0]))
        self.last_action_speed = -1.0
        self.last_action_jump = 0.0
        self.last_distance_progress = 0.0
        self.last_movement = 0.0
        self.no_progress_time_est = 0.0
        self.low_movement_time_est = 0.0
        return self._get_observable_obs(), info

    def step(self, action):
        base = self.base_env
        prev_pos = base.pos.copy()
        prev_distance = float(np.linalg.norm(base.target - base.pos))
        delta = base.target - base.pos
        target_angle = float(np.arctan2(delta[1], delta[0]))
        action_arr = np.asarray(action, dtype=np.float32)
        self.last_cmd_angle = target_angle + float(action_arr[0]) * np.pi
        self.last_action_speed = float(action_arr[1] * 2.0 - 1.0)
        self.last_action_jump = float(action_arr[2])

        _, reward, terminated, truncated, info = self.env.step(action)

        current_distance = float(np.linalg.norm(base.target - base.pos))
        self.last_distance_progress = prev_distance - current_distance
        self.last_movement = float(np.linalg.norm(base.pos - prev_pos))
        if self.last_distance_progress < 0.2:
            self.no_progress_time_est += base.dt
        else:
            self.no_progress_time_est = max(0.0, self.no_progress_time_est - base.dt * 2.0)
        if self.last_movement < 1.0:
            self.low_movement_time_est += base.dt
        else:
            self.low_movement_time_est = max(0.0, self.low_movement_time_est - base.dt * 2.0)

        self.prev_pos = base.pos.copy()
        self.prev_distance = current_distance
        return self._get_observable_obs(), reward, terminated, truncated, info

    def _get_observable_obs(self) -> np.ndarray:
        base = self.base_env
        delta = base.target - base.pos
        return np.array(
            [
                np.clip(delta[0] / base.world_size, -1.0, 1.0),
                np.clip(delta[1] / base.world_size, -1.0, 1.0),
                np.sin(self.last_cmd_angle),
                np.cos(self.last_cmd_angle),
                np.clip(self.last_action_speed, -1.0, 1.0),
                np.clip(self.last_distance_progress / 30.0, -1.0, 1.0),
                np.clip(self.last_movement / 40.0, 0.0, 1.0),
                np.clip(self.no_progress_time_est / 5.0, 0.0, 1.0),
                np.clip(self.low_movement_time_est / 5.0, 0.0, 1.0),
                np.clip(self.last_action_jump, -1.0, 1.0),
            ],
            dtype=np.float32,
        )


class Observable8Wrapper(Observable10Wrapper):
    """LSTM-friendly observation without hand-crafted time accumulators."""

    def __init__(self, env: BlindNavEnv) -> None:
        super().__init__(env)
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(OBSERVABLE8_DIM,), dtype=np.float32)

    def _get_observable_obs(self) -> np.ndarray:
        base = self.base_env
        delta = base.target - base.pos
        return np.array(
            [
                np.clip(delta[0] / base.world_size, -1.0, 1.0),
                np.clip(delta[1] / base.world_size, -1.0, 1.0),
                np.sin(self.last_cmd_angle),
                np.cos(self.last_cmd_angle),
                np.clip(self.last_action_speed, -1.0, 1.0),
                np.clip(self.last_distance_progress / 30.0, -1.0, 1.0),
                np.clip(self.last_movement / 40.0, 0.0, 1.0),
                np.clip(self.last_action_jump, -1.0, 1.0),
            ],
            dtype=np.float32,
        )


class Observable8DeltaWrapper(gym.Wrapper):
    """Delta-angle control observation with 8 compact features."""

    def __init__(self, env: BlindNavEnv) -> None:
        super().__init__(env)
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(OBSERVABLE8_DIM,), dtype=np.float32)
        self.prev_angle_error = 0.0
        self.distance_progress = 0.0
        self.movement_dist = 0.0
        self.delta_angle_error = 0.0
        self.last_action_speed = -1.0
        self.last_action_jump = 0.0

    @property
    def base_env(self) -> BlindNavEnv:
        return self.env.unwrapped

    def _angle_error(self) -> float:
        base = self.base_env
        delta = base.target - base.pos
        target_angle = float(np.arctan2(delta[1], delta[0]))
        return float(np.arctan2(np.sin(target_angle - base.heading), np.cos(target_angle - base.heading)))

    def reset(self, **kwargs):
        _, info = self.env.reset(**kwargs)
        self.prev_angle_error = self._angle_error()
        self.distance_progress = 0.0
        self.movement_dist = 0.0
        self.delta_angle_error = 0.0
        self.last_action_speed = -1.0
        self.last_action_jump = 0.0
        return self._get_obs(), info

    def step(self, action):
        base = self.base_env
        prev_pos = base.pos.copy()
        prev_distance = float(np.linalg.norm(base.target - base.pos))
        prev_angle_error = self._angle_error()
        action_arr = np.asarray(action, dtype=np.float32)
        self.last_action_speed = float(action_arr[1] * 2.0 - 1.0)
        self.last_action_jump = float(action_arr[2])
        _, reward, terminated, truncated, info = self.env.step(action)
        current_distance = float(np.linalg.norm(base.target - base.pos))
        angle_error = self._angle_error()
        self.distance_progress = prev_distance - current_distance
        self.movement_dist = float(np.linalg.norm(base.pos - prev_pos))
        self.delta_angle_error = prev_angle_error - angle_error
        self.prev_angle_error = angle_error
        return self._get_obs(), reward, terminated, truncated, info

    def _get_obs(self) -> np.ndarray:
        base = self.base_env
        delta = base.target - base.pos
        return np.array(
            [
                np.clip(delta[0] / base.world_size, -1.0, 1.0),
                np.clip(delta[1] / base.world_size, -1.0, 1.0),
                np.clip(self.prev_angle_error / np.pi, -1.0, 1.0),
                np.clip(self.delta_angle_error / np.pi, -1.0, 1.0),
                np.clip(self.distance_progress / 30.0, -1.0, 1.0),
                np.clip(self.movement_dist / 40.0, 0.0, 1.0),
                np.clip(self.last_action_speed, -1.0, 1.0),
                np.clip(self.last_action_jump, -1.0, 1.0),
            ],
            dtype=np.float32,
        )


class Observable7StopWrapper(gym.Wrapper):
    """Deployment-oriented 7-feature state with stale coordinates and live heading."""

    def __init__(self, env: BlindNavEnv) -> None:
        super().__init__(env)
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(OBSERVABLE7_DIM,), dtype=np.float32)
        self.position_obs_dt = 0.5
        self.coord_age_scale = 1.0
        self.stop_center = 14.0
        self.stop_half_range = 6.0
        self.observed_pos = np.zeros(2, dtype=np.float32)
        self.coord_age = 0.0
        self.last_action_speed = -1.0

    @property
    def base_env(self) -> BlindNavEnv:
        return self.env.unwrapped

    def reset(self, **kwargs):
        _, info = self.env.reset(**kwargs)
        base = self.base_env
        self._set_stop_scale()
        self.observed_pos = base.pos.copy()
        if base.coord_noise_std > 0.0:
            self.observed_pos = self._noisy_pos(self.observed_pos)
        self.coord_age = 0.0
        self._sample_position_obs_dt()
        self.last_action_speed = -1.0
        return self._get_obs(), info

    def _set_stop_scale(self) -> None:
        base = self.base_env
        if base.stop_distance_range is None:
            self.stop_center = max(float(base.target_radius), 1.0)
            self.stop_half_range = max(float(base.target_radius), 1.0)
            return
        low, high = base.stop_distance_range
        self.stop_center = (float(low) + float(high)) * 0.5
        self.stop_half_range = max((float(high) - float(low)) * 0.5, 1.0)

    def step(self, action):
        base = self.base_env
        action_arr = np.asarray(action, dtype=np.float32)
        self.last_action_speed = float(action_arr[1] * 2.0 - 1.0)
        _, reward, terminated, truncated, info = self.env.step(action)
        self.coord_age += base.dt
        if self.coord_age >= self.position_obs_dt:
            self.observed_pos = base.pos.copy()
            if base.coord_noise_std > 0.0:
                self.observed_pos = self._noisy_pos(self.observed_pos)
            self.coord_age = 0.0
            self._sample_position_obs_dt()
        return self._get_obs(), reward, terminated, truncated, info

    def _sample_position_obs_dt(self) -> None:
        base = self.base_env
        if base.position_obs_dt_range is None:
            self.position_obs_dt = 0.5
            self.coord_age_scale = 1.0
            return
        low, high = base.position_obs_dt_range
        self.position_obs_dt = float(base.np_random.uniform(float(low), float(high)))
        self.coord_age_scale = max(float(high), 1e-6)

    def _noisy_pos(self, pos: np.ndarray) -> np.ndarray:
        base = self.base_env
        noise = base.np_random.normal(0.0, base.coord_noise_std, size=2)
        return (pos + noise).astype(np.float32)

    def _get_obs(self) -> np.ndarray:
        base = self.base_env
        delta = base.target - self.observed_pos
        return np.array(
            [
                np.clip(delta[0] / base.world_size, -1.0, 1.0),
                np.clip(delta[1] / base.world_size, -1.0, 1.0),
                np.sin(base.heading),
                np.cos(base.heading),
                np.clip((base.desired_stop_distance - self.stop_center) / self.stop_half_range, -1.0, 1.0),
                np.clip(self.coord_age / self.coord_age_scale, 0.0, 1.0),
                np.clip(self.last_action_speed, -1.0, 1.0),
            ],
            dtype=np.float32,
        )


class Observable11StopWrapper(Observable7StopWrapper):
    """7D stop state plus deployable target alignment and progress hints."""

    def __init__(self, env: BlindNavEnv) -> None:
        super().__init__(env)
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(OBSERVABLE11_DIM,), dtype=np.float32)
        self.last_distance_progress = 0.0
        self.prev_observed_distance = 0.0

    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        base = self.base_env
        self.prev_observed_distance = float(np.linalg.norm(base.target - self.observed_pos))
        self.last_distance_progress = 0.0
        return self._get_obs(), info

    def step(self, action):
        base = self.base_env
        prev_distance = float(np.linalg.norm(base.target - self.observed_pos))
        action_arr = np.asarray(action, dtype=np.float32)
        self.last_action_speed = float(action_arr[1] * 2.0 - 1.0)
        _, reward, terminated, truncated, info = self.env.step(action)
        self.coord_age += base.dt
        if self.coord_age >= self.position_obs_dt:
            self.observed_pos = base.pos.copy()
            if base.coord_noise_std > 0.0:
                self.observed_pos = self._noisy_pos(self.observed_pos)
            self.coord_age = 0.0
            self._sample_position_obs_dt()
        current_distance = float(np.linalg.norm(base.target - self.observed_pos))
        self.last_distance_progress = prev_distance - current_distance
        self.prev_observed_distance = current_distance
        return self._get_obs(), reward, terminated, truncated, info

    def _get_obs(self) -> np.ndarray:
        base = self.base_env
        delta = base.target - self.observed_pos
        target_angle = float(np.arctan2(delta[1], delta[0]))
        angle_error = float(np.arctan2(np.sin(target_angle - base.heading), np.cos(target_angle - base.heading)))
        distance = float(np.linalg.norm(delta))
        stop_error = distance - float(base.desired_stop_distance)
        return np.array(
            [
                np.clip(delta[0] / base.world_size, -1.0, 1.0),
                np.clip(delta[1] / base.world_size, -1.0, 1.0),
                np.sin(base.heading),
                np.cos(base.heading),
                np.clip((base.desired_stop_distance - self.stop_center) / self.stop_half_range, -1.0, 1.0),
                np.clip(self.coord_age / self.coord_age_scale, 0.0, 1.0),
                np.clip(self.last_action_speed, -1.0, 1.0),
                np.sin(angle_error),
                np.cos(angle_error),
                np.clip(stop_error / 160.0, -1.0, 1.0),
                np.clip(self.last_distance_progress / 30.0, -1.0, 1.0),
            ],
            dtype=np.float32,
        )


class Observable8TargetWrapper(gym.Wrapper):
    """8D target-circle navigation state from stale coordinates and live heading."""

    def __init__(self, env: BlindNavEnv) -> None:
        super().__init__(env)
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(OBSERVABLE8_TARGET_DIM,), dtype=np.float32)
        self.position_obs_dt = 0.5
        self.coord_age_scale = 1.0
        self.observed_pos = np.zeros(2, dtype=np.float32)
        self.coord_age = 0.0
        self.prev_observed_distance = 0.0
        self.last_distance_progress = 0.0

    @property
    def base_env(self) -> BlindNavEnv:
        return self.env.unwrapped

    def reset(self, **kwargs):
        _, info = self.env.reset(**kwargs)
        base = self.base_env
        self.observed_pos = base.pos.copy()
        if base.coord_noise_std > 0.0:
            self.observed_pos = self._noisy_pos(self.observed_pos)
        self.coord_age = 0.0
        self._sample_position_obs_dt()
        self.prev_observed_distance = float(np.linalg.norm(base.target - self.observed_pos))
        self.last_distance_progress = 0.0
        return self._get_obs(), info

    def step(self, action):
        base = self.base_env
        prev_distance = float(np.linalg.norm(base.target - self.observed_pos))
        _, reward, terminated, truncated, info = self.env.step(action)
        self.coord_age += base.dt
        if self.coord_age >= self.position_obs_dt:
            self.observed_pos = base.pos.copy()
            if base.coord_noise_std > 0.0:
                self.observed_pos = self._noisy_pos(self.observed_pos)
            self.coord_age = 0.0
            self._sample_position_obs_dt()
        current_distance = float(np.linalg.norm(base.target - self.observed_pos))
        self.last_distance_progress = prev_distance - current_distance
        self.prev_observed_distance = current_distance
        return self._get_obs(), reward, terminated, truncated, info

    def _sample_position_obs_dt(self) -> None:
        base = self.base_env
        if base.position_obs_dt_range is None:
            self.position_obs_dt = 0.5
            self.coord_age_scale = 1.0
            return
        low, high = base.position_obs_dt_range
        self.position_obs_dt = float(base.np_random.uniform(float(low), float(high)))
        self.coord_age_scale = max(float(high), 1e-6)

    def _noisy_pos(self, pos: np.ndarray) -> np.ndarray:
        base = self.base_env
        noise = base.np_random.normal(0.0, base.coord_noise_std, size=2)
        return (pos + noise).astype(np.float32)

    def _get_obs(self) -> np.ndarray:
        base = self.base_env
        delta = base.target - self.observed_pos
        target_angle = float(np.arctan2(delta[1], delta[0]))
        angle_error = float(np.arctan2(np.sin(target_angle - base.heading), np.cos(target_angle - base.heading)))
        return np.array(
            [
                np.clip(delta[0] / base.world_size, -1.0, 1.0),
                np.clip(delta[1] / base.world_size, -1.0, 1.0),
                np.sin(base.heading),
                np.cos(base.heading),
                np.sin(angle_error),
                np.cos(angle_error),
                np.clip(self.coord_age / self.coord_age_scale, 0.0, 1.0),
                np.clip(self.last_distance_progress / 30.0, -1.0, 1.0),
            ],
            dtype=np.float32,
        )


class Observable12TargetRecoveryWrapper(Observable8TargetWrapper):
    """12D target-circle state with deployable progress history, no obstacle geometry."""

    def __init__(self, env: BlindNavEnv) -> None:
        super().__init__(env)
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(OBSERVABLE12_TARGET_DIM,), dtype=np.float32)
        self.last_action_angle = 0.0
        self.last_recovery_mode = 0.0

    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        self.last_action_angle = 0.0
        self.last_recovery_mode = 0.0
        return self._get_obs(), info

    def step(self, action):
        base = self.base_env
        action_arr = np.asarray(action, dtype=np.float32).reshape(-1)
        env_action = self._map_v4_recovery_action(action)
        _, reward, terminated, truncated, info = super().step(env_action)
        self.last_action_angle = float(np.clip(action_arr[0], -1.0, 1.0))
        self.last_recovery_mode = float(int(info.get("recovery_mode", 0) or 0)) / 2.0
        return self._get_obs(), reward, terminated, truncated, info

    @staticmethod
    def _map_v4_recovery_action(action) -> np.ndarray:
        env_action = np.asarray(action, dtype=np.float32).copy()
        recovery_signal = float(env_action[3])
        if recovery_signal < -0.5:
            env_action[3] = 0.25  # normal
        elif recovery_signal < 0.0:
            env_action[3] = -0.25  # back
        elif recovery_signal < 0.5:
            env_action[3] = -0.75  # back-left
        else:
            env_action[3] = 0.75  # back-right
        return env_action

    def _get_obs(self) -> np.ndarray:
        base = self.base_env
        delta = base.target - self.observed_pos
        target_angle = float(np.arctan2(delta[1], delta[0]))
        angle_error = float(np.arctan2(np.sin(target_angle - base.heading), np.cos(target_angle - base.heading)))
        distance = float(np.linalg.norm(delta))
        return np.array(
            [
                np.clip(delta[0] / base.world_size, -1.0, 1.0),
                np.clip(delta[1] / base.world_size, -1.0, 1.0),
                np.sin(base.heading),
                np.cos(base.heading),
                np.sin(angle_error),
                np.cos(angle_error),
                np.clip(self.coord_age / self.coord_age_scale, 0.0, 1.0),
                np.clip(self.last_distance_progress / 30.0, -1.0, 1.0),
                np.clip(base.no_progress_time / 3.0, 0.0, 1.0),
                np.clip(distance / base.world_size, 0.0, 1.0),
                np.clip(self.last_recovery_mode, -1.0, 1.0),
                np.clip(self.last_action_angle, -1.0, 1.0),
            ],
            dtype=np.float32,
        )


class Observable9ContinuousFixedRecoveryWrapper(Observable8TargetWrapper):
    """9D blind state for continuous steering with env-owned fixed recovery."""

    def __init__(self, env: BlindNavEnv) -> None:
        super().__init__(env)
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(OBSERVABLE9_TARGET_DIM,), dtype=np.float32)

    def reset(self, **kwargs):
        _, info = super().reset(**kwargs)
        return self._get_obs(), info

    def step(self, action):
        base = self.base_env
        prev_distance = float(np.linalg.norm(base.target - self.observed_pos))
        self.coord_age += base.dt
        action_arr = np.asarray(action, dtype=np.float32).reshape(-1)
        if action_arr.size != 4:
            raise ValueError(f"expected 4 continuous action values, got shape {action_arr.shape}")
        env_action = action_arr.copy()
        env_action[2] = -1.0
        _, reward, terminated, truncated, info = self.env.step(env_action)
        true_distance = float(np.linalg.norm(base.target - base.pos))
        if self.coord_age >= self.position_obs_dt or terminated or truncated:
            self.observed_pos = base.pos.copy()
            if base.coord_noise_std > 0.0:
                self.observed_pos = self._noisy_pos(self.observed_pos)
            self.coord_age = 0.0
            self._sample_position_obs_dt()
            self.last_distance_progress = prev_distance - true_distance
        else:
            self.last_distance_progress = 0.0
        return self._get_obs(), reward, terminated, truncated, info

    def _get_obs(self) -> np.ndarray:
        base = self.base_env
        delta = base.target - self.observed_pos
        target_angle = float(np.arctan2(delta[1], delta[0]))
        angle_error = float(np.arctan2(np.sin(target_angle - base.heading), np.cos(target_angle - base.heading)))
        return np.array(
            [
                np.clip(delta[0] / base.world_size, -1.0, 1.0),
                np.clip(delta[1] / base.world_size, -1.0, 1.0),
                np.sin(base.heading),
                np.cos(base.heading),
                np.sin(angle_error),
                np.cos(angle_error),
                np.clip(self.coord_age / self.coord_age_scale, 0.0, 1.0),
                np.clip(self.last_distance_progress / 30.0, -1.0, 1.0),
                np.clip(base.no_progress_time / 3.0, 0.0, 1.0),
            ],
            dtype=np.float32,
        )


class Observable10ContinuousFixedRecoveryWrapper(Observable9ContinuousFixedRecoveryWrapper):
    """10D blind state with recent collision signal for trigger learning."""

    def __init__(self, env: BlindNavEnv) -> None:
        super().__init__(env)
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(OBSERVABLE10_TARGET_DIM,), dtype=np.float32)

    def _get_obs(self) -> np.ndarray:
        obs = super()._get_obs()
        base = self.base_env
        recent_collision_norm = 1.0 - np.clip(base.time_since_collision / 1.5, 0.0, 1.0)
        return np.concatenate([obs, np.array([recent_collision_norm], dtype=np.float32)], axis=0)


class Observable12DiscreteMacroWrapper(Observable12TargetRecoveryWrapper):
    """12D state with a discrete macro-action interface for recovery exploration."""

    ANGLE_OFFSETS = np.asarray([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=np.float32)
    SPEED_VALUES = np.asarray([0.0, 1.0], dtype=np.float32)
    RECOVERY_SIGNALS = np.asarray([0.25, -0.25, -0.75, 0.75, 0.0], dtype=np.float32)

    def __init__(self, env: BlindNavEnv) -> None:
        super().__init__(env)
        # [move_mode, angle_offset_bin, speed_bin]
        # move_mode: 0 normal, 1 back, 2 back-left, 3 back-right, 4 normal fine adjust.
        self.action_space = spaces.MultiDiscrete([5, 5, 2])
        self.last_move_mode = 0.0
        self.macro_phase_norm = 0.0

    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        self.last_move_mode = 0.0
        self.macro_phase_norm = 0.0
        return self._get_obs(), info

    def step(self, action):
        base = self.base_env
        prev_distance = float(np.linalg.norm(base.target - self.observed_pos))
        self.coord_age += base.dt
        discrete = np.asarray(action, dtype=np.int64).reshape(-1)
        if discrete.size != 3:
            raise ValueError(f"expected 3 discrete action values, got shape {discrete.shape}")
        move_mode = int(np.clip(discrete[0], 0, 4))
        angle_bin = int(np.clip(discrete[1], 0, 4))
        speed_bin = int(np.clip(discrete[2], 0, 1))
        angle_offset = float(self.ANGLE_OFFSETS[angle_bin])
        if move_mode == 4:
            angle_offset *= 0.5
        env_action = np.array(
            [
                angle_offset,
                float(self.SPEED_VALUES[speed_bin]),
                -1.0,
                float(self.RECOVERY_SIGNALS[move_mode]),
            ],
            dtype=np.float32,
        )
        _, reward, terminated, truncated, info = self.env.step(env_action)
        true_distance = float(np.linalg.norm(base.target - base.pos))
        if self.coord_age >= self.position_obs_dt or terminated or truncated:
            self.observed_pos = base.pos.copy()
            if base.coord_noise_std > 0.0:
                self.observed_pos = self._noisy_pos(self.observed_pos)
            self.coord_age = 0.0
            self._sample_position_obs_dt()
            self.last_distance_progress = prev_distance - true_distance
        else:
            self.last_distance_progress = 0.0
        self.last_action_angle = angle_offset
        self.last_recovery_mode = float(int(info.get("recovery_mode", 0) or 0)) / 2.0
        self.last_move_mode = (float(move_mode) / 2.0) - 1.0
        self.macro_phase_norm = np.clip(
            float(info.get("macro_recovery_steps_left", 0) or 0) / max(float(base.macro_recovery_steps), 1.0),
            0.0,
            1.0,
        )
        return self._get_obs(), reward, terminated, truncated, info

    def _get_obs(self) -> np.ndarray:
        obs = super()._get_obs().copy()
        obs[10] = np.clip(self.last_move_mode, -1.0, 1.0)
        obs[11] = np.clip(self.macro_phase_norm, -1.0, 1.0)
        return obs


class Observable12DiscreteMacroV7Wrapper(Observable12TargetRecoveryWrapper):
    """12D state with tighter steering bins and heading-relative two-phase recovery."""

    ANGLE_OFFSETS = np.asarray(
        [
            -1.0,
            -0.33333334,
            0.0,
            0.33333334,
            1.0,
        ],
        dtype=np.float32,
    )
    SPEED_VALUES = np.asarray([0.0, 1.0], dtype=np.float32)
    RECOVERY_SIGNALS = np.asarray([0.25, -0.25, -0.75, 0.75, 0.0], dtype=np.float32)

    def __init__(self, env: BlindNavEnv) -> None:
        super().__init__(env)
        self.action_space = spaces.MultiDiscrete([5, 5, 2])
        self.last_move_mode = 0.0
        self.macro_phase_norm = 0.0

    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        self.last_move_mode = 0.0
        self.macro_phase_norm = 0.0
        return self._get_obs(), info

    def step(self, action):
        base = self.base_env
        prev_distance = float(np.linalg.norm(base.target - self.observed_pos))
        self.coord_age += base.dt
        discrete = np.asarray(action, dtype=np.int64).reshape(-1)
        if discrete.size != 3:
            raise ValueError(f"expected 3 discrete action values, got shape {discrete.shape}")
        move_mode = int(np.clip(discrete[0], 0, 4))
        angle_bin = int(np.clip(discrete[1], 0, 4))
        speed_bin = int(np.clip(discrete[2], 0, 1))
        angle_offset = float(self.ANGLE_OFFSETS[angle_bin])
        if move_mode == 4:
            angle_offset *= 0.5
        env_action = np.array(
            [
                angle_offset,
                float(self.SPEED_VALUES[speed_bin]),
                -1.0,
                float(self.RECOVERY_SIGNALS[move_mode]),
            ],
            dtype=np.float32,
        )
        _, reward, terminated, truncated, info = self.env.step(env_action)
        true_distance = float(np.linalg.norm(base.target - base.pos))
        if self.coord_age >= self.position_obs_dt or terminated or truncated:
            self.observed_pos = base.pos.copy()
            if base.coord_noise_std > 0.0:
                self.observed_pos = self._noisy_pos(self.observed_pos)
            self.coord_age = 0.0
            self._sample_position_obs_dt()
            self.last_distance_progress = prev_distance - true_distance
        else:
            self.last_distance_progress = 0.0
        self.last_action_angle = angle_offset
        self.last_recovery_mode = float(int(info.get("recovery_mode", 0) or 0)) / 2.0
        self.last_move_mode = (float(move_mode) / 2.0) - 1.0
        self.macro_phase_norm = np.clip(
            float(info.get("macro_recovery_steps_left", 0) or 0) / max(float(base.macro_recovery_steps), 1.0),
            0.0,
            1.0,
        )
        return self._get_obs(), reward, terminated, truncated, info

    def _get_obs(self) -> np.ndarray:
        obs = super()._get_obs().copy()
        obs[10] = np.clip(self.last_move_mode, -1.0, 1.0)
        obs[11] = np.clip(self.macro_phase_norm, -1.0, 1.0)
        return obs


class Observable12DiscreteTriggerV9Wrapper(Observable12TargetRecoveryWrapper):
    """12D blind state with no-policy-jump discrete trigger recovery actions."""

    ANGLE_OFFSETS = np.asarray(
        [
            -1.0,
            -0.33333334,
            0.0,
            0.33333334,
            1.0,
        ],
        dtype=np.float32,
    )
    SPEED_VALUES = np.asarray([0.0, 1.0], dtype=np.float32)
    TRIGGER_VALUES = np.asarray([-1.0, 1.0], dtype=np.float32)

    def __init__(self, env: BlindNavEnv) -> None:
        super().__init__(env)
        # [direction_bin, speed_bin, recovery_trigger_bin]
        self.action_space = spaces.MultiDiscrete([5, 2, 2])
        self.last_move_mode = 0.0
        self.macro_phase_norm = 0.0

    def reset(self, **kwargs):
        _, info = super().reset(**kwargs)
        self.last_move_mode = 0.0
        self.macro_phase_norm = 0.0
        return self._get_obs(), info

    def step(self, action):
        base = self.base_env
        prev_distance = float(np.linalg.norm(base.target - self.observed_pos))
        self.coord_age += base.dt
        discrete = np.asarray(action, dtype=np.int64).reshape(-1)
        if discrete.size != 3:
            raise ValueError(f"expected 3 discrete action values, got shape {discrete.shape}")
        angle_bin = int(np.clip(discrete[0], 0, 4))
        speed_bin = int(np.clip(discrete[1], 0, 1))
        trigger_bin = int(np.clip(discrete[2], 0, 1))
        angle_offset = float(self.ANGLE_OFFSETS[angle_bin])
        trigger_signal = float(self.TRIGGER_VALUES[trigger_bin])
        env_action = np.array(
            [
                angle_offset,
                float(self.SPEED_VALUES[speed_bin]),
                -1.0,
                trigger_signal,
            ],
            dtype=np.float32,
        )
        _, reward, terminated, truncated, info = self.env.step(env_action)
        true_distance = float(np.linalg.norm(base.target - base.pos))
        if self.coord_age >= self.position_obs_dt or terminated or truncated:
            self.observed_pos = base.pos.copy()
            if base.coord_noise_std > 0.0:
                self.observed_pos = self._noisy_pos(self.observed_pos)
            self.coord_age = 0.0
            self._sample_position_obs_dt()
            self.last_distance_progress = prev_distance - true_distance
        else:
            self.last_distance_progress = 0.0
        self.last_action_angle = angle_offset
        self.last_recovery_mode = float(int(info.get("recovery_mode", 0) or 0)) / 2.0
        self.last_move_mode = 1.0 if int(info.get("recovery_mode", 0) or 0) != 0 else 0.0
        self.macro_phase_norm = np.clip(
            float(info.get("macro_recovery_steps_left", 0) or 0) / max(float(base.macro_recovery_steps), 1.0),
            0.0,
            1.0,
        )
        return self._get_obs(), reward, terminated, truncated, info

    def _get_obs(self) -> np.ndarray:
        obs = super()._get_obs().copy()
        obs[10] = np.clip(self.last_move_mode, -1.0, 1.0)
        obs[11] = np.clip(self.macro_phase_norm, -1.0, 1.0)
        return obs


class Observable12DiscreteRecoveryV10Wrapper(Observable12TargetRecoveryWrapper):
    """12D blind state with pre-trigger stuck signals and explicit recovery mode bins."""

    ANGLE_OFFSETS = np.asarray(
        [
            -1.0,
            -0.33333334,
            0.0,
            0.33333334,
            1.0,
        ],
        dtype=np.float32,
    )
    SPEED_VALUES = np.asarray([0.0, 1.0], dtype=np.float32)
    RECOVERY_SIGNALS = np.asarray([0.25, -0.75, -0.25, 0.75], dtype=np.float32)

    def __init__(
        self,
        env: BlindNavEnv,
        *,
        recovery_enabled: bool = True,
        selective_recovery: bool = False,
        selective_no_progress_threshold: float = 1.10,
        selective_collision_threshold: float = 0.90,
        force_recovery_high_speed: bool = False,
    ) -> None:
        super().__init__(env)
        # [direction_bin, speed_bin, recovery_mode_bin]
        # recovery_mode_bin: 0 none, 1 back-left, 2 back, 3 back-right
        self.action_space = spaces.MultiDiscrete([5, 2, 4])
        self.recovery_enabled = bool(recovery_enabled)
        self.selective_recovery = bool(selective_recovery)
        self.selective_no_progress_threshold = float(selective_no_progress_threshold)
        self.selective_collision_threshold = float(selective_collision_threshold)
        self.force_recovery_high_speed = bool(force_recovery_high_speed)

    def reset(self, **kwargs):
        _, info = super().reset(**kwargs)
        return self._get_obs(), info

    def step(self, action):
        base = self.base_env
        prev_distance = float(np.linalg.norm(base.target - self.observed_pos))
        self.coord_age += base.dt
        discrete = np.asarray(action, dtype=np.int64).reshape(-1)
        if discrete.size != 3:
            raise ValueError(f"expected 3 discrete action values, got shape {discrete.shape}")
        angle_bin = int(np.clip(discrete[0], 0, 4))
        speed_bin = int(np.clip(discrete[1], 0, 1))
        recovery_bin = int(np.clip(discrete[2], 0, 3))
        angle_offset = float(self.ANGLE_OFFSETS[angle_bin])
        recovery_signal = 0.25 if not self.recovery_enabled else float(self.RECOVERY_SIGNALS[recovery_bin])
        if self.selective_recovery and recovery_signal != 0.25:
            recent_collision_norm = 1.0 - np.clip(base.time_since_collision / 3.0, 0.0, 1.0)
            allow_recovery = (
                base.no_progress_time >= self.selective_no_progress_threshold
                or recent_collision_norm >= self.selective_collision_threshold
            )
            if not allow_recovery:
                recovery_signal = 0.25
        speed_value = float(self.SPEED_VALUES[speed_bin])
        if self.force_recovery_high_speed and recovery_signal != 0.25:
            speed_value = 1.0
        env_action = np.array(
            [
                angle_offset,
                speed_value,
                -1.0,
                recovery_signal,
            ],
            dtype=np.float32,
        )
        _, reward, terminated, truncated, info = self.env.step(env_action)
        true_distance = float(np.linalg.norm(base.target - base.pos))
        if self.coord_age >= self.position_obs_dt or terminated or truncated:
            self.observed_pos = base.pos.copy()
            if base.coord_noise_std > 0.0:
                self.observed_pos = self._noisy_pos(self.observed_pos)
            self.coord_age = 0.0
            self._sample_position_obs_dt()
            self.last_distance_progress = prev_distance - true_distance
        else:
            self.last_distance_progress = 0.0
        self.last_action_angle = angle_offset
        self.last_recovery_mode = float(int(info.get("recovery_mode", 0) or 0)) / 2.0
        return self._get_obs(), reward, terminated, truncated, info

    def _get_obs(self) -> np.ndarray:
        base = self.base_env
        delta = base.target - self.observed_pos
        target_angle = float(np.arctan2(delta[1], delta[0]))
        angle_error = float(np.arctan2(np.sin(target_angle - base.heading), np.cos(target_angle - base.heading)))
        distance = float(np.linalg.norm(delta))
        recent_collision_norm = 1.0 - np.clip(base.time_since_collision / 3.0, 0.0, 1.0)
        return np.array(
            [
                np.clip(delta[0] / base.world_size, -1.0, 1.0),
                np.clip(delta[1] / base.world_size, -1.0, 1.0),
                np.sin(base.heading),
                np.cos(base.heading),
                np.sin(angle_error),
                np.cos(angle_error),
                np.clip(self.coord_age / self.coord_age_scale, 0.0, 1.0),
                np.clip(self.last_distance_progress / 30.0, -1.0, 1.0),
                np.clip(base.no_progress_time / 3.0, 0.0, 1.0),
                np.clip(distance / base.world_size, 0.0, 1.0),
                np.clip(recent_collision_norm, 0.0, 1.0),
                np.clip(base.stuck_time / 3.0, 0.0, 1.0),
            ],
            dtype=np.float32,
        )


class Observable12DiscreteRecoveryV10B9Wrapper(Observable12DiscreteRecoveryV10Wrapper):
    """V10 recovery interface with 9 direction bins instead of 5."""

    ANGLE_OFFSETS = np.asarray(
        [
            -1.0,
            -0.75,
            -0.5,
            -0.25,
            0.0,
            0.25,
            0.5,
            0.75,
            1.0,
        ],
        dtype=np.float32,
    )

    def __init__(self, env: BlindNavEnv, **kwargs) -> None:
        super().__init__(env, **kwargs)
        self.action_space = spaces.MultiDiscrete([9, 2, 4])

    def step(self, action):
        base = self.base_env
        prev_distance = float(np.linalg.norm(base.target - self.observed_pos))
        self.coord_age += base.dt
        discrete = np.asarray(action, dtype=np.int64).reshape(-1)
        if discrete.size != 3:
            raise ValueError(f"expected 3 discrete action values, got shape {discrete.shape}")
        angle_bin = int(np.clip(discrete[0], 0, 8))
        speed_bin = int(np.clip(discrete[1], 0, 1))
        recovery_bin = int(np.clip(discrete[2], 0, 3))
        angle_offset = float(self.ANGLE_OFFSETS[angle_bin])
        recovery_signal = 0.25 if not self.recovery_enabled else float(self.RECOVERY_SIGNALS[recovery_bin])
        if self.selective_recovery and recovery_signal != 0.25:
            recent_collision_norm = 1.0 - np.clip(base.time_since_collision / 3.0, 0.0, 1.0)
            allow_recovery = (
                base.no_progress_time >= self.selective_no_progress_threshold
                or recent_collision_norm >= self.selective_collision_threshold
            )
            if not allow_recovery:
                recovery_signal = 0.25
        speed_value = float(self.SPEED_VALUES[speed_bin])
        if self.force_recovery_high_speed and recovery_signal != 0.25:
            speed_value = 1.0
        env_action = np.array(
            [
                angle_offset,
                speed_value,
                -1.0,
                recovery_signal,
            ],
            dtype=np.float32,
        )
        _, reward, terminated, truncated, info = self.env.step(env_action)
        true_distance = float(np.linalg.norm(base.target - base.pos))
        if self.coord_age >= self.position_obs_dt or terminated or truncated:
            self.observed_pos = base.pos.copy()
            if base.coord_noise_std > 0.0:
                self.observed_pos = self._noisy_pos(self.observed_pos)
            self.coord_age = 0.0
            self._sample_position_obs_dt()
            self.last_distance_progress = prev_distance - true_distance
        else:
            self.last_distance_progress = 0.0
        self.last_action_angle = angle_offset
        self.last_recovery_mode = float(int(info.get("recovery_mode", 0) or 0)) / 2.0
        return self._get_obs(), reward, terminated, truncated, info


class Observable12MacroLibraryV11Wrapper(Observable12DiscreteRecoveryV10Wrapper):
    """V11 recovery interface with a finite library of escape macro templates."""

    MACRO_SIGNALS = np.asarray(
        [
            -1.0,
            -0.6666667,
            -0.33333334,
            0.0,
            0.33333334,
            0.6666667,
            1.0,
        ],
        dtype=np.float32,
    )

    def __init__(self, env: BlindNavEnv, **kwargs) -> None:
        stuck_macro_explore_prob = kwargs.pop("stuck_macro_explore_prob", 0.0)
        stuck_macro_no_progress_threshold = kwargs.pop("stuck_macro_no_progress_threshold", 0.55)
        stuck_macro_collision_threshold = kwargs.pop("stuck_macro_collision_threshold", 0.82)
        stuck_macro_min_distance = kwargs.pop("stuck_macro_min_distance", None)
        stuck_macro_max_distance = kwargs.pop("stuck_macro_max_distance", None)
        stuck_macro_mid_distance_min = kwargs.pop("stuck_macro_mid_distance_min", None)
        stuck_macro_mid_distance_max = kwargs.pop("stuck_macro_mid_distance_max", None)
        stuck_macro_mid_max_triggers = kwargs.pop("stuck_macro_mid_max_triggers", None)
        stuck_macro_mid_max_stuck_time = kwargs.pop("stuck_macro_mid_max_stuck_time", None)
        stuck_macro_mid_choices = kwargs.pop("stuck_macro_mid_choices", None)
        stuck_macro_mid_weights = kwargs.pop("stuck_macro_mid_weights", None)
        stuck_macro_choices = kwargs.pop("stuck_macro_choices", (1, 2, 3, 4, 5, 6))
        stuck_macro_weights = kwargs.pop("stuck_macro_weights", None)
        policy_macro_requires_release = kwargs.pop("policy_macro_requires_release", False)
        policy_macro_release_gate_bins = tuple(int(v) for v in kwargs.pop("policy_macro_release_gate_bins", ()))
        policy_macro_rearm_on_clear = kwargs.pop("policy_macro_rearm_on_clear", False)
        policy_macro_rearm_clear_time_threshold = kwargs.pop("policy_macro_rearm_clear_time_threshold", 1.5)
        policy_macro_rearm_clear_no_progress_threshold = kwargs.pop("policy_macro_rearm_clear_no_progress_threshold", 0.25)
        policy_macro_rearm_clear_stuck_threshold = kwargs.pop("policy_macro_rearm_clear_stuck_threshold", 0.25)
        policy_macro_rearm_max_count = kwargs.pop("policy_macro_rearm_max_count", None)
        policy_macro_rearm_min_distance = kwargs.pop("policy_macro_rearm_min_distance", None)
        policy_macro_rearm_max_distance = kwargs.pop("policy_macro_rearm_max_distance", None)
        targeted_stuck_macro_explore_prob = kwargs.pop("targeted_stuck_macro_explore_prob", 0.0)
        targeted_stuck_macro_no_progress_threshold = kwargs.pop(
            "targeted_stuck_macro_no_progress_threshold",
            stuck_macro_no_progress_threshold,
        )
        targeted_stuck_macro_collision_threshold = kwargs.pop(
            "targeted_stuck_macro_collision_threshold",
            stuck_macro_collision_threshold,
        )
        targeted_stuck_macro_angle_error_threshold = kwargs.pop("targeted_stuck_macro_angle_error_threshold", None)
        targeted_stuck_macro_distance_threshold = kwargs.pop("targeted_stuck_macro_distance_threshold", None)
        targeted_stuck_macro_max_distance_threshold = kwargs.pop("targeted_stuck_macro_max_distance_threshold", None)
        targeted_stuck_macro_stuck_time_threshold = kwargs.pop("targeted_stuck_macro_stuck_time_threshold", None)
        targeted_stuck_macro_mid_trigger_count_threshold = kwargs.pop(
            "targeted_stuck_macro_mid_trigger_count_threshold",
            None,
        )
        targeted_stuck_macro_allow_after_suppressed_policy = kwargs.pop(
            "targeted_stuck_macro_allow_after_suppressed_policy",
            False,
        )
        targeted_stuck_macro_choices = kwargs.pop("targeted_stuck_macro_choices", ())
        targeted_stuck_macro_weights = kwargs.pop("targeted_stuck_macro_weights", None)
        suppressed_policy_macro_override_prob = kwargs.pop("suppressed_policy_macro_override_prob", 0.0)
        suppressed_policy_macro_distance_threshold = kwargs.pop("suppressed_policy_macro_distance_threshold", None)
        suppressed_policy_macro_max_distance_threshold = kwargs.pop("suppressed_policy_macro_max_distance_threshold", None)
        suppressed_policy_macro_stuck_time_threshold = kwargs.pop("suppressed_policy_macro_stuck_time_threshold", None)
        suppressed_policy_macro_mid_trigger_count_threshold = kwargs.pop(
            "suppressed_policy_macro_mid_trigger_count_threshold",
            None,
        )
        suppressed_policy_macro_choices = kwargs.pop("suppressed_policy_macro_choices", ())
        suppressed_policy_macro_weights = kwargs.pop("suppressed_policy_macro_weights", None)
        suppressed_policy_generic_redirect_prob = kwargs.pop("suppressed_policy_generic_redirect_prob", 0.0)
        suppressed_policy_generic_redirect_distance_threshold = kwargs.pop(
            "suppressed_policy_generic_redirect_distance_threshold",
            None,
        )
        suppressed_policy_generic_redirect_max_distance_threshold = kwargs.pop(
            "suppressed_policy_generic_redirect_max_distance_threshold",
            None,
        )
        suppressed_policy_generic_redirect_stuck_time_threshold = kwargs.pop(
            "suppressed_policy_generic_redirect_stuck_time_threshold",
            None,
        )
        suppressed_policy_generic_redirect_mid_trigger_count_threshold = kwargs.pop(
            "suppressed_policy_generic_redirect_mid_trigger_count_threshold",
            None,
        )
        suppressed_policy_generic_redirect_choices = kwargs.pop("suppressed_policy_generic_redirect_choices", ())
        suppressed_policy_generic_redirect_weights = kwargs.pop("suppressed_policy_generic_redirect_weights", None)
        suppressed_policy_macro_passthrough_prob = kwargs.pop("suppressed_policy_macro_passthrough_prob", 0.0)
        suppressed_policy_macro_passthrough_bins = kwargs.pop("suppressed_policy_macro_passthrough_bins", ())
        suppressed_policy_macro_passthrough_distance_threshold = kwargs.pop(
            "suppressed_policy_macro_passthrough_distance_threshold",
            None,
        )
        suppressed_policy_macro_passthrough_max_distance_threshold = kwargs.pop(
            "suppressed_policy_macro_passthrough_max_distance_threshold",
            None,
        )
        suppressed_policy_macro_passthrough_stuck_time_threshold = kwargs.pop(
            "suppressed_policy_macro_passthrough_stuck_time_threshold",
            None,
        )
        suppressed_policy_macro_passthrough_mid_trigger_count_threshold = kwargs.pop(
            "suppressed_policy_macro_passthrough_mid_trigger_count_threshold",
            None,
        )
        super().__init__(env, **kwargs)
        # [direction_bin, speed_bin, macro_id]
        # macro_id: 0 normal, 1 short_back, 2 back_left, 3 back_right,
        # 4 wide_left, 5 wide_right, 6 back_then_realign.
        self.action_space = spaces.MultiDiscrete([5, 2, 7])
        self.stuck_macro_explore_prob = float(np.clip(stuck_macro_explore_prob, 0.0, 1.0))
        self.stuck_macro_no_progress_threshold = float(stuck_macro_no_progress_threshold)
        self.stuck_macro_collision_threshold = float(stuck_macro_collision_threshold)
        self.stuck_macro_min_distance = None if stuck_macro_min_distance is None else float(stuck_macro_min_distance)
        self.stuck_macro_max_distance = None if stuck_macro_max_distance is None else float(stuck_macro_max_distance)
        self.stuck_macro_mid_distance_min = (
            None if stuck_macro_mid_distance_min is None else float(stuck_macro_mid_distance_min)
        )
        self.stuck_macro_mid_distance_max = (
            None if stuck_macro_mid_distance_max is None else float(stuck_macro_mid_distance_max)
        )
        self.stuck_macro_mid_max_triggers = (
            None if stuck_macro_mid_max_triggers is None else int(stuck_macro_mid_max_triggers)
        )
        self.stuck_macro_mid_max_stuck_time = (
            None if stuck_macro_mid_max_stuck_time is None else float(stuck_macro_mid_max_stuck_time)
        )
        self.stuck_macro_choices = tuple(int(v) for v in stuck_macro_choices)
        if stuck_macro_weights is None:
            self.stuck_macro_weights = None
        else:
            weights = np.asarray(stuck_macro_weights, dtype=np.float32).reshape(-1)
            if len(weights) != len(self.stuck_macro_choices):
                raise ValueError("stuck_macro_weights length must match stuck_macro_choices")
            weights = np.clip(weights, 0.0, None)
            total = float(weights.sum())
            self.stuck_macro_weights = None if total <= 0.0 else (weights / total)
        if stuck_macro_mid_choices is None:
            self.stuck_macro_mid_choices = None
        else:
            self.stuck_macro_mid_choices = tuple(int(v) for v in stuck_macro_mid_choices)
        if stuck_macro_mid_weights is None:
            self.stuck_macro_mid_weights = None
        else:
            if self.stuck_macro_mid_choices is None:
                raise ValueError("stuck_macro_mid_weights requires stuck_macro_mid_choices")
            mid_weights = np.asarray(stuck_macro_mid_weights, dtype=np.float32).reshape(-1)
            if len(mid_weights) != len(self.stuck_macro_mid_choices):
                raise ValueError("stuck_macro_mid_weights length must match stuck_macro_mid_choices")
            mid_weights = np.clip(mid_weights, 0.0, None)
            mid_total = float(mid_weights.sum())
            self.stuck_macro_mid_weights = None if mid_total <= 0.0 else (mid_weights / mid_total)
        self.targeted_stuck_macro_explore_prob = float(np.clip(targeted_stuck_macro_explore_prob, 0.0, 1.0))
        self.targeted_stuck_macro_no_progress_threshold = float(targeted_stuck_macro_no_progress_threshold)
        self.targeted_stuck_macro_collision_threshold = float(targeted_stuck_macro_collision_threshold)
        self.targeted_stuck_macro_angle_error_threshold = (
            None
            if targeted_stuck_macro_angle_error_threshold is None
            else float(targeted_stuck_macro_angle_error_threshold)
        )
        self.targeted_stuck_macro_distance_threshold = (
            None if targeted_stuck_macro_distance_threshold is None else float(targeted_stuck_macro_distance_threshold)
        )
        self.targeted_stuck_macro_max_distance_threshold = (
            None
            if targeted_stuck_macro_max_distance_threshold is None
            else float(targeted_stuck_macro_max_distance_threshold)
        )
        self.targeted_stuck_macro_stuck_time_threshold = (
            None if targeted_stuck_macro_stuck_time_threshold is None else float(targeted_stuck_macro_stuck_time_threshold)
        )
        self.targeted_stuck_macro_mid_trigger_count_threshold = (
            None
            if targeted_stuck_macro_mid_trigger_count_threshold is None
            else int(targeted_stuck_macro_mid_trigger_count_threshold)
        )
        self.targeted_stuck_macro_allow_after_suppressed_policy = bool(
            targeted_stuck_macro_allow_after_suppressed_policy
        )
        self.targeted_stuck_macro_choices = tuple(int(v) for v in targeted_stuck_macro_choices)
        if targeted_stuck_macro_weights is None:
            self.targeted_stuck_macro_weights = None
        else:
            targeted_weights = np.asarray(targeted_stuck_macro_weights, dtype=np.float32).reshape(-1)
            if len(targeted_weights) != len(self.targeted_stuck_macro_choices):
                raise ValueError(
                    "targeted_stuck_macro_weights length must match targeted_stuck_macro_choices"
                )
            targeted_weights = np.clip(targeted_weights, 0.0, None)
            targeted_total = float(targeted_weights.sum())
            self.targeted_stuck_macro_weights = None if targeted_total <= 0.0 else (targeted_weights / targeted_total)
        self.suppressed_policy_macro_override_prob = float(np.clip(suppressed_policy_macro_override_prob, 0.0, 1.0))
        self.suppressed_policy_macro_distance_threshold = (
            None
            if suppressed_policy_macro_distance_threshold is None
            else float(suppressed_policy_macro_distance_threshold)
        )
        self.suppressed_policy_macro_max_distance_threshold = (
            None
            if suppressed_policy_macro_max_distance_threshold is None
            else float(suppressed_policy_macro_max_distance_threshold)
        )
        self.suppressed_policy_macro_stuck_time_threshold = (
            None
            if suppressed_policy_macro_stuck_time_threshold is None
            else float(suppressed_policy_macro_stuck_time_threshold)
        )
        self.suppressed_policy_macro_mid_trigger_count_threshold = (
            None
            if suppressed_policy_macro_mid_trigger_count_threshold is None
            else int(suppressed_policy_macro_mid_trigger_count_threshold)
        )
        self.suppressed_policy_macro_choices = tuple(int(v) for v in suppressed_policy_macro_choices)
        if suppressed_policy_macro_weights is None:
            self.suppressed_policy_macro_weights = None
        else:
            override_weights = np.asarray(suppressed_policy_macro_weights, dtype=np.float32).reshape(-1)
            if len(override_weights) != len(self.suppressed_policy_macro_choices):
                raise ValueError(
                    "suppressed_policy_macro_weights length must match suppressed_policy_macro_choices"
                )
            override_weights = np.clip(override_weights, 0.0, None)
            override_total = float(override_weights.sum())
            self.suppressed_policy_macro_weights = None if override_total <= 0.0 else (override_weights / override_total)
        self.suppressed_policy_generic_redirect_prob = float(
            np.clip(suppressed_policy_generic_redirect_prob, 0.0, 1.0)
        )
        self.suppressed_policy_generic_redirect_distance_threshold = (
            None
            if suppressed_policy_generic_redirect_distance_threshold is None
            else float(suppressed_policy_generic_redirect_distance_threshold)
        )
        self.suppressed_policy_generic_redirect_max_distance_threshold = (
            None
            if suppressed_policy_generic_redirect_max_distance_threshold is None
            else float(suppressed_policy_generic_redirect_max_distance_threshold)
        )
        self.suppressed_policy_generic_redirect_stuck_time_threshold = (
            None
            if suppressed_policy_generic_redirect_stuck_time_threshold is None
            else float(suppressed_policy_generic_redirect_stuck_time_threshold)
        )
        self.suppressed_policy_generic_redirect_mid_trigger_count_threshold = (
            None
            if suppressed_policy_generic_redirect_mid_trigger_count_threshold is None
            else int(suppressed_policy_generic_redirect_mid_trigger_count_threshold)
        )
        self.suppressed_policy_generic_redirect_choices = tuple(
            int(v) for v in suppressed_policy_generic_redirect_choices
        )
        if suppressed_policy_generic_redirect_weights is None:
            self.suppressed_policy_generic_redirect_weights = None
        else:
            redirect_weights = np.asarray(suppressed_policy_generic_redirect_weights, dtype=np.float32).reshape(-1)
            if len(redirect_weights) != len(self.suppressed_policy_generic_redirect_choices):
                raise ValueError(
                    "suppressed_policy_generic_redirect_weights length must match "
                    "suppressed_policy_generic_redirect_choices"
                )
            redirect_weights = np.clip(redirect_weights, 0.0, None)
            redirect_total = float(redirect_weights.sum())
            self.suppressed_policy_generic_redirect_weights = (
                None if redirect_total <= 0.0 else (redirect_weights / redirect_total)
            )
        self.suppressed_policy_macro_passthrough_prob = float(
            np.clip(suppressed_policy_macro_passthrough_prob, 0.0, 1.0)
        )
        self.suppressed_policy_macro_passthrough_bins = tuple(
            int(v) for v in suppressed_policy_macro_passthrough_bins
        )
        self.suppressed_policy_macro_passthrough_distance_threshold = (
            None
            if suppressed_policy_macro_passthrough_distance_threshold is None
            else float(suppressed_policy_macro_passthrough_distance_threshold)
        )
        self.suppressed_policy_macro_passthrough_max_distance_threshold = (
            None
            if suppressed_policy_macro_passthrough_max_distance_threshold is None
            else float(suppressed_policy_macro_passthrough_max_distance_threshold)
        )
        self.suppressed_policy_macro_passthrough_stuck_time_threshold = (
            None
            if suppressed_policy_macro_passthrough_stuck_time_threshold is None
            else float(suppressed_policy_macro_passthrough_stuck_time_threshold)
        )
        self.suppressed_policy_macro_passthrough_mid_trigger_count_threshold = (
            None
            if suppressed_policy_macro_passthrough_mid_trigger_count_threshold is None
            else int(suppressed_policy_macro_passthrough_mid_trigger_count_threshold)
        )
        self.policy_macro_requires_release = bool(policy_macro_requires_release)
        self.policy_macro_release_gate_bins = tuple(int(v) for v in policy_macro_release_gate_bins)
        self.policy_macro_rearm_on_clear = bool(policy_macro_rearm_on_clear)
        self.policy_macro_rearm_clear_time_threshold = float(policy_macro_rearm_clear_time_threshold)
        self.policy_macro_rearm_clear_no_progress_threshold = float(policy_macro_rearm_clear_no_progress_threshold)
        self.policy_macro_rearm_clear_stuck_threshold = float(policy_macro_rearm_clear_stuck_threshold)
        self.policy_macro_rearm_max_count = (
            None if policy_macro_rearm_max_count is None else int(policy_macro_rearm_max_count)
        )
        self.policy_macro_rearm_min_distance = (
            None if policy_macro_rearm_min_distance is None else float(policy_macro_rearm_min_distance)
        )
        self.policy_macro_rearm_max_distance = (
            None if policy_macro_rearm_max_distance is None else float(policy_macro_rearm_max_distance)
        )
        self.policy_macro_armed = True
        self.policy_macro_rearm_count = 0
        self.policy_macro_rearm_pending = False
        self.stuck_macro_mid_trigger_count = 0
        self.last_policy_macro_bin = 0

    def reset(self, **kwargs):
        _, info = super().reset(**kwargs)
        self.policy_macro_armed = True
        self.policy_macro_rearm_count = 0
        self.policy_macro_rearm_pending = False
        self.stuck_macro_mid_trigger_count = 0
        self.last_policy_macro_bin = 0
        return self._get_obs(), info

    def _sample_macro_choice(self, base: BlindNavEnv, macro_choices: tuple[int, ...], macro_weights):
        choices = np.asarray(macro_choices, dtype=np.int64)
        return int(base.np_random.choice(choices, p=macro_weights))

    def _policy_macro_clear_rearm(self, base: BlindNavEnv) -> bool:
        distance = float(np.linalg.norm(base.target - self.observed_pos))
        distance_ok = True
        if self.policy_macro_rearm_min_distance is not None:
            distance_ok = distance >= self.policy_macro_rearm_min_distance
        if self.policy_macro_rearm_max_distance is not None:
            distance_ok = distance_ok and distance <= self.policy_macro_rearm_max_distance
        return (
            self.policy_macro_rearm_on_clear
            and (
                self.policy_macro_rearm_max_count is None
                or self.policy_macro_rearm_count < self.policy_macro_rearm_max_count
            )
            and distance_ok
            and base.time_since_collision >= self.policy_macro_rearm_clear_time_threshold
            and base.no_progress_time <= self.policy_macro_rearm_clear_no_progress_threshold
            and base.stuck_time <= self.policy_macro_rearm_clear_stuck_threshold
        )

    def _targeted_stuck_macro_context(
        self,
        base: BlindNavEnv,
        *,
        recent_collision_norm: float,
        distance: float,
        angle_error_abs: float,
    ) -> bool:
        context = (
            base.no_progress_time >= self.targeted_stuck_macro_no_progress_threshold
            or recent_collision_norm >= self.targeted_stuck_macro_collision_threshold
        )
        if not context:
            return False
        if (
            self.targeted_stuck_macro_angle_error_threshold is not None
            and angle_error_abs < self.targeted_stuck_macro_angle_error_threshold
        ):
            return False
        if self.targeted_stuck_macro_distance_threshold is not None and distance < self.targeted_stuck_macro_distance_threshold:
            return False
        if (
            self.targeted_stuck_macro_max_distance_threshold is not None
            and distance > self.targeted_stuck_macro_max_distance_threshold
        ):
            return False
        if (
            self.targeted_stuck_macro_stuck_time_threshold is not None
            and base.stuck_time < self.targeted_stuck_macro_stuck_time_threshold
        ):
            return False
        if (
            self.targeted_stuck_macro_mid_trigger_count_threshold is not None
            and self.stuck_macro_mid_trigger_count < self.targeted_stuck_macro_mid_trigger_count_threshold
        ):
            return False
        return True

    def _generic_stuck_macro_allowed(self, distance: float, stuck_time: float) -> bool:
        if self.stuck_macro_min_distance is not None and distance < self.stuck_macro_min_distance:
            return False
        if self.stuck_macro_max_distance is not None and distance > self.stuck_macro_max_distance:
            return False
        if self.stuck_macro_mid_max_triggers is not None:
            in_mid_band = True
            if self.stuck_macro_mid_distance_min is not None and distance < self.stuck_macro_mid_distance_min:
                in_mid_band = False
            if self.stuck_macro_mid_distance_max is not None and distance > self.stuck_macro_mid_distance_max:
                in_mid_band = False
            if in_mid_band and self.stuck_macro_mid_trigger_count >= self.stuck_macro_mid_max_triggers:
                return False
        if self.stuck_macro_mid_max_stuck_time is not None and self._in_mid_stuck_macro_band(distance):
            if stuck_time > self.stuck_macro_mid_max_stuck_time:
                return False
        return True

    def _suppressed_policy_macro_override_context(self, distance: float, stuck_time: float) -> bool:
        if self.suppressed_policy_macro_override_prob <= 0.0 or not self.suppressed_policy_macro_choices:
            return False
        if (
            self.suppressed_policy_macro_distance_threshold is not None
            and distance < self.suppressed_policy_macro_distance_threshold
        ):
            return False
        if (
            self.suppressed_policy_macro_max_distance_threshold is not None
            and distance > self.suppressed_policy_macro_max_distance_threshold
        ):
            return False
        if (
            self.suppressed_policy_macro_stuck_time_threshold is not None
            and stuck_time < self.suppressed_policy_macro_stuck_time_threshold
        ):
            return False
        if (
            self.suppressed_policy_macro_mid_trigger_count_threshold is not None
            and self.stuck_macro_mid_trigger_count < self.suppressed_policy_macro_mid_trigger_count_threshold
        ):
            return False
        return True

    def _suppressed_policy_generic_redirect_context(self, distance: float, stuck_time: float) -> bool:
        if self.suppressed_policy_generic_redirect_prob <= 0.0 or not self.suppressed_policy_generic_redirect_choices:
            return False
        if (
            self.suppressed_policy_generic_redirect_distance_threshold is not None
            and distance < self.suppressed_policy_generic_redirect_distance_threshold
        ):
            return False
        if (
            self.suppressed_policy_generic_redirect_max_distance_threshold is not None
            and distance > self.suppressed_policy_generic_redirect_max_distance_threshold
        ):
            return False
        if (
            self.suppressed_policy_generic_redirect_stuck_time_threshold is not None
            and stuck_time < self.suppressed_policy_generic_redirect_stuck_time_threshold
        ):
            return False
        if (
            self.suppressed_policy_generic_redirect_mid_trigger_count_threshold is not None
            and self.stuck_macro_mid_trigger_count
            < self.suppressed_policy_generic_redirect_mid_trigger_count_threshold
        ):
            return False
        return True

    def _suppressed_policy_macro_passthrough_context(
        self,
        requested_macro_bin: int,
        distance: float,
        stuck_time: float,
    ) -> bool:
        if self.suppressed_policy_macro_passthrough_prob <= 0.0:
            return False
        if (
            self.suppressed_policy_macro_passthrough_bins
            and requested_macro_bin not in self.suppressed_policy_macro_passthrough_bins
        ):
            return False
        if (
            self.suppressed_policy_macro_passthrough_distance_threshold is not None
            and distance < self.suppressed_policy_macro_passthrough_distance_threshold
        ):
            return False
        if (
            self.suppressed_policy_macro_passthrough_max_distance_threshold is not None
            and distance > self.suppressed_policy_macro_passthrough_max_distance_threshold
        ):
            return False
        if (
            self.suppressed_policy_macro_passthrough_stuck_time_threshold is not None
            and stuck_time < self.suppressed_policy_macro_passthrough_stuck_time_threshold
        ):
            return False
        if (
            self.suppressed_policy_macro_passthrough_mid_trigger_count_threshold is not None
            and self.stuck_macro_mid_trigger_count
            < self.suppressed_policy_macro_passthrough_mid_trigger_count_threshold
        ):
            return False
        return True

    def _in_mid_stuck_macro_band(self, distance: float) -> bool:
        if self.stuck_macro_mid_distance_min is not None and distance < self.stuck_macro_mid_distance_min:
            return False
        if self.stuck_macro_mid_distance_max is not None and distance > self.stuck_macro_mid_distance_max:
            return False
        return (
            self.stuck_macro_mid_distance_min is not None
            or self.stuck_macro_mid_distance_max is not None
        )

    def step(self, action):
        base = self.base_env
        prev_distance = float(np.linalg.norm(base.target - self.observed_pos))
        self.coord_age += base.dt
        discrete = np.asarray(action, dtype=np.int64).reshape(-1)
        if discrete.size != 3:
            raise ValueError(f"expected 3 discrete action values, got shape {discrete.shape}")
        angle_bin = int(np.clip(discrete[0], 0, 4))
        speed_bin = int(np.clip(discrete[1], 0, 1))
        macro_bin = int(np.clip(discrete[2], 0, 6))
        requested_macro_bin = macro_bin
        policy_macro_suppressed = False
        angle_offset = float(self.ANGLE_OFFSETS[angle_bin])
        recent_collision_norm = 1.0 - np.clip(base.time_since_collision / 3.0, 0.0, 1.0)
        delta = base.target - self.observed_pos
        distance = float(np.linalg.norm(delta))
        target_angle = float(np.arctan2(delta[1], delta[0]))
        angle_error = float(np.arctan2(np.sin(target_angle - base.heading), np.cos(target_angle - base.heading)))
        angle_error_abs = abs(angle_error)
        stuck_macro_context = (
            base.no_progress_time >= self.stuck_macro_no_progress_threshold
            or recent_collision_norm >= self.stuck_macro_collision_threshold
        )
        clear_rearm_triggered = False
        if requested_macro_bin == 0:
            self.policy_macro_armed = True
            self.policy_macro_rearm_pending = False
        elif self._policy_macro_clear_rearm(base):
            clear_rearm_triggered = not self.policy_macro_armed
            self.policy_macro_armed = True
            if clear_rearm_triggered:
                self.policy_macro_rearm_pending = True
        suppressed_policy_macro_passthrough_triggered = False
        if (
            self.recovery_enabled
            and requested_macro_bin != 0
            and self.policy_macro_requires_release
            and (
                not self.policy_macro_release_gate_bins
                or requested_macro_bin in self.policy_macro_release_gate_bins
            )
            and not self.policy_macro_armed
        ):
            if (
                self._suppressed_policy_macro_passthrough_context(
                    requested_macro_bin,
                    distance,
                    base.stuck_time,
                )
                and float(base.np_random.random()) < self.suppressed_policy_macro_passthrough_prob
            ):
                macro_bin = requested_macro_bin
                suppressed_policy_macro_passthrough_triggered = True
            else:
                macro_bin = 0
                policy_macro_suppressed = True
        targeted_macro_candidate = macro_bin == 0 or (
            policy_macro_suppressed and self.targeted_stuck_macro_allow_after_suppressed_policy
        )
        targeted_stuck_macro_context = (
            self.targeted_stuck_macro_explore_prob > 0.0
            and targeted_macro_candidate
            and self.targeted_stuck_macro_choices
            and self._targeted_stuck_macro_context(
                base,
                recent_collision_norm=recent_collision_norm,
                distance=distance,
                angle_error_abs=angle_error_abs,
            )
        )
        targeted_macro_triggered = False
        suppressed_override_triggered = False
        suppressed_generic_redirect_triggered = False
        generic_macro_triggered = False
        if (
            self.recovery_enabled
            and policy_macro_suppressed
            and self._suppressed_policy_macro_override_context(distance, base.stuck_time)
            and float(base.np_random.random()) < self.suppressed_policy_macro_override_prob
        ):
            macro_bin = self._sample_macro_choice(
                base,
                self.suppressed_policy_macro_choices,
                self.suppressed_policy_macro_weights,
            )
            suppressed_override_triggered = True
        if (
            self.recovery_enabled
            and not suppressed_override_triggered
            and targeted_stuck_macro_context
            and float(base.np_random.random()) < self.targeted_stuck_macro_explore_prob
        ):
            macro_bin = self._sample_macro_choice(base, self.targeted_stuck_macro_choices, self.targeted_stuck_macro_weights)
            targeted_macro_triggered = True
        if (
            self.recovery_enabled
            and not suppressed_override_triggered
            and macro_bin == 0
            and self.stuck_macro_explore_prob > 0.0
            and stuck_macro_context
            and self._generic_stuck_macro_allowed(distance, base.stuck_time)
            and self.stuck_macro_choices
            and float(base.np_random.random()) < self.stuck_macro_explore_prob
        ):
            if (
                policy_macro_suppressed
                and self._suppressed_policy_generic_redirect_context(distance, base.stuck_time)
                and float(base.np_random.random()) < self.suppressed_policy_generic_redirect_prob
            ):
                macro_bin = self._sample_macro_choice(
                    base,
                    self.suppressed_policy_generic_redirect_choices,
                    self.suppressed_policy_generic_redirect_weights,
                )
                suppressed_generic_redirect_triggered = True
            elif self._in_mid_stuck_macro_band(distance) and self.stuck_macro_mid_choices:
                macro_bin = self._sample_macro_choice(base, self.stuck_macro_mid_choices, self.stuck_macro_mid_weights)
            else:
                macro_bin = self._sample_macro_choice(base, self.stuck_macro_choices, self.stuck_macro_weights)
            generic_macro_triggered = True
        macro_signal = -1.0 if not self.recovery_enabled else float(self.MACRO_SIGNALS[macro_bin])
        if self.selective_recovery and macro_bin != 0:
            allow_recovery = (
                base.no_progress_time >= self.selective_no_progress_threshold
                or recent_collision_norm >= self.selective_collision_threshold
            )
            if not allow_recovery:
                macro_signal = -1.0
                macro_bin = 0
        speed_value = float(self.SPEED_VALUES[speed_bin])
        if self.force_recovery_high_speed and macro_bin != 0:
            speed_value = 1.0
        env_action = np.array(
            [
                angle_offset,
                speed_value,
                -1.0,
                macro_signal,
            ],
            dtype=np.float32,
        )
        _, reward, terminated, truncated, info = self.env.step(env_action)
        if (
            self.recovery_enabled
            and requested_macro_bin != 0
            and not policy_macro_suppressed
            and not suppressed_override_triggered
            and not targeted_macro_triggered
            and not generic_macro_triggered
            and macro_bin != 0
        ):
            if self.policy_macro_rearm_pending:
                self.policy_macro_rearm_count += 1
                self.policy_macro_rearm_pending = False
            self.policy_macro_armed = False
        if generic_macro_triggered:
            in_mid_band = True
            if self.stuck_macro_mid_distance_min is not None and distance < self.stuck_macro_mid_distance_min:
                in_mid_band = False
            if self.stuck_macro_mid_distance_max is not None and distance > self.stuck_macro_mid_distance_max:
                in_mid_band = False
            if in_mid_band:
                self.stuck_macro_mid_trigger_count += 1
        self.last_policy_macro_bin = requested_macro_bin
        info["policy_macro_bin"] = requested_macro_bin
        info["effective_macro_bin"] = macro_bin
        info["policy_macro_suppressed"] = bool(policy_macro_suppressed)
        info["policy_macro_armed"] = bool(self.policy_macro_armed)
        info["policy_macro_clear_rearm_triggered"] = bool(clear_rearm_triggered)
        info["policy_macro_rearm_count"] = int(self.policy_macro_rearm_count)
        info["policy_macro_rearm_pending"] = bool(self.policy_macro_rearm_pending)
        info["stuck_macro_mid_trigger_count"] = int(self.stuck_macro_mid_trigger_count)
        info["stuck_macro_context"] = bool(stuck_macro_context)
        info["targeted_stuck_macro_context"] = bool(targeted_stuck_macro_context)
        info["suppressed_policy_macro_passthrough_triggered"] = bool(
            suppressed_policy_macro_passthrough_triggered
        )
        info["suppressed_policy_macro_override_triggered"] = bool(suppressed_override_triggered)
        info["suppressed_policy_generic_redirect_triggered"] = bool(suppressed_generic_redirect_triggered)
        info["stuck_macro_triggered"] = bool(generic_macro_triggered)
        info["targeted_stuck_macro_triggered"] = bool(targeted_macro_triggered)
        info["wrapper_recent_collision_norm"] = float(recent_collision_norm)
        info["wrapper_angle_error"] = float(angle_error)
        info["wrapper_distance"] = float(distance)
        true_distance = float(np.linalg.norm(base.target - base.pos))
        if self.coord_age >= self.position_obs_dt or terminated or truncated:
            self.observed_pos = base.pos.copy()
            if base.coord_noise_std > 0.0:
                self.observed_pos = self._noisy_pos(self.observed_pos)
            self.coord_age = 0.0
            self._sample_position_obs_dt()
            self.last_distance_progress = prev_distance - true_distance
        else:
            self.last_distance_progress = 0.0
        self.last_action_angle = angle_offset
        self.last_recovery_mode = float(int(info.get("recovery_mode", 0) or 0)) / 6.0
        return self._get_obs(), reward, terminated, truncated, info


class Observable12MacroLibraryV11RxWrapper(Observable12MacroLibraryV11Wrapper):
    """V11 recovery library with one extra short left macro that returns to target earlier."""

    MACRO_SIGNALS = np.asarray(
        [
            -1.0,
            -0.71428573,
            -0.42857143,
            -0.14285715,
            0.14285715,
            0.42857143,
            0.71428573,
            1.0,
        ],
        dtype=np.float32,
    )

    def __init__(self, env: BlindNavEnv, **kwargs) -> None:
        stuck_macro_choices = kwargs.pop("stuck_macro_choices", (1, 2, 3, 4, 5, 6, 7))
        super().__init__(env, stuck_macro_choices=stuck_macro_choices, **kwargs)
        # [direction_bin, speed_bin, macro_id]
        # macro_id 7: short_wide_left_then_target
        self.action_space = spaces.MultiDiscrete([5, 2, 8])

    def step(self, action):
        base = self.base_env
        prev_distance = float(np.linalg.norm(base.target - self.observed_pos))
        self.coord_age += base.dt
        discrete = np.asarray(action, dtype=np.int64).reshape(-1)
        if discrete.size != 3:
            raise ValueError(f"expected 3 discrete action values, got shape {discrete.shape}")
        angle_bin = int(np.clip(discrete[0], 0, 4))
        speed_bin = int(np.clip(discrete[1], 0, 1))
        macro_bin = int(np.clip(discrete[2], 0, 7))
        requested_macro_bin = macro_bin
        policy_macro_suppressed = False
        angle_offset = float(self.ANGLE_OFFSETS[angle_bin])
        recent_collision_norm = 1.0 - np.clip(base.time_since_collision / 3.0, 0.0, 1.0)
        delta = base.target - self.observed_pos
        distance = float(np.linalg.norm(delta))
        target_angle = float(np.arctan2(delta[1], delta[0]))
        angle_error = float(np.arctan2(np.sin(target_angle - base.heading), np.cos(target_angle - base.heading)))
        angle_error_abs = abs(angle_error)
        stuck_macro_context = (
            base.no_progress_time >= self.stuck_macro_no_progress_threshold
            or recent_collision_norm >= self.stuck_macro_collision_threshold
        )
        clear_rearm_triggered = False
        if requested_macro_bin == 0:
            self.policy_macro_armed = True
            self.policy_macro_rearm_pending = False
        elif self._policy_macro_clear_rearm(base):
            clear_rearm_triggered = not self.policy_macro_armed
            self.policy_macro_armed = True
            if clear_rearm_triggered:
                self.policy_macro_rearm_pending = True
        suppressed_policy_macro_passthrough_triggered = False
        if (
            self.recovery_enabled
            and requested_macro_bin != 0
            and self.policy_macro_requires_release
            and (
                not self.policy_macro_release_gate_bins
                or requested_macro_bin in self.policy_macro_release_gate_bins
            )
            and not self.policy_macro_armed
        ):
            if (
                self._suppressed_policy_macro_passthrough_context(
                    requested_macro_bin,
                    distance,
                    base.stuck_time,
                )
                and float(base.np_random.random()) < self.suppressed_policy_macro_passthrough_prob
            ):
                macro_bin = requested_macro_bin
                suppressed_policy_macro_passthrough_triggered = True
            else:
                macro_bin = 0
                policy_macro_suppressed = True
        targeted_macro_candidate = macro_bin == 0 or (
            policy_macro_suppressed and self.targeted_stuck_macro_allow_after_suppressed_policy
        )
        targeted_stuck_macro_context = (
            self.targeted_stuck_macro_explore_prob > 0.0
            and targeted_macro_candidate
            and self.targeted_stuck_macro_choices
            and self._targeted_stuck_macro_context(
                base,
                recent_collision_norm=recent_collision_norm,
                distance=distance,
                angle_error_abs=angle_error_abs,
            )
        )
        targeted_macro_triggered = False
        suppressed_override_triggered = False
        suppressed_generic_redirect_triggered = False
        generic_macro_triggered = False
        if (
            self.recovery_enabled
            and policy_macro_suppressed
            and self._suppressed_policy_macro_override_context(distance, base.stuck_time)
            and float(base.np_random.random()) < self.suppressed_policy_macro_override_prob
        ):
            macro_bin = self._sample_macro_choice(
                base,
                self.suppressed_policy_macro_choices,
                self.suppressed_policy_macro_weights,
            )
            suppressed_override_triggered = True
        if (
            self.recovery_enabled
            and not suppressed_override_triggered
            and targeted_stuck_macro_context
            and float(base.np_random.random()) < self.targeted_stuck_macro_explore_prob
        ):
            macro_bin = self._sample_macro_choice(base, self.targeted_stuck_macro_choices, self.targeted_stuck_macro_weights)
            targeted_macro_triggered = True
        if (
            self.recovery_enabled
            and not suppressed_override_triggered
            and macro_bin == 0
            and self.stuck_macro_explore_prob > 0.0
            and stuck_macro_context
            and self._generic_stuck_macro_allowed(distance, base.stuck_time)
            and self.stuck_macro_choices
            and float(base.np_random.random()) < self.stuck_macro_explore_prob
        ):
            if (
                policy_macro_suppressed
                and self._suppressed_policy_generic_redirect_context(distance, base.stuck_time)
                and float(base.np_random.random()) < self.suppressed_policy_generic_redirect_prob
            ):
                macro_bin = self._sample_macro_choice(
                    base,
                    self.suppressed_policy_generic_redirect_choices,
                    self.suppressed_policy_generic_redirect_weights,
                )
                suppressed_generic_redirect_triggered = True
            elif self._in_mid_stuck_macro_band(distance) and self.stuck_macro_mid_choices:
                macro_bin = self._sample_macro_choice(base, self.stuck_macro_mid_choices, self.stuck_macro_mid_weights)
            else:
                macro_bin = self._sample_macro_choice(base, self.stuck_macro_choices, self.stuck_macro_weights)
            generic_macro_triggered = True
        macro_signal = -1.0 if not self.recovery_enabled else float(self.MACRO_SIGNALS[macro_bin])
        if self.selective_recovery and macro_bin != 0:
            allow_recovery = (
                base.no_progress_time >= self.selective_no_progress_threshold
                or recent_collision_norm >= self.selective_collision_threshold
            )
            if not allow_recovery:
                macro_signal = -1.0
                macro_bin = 0
        speed_value = float(self.SPEED_VALUES[speed_bin])
        if self.force_recovery_high_speed and macro_bin != 0:
            speed_value = 1.0
        env_action = np.array([angle_offset, speed_value, -1.0, macro_signal], dtype=np.float32)
        _, reward, terminated, truncated, info = self.env.step(env_action)
        if (
            self.recovery_enabled
            and requested_macro_bin != 0
            and not policy_macro_suppressed
            and not suppressed_override_triggered
            and not targeted_macro_triggered
            and not generic_macro_triggered
            and macro_bin != 0
        ):
            if self.policy_macro_rearm_pending:
                self.policy_macro_rearm_count += 1
                self.policy_macro_rearm_pending = False
            self.policy_macro_armed = False
        if generic_macro_triggered:
            in_mid_band = True
            if self.stuck_macro_mid_distance_min is not None and distance < self.stuck_macro_mid_distance_min:
                in_mid_band = False
            if self.stuck_macro_mid_distance_max is not None and distance > self.stuck_macro_mid_distance_max:
                in_mid_band = False
            if in_mid_band:
                self.stuck_macro_mid_trigger_count += 1
        self.last_policy_macro_bin = requested_macro_bin
        info["policy_macro_bin"] = requested_macro_bin
        info["effective_macro_bin"] = macro_bin
        info["policy_macro_suppressed"] = bool(policy_macro_suppressed)
        info["policy_macro_armed"] = bool(self.policy_macro_armed)
        info["policy_macro_clear_rearm_triggered"] = bool(clear_rearm_triggered)
        info["policy_macro_rearm_count"] = int(self.policy_macro_rearm_count)
        info["policy_macro_rearm_pending"] = bool(self.policy_macro_rearm_pending)
        info["stuck_macro_mid_trigger_count"] = int(self.stuck_macro_mid_trigger_count)
        info["stuck_macro_context"] = bool(stuck_macro_context)
        info["targeted_stuck_macro_context"] = bool(targeted_stuck_macro_context)
        info["suppressed_policy_macro_passthrough_triggered"] = bool(
            suppressed_policy_macro_passthrough_triggered
        )
        info["suppressed_policy_macro_override_triggered"] = bool(suppressed_override_triggered)
        info["suppressed_policy_generic_redirect_triggered"] = bool(suppressed_generic_redirect_triggered)
        info["stuck_macro_triggered"] = bool(generic_macro_triggered)
        info["targeted_stuck_macro_triggered"] = bool(targeted_macro_triggered)
        info["wrapper_recent_collision_norm"] = float(recent_collision_norm)
        info["wrapper_angle_error"] = float(angle_error)
        info["wrapper_distance"] = float(distance)
        true_distance = float(np.linalg.norm(base.target - base.pos))
        if self.coord_age >= self.position_obs_dt or terminated or truncated:
            self.observed_pos = base.pos.copy()
            if base.coord_noise_std > 0.0:
                self.observed_pos = self._noisy_pos(self.observed_pos)
            self.coord_age = 0.0
            self._sample_position_obs_dt()
            self.last_distance_progress = prev_distance - true_distance
        else:
            self.last_distance_progress = 0.0
        self.last_action_angle = angle_offset
        self.last_recovery_mode = float(int(info.get("recovery_mode", 0) or 0)) / 7.0
        return self._get_obs(), reward, terminated, truncated, info


class Observable12MacroLibraryV11RxContextWrapper(Observable12MacroLibraryV11RxWrapper):
    """V11rx with explicit targeted stuck-context features in the observation."""

    def _get_obs(self) -> np.ndarray:
        obs = super()._get_obs().copy()
        base = self.base_env
        delta = base.target - self.observed_pos
        distance = float(np.linalg.norm(delta))
        target_angle = float(np.arctan2(delta[1], delta[0]))
        angle_error = float(np.arctan2(np.sin(target_angle - base.heading), np.cos(target_angle - base.heading)))
        angle_error_abs = abs(angle_error)
        recent_collision_norm = 1.0 - np.clip(base.time_since_collision / 3.0, 0.0, 1.0)
        targeted_context = self._targeted_stuck_macro_context(
            base,
            recent_collision_norm=recent_collision_norm,
            distance=distance,
            angle_error_abs=angle_error_abs,
        )
        distance_pressure = 0.0
        if self.targeted_stuck_macro_distance_threshold is not None and self.targeted_stuck_macro_distance_threshold > 0.0:
            distance_pressure = np.clip(distance / self.targeted_stuck_macro_distance_threshold, 0.0, 1.5)
        obs[7] = np.clip(base.no_progress_time / 3.0, 0.0, 1.0)
        obs[8] = 1.0 if targeted_context else 0.0
        obs[11] = np.clip(distance_pressure, 0.0, 1.0)
        return obs


class Observable12MacroLibraryV11DurationWrapper(Observable12MacroLibraryV11Wrapper):
    """V11 duration-only extension: keep macro library, add duration bin."""

    DURATION_SIGNALS = np.asarray([-1.0, 0.0, 1.0], dtype=np.float32)

    def __init__(self, env: BlindNavEnv, **kwargs) -> None:
        super().__init__(env, **kwargs)
        # [direction_bin, speed_bin, macro_id, duration_bin]
        self.action_space = spaces.MultiDiscrete([5, 2, 7, 3])

    def step(self, action):
        base = self.base_env
        prev_distance = float(np.linalg.norm(base.target - self.observed_pos))
        self.coord_age += base.dt
        discrete = np.asarray(action, dtype=np.int64).reshape(-1)
        if discrete.size != 4:
            raise ValueError(f"expected 4 discrete action values, got shape {discrete.shape}")
        angle_bin = int(np.clip(discrete[0], 0, 4))
        speed_bin = int(np.clip(discrete[1], 0, 1))
        macro_bin = int(np.clip(discrete[2], 0, 6))
        duration_bin = int(np.clip(discrete[3], 0, 2))
        angle_offset = float(self.ANGLE_OFFSETS[angle_bin])
        recent_collision_norm = 1.0 - np.clip(base.time_since_collision / 3.0, 0.0, 1.0)
        stuck_macro_context = (
            base.no_progress_time >= self.stuck_macro_no_progress_threshold
            or recent_collision_norm >= self.stuck_macro_collision_threshold
        )
        if (
            self.recovery_enabled
            and macro_bin == 0
            and self.stuck_macro_explore_prob > 0.0
            and stuck_macro_context
            and self.stuck_macro_choices
            and float(base.np_random.random()) < self.stuck_macro_explore_prob
        ):
            macro_bin = int(base.np_random.choice(np.asarray(self.stuck_macro_choices, dtype=np.int64)))
        macro_signal = -1.0 if not self.recovery_enabled else float(self.MACRO_SIGNALS[macro_bin])
        if self.selective_recovery and macro_bin != 0:
            allow_recovery = (
                base.no_progress_time >= self.selective_no_progress_threshold
                or recent_collision_norm >= self.selective_collision_threshold
            )
            if not allow_recovery:
                macro_signal = -1.0
                macro_bin = 0
        speed_value = float(self.SPEED_VALUES[speed_bin])
        if self.force_recovery_high_speed and macro_bin != 0:
            speed_value = 1.0
        env_action = np.array(
            [
                angle_offset,
                speed_value,
                -1.0,
                macro_signal,
                float(self.DURATION_SIGNALS[duration_bin]),
            ],
            dtype=np.float32,
        )
        _, reward, terminated, truncated, info = self.env.step(env_action)
        true_distance = float(np.linalg.norm(base.target - base.pos))
        if self.coord_age >= self.position_obs_dt or terminated or truncated:
            self.observed_pos = base.pos.copy()
            if base.coord_noise_std > 0.0:
                self.observed_pos = self._noisy_pos(self.observed_pos)
            self.coord_age = 0.0
            self._sample_position_obs_dt()
            self.last_distance_progress = prev_distance - true_distance
        else:
            self.last_distance_progress = 0.0
        self.last_action_angle = angle_offset
        self.last_recovery_mode = float(int(info.get("recovery_mode", 0) or 0)) / 6.0
        return self._get_obs(), reward, terminated, truncated, info


class Observable12MacroLibraryV11DurationLiteWrapper(Observable12MacroLibraryV11DurationWrapper):
    """Conservative duration control: only selected recovery macros can vary duration."""

    def __init__(self, env: BlindNavEnv, **kwargs) -> None:
        duration_control_macro_choices = kwargs.pop("duration_control_macro_choices", (4, 5, 6))
        duration_long_macro_choices = kwargs.pop("duration_long_macro_choices", (4, 5))
        super().__init__(env, **kwargs)
        self.duration_control_macro_choices = tuple(int(v) for v in duration_control_macro_choices)
        self.duration_long_macro_choices = tuple(int(v) for v in duration_long_macro_choices)

    def step(self, action):
        discrete = np.asarray(action, dtype=np.int64).reshape(-1)
        if discrete.size != 4:
            raise ValueError(f"expected 4 discrete action values, got shape {discrete.shape}")
        adjusted = discrete.copy()
        macro_bin = int(np.clip(adjusted[2], 0, 6))
        duration_bin = int(np.clip(adjusted[3], 0, 2))
        # Keep normal navigation on the default duration and only allow
        # selected escape macros to vary. Restrict long duration further.
        if macro_bin == 0 or macro_bin not in self.duration_control_macro_choices:
            adjusted[3] = 1
        elif duration_bin == 2 and macro_bin not in self.duration_long_macro_choices:
            adjusted[3] = 1
        return super().step(adjusted)


class Observable12MacroLibraryV11TriggerMacroWrapper(Observable12MacroLibraryV11Wrapper):
    """Separate recovery trigger from macro selection for easier credit assignment."""

    TRIGGER_SIGNALS = np.asarray([-1.0, 1.0], dtype=np.float32)

    def __init__(self, env: BlindNavEnv, **kwargs) -> None:
        trigger_explore_prob = kwargs.pop("trigger_explore_prob", 0.0)
        super().__init__(env, **kwargs)
        # [direction_bin, speed_bin, trigger_bin, macro_id_without_normal]
        self.action_space = spaces.MultiDiscrete([5, 2, 2, 6])
        self.trigger_explore_prob = float(np.clip(trigger_explore_prob, 0.0, 1.0))

    def step(self, action):
        base = self.base_env
        prev_distance = float(np.linalg.norm(base.target - self.observed_pos))
        self.coord_age += base.dt
        discrete = np.asarray(action, dtype=np.int64).reshape(-1)
        if discrete.size != 4:
            raise ValueError(f"expected 4 discrete action values, got shape {discrete.shape}")
        angle_bin = int(np.clip(discrete[0], 0, 4))
        speed_bin = int(np.clip(discrete[1], 0, 1))
        trigger_bin = int(np.clip(discrete[2], 0, 1))
        macro_choice_bin = int(np.clip(discrete[3], 0, 5))
        macro_bin = macro_choice_bin + 1
        angle_offset = float(self.ANGLE_OFFSETS[angle_bin])
        recent_collision_norm = 1.0 - np.clip(base.time_since_collision / 3.0, 0.0, 1.0)
        stuck_macro_context = (
            base.no_progress_time >= self.stuck_macro_no_progress_threshold
            or recent_collision_norm >= self.stuck_macro_collision_threshold
        )
        if (
            self.recovery_enabled
            and trigger_bin == 0
            and self.trigger_explore_prob > 0.0
            and stuck_macro_context
            and float(base.np_random.random()) < self.trigger_explore_prob
        ):
            trigger_bin = 1
        if (
            self.recovery_enabled
            and trigger_bin == 1
            and self.stuck_macro_explore_prob > 0.0
            and stuck_macro_context
            and self.stuck_macro_choices
            and float(base.np_random.random()) < self.stuck_macro_explore_prob
        ):
            macro_choices = np.asarray(self.stuck_macro_choices, dtype=np.int64)
            macro_bin = int(base.np_random.choice(macro_choices, p=self.stuck_macro_weights))
        if not self.recovery_enabled or trigger_bin == 0:
            macro_signal = -1.0
            macro_bin = 0
        else:
            macro_signal = float(self.MACRO_SIGNALS[macro_bin])
        if self.selective_recovery and macro_bin != 0:
            allow_recovery = (
                base.no_progress_time >= self.selective_no_progress_threshold
                or recent_collision_norm >= self.selective_collision_threshold
            )
            if not allow_recovery:
                macro_signal = -1.0
                macro_bin = 0
        speed_value = float(self.SPEED_VALUES[speed_bin])
        if self.force_recovery_high_speed and macro_bin != 0:
            speed_value = 1.0
        env_action = np.array(
            [
                angle_offset,
                speed_value,
                -1.0,
                macro_signal,
            ],
            dtype=np.float32,
        )
        _, reward, terminated, truncated, info = self.env.step(env_action)
        true_distance = float(np.linalg.norm(base.target - base.pos))
        if self.coord_age >= self.position_obs_dt or terminated or truncated:
            self.observed_pos = base.pos.copy()
            if base.coord_noise_std > 0.0:
                self.observed_pos = self._noisy_pos(self.observed_pos)
            self.coord_age = 0.0
            self._sample_position_obs_dt()
            self.last_distance_progress = prev_distance - true_distance
        else:
            self.last_distance_progress = 0.0
        self.last_action_angle = angle_offset
        self.last_recovery_mode = float(int(info.get("recovery_mode", 0) or 0)) / 6.0
        return self._get_obs(), reward, terminated, truncated, info


class Observable6Wrapper(gym.Wrapper):
    """Compact control-oriented observation for delta-angle joystick control."""

    def __init__(self, env: BlindNavEnv) -> None:
        super().__init__(env)
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(OBSERVABLE6_DIM,), dtype=np.float32)
        self.prev_distance = 0.0
        self.prev_angle_error = 0.0
        self.distance_progress = 0.0
        self.movement_dist = 0.0
        self.delta_angle_error = 0.0

    @property
    def base_env(self) -> BlindNavEnv:
        return self.env.unwrapped

    def _angle_error(self) -> float:
        base = self.base_env
        delta = base.target - base.pos
        target_angle = float(np.arctan2(delta[1], delta[0]))
        return float(np.arctan2(np.sin(target_angle - base.heading), np.cos(target_angle - base.heading)))

    def reset(self, **kwargs):
        _, info = self.env.reset(**kwargs)
        base = self.base_env
        self.prev_distance = float(np.linalg.norm(base.target - base.pos))
        self.prev_angle_error = self._angle_error()
        self.distance_progress = 0.0
        self.movement_dist = 0.0
        self.delta_angle_error = 0.0
        return self._get_obs(), info

    def step(self, action):
        base = self.base_env
        prev_pos = base.pos.copy()
        prev_distance = float(np.linalg.norm(base.target - base.pos))
        prev_angle_error = self._angle_error()
        _, reward, terminated, truncated, info = self.env.step(action)
        current_distance = float(np.linalg.norm(base.target - base.pos))
        angle_error = self._angle_error()
        self.distance_progress = prev_distance - current_distance
        self.movement_dist = float(np.linalg.norm(base.pos - prev_pos))
        self.delta_angle_error = prev_angle_error - angle_error
        self.prev_distance = current_distance
        self.prev_angle_error = angle_error
        return self._get_obs(), reward, terminated, truncated, info

    def _get_obs(self) -> np.ndarray:
        base = self.base_env
        delta = base.target - base.pos
        return np.array(
            [
                np.clip(delta[0] / base.world_size, -1.0, 1.0),
                np.clip(delta[1] / base.world_size, -1.0, 1.0),
                np.clip(self.prev_angle_error / np.pi, -1.0, 1.0),
                np.clip(self.delta_angle_error / np.pi, -1.0, 1.0),
                np.clip(self.distance_progress / 30.0, -1.0, 1.0),
                np.clip(self.movement_dist / 40.0, 0.0, 1.0),
            ],
            dtype=np.float32,
        )


class EpisodeStopAndCheckpointCallback(BaseCallback):
    def __init__(
        self,
        target_episodes: int,
        checkpoint_dir: Path,
        *,
        checkpoint_interval: int = CHECKPOINT_INTERVAL,
        progress_interval: int = PROGRESS_INTERVAL,
        episode_offset: int = 0,
    ) -> None:
        super().__init__()
        self.episode_offset = max(0, int(episode_offset))
        self.target_episodes = self.episode_offset + int(target_episodes)
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_interval = max(1, int(checkpoint_interval))
        self.progress_interval = max(1, int(progress_interval))
        self.episodes = self.episode_offset
        self._saved: set[int] = set()

    def _on_step(self) -> bool:
        dones = self.locals.get("dones")
        if dones is None:
            return True
        completed = int(np.asarray(dones).sum())
        if completed <= 0:
            return True
        prev = self.episodes
        self.episodes += completed

        start_progress = (prev // self.progress_interval + 1) * self.progress_interval
        for mark in range(start_progress, self.episodes + 1, self.progress_interval):
            print(f"progress episodes={mark}/{self.target_episodes}", flush=True)

        start_ckpt = (prev // self.checkpoint_interval + 1) * self.checkpoint_interval
        for mark in range(start_ckpt, self.episodes + 1, self.checkpoint_interval):
            if mark in self._saved:
                continue
            ckpt = self.checkpoint_dir / f"checkpoint_ep_{mark:05d}"
            self.model.save(str(ckpt))
            self._saved.add(mark)
            print(f"checkpoint_saved episode={mark} path={ckpt}.zip", flush=True)

        return self.episodes < self.target_episodes


def select_obs(obs: np.ndarray, state_mode: str) -> np.ndarray:
    if state_mode in {
        "observable10",
        "observable8",
        *OBSERVABLE7_STOP_MODES,
        *OBSERVABLE11_STOP_MODES,
        *OBSERVABLE8_TARGET_MODES,
        *OBSERVABLE12_TARGET_MODES,
    }:
        raise ValueError(f"{state_mode} must be produced by an observable wrapper")
    return np.asarray(obs, dtype=np.float32)[STATE_MODES[state_mode]]


def canonical_env_mode(state_mode: str) -> str:
    if state_mode in OBSERVABLE11_STOP_MODES:
        return state_mode.replace("observable11", "observable7", 1)
    return state_mode


def load_env_overrides(path: str | Path | None) -> dict[str, dict[str, Any]]:
    if not path:
        return {}
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise TypeError("env overrides must be a JSON object")
    out: dict[str, dict[str, Any]] = {}
    for key, value in data.items():
        if not isinstance(value, dict):
            raise TypeError(f"env override for {key!r} must be a JSON object")
        out[str(key)] = dict(value)
    return out


def _extract_override_section(value: dict[str, Any], section: str) -> dict[str, Any]:
    nested = value.get(section)
    if isinstance(nested, dict):
        return dict(nested)
    if any(key in value for key in ("__env__", "__wrapper__")):
        return {}
    return dict(value)


def resolve_env_overrides(state_mode: str, env_overrides: dict[str, dict[str, Any]] | None) -> dict[str, Any]:
    if not env_overrides:
        return {}
    canonical = canonical_env_mode(state_mode)
    resolved: dict[str, Any] = {}
    shared = env_overrides.get("__all__")
    if isinstance(shared, dict):
        resolved.update(_extract_override_section(shared, "__env__"))
    for key in (state_mode, canonical):
        value = env_overrides.get(key)
        if isinstance(value, dict):
            resolved.update(_extract_override_section(value, "__env__"))
    return resolved


def resolve_wrapper_overrides(state_mode: str, env_overrides: dict[str, dict[str, Any]] | None) -> dict[str, Any]:
    if not env_overrides:
        return {}
    canonical = canonical_env_mode(state_mode)
    resolved: dict[str, Any] = {}
    shared = env_overrides.get("__all__")
    if isinstance(shared, dict):
        resolved.update(_extract_override_section(shared, "__wrapper__"))
    for key in (state_mode, canonical):
        value = env_overrides.get(key)
        if isinstance(value, dict):
            resolved.update(_extract_override_section(value, "__wrapper__"))
    return resolved


def observation_dim(state_mode: str) -> int:
    if state_mode == "observable10":
        return OBSERVABLE10_DIM
    if state_mode in OBSERVABLE10_TARGET_MODES:
        return OBSERVABLE10_TARGET_DIM
    if state_mode in {"observable8", "observable8_delta45"}:
        return OBSERVABLE8_DIM
    if state_mode in OBSERVABLE8_TARGET_MODES:
        return OBSERVABLE8_TARGET_DIM
    if state_mode in OBSERVABLE9_TARGET_MODES:
        return OBSERVABLE9_TARGET_DIM
    if state_mode in OBSERVABLE12_TARGET_MODES:
        return OBSERVABLE12_TARGET_DIM
    if state_mode == "observable6":
        return OBSERVABLE6_DIM
    if state_mode in OBSERVABLE7_STOP_MODES:
        return OBSERVABLE7_DIM
    if state_mode in OBSERVABLE11_STOP_MODES:
        return OBSERVABLE11_DIM
    return len(STATE_MODES[state_mode])


def wrap_policy_env(
    env: BlindNavEnv,
    state_mode: str,
    env_overrides: dict[str, dict[str, Any]] | None = None,
) -> gym.Env:
    wrapper_overrides = resolve_wrapper_overrides(state_mode, env_overrides)
    if state_mode == "observable10":
        return Observable10Wrapper(env)
    if state_mode == "observable8":
        return Observable8Wrapper(env)
    if state_mode == "observable6":
        return Observable6Wrapper(env)
    if state_mode == "observable8_delta45":
        return Observable8DeltaWrapper(env)
    if state_mode in OBSERVABLE8_TARGET_MODES:
        return Observable8TargetWrapper(env)
    if state_mode in OBSERVABLE10_TARGET_MODES:
        return Observable10ContinuousFixedRecoveryWrapper(env)
    if state_mode in OBSERVABLE9_TARGET_MODES:
        return Observable9ContinuousFixedRecoveryWrapper(env)
    if state_mode in {"observable12_target8_discrete_macro_v5", "observable12_target8_discrete_macro_v6"}:
        return Observable12DiscreteMacroWrapper(env)
    if state_mode == "observable12_target8_discrete_macro_v8":
        return Observable12DiscreteMacroV7Wrapper(env)
    if state_mode == "observable12_target8_discrete_macro_v7":
        return Observable12DiscreteMacroV7Wrapper(env)
    if state_mode in {
        "observable12_target8_discrete_recovery_v10_stage1",
        "observable12_target8_discrete_recovery_v10_stage2a",
        "observable12_target8_discrete_recovery_v10_stage2",
        "observable12_target8_discrete_recovery_v10_stage3",
    }:
        return Observable12DiscreteRecoveryV10Wrapper(
            env,
            recovery_enabled=state_mode != "observable12_target8_discrete_recovery_v10_stage1",
            selective_recovery=state_mode in {
                "observable12_target8_discrete_recovery_v10_stage2a",
                "observable12_target8_discrete_recovery_v10_stage2",
                "observable12_target8_discrete_recovery_v10_stage3",
            },
            selective_no_progress_threshold=float(wrapper_overrides.get("selective_no_progress_threshold", 1.10)),
            selective_collision_threshold=float(wrapper_overrides.get("selective_collision_threshold", 0.90)),
            force_recovery_high_speed=state_mode in {
                "observable12_target8_discrete_recovery_v10_stage2a",
                "observable12_target8_discrete_recovery_v10_stage2",
                "observable12_target8_discrete_recovery_v10_stage3",
            },
        )
    if state_mode in {
        "observable12_target8_discrete_recovery_v10b9_stage1",
        "observable12_target8_discrete_recovery_v10b9_stage2a",
        "observable12_target8_discrete_recovery_v10b9_stage2",
        "observable12_target8_discrete_recovery_v10b9_stage3",
    }:
        return Observable12DiscreteRecoveryV10B9Wrapper(
            env,
            recovery_enabled=state_mode != "observable12_target8_discrete_recovery_v10b9_stage1",
            selective_recovery=state_mode in {
                "observable12_target8_discrete_recovery_v10b9_stage2a",
                "observable12_target8_discrete_recovery_v10b9_stage2",
                "observable12_target8_discrete_recovery_v10b9_stage3",
            },
            selective_no_progress_threshold=float(wrapper_overrides.get("selective_no_progress_threshold", 1.10)),
            selective_collision_threshold=float(wrapper_overrides.get("selective_collision_threshold", 0.90)),
            force_recovery_high_speed=state_mode in {
                "observable12_target8_discrete_recovery_v10b9_stage2a",
                "observable12_target8_discrete_recovery_v10b9_stage2",
                "observable12_target8_discrete_recovery_v10b9_stage3",
            },
        )
    if state_mode in {
        "observable12_target8_macro_library_v11_stage1",
        "observable12_target8_macro_library_v11_stage2a",
        "observable12_target8_macro_library_v11_stage2",
        "observable12_target8_macro_library_v11_stage3",
    }:
        return Observable12MacroLibraryV11Wrapper(
            env,
            recovery_enabled=state_mode != "observable12_target8_macro_library_v11_stage1",
            selective_recovery=state_mode in {
                "observable12_target8_macro_library_v11_stage2a",
                "observable12_target8_macro_library_v11_stage2",
                "observable12_target8_macro_library_v11_stage3",
            },
            selective_no_progress_threshold=float(wrapper_overrides.get("selective_no_progress_threshold", 1.10)),
            selective_collision_threshold=float(wrapper_overrides.get("selective_collision_threshold", 0.90)),
            force_recovery_high_speed=state_mode in {
                "observable12_target8_macro_library_v11_stage2a",
                "observable12_target8_macro_library_v11_stage2",
                "observable12_target8_macro_library_v11_stage3",
            },
            stuck_macro_explore_prob=float(wrapper_overrides.get("stuck_macro_explore_prob", 0.0)),
            stuck_macro_no_progress_threshold=float(wrapper_overrides.get("stuck_macro_no_progress_threshold", 0.55)),
            stuck_macro_collision_threshold=float(wrapper_overrides.get("stuck_macro_collision_threshold", 0.82)),
            stuck_macro_min_distance=wrapper_overrides.get("stuck_macro_min_distance", None),
            stuck_macro_max_distance=wrapper_overrides.get("stuck_macro_max_distance", None),
            stuck_macro_mid_distance_min=wrapper_overrides.get("stuck_macro_mid_distance_min", None),
            stuck_macro_mid_distance_max=wrapper_overrides.get("stuck_macro_mid_distance_max", None),
            stuck_macro_mid_max_triggers=wrapper_overrides.get("stuck_macro_mid_max_triggers", None),
            stuck_macro_mid_max_stuck_time=wrapper_overrides.get("stuck_macro_mid_max_stuck_time", None),
            stuck_macro_mid_choices=wrapper_overrides.get("stuck_macro_mid_choices", None),
            stuck_macro_mid_weights=wrapper_overrides.get("stuck_macro_mid_weights", None),
            stuck_macro_choices=tuple(wrapper_overrides.get("stuck_macro_choices", (1, 2, 3, 4, 5, 6))),
            stuck_macro_weights=wrapper_overrides.get("stuck_macro_weights", None),
            targeted_stuck_macro_explore_prob=float(wrapper_overrides.get("targeted_stuck_macro_explore_prob", 0.0)),
            targeted_stuck_macro_no_progress_threshold=float(
                wrapper_overrides.get(
                    "targeted_stuck_macro_no_progress_threshold",
                    wrapper_overrides.get("stuck_macro_no_progress_threshold", 0.55),
                )
            ),
            targeted_stuck_macro_collision_threshold=float(
                wrapper_overrides.get(
                    "targeted_stuck_macro_collision_threshold",
                    wrapper_overrides.get("stuck_macro_collision_threshold", 0.82),
                )
            ),
            targeted_stuck_macro_angle_error_threshold=wrapper_overrides.get(
                "targeted_stuck_macro_angle_error_threshold",
                None,
            ),
            targeted_stuck_macro_distance_threshold=wrapper_overrides.get(
                "targeted_stuck_macro_distance_threshold",
                None,
            ),
            targeted_stuck_macro_max_distance_threshold=wrapper_overrides.get(
                "targeted_stuck_macro_max_distance_threshold",
                None,
            ),
            targeted_stuck_macro_stuck_time_threshold=wrapper_overrides.get(
                "targeted_stuck_macro_stuck_time_threshold",
                None,
            ),
            targeted_stuck_macro_mid_trigger_count_threshold=wrapper_overrides.get(
                "targeted_stuck_macro_mid_trigger_count_threshold",
                None,
            ),
            targeted_stuck_macro_allow_after_suppressed_policy=bool(
                wrapper_overrides.get("targeted_stuck_macro_allow_after_suppressed_policy", False)
            ),
            targeted_stuck_macro_choices=tuple(wrapper_overrides.get("targeted_stuck_macro_choices", ())),
            targeted_stuck_macro_weights=wrapper_overrides.get("targeted_stuck_macro_weights", None),
            suppressed_policy_macro_override_prob=float(
                wrapper_overrides.get("suppressed_policy_macro_override_prob", 0.0)
            ),
            suppressed_policy_macro_distance_threshold=wrapper_overrides.get(
                "suppressed_policy_macro_distance_threshold",
                None,
            ),
            suppressed_policy_macro_max_distance_threshold=wrapper_overrides.get(
                "suppressed_policy_macro_max_distance_threshold",
                None,
            ),
            suppressed_policy_macro_stuck_time_threshold=wrapper_overrides.get(
                "suppressed_policy_macro_stuck_time_threshold",
                None,
            ),
            suppressed_policy_macro_mid_trigger_count_threshold=wrapper_overrides.get(
                "suppressed_policy_macro_mid_trigger_count_threshold",
                None,
            ),
            suppressed_policy_macro_choices=tuple(wrapper_overrides.get("suppressed_policy_macro_choices", ())),
            suppressed_policy_macro_weights=wrapper_overrides.get("suppressed_policy_macro_weights", None),
            suppressed_policy_generic_redirect_prob=float(
                wrapper_overrides.get("suppressed_policy_generic_redirect_prob", 0.0)
            ),
            suppressed_policy_generic_redirect_distance_threshold=wrapper_overrides.get(
                "suppressed_policy_generic_redirect_distance_threshold",
                None,
            ),
            suppressed_policy_generic_redirect_max_distance_threshold=wrapper_overrides.get(
                "suppressed_policy_generic_redirect_max_distance_threshold",
                None,
            ),
            suppressed_policy_generic_redirect_stuck_time_threshold=wrapper_overrides.get(
                "suppressed_policy_generic_redirect_stuck_time_threshold",
                None,
            ),
            suppressed_policy_generic_redirect_mid_trigger_count_threshold=wrapper_overrides.get(
                "suppressed_policy_generic_redirect_mid_trigger_count_threshold",
                None,
            ),
            suppressed_policy_generic_redirect_choices=tuple(
                wrapper_overrides.get("suppressed_policy_generic_redirect_choices", ())
            ),
            suppressed_policy_generic_redirect_weights=wrapper_overrides.get(
                "suppressed_policy_generic_redirect_weights",
                None,
            ),
            suppressed_policy_macro_passthrough_prob=float(
                wrapper_overrides.get("suppressed_policy_macro_passthrough_prob", 0.0)
            ),
            suppressed_policy_macro_passthrough_bins=tuple(
                wrapper_overrides.get("suppressed_policy_macro_passthrough_bins", ())
            ),
            suppressed_policy_macro_passthrough_distance_threshold=wrapper_overrides.get(
                "suppressed_policy_macro_passthrough_distance_threshold",
                None,
            ),
            suppressed_policy_macro_passthrough_max_distance_threshold=wrapper_overrides.get(
                "suppressed_policy_macro_passthrough_max_distance_threshold",
                None,
            ),
            suppressed_policy_macro_passthrough_stuck_time_threshold=wrapper_overrides.get(
                "suppressed_policy_macro_passthrough_stuck_time_threshold",
                None,
            ),
            suppressed_policy_macro_passthrough_mid_trigger_count_threshold=wrapper_overrides.get(
                "suppressed_policy_macro_passthrough_mid_trigger_count_threshold",
                None,
            ),
            policy_macro_requires_release=bool(wrapper_overrides.get("policy_macro_requires_release", False)),
            policy_macro_release_gate_bins=tuple(wrapper_overrides.get("policy_macro_release_gate_bins", ())),
            policy_macro_rearm_on_clear=bool(wrapper_overrides.get("policy_macro_rearm_on_clear", False)),
            policy_macro_rearm_clear_time_threshold=float(
                wrapper_overrides.get("policy_macro_rearm_clear_time_threshold", 1.5)
            ),
            policy_macro_rearm_clear_no_progress_threshold=float(
                wrapper_overrides.get("policy_macro_rearm_clear_no_progress_threshold", 0.25)
            ),
            policy_macro_rearm_clear_stuck_threshold=float(
                wrapper_overrides.get("policy_macro_rearm_clear_stuck_threshold", 0.25)
            ),
            policy_macro_rearm_max_count=wrapper_overrides.get(
                "policy_macro_rearm_max_count",
                None,
            ),
            policy_macro_rearm_min_distance=wrapper_overrides.get(
                "policy_macro_rearm_min_distance",
                None,
            ),
            policy_macro_rearm_max_distance=wrapper_overrides.get(
                "policy_macro_rearm_max_distance",
                None,
            ),
        )
    if state_mode == "observable12_target8_macro_library_v11ds_stage2":
        return Observable12MacroLibraryV11DurationWrapper(
            env,
            recovery_enabled=True,
            selective_recovery=True,
            selective_no_progress_threshold=float(wrapper_overrides.get("selective_no_progress_threshold", 1.10)),
            selective_collision_threshold=float(wrapper_overrides.get("selective_collision_threshold", 0.90)),
            force_recovery_high_speed=True,
            stuck_macro_explore_prob=float(wrapper_overrides.get("stuck_macro_explore_prob", 0.0)),
            stuck_macro_no_progress_threshold=float(wrapper_overrides.get("stuck_macro_no_progress_threshold", 0.55)),
            stuck_macro_collision_threshold=float(wrapper_overrides.get("stuck_macro_collision_threshold", 0.82)),
            stuck_macro_choices=tuple(wrapper_overrides.get("stuck_macro_choices", (1, 2, 3, 4, 5, 6))),
            stuck_macro_weights=wrapper_overrides.get("stuck_macro_weights", None),
        )
    if state_mode == "observable12_target8_macro_library_v11dsl_stage2":
        return Observable12MacroLibraryV11DurationLiteWrapper(
            env,
            recovery_enabled=True,
            selective_recovery=True,
            selective_no_progress_threshold=float(wrapper_overrides.get("selective_no_progress_threshold", 1.10)),
            selective_collision_threshold=float(wrapper_overrides.get("selective_collision_threshold", 0.90)),
            force_recovery_high_speed=True,
            stuck_macro_explore_prob=float(wrapper_overrides.get("stuck_macro_explore_prob", 0.0)),
            stuck_macro_no_progress_threshold=float(wrapper_overrides.get("stuck_macro_no_progress_threshold", 0.55)),
            stuck_macro_collision_threshold=float(wrapper_overrides.get("stuck_macro_collision_threshold", 0.82)),
            stuck_macro_choices=tuple(wrapper_overrides.get("stuck_macro_choices", (1, 2, 3, 4, 5, 6))),
            stuck_macro_weights=wrapper_overrides.get("stuck_macro_weights", None),
            duration_control_macro_choices=tuple(wrapper_overrides.get("duration_control_macro_choices", (4, 5, 6))),
            duration_long_macro_choices=tuple(wrapper_overrides.get("duration_long_macro_choices", (4, 5))),
        )
    if state_mode == "observable12_target8_macro_library_v11tm_stage2":
        return Observable12MacroLibraryV11TriggerMacroWrapper(
            env,
            recovery_enabled=True,
            selective_recovery=True,
            selective_no_progress_threshold=float(wrapper_overrides.get("selective_no_progress_threshold", 1.10)),
            selective_collision_threshold=float(wrapper_overrides.get("selective_collision_threshold", 0.90)),
            force_recovery_high_speed=True,
            stuck_macro_explore_prob=float(wrapper_overrides.get("stuck_macro_explore_prob", 0.0)),
            stuck_macro_no_progress_threshold=float(wrapper_overrides.get("stuck_macro_no_progress_threshold", 0.55)),
            stuck_macro_collision_threshold=float(wrapper_overrides.get("stuck_macro_collision_threshold", 0.82)),
            stuck_macro_choices=tuple(wrapper_overrides.get("stuck_macro_choices", (1, 2, 3, 4, 5, 6))),
            stuck_macro_weights=wrapper_overrides.get("stuck_macro_weights", None),
            trigger_explore_prob=float(wrapper_overrides.get("trigger_explore_prob", 0.0)),
        )
    if state_mode == "observable12_target8_macro_library_v11re_stage2":
        return Observable12MacroLibraryV11Wrapper(
            env,
            recovery_enabled=True,
            selective_recovery=True,
            selective_no_progress_threshold=float(wrapper_overrides.get("selective_no_progress_threshold", 1.10)),
            selective_collision_threshold=float(wrapper_overrides.get("selective_collision_threshold", 0.90)),
            force_recovery_high_speed=True,
            stuck_macro_explore_prob=float(wrapper_overrides.get("stuck_macro_explore_prob", 0.0)),
            stuck_macro_no_progress_threshold=float(wrapper_overrides.get("stuck_macro_no_progress_threshold", 0.55)),
            stuck_macro_collision_threshold=float(wrapper_overrides.get("stuck_macro_collision_threshold", 0.82)),
            stuck_macro_choices=tuple(wrapper_overrides.get("stuck_macro_choices", (1, 2, 3, 4, 5, 6))),
            stuck_macro_weights=wrapper_overrides.get("stuck_macro_weights", None),
            targeted_stuck_macro_explore_prob=float(wrapper_overrides.get("targeted_stuck_macro_explore_prob", 0.0)),
            targeted_stuck_macro_no_progress_threshold=float(
                wrapper_overrides.get(
                    "targeted_stuck_macro_no_progress_threshold",
                    wrapper_overrides.get("stuck_macro_no_progress_threshold", 0.55),
                )
            ),
            targeted_stuck_macro_collision_threshold=float(
                wrapper_overrides.get(
                    "targeted_stuck_macro_collision_threshold",
                    wrapper_overrides.get("stuck_macro_collision_threshold", 0.82),
                )
            ),
            targeted_stuck_macro_angle_error_threshold=wrapper_overrides.get(
                "targeted_stuck_macro_angle_error_threshold",
                None,
            ),
            targeted_stuck_macro_distance_threshold=wrapper_overrides.get(
                "targeted_stuck_macro_distance_threshold",
                None,
            ),
            targeted_stuck_macro_max_distance_threshold=wrapper_overrides.get(
                "targeted_stuck_macro_max_distance_threshold",
                None,
            ),
            targeted_stuck_macro_stuck_time_threshold=wrapper_overrides.get(
                "targeted_stuck_macro_stuck_time_threshold",
                None,
            ),
            targeted_stuck_macro_mid_trigger_count_threshold=wrapper_overrides.get(
                "targeted_stuck_macro_mid_trigger_count_threshold",
                None,
            ),
            targeted_stuck_macro_allow_after_suppressed_policy=bool(
                wrapper_overrides.get("targeted_stuck_macro_allow_after_suppressed_policy", False)
            ),
            targeted_stuck_macro_choices=tuple(wrapper_overrides.get("targeted_stuck_macro_choices", ())),
            targeted_stuck_macro_weights=wrapper_overrides.get("targeted_stuck_macro_weights", None),
            suppressed_policy_macro_override_prob=float(
                wrapper_overrides.get("suppressed_policy_macro_override_prob", 0.0)
            ),
            suppressed_policy_macro_distance_threshold=wrapper_overrides.get(
                "suppressed_policy_macro_distance_threshold",
                None,
            ),
            suppressed_policy_macro_max_distance_threshold=wrapper_overrides.get(
                "suppressed_policy_macro_max_distance_threshold",
                None,
            ),
            suppressed_policy_macro_stuck_time_threshold=wrapper_overrides.get(
                "suppressed_policy_macro_stuck_time_threshold",
                None,
            ),
            suppressed_policy_macro_mid_trigger_count_threshold=wrapper_overrides.get(
                "suppressed_policy_macro_mid_trigger_count_threshold",
                None,
            ),
            suppressed_policy_macro_choices=tuple(wrapper_overrides.get("suppressed_policy_macro_choices", ())),
            suppressed_policy_macro_weights=wrapper_overrides.get("suppressed_policy_macro_weights", None),
            suppressed_policy_generic_redirect_prob=float(
                wrapper_overrides.get("suppressed_policy_generic_redirect_prob", 0.0)
            ),
            suppressed_policy_generic_redirect_distance_threshold=wrapper_overrides.get(
                "suppressed_policy_generic_redirect_distance_threshold",
                None,
            ),
            suppressed_policy_generic_redirect_max_distance_threshold=wrapper_overrides.get(
                "suppressed_policy_generic_redirect_max_distance_threshold",
                None,
            ),
            suppressed_policy_generic_redirect_stuck_time_threshold=wrapper_overrides.get(
                "suppressed_policy_generic_redirect_stuck_time_threshold",
                None,
            ),
            suppressed_policy_generic_redirect_mid_trigger_count_threshold=wrapper_overrides.get(
                "suppressed_policy_generic_redirect_mid_trigger_count_threshold",
                None,
            ),
            suppressed_policy_generic_redirect_choices=tuple(
                wrapper_overrides.get("suppressed_policy_generic_redirect_choices", ())
            ),
            suppressed_policy_generic_redirect_weights=wrapper_overrides.get(
                "suppressed_policy_generic_redirect_weights",
                None,
            ),
            suppressed_policy_macro_passthrough_prob=float(
                wrapper_overrides.get("suppressed_policy_macro_passthrough_prob", 0.0)
            ),
            suppressed_policy_macro_passthrough_bins=tuple(
                wrapper_overrides.get("suppressed_policy_macro_passthrough_bins", ())
            ),
            suppressed_policy_macro_passthrough_distance_threshold=wrapper_overrides.get(
                "suppressed_policy_macro_passthrough_distance_threshold",
                None,
            ),
            suppressed_policy_macro_passthrough_max_distance_threshold=wrapper_overrides.get(
                "suppressed_policy_macro_passthrough_max_distance_threshold",
                None,
            ),
            suppressed_policy_macro_passthrough_stuck_time_threshold=wrapper_overrides.get(
                "suppressed_policy_macro_passthrough_stuck_time_threshold",
                None,
            ),
            suppressed_policy_macro_passthrough_mid_trigger_count_threshold=wrapper_overrides.get(
                "suppressed_policy_macro_passthrough_mid_trigger_count_threshold",
                None,
            ),
            policy_macro_requires_release=bool(wrapper_overrides.get("policy_macro_requires_release", False)),
            policy_macro_release_gate_bins=tuple(wrapper_overrides.get("policy_macro_release_gate_bins", ())),
            policy_macro_rearm_on_clear=bool(wrapper_overrides.get("policy_macro_rearm_on_clear", False)),
            policy_macro_rearm_clear_time_threshold=float(
                wrapper_overrides.get("policy_macro_rearm_clear_time_threshold", 1.5)
            ),
            policy_macro_rearm_clear_no_progress_threshold=float(
                wrapper_overrides.get("policy_macro_rearm_clear_no_progress_threshold", 0.25)
            ),
            policy_macro_rearm_clear_stuck_threshold=float(
                wrapper_overrides.get("policy_macro_rearm_clear_stuck_threshold", 0.25)
            ),
            policy_macro_rearm_max_count=wrapper_overrides.get(
                "policy_macro_rearm_max_count",
                None,
            ),
            policy_macro_rearm_min_distance=wrapper_overrides.get(
                "policy_macro_rearm_min_distance",
                None,
            ),
            policy_macro_rearm_max_distance=wrapper_overrides.get(
                "policy_macro_rearm_max_distance",
                None,
            ),
        )
    if state_mode == "observable12_target8_macro_library_v11rx_stage2":
        return Observable12MacroLibraryV11RxWrapper(
            env,
            recovery_enabled=True,
            selective_recovery=True,
            selective_no_progress_threshold=float(wrapper_overrides.get("selective_no_progress_threshold", 1.10)),
            selective_collision_threshold=float(wrapper_overrides.get("selective_collision_threshold", 0.90)),
            force_recovery_high_speed=True,
            stuck_macro_explore_prob=float(wrapper_overrides.get("stuck_macro_explore_prob", 0.0)),
            stuck_macro_no_progress_threshold=float(wrapper_overrides.get("stuck_macro_no_progress_threshold", 0.55)),
            stuck_macro_collision_threshold=float(wrapper_overrides.get("stuck_macro_collision_threshold", 0.82)),
            stuck_macro_min_distance=wrapper_overrides.get("stuck_macro_min_distance", None),
            stuck_macro_max_distance=wrapper_overrides.get("stuck_macro_max_distance", None),
            stuck_macro_mid_distance_min=wrapper_overrides.get("stuck_macro_mid_distance_min", None),
            stuck_macro_mid_distance_max=wrapper_overrides.get("stuck_macro_mid_distance_max", None),
            stuck_macro_mid_max_triggers=wrapper_overrides.get("stuck_macro_mid_max_triggers", None),
            stuck_macro_mid_max_stuck_time=wrapper_overrides.get("stuck_macro_mid_max_stuck_time", None),
            stuck_macro_mid_choices=wrapper_overrides.get("stuck_macro_mid_choices", None),
            stuck_macro_mid_weights=wrapper_overrides.get("stuck_macro_mid_weights", None),
            stuck_macro_choices=tuple(wrapper_overrides.get("stuck_macro_choices", (1, 2, 3, 4, 5, 6, 7))),
            stuck_macro_weights=wrapper_overrides.get("stuck_macro_weights", None),
            targeted_stuck_macro_explore_prob=float(wrapper_overrides.get("targeted_stuck_macro_explore_prob", 0.0)),
            targeted_stuck_macro_no_progress_threshold=float(
                wrapper_overrides.get(
                    "targeted_stuck_macro_no_progress_threshold",
                    wrapper_overrides.get("stuck_macro_no_progress_threshold", 0.55),
                )
            ),
            targeted_stuck_macro_collision_threshold=float(
                wrapper_overrides.get(
                    "targeted_stuck_macro_collision_threshold",
                    wrapper_overrides.get("stuck_macro_collision_threshold", 0.82),
                )
            ),
            targeted_stuck_macro_angle_error_threshold=wrapper_overrides.get(
                "targeted_stuck_macro_angle_error_threshold",
                None,
            ),
            targeted_stuck_macro_distance_threshold=wrapper_overrides.get(
                "targeted_stuck_macro_distance_threshold",
                None,
            ),
            targeted_stuck_macro_max_distance_threshold=wrapper_overrides.get(
                "targeted_stuck_macro_max_distance_threshold",
                None,
            ),
            targeted_stuck_macro_stuck_time_threshold=wrapper_overrides.get(
                "targeted_stuck_macro_stuck_time_threshold",
                None,
            ),
            targeted_stuck_macro_mid_trigger_count_threshold=wrapper_overrides.get(
                "targeted_stuck_macro_mid_trigger_count_threshold",
                None,
            ),
            targeted_stuck_macro_allow_after_suppressed_policy=bool(
                wrapper_overrides.get("targeted_stuck_macro_allow_after_suppressed_policy", False)
            ),
            targeted_stuck_macro_choices=tuple(wrapper_overrides.get("targeted_stuck_macro_choices", ())),
            targeted_stuck_macro_weights=wrapper_overrides.get("targeted_stuck_macro_weights", None),
            suppressed_policy_macro_override_prob=float(
                wrapper_overrides.get("suppressed_policy_macro_override_prob", 0.0)
            ),
            suppressed_policy_macro_distance_threshold=wrapper_overrides.get(
                "suppressed_policy_macro_distance_threshold",
                None,
            ),
            suppressed_policy_macro_max_distance_threshold=wrapper_overrides.get(
                "suppressed_policy_macro_max_distance_threshold",
                None,
            ),
            suppressed_policy_macro_stuck_time_threshold=wrapper_overrides.get(
                "suppressed_policy_macro_stuck_time_threshold",
                None,
            ),
            suppressed_policy_macro_mid_trigger_count_threshold=wrapper_overrides.get(
                "suppressed_policy_macro_mid_trigger_count_threshold",
                None,
            ),
            suppressed_policy_macro_choices=tuple(wrapper_overrides.get("suppressed_policy_macro_choices", ())),
            suppressed_policy_macro_weights=wrapper_overrides.get("suppressed_policy_macro_weights", None),
            suppressed_policy_generic_redirect_prob=float(
                wrapper_overrides.get("suppressed_policy_generic_redirect_prob", 0.0)
            ),
            suppressed_policy_generic_redirect_distance_threshold=wrapper_overrides.get(
                "suppressed_policy_generic_redirect_distance_threshold",
                None,
            ),
            suppressed_policy_generic_redirect_max_distance_threshold=wrapper_overrides.get(
                "suppressed_policy_generic_redirect_max_distance_threshold",
                None,
            ),
            suppressed_policy_generic_redirect_stuck_time_threshold=wrapper_overrides.get(
                "suppressed_policy_generic_redirect_stuck_time_threshold",
                None,
            ),
            suppressed_policy_generic_redirect_mid_trigger_count_threshold=wrapper_overrides.get(
                "suppressed_policy_generic_redirect_mid_trigger_count_threshold",
                None,
            ),
            suppressed_policy_generic_redirect_choices=tuple(
                wrapper_overrides.get("suppressed_policy_generic_redirect_choices", ())
            ),
            suppressed_policy_generic_redirect_weights=wrapper_overrides.get(
                "suppressed_policy_generic_redirect_weights",
                None,
            ),
            suppressed_policy_macro_passthrough_prob=float(
                wrapper_overrides.get("suppressed_policy_macro_passthrough_prob", 0.0)
            ),
            suppressed_policy_macro_passthrough_bins=tuple(
                wrapper_overrides.get("suppressed_policy_macro_passthrough_bins", ())
            ),
            suppressed_policy_macro_passthrough_distance_threshold=wrapper_overrides.get(
                "suppressed_policy_macro_passthrough_distance_threshold",
                None,
            ),
            suppressed_policy_macro_passthrough_max_distance_threshold=wrapper_overrides.get(
                "suppressed_policy_macro_passthrough_max_distance_threshold",
                None,
            ),
            suppressed_policy_macro_passthrough_stuck_time_threshold=wrapper_overrides.get(
                "suppressed_policy_macro_passthrough_stuck_time_threshold",
                None,
            ),
            suppressed_policy_macro_passthrough_mid_trigger_count_threshold=wrapper_overrides.get(
                "suppressed_policy_macro_passthrough_mid_trigger_count_threshold",
                None,
            ),
            policy_macro_requires_release=bool(wrapper_overrides.get("policy_macro_requires_release", False)),
            policy_macro_release_gate_bins=tuple(wrapper_overrides.get("policy_macro_release_gate_bins", ())),
            policy_macro_rearm_on_clear=bool(wrapper_overrides.get("policy_macro_rearm_on_clear", False)),
            policy_macro_rearm_clear_time_threshold=float(
                wrapper_overrides.get("policy_macro_rearm_clear_time_threshold", 1.5)
            ),
            policy_macro_rearm_clear_no_progress_threshold=float(
                wrapper_overrides.get("policy_macro_rearm_clear_no_progress_threshold", 0.25)
            ),
            policy_macro_rearm_clear_stuck_threshold=float(
                wrapper_overrides.get("policy_macro_rearm_clear_stuck_threshold", 0.25)
            ),
            policy_macro_rearm_max_count=wrapper_overrides.get(
                "policy_macro_rearm_max_count",
                None,
            ),
            policy_macro_rearm_min_distance=wrapper_overrides.get(
                "policy_macro_rearm_min_distance",
                None,
            ),
            policy_macro_rearm_max_distance=wrapper_overrides.get(
                "policy_macro_rearm_max_distance",
                None,
            ),
        )
    if state_mode == "observable12_target8_macro_library_v11rxc_stage2":
        return Observable12MacroLibraryV11RxContextWrapper(
            env,
            recovery_enabled=True,
            selective_recovery=True,
            selective_no_progress_threshold=float(wrapper_overrides.get("selective_no_progress_threshold", 1.10)),
            selective_collision_threshold=float(wrapper_overrides.get("selective_collision_threshold", 0.90)),
            force_recovery_high_speed=True,
            stuck_macro_explore_prob=float(wrapper_overrides.get("stuck_macro_explore_prob", 0.0)),
            stuck_macro_no_progress_threshold=float(wrapper_overrides.get("stuck_macro_no_progress_threshold", 0.55)),
            stuck_macro_collision_threshold=float(wrapper_overrides.get("stuck_macro_collision_threshold", 0.82)),
            stuck_macro_min_distance=wrapper_overrides.get("stuck_macro_min_distance", None),
            stuck_macro_max_distance=wrapper_overrides.get("stuck_macro_max_distance", None),
            stuck_macro_mid_distance_min=wrapper_overrides.get("stuck_macro_mid_distance_min", None),
            stuck_macro_mid_distance_max=wrapper_overrides.get("stuck_macro_mid_distance_max", None),
            stuck_macro_mid_max_triggers=wrapper_overrides.get("stuck_macro_mid_max_triggers", None),
            stuck_macro_mid_max_stuck_time=wrapper_overrides.get("stuck_macro_mid_max_stuck_time", None),
            stuck_macro_mid_choices=wrapper_overrides.get("stuck_macro_mid_choices", None),
            stuck_macro_mid_weights=wrapper_overrides.get("stuck_macro_mid_weights", None),
            stuck_macro_choices=tuple(wrapper_overrides.get("stuck_macro_choices", (1, 2, 3, 4, 5, 6, 7))),
            stuck_macro_weights=wrapper_overrides.get("stuck_macro_weights", None),
            targeted_stuck_macro_explore_prob=float(wrapper_overrides.get("targeted_stuck_macro_explore_prob", 0.0)),
            targeted_stuck_macro_no_progress_threshold=float(
                wrapper_overrides.get(
                    "targeted_stuck_macro_no_progress_threshold",
                    wrapper_overrides.get("stuck_macro_no_progress_threshold", 0.55),
                )
            ),
            targeted_stuck_macro_collision_threshold=float(
                wrapper_overrides.get(
                    "targeted_stuck_macro_collision_threshold",
                    wrapper_overrides.get("stuck_macro_collision_threshold", 0.82),
                )
            ),
            targeted_stuck_macro_angle_error_threshold=wrapper_overrides.get(
                "targeted_stuck_macro_angle_error_threshold",
                None,
            ),
            targeted_stuck_macro_distance_threshold=wrapper_overrides.get(
                "targeted_stuck_macro_distance_threshold",
                None,
            ),
            targeted_stuck_macro_max_distance_threshold=wrapper_overrides.get(
                "targeted_stuck_macro_max_distance_threshold",
                None,
            ),
            targeted_stuck_macro_stuck_time_threshold=wrapper_overrides.get(
                "targeted_stuck_macro_stuck_time_threshold",
                None,
            ),
            targeted_stuck_macro_mid_trigger_count_threshold=wrapper_overrides.get(
                "targeted_stuck_macro_mid_trigger_count_threshold",
                None,
            ),
            targeted_stuck_macro_allow_after_suppressed_policy=bool(
                wrapper_overrides.get("targeted_stuck_macro_allow_after_suppressed_policy", False)
            ),
            targeted_stuck_macro_choices=tuple(wrapper_overrides.get("targeted_stuck_macro_choices", ())),
            targeted_stuck_macro_weights=wrapper_overrides.get("targeted_stuck_macro_weights", None),
            suppressed_policy_macro_override_prob=float(
                wrapper_overrides.get("suppressed_policy_macro_override_prob", 0.0)
            ),
            suppressed_policy_macro_distance_threshold=wrapper_overrides.get(
                "suppressed_policy_macro_distance_threshold",
                None,
            ),
            suppressed_policy_macro_max_distance_threshold=wrapper_overrides.get(
                "suppressed_policy_macro_max_distance_threshold",
                None,
            ),
            suppressed_policy_macro_stuck_time_threshold=wrapper_overrides.get(
                "suppressed_policy_macro_stuck_time_threshold",
                None,
            ),
            suppressed_policy_macro_mid_trigger_count_threshold=wrapper_overrides.get(
                "suppressed_policy_macro_mid_trigger_count_threshold",
                None,
            ),
            suppressed_policy_macro_choices=tuple(wrapper_overrides.get("suppressed_policy_macro_choices", ())),
            suppressed_policy_macro_weights=wrapper_overrides.get("suppressed_policy_macro_weights", None),
            suppressed_policy_generic_redirect_prob=float(
                wrapper_overrides.get("suppressed_policy_generic_redirect_prob", 0.0)
            ),
            suppressed_policy_generic_redirect_distance_threshold=wrapper_overrides.get(
                "suppressed_policy_generic_redirect_distance_threshold",
                None,
            ),
            suppressed_policy_generic_redirect_max_distance_threshold=wrapper_overrides.get(
                "suppressed_policy_generic_redirect_max_distance_threshold",
                None,
            ),
            suppressed_policy_generic_redirect_stuck_time_threshold=wrapper_overrides.get(
                "suppressed_policy_generic_redirect_stuck_time_threshold",
                None,
            ),
            suppressed_policy_generic_redirect_mid_trigger_count_threshold=wrapper_overrides.get(
                "suppressed_policy_generic_redirect_mid_trigger_count_threshold",
                None,
            ),
            suppressed_policy_generic_redirect_choices=tuple(
                wrapper_overrides.get("suppressed_policy_generic_redirect_choices", ())
            ),
            suppressed_policy_generic_redirect_weights=wrapper_overrides.get(
                "suppressed_policy_generic_redirect_weights",
                None,
            ),
            suppressed_policy_macro_passthrough_prob=float(
                wrapper_overrides.get("suppressed_policy_macro_passthrough_prob", 0.0)
            ),
            suppressed_policy_macro_passthrough_bins=tuple(
                wrapper_overrides.get("suppressed_policy_macro_passthrough_bins", ())
            ),
            suppressed_policy_macro_passthrough_distance_threshold=wrapper_overrides.get(
                "suppressed_policy_macro_passthrough_distance_threshold",
                None,
            ),
            suppressed_policy_macro_passthrough_max_distance_threshold=wrapper_overrides.get(
                "suppressed_policy_macro_passthrough_max_distance_threshold",
                None,
            ),
            suppressed_policy_macro_passthrough_stuck_time_threshold=wrapper_overrides.get(
                "suppressed_policy_macro_passthrough_stuck_time_threshold",
                None,
            ),
            suppressed_policy_macro_passthrough_mid_trigger_count_threshold=wrapper_overrides.get(
                "suppressed_policy_macro_passthrough_mid_trigger_count_threshold",
                None,
            ),
            policy_macro_requires_release=bool(wrapper_overrides.get("policy_macro_requires_release", False)),
            policy_macro_release_gate_bins=tuple(wrapper_overrides.get("policy_macro_release_gate_bins", ())),
            policy_macro_rearm_on_clear=bool(wrapper_overrides.get("policy_macro_rearm_on_clear", False)),
            policy_macro_rearm_clear_time_threshold=float(
                wrapper_overrides.get("policy_macro_rearm_clear_time_threshold", 1.5)
            ),
            policy_macro_rearm_clear_no_progress_threshold=float(
                wrapper_overrides.get("policy_macro_rearm_clear_no_progress_threshold", 0.25)
            ),
            policy_macro_rearm_clear_stuck_threshold=float(
                wrapper_overrides.get("policy_macro_rearm_clear_stuck_threshold", 0.25)
            ),
            policy_macro_rearm_max_count=wrapper_overrides.get(
                "policy_macro_rearm_max_count",
                None,
            ),
            policy_macro_rearm_min_distance=wrapper_overrides.get(
                "policy_macro_rearm_min_distance",
                None,
            ),
            policy_macro_rearm_max_distance=wrapper_overrides.get(
                "policy_macro_rearm_max_distance",
                None,
            ),
        )
    if state_mode in {
        "observable12_target8_discrete_trigger_v9_stage1",
        "observable12_target8_discrete_trigger_v9_stage2",
    }:
        return Observable12DiscreteTriggerV9Wrapper(env)
    if state_mode in OBSERVABLE12_TARGET_MODES:
        return Observable12TargetRecoveryWrapper(env)
    if state_mode in OBSERVABLE7_STOP_MODES:
        return Observable7StopWrapper(env)
    if state_mode in OBSERVABLE11_STOP_MODES:
        return Observable11StopWrapper(env)
    return SelectObsWrapper(env, STATE_MODES[state_mode])


def make_base_env(seed: int) -> BlindNavEnv:
    return BlindNavEnv(
        seed=seed,
        tree_count=TREE_COUNT,
        mountain_count=MOUNTAIN_COUNT,
        tree_radius=TREE_RADIUS,
        max_steps=MAX_STEPS,
    )


def make_state_env(seed: int, state_mode: str, env_overrides: dict[str, dict[str, Any]] | None = None) -> BlindNavEnv:
    state_mode = canonical_env_mode(state_mode)
    kwargs: dict[str, Any] = {
        "seed": seed,
        "tree_count": TREE_COUNT,
        "mountain_count": MOUNTAIN_COUNT,
        "tree_radius": TREE_RADIUS,
        "max_steps": MAX_STEPS,
    }
    if state_mode == "observable6":
        kwargs.update({"action_mode": "stick_delta", "delta_angle_limit": np.pi / 2.0})
    elif state_mode == "observable8_delta45":
        kwargs.update({"action_mode": "stick_delta", "delta_angle_limit": np.pi / 4.0})
    elif state_mode == "observable8_target8_line_delta90":
        kwargs.update(
            {
                "world_size": 1200.0,
                "max_steps": 180,
                "action_mode": "stick_angle_delta",
                "delta_angle_limit": np.pi / 2.0,
                "binary_speed": True,
                "allow_zero_speed": False,
                "random_initial_heading": True,
                "tree_count": 45,
                "mountain_count": 8,
                "tree_radius": 18.0,
                "mountain_radius_range": (70.0, 160.0),
                "target_radius": 8.0,
                "stop_distance_range": None,
                "start_target_distance_range": (280.0, 700.0),
                "align_angle_tolerance": np.deg2rad(12.0),
                "reward_profile": "target_line",
                "position_obs_dt_range": (0.35, 0.75),
                "coord_noise_std": 1.5,
            }
        )
    elif state_mode == "observable8_target8_v2_offset60":
        kwargs.update(
            {
                "world_size": 1200.0,
                "max_steps": 180,
                "action_mode": "target_relative_offset",
                "delta_angle_limit": np.deg2rad(60.0),
                "binary_speed": True,
                "allow_zero_speed": False,
                "random_initial_heading": True,
                "tree_count": 45,
                "mountain_count": 8,
                "tree_radius": 18.0,
                "mountain_radius_range": (70.0, 160.0),
                "target_radius": 8.0,
                "stop_distance_range": None,
                "start_target_distance_range": (280.0, 700.0),
                "align_angle_tolerance": np.deg2rad(12.0),
                "reward_profile": "target_line_v2",
                "position_obs_dt_range": (0.35, 0.75),
                "coord_noise_std": 1.5,
            }
        )
    elif state_mode == "observable8_target8_recovery_angle":
        kwargs.update(
            {
                "world_size": 1200.0,
                "max_steps": 200,
                "action_mode": "target_relative_recovery",
                "delta_angle_limit": np.deg2rad(60.0),
                "binary_speed": True,
                "allow_zero_speed": False,
                "random_initial_heading": True,
                "tree_count": 45,
                "mountain_count": 10,
                "tree_radius": 18.0,
                "mountain_radius_range": (70.0, 160.0),
                "concave_mountain_probability": 0.78,
                "deep_concave_mountain_probability": 0.25,
                "target_radius": 8.0,
                "stop_distance_range": None,
                "start_target_distance_range": (280.0, 700.0),
                "align_angle_tolerance": np.deg2rad(10.0),
                "reward_profile": "target_line_recovery_angle",
                "position_obs_dt_range": (0.35, 0.75),
                "coord_noise_std": 1.5,
            }
        )
    elif state_mode == "observable8_target8_recovery_pressure":
        kwargs.update(
            {
                "world_size": 1200.0,
                "max_steps": 260,
                "action_mode": "target_relative_recovery",
                "delta_angle_limit": np.deg2rad(60.0),
                "binary_speed": True,
                "allow_zero_speed": False,
                "random_initial_heading": True,
                "tree_count": 45,
                "mountain_count": 18,
                "tree_radius": 18.0,
                "mountain_radius_range": (100.0, 190.0),
                "concave_mountain_probability": 0.95,
                "deep_concave_mountain_probability": 0.55,
                "target_radius": 8.0,
                "stop_distance_range": None,
                "start_target_distance_range": (450.0, 850.0),
                "align_angle_tolerance": np.deg2rad(10.0),
                "reward_profile": "target_line_recovery_pressure",
                "position_obs_dt_range": (0.35, 0.75),
                "coord_noise_std": 1.5,
            }
        )
    elif state_mode == "observable8_target8_full180_pressure":
        kwargs.update(
            {
                "world_size": 1200.0,
                "max_steps": 260,
                "action_mode": "target_relative_full",
                "binary_speed": True,
                "allow_zero_speed": False,
                "random_initial_heading": True,
                "tree_count": 45,
                "mountain_count": 18,
                "tree_radius": 18.0,
                "mountain_radius_range": (100.0, 190.0),
                "concave_mountain_probability": 0.95,
                "deep_concave_mountain_probability": 0.55,
                "target_radius": 8.0,
                "stop_distance_range": None,
                "start_target_distance_range": (450.0, 850.0),
                "align_angle_tolerance": np.deg2rad(10.0),
                "reward_profile": "target_line_full180_pressure",
                "position_obs_dt_range": (0.35, 0.75),
                "coord_noise_std": 1.5,
            }
        )
    elif state_mode == "observable8_target8_macro_recovery_dense_halfmountain":
        kwargs.update(
            {
                "world_size": 1200.0,
                "max_steps": 220,
                "action_mode": "target_relative_macro_recovery",
                "delta_angle_limit": np.deg2rad(60.0),
                "macro_recovery_steps": 4,
                "binary_speed": True,
                "allow_zero_speed": False,
                "random_initial_heading": True,
                "tree_count": 45,
                "mountain_count": 16,
                "tree_radius": 18.0,
                "mountain_radius_range": (35.0, 80.0),
                "concave_mountain_probability": 0.85,
                "deep_concave_mountain_probability": 0.30,
                "target_radius": 8.0,
                "stop_distance_range": None,
                "start_target_distance_range": (280.0, 700.0),
                "align_angle_tolerance": np.deg2rad(10.0),
                "reward_profile": "target_line_macro_recovery",
                "position_obs_dt_range": (0.35, 0.75),
                "coord_noise_std": 1.5,
            }
        )
    elif state_mode == "observable8_target8_macro_trigger_dense_halfmountain":
        kwargs.update(
            {
                "world_size": 1200.0,
                "max_steps": 220,
                "action_mode": "target_relative_macro_trigger_recovery",
                "delta_angle_limit": np.deg2rad(60.0),
                "macro_recovery_steps": 4,
                "binary_speed": True,
                "allow_zero_speed": False,
                "random_initial_heading": True,
                "tree_count": 45,
                "mountain_count": 16,
                "tree_radius": 18.0,
                "mountain_radius_range": (35.0, 80.0),
                "concave_mountain_probability": 0.85,
                "deep_concave_mountain_probability": 0.30,
                "target_radius": 8.0,
                "stop_distance_range": None,
                "start_target_distance_range": (280.0, 700.0),
                "align_angle_tolerance": np.deg2rad(10.0),
                "reward_profile": "target_line_macro_recovery",
                "position_obs_dt_range": (0.35, 0.75),
                "coord_noise_std": 1.5,
            }
        )
    elif state_mode == "observable12_target8_recovery_v4":
        kwargs.update(
            {
                "world_size": 1200.0,
                "max_steps": 220,
                "action_mode": "target_relative_macro_recovery",
                "delta_angle_limit": np.deg2rad(60.0),
                "macro_recovery_steps": 4,
                "binary_speed": True,
                "allow_zero_speed": False,
                "random_initial_heading": True,
                "tree_count": 45,
                "mountain_count": 16,
                "tree_radius": 18.0,
                "mountain_radius_range": (35.0, 80.0),
                "concave_mountain_probability": 0.85,
                "deep_concave_mountain_probability": 0.30,
                "target_radius": 8.0,
                "stop_distance_range": None,
                "start_target_distance_range": (280.0, 700.0),
                "align_angle_tolerance": np.deg2rad(10.0),
                "reward_profile": "target_line_macro_recovery",
                "position_obs_dt_range": (0.35, 0.75),
                "coord_noise_std": 1.5,
            }
        )
    elif state_mode == "observable12_target8_discrete_macro_v5":
        kwargs.update(
            {
                "world_size": 1200.0,
                "max_steps": 260,
                "action_mode": "target_relative_macro_recovery",
                "delta_angle_limit": np.deg2rad(60.0),
                "macro_recovery_steps": 6,
                "binary_speed": True,
                "allow_zero_speed": False,
                "random_initial_heading": True,
                "tree_count": 45,
                "mountain_count": 24,
                "tree_radius": 18.0,
                "mountain_radius_range": (35.0, 80.0),
                "concave_mountain_probability": 0.92,
                "deep_concave_mountain_probability": 0.45,
                "target_radius": 8.0,
                "stop_distance_range": None,
                "start_target_distance_range": (320.0, 780.0),
                "align_angle_tolerance": np.deg2rad(10.0),
                "reward_profile": "target_line_macro_recovery",
                "position_obs_dt_range": (0.35, 0.75),
                "coord_noise_std": 1.5,
            }
        )
    elif state_mode == "observable12_target8_discrete_macro_v6":
        kwargs.update(
            {
                "world_size": 1200.0,
                "max_steps": 260,
                "action_mode": "target_relative_macro_recovery",
                "delta_angle_limit": np.deg2rad(60.0),
                "macro_recovery_steps": 6,
                "binary_speed": True,
                "allow_zero_speed": False,
                "random_initial_heading": True,
                "tree_count": 45,
                "mountain_count": 28,
                "tree_radius": 18.0,
                "mountain_radius_range": (35.0, 85.0),
                "concave_mountain_probability": 0.96,
                "deep_concave_mountain_probability": 0.60,
                "target_radius": 8.0,
                "stop_distance_range": None,
                "start_target_distance_range": (360.0, 840.0),
                "align_angle_tolerance": np.deg2rad(10.0),
                "reward_profile": "target_line_macro_recovery_v6",
                "position_obs_dt_range": (0.35, 0.75),
                "coord_noise_std": 1.5,
            }
        )
    elif state_mode == "observable12_target8_discrete_macro_v7":
        kwargs.update(
            {
                "world_size": 1200.0,
                "max_steps": 280,
                "action_mode": "heading_relative_two_phase_macro_recovery",
                "delta_angle_limit": np.deg2rad(60.0),
                "macro_recovery_steps": 6,
                "binary_speed": True,
                "allow_zero_speed": False,
                "random_initial_heading": True,
                "tree_count": 45,
                "mountain_count": 32,
                "tree_radius": 18.0,
                "mountain_radius_range": (35.0, 85.0),
                "concave_mountain_probability": 0.98,
                "deep_concave_mountain_probability": 0.72,
                "target_radius": 8.0,
                "stop_distance_range": None,
                "start_target_distance_range": (360.0, 860.0),
                "align_angle_tolerance": np.deg2rad(10.0),
                "reward_profile": "target_line_macro_recovery_v7",
                "position_obs_dt_range": (0.35, 0.75),
                "coord_noise_std": 1.5,
            }
        )
    elif state_mode == "observable12_target8_discrete_macro_v8":
        kwargs.update(
            {
                "world_size": 1200.0,
                "max_steps": 280,
                "action_mode": "heading_relative_two_phase_macro_recovery",
                "delta_angle_limit": np.deg2rad(60.0),
                "macro_recovery_steps": 6,
                "binary_speed": True,
                "allow_zero_speed": False,
                "random_initial_heading": True,
                "tree_count": 45,
                "mountain_count": 34,
                "tree_radius": 18.0,
                "mountain_radius_range": (35.0, 85.0),
                "concave_mountain_probability": 0.99,
                "deep_concave_mountain_probability": 0.82,
                "target_radius": 8.0,
                "stop_distance_range": None,
                "start_target_distance_range": (360.0, 860.0),
                "align_angle_tolerance": np.deg2rad(10.0),
                "reward_profile": "target_line_macro_recovery_v7",
                "position_obs_dt_range": (0.35, 0.75),
                "coord_noise_std": 1.5,
                "trapped_start_probability": 0.55,
            }
        )
    elif state_mode == "observable12_target8_discrete_trigger_v9_stage1":
        kwargs.update(
            {
                "world_size": 1200.0,
                "max_steps": 220,
                "action_mode": "target_relative_macro_trigger_recovery",
                "delta_angle_limit": np.deg2rad(60.0),
                "macro_recovery_steps": 6,
                "binary_speed": True,
                "allow_zero_speed": False,
                "random_initial_heading": True,
                "tree_count": 45,
                "mountain_count": 12,
                "tree_radius": 18.0,
                "mountain_radius_range": (45.0, 90.0),
                "concave_mountain_probability": 0.84,
                "deep_concave_mountain_probability": 0.22,
                "target_radius": 8.0,
                "stop_distance_range": None,
                "start_target_distance_range": (300.0, 760.0),
                "align_angle_tolerance": np.deg2rad(10.0),
                "reward_profile": "target_line_macro_recovery_v7",
                "position_obs_dt_range": (0.35, 0.75),
                "coord_noise_std": 1.5,
                "opposite_heading_probability": 0.35,
            }
        )
    elif state_mode == "observable12_target8_discrete_trigger_v9_stage2":
        kwargs.update(
            {
                "world_size": 1200.0,
                "max_steps": 280,
                "action_mode": "target_relative_macro_trigger_recovery",
                "delta_angle_limit": np.deg2rad(60.0),
                "macro_recovery_steps": 6,
                "binary_speed": True,
                "allow_zero_speed": False,
                "random_initial_heading": True,
                "tree_count": 45,
                "mountain_count": 34,
                "tree_radius": 18.0,
                "mountain_radius_range": (35.0, 85.0),
                "concave_mountain_probability": 0.99,
                "deep_concave_mountain_probability": 0.82,
                "target_radius": 8.0,
                "stop_distance_range": None,
                "start_target_distance_range": (360.0, 860.0),
                "align_angle_tolerance": np.deg2rad(10.0),
                "reward_profile": "target_line_macro_recovery_v7",
                "position_obs_dt_range": (0.45, 0.85),
                "coord_noise_std": 1.8,
                "trapped_start_probability": 0.55,
                "opposite_heading_probability": 0.50,
            }
        )
    elif state_mode == "observable12_target8_discrete_recovery_v10_stage1":
        kwargs.update(
            {
                "world_size": 1200.0,
                "max_steps": 220,
                "action_mode": "target_relative_macro_recovery",
                "delta_angle_limit": np.deg2rad(60.0),
                "macro_recovery_steps": 6,
                "binary_speed": True,
                "allow_zero_speed": False,
                "random_initial_heading": True,
                "tree_count": 45,
                "mountain_count": 10,
                "tree_radius": 18.0,
                "mountain_radius_range": (45.0, 90.0),
                "concave_mountain_probability": 0.80,
                "deep_concave_mountain_probability": 0.18,
                "target_radius": 8.0,
                "stop_distance_range": None,
                "start_target_distance_range": (300.0, 760.0),
                "align_angle_tolerance": np.deg2rad(10.0),
                "reward_profile": "target_line_macro_recovery_v10_stage1",
                "position_obs_dt_range": (0.35, 0.75),
                "coord_noise_std": 1.5,
                "opposite_heading_probability": 0.35,
            }
        )
    elif state_mode == "observable12_target8_discrete_recovery_v10_stage2a":
        kwargs.update(
            {
                "world_size": 1200.0,
                "max_steps": 240,
                "action_mode": "heading_relative_two_phase_macro_recovery",
                "delta_angle_limit": np.deg2rad(60.0),
                "macro_recovery_steps": 6,
                "binary_speed": True,
                "allow_zero_speed": False,
                "random_initial_heading": True,
                "tree_count": 45,
                "mountain_count": 16,
                "tree_radius": 18.0,
                "mountain_radius_range": (40.0, 85.0),
                "concave_mountain_probability": 0.86,
                "deep_concave_mountain_probability": 0.22,
                "target_radius": 8.0,
                "stop_distance_range": None,
                "start_target_distance_range": (280.0, 700.0),
                "align_angle_tolerance": np.deg2rad(10.0),
                "reward_profile": "target_line_macro_recovery_v10_stage2",
                "position_obs_dt_range": (0.38, 0.78),
                "coord_noise_std": 1.55,
                "trapped_start_probability": 0.18,
                "opposite_heading_probability": 0.40,
            }
        )
    elif state_mode == "observable12_target8_discrete_recovery_v10_stage2":
        kwargs.update(
            {
                "world_size": 1200.0,
                "max_steps": 260,
                "action_mode": "heading_relative_two_phase_macro_recovery",
                "delta_angle_limit": np.deg2rad(60.0),
                "macro_recovery_steps": 6,
                "binary_speed": True,
                "allow_zero_speed": False,
                "random_initial_heading": True,
                "tree_count": 45,
                "mountain_count": 24,
                "tree_radius": 18.0,
                "mountain_radius_range": (35.0, 85.0),
                "concave_mountain_probability": 0.92,
                "deep_concave_mountain_probability": 0.45,
                "target_radius": 8.0,
                "stop_distance_range": None,
                "start_target_distance_range": (260.0, 680.0),
                "align_angle_tolerance": np.deg2rad(10.0),
                "reward_profile": "target_line_macro_recovery_v10_stage2",
                "position_obs_dt_range": (0.40, 0.80),
                "coord_noise_std": 1.6,
                "trapped_start_probability": 0.30,
                "opposite_heading_probability": 0.45,
            }
        )
    elif state_mode == "observable12_target8_discrete_recovery_v10_stage3":
        kwargs.update(
            {
                "world_size": 1200.0,
                "max_steps": 280,
                "action_mode": "heading_relative_two_phase_macro_recovery",
                "delta_angle_limit": np.deg2rad(60.0),
                "macro_recovery_steps": 6,
                "binary_speed": True,
                "allow_zero_speed": False,
                "random_initial_heading": True,
                "tree_count": 45,
                "mountain_count": 30,
                "tree_radius": 18.0,
                "mountain_radius_range": (35.0, 85.0),
                "concave_mountain_probability": 0.97,
                "deep_concave_mountain_probability": 0.60,
                "target_radius": 8.0,
                "stop_distance_range": None,
                "start_target_distance_range": (260.0, 860.0),
                "align_angle_tolerance": np.deg2rad(10.0),
                "reward_profile": "target_line_macro_recovery_v10",
                "position_obs_dt_range": (0.45, 0.85),
                "coord_noise_std": 1.8,
                "trapped_start_probability": 0.50,
                "opposite_heading_probability": 0.42,
            }
        )
    elif state_mode == "observable12_target8_macro_library_v11_stage1":
        kwargs.update(
            {
                "world_size": 1200.0,
                "max_steps": 220,
                "action_mode": "heading_relative_macro_library_recovery",
                "delta_angle_limit": np.deg2rad(60.0),
                "macro_recovery_steps": 6,
                "binary_speed": True,
                "allow_zero_speed": False,
                "random_initial_heading": True,
                "tree_count": 45,
                "mountain_count": 10,
                "tree_radius": 18.0,
                "mountain_radius_range": (45.0, 90.0),
                "concave_mountain_probability": 0.80,
                "deep_concave_mountain_probability": 0.18,
                "target_radius": 8.0,
                "stop_distance_range": None,
                "start_target_distance_range": (300.0, 760.0),
                "align_angle_tolerance": np.deg2rad(10.0),
                "reward_profile": "target_line_macro_recovery_v10_stage1",
                "position_obs_dt_range": (0.35, 0.75),
                "coord_noise_std": 1.5,
                "opposite_heading_probability": 0.35,
            }
        )
    elif state_mode == "observable12_target8_macro_library_v11_stage2a":
        kwargs.update(
            {
                "world_size": 1200.0,
                "max_steps": 240,
                "action_mode": "heading_relative_macro_library_recovery",
                "delta_angle_limit": np.deg2rad(60.0),
                "macro_recovery_steps": 6,
                "binary_speed": True,
                "allow_zero_speed": False,
                "random_initial_heading": True,
                "tree_count": 45,
                "mountain_count": 16,
                "tree_radius": 18.0,
                "mountain_radius_range": (40.0, 85.0),
                "concave_mountain_probability": 0.86,
                "deep_concave_mountain_probability": 0.22,
                "target_radius": 8.0,
                "stop_distance_range": None,
                "start_target_distance_range": (280.0, 700.0),
                "align_angle_tolerance": np.deg2rad(10.0),
                "reward_profile": "target_line_macro_recovery_v10_stage2",
                "position_obs_dt_range": (0.38, 0.78),
                "coord_noise_std": 1.55,
                "trapped_start_probability": 0.18,
                "opposite_heading_probability": 0.40,
            }
        )
    elif state_mode == "observable12_target8_macro_library_v11_stage2":
        kwargs.update(
            {
                "world_size": 1200.0,
                "max_steps": 260,
                "action_mode": "heading_relative_macro_library_recovery",
                "delta_angle_limit": np.deg2rad(60.0),
                "macro_recovery_steps": 6,
                "binary_speed": True,
                "allow_zero_speed": False,
                "random_initial_heading": True,
                "tree_count": 45,
                "mountain_count": 24,
                "tree_radius": 18.0,
                "mountain_radius_range": (35.0, 85.0),
                "concave_mountain_probability": 0.92,
                "deep_concave_mountain_probability": 0.45,
                "target_radius": 8.0,
                "stop_distance_range": None,
                "start_target_distance_range": (260.0, 680.0),
                "align_angle_tolerance": np.deg2rad(10.0),
                "reward_profile": "target_line_macro_recovery_v10_stage2",
                "position_obs_dt_range": (0.40, 0.80),
                "coord_noise_std": 1.6,
                "trapped_start_probability": 0.30,
                "opposite_heading_probability": 0.45,
            }
        )
    elif state_mode == "observable12_target8_macro_library_v11_stage3":
        kwargs.update(
            {
                "world_size": 1200.0,
                "max_steps": 280,
                "action_mode": "heading_relative_macro_library_recovery",
                "delta_angle_limit": np.deg2rad(60.0),
                "macro_recovery_steps": 6,
                "binary_speed": True,
                "allow_zero_speed": False,
                "random_initial_heading": True,
                "tree_count": 45,
                "mountain_count": 30,
                "tree_radius": 18.0,
                "mountain_radius_range": (35.0, 85.0),
                "concave_mountain_probability": 0.97,
                "deep_concave_mountain_probability": 0.60,
                "target_radius": 8.0,
                "stop_distance_range": None,
                "start_target_distance_range": (260.0, 860.0),
                "align_angle_tolerance": np.deg2rad(10.0),
                "reward_profile": "target_line_macro_recovery_v10",
                "position_obs_dt_range": (0.45, 0.85),
                "coord_noise_std": 1.8,
                "trapped_start_probability": 0.50,
                "opposite_heading_probability": 0.42,
            }
        )
    elif state_mode == "observable12_target8_macro_library_v11ds_stage2":
        kwargs.update(
            {
                "world_size": 1200.0,
                "max_steps": 260,
                "action_mode": "heading_relative_macro_library_recovery",
                "delta_angle_limit": np.deg2rad(60.0),
                "macro_recovery_steps": 6,
                "binary_speed": True,
                "allow_zero_speed": False,
                "random_initial_heading": True,
                "tree_count": 45,
                "mountain_count": 24,
                "tree_radius": 18.0,
                "mountain_radius_range": (35.0, 85.0),
                "concave_mountain_probability": 0.92,
                "deep_concave_mountain_probability": 0.45,
                "target_radius": 8.0,
                "stop_distance_range": None,
                "start_target_distance_range": (260.0, 680.0),
                "align_angle_tolerance": np.deg2rad(10.0),
                "reward_profile": "target_line_macro_recovery_v10_stage2",
                "position_obs_dt_range": (0.40, 0.80),
                "coord_noise_std": 1.6,
                "trapped_start_probability": 0.30,
                "opposite_heading_probability": 0.45,
            }
        )
    elif state_mode == "observable12_target8_macro_library_v11dsl_stage2":
        kwargs.update(
            {
                "world_size": 1200.0,
                "max_steps": 260,
                "action_mode": "heading_relative_macro_library_recovery",
                "delta_angle_limit": np.deg2rad(60.0),
                "macro_recovery_steps": 6,
                "binary_speed": True,
                "allow_zero_speed": False,
                "random_initial_heading": True,
                "tree_count": 45,
                "mountain_count": 24,
                "tree_radius": 18.0,
                "mountain_radius_range": (35.0, 85.0),
                "concave_mountain_probability": 0.92,
                "deep_concave_mountain_probability": 0.45,
                "target_radius": 8.0,
                "stop_distance_range": None,
                "start_target_distance_range": (260.0, 680.0),
                "align_angle_tolerance": np.deg2rad(10.0),
                "reward_profile": "target_line_macro_recovery_v10_stage2",
                "position_obs_dt_range": (0.40, 0.80),
                "coord_noise_std": 1.6,
                "trapped_start_probability": 0.30,
                "opposite_heading_probability": 0.45,
            }
        )
    elif state_mode == "observable12_target8_macro_library_v11tm_stage2":
        kwargs.update(
            {
                "world_size": 1200.0,
                "max_steps": 260,
                "action_mode": "heading_relative_macro_library_recovery",
                "delta_angle_limit": np.deg2rad(60.0),
                "macro_recovery_steps": 6,
                "binary_speed": True,
                "allow_zero_speed": False,
                "random_initial_heading": True,
                "tree_count": 45,
                "mountain_count": 24,
                "tree_radius": 18.0,
                "mountain_radius_range": (35.0, 85.0),
                "concave_mountain_probability": 0.92,
                "deep_concave_mountain_probability": 0.45,
                "target_radius": 8.0,
                "stop_distance_range": None,
                "start_target_distance_range": (260.0, 680.0),
                "align_angle_tolerance": np.deg2rad(10.0),
                "reward_profile": "target_line_macro_recovery_v10_stage2",
                "position_obs_dt_range": (0.40, 0.80),
                "coord_noise_std": 1.6,
                "trapped_start_probability": 0.30,
                "opposite_heading_probability": 0.45,
            }
        )
    elif state_mode == "observable12_target8_macro_library_v11re_stage2":
        kwargs.update(
            {
                "world_size": 1200.0,
                "max_steps": 260,
                "action_mode": "heading_relative_macro_library_recovery",
                "delta_angle_limit": np.deg2rad(60.0),
                "macro_recovery_steps": 6,
                "binary_speed": True,
                "allow_zero_speed": False,
                "random_initial_heading": True,
                "tree_count": 45,
                "mountain_count": 24,
                "tree_radius": 18.0,
                "mountain_radius_range": (35.0, 85.0),
                "concave_mountain_probability": 0.92,
                "deep_concave_mountain_probability": 0.45,
                "target_radius": 8.0,
                "stop_distance_range": None,
                "start_target_distance_range": (260.0, 680.0),
                "align_angle_tolerance": np.deg2rad(10.0),
                "reward_profile": "target_line_macro_recovery_v10",
                "position_obs_dt_range": (0.40, 0.80),
                "coord_noise_std": 1.6,
                "trapped_start_probability": 0.30,
                "opposite_heading_probability": 0.45,
            }
        )
    elif state_mode == "observable12_target8_macro_library_v11rx_stage2":
        kwargs.update(
            {
                "world_size": 1200.0,
                "max_steps": 260,
                "action_mode": "heading_relative_macro_library_recovery",
                "delta_angle_limit": np.deg2rad(60.0),
                "macro_recovery_steps": 6,
                "binary_speed": True,
                "allow_zero_speed": False,
                "random_initial_heading": True,
                "tree_count": 45,
                "mountain_count": 24,
                "tree_radius": 18.0,
                "mountain_radius_range": (35.0, 85.0),
                "concave_mountain_probability": 0.92,
                "deep_concave_mountain_probability": 0.45,
                "target_radius": 8.0,
                "stop_distance_range": None,
                "start_target_distance_range": (260.0, 680.0),
                "align_angle_tolerance": np.deg2rad(10.0),
                "reward_profile": "target_line_macro_recovery_v10",
                "position_obs_dt_range": (0.40, 0.80),
                "coord_noise_std": 1.6,
                "trapped_start_probability": 0.30,
                "opposite_heading_probability": 0.45,
            }
        )
    elif state_mode == "observable12_target8_macro_library_v11rxc_stage2":
        kwargs.update(
            {
                "world_size": 1200.0,
                "max_steps": 260,
                "action_mode": "heading_relative_macro_library_recovery",
                "delta_angle_limit": np.deg2rad(60.0),
                "macro_recovery_steps": 6,
                "binary_speed": True,
                "allow_zero_speed": False,
                "random_initial_heading": True,
                "tree_count": 45,
                "mountain_count": 24,
                "tree_radius": 18.0,
                "mountain_radius_range": (35.0, 85.0),
                "concave_mountain_probability": 0.92,
                "deep_concave_mountain_probability": 0.45,
                "target_radius": 8.0,
                "stop_distance_range": None,
                "start_target_distance_range": (260.0, 680.0),
                "align_angle_tolerance": np.deg2rad(10.0),
                "reward_profile": "target_line_macro_recovery_v10",
                "position_obs_dt_range": (0.40, 0.80),
                "coord_noise_std": 1.6,
                "trapped_start_probability": 0.30,
                "opposite_heading_probability": 0.45,
            }
        )
    elif state_mode == "observable12_target8_discrete_recovery_v10b9_stage1":
        kwargs.update(
            {
                "world_size": 1200.0,
                "max_steps": 220,
                "action_mode": "target_relative_macro_recovery",
                "delta_angle_limit": np.deg2rad(60.0),
                "macro_recovery_steps": 6,
                "binary_speed": True,
                "allow_zero_speed": False,
                "random_initial_heading": True,
                "tree_count": 45,
                "mountain_count": 10,
                "tree_radius": 18.0,
                "mountain_radius_range": (45.0, 90.0),
                "concave_mountain_probability": 0.80,
                "deep_concave_mountain_probability": 0.18,
                "target_radius": 8.0,
                "stop_distance_range": None,
                "start_target_distance_range": (300.0, 760.0),
                "align_angle_tolerance": np.deg2rad(10.0),
                "reward_profile": "target_line_macro_recovery_v10_stage1",
                "position_obs_dt_range": (0.35, 0.75),
                "coord_noise_std": 1.5,
                "opposite_heading_probability": 0.35,
            }
        )
    elif state_mode == "observable12_target8_discrete_recovery_v10b9_stage2a":
        kwargs.update(
            {
                "world_size": 1200.0,
                "max_steps": 240,
                "action_mode": "heading_relative_two_phase_macro_recovery",
                "delta_angle_limit": np.deg2rad(60.0),
                "macro_recovery_steps": 6,
                "binary_speed": True,
                "allow_zero_speed": False,
                "random_initial_heading": True,
                "tree_count": 45,
                "mountain_count": 16,
                "tree_radius": 18.0,
                "mountain_radius_range": (40.0, 85.0),
                "concave_mountain_probability": 0.86,
                "deep_concave_mountain_probability": 0.22,
                "target_radius": 8.0,
                "stop_distance_range": None,
                "start_target_distance_range": (280.0, 700.0),
                "align_angle_tolerance": np.deg2rad(10.0),
                "reward_profile": "target_line_macro_recovery_v10_stage2",
                "position_obs_dt_range": (0.38, 0.78),
                "coord_noise_std": 1.55,
                "trapped_start_probability": 0.18,
                "opposite_heading_probability": 0.40,
            }
        )
    elif state_mode == "observable12_target8_discrete_recovery_v10b9_stage2":
        kwargs.update(
            {
                "world_size": 1200.0,
                "max_steps": 260,
                "action_mode": "heading_relative_two_phase_macro_recovery",
                "delta_angle_limit": np.deg2rad(60.0),
                "macro_recovery_steps": 6,
                "binary_speed": True,
                "allow_zero_speed": False,
                "random_initial_heading": True,
                "tree_count": 45,
                "mountain_count": 24,
                "tree_radius": 18.0,
                "mountain_radius_range": (35.0, 85.0),
                "concave_mountain_probability": 0.92,
                "deep_concave_mountain_probability": 0.45,
                "target_radius": 8.0,
                "stop_distance_range": None,
                "start_target_distance_range": (260.0, 680.0),
                "align_angle_tolerance": np.deg2rad(10.0),
                "reward_profile": "target_line_macro_recovery_v10_stage2",
                "position_obs_dt_range": (0.40, 0.80),
                "coord_noise_std": 1.6,
                "trapped_start_probability": 0.30,
                "opposite_heading_probability": 0.45,
            }
        )
    elif state_mode == "observable12_target8_discrete_recovery_v10b9_stage3":
        kwargs.update(
            {
                "world_size": 1200.0,
                "max_steps": 280,
                "action_mode": "heading_relative_two_phase_macro_recovery",
                "delta_angle_limit": np.deg2rad(60.0),
                "macro_recovery_steps": 6,
                "binary_speed": True,
                "allow_zero_speed": False,
                "random_initial_heading": True,
                "tree_count": 45,
                "mountain_count": 30,
                "tree_radius": 18.0,
                "mountain_radius_range": (35.0, 85.0),
                "concave_mountain_probability": 0.97,
                "deep_concave_mountain_probability": 0.60,
                "target_radius": 8.0,
                "stop_distance_range": None,
                "start_target_distance_range": (260.0, 860.0),
                "align_angle_tolerance": np.deg2rad(10.0),
                "reward_profile": "target_line_macro_recovery_v10",
                "position_obs_dt_range": (0.45, 0.85),
                "coord_noise_std": 1.8,
                "trapped_start_probability": 0.50,
                "opposite_heading_probability": 0.42,
            }
        )
    elif state_mode == "observable9_target8_continuous_fixed_recovery_v1":
        kwargs.update(
            {
                "world_size": 1200.0,
                "max_steps": 280,
                "action_mode": "continuous_trigger_fixed_recovery",
                "delta_angle_limit": np.deg2rad(60.0),
                "fixed_recovery_back_duration": 2.0,
                "fixed_recovery_side_duration": 2.0,
                "binary_speed": True,
                "allow_zero_speed": False,
                "random_initial_heading": True,
                "tree_count": 45,
                "mountain_count": 30,
                "tree_radius": 18.0,
                "mountain_radius_range": (35.0, 85.0),
                "concave_mountain_probability": 0.97,
                "deep_concave_mountain_probability": 0.60,
                "target_radius": 8.0,
                "stop_distance_range": None,
                "start_target_distance_range": (260.0, 860.0),
                "align_angle_tolerance": np.deg2rad(10.0),
                "reward_profile": "target_line_macro_recovery_v10",
                "position_obs_dt_range": (0.45, 0.85),
                "coord_noise_std": 1.8,
                "trapped_start_probability": 0.50,
                "opposite_heading_probability": 0.42,
                "collision_slide_enabled": False,
            }
        )
    elif state_mode == "observable10_target8_continuous_fixed_recovery_v2":
        kwargs.update(
            {
                "world_size": 1200.0,
                "max_steps": 280,
                "action_mode": "continuous_trigger_fixed_recovery",
                "delta_angle_limit": np.deg2rad(60.0),
                "fixed_recovery_back_duration": 2.0,
                "fixed_recovery_side_duration": 2.0,
                "fixed_recovery_trigger_threshold": 0.1,
                "binary_speed": True,
                "allow_zero_speed": False,
                "random_initial_heading": True,
                "tree_count": 45,
                "mountain_count": 34,
                "tree_radius": 18.0,
                "mountain_radius_range": (35.0, 85.0),
                "concave_mountain_probability": 0.98,
                "deep_concave_mountain_probability": 0.70,
                "target_radius": 8.0,
                "stop_distance_range": None,
                "start_target_distance_range": (260.0, 860.0),
                "align_angle_tolerance": np.deg2rad(10.0),
                "reward_profile": "target_line_macro_recovery_v10",
                "position_obs_dt_range": (0.45, 0.85),
                "coord_noise_std": 1.8,
                "trapped_start_probability": 0.65,
                "opposite_heading_probability": 0.50,
                "collision_slide_enabled": False,
            }
        )
    elif state_mode == "observable10_target8_continuous_fixed_recovery_v3":
        kwargs.update(
            {
                "world_size": 1200.0,
                "max_steps": 280,
                "action_mode": "continuous_trigger_fixed_recovery",
                "delta_angle_limit": np.deg2rad(60.0),
                "fixed_recovery_back_duration": 2.0,
                "fixed_recovery_side_duration": 2.0,
                "fixed_recovery_trigger_threshold": 0.2,
                "fixed_recovery_cooldown_duration": 1.2,
                "binary_speed": True,
                "allow_zero_speed": False,
                "random_initial_heading": True,
                "tree_count": 45,
                "mountain_count": 34,
                "tree_radius": 18.0,
                "mountain_radius_range": (35.0, 85.0),
                "concave_mountain_probability": 0.98,
                "deep_concave_mountain_probability": 0.70,
                "target_radius": 8.0,
                "stop_distance_range": None,
                "start_target_distance_range": (260.0, 860.0),
                "align_angle_tolerance": np.deg2rad(10.0),
                "reward_profile": "target_line_macro_recovery_v10",
                "position_obs_dt_range": (0.45, 0.85),
                "coord_noise_std": 1.8,
                "trapped_start_probability": 0.65,
                "opposite_heading_probability": 0.50,
                "collision_slide_enabled": False,
            }
        )
    elif state_mode == "observable10_target8_continuous_fixed_recovery_v4":
        kwargs.update(
            {
                "world_size": 1200.0,
                "max_steps": 280,
                "action_mode": "continuous_trigger_fixed_recovery",
                "delta_angle_limit": np.deg2rad(45.0),
                "fixed_recovery_back_duration": 2.0,
                "fixed_recovery_side_duration": 2.0,
                "fixed_recovery_trigger_threshold": 0.35,
                "fixed_recovery_cooldown_duration": 1.5,
                "binary_speed": True,
                "allow_zero_speed": False,
                "random_initial_heading": True,
                "tree_count": 45,
                "mountain_count": 34,
                "tree_radius": 18.0,
                "mountain_radius_range": (35.0, 85.0),
                "concave_mountain_probability": 0.98,
                "deep_concave_mountain_probability": 0.70,
                "target_radius": 8.0,
                "stop_distance_range": None,
                "start_target_distance_range": (260.0, 860.0),
                "align_angle_tolerance": np.deg2rad(10.0),
                "reward_profile": "target_line_v2_fixed_trigger",
                "position_obs_dt_range": (0.45, 0.85),
                "coord_noise_std": 1.8,
                "trapped_start_probability": 0.65,
                "opposite_heading_probability": 0.50,
                "collision_slide_enabled": False,
            }
        )
    elif state_mode == "observable10_target8_continuous_fixed_recovery_v5":
        kwargs.update(
            {
                "world_size": 1200.0,
                "max_steps": 280,
                "action_mode": "continuous_trigger_fixed_recovery",
                "delta_angle_limit": np.deg2rad(45.0),
                "fixed_recovery_back_duration": 2.0,
                "fixed_recovery_side_duration": 2.0,
                "fixed_recovery_trigger_threshold": 0.35,
                "fixed_recovery_cooldown_duration": 1.5,
                "fixed_recovery_policy_side_control": True,
                "binary_speed": True,
                "allow_zero_speed": False,
                "random_initial_heading": True,
                "tree_count": 45,
                "mountain_count": 34,
                "tree_radius": 18.0,
                "mountain_radius_range": (35.0, 85.0),
                "concave_mountain_probability": 0.98,
                "deep_concave_mountain_probability": 0.70,
                "target_radius": 8.0,
                "stop_distance_range": None,
                "start_target_distance_range": (260.0, 860.0),
                "align_angle_tolerance": np.deg2rad(10.0),
                "reward_profile": "target_line_macro_recovery_v10",
                "position_obs_dt_range": (0.45, 0.85),
                "coord_noise_std": 1.8,
                "trapped_start_probability": 0.65,
                "opposite_heading_probability": 0.50,
                "collision_slide_enabled": False,
            }
        )
    elif state_mode == "observable10_target8_continuous_fixed_recovery_v6":
        kwargs.update(
            {
                "world_size": 1200.0,
                "max_steps": 280,
                "action_mode": "continuous_trigger_fixed_recovery",
                "delta_angle_limit": np.deg2rad(45.0),
                "fixed_recovery_back_duration": 2.0,
                "fixed_recovery_side_duration": 2.0,
                "fixed_recovery_trigger_threshold": 0.35,
                "fixed_recovery_cooldown_duration": 1.5,
                "fixed_recovery_policy_side_control": True,
                "fixed_recovery_auto_trigger": True,
                "fixed_recovery_auto_no_progress_threshold": 0.75,
                "fixed_recovery_auto_collision_threshold": 0.72,
                "binary_speed": True,
                "allow_zero_speed": False,
                "random_initial_heading": True,
                "tree_count": 45,
                "mountain_count": 34,
                "tree_radius": 18.0,
                "mountain_radius_range": (35.0, 85.0),
                "concave_mountain_probability": 0.98,
                "deep_concave_mountain_probability": 0.70,
                "target_radius": 8.0,
                "stop_distance_range": None,
                "start_target_distance_range": (260.0, 860.0),
                "align_angle_tolerance": np.deg2rad(10.0),
                "reward_profile": "target_line_macro_recovery_v10",
                "position_obs_dt_range": (0.45, 0.85),
                "coord_noise_std": 1.8,
                "trapped_start_probability": 0.65,
                "opposite_heading_probability": 0.50,
                "collision_slide_enabled": False,
            }
        )
    elif state_mode == "observable7_stop_stage1_delta45":
        kwargs.update(
            {
                "action_mode": "stick_delta",
                "delta_angle_limit": np.pi / 4.0,
                "allow_zero_speed": False,
                "random_initial_heading": False,
                "tree_count": 16,
                "mountain_count": 3,
                "target_radius": 55.0,
                "stop_distance_range": None,
                "start_target_distance_range": (160.0, 360.0),
            }
        )
    elif state_mode == "observable7_stop_stage2_delta45":
        kwargs.update(
            {
                "action_mode": "stick_delta",
                "delta_angle_limit": np.pi / 4.0,
                "allow_zero_speed": True,
                "random_initial_heading": False,
                "tree_count": 48,
                "mountain_count": 8,
                "stop_distance_range": (35.0, 55.0),
                "start_target_distance_range": (260.0, 560.0),
                "stop_distance_tolerance": 12.0,
                "align_angle_tolerance": np.pi,
                "stop_speed_threshold": 100.0,
            }
        )
    elif state_mode == "observable7_stop_stage3_delta45":
        kwargs.update(
            {
                "action_mode": "stick_delta",
                "delta_angle_limit": np.pi / 4.0,
                "allow_zero_speed": True,
                "random_initial_heading": False,
                "tree_count": 72,
                "mountain_count": 12,
                "stop_distance_range": (28.0, 42.0),
                "start_target_distance_range": (320.0, 680.0),
                "stop_distance_tolerance": 10.0,
                "align_angle_tolerance": np.pi,
                "stop_speed_threshold": 100.0,
            }
        )
    elif state_mode == "observable7_stop_stage3_heading_delta45":
        kwargs.update(
            {
                "action_mode": "stick_delta",
                "delta_angle_limit": np.pi / 4.0,
                "allow_zero_speed": True,
                "random_initial_heading": True,
                "tree_count": 72,
                "mountain_count": 12,
                "stop_distance_range": (24.0, 42.0),
                "start_target_distance_range": (320.0, 720.0),
                "stop_distance_tolerance": 10.0,
                "align_angle_tolerance": np.deg2rad(60.0),
                "stop_speed_threshold": 100.0,
            }
        )
    elif state_mode == "observable7_stop_final_warm1_delta45":
        kwargs.update(
            {
                "action_mode": "stick_delta",
                "delta_angle_limit": np.pi / 4.0,
                "allow_zero_speed": True,
                "random_initial_heading": True,
                "tree_count": 90,
                "mountain_count": 16,
                "stop_distance_range": (18.0, 34.0),
                "start_target_distance_range": (480.0, 920.0),
                "stop_distance_tolerance": 8.0,
                "align_angle_tolerance": np.deg2rad(40.0),
                "stop_speed_threshold": 70.0,
            }
        )
    elif state_mode == "observable7_stop_final_warm2_delta45":
        kwargs.update(
            {
                "action_mode": "stick_delta",
                "delta_angle_limit": np.pi / 4.0,
                "allow_zero_speed": True,
                "random_initial_heading": True,
                "tree_count": 90,
                "mountain_count": 16,
                "stop_distance_range": (12.0, 26.0),
                "start_target_distance_range": (650.0, 1150.0),
                "stop_distance_tolerance": 5.0,
                "align_angle_tolerance": np.deg2rad(24.0),
                "stop_speed_threshold": 35.0,
            }
        )
    elif state_mode == "observable7_stop_delta45":
        kwargs.update(
            {
                "action_mode": "stick_delta",
                "delta_angle_limit": np.pi / 4.0,
                "allow_zero_speed": True,
                "random_initial_heading": True,
                "stop_distance_range": (8.0, 20.0),
                "start_target_distance_range": (800.0, 1400.0),
                "stop_distance_tolerance": 3.0,
                "align_angle_tolerance": np.deg2rad(10.0),
                "stop_speed_threshold": 12.0,
            }
        )
    elif state_mode == "observable7_stop_quality_delta60":
        kwargs.update(
            {
                "action_mode": "stick_delta",
                "delta_angle_limit": np.deg2rad(60.0),
                "allow_zero_speed": True,
                "random_initial_heading": True,
                "tree_count": 90,
                "mountain_count": 16,
                "stop_distance_range": (8.0, 20.0),
                "start_target_distance_range": (800.0, 1400.0),
                "stop_distance_tolerance": 3.0,
                "align_angle_tolerance": np.deg2rad(10.0),
                "stop_speed_threshold": 12.0,
                "reward_profile": "quality_stop",
                "position_obs_dt_range": (0.35, 0.75),
                "coord_noise_std": 2.0,
            }
        )
    elif state_mode == "observable7_stop_line_delta45":
        kwargs.update(
            {
                "action_mode": "target_tracking",
                "delta_angle_limit": np.deg2rad(45.0),
                "tracking_residual_limit": np.deg2rad(12.0),
                "allow_zero_speed": True,
                "random_initial_heading": True,
                "tree_count": 90,
                "mountain_count": 16,
                "stop_distance_range": (8.0, 20.0),
                "start_target_distance_range": (800.0, 1400.0),
                "stop_distance_tolerance": 3.0,
                "align_angle_tolerance": np.deg2rad(10.0),
                "stop_speed_threshold": 12.0,
                "reward_profile": "line_stop",
                "position_obs_dt_range": (0.35, 0.75),
                "coord_noise_std": 1.5,
            }
        )
    elif state_mode == "observable7_stop_curriculum_delta45":
        kwargs.update(
            {
                "action_mode": "stick_delta",
                "delta_angle_limit": np.pi / 4.0,
                "allow_zero_speed": True,
                "random_initial_heading": True,
                "stop_distance_range": (8.0, 20.0),
                "start_target_distance_range": (420.0, 820.0),
                "stop_distance_tolerance": 10.0,
                "align_angle_tolerance": np.deg2rad(35.0),
                "stop_speed_threshold": 28.0,
            }
        )
    kwargs.update(resolve_env_overrides(state_mode, env_overrides))
    return BlindNavEnv(**kwargs)


def make_env(seed_base: int, rank: int, state_mode: str, env_overrides: dict[str, dict[str, Any]] | None = None):
    def _init():
        env = make_state_env(seed_base + rank, state_mode, env_overrides=env_overrides)
        return Monitor(wrap_policy_env(env, state_mode, env_overrides=env_overrides))

    return _init


def make_model(exp: Experiment, env: SubprocVecEnv, seed: int, device: str) -> BaseAlgorithm:
    kwargs: dict[str, Any] = {
        "env": env,
        "verbose": 0,
        "device": device,
        "seed": seed,
    }
    if exp.algo == "RecurrentPPO":
        if RecurrentPPO is None:
            raise RuntimeError("sb3-contrib is not installed; RecurrentPPO is unavailable")
        kwargs.update(
            {
                "policy_kwargs": {
                    "lstm_hidden_size": exp.lstm_hidden_size,
                    "n_lstm_layers": exp.n_lstm_layers,
                    "net_arch": exp.net_arch,
                },
                "learning_rate": 3e-4,
                "n_steps": 64,
                "batch_size": 1024,
                "n_epochs": 5,
                "gamma": 0.98,
                "gae_lambda": 0.95,
            }
        )
        return RecurrentPPO(exp.policy, **kwargs)
    kwargs["policy_kwargs"] = {"net_arch": exp.net_arch}
    if exp.algo == "A2C":
        kwargs.update(
            {
                "learning_rate": 7e-4,
                "n_steps": 8,
                "gamma": 0.98,
                "gae_lambda": 0.95,
            }
        )
        return A2C(exp.policy, **kwargs)
    if exp.algo == "PPO":
        kwargs.update(
            {
                "learning_rate": 3e-4,
                "n_steps": 64,
                "batch_size": 1024,
                "n_epochs": 5,
                "gamma": 0.98,
                "gae_lambda": 0.95,
            }
        )
        return PPO(exp.policy, **kwargs)
    if exp.algo == "SAC":
        kwargs.update(
            {
                "learning_rate": 3e-4,
                "buffer_size": 500_000,
                "learning_starts": 8_192,
                "batch_size": 512,
                "gamma": 0.98,
                "tau": 0.01,
                "train_freq": (1, "step"),
                "gradient_steps": 1,
            }
        )
        return SAC(exp.policy, **kwargs)
    raise ValueError(exp.algo)


def load_model(exp: Experiment, env: SubprocVecEnv, model_path: Path, device: str, seed: int | None = None) -> BaseAlgorithm:
    if exp.algo == "RecurrentPPO":
        if RecurrentPPO is None:
            raise RuntimeError("sb3-contrib is not installed; RecurrentPPO is unavailable")
        model = RecurrentPPO.load(str(model_path), env=env, device=device)
    elif exp.algo == "A2C":
        model = A2C.load(str(model_path), env=env, device=device)
    elif exp.algo == "PPO":
        model = PPO.load(str(model_path), env=env, device=device)
    elif exp.algo == "SAC":
        model = SAC.load(str(model_path), env=env, device=device)
    else:
        raise ValueError(exp.algo)
    if seed is not None:
        model.set_random_seed(int(seed))
        model.seed = int(seed)
        if hasattr(model, "_last_obs"):
            model._last_obs = None
        if hasattr(model, "_last_episode_starts"):
            model._last_episode_starts = None
        if hasattr(model, "_last_original_obs"):
            model._last_original_obs = None
    return model


def predict_action(model: BaseAlgorithm, obs: np.ndarray, episode_start: bool, state: Any):
    if model.__class__.__name__ == "RecurrentPPO":
        action, state = model.predict(obs, state=state, episode_start=np.array([episode_start]), deterministic=True)
        return action, state
    action, _ = model.predict(obs, deterministic=True)
    return action, state


def count_params(model: BaseAlgorithm) -> tuple[int, int]:
    total = sum(p.numel() for p in model.policy.parameters())
    trainable = sum(p.numel() for p in model.policy.parameters() if p.requires_grad)
    if hasattr(model, "log_ent_coef"):
        coef = getattr(model, "log_ent_coef")
        total += int(coef.numel())
        trainable += int(coef.numel())
    return total, trainable


def eval_model(model: BaseAlgorithm, exp: Experiment, exp_dir: Path, seed_start: int, eval_episodes: int, video_episodes: int) -> dict[str, float]:
    if eval_episodes <= 0:
        return {
            "success_rate": 0.0,
            "avg_steps": 0.0,
            "avg_reward": 0.0,
            "avg_final_distance": 0.0,
            "avg_collision_count": 0.0,
            "avg_jump_attempts": 0.0,
            "avg_jump_triggers": 0.0,
        }

    rewards: list[float] = []
    steps: list[int] = []
    final_distances: list[float] = []
    successes: list[float] = []
    collision_counts: list[int] = []
    jump_attempt_counts: list[int] = []
    jump_trigger_counts: list[int] = []
    videos: list[Path] = []

    for i in range(eval_episodes):
        seed = seed_start + i
        env = make_state_env(seed, exp.state_mode)
        policy_env = wrap_policy_env(env, exp.state_mode)
        policy_obs, _ = policy_env.reset(seed=seed)
        path = [env.pos.copy()]
        headings = [env.heading]
        total_reward = 0.0
        terminated = False
        truncated = False
        collision_count = 0
        jump_attempt_count = 0
        jump_trigger_count = 0
        state = None
        episode_start = True

        while not (terminated or truncated):
            action, state = predict_action(model, policy_obs, episode_start, state)
            if exp.state_mode in NO_POLICY_JUMP_STATE_MODES:
                jump_attempt = 0
            else:
                jump_attempt = int(float(action[2]) > env.jump_threshold)
            jump_trigger = int(jump_attempt and env.jump_cooldown <= 0.0)
            policy_obs, reward, terminated, truncated, info = policy_env.step(action)
            episode_start = False
            if float(info.get("time_since_collision", 1.0)) <= 1e-8:
                collision_count += 1
            jump_attempt_count += jump_attempt
            jump_trigger_count += jump_trigger
            path.append(env.pos.copy())
            headings.append(env.heading)
            total_reward += float(reward)

        path_arr = np.asarray(path)
        heading_arr = np.asarray(headings, dtype=np.float32)
        final_distance = float(np.linalg.norm(env.target - path_arr[-1]))
        rewards.append(total_reward)
        steps.append(len(path_arr) - 1)
        final_distances.append(final_distance)
        successes.append(float(info.get("is_success", final_distance <= env.target_radius)))
        collision_counts.append(collision_count)
        jump_attempt_counts.append(jump_attempt_count)
        jump_trigger_counts.append(jump_trigger_count)

        if i < video_episodes:
            video = exp_dir / f"eval_{i + 1:02d}.mp4"
            render_episode_video(env, path_arr, total_reward, video, fps=20, headings=heading_arr)
            videos.append(video)
        print(
            f"eval index={i + 1} steps={steps[-1]} reward={total_reward:.3f} "
            f"final_distance={final_distance:.3f} collisions={collision_count} "
            f"jump_attempts={jump_attempt_count} jump_triggers={jump_trigger_count}",
            flush=True,
        )

    if videos:
        concat_and_compress(videos, exp_dir / "eval_concat.mp4", exp_dir / "eval_concat_compressed.mp4", fps=20)

    return {
        "success_rate": float(np.mean(successes)),
        "avg_steps": float(np.mean(steps)),
        "avg_reward": float(np.mean(rewards)),
        "avg_final_distance": float(np.mean(final_distances)),
        "avg_collision_count": float(np.mean(collision_counts)),
        "avg_jump_attempts": float(np.mean(jump_attempt_counts)),
        "avg_jump_triggers": float(np.mean(jump_trigger_counts)),
    }


def load_existing(csv_path: Path) -> tuple[list[dict[str, Any]], set[str]]:
    if not csv_path.exists():
        return [], set()
    rows = list(csv.DictReader(csv_path.open("r", encoding="utf-8")))
    return rows, {row["name"] for row in rows if row.get("status") == "ok"}


def write_results(results: list[dict[str, Any]], csv_path: Path, md_path: Path) -> None:
    fieldnames = [
        "name",
        "algo",
        "state_mode",
        "obs_dim",
        "net_arch",
        "param_trainable",
        "model_size_mb",
        "train_seconds",
        "trained_episodes",
        "success_rate",
        "avg_steps",
        "avg_reward",
        "avg_final_distance",
        "avg_collision_count",
        "avg_jump_attempts",
        "avg_jump_triggers",
        "status",
        "model_path",
        "video_path",
        "error",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow({k: row.get(k, "") for k in fieldnames})

    lines = [
        "| name | algo | obs_dim | net | params | train_s | success | avg_steps | avg_reward | avg_dist | avg_collisions | avg_jump_attempts | avg_jump_triggers | video | status |",
        "|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in results:
        lines.append(
            "| {name} | {algo} | {obs_dim} | {net_arch} | {params} | {train_s:.1f} | "
            "{success:.2f} | {steps:.1f} | {reward:.2f} | {dist:.2f} | {coll:.1f} | {jump_attempts:.1f} | {jump_triggers:.1f} | {video} | {status} |".format(
                name=row.get("name", ""),
                algo=row.get("algo", ""),
                obs_dim=int(float(row.get("obs_dim", 0) or 0)),
                net_arch=row.get("net_arch", ""),
                params=int(float(row.get("param_trainable", 0) or 0)),
                train_s=float(row.get("train_seconds", 0) or 0),
                success=float(row.get("success_rate", 0) or 0),
                steps=float(row.get("avg_steps", 0) or 0),
                reward=float(row.get("avg_reward", 0) or 0),
                dist=float(row.get("avg_final_distance", 0) or 0),
                coll=float(row.get("avg_collision_count", 0) or 0),
                jump_attempts=float(row.get("avg_jump_attempts", 0) or 0),
                jump_triggers=float(row.get("avg_jump_triggers", 0) or 0),
                video=row.get("video_path", ""),
                status=row.get("status", ""),
            )
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def format_net_arch(net_arch: Any) -> str:
    if isinstance(net_arch, dict):
        return str(net_arch)
    if isinstance(net_arch, (list, tuple)):
        return ",".join(map(str, net_arch))
    return str(net_arch)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="/home/weiaokang/screencap/file/blind_nav_state_dims_10k")
    parser.add_argument("--episodes", type=int, default=TARGET_EPISODES)
    parser.add_argument("--n-envs", type=int, default=N_ENVS)
    parser.add_argument("--eval-episodes", type=int, default=EVAL_EPISODES)
    parser.add_argument("--video-episodes", type=int, default=VIDEO_EPISODES)
    parser.add_argument("--only", nargs="*", default=None)
    parser.add_argument("--resume-model", default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--seed-base", type=int, default=900_000)
    parser.add_argument("--eval-seed-base", type=int, default=1_000_000)
    args = parser.parse_args()

    thread_count = max(1, int(args.torch_threads))
    os.environ["OMP_NUM_THREADS"] = str(thread_count)
    os.environ["MKL_NUM_THREADS"] = str(thread_count)
    os.environ["OPENBLAS_NUM_THREADS"] = str(thread_count)
    torch.set_num_threads(thread_count)
    print(f"thread_config torch_threads={thread_count}", flush=True)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "state_dim_results.csv"
    md_path = output_dir / "state_dim_results.md"
    results, done = load_existing(csv_path)
    selected = [exp for exp in EXPERIMENTS if args.only is None or exp.name in set(args.only)]

    print(f"state_dim_benchmark_start output_dir={output_dir} experiments={len(selected)}", flush=True)
    for idx, exp in enumerate(selected):
        if exp.name in done:
            print(f"skip_done name={exp.name}", flush=True)
            continue

        exp_dir = output_dir / exp.name
        exp_dir.mkdir(parents=True, exist_ok=True)
        ckpt_dir = exp_dir / "models"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        obs_dim = observation_dim(exp.state_mode)
        row: dict[str, Any] = {
            "name": exp.name,
            "algo": exp.algo,
            "state_mode": exp.state_mode,
            "obs_dim": obs_dim,
            "net_arch": format_net_arch(exp.net_arch),
            "trained_episodes": args.episodes,
            "status": "failed",
            "error": "",
        }
        env = None
        try:
            seed = args.seed_base + idx * 10_000
            print(f"experiment_start index={idx + 1}/{len(selected)} name={exp.name}", flush=True)
            env = SubprocVecEnv([make_env(seed, rank, exp.state_mode) for rank in range(args.n_envs)], start_method="fork")
            if args.resume_model:
                model = load_model(exp, env, Path(args.resume_model), args.device, seed=seed)
                print(f"resume_model path={args.resume_model}", flush=True)
            else:
                model = make_model(exp, env, seed, args.device)
            _, param_trainable = count_params(model)
            row["param_trainable"] = param_trainable
            callback = EpisodeStopAndCheckpointCallback(args.episodes, ckpt_dir)
            start = time.perf_counter()
            model.learn(total_timesteps=args.episodes * MAX_STEPS, callback=callback, progress_bar=False)
            row["train_seconds"] = time.perf_counter() - start
            model_path = exp_dir / f"{exp.name}.zip"
            model.save(str(model_path.with_suffix("")))
            row["model_path"] = str(model_path)
            row["model_size_mb"] = model_path.stat().st_size / (1024 * 1024)

            env.close()
            env = None
            gc.collect()

            row.update(
                eval_model(
                    model,
                    exp,
                    exp_dir,
                    seed_start=args.eval_seed_base + idx * 1_000,
                    eval_episodes=args.eval_episodes,
                    video_episodes=args.video_episodes,
                )
            )
            compressed = exp_dir / "eval_concat_compressed.mp4"
            row["video_path"] = str(compressed if compressed.exists() else exp_dir)
            row["status"] = "ok"
            print(f"experiment_done name={exp.name}", flush=True)
        except Exception as exc:
            row["error"] = repr(exc)
            print(f"experiment_failed name={exp.name} error={exc!r}", flush=True)
        finally:
            if env is not None:
                env.close()
            gc.collect()
            results = [r for r in results if r.get("name") != exp.name]
            results.append(row)
            write_results(results, csv_path, md_path)
            print(f"results_updated csv={csv_path} md={md_path}", flush=True)

    print("state_dim_benchmark_done", flush=True)


if __name__ == "__main__":
    main()
