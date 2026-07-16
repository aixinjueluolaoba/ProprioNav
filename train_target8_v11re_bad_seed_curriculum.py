from __future__ import annotations

import argparse
import gc
import os
import shutil
import time
from pathlib import Path

import torch
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv

from benchmark_state_dims_10k import (
    EpisodeStopAndCheckpointCallback,
    Experiment,
    MAX_STEPS,
    N_ENVS,
    count_params,
    eval_model,
    load_env_overrides,
    make_model,
    observation_dim,
    write_results,
    wrap_policy_env,
)
from train_target8_v11re_reward_exit_curriculum import warm_start_policy


STAGE = Experiment(
    name="rppo_medium_lstm128x2_observable12_target8_macro_library_v11re_stage2",
    algo="RecurrentPPO",
    policy="MlpLstmPolicy",
    net_arch={"pi": [128, 128], "vf": [128, 128]},
    state_mode="observable12_target8_macro_library_v11re_stage2",
    lstm_hidden_size=128,
    n_lstm_layers=2,
)

DEFAULT_BAD_SEEDS = [
    6_300_058,
    6_300_069,
    6_300_025,
    6_300_031,
    6_300_057,
    6_300_067,
    6_300_089,
    6_300_006,
]


def make_bad_seed_base_env(
    seed: int,
    state_mode: str,
    env_overrides: dict[str, dict[str, object]] | None = None,
):
    from benchmark_state_dims_10k import make_state_env

    env = make_state_env(seed, state_mode, env_overrides=env_overrides)
    env.tree_count = 85
    env.mountain_count = 58
    env.mountain_radius_range = (45.0, 105.0)
    env.concave_mountain_probability = 0.995
    env.deep_concave_mountain_probability = 0.90
    env.trapped_start_probability = 0.0
    env.opposite_heading_probability = 0.65

    start = torch.tensor([-520.0, -520.0], dtype=torch.float32).numpy()
    target = torch.tensor([520.0, 520.0], dtype=torch.float32).numpy()

    def fixed_start_and_target():
        env._forced_start_mountain = None
        return start.copy(), target.copy()

    env._sample_start_and_target = fixed_start_and_target  # type: ignore[method-assign]
    return env


class SeedCycleMonitor(Monitor):
    def __init__(self, env, seed_pool: list[int], seed_offset: int = 0):
        super().__init__(env)
        self.seed_pool = [int(seed) for seed in seed_pool]
        if not self.seed_pool:
            raise ValueError("seed_pool must not be empty")
        self.seed_index = int(seed_offset) % len(self.seed_pool)

    def reset(self, **kwargs):
        kwargs["seed"] = self.seed_pool[self.seed_index]
        self.seed_index = (self.seed_index + 1) % len(self.seed_pool)
        return super().reset(**kwargs)


def parse_seed_pool(seed_pool: str | None) -> list[int]:
    if not seed_pool:
        return list(DEFAULT_BAD_SEEDS)
    out: list[int] = []
    for value in seed_pool.split(","):
        value = value.strip()
        if not value:
            continue
        out.append(int(value))
    if not out:
        raise ValueError("parsed empty seed pool")
    return out


def make_bad_seed_env(
    rank: int,
    state_mode: str,
    seed_pool: list[int],
    env_overrides: dict[str, dict[str, object]] | None = None,
):
    def _init():
        env = make_bad_seed_base_env(seed_pool[rank % len(seed_pool)], state_mode, env_overrides=env_overrides)
        wrapped = wrap_policy_env(env, state_mode, env_overrides=env_overrides)
        return SeedCycleMonitor(wrapped, seed_pool=seed_pool, seed_offset=rank)

    return _init


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tiny", action="store_true")
    parser.add_argument("--micro", action="store_true")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--stage-episodes", type=int, default=None)
    parser.add_argument("--n-envs", type=int, default=N_ENVS)
    parser.add_argument("--eval-episodes", type=int, default=None)
    parser.add_argument("--video-episodes", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--seed-base", type=int, default=10_741_000)
    parser.add_argument("--eval-seed-base", type=int, default=10_751_000)
    parser.add_argument("--checkpoint-interval", type=int, default=None)
    parser.add_argument("--progress-interval", type=int, default=None)
    parser.add_argument("--env-overrides-file", default=None)
    parser.add_argument("--resume-model", default=None)
    parser.add_argument("--seed-pool", default=None)
    args = parser.parse_args()

    tiny_mode = bool(args.tiny)
    micro_mode = bool(args.micro)
    quick_mode = bool(args.quick)
    if tiny_mode:
        stage_episodes = int(args.stage_episodes or 64)
        eval_episodes = int(args.eval_episodes or 4)
        checkpoint_interval = int(args.checkpoint_interval or 32)
        progress_interval = int(args.progress_interval or 32)
    elif micro_mode:
        stage_episodes = int(args.stage_episodes or 250)
        eval_episodes = int(args.eval_episodes or 4)
        checkpoint_interval = int(args.checkpoint_interval or 125)
        progress_interval = int(args.progress_interval or 125)
    else:
        stage_episodes = int(args.stage_episodes or (500 if quick_mode else 1_000))
        eval_episodes = int(args.eval_episodes or (6 if quick_mode else 8))
        checkpoint_interval = int(args.checkpoint_interval or (250 if quick_mode else 500))
        progress_interval = int(args.progress_interval or (125 if quick_mode else 250))
    seed_pool = parse_seed_pool(args.seed_pool)

    thread_count = max(1, int(args.torch_threads))
    os.environ["OMP_NUM_THREADS"] = str(thread_count)
    os.environ["MKL_NUM_THREADS"] = str(thread_count)
    os.environ["OPENBLAS_NUM_THREADS"] = str(thread_count)
    torch.set_num_threads(thread_count)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    env_overrides = load_env_overrides(args.env_overrides_file)
    if args.env_overrides_file:
        shutil.copy2(args.env_overrides_file, output_dir / "env_overrides.json")
    (output_dir / "bad_seed_pool.txt").write_text("\n".join(str(seed) for seed in seed_pool) + "\n", encoding="utf-8")
    csv_path = output_dir / "v11re_bad_seed_curriculum_results.csv"
    md_path = output_dir / "v11re_bad_seed_curriculum_results.md"
    stage_dir = output_dir / STAGE.name
    stage_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = stage_dir / "models"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    env = None
    model = None
    results: list[dict[str, object]] = []
    try:
        env = SubprocVecEnv(
            [
                make_bad_seed_env(rank, STAGE.state_mode, seed_pool=seed_pool, env_overrides=env_overrides)
                for rank in range(int(args.n_envs))
            ],
            start_method="fork",
        )
        model = make_model(STAGE, env, int(args.seed_base), args.device)
        if args.resume_model:
            matched = warm_start_policy(model, Path(args.resume_model), args.device)
            print(f"warm_start matched_policy_tensors={matched}", flush=True)
        _, trainable = count_params(model)
        callback = EpisodeStopAndCheckpointCallback(
            stage_episodes,
            ckpt_dir,
            checkpoint_interval=checkpoint_interval,
            progress_interval=progress_interval,
        )
        start = time.perf_counter()
        model.learn(
            total_timesteps=stage_episodes * MAX_STEPS,
            callback=callback,
            progress_bar=False,
            reset_num_timesteps=True,
        )
        train_seconds = time.perf_counter() - start
        model_path = stage_dir / f"{STAGE.name}.zip"
        model.save(str(model_path.with_suffix("")))

        env.close()
        env = None
        gc.collect()

        row: dict[str, object] = {
            "name": STAGE.name,
            "algo": STAGE.algo,
            "state_mode": STAGE.state_mode,
            "obs_dim": observation_dim(STAGE.state_mode),
            "net_arch": str(STAGE.net_arch),
            "param_trainable": trainable,
            "trained_episodes": stage_episodes,
            "train_seconds": train_seconds,
            "model_path": str(model_path),
            "model_size_mb": model_path.stat().st_size / (1024 * 1024),
            "status": "ok",
            "error": "",
            "bad_seed_pool_size": len(seed_pool),
            "bad_seed_pool_path": str(output_dir / "bad_seed_pool.txt"),
        }
        row.update(
            eval_model(
                model,
                STAGE,
                stage_dir,
                seed_start=int(args.eval_seed_base),
                eval_episodes=eval_episodes,
                video_episodes=int(args.video_episodes),
            )
        )
        results.append(row)
        write_results(results, csv_path, md_path)
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    main()
