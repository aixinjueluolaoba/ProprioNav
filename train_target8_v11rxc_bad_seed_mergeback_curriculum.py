from __future__ import annotations

import argparse
import gc
import os
import shutil
import time
from pathlib import Path

import torch
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import SubprocVecEnv

from benchmark_state_dims_10k import (
    EpisodeStopAndCheckpointCallback,
    Experiment,
    MAX_STEPS,
    N_ENVS,
    count_params,
    eval_model,
    load_env_overrides,
    make_env,
    make_model,
    observation_dim,
    write_results,
)
from train_target8_v11re_bad_seed_mergeback_curriculum import set_model_learning_rate
from train_target8_v11re_reward_exit_curriculum import warm_start_policy
from train_target8_v11rxc_bad_seed_curriculum import make_bad_seed_env, parse_seed_pool


STAGE = Experiment(
    name="rppo_medium_lstm128x2_observable12_target8_macro_library_v11rxc_stage2",
    algo="RecurrentPPO",
    policy="MlpLstmPolicy",
    net_arch={"pi": [128, 128], "vf": [128, 128]},
    state_mode="observable12_target8_macro_library_v11rxc_stage2",
    lstm_hidden_size=128,
    n_lstm_layers=2,
)


def load_trained_model(model_path: Path, env: SubprocVecEnv, device: str, seed: int) -> RecurrentPPO:
    model = RecurrentPPO.load(str(model_path), env=env, device=device)
    model.set_random_seed(int(seed))
    model.seed = int(seed)
    return model


def copy_optional(src: str | None, dst: Path) -> None:
    if src:
        shutil.copy2(src, dst)


def make_mixed_phase2_env(
    seed_base: int,
    rank: int,
    state_mode: str,
    seed_pool: list[int],
    bad_seed_env_count: int,
    env_overrides: dict[str, dict[str, object]] | None = None,
):
    if rank < bad_seed_env_count:
        return make_bad_seed_env(rank, state_mode, seed_pool=seed_pool, env_overrides=env_overrides)
    return make_env(seed_base, rank, state_mode, env_overrides=env_overrides)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tiny", action="store_true")
    parser.add_argument("--micro", action="store_true")
    parser.add_argument("--phase1-episodes", type=int, default=None)
    parser.add_argument("--phase2-episodes", type=int, default=None)
    parser.add_argument("--n-envs", type=int, default=N_ENVS)
    parser.add_argument("--eval-episodes", type=int, default=6)
    parser.add_argument("--video-episodes", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--seed-base", type=int, default=11_041_000)
    parser.add_argument("--eval-seed-base", type=int, default=11_051_000)
    parser.add_argument("--phase1-checkpoint-interval", type=int, default=None)
    parser.add_argument("--phase2-checkpoint-interval", type=int, default=None)
    parser.add_argument("--progress-interval", type=int, default=None)
    parser.add_argument("--phase2-bad-seed-ratio", type=float, default=0.25)
    parser.add_argument("--phase1-learning-rate", type=float, default=None)
    parser.add_argument("--phase2-learning-rate", type=float, default=None)
    parser.add_argument("--bad-seed-env-overrides-file", default=None)
    parser.add_argument("--mergeback-env-overrides-file", default=None)
    parser.add_argument("--resume-model", default=None)
    parser.add_argument("--seed-pool", default=None)
    parser.add_argument("--phase1-seed-pool", default=None)
    parser.add_argument("--phase2-seed-pool", default=None)
    args = parser.parse_args()

    tiny_mode = bool(args.tiny)
    micro_mode = bool(args.micro)
    if tiny_mode:
        phase1_episodes = int(args.phase1_episodes or 32)
        phase2_episodes = int(args.phase2_episodes or 32)
        progress_interval = int(args.progress_interval or 16)
        phase1_checkpoint_interval = int(args.phase1_checkpoint_interval or 32)
        phase2_checkpoint_interval = int(args.phase2_checkpoint_interval or 32)
    else:
        phase1_episodes = int(args.phase1_episodes or (64 if micro_mode else 125))
        phase2_episodes = int(args.phase2_episodes or (32 if micro_mode else 64))
        progress_interval = int(args.progress_interval or (16 if micro_mode else 32))
        phase1_checkpoint_interval = int(args.phase1_checkpoint_interval or phase1_episodes)
        phase2_checkpoint_interval = int(args.phase2_checkpoint_interval or phase2_episodes)
    default_seed_pool = parse_seed_pool(args.seed_pool)
    phase1_seed_pool = parse_seed_pool(args.phase1_seed_pool) if args.phase1_seed_pool else list(default_seed_pool)
    phase2_seed_pool = parse_seed_pool(args.phase2_seed_pool) if args.phase2_seed_pool else list(default_seed_pool)
    phase2_bad_seed_ratio = float(max(0.0, min(1.0, args.phase2_bad_seed_ratio)))

    thread_count = max(1, int(args.torch_threads))
    os.environ["OMP_NUM_THREADS"] = str(thread_count)
    os.environ["MKL_NUM_THREADS"] = str(thread_count)
    os.environ["OPENBLAS_NUM_THREADS"] = str(thread_count)
    torch.set_num_threads(thread_count)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    phase1_overrides = load_env_overrides(args.bad_seed_env_overrides_file)
    phase2_overrides = load_env_overrides(args.mergeback_env_overrides_file)
    copy_optional(args.bad_seed_env_overrides_file, output_dir / "phase1_env_overrides.json")
    copy_optional(args.mergeback_env_overrides_file, output_dir / "phase2_env_overrides.json")
    (output_dir / "bad_seed_pool.txt").write_text("\n".join(str(seed) for seed in default_seed_pool) + "\n", encoding="utf-8")
    (output_dir / "phase1_seed_pool.txt").write_text(
        "\n".join(str(seed) for seed in phase1_seed_pool) + "\n",
        encoding="utf-8",
    )
    (output_dir / "phase2_seed_pool.txt").write_text(
        "\n".join(str(seed) for seed in phase2_seed_pool) + "\n",
        encoding="utf-8",
    )

    csv_path = output_dir / "v11rxc_bad_seed_mergeback_results.csv"
    md_path = output_dir / "v11rxc_bad_seed_mergeback_results.md"
    stage_dir = output_dir / STAGE.name
    stage_dir.mkdir(parents=True, exist_ok=True)
    phase1_ckpt_dir = stage_dir / "phase1_models"
    phase2_ckpt_dir = stage_dir / "phase2_models"
    phase1_ckpt_dir.mkdir(parents=True, exist_ok=True)
    phase2_ckpt_dir.mkdir(parents=True, exist_ok=True)

    env = None
    model: RecurrentPPO | None = None
    results: list[dict[str, object]] = []
    phase1_model_path = stage_dir / "phase1_model.zip"
    final_model_path = stage_dir / f"{STAGE.name}.zip"
    total_train_seconds = 0.0
    try:
        env = SubprocVecEnv(
            [
                make_bad_seed_env(rank, STAGE.state_mode, seed_pool=phase1_seed_pool, env_overrides=phase1_overrides)
                for rank in range(int(args.n_envs))
            ],
            start_method="fork",
        )
        model = make_model(STAGE, env, int(args.seed_base), args.device)
        if args.resume_model:
            matched = warm_start_policy(model, Path(args.resume_model), args.device)
            print(f"warm_start matched_policy_tensors={matched}", flush=True)
        if args.phase1_learning_rate is not None:
            set_model_learning_rate(model, float(args.phase1_learning_rate))
            print(f"phase1_learning_rate={float(args.phase1_learning_rate)}", flush=True)
        _, trainable = count_params(model)

        phase1_callback = EpisodeStopAndCheckpointCallback(
            phase1_episodes,
            phase1_ckpt_dir,
            checkpoint_interval=phase1_checkpoint_interval,
            progress_interval=progress_interval,
        )
        phase1_start = time.perf_counter()
        model.learn(
            total_timesteps=phase1_episodes * MAX_STEPS,
            callback=phase1_callback,
            progress_bar=False,
            reset_num_timesteps=True,
        )
        total_train_seconds += time.perf_counter() - phase1_start
        model.save(str(phase1_model_path.with_suffix("")))

        env.close()
        env = None
        gc.collect()

        phase2_bad_seed_env_count = int(round(int(args.n_envs) * phase2_bad_seed_ratio))
        env = SubprocVecEnv(
            [
                make_mixed_phase2_env(
                    int(args.seed_base) + 50_000,
                    rank,
                    STAGE.state_mode,
                    seed_pool=phase2_seed_pool,
                    bad_seed_env_count=phase2_bad_seed_env_count,
                    env_overrides=phase2_overrides,
                )
                for rank in range(int(args.n_envs))
            ],
            start_method="fork",
        )
        model = load_trained_model(phase1_model_path, env, args.device, int(args.seed_base) + 50_000)
        if args.phase2_learning_rate is not None:
            set_model_learning_rate(model, float(args.phase2_learning_rate))
            print(f"phase2_learning_rate={float(args.phase2_learning_rate)}", flush=True)
        phase2_callback = EpisodeStopAndCheckpointCallback(
            phase2_episodes,
            phase2_ckpt_dir,
            checkpoint_interval=phase2_checkpoint_interval,
            progress_interval=progress_interval,
        )
        phase2_start = time.perf_counter()
        model.learn(
            total_timesteps=phase2_episodes * MAX_STEPS,
            callback=phase2_callback,
            progress_bar=False,
            reset_num_timesteps=False,
        )
        total_train_seconds += time.perf_counter() - phase2_start
        model.save(str(final_model_path.with_suffix("")))

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
            "phase1_episodes": phase1_episodes,
            "phase2_episodes": phase2_episodes,
            "trained_episodes": phase1_episodes + phase2_episodes,
            "train_seconds": total_train_seconds,
            "model_path": str(final_model_path),
            "phase1_model_path": str(phase1_model_path),
            "model_size_mb": final_model_path.stat().st_size / (1024 * 1024),
            "status": "ok",
            "error": "",
            "bad_seed_pool_size": len(default_seed_pool),
            "bad_seed_pool_path": str(output_dir / "bad_seed_pool.txt"),
            "phase1_seed_pool_size": len(phase1_seed_pool),
            "phase1_seed_pool_path": str(output_dir / "phase1_seed_pool.txt"),
            "phase2_seed_pool_size": len(phase2_seed_pool),
            "phase2_seed_pool_path": str(output_dir / "phase2_seed_pool.txt"),
            "phase2_bad_seed_ratio": phase2_bad_seed_ratio,
            "phase2_bad_seed_env_count": phase2_bad_seed_env_count,
            "phase1_learning_rate": float(args.phase1_learning_rate) if args.phase1_learning_rate is not None else "",
            "phase2_learning_rate": float(args.phase2_learning_rate) if args.phase2_learning_rate is not None else "",
        }
        row.update(
            eval_model(
                model,
                STAGE,
                stage_dir,
                seed_start=int(args.eval_seed_base),
                eval_episodes=int(args.eval_episodes),
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
