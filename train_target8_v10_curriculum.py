from __future__ import annotations

import argparse
import gc
import os
import time
from pathlib import Path
import shutil

import torch
from stable_baselines3.common.vec_env import SubprocVecEnv

from benchmark_state_dims_10k import (
    EpisodeStopAndCheckpointCallback,
    Experiment,
    MAX_STEPS,
    N_ENVS,
    count_params,
    eval_model,
    load_model,
    load_env_overrides,
    make_env,
    make_model,
    observation_dim,
    write_results,
)


STAGES: list[Experiment] = [
    Experiment(
        name="rppo_medium_lstm128x2_observable12_target8_discrete_recovery_v10_stage1",
        algo="RecurrentPPO",
        policy="MlpLstmPolicy",
        net_arch={"pi": [128, 128], "vf": [128, 128]},
        state_mode="observable12_target8_discrete_recovery_v10_stage1",
        lstm_hidden_size=128,
        n_lstm_layers=2,
    ),
    Experiment(
        name="rppo_medium_lstm128x2_observable12_target8_discrete_recovery_v10_stage2a",
        algo="RecurrentPPO",
        policy="MlpLstmPolicy",
        net_arch={"pi": [128, 128], "vf": [128, 128]},
        state_mode="observable12_target8_discrete_recovery_v10_stage2a",
        lstm_hidden_size=128,
        n_lstm_layers=2,
    ),
    Experiment(
        name="rppo_medium_lstm128x2_observable12_target8_discrete_recovery_v10_stage2",
        algo="RecurrentPPO",
        policy="MlpLstmPolicy",
        net_arch={"pi": [128, 128], "vf": [128, 128]},
        state_mode="observable12_target8_discrete_recovery_v10_stage2",
        lstm_hidden_size=128,
        n_lstm_layers=2,
    ),
    Experiment(
        name="rppo_medium_lstm128x2_observable12_target8_discrete_recovery_v10_stage3",
        algo="RecurrentPPO",
        policy="MlpLstmPolicy",
        net_arch={"pi": [128, 128], "vf": [128, 128]},
        state_mode="observable12_target8_discrete_recovery_v10_stage3",
        lstm_hidden_size=128,
        n_lstm_layers=2,
    ),
]


def stage_names() -> list[str]:
    return [exp.name for exp in STAGES]


def stage_index_by_name(name: str) -> int:
    names = stage_names()
    if name not in names:
        raise ValueError(f"unknown stage name: {name}")
    return names.index(name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--stage1-episodes", type=int, default=20_000)
    parser.add_argument("--stage2a-episodes", type=int, default=10_000)
    parser.add_argument("--stage2-episodes", type=int, default=30_000)
    parser.add_argument("--stage3-episodes", type=int, default=50_000)
    parser.add_argument("--n-envs", type=int, default=N_ENVS)
    parser.add_argument("--stage1-n-envs", type=int, default=None)
    parser.add_argument("--stage2a-n-envs", type=int, default=None)
    parser.add_argument("--stage2-n-envs", type=int, default=None)
    parser.add_argument("--stage3-n-envs", type=int, default=None)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--video-episodes", type=int, default=3)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--seed-base", type=int, default=9_200_000)
    parser.add_argument("--eval-seed-base", type=int, default=9_300_000)
    parser.add_argument("--checkpoint-interval", type=int, default=1_000)
    parser.add_argument("--progress-interval", type=int, default=500)
    parser.add_argument("--env-overrides-file", default=None)
    parser.add_argument("--resume-model", default=None)
    parser.add_argument("--start-stage", choices=stage_names(), default=STAGES[0].name)
    parser.add_argument("--end-stage", choices=stage_names(), default=STAGES[-1].name)
    args = parser.parse_args()

    thread_count = max(1, int(args.torch_threads))
    os.environ["OMP_NUM_THREADS"] = str(thread_count)
    os.environ["MKL_NUM_THREADS"] = str(thread_count)
    os.environ["OPENBLAS_NUM_THREADS"] = str(thread_count)
    torch.set_num_threads(thread_count)
    print(f"thread_config torch_threads={thread_count}", flush=True)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    env_overrides = load_env_overrides(args.env_overrides_file)
    if args.env_overrides_file:
        shutil.copy2(args.env_overrides_file, output_dir / "env_overrides.json")
    csv_path = output_dir / "v10_curriculum_results.csv"
    md_path = output_dir / "v10_curriculum_results.md"

    episode_schedule = {
        STAGES[0].name: int(args.stage1_episodes),
        STAGES[1].name: int(args.stage2a_episodes),
        STAGES[2].name: int(args.stage2_episodes),
        STAGES[3].name: int(args.stage3_episodes),
    }
    env_schedule = {
        STAGES[0].name: int(args.stage1_n_envs or args.n_envs),
        STAGES[1].name: int(args.stage2a_n_envs or args.n_envs),
        STAGES[2].name: int(args.stage2_n_envs or args.n_envs),
        STAGES[3].name: int(args.stage3_n_envs or args.n_envs),
    }
    start_idx = stage_index_by_name(args.start_stage)
    end_idx = stage_index_by_name(args.end_stage)
    if start_idx > end_idx:
        raise ValueError("--start-stage must not come after --end-stage")
    selected_stages = STAGES[start_idx : end_idx + 1]
    prev_model = Path(args.resume_model) if args.resume_model else None
    if start_idx > 0 and prev_model is None:
        raise ValueError("--resume-model is required when starting after stage1")
    results: list[dict[str, object]] = []

    for idx, exp in enumerate(selected_stages):
        global_stage_idx = start_idx + idx
        stage_seed = args.seed_base + global_stage_idx * 10_000
        stage_dir = output_dir / exp.name
        stage_dir.mkdir(parents=True, exist_ok=True)
        ckpt_dir = stage_dir / "models"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        env = None
        model = None
        try:
            stage_envs = env_schedule[exp.name]
            print(
                f"v10_stage_start stage={global_stage_idx + 1}/{len(STAGES)} "
                f"name={exp.name} n_envs={stage_envs}",
                flush=True,
            )
            env = SubprocVecEnv(
                [make_env(stage_seed, rank, exp.state_mode, env_overrides=env_overrides) for rank in range(stage_envs)],
                start_method="fork",
            )
            if prev_model is None:
                model = make_model(exp, env, stage_seed, args.device)
            else:
                model = load_model(exp, env, prev_model, args.device, seed=stage_seed)
                print(f"v10_stage_resume model={prev_model}", flush=True)
            _, trainable = count_params(model)
            target_episodes = episode_schedule[exp.name]
            callback = EpisodeStopAndCheckpointCallback(
                target_episodes,
                ckpt_dir,
                checkpoint_interval=args.checkpoint_interval,
                progress_interval=args.progress_interval,
            )
            start = time.perf_counter()
            model.learn(
                total_timesteps=target_episodes * MAX_STEPS,
                callback=callback,
                progress_bar=False,
                reset_num_timesteps=prev_model is None,
            )
            train_seconds = time.perf_counter() - start
            model_path = stage_dir / f"{exp.name}.zip"
            model.save(str(model_path.with_suffix("")))
            prev_model = model_path

            env.close()
            env = None
            gc.collect()

            row: dict[str, object] = {
                "name": exp.name,
                "algo": exp.algo,
                "state_mode": exp.state_mode,
                "obs_dim": observation_dim(exp.state_mode),
                "net_arch": str(exp.net_arch),
                "param_trainable": trainable,
                "trained_episodes": target_episodes,
                "train_seconds": train_seconds,
                "model_path": str(model_path),
                "model_size_mb": model_path.stat().st_size / (1024 * 1024),
                "status": "ok",
                "error": "",
            }
            row.update(
                eval_model(
                    model,
                    exp,
                    stage_dir,
                    seed_start=args.eval_seed_base + global_stage_idx * 1_000,
                    eval_episodes=args.eval_episodes,
                    video_episodes=args.video_episodes if global_stage_idx == len(STAGES) - 1 else 0,
                )
            )
            compressed = stage_dir / "eval_concat_compressed.mp4"
            row["video_path"] = str(compressed if compressed.exists() else stage_dir)
            results = [r for r in results if r.get("name") != exp.name]
            results.append(row)
            write_results(results, csv_path, md_path)
            print(f"v10_stage_done name={exp.name}", flush=True)
        finally:
            if env is not None:
                env.close()
            if model is not None:
                del model
            gc.collect()

    print("v10_curriculum_done", flush=True)


if __name__ == "__main__":
    main()
