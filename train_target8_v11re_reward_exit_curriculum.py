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


STAGE = Experiment(
    name="rppo_medium_lstm128x2_observable12_target8_macro_library_v11re_stage2",
    algo="RecurrentPPO",
    policy="MlpLstmPolicy",
    net_arch={"pi": [128, 128], "vf": [128, 128]},
    state_mode="observable12_target8_macro_library_v11re_stage2",
    lstm_hidden_size=128,
    n_lstm_layers=2,
)


def warm_start_policy(model: RecurrentPPO, resume_model: Path, device: str) -> int:
    source = RecurrentPPO.load(str(resume_model), device=device)
    target_state = model.policy.state_dict()
    source_state = source.policy.state_dict()
    matched: dict[str, torch.Tensor] = {}
    for key, value in source_state.items():
        if key in target_state and target_state[key].shape == value.shape:
            matched[key] = value.detach().clone()
    target_state.update(matched)
    model.policy.load_state_dict(target_state, strict=False)
    return len(matched)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--stage-episodes", type=int, default=None)
    parser.add_argument("--n-envs", type=int, default=N_ENVS)
    parser.add_argument("--eval-episodes", type=int, default=None)
    parser.add_argument("--video-episodes", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--seed-base", type=int, default=10_641_000)
    parser.add_argument("--eval-seed-base", type=int, default=10_651_000)
    parser.add_argument("--checkpoint-interval", type=int, default=None)
    parser.add_argument("--progress-interval", type=int, default=None)
    parser.add_argument("--env-overrides-file", default=None)
    parser.add_argument("--resume-model", default=None)
    args = parser.parse_args()

    quick_mode = bool(args.quick)
    stage_episodes = int(args.stage_episodes or (500 if quick_mode else 1_000))
    eval_episodes = int(args.eval_episodes or (6 if quick_mode else 8))
    checkpoint_interval = int(args.checkpoint_interval or (250 if quick_mode else 500))
    progress_interval = int(args.progress_interval or (125 if quick_mode else 250))

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
    csv_path = output_dir / "v11re_curriculum_results.csv"
    md_path = output_dir / "v11re_curriculum_results.md"
    stage_dir = output_dir / STAGE.name
    stage_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = stage_dir / "models"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    env = None
    model = None
    results: list[dict[str, object]] = []
    try:
        env = SubprocVecEnv(
            [make_env(int(args.seed_base), rank, STAGE.state_mode, env_overrides=env_overrides) for rank in range(int(args.n_envs))],
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
