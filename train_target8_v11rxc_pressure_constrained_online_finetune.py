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
    make_model,
    observation_dim,
    write_results,
)
from eval_target8_pressure import make_pressure_env
from train_target8_v11re_reward_exit_curriculum import warm_start_policy
from train_target8_v11rxc_bad_seed_curriculum import SeedCycleMonitor
from train_target8_v11rxc_constrained_online_finetune import (
    apply_gradient_mask_for_mode,
    configure_trainable_params,
    rebuild_policy_optimizer,
    restore_full_policy_optimizer_for_save,
)
from train_target8_v11rxc_pressure_seed_curriculum import parse_seed_pool
from benchmark_state_dims_10k import wrap_policy_env


STAGE = Experiment(
    name="rppo_medium_lstm128x2_observable12_target8_macro_library_v11rxc_stage2",
    algo="RecurrentPPO",
    policy="MlpLstmPolicy",
    net_arch={"pi": [128, 128], "vf": [128, 128]},
    state_mode="observable12_target8_macro_library_v11rxc_stage2",
    lstm_hidden_size=128,
    n_lstm_layers=2,
)


def make_pressure_seed_env(
    rank: int,
    state_mode: str,
    seed_pool: list[int],
    env_overrides: dict[str, dict[str, object]] | None = None,
):
    def _init():
        seed = seed_pool[rank % len(seed_pool)]
        env = make_pressure_env(
            seed,
            "v11",
            state_mode=state_mode,
            env_overrides=env_overrides,
        )
        wrapped = wrap_policy_env(env, state_mode, env_overrides=env_overrides)
        return SeedCycleMonitor(wrapped, seed_pool=seed_pool, seed_offset=rank)

    return _init


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tiny", action="store_true")
    parser.add_argument("--micro", action="store_true")
    parser.add_argument("--stage-episodes", type=int, default=None)
    parser.add_argument("--n-envs", type=int, default=N_ENVS)
    parser.add_argument("--eval-episodes", type=int, default=None)
    parser.add_argument("--video-episodes", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--seed-base", type=int, default=11_171_000)
    parser.add_argument("--eval-seed-base", type=int, default=11_181_000)
    parser.add_argument("--checkpoint-interval", type=int, default=None)
    parser.add_argument("--progress-interval", type=int, default=None)
    parser.add_argument("--env-overrides-file", default=None)
    parser.add_argument("--resume-model", required=True)
    parser.add_argument("--seed-pool", default=None)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--trainable-param-mode", default="action_net_only")
    args = parser.parse_args()

    if args.tiny:
        stage_episodes = int(args.stage_episodes or 32)
        eval_episodes = int(args.eval_episodes or 4)
        checkpoint_interval = int(args.checkpoint_interval or 16)
        progress_interval = int(args.progress_interval or 16)
    elif args.micro:
        stage_episodes = int(args.stage_episodes or 64)
        eval_episodes = int(args.eval_episodes or 4)
        checkpoint_interval = int(args.checkpoint_interval or 32)
        progress_interval = int(args.progress_interval or 32)
    else:
        stage_episodes = int(args.stage_episodes or 125)
        eval_episodes = int(args.eval_episodes or 6)
        checkpoint_interval = int(args.checkpoint_interval or 64)
        progress_interval = int(args.progress_interval or 32)
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
    (output_dir / "pressure_seed_pool.txt").write_text(
        "\n".join(str(seed) for seed in seed_pool) + "\n",
        encoding="utf-8",
    )
    csv_path = output_dir / "v11rxc_pressure_constrained_online_results.csv"
    md_path = output_dir / "v11rxc_pressure_constrained_online_results.md"
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
                make_pressure_seed_env(rank, STAGE.state_mode, seed_pool=seed_pool, env_overrides=env_overrides)
                for rank in range(int(args.n_envs))
            ],
            start_method="fork",
        )
        model = make_model(STAGE, env, int(args.seed_base), args.device)
        matched = warm_start_policy(model, Path(args.resume_model), args.device)
        print(f"warm_start matched_policy_tensors={matched}", flush=True)
        trainable_param_mode, macro_slice, anchor_trainable = configure_trainable_params(
            model, args.trainable_param_mode
        )
        print(
            f"trainable_param_mode={trainable_param_mode} trainable_params={anchor_trainable} macro_slice={macro_slice}",
            flush=True,
        )
        rebuild_policy_optimizer(model, float(args.learning_rate))
        _, trainable = count_params(model)
        original_step = model.policy.optimizer.step

        def masked_step(closure=None):
            apply_gradient_mask_for_mode(model, trainable_param_mode, macro_slice)
            if closure is None:
                return original_step()
            return original_step(closure)

        model.policy.optimizer.step = masked_step  # type: ignore[method-assign]

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
        restore_full_policy_optimizer_for_save(model, float(args.learning_rate))
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
            "trainable_param_mode": trainable_param_mode,
            "trainable_slice_params": anchor_trainable,
            "trained_episodes": stage_episodes,
            "train_seconds": train_seconds,
            "model_path": str(model_path),
            "model_size_mb": model_path.stat().st_size / (1024 * 1024),
            "status": "ok",
            "error": "",
            "pressure_seed_pool_size": len(seed_pool),
            "pressure_seed_pool_path": str(output_dir / "pressure_seed_pool.txt"),
            "learning_rate": float(args.learning_rate),
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
