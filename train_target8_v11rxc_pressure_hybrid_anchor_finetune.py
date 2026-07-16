from __future__ import annotations

import argparse
import gc
import os
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from sb3_contrib import RecurrentPPO
from sb3_contrib.common.recurrent.type_aliases import RNNStates
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
from eval_target8_pressure import make_pressure_env
from train_target8_v11re_bad_seed_mergeback_curriculum import set_model_learning_rate
from train_target8_v11re_reward_exit_curriculum import warm_start_policy
from train_target8_v11rxc_bad_seed_curriculum import SeedCycleMonitor
from train_target8_v11rxc_hybrid_anchor_finetune import (
    apply_anchor_gradient_mask,
    compute_anchor_weights,
    configure_anchor_trainable_params,
    load_anchor_dataset,
)
from train_target8_v11rxc_pressure_seed_curriculum import parse_seed_pool


STAGE = Experiment(
    name="rppo_medium_lstm128x2_observable12_target8_macro_library_v11rxc_stage2",
    algo="RecurrentPPO",
    policy="MlpLstmPolicy",
    net_arch={"pi": [128, 128], "vf": [128, 128]},
    state_mode="observable12_target8_macro_library_v11rxc_stage2",
    lstm_hidden_size=128,
    n_lstm_layers=2,
)


def make_zero_states(model: RecurrentPPO, batch_size: int, device: torch.device) -> RNNStates:
    shape = model.policy.lstm_hidden_state_shape
    zeros = torch.zeros(shape[0], batch_size, shape[2], device=device)
    return RNNStates((zeros.clone(), zeros.clone()), (zeros.clone(), zeros.clone()))


def run_anchor_bc_epoch(
    model: RecurrentPPO,
    dataset: dict[str, np.ndarray],
    sample_weights: np.ndarray,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch_index: int,
    trainable_param_mode: str,
    macro_slice: tuple[int, int] | None,
) -> float:
    obs = torch.as_tensor(dataset["obs"], dtype=torch.float32, device=device)
    actions = torch.as_tensor(dataset["actions"], dtype=torch.long, device=device)
    episode_starts = torch.as_tensor(dataset["episode_starts"], dtype=torch.float32, device=device)
    weights = torch.as_tensor(sample_weights, dtype=torch.float32, device=device)
    batch_states = make_zero_states(model, 1, device)
    optimizer.zero_grad(set_to_none=True)
    values, log_prob, entropy = model.policy.evaluate_actions(obs, actions, batch_states, episode_starts)
    del values
    loss = -(log_prob * weights).sum() / weights.sum().clamp_min(1e-6) - 1e-3 * entropy.mean()
    loss.backward()
    apply_anchor_gradient_mask(model, trainable_param_mode, macro_slice)
    torch.nn.utils.clip_grad_norm_(model.policy.parameters(), 1.0)
    optimizer.step()
    loss_value = float(loss.detach().cpu())
    print(f"anchor_epoch={epoch_index} anchor_loss={loss_value:.6f}", flush=True)
    return loss_value


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
    parser.add_argument("--resume-model", required=True)
    parser.add_argument("--anchor-datasets", nargs="+", required=True)
    parser.add_argument("--tiny", action="store_true")
    parser.add_argument("--micro", action="store_true")
    parser.add_argument("--hybrid-rounds", type=int, default=None)
    parser.add_argument("--ppo-episodes", type=int, default=None)
    parser.add_argument("--anchor-epochs", type=int, default=None)
    parser.add_argument("--n-envs", type=int, default=N_ENVS)
    parser.add_argument("--eval-episodes", type=int, default=4)
    parser.add_argument("--video-episodes", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--seed-base", type=int, default=11_191_000)
    parser.add_argument("--eval-seed-base", type=int, default=11_201_000)
    parser.add_argument("--progress-interval", type=int, default=None)
    parser.add_argument("--checkpoint-interval", type=int, default=None)
    parser.add_argument("--ppo-learning-rate", type=float, default=3e-5)
    parser.add_argument("--anchor-learning-rate", type=float, default=1e-4)
    parser.add_argument("--anchor-max-samples", type=int, default=None)
    parser.add_argument("--anchor-success-only", action="store_true")
    parser.add_argument("--anchor-dataset-weights", default=None)
    parser.add_argument("--anchor-dataset-roles", default=None)
    parser.add_argument("--anchor-trainable-param-mode", default="all")
    parser.add_argument("--recovery-weight", type=float, default=1.0)
    parser.add_argument("--stuck-weight", type=float, default=1.0)
    parser.add_argument("--collision-weight", type=float, default=1.0)
    parser.add_argument("--stuck-no-progress-threshold", type=float, default=0.8)
    parser.add_argument("--stuck-time-threshold", type=float, default=0.5)
    parser.add_argument("--seed-pool", default=None)
    parser.add_argument("--env-overrides-file", default=None)
    args = parser.parse_args()

    if args.tiny:
        hybrid_rounds = int(args.hybrid_rounds or 2)
        ppo_episodes = int(args.ppo_episodes or 16)
        anchor_epochs = int(args.anchor_epochs or 2)
        progress_interval = int(args.progress_interval or 8)
        checkpoint_interval = int(args.checkpoint_interval or 16)
    elif args.micro:
        hybrid_rounds = int(args.hybrid_rounds or 4)
        ppo_episodes = int(args.ppo_episodes or 32)
        anchor_epochs = int(args.anchor_epochs or 3)
        progress_interval = int(args.progress_interval or 16)
        checkpoint_interval = int(args.checkpoint_interval or 32)
    else:
        hybrid_rounds = int(args.hybrid_rounds or 6)
        ppo_episodes = int(args.ppo_episodes or 64)
        anchor_epochs = int(args.anchor_epochs or 4)
        progress_interval = int(args.progress_interval or 32)
        checkpoint_interval = int(args.checkpoint_interval or 64)

    seed_pool = parse_seed_pool(args.seed_pool)
    thread_count = max(1, int(args.torch_threads))
    os.environ["OMP_NUM_THREADS"] = str(thread_count)
    os.environ["MKL_NUM_THREADS"] = str(thread_count)
    os.environ["OPENBLAS_NUM_THREADS"] = str(thread_count)
    torch.set_num_threads(thread_count)
    device = torch.device(args.device)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.env_overrides_file:
        shutil.copy2(args.env_overrides_file, output_dir / "env_overrides.json")
    (output_dir / "pressure_seed_pool.txt").write_text(
        "\n".join(str(seed) for seed in seed_pool) + "\n",
        encoding="utf-8",
    )
    csv_path = output_dir / "v11rxc_pressure_hybrid_anchor_results.csv"
    md_path = output_dir / "v11rxc_pressure_hybrid_anchor_results.md"
    stage_dir = output_dir / STAGE.name
    stage_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = stage_dir / "models"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    env_overrides = load_env_overrides(args.env_overrides_file)

    anchor_dataset_weights = [1.0] * len(args.anchor_datasets)
    if args.anchor_dataset_weights:
        parsed = [float(value.strip()) for value in args.anchor_dataset_weights.split(",") if value.strip()]
        if len(parsed) != len(args.anchor_datasets):
            raise ValueError("anchor_dataset_weights count must match anchor_datasets")
        anchor_dataset_weights = parsed
    anchor_dataset_roles = ["generic"] * len(args.anchor_datasets)
    if args.anchor_dataset_roles:
        parsed = [value.strip() for value in args.anchor_dataset_roles.split(",") if value.strip()]
        if len(parsed) != len(args.anchor_datasets):
            raise ValueError("anchor_dataset_roles count must match anchor_datasets")
        anchor_dataset_roles = parsed

    anchor_dataset = load_anchor_dataset(
        dataset_paths=[str(path) for path in args.anchor_datasets],
        dataset_weights=anchor_dataset_weights,
        dataset_roles=anchor_dataset_roles,
        success_only=bool(args.anchor_success_only),
        stuck_no_progress_threshold=float(args.stuck_no_progress_threshold),
        stuck_time_threshold=float(args.stuck_time_threshold),
        max_samples=args.anchor_max_samples,
    )
    anchor_weights = compute_anchor_weights(
        anchor_dataset,
        recovery_weight=float(args.recovery_weight),
        stuck_weight=float(args.stuck_weight),
        collision_weight=float(args.collision_weight),
        stuck_no_progress_threshold=float(args.stuck_no_progress_threshold),
        stuck_time_threshold=float(args.stuck_time_threshold),
    )

    env = None
    model: RecurrentPPO | None = None
    results: list[dict[str, object]] = []
    total_train_seconds = 0.0
    anchor_loss_last = float("nan")
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
        _, trainable = count_params(model)
        anchor_trainable_mode, macro_slice, trainable_tensors, anchor_trainable = configure_anchor_trainable_params(
            model, args.anchor_trainable_param_mode
        )
        print(
            f"anchor_trainable_param_mode={anchor_trainable_mode} anchor_trainable_params={anchor_trainable} macro_slice={macro_slice}",
            flush=True,
        )
        if not trainable_tensors:
            raise RuntimeError("no anchor-trainable parameters selected")
        optimizer = torch.optim.Adam(trainable_tensors, lr=float(args.anchor_learning_rate))
        total_episodes = 0
        for round_index in range(int(hybrid_rounds)):
            set_model_learning_rate(model, float(args.ppo_learning_rate))
            callback = EpisodeStopAndCheckpointCallback(
                int(ppo_episodes),
                ckpt_dir,
                checkpoint_interval=int(checkpoint_interval),
                progress_interval=int(progress_interval),
                episode_offset=total_episodes,
            )
            learn_start = time.perf_counter()
            model.learn(
                total_timesteps=int(ppo_episodes) * MAX_STEPS,
                callback=callback,
                progress_bar=False,
                reset_num_timesteps=(round_index == 0),
            )
            total_train_seconds += time.perf_counter() - learn_start
            total_episodes += int(ppo_episodes)
            for anchor_epoch in range(int(anchor_epochs)):
                anchor_loss_last = run_anchor_bc_epoch(
                    model,
                    anchor_dataset,
                    anchor_weights,
                    optimizer,
                    device,
                    epoch_index=round_index * int(anchor_epochs) + anchor_epoch + 1,
                    trainable_param_mode=anchor_trainable_mode,
                    macro_slice=macro_slice,
                )
            round_checkpoint_path = ckpt_dir / f"checkpoint_ep_{total_episodes:05d}"
            model.save(str(round_checkpoint_path))
            round_model_path = ckpt_dir / f"hybrid_round_{round_index + 1:02d}"
            model.save(str(round_model_path))

        final_model_path = stage_dir / f"{STAGE.name}.zip"
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
            "hybrid_rounds": hybrid_rounds,
            "ppo_episodes": ppo_episodes,
            "anchor_epochs": anchor_epochs,
            "trained_episodes": total_episodes,
            "train_seconds": total_train_seconds,
            "model_path": str(final_model_path),
            "model_size_mb": final_model_path.stat().st_size / (1024 * 1024),
            "status": "ok",
            "error": "",
            "pressure_seed_pool_size": len(seed_pool),
            "pressure_seed_pool_path": str(output_dir / "pressure_seed_pool.txt"),
            "ppo_learning_rate": float(args.ppo_learning_rate),
            "anchor_learning_rate": float(args.anchor_learning_rate),
            "anchor_trainable_param_mode": anchor_trainable_mode,
            "anchor_trainable_params": anchor_trainable,
            "anchor_samples": int(anchor_dataset["obs"].shape[0]),
            "anchor_loss_last": anchor_loss_last,
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
