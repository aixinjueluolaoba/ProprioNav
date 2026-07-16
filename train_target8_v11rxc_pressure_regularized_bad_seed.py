from __future__ import annotations

import argparse
import gc
import os
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
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


def make_zero_states(model: RecurrentPPO, batch_size: int, device: torch.device) -> RNNStates:
    shape = model.policy.lstm_hidden_state_shape
    zeros = torch.zeros(shape[0], batch_size, shape[2], device=device)
    return RNNStates((zeros.clone(), zeros.clone()), (zeros.clone(), zeros.clone()))


def macro_head_slice(model: RecurrentPPO) -> tuple[int, int]:
    action_dims = getattr(model.policy.action_dist, "action_dims", None)
    if action_dims is None or len(action_dims) != 3:
        raise ValueError(f"expected 3-head MultiCategorical action dims, got {action_dims}")
    start = int(action_dims[0]) + int(action_dims[1])
    end = start + int(action_dims[2])
    return start, end


def configure_trainable_params(model: RecurrentPPO, mode: str) -> tuple[str, tuple[int, int] | None, int]:
    normalized = mode.strip().lower()
    for param in model.policy.parameters():
        param.requires_grad_(False)
    macro_slice: tuple[int, int] | None = None
    if normalized == "all":
        for param in model.policy.parameters():
            param.requires_grad_(True)
    elif normalized == "action_net_only":
        model.policy.action_net.weight.requires_grad_(True)
        if model.policy.action_net.bias is not None:
            model.policy.action_net.bias.requires_grad_(True)
    elif normalized == "policy_head_only":
        for param in model.policy.mlp_extractor.policy_net.parameters():
            param.requires_grad_(True)
        model.policy.action_net.weight.requires_grad_(True)
        if model.policy.action_net.bias is not None:
            model.policy.action_net.bias.requires_grad_(True)
    elif normalized == "macro_slice_only":
        model.policy.action_net.weight.requires_grad_(True)
        if model.policy.action_net.bias is not None:
            model.policy.action_net.bias.requires_grad_(True)
        macro_slice = macro_head_slice(model)
    else:
        raise ValueError(f"unsupported trainable param mode: {mode}")
    trainable = sum(param.numel() for param in model.policy.parameters() if param.requires_grad)
    return normalized, macro_slice, int(trainable)


def apply_gradient_mask_for_mode(
    model: RecurrentPPO,
    trainable_param_mode: str,
    macro_slice: tuple[int, int] | None,
) -> None:
    if trainable_param_mode != "macro_slice_only":
        return
    if macro_slice is None:
        raise RuntimeError("macro_slice is required for macro_slice_only mode")
    start, end = macro_slice
    weight_grad = model.policy.action_net.weight.grad
    if weight_grad is not None:
        weight_grad[:start].zero_()
        weight_grad[end:].zero_()
    bias_grad = model.policy.action_net.bias.grad
    if bias_grad is not None:
        bias_grad[:start].zero_()
        bias_grad[end:].zero_()


def rebuild_policy_optimizer(model: RecurrentPPO, learning_rate: float) -> None:
    lr = float(learning_rate)
    if lr <= 0.0:
        raise ValueError(f"learning_rate must be positive, got {learning_rate}")
    trainable_tensors = [param for param in model.policy.parameters() if param.requires_grad]
    if not trainable_tensors:
        raise RuntimeError("no trainable parameters selected")
    optimizer_class = type(model.policy.optimizer)
    optimizer_kwargs = dict(model.policy.optimizer.defaults)
    optimizer_kwargs["lr"] = lr
    model.policy.optimizer = optimizer_class(trainable_tensors, **optimizer_kwargs)
    set_model_learning_rate(model, lr)


def restore_full_policy_optimizer_for_save(model: RecurrentPPO, learning_rate: float) -> None:
    for param in model.policy.parameters():
        param.requires_grad_(True)
    rebuild_policy_optimizer(model, learning_rate)


def parse_loss_head_mode(mode: str) -> np.ndarray:
    normalized = mode.strip().lower()
    mapping = {
        "all": np.asarray([1.0, 1.0, 1.0], dtype=np.float32),
        "macro_only": np.asarray([0.0, 0.0, 1.0], dtype=np.float32),
        "direction_macro": np.asarray([1.0, 0.0, 1.0], dtype=np.float32),
        "speed_macro": np.asarray([0.0, 1.0, 1.0], dtype=np.float32),
        "direction_speed": np.asarray([1.0, 1.0, 0.0], dtype=np.float32),
    }
    if normalized not in mapping:
        raise ValueError(f"unsupported loss head mode: {mode}")
    return mapping[normalized]


class PressureRegularizer:
    def __init__(
        self,
        dataset_path: str,
        device: torch.device,
        batch_size: int,
        loss_weight: float,
        loss_head_mode: str,
        max_samples: int | None = None,
    ):
        data = np.load(dataset_path)
        obs_np = np.asarray(data["obs"], dtype=np.float32)
        actions_np = np.asarray(data["actions"], dtype=np.int64)
        episode_starts_np = np.asarray(data["episode_starts"], dtype=np.float32)
        if max_samples is not None:
            limit = min(int(max_samples), int(obs_np.shape[0]))
            obs_np = obs_np[:limit]
            actions_np = actions_np[:limit]
            episode_starts_np = episode_starts_np[:limit]
        self.obs = torch.as_tensor(obs_np, dtype=torch.float32, device=device)
        self.actions = torch.as_tensor(actions_np, dtype=torch.long, device=device)
        self.episode_starts = torch.as_tensor(episode_starts_np, dtype=torch.float32, device=device)
        self.loss_head_mask = torch.as_tensor(
            np.repeat(parse_loss_head_mode(loss_head_mode)[None, :], int(obs_np.shape[0]), axis=0),
            dtype=torch.float32,
            device=device,
        )
        self.batch_size = max(1, int(batch_size))
        self.loss_weight = float(loss_weight)
        self.size = int(obs_np.shape[0])
        self.cursor = 0

    def sample(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.batch_size >= self.size:
            return self.obs, self.actions, self.episode_starts, self.loss_head_mask
        end = self.cursor + self.batch_size
        if end <= self.size:
            sl = slice(self.cursor, end)
            self.cursor = end % self.size
            return self.obs[sl], self.actions[sl], self.episode_starts[sl], self.loss_head_mask[sl]
        first = self.size - self.cursor
        second = self.batch_size - first
        idx = torch.cat(
            (
                torch.arange(self.cursor, self.size, device=self.obs.device),
                torch.arange(0, second, device=self.obs.device),
            )
        )
        self.cursor = second
        return self.obs[idx], self.actions[idx], self.episode_starts[idx], self.loss_head_mask[idx]

    def backward(self, model: RecurrentPPO, device: torch.device) -> float:
        if self.loss_weight <= 0.0:
            return 0.0
        obs, actions, episode_starts, loss_head_mask = self.sample()
        batch_states = make_zero_states(model, 1, device)
        dist, _ = model.policy.get_distribution(obs, batch_states.pi, episode_starts)
        per_head_log_prob: list[torch.Tensor] = []
        for head_index, categorical in enumerate(dist.distribution):
            logits = categorical.logits
            per_head_log_prob.append(-F.cross_entropy(logits, actions[:, head_index], reduction="none"))
        log_prob_heads = torch.stack(per_head_log_prob, dim=1)
        normalizer = loss_head_mask.sum().clamp_min(1e-6)
        loss = -(log_prob_heads * loss_head_mask).sum() / normalizer
        scaled = loss * self.loss_weight
        scaled.backward()
        return float(scaled.detach().cpu())


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
    parser.add_argument("--seed-base", type=int, default=11_281_000)
    parser.add_argument("--eval-seed-base", type=int, default=11_291_000)
    parser.add_argument("--checkpoint-interval", type=int, default=None)
    parser.add_argument("--progress-interval", type=int, default=None)
    parser.add_argument("--env-overrides-file", default=None)
    parser.add_argument("--resume-model", required=True)
    parser.add_argument("--seed-pool", default=None)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--trainable-param-mode", default="all")
    parser.add_argument("--pressure-dataset", required=True)
    parser.add_argument("--pressure-loss-weight", type=float, default=0.5)
    parser.add_argument("--pressure-batch-size", type=int, default=128)
    parser.add_argument("--pressure-loss-heads", default="all")
    parser.add_argument("--pressure-max-samples", type=int, default=None)
    args = parser.parse_args()

    tiny_mode = bool(args.tiny)
    micro_mode = bool(args.micro)
    if tiny_mode:
        stage_episodes = int(args.stage_episodes or 32)
        eval_episodes = int(args.eval_episodes or 4)
        checkpoint_interval = int(args.checkpoint_interval or 16)
        progress_interval = int(args.progress_interval or 16)
    elif micro_mode:
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
    device = torch.device(args.device)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    env_overrides = load_env_overrides(args.env_overrides_file)
    if args.env_overrides_file:
        shutil.copy2(args.env_overrides_file, output_dir / "env_overrides.json")
    (output_dir / "bad_seed_pool.txt").write_text("\n".join(str(seed) for seed in seed_pool) + "\n", encoding="utf-8")
    csv_path = output_dir / "v11rxc_pressure_regularized_bad_seed_results.csv"
    md_path = output_dir / "v11rxc_pressure_regularized_bad_seed_results.md"
    stage_dir = output_dir / STAGE.name
    stage_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = stage_dir / "models"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    env = None
    model = None
    results: list[dict[str, object]] = []
    pressure_loss_last = float("nan")
    try:
        env = SubprocVecEnv(
            [
                make_bad_seed_env(rank, STAGE.state_mode, seed_pool=seed_pool, env_overrides=env_overrides)
                for rank in range(int(args.n_envs))
            ],
            start_method="fork",
        )
        model = make_model(STAGE, env, int(args.seed_base), args.device)
        matched = warm_start_policy(model, Path(args.resume_model), args.device)
        print(f"warm_start matched_policy_tensors={matched}", flush=True)
        trainable_param_mode, macro_slice, anchor_trainable = configure_trainable_params(model, args.trainable_param_mode)
        print(
            f"trainable_param_mode={trainable_param_mode} trainable_params={anchor_trainable} macro_slice={macro_slice}",
            flush=True,
        )
        regularizer = PressureRegularizer(
            dataset_path=args.pressure_dataset,
            device=device,
            batch_size=int(args.pressure_batch_size),
            loss_weight=float(args.pressure_loss_weight),
            loss_head_mode=args.pressure_loss_heads,
            max_samples=args.pressure_max_samples,
        )
        print(
            f"pressure_regularizer samples={regularizer.size} batch_size={regularizer.batch_size} loss_weight={regularizer.loss_weight} loss_heads={args.pressure_loss_heads}",
            flush=True,
        )
        rebuild_policy_optimizer(model, float(args.learning_rate))
        _, trainable = count_params(model)
        original_step = model.policy.optimizer.step

        def masked_step(closure=None):
            nonlocal pressure_loss_last
            pressure_loss_last = regularizer.backward(model, device)
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
            "bad_seed_pool_size": len(seed_pool),
            "bad_seed_pool_path": str(output_dir / "bad_seed_pool.txt"),
            "learning_rate": float(args.learning_rate),
            "pressure_dataset": args.pressure_dataset,
            "pressure_dataset_samples": regularizer.size,
            "pressure_loss_weight": float(args.pressure_loss_weight),
            "pressure_loss_heads": args.pressure_loss_heads,
            "pressure_batch_size": int(args.pressure_batch_size),
            "pressure_loss_last": pressure_loss_last,
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
