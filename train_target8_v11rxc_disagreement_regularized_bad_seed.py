from __future__ import annotations

import argparse
import gc
import os
import re
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


class AuxiliaryRegularizer:
    def __init__(
        self,
        dataset_path: str,
        device: torch.device,
        batch_size: int,
        loss_weight: float,
        loss_head_mode: str,
        max_samples: int | None = None,
        disagreement_filter: str = "none",
        seed_whitelist: set[int] | None = None,
        seed_mode_whitelist: dict[int, set[int] | None] | None = None,
        min_stuck_time: float | None = None,
        min_no_progress_time: float | None = None,
        recovery_mode_whitelist: set[int] | None = None,
    ):
        data = np.load(dataset_path)
        obs_np = np.asarray(data["obs"], dtype=np.float32)
        actions_np = np.asarray(data["actions"], dtype=np.int64)
        episode_starts_np = np.asarray(data["episode_starts"], dtype=np.float32)
        seeds_np = np.asarray(data["seeds"], dtype=np.int64) if "seeds" in data else None
        stuck_time_np = np.asarray(data["stuck_time"], dtype=np.float32) if "stuck_time" in data else None
        no_progress_np = np.asarray(data["no_progress_time"], dtype=np.float32) if "no_progress_time" in data else None
        recovery_mode_np = np.asarray(data["recovery_mode"], dtype=np.int64) if "recovery_mode" in data else None
        mask = np.ones(int(obs_np.shape[0]), dtype=bool)
        if disagreement_filter != "none":
            if "policy_actions" not in data:
                raise ValueError(f"dataset missing policy_actions for disagreement filter: {dataset_path}")
            policy_actions_np = np.asarray(data["policy_actions"], dtype=np.int64)
            if disagreement_filter == "macro":
                mask &= actions_np[:, 2] != policy_actions_np[:, 2]
            elif disagreement_filter == "any":
                mask &= np.any(actions_np != policy_actions_np, axis=1)
            else:
                raise ValueError(f"unsupported disagreement_filter: {disagreement_filter}")
        if seed_whitelist is not None:
            if seeds_np is None:
                raise ValueError(f"dataset missing seeds for seed_whitelist: {dataset_path}")
            mask &= np.isin(seeds_np, np.asarray(sorted(seed_whitelist), dtype=np.int64))
        if seed_mode_whitelist is not None:
            if seeds_np is None:
                raise ValueError(f"dataset missing seeds for seed_mode_whitelist: {dataset_path}")
            if recovery_mode_np is None:
                raise ValueError(f"dataset missing recovery_mode for seed_mode_whitelist: {dataset_path}")
            seed_mode_mask = np.zeros(int(obs_np.shape[0]), dtype=bool)
            for seed_value, mode_values in seed_mode_whitelist.items():
                seed_match = seeds_np == int(seed_value)
                if mode_values is None:
                    seed_mode_mask |= seed_match
                else:
                    seed_mode_mask |= seed_match & np.isin(
                        recovery_mode_np,
                        np.asarray(sorted(mode_values), dtype=np.int64),
                    )
            mask &= seed_mode_mask
        if min_stuck_time is not None:
            if stuck_time_np is None:
                raise ValueError(f"dataset missing stuck_time for min_stuck_time: {dataset_path}")
            mask &= stuck_time_np >= float(min_stuck_time)
        if min_no_progress_time is not None:
            if no_progress_np is None:
                raise ValueError(f"dataset missing no_progress_time for min_no_progress_time: {dataset_path}")
            mask &= no_progress_np >= float(min_no_progress_time)
        if recovery_mode_whitelist is not None:
            if recovery_mode_np is None:
                raise ValueError(f"dataset missing recovery_mode for recovery_mode_whitelist: {dataset_path}")
            mask &= np.isin(recovery_mode_np, np.asarray(sorted(recovery_mode_whitelist), dtype=np.int64))
        if not mask.all():
            obs_np = obs_np[mask]
            actions_np = actions_np[mask]
            episode_starts_np = episode_starts_np[mask]
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
        self.base_loss_weight = float(loss_weight)
        self.loss_weight = float(loss_weight)
        self.size = int(obs_np.shape[0])
        self.cursor = 0
        self.disagreement_filter = disagreement_filter
        self.seed_whitelist = seed_whitelist
        self.seed_mode_whitelist = seed_mode_whitelist
        self.min_stuck_time = min_stuck_time
        self.min_no_progress_time = min_no_progress_time
        self.recovery_mode_whitelist = recovery_mode_whitelist
        if self.size <= 0:
            raise ValueError(f"aux dataset became empty after filtering: path={dataset_path} disagreement_filter={disagreement_filter}")

    def set_loss_weight(self, value: float) -> None:
        self.loss_weight = float(value)

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


def parse_optional_int_set(value: str | None) -> set[int] | None:
    if value is None:
        return None
    text = value.strip()
    if not text:
        return None
    out = {int(part.strip()) for part in text.split(",") if part.strip()}
    return out or None


def parse_seed_mode_whitelist(value: str | None) -> dict[int, set[int] | None] | None:
    if value is None:
        return None
    text = value.strip()
    if not text:
        return None
    result: dict[int, set[int] | None] = {}
    for chunk in re.split(r"[;+]", text):
        item = chunk.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"invalid seed_mode_whitelist item: {item}")
        seed_text, mode_text = item.split(":", 1)
        seed_value = int(seed_text.strip())
        mode_spec = mode_text.strip()
        if not mode_spec or mode_spec == "*":
            result[seed_value] = None
            continue
        result[seed_value] = {int(part.strip()) for part in mode_spec.split(",") if part.strip()}
    return result or None


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
    parser.add_argument("--seed-base", type=int, default=11_381_000)
    parser.add_argument("--eval-seed-base", type=int, default=11_391_000)
    parser.add_argument("--checkpoint-interval", type=int, default=None)
    parser.add_argument("--progress-interval", type=int, default=None)
    parser.add_argument("--env-overrides-file", default=None)
    parser.add_argument("--resume-model", required=True)
    parser.add_argument("--seed-pool", default=None)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--trainable-param-mode", default="all")
    parser.add_argument("--pressure-dataset", required=True)
    parser.add_argument("--pressure-loss-weight", type=float, default=0.5)
    parser.add_argument("--pressure-loss-weight-late", type=float, default=None)
    parser.add_argument("--pressure-batch-size", type=int, default=128)
    parser.add_argument("--pressure-loss-heads", default="all")
    parser.add_argument("--pressure-max-samples", type=int, default=None)
    parser.add_argument("--specialist-dataset", required=True)
    parser.add_argument("--specialist-loss-weight", type=float, default=1.0)
    parser.add_argument("--specialist-loss-weight-late", type=float, default=None)
    parser.add_argument("--specialist-batch-size", type=int, default=128)
    parser.add_argument("--specialist-loss-heads", default="macro_only")
    parser.add_argument("--specialist-max-samples", type=int, default=None)
    parser.add_argument("--specialist-disagreement-filter", default="macro")
    parser.add_argument("--specialist-seed-whitelist", default=None)
    parser.add_argument("--specialist-seed-mode-whitelist", default=None)
    parser.add_argument("--specialist-min-stuck-time", type=float, default=None)
    parser.add_argument("--specialist-min-no-progress-time", type=float, default=None)
    parser.add_argument("--specialist-recovery-modes", default=None)
    parser.add_argument("--loss-weight-switch-frac", type=float, default=None)
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
    csv_path = output_dir / "v11rxc_disagreement_regularized_bad_seed_results.csv"
    md_path = output_dir / "v11rxc_disagreement_regularized_bad_seed_results.md"
    stage_dir = output_dir / STAGE.name
    stage_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = stage_dir / "models"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    env = None
    model = None
    results: list[dict[str, object]] = []
    pressure_loss_last = float("nan")
    specialist_loss_last = float("nan")
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
        pressure_regularizer = AuxiliaryRegularizer(
            dataset_path=args.pressure_dataset,
            device=device,
            batch_size=int(args.pressure_batch_size),
            loss_weight=float(args.pressure_loss_weight),
            loss_head_mode=args.pressure_loss_heads,
            max_samples=args.pressure_max_samples,
            disagreement_filter="none",
        )
        specialist_regularizer = AuxiliaryRegularizer(
            dataset_path=args.specialist_dataset,
            device=device,
            batch_size=int(args.specialist_batch_size),
            loss_weight=float(args.specialist_loss_weight),
            loss_head_mode=args.specialist_loss_heads,
            max_samples=args.specialist_max_samples,
            disagreement_filter=args.specialist_disagreement_filter,
            seed_whitelist=parse_optional_int_set(args.specialist_seed_whitelist),
            seed_mode_whitelist=parse_seed_mode_whitelist(args.specialist_seed_mode_whitelist),
            min_stuck_time=args.specialist_min_stuck_time,
            min_no_progress_time=args.specialist_min_no_progress_time,
            recovery_mode_whitelist=parse_optional_int_set(args.specialist_recovery_modes),
        )
        print(
            f"pressure_regularizer samples={pressure_regularizer.size} batch_size={pressure_regularizer.batch_size} loss_weight={pressure_regularizer.loss_weight} loss_heads={args.pressure_loss_heads}",
            flush=True,
        )
        print(
            "specialist_regularizer samples={samples} batch_size={batch} loss_weight={weight} loss_heads={heads} disagreement_filter={disagreement} seed_whitelist={seed_whitelist} seed_mode_whitelist={seed_mode_whitelist} min_stuck_time={min_stuck} min_no_progress_time={min_no_progress} recovery_modes={recovery_modes}".format(
                samples=specialist_regularizer.size,
                batch=specialist_regularizer.batch_size,
                weight=specialist_regularizer.loss_weight,
                heads=args.specialist_loss_heads,
                disagreement=args.specialist_disagreement_filter,
                seed_whitelist=args.specialist_seed_whitelist,
                seed_mode_whitelist=args.specialist_seed_mode_whitelist,
                min_stuck=args.specialist_min_stuck_time,
                min_no_progress=args.specialist_min_no_progress_time,
                recovery_modes=args.specialist_recovery_modes,
            ),
            flush=True,
        )
        rebuild_policy_optimizer(model, float(args.learning_rate))
        _, trainable = count_params(model)
        original_step = model.policy.optimizer.step
        total_timesteps = stage_episodes * MAX_STEPS
        loss_weight_switch_frac = args.loss_weight_switch_frac
        pressure_loss_weight_late = args.pressure_loss_weight_late
        specialist_loss_weight_late = args.specialist_loss_weight_late
        switch_timesteps = None
        if loss_weight_switch_frac is not None:
            switch_frac = float(loss_weight_switch_frac)
            if not 0.0 < switch_frac < 1.0:
                raise ValueError(f"loss_weight_switch_frac must be in (0, 1), got {switch_frac}")
            switch_timesteps = max(1, int(total_timesteps * switch_frac))
            print(
                f"loss_weight_schedule switch_frac={switch_frac} switch_timesteps={switch_timesteps} pressure_late={pressure_loss_weight_late} specialist_late={specialist_loss_weight_late}",
                flush=True,
            )

        def masked_step(closure=None):
            nonlocal pressure_loss_last, specialist_loss_last
            if switch_timesteps is not None:
                use_late = int(model.num_timesteps) >= int(switch_timesteps)
                pressure_regularizer.set_loss_weight(
                    pressure_loss_weight_late if use_late and pressure_loss_weight_late is not None else pressure_regularizer.base_loss_weight
                )
                specialist_regularizer.set_loss_weight(
                    specialist_loss_weight_late if use_late and specialist_loss_weight_late is not None else specialist_regularizer.base_loss_weight
                )
            pressure_loss_last = pressure_regularizer.backward(model, device)
            specialist_loss_last = specialist_regularizer.backward(model, device)
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
            total_timesteps=total_timesteps,
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
            "pressure_dataset_samples": pressure_regularizer.size,
            "pressure_loss_weight": float(args.pressure_loss_weight),
            "pressure_loss_weight_late": "" if args.pressure_loss_weight_late is None else float(args.pressure_loss_weight_late),
            "pressure_loss_heads": args.pressure_loss_heads,
            "pressure_batch_size": int(args.pressure_batch_size),
            "pressure_loss_last": pressure_loss_last,
            "specialist_dataset": args.specialist_dataset,
            "specialist_dataset_samples": specialist_regularizer.size,
            "specialist_loss_weight": float(args.specialist_loss_weight),
            "specialist_loss_weight_late": "" if args.specialist_loss_weight_late is None else float(args.specialist_loss_weight_late),
            "specialist_loss_heads": args.specialist_loss_heads,
            "specialist_batch_size": int(args.specialist_batch_size),
            "specialist_disagreement_filter": args.specialist_disagreement_filter,
            "specialist_seed_whitelist": args.specialist_seed_whitelist or "",
            "specialist_seed_mode_whitelist": args.specialist_seed_mode_whitelist or "",
            "specialist_min_stuck_time": "" if args.specialist_min_stuck_time is None else float(args.specialist_min_stuck_time),
            "specialist_min_no_progress_time": "" if args.specialist_min_no_progress_time is None else float(args.specialist_min_no_progress_time),
            "specialist_recovery_modes": args.specialist_recovery_modes or "",
            "specialist_loss_last": specialist_loss_last,
            "loss_weight_switch_frac": "" if args.loss_weight_switch_frac is None else float(args.loss_weight_switch_frac),
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
