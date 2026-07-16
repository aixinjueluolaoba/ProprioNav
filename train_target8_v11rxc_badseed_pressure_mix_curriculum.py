from __future__ import annotations

import argparse
import gc
import numpy as np
import os
import shutil
import time
from pathlib import Path

import torch
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv

from benchmark_state_dims_10k import (
    CHECKPOINT_INTERVAL,
    EpisodeStopAndCheckpointCallback,
    Experiment,
    MAX_STEPS,
    N_ENVS,
    PROGRESS_INTERVAL,
    count_params,
    eval_model,
    load_env_overrides,
    make_model,
    observation_dim,
    write_results,
    wrap_policy_env,
)
from eval_target8_pressure import make_pressure_env
from train_target8_v11re_reward_exit_curriculum import warm_start_policy
from train_target8_v11rxc_bad_seed_curriculum import make_bad_seed_env, parse_seed_pool
from train_target8_v11rxc_disagreement_regularized_bad_seed import (
    AuxiliaryRegularizer,
    apply_gradient_mask_for_mode,
    configure_trainable_params,
    parse_optional_int_set,
    parse_seed_mode_whitelist,
    rebuild_policy_optimizer,
    restore_full_policy_optimizer_for_save,
)


STAGE = Experiment(
    name="rppo_medium_lstm128x2_observable12_target8_macro_library_v11rxc_stage2",
    algo="RecurrentPPO",
    policy="MlpLstmPolicy",
    net_arch={"pi": [128, 128], "vf": [128, 128]},
    state_mode="observable12_target8_macro_library_v11rxc_stage2",
    lstm_hidden_size=128,
    n_lstm_layers=2,
)


DEFAULT_PRESSURE_SEEDS = [
    5_200_015,
    5_200_015,
    5_200_015,
    5_200_015,
    5_200_008,
    5_200_022,
]


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


class PressureTriggeredSpecialistCallback(EpisodeStopAndCheckpointCallback):
    def __init__(
        self,
        target_episodes: int,
        checkpoint_dir: Path,
        *,
        specialist_regularizer: AuxiliaryRegularizer,
        pressure_env_start: int,
        trigger_min_pressure_active_frac: float = 0.0,
        trigger_scale_by_pressure_active_frac: bool = False,
        trigger_min_no_progress_time: float | None = None,
        trigger_min_stuck_time: float | None = None,
        trigger_recovery_modes: set[int] | None = None,
        trigger_min_recent_collision_norm: float | None = None,
        trigger_require_stuck_macro_context: bool = False,
        trigger_require_targeted_stuck_macro_context: bool = False,
        checkpoint_interval: int = CHECKPOINT_INTERVAL,
        progress_interval: int = PROGRESS_INTERVAL,
        episode_offset: int = 0,
    ) -> None:
        super().__init__(
            target_episodes,
            checkpoint_dir,
            checkpoint_interval=checkpoint_interval,
            progress_interval=progress_interval,
            episode_offset=episode_offset,
        )
        self.specialist_regularizer = specialist_regularizer
        self.pressure_env_start = max(0, int(pressure_env_start))
        self.trigger_min_pressure_active_frac = max(0.0, float(trigger_min_pressure_active_frac))
        self.trigger_scale_by_pressure_active_frac = bool(trigger_scale_by_pressure_active_frac)
        self.trigger_min_no_progress_time = (
            None if trigger_min_no_progress_time is None else float(trigger_min_no_progress_time)
        )
        self.trigger_min_stuck_time = None if trigger_min_stuck_time is None else float(trigger_min_stuck_time)
        self.trigger_recovery_modes = None if trigger_recovery_modes is None else {int(value) for value in trigger_recovery_modes}
        self.trigger_min_recent_collision_norm = (
            None if trigger_min_recent_collision_norm is None else float(trigger_min_recent_collision_norm)
        )
        self.trigger_require_stuck_macro_context = bool(trigger_require_stuck_macro_context)
        self.trigger_require_targeted_stuck_macro_context = bool(trigger_require_targeted_stuck_macro_context)
        self.rollout_pressure_steps = 0
        self.rollout_triggered_steps = 0
        self.rollout_index = 0
        self.last_pressure_active_frac = 0.0
        self.last_applied_loss_weight = 0.0
        self.last_pressure_steps = 0
        self.last_triggered_steps = 0

    def _info_matches_trigger(self, info: dict[str, object]) -> bool:
        if self.trigger_min_no_progress_time is not None:
            if float(info.get("no_progress_time", 0.0) or 0.0) < self.trigger_min_no_progress_time:
                return False
        if self.trigger_min_stuck_time is not None:
            if float(info.get("stuck_time", 0.0) or 0.0) < self.trigger_min_stuck_time:
                return False
        if self.trigger_recovery_modes is not None:
            if int(info.get("recovery_mode", 0) or 0) not in self.trigger_recovery_modes:
                return False
        if self.trigger_min_recent_collision_norm is not None:
            if float(info.get("wrapper_recent_collision_norm", 0.0) or 0.0) < self.trigger_min_recent_collision_norm:
                return False
        if self.trigger_require_stuck_macro_context and not bool(info.get("stuck_macro_context", False)):
            return False
        if self.trigger_require_targeted_stuck_macro_context and not bool(info.get("targeted_stuck_macro_context", False)):
            return False
        return True

    def _on_step(self) -> bool:
        keep_going = super()._on_step()
        infos = self.locals.get("infos")
        if infos is None:
            return keep_going
        info_list = list(infos)
        if self.pressure_env_start >= len(info_list):
            return keep_going
        for info in info_list[self.pressure_env_start:]:
            if not isinstance(info, dict):
                continue
            self.rollout_pressure_steps += 1
            if self._info_matches_trigger(info):
                self.rollout_triggered_steps += 1
        return keep_going

    def _on_rollout_end(self) -> None:
        pressure_steps = int(self.rollout_pressure_steps)
        triggered_steps = int(self.rollout_triggered_steps)
        pressure_active_frac = float(triggered_steps / pressure_steps) if pressure_steps > 0 else 0.0
        if pressure_active_frac >= self.trigger_min_pressure_active_frac:
            applied_weight = self.specialist_regularizer.base_loss_weight
            if self.trigger_scale_by_pressure_active_frac:
                applied_weight *= pressure_active_frac
        else:
            applied_weight = 0.0
        self.specialist_regularizer.set_loss_weight(applied_weight)
        self.rollout_index += 1
        self.last_pressure_active_frac = pressure_active_frac
        self.last_applied_loss_weight = float(applied_weight)
        self.last_pressure_steps = pressure_steps
        self.last_triggered_steps = triggered_steps
        print(
            "specialist_trigger rollout={rollout} pressure_steps={pressure_steps} triggered_steps={triggered_steps} pressure_active_frac={frac:.4f} applied_loss_weight={weight:.6f}".format(
                rollout=self.rollout_index,
                pressure_steps=pressure_steps,
                triggered_steps=triggered_steps,
                frac=pressure_active_frac,
                weight=float(applied_weight),
            ),
            flush=True,
        )
        self.rollout_pressure_steps = 0
        self.rollout_triggered_steps = 0


def parse_pressure_seed_pool(seed_pool: str | None) -> list[int]:
    if not seed_pool:
        return list(DEFAULT_PRESSURE_SEEDS)
    out: list[int] = []
    for value in seed_pool.split(","):
        value = value.strip()
        if not value:
            continue
        out.append(int(value))
    if not out:
        raise ValueError("parsed empty pressure seed pool")
    return out


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


def make_mixed_env(
    rank: int,
    state_mode: str,
    bad_seed_pool: list[int],
    pressure_seed_pool: list[int],
    bad_seed_env_count: int,
    env_overrides: dict[str, dict[str, object]] | None = None,
):
    if rank < bad_seed_env_count:
        return make_bad_seed_env(rank, state_mode, seed_pool=bad_seed_pool, env_overrides=env_overrides)
    return make_pressure_seed_env(rank - bad_seed_env_count, state_mode, seed_pool=pressure_seed_pool, env_overrides=env_overrides)


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
    parser.add_argument("--seed-base", type=int, default=11_481_000)
    parser.add_argument("--eval-seed-base", type=int, default=11_491_000)
    parser.add_argument("--checkpoint-interval", type=int, default=None)
    parser.add_argument("--progress-interval", type=int, default=None)
    parser.add_argument("--env-overrides-file", default=None)
    parser.add_argument("--resume-model", required=True)
    parser.add_argument("--bad-seed-pool", default=None)
    parser.add_argument("--pressure-seed-pool", default=None)
    parser.add_argument("--bad-seed-ratio", type=float, default=0.5)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--trainable-param-mode", default="all")
    parser.add_argument("--specialist-dataset", default=None)
    parser.add_argument("--specialist-loss-weight", type=float, default=0.0)
    parser.add_argument("--specialist-batch-size", type=int, default=128)
    parser.add_argument("--specialist-loss-heads", default="macro_only")
    parser.add_argument("--specialist-max-samples", type=int, default=None)
    parser.add_argument("--specialist-disagreement-filter", default="macro")
    parser.add_argument("--specialist-seed-whitelist", default=None)
    parser.add_argument("--specialist-seed-mode-whitelist", default=None)
    parser.add_argument("--specialist-min-stuck-time", type=float, default=None)
    parser.add_argument("--specialist-min-no-progress-time", type=float, default=None)
    parser.add_argument("--specialist-recovery-modes", default=None)
    parser.add_argument("--specialist-trigger-min-pressure-active-frac", type=float, default=None)
    parser.add_argument("--specialist-trigger-scale-by-pressure-active-frac", action="store_true")
    parser.add_argument("--specialist-trigger-min-no-progress-time", type=float, default=None)
    parser.add_argument("--specialist-trigger-min-stuck-time", type=float, default=None)
    parser.add_argument("--specialist-trigger-recovery-modes", default=None)
    parser.add_argument("--specialist-trigger-min-recent-collision-norm", type=float, default=None)
    parser.add_argument("--specialist-trigger-require-stuck-macro-context", action="store_true")
    parser.add_argument("--specialist-trigger-require-targeted-stuck-macro-context", action="store_true")
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
    bad_seed_pool = parse_seed_pool(args.bad_seed_pool)
    pressure_seed_pool = parse_pressure_seed_pool(args.pressure_seed_pool)
    bad_seed_ratio = float(max(0.0, min(1.0, args.bad_seed_ratio)))

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
    (output_dir / "bad_seed_pool.txt").write_text("\n".join(str(seed) for seed in bad_seed_pool) + "\n", encoding="utf-8")
    (output_dir / "pressure_seed_pool.txt").write_text("\n".join(str(seed) for seed in pressure_seed_pool) + "\n", encoding="utf-8")

    csv_path = output_dir / "v11rxc_badseed_pressure_mix_results.csv"
    md_path = output_dir / "v11rxc_badseed_pressure_mix_results.md"
    stage_dir = output_dir / STAGE.name
    stage_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = stage_dir / "models"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    env = None
    model = None
    results: list[dict[str, object]] = []
    specialist_regularizer = None
    specialist_loss_last = float("nan")
    specialist_trigger_callback = None
    try:
        bad_seed_env_count = int(round(int(args.n_envs) * bad_seed_ratio))
        env = SubprocVecEnv(
            [
                make_mixed_env(
                    rank,
                    STAGE.state_mode,
                    bad_seed_pool=bad_seed_pool,
                    pressure_seed_pool=pressure_seed_pool,
                    bad_seed_env_count=bad_seed_env_count,
                    env_overrides=env_overrides,
                )
                for rank in range(int(args.n_envs))
            ],
            start_method="fork",
        )
        model = make_model(STAGE, env, int(args.seed_base), args.device)
        matched = warm_start_policy(model, Path(args.resume_model), args.device)
        print(f"warm_start matched_policy_tensors={matched}", flush=True)
        trainable_param_mode, macro_slice, trainable_slice_params = configure_trainable_params(model, args.trainable_param_mode)
        print(
            f"trainable_param_mode={trainable_param_mode} trainable_params={trainable_slice_params} macro_slice={macro_slice}",
            flush=True,
        )
        rebuild_policy_optimizer(model, float(args.learning_rate))
        original_step = model.policy.optimizer.step
        needs_masked_step = bool(args.specialist_dataset and float(args.specialist_loss_weight) > 0.0) or trainable_param_mode == "macro_slice_only"
        if args.specialist_dataset and float(args.specialist_loss_weight) > 0.0:
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
                "specialist_regularizer samples={samples} batch_size={batch} loss_weight={weight} loss_heads={heads} disagreement_filter={disagreement} seed_whitelist={seed_whitelist} seed_mode_whitelist={seed_mode_whitelist} min_stuck_time={min_stuck_time} min_no_progress_time={min_no_progress_time} recovery_modes={recovery_modes}".format(
                    samples=specialist_regularizer.size,
                    batch=specialist_regularizer.batch_size,
                    weight=specialist_regularizer.loss_weight,
                    heads=args.specialist_loss_heads,
                    disagreement=args.specialist_disagreement_filter,
                    seed_whitelist=args.specialist_seed_whitelist or "",
                    seed_mode_whitelist=args.specialist_seed_mode_whitelist or "",
                    min_stuck_time="" if args.specialist_min_stuck_time is None else args.specialist_min_stuck_time,
                    min_no_progress_time="" if args.specialist_min_no_progress_time is None else args.specialist_min_no_progress_time,
                    recovery_modes=args.specialist_recovery_modes or "",
                ),
                flush=True,
            )
        trigger_enabled = (
            specialist_regularizer is not None
            and (
                args.specialist_trigger_min_pressure_active_frac is not None
                or args.specialist_trigger_scale_by_pressure_active_frac
                or args.specialist_trigger_min_no_progress_time is not None
                or args.specialist_trigger_min_stuck_time is not None
                or args.specialist_trigger_recovery_modes not in {None, ""}
                or args.specialist_trigger_min_recent_collision_norm is not None
                or args.specialist_trigger_require_stuck_macro_context
                or args.specialist_trigger_require_targeted_stuck_macro_context
            )
        )
        if trigger_enabled:
            specialist_regularizer.set_loss_weight(0.0)
        if needs_masked_step:
            def masked_step(closure=None):
                nonlocal specialist_loss_last
                if specialist_regularizer is not None:
                    specialist_loss_last = specialist_regularizer.backward(model, device)
                apply_gradient_mask_for_mode(model, trainable_param_mode, macro_slice)
                if closure is None:
                    return original_step()
                return original_step(closure)

            model.policy.optimizer.step = masked_step  # type: ignore[method-assign]
        _, trainable = count_params(model)
        if trigger_enabled:
            specialist_trigger_callback = PressureTriggeredSpecialistCallback(
                stage_episodes,
                ckpt_dir,
                specialist_regularizer=specialist_regularizer,
                pressure_env_start=bad_seed_env_count,
                trigger_min_pressure_active_frac=(
                    0.0
                    if args.specialist_trigger_min_pressure_active_frac is None
                    else float(args.specialist_trigger_min_pressure_active_frac)
                ),
                trigger_scale_by_pressure_active_frac=bool(args.specialist_trigger_scale_by_pressure_active_frac),
                trigger_min_no_progress_time=args.specialist_trigger_min_no_progress_time,
                trigger_min_stuck_time=args.specialist_trigger_min_stuck_time,
                trigger_recovery_modes=parse_optional_int_set(args.specialist_trigger_recovery_modes),
                trigger_min_recent_collision_norm=args.specialist_trigger_min_recent_collision_norm,
                trigger_require_stuck_macro_context=bool(args.specialist_trigger_require_stuck_macro_context),
                trigger_require_targeted_stuck_macro_context=bool(
                    args.specialist_trigger_require_targeted_stuck_macro_context
                ),
                checkpoint_interval=checkpoint_interval,
                progress_interval=progress_interval,
            )
            callback = specialist_trigger_callback
        else:
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
        if specialist_regularizer is not None:
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
            "trained_episodes": stage_episodes,
            "train_seconds": train_seconds,
            "model_path": str(model_path),
            "model_size_mb": model_path.stat().st_size / (1024 * 1024),
            "status": "ok",
            "error": "",
            "bad_seed_pool_size": len(bad_seed_pool),
            "bad_seed_pool_path": str(output_dir / "bad_seed_pool.txt"),
            "pressure_seed_pool_size": len(pressure_seed_pool),
            "pressure_seed_pool_path": str(output_dir / "pressure_seed_pool.txt"),
            "bad_seed_ratio": bad_seed_ratio,
            "bad_seed_env_count": bad_seed_env_count,
            "learning_rate": float(args.learning_rate),
            "trainable_param_mode": trainable_param_mode,
            "trainable_slice_params": trainable_slice_params,
            "specialist_dataset": args.specialist_dataset or "",
            "specialist_dataset_samples": "" if specialist_regularizer is None else specialist_regularizer.size,
            "specialist_loss_weight": float(args.specialist_loss_weight),
            "specialist_loss_heads": args.specialist_loss_heads,
            "specialist_batch_size": int(args.specialist_batch_size),
            "specialist_disagreement_filter": args.specialist_disagreement_filter,
            "specialist_seed_whitelist": args.specialist_seed_whitelist or "",
            "specialist_seed_mode_whitelist": args.specialist_seed_mode_whitelist or "",
            "specialist_min_stuck_time": "" if args.specialist_min_stuck_time is None else args.specialist_min_stuck_time,
            "specialist_min_no_progress_time": "" if args.specialist_min_no_progress_time is None else args.specialist_min_no_progress_time,
            "specialist_recovery_modes": args.specialist_recovery_modes or "",
            "specialist_trigger_min_pressure_active_frac": (
                "" if args.specialist_trigger_min_pressure_active_frac is None else args.specialist_trigger_min_pressure_active_frac
            ),
            "specialist_trigger_scale_by_pressure_active_frac": bool(
                args.specialist_trigger_scale_by_pressure_active_frac
            ),
            "specialist_trigger_min_no_progress_time": (
                "" if args.specialist_trigger_min_no_progress_time is None else args.specialist_trigger_min_no_progress_time
            ),
            "specialist_trigger_min_stuck_time": (
                "" if args.specialist_trigger_min_stuck_time is None else args.specialist_trigger_min_stuck_time
            ),
            "specialist_trigger_recovery_modes": args.specialist_trigger_recovery_modes or "",
            "specialist_trigger_min_recent_collision_norm": (
                ""
                if args.specialist_trigger_min_recent_collision_norm is None
                else args.specialist_trigger_min_recent_collision_norm
            ),
            "specialist_trigger_require_stuck_macro_context": bool(args.specialist_trigger_require_stuck_macro_context),
            "specialist_trigger_require_targeted_stuck_macro_context": bool(
                args.specialist_trigger_require_targeted_stuck_macro_context
            ),
            "specialist_trigger_last_pressure_active_frac": (
                "" if specialist_trigger_callback is None else specialist_trigger_callback.last_pressure_active_frac
            ),
            "specialist_trigger_last_applied_loss_weight": (
                "" if specialist_trigger_callback is None else specialist_trigger_callback.last_applied_loss_weight
            ),
            "specialist_trigger_last_pressure_steps": (
                "" if specialist_trigger_callback is None else specialist_trigger_callback.last_pressure_steps
            ),
            "specialist_trigger_last_triggered_steps": (
                "" if specialist_trigger_callback is None else specialist_trigger_callback.last_triggered_steps
            ),
            "specialist_loss_last": specialist_loss_last,
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
