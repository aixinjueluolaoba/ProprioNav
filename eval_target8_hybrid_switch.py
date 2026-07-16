from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
from sb3_contrib import RecurrentPPO

from benchmark_state_dims_10k import load_env_overrides, make_state_env, predict_action, wrap_policy_env
from eval_target8_bad_seeds import DEFAULT_BAD_SEEDS, make_diagonal_env, parse_seed_list, summarize_bad_seed_rows, write_csv
from eval_target8_sweep import path_metrics, stall_metrics


DEFAULT_STATE_MODE = "observable10_target8_continuous_fixed_recovery_v6"


def should_switch(
    *,
    step: int,
    collision_count: int,
    no_progress_time: float,
    stuck_time: float,
    min_switch_step: int,
    no_progress_threshold: float,
    stuck_threshold: float,
    collision_threshold: int,
) -> bool:
    if step < min_switch_step:
        return False
    return (
        no_progress_time >= no_progress_threshold
        or stuck_time >= stuck_threshold
        or collision_count >= collision_threshold
    )


def evaluate_one(
    primary_model: RecurrentPPO,
    fallback_model: RecurrentPPO,
    seed: int,
    state_mode: str,
    *,
    diagonal: bool,
    env_overrides: dict[str, dict[str, Any]] | None,
    min_switch_step: int,
    no_progress_threshold: float,
    stuck_threshold: float,
    collision_threshold: int,
) -> dict[str, Any]:
    env = make_diagonal_env(seed, state_mode, env_overrides=env_overrides) if diagonal else make_state_env(seed, state_mode, env_overrides=env_overrides)
    policy_env = wrap_policy_env(env, state_mode, env_overrides=env_overrides)
    obs, _ = policy_env.reset(seed=seed)
    path = [env.pos.copy()]
    headings = [env.heading]
    actions: list[np.ndarray] = []
    total_reward = 0.0
    primary_state = None
    fallback_state = None
    primary_episode_start = True
    fallback_episode_start = True
    use_fallback = False
    switch_step = -1
    terminated = False
    truncated = False
    info: dict[str, Any] = {}
    collision_count = 0
    collision_flags: list[bool] = []
    recovery_count = 0
    stuck_run_steps = 0
    max_stuck_run_steps = 0
    while not (terminated or truncated):
        if use_fallback:
            action, fallback_state = predict_action(fallback_model, obs, fallback_episode_start, fallback_state)
            fallback_episode_start = False
        else:
            action, primary_state = predict_action(primary_model, obs, primary_episode_start, primary_state)
            primary_episode_start = False
        action_arr = np.asarray(action, dtype=np.float32).reshape(-1)
        obs, reward, terminated, truncated, info = policy_env.step(action)
        collided = float(info.get("time_since_collision", 1.0)) <= 1e-8
        collision_flags.append(collided)
        if collided:
            collision_count += 1
        no_progress_time = float(info.get("no_progress_time", 0.0) or 0.0)
        stuck_time = float(info.get("stuck_time", 0.0) or 0.0)
        currently_stuck = no_progress_time >= 0.8 or stuck_time >= 0.5
        if currently_stuck:
            stuck_run_steps += 1
            max_stuck_run_steps = max(max_stuck_run_steps, stuck_run_steps)
        else:
            stuck_run_steps = 0
        if int(info.get("recovery_mode", 0) or 0) != 0:
            recovery_count += 1
        step = len(actions) + 1
        if not use_fallback and should_switch(
            step=step,
            collision_count=collision_count,
            no_progress_time=no_progress_time,
            stuck_time=stuck_time,
            min_switch_step=min_switch_step,
            no_progress_threshold=no_progress_threshold,
            stuck_threshold=stuck_threshold,
            collision_threshold=collision_threshold,
        ):
            use_fallback = True
            fallback_episode_start = True
            switch_step = step
        total_reward += float(reward)
        actions.append(action_arr.copy())
        path.append(env.pos.copy())
        headings.append(env.heading)

    path_arr = np.asarray(path, dtype=np.float32)
    heading_arr = np.asarray(headings, dtype=np.float32)
    action_arr = np.asarray(actions, dtype=np.float32)
    final_distance = float(np.linalg.norm(env.target - path_arr[-1]))
    metrics = path_metrics(path_arr, heading_arr, action_arr, env.target)
    metrics.update(stall_metrics(path_arr, np.asarray(collision_flags, dtype=bool)))
    metrics["recovery_action_count"] = float(recovery_count)
    metrics["recovery_action_rate"] = recovery_count / max(float(len(action_arr)), 1.0)
    metrics["max_stuck_run_steps"] = float(max_stuck_run_steps)
    metrics["recovery_escape_success"] = 0.0
    metrics["avg_recovery_resolve_steps"] = float(len(action_arr))
    policy_env.close()
    return {
        "seed": seed,
        "state_mode": state_mode,
        "diagonal": int(diagonal),
        "steps": len(path_arr) - 1,
        "reward": total_reward,
        "final_distance": final_distance,
        "success": int(bool(info.get("is_success", final_distance <= env.target_radius))),
        "collisions": collision_count,
        "used_fallback": int(use_fallback),
        "switch_step": switch_step,
        **metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--primary-model", required=True)
    parser.add_argument("--fallback-model", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed-list", default=None)
    parser.add_argument("--seed-file", default=None)
    parser.add_argument("--seed-start", type=int, default=None)
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--diagonal", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--env-overrides-file", default=None)
    parser.add_argument("--state-mode", default=DEFAULT_STATE_MODE)
    parser.add_argument("--min-switch-step", type=int, default=45)
    parser.add_argument("--no-progress-threshold", type=float, default=1.4)
    parser.add_argument("--stuck-threshold", type=float, default=0.9)
    parser.add_argument("--collision-threshold", type=int, default=35)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    primary_model = RecurrentPPO.load(args.primary_model, device=args.device)
    fallback_model = RecurrentPPO.load(args.fallback_model, device=args.device)
    env_overrides = load_env_overrides(args.env_overrides_file)
    if args.seed_start is not None:
        episodes = args.episodes or len(DEFAULT_BAD_SEEDS)
        seeds = [args.seed_start + index for index in range(episodes)]
    else:
        seeds = parse_seed_list(args.seed_list, args.seed_file)

    rows: list[dict[str, Any]] = []
    for index, seed in enumerate(seeds):
        row = evaluate_one(
            primary_model,
            fallback_model,
            seed,
            args.state_mode,
            diagonal=args.diagonal,
            env_overrides=env_overrides,
            min_switch_step=args.min_switch_step,
            no_progress_threshold=args.no_progress_threshold,
            stuck_threshold=args.stuck_threshold,
            collision_threshold=args.collision_threshold,
        )
        rows.append(row)
        print(
            "hybrid_eval index={index} seed={seed} success={success} steps={steps} "
            "fallback={fallback} switch={switch} final_angle={angle:.3f} collisions={collisions}".format(
                index=index + 1,
                seed=seed,
                success=row["success"],
                steps=row["steps"],
                fallback=row["used_fallback"],
                switch=row["switch_step"],
                angle=float(row["final_abs_angle_error_deg"]),
                collisions=row["collisions"],
            ),
            flush=True,
        )
    write_csv(out_dir / "hybrid_eval_results.csv", rows)
    write_csv(out_dir / "hybrid_eval_summary.csv", [summarize_bad_seed_rows(rows)])
    (out_dir / "hybrid_seed_list.txt").write_text("\n".join(str(seed) for seed in seeds) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
