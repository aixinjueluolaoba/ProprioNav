from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any

import numpy as np
from sb3_contrib import RecurrentPPO

from benchmark_state_dims_10k import load_env_overrides, make_state_env, predict_action, wrap_policy_env
from eval_target8_sweep import path_metrics, stall_metrics


DEFAULT_STATE_MODE = "observable12_target8_macro_library_v11re_stage2"
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
START = np.asarray([-520.0, -520.0], dtype=np.float32)
TARGET = np.asarray([520.0, 520.0], dtype=np.float32)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def parse_seed_list(seed_list: str | None, seed_file: str | None) -> list[int]:
    if seed_file:
        values = [line.strip() for line in Path(seed_file).read_text(encoding="utf-8").splitlines()]
        seeds = [int(value) for value in values if value]
        if seeds:
            return seeds
    if seed_list:
        seeds = [int(value.strip()) for value in seed_list.split(",") if value.strip()]
        if seeds:
            return seeds
    return list(DEFAULT_BAD_SEEDS)


def make_diagonal_env(seed: int, state_mode: str, env_overrides: dict[str, dict[str, Any]] | None = None):
    env = make_state_env(seed, state_mode, env_overrides=env_overrides)
    env.tree_count = 85
    env.mountain_count = 58
    env.mountain_radius_range = (45.0, 105.0)
    env.concave_mountain_probability = 0.995
    env.deep_concave_mountain_probability = 0.90
    env.trapped_start_probability = 0.0
    env.opposite_heading_probability = 0.65

    def fixed_start_and_target() -> tuple[np.ndarray, np.ndarray]:
        env._forced_start_mountain = None
        return START.copy(), TARGET.copy()

    env._sample_start_and_target = fixed_start_and_target  # type: ignore[method-assign]
    return env


def evaluate_one(
    model: RecurrentPPO,
    seed: int,
    state_mode: str,
    env_overrides: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    env = make_diagonal_env(seed, state_mode, env_overrides=env_overrides)
    policy_env = wrap_policy_env(env, state_mode, env_overrides=env_overrides)
    obs, _ = policy_env.reset(seed=seed)
    path = [env.pos.copy()]
    headings = [env.heading]
    actions: list[np.ndarray] = []
    total_reward = 0.0
    state = None
    episode_start = True
    terminated = False
    truncated = False
    info: dict[str, Any] = {}
    collision_count = 0
    collision_flags: list[bool] = []
    recovery_count = 0
    stuck_run_steps = 0
    max_stuck_run_steps = 0
    recovery_active = False
    recovery_start_step: int | None = None
    recovery_resolve_steps: list[int] = []
    while not (terminated or truncated):
        action, state = predict_action(model, obs, episode_start, state)
        action_arr = np.asarray(action, dtype=np.float32).reshape(-1)
        obs, reward, terminated, truncated, info = policy_env.step(action)
        episode_start = False
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
        recovery_mode = int(info.get("recovery_mode", 0) or 0)
        if recovery_mode != 0:
            recovery_count += 1
            if not recovery_active:
                recovery_active = True
                recovery_start_step = len(path) - 1
        elif recovery_active:
            if recovery_start_step is not None and no_progress_time <= 0.25 and stuck_time <= 0.25:
                recovery_resolve_steps.append((len(path) - 1) - recovery_start_step)
            recovery_active = False
            recovery_start_step = None
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
    metrics["recovery_escape_success"] = float(1 if recovery_resolve_steps else 0)
    metrics["avg_recovery_resolve_steps"] = float(np.mean(recovery_resolve_steps)) if recovery_resolve_steps else float(len(action_arr))
    policy_env.close()
    return {
        "seed": seed,
        "state_mode": state_mode,
        "env_action_mode": env.action_mode,
        "steps": len(path_arr) - 1,
        "reward": total_reward,
        "final_distance": final_distance,
        "success": int(bool(info.get("is_success", final_distance <= env.target_radius))),
        "collisions": collision_count,
        **metrics,
    }


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def avg(key: str) -> float:
        return float(np.mean([float(row[key]) for row in rows])) if rows else 0.0

    return {
        "episodes": len(rows),
        "success_rate": avg("success"),
        "avg_steps": avg("steps"),
        "avg_final_distance": avg("final_distance"),
        "avg_path_efficiency": avg("path_efficiency"),
        "avg_abs_angle_error_deg": avg("avg_abs_angle_error_deg"),
        "avg_final_abs_angle_error_deg": avg("final_abs_angle_error_deg"),
        "avg_last5_abs_angle_error_deg": avg("last5_abs_angle_error_deg"),
        "avg_recovery_action_rate": avg("recovery_action_rate"),
        "recovery_escape_rate": avg("recovery_escape_success"),
        "avg_recovery_resolve_steps": avg("avg_recovery_resolve_steps"),
        "avg_collision_count": avg("collisions"),
        "avg_still_step_count": avg("still_step_count"),
        "avg_still_step_rate": avg("still_step_rate"),
        "avg_collision_still_step_count": avg("collision_still_step_count"),
        "avg_collision_still_step_rate": avg("collision_still_step_rate"),
        "avg_max_still_run": avg("max_still_run"),
        "avg_max_stuck_run_steps": avg("max_stuck_run_steps"),
    }


def summarize_bad_seed_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary = summarize_rows(rows)
    success_rows = [row for row in rows if int(row["success"]) == 1]
    failed_rows = [row for row in rows if int(row["success"]) == 0]
    summary["success_seed_count"] = len(success_rows)
    summary["failed_seed_count"] = len(failed_rows)
    summary["failed_seed_list"] = ",".join(str(int(row["seed"])) for row in failed_rows)
    if rows:
        worst = max(rows, key=lambda row: (int(row["collisions"]), int(row["steps"])))
        summary["worst_seed"] = int(worst["seed"])
        summary["worst_collisions"] = int(worst["collisions"])
        summary["worst_steps"] = int(worst["steps"])
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed-list", default=None)
    parser.add_argument("--seed-file", default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--env-overrides-file", default=None)
    parser.add_argument("--state-mode", default=DEFAULT_STATE_MODE)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model = RecurrentPPO.load(args.model, device=args.device)
    env_overrides = load_env_overrides(args.env_overrides_file)
    seeds = parse_seed_list(args.seed_list, args.seed_file)

    rows: list[dict[str, Any]] = []
    for index, seed in enumerate(seeds):
        row = evaluate_one(model, seed, args.state_mode, env_overrides=env_overrides)
        rows.append(row)
        print(
            "bad_seed_eval index={index} seed={seed} success={success} steps={steps} "
            "final_angle={angle:.3f} recovery={recovery:.3f} collisions={collisions}".format(
                index=index + 1,
                seed=seed,
                success=row["success"],
                steps=row["steps"],
                angle=float(row["final_abs_angle_error_deg"]),
                recovery=float(row["recovery_action_rate"]),
                collisions=row["collisions"],
            ),
            flush=True,
        )
    write_csv(out_dir / "bad_seed_eval_results.csv", rows)
    write_csv(out_dir / "bad_seed_eval_summary.csv", [summarize_bad_seed_rows(rows)])
    (out_dir / "bad_seed_list.txt").write_text("\n".join(str(seed) for seed in seeds) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
