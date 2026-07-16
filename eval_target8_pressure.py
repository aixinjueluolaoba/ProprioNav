from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any

import numpy as np
from sb3_contrib import RecurrentPPO

from benchmark_state_dims_10k import load_env_overrides, predict_action, resolve_env_overrides, wrap_policy_env
from blind_nav_rl.env import BlindNavEnv, MountainObstacle, TreeObstacle
from eval_target8_sweep import path_metrics, stall_metrics, write_csv
from render_target8_eval import EpisodeTrace, concat_videos, render_episode


INFO_RECOVERY_ACTION_MODES = {
    "target_relative_macro_recovery",
    "target_relative_macro_trigger_recovery",
    "heading_relative_two_phase_macro_recovery",
    "heading_relative_macro_library_recovery",
}


def make_pressure_env(
    seed: int,
    action_mode: str = "recovery",
    *,
    state_mode: str | None = None,
    env_overrides: dict[str, dict[str, Any]] | None = None,
) -> BlindNavEnv:
    if action_mode == "full180":
        env_action_mode = "target_relative_full"
        reward_profile = "target_line_full180_pressure"
        mountain_count = 36
        mountain_radius_range = (50.0, 95.0)
    elif action_mode == "recovery":
        env_action_mode = "target_relative_recovery"
        reward_profile = "target_line_recovery_angle"
        mountain_count = 36
        mountain_radius_range = (50.0, 95.0)
    elif action_mode == "macro":
        env_action_mode = "target_relative_macro_recovery"
        reward_profile = "target_line_macro_recovery"
        mountain_count = 36
        mountain_radius_range = (50.0, 95.0)
    elif action_mode == "macro_trigger":
        env_action_mode = "target_relative_macro_trigger_recovery"
        reward_profile = "target_line_macro_recovery"
        mountain_count = 36
        mountain_radius_range = (50.0, 95.0)
    elif action_mode == "v4":
        env_action_mode = "target_relative_macro_recovery"
        reward_profile = "target_line_macro_recovery"
        mountain_count = 36
        mountain_radius_range = (50.0, 95.0)
    elif action_mode == "v5":
        env_action_mode = "target_relative_macro_recovery"
        reward_profile = "target_line_macro_recovery"
        mountain_count = 36
        mountain_radius_range = (35.0, 85.0)
    elif action_mode == "v6":
        env_action_mode = "target_relative_macro_recovery"
        reward_profile = "target_line_macro_recovery_v6"
        mountain_count = 40
        mountain_radius_range = (35.0, 85.0)
    elif action_mode == "v7":
        env_action_mode = "heading_relative_two_phase_macro_recovery"
        reward_profile = "target_line_macro_recovery_v7"
        mountain_count = 44
        mountain_radius_range = (35.0, 85.0)
    elif action_mode == "v8":
        env_action_mode = "heading_relative_two_phase_macro_recovery"
        reward_profile = "target_line_macro_recovery_v7"
        mountain_count = 48
        mountain_radius_range = (35.0, 85.0)
    elif action_mode == "v9":
        env_action_mode = "target_relative_macro_trigger_recovery"
        reward_profile = "target_line_macro_recovery_v7"
        mountain_count = 48
        mountain_radius_range = (35.0, 85.0)
    elif action_mode == "v10":
        env_action_mode = "heading_relative_two_phase_macro_recovery"
        reward_profile = "target_line_macro_recovery_v10"
        mountain_count = 48
        mountain_radius_range = (35.0, 85.0)
    elif action_mode == "v10b9":
        env_action_mode = "heading_relative_two_phase_macro_recovery"
        reward_profile = "target_line_macro_recovery_v10"
        mountain_count = 48
        mountain_radius_range = (35.0, 85.0)
    elif action_mode == "v11":
        env_action_mode = "heading_relative_macro_library_recovery"
        reward_profile = "target_line_macro_recovery_v10"
        mountain_count = 48
        mountain_radius_range = (35.0, 85.0)
    else:
        raise ValueError(f"Unsupported action_mode: {action_mode}")
    kwargs = dict(
        seed=seed,
        world_size=1200.0,
        max_steps=260,
        action_mode=env_action_mode,
        delta_angle_limit=math.radians(60.0),
        binary_speed=True,
        allow_zero_speed=False,
        random_initial_heading=True,
        tree_count=45,
        mountain_count=mountain_count,
        tree_radius=18.0,
        mountain_radius_range=mountain_radius_range,
        concave_mountain_probability=0.995 if action_mode in {"v10", "v10b9", "v11"} else (0.99 if action_mode == "v8" else (0.98 if action_mode == "v7" else (0.96 if action_mode == "v6" else 0.95))),
        deep_concave_mountain_probability=0.88 if action_mode in {"v10", "v10b9", "v11"} else (0.82 if action_mode == "v8" else (0.72 if action_mode == "v7" else (0.60 if action_mode == "v6" else 0.55))),
        target_radius=8.0,
        stop_distance_range=None,
        start_target_distance_range=(260.0, 860.0) if action_mode in {"v10", "v10b9", "v11"} else ((450.0, 920.0) if action_mode in {"v8", "v9"} else ((450.0, 900.0) if action_mode == "v7" else (450.0, 850.0))),
        align_angle_tolerance=math.radians(10.0),
        reward_profile=reward_profile,
        position_obs_dt_range=(0.45, 0.85) if action_mode in {"v9", "v10", "v10b9", "v11"} else (0.35, 0.75),
        coord_noise_std=1.8 if action_mode in {"v9", "v10", "v10b9", "v11"} else 1.5,
        trapped_start_probability=0.75 if action_mode in {"v10", "v10b9", "v11"} else (0.70 if action_mode in {"v8", "v9"} else 0.0),
        opposite_heading_probability=0.55 if action_mode in {"v10", "v10b9"} else (0.50 if action_mode == "v9" else 0.0),
    )
    override_mode = state_mode or default_state_mode(action_mode)
    kwargs.update(resolve_env_overrides(override_mode, env_overrides))
    return BlindNavEnv(**kwargs)


def straight_line_hits(env: BlindNavEnv) -> tuple[int, int]:
    mountain_hits = 0
    tree_hits = 0
    for obstacle in env.obstacles:
        if isinstance(obstacle, MountainObstacle):
            hit, _ = env._segment_polygon_hit(env.pos, env.target, obstacle)
            mountain_hits += int(hit is not None)
        elif isinstance(obstacle, TreeObstacle):
            hit, _ = env._segment_circle_hit(env.pos, env.target, obstacle.center, obstacle.radius)
            tree_hits += int(hit)
    return mountain_hits, tree_hits


def find_pressure_seeds(
    seed_start: int,
    episodes: int,
    max_scan: int,
    action_mode: str,
    *,
    state_mode: str | None = None,
    env_overrides: dict[str, dict[str, Any]] | None = None,
) -> list[int]:
    seeds: list[int] = []
    for seed in range(seed_start, seed_start + max_scan):
        env = make_pressure_env(seed, action_mode, state_mode=state_mode, env_overrides=env_overrides)
        mountain_hits, _ = straight_line_hits(env)
        if mountain_hits >= 1:
            seeds.append(seed)
            if len(seeds) >= episodes:
                break
    if len(seeds) < episodes:
        raise RuntimeError(f"found only {len(seeds)} pressure seeds from scan {max_scan}")
    return seeds


def default_state_mode(action_mode: str) -> str:
    if action_mode == "v5":
        return "observable12_target8_discrete_macro_v5"
    if action_mode == "v6":
        return "observable12_target8_discrete_macro_v6"
    if action_mode == "v7":
        return "observable12_target8_discrete_macro_v7"
    if action_mode == "v8":
        return "observable12_target8_discrete_macro_v8"
    if action_mode == "v9":
        return "observable12_target8_discrete_trigger_v9_stage2"
    if action_mode == "v10":
        return "observable12_target8_discrete_recovery_v10_stage3"
    if action_mode == "v10b9":
        return "observable12_target8_discrete_recovery_v10b9_stage3"
    if action_mode == "v11":
        return "observable12_target8_macro_library_v11_stage2"
    if action_mode == "v4":
        return "observable12_target8_recovery_v4"
    if action_mode == "full180":
        return "observable8_target8_full180_pressure"
    if action_mode == "macro":
        return "observable8_target8_macro_recovery_dense_halfmountain"
    if action_mode == "macro_trigger":
        return "observable8_target8_macro_trigger_dense_halfmountain"
    return "observable8_target8_recovery_angle"


def evaluate_one(
    model: RecurrentPPO,
    seed: int,
    action_mode: str,
    state_mode: str,
    env_overrides: dict[str, dict[str, Any]] | None = None,
) -> tuple[dict[str, Any], EpisodeTrace]:
    env = make_pressure_env(seed, action_mode, state_mode=state_mode, env_overrides=env_overrides)
    mountain_hits, tree_hits = straight_line_hits(env)
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
    collisions = 0
    collision_flags: list[bool] = []
    effective_recovery_count = 0
    recovery_trigger_step: int | None = None
    recovery_escape_success = 0
    stuck_before_recovery = False
    stuck_run_steps = 0
    max_stuck_run_steps = 0
    recovery_resolve_steps: list[int] = []
    while not (terminated or truncated):
        action, state = predict_action(model, obs, episode_start, state)
        action_arr = np.asarray(action, dtype=np.float32).reshape(-1)
        obs, reward, terminated, truncated, info = policy_env.step(action)
        episode_start = False
        collided = float(info.get("time_since_collision", 1.0)) <= 1e-8
        collision_flags.append(collided)
        if collided:
            collisions += 1
        no_progress_time = float(info.get("no_progress_time", 0.0) or 0.0)
        stuck_time = float(info.get("stuck_time", 0.0) or 0.0)
        currently_stuck = no_progress_time >= 0.8 or stuck_time >= 0.5
        if currently_stuck:
            stuck_run_steps += 1
            max_stuck_run_steps = max(max_stuck_run_steps, stuck_run_steps)
        else:
            stuck_run_steps = 0
        if currently_stuck:
            stuck_before_recovery = True
        if int(info.get("recovery_mode", 0) or 0) != 0:
            effective_recovery_count += 1
            if recovery_trigger_step is None:
                recovery_trigger_step = len(path) - 1
        if stuck_before_recovery and recovery_trigger_step is not None and no_progress_time <= 0.25 and stuck_time <= 0.25:
            recovery_escape_success = 1
            recovery_resolve_steps.append((len(path) - 1) - recovery_trigger_step)
            stuck_before_recovery = False
            recovery_trigger_step = None
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
    if env.action_mode == "target_relative_full" and len(action_arr):
        full180_recovery_count = float(np.count_nonzero(np.abs(action_arr[:, 0]) > 0.67))
        metrics["recovery_action_count"] = full180_recovery_count
        metrics["recovery_action_rate"] = full180_recovery_count / max(float(len(action_arr)), 1.0)
    elif env.action_mode in INFO_RECOVERY_ACTION_MODES:
        metrics["recovery_action_count"] = float(effective_recovery_count)
        metrics["recovery_action_rate"] = effective_recovery_count / max(float(len(action_arr)), 1.0)
    row = {
        "seed": seed,
        "state_mode": state_mode,
        "action_mode": action_mode,
        "env_action_mode": env.action_mode,
        "steps": len(path_arr) - 1,
        "reward": total_reward,
        "final_distance": final_distance,
        "success": int(bool(info.get("is_success", final_distance <= env.target_radius))),
        "collisions": collisions,
        "recovery_steps_to_trigger": float(recovery_trigger_step if recovery_trigger_step is not None else len(path_arr) - 1),
        "recovery_escape_success": float(recovery_escape_success),
        "max_stuck_run_steps": float(max_stuck_run_steps),
        "avg_recovery_resolve_steps": float(np.mean(recovery_resolve_steps)) if recovery_resolve_steps else float(len(path_arr) - 1),
        "straight_mountain_hits": mountain_hits,
        "straight_tree_hits": tree_hits,
        **metrics,
    }
    trace = EpisodeTrace(env=env, path=path_arr, headings=heading_arr, total_reward=total_reward, row=row)
    return row, trace


def write_summary(path: Path, rows: list[dict[str, Any]], video_path: Path | None) -> None:
    def avg(key: str) -> float:
        return float(np.mean([float(row[key]) for row in rows])) if rows else 0.0

    summary = {
        "episodes": len(rows),
        "state_mode": rows[0].get("state_mode", "") if rows else "",
        "action_mode": rows[0].get("action_mode", "") if rows else "",
        "env_action_mode": rows[0].get("env_action_mode", "") if rows else "",
        "success_rate": avg("success"),
        "avg_steps": avg("steps"),
        "avg_final_distance": avg("final_distance"),
        "avg_path_efficiency": avg("path_efficiency"),
        "avg_abs_angle_error_deg": avg("avg_abs_angle_error_deg"),
        "avg_final_abs_angle_error_deg": avg("final_abs_angle_error_deg"),
        "avg_last5_abs_angle_error_deg": avg("last5_abs_angle_error_deg"),
        "avg_recovery_action_rate": avg("recovery_action_rate"),
        "avg_recovery_action_count": avg("recovery_action_count"),
        "avg_recovery_steps_to_trigger": avg("recovery_steps_to_trigger"),
        "recovery_escape_rate": avg("recovery_escape_success"),
        "avg_max_stuck_run_steps": avg("max_stuck_run_steps"),
        "avg_recovery_resolve_steps": avg("avg_recovery_resolve_steps"),
        "avg_collision_count": avg("collisions"),
        "avg_still_step_count": avg("still_step_count"),
        "avg_still_step_rate": avg("still_step_rate"),
        "avg_collision_still_step_count": avg("collision_still_step_count"),
        "avg_collision_still_step_rate": avg("collision_still_step_rate"),
        "avg_max_still_run": avg("max_still_run"),
        "video_path": str(video_path or ""),
    }
    write_csv(path, [summary])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--video-episodes", type=int, default=20)
    parser.add_argument("--seed-start", type=int, default=5_200_000)
    parser.add_argument("--max-scan", type=int, default=5000)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--action-mode", choices=["recovery", "full180", "macro", "macro_trigger", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v10b9", "v11"], default="v11")
    parser.add_argument("--state-mode", default=None)
    parser.add_argument("--env-overrides-file", default=None)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model = RecurrentPPO.load(args.model, device=args.device)
    state_mode = args.state_mode or default_state_mode(args.action_mode)
    env_overrides = load_env_overrides(args.env_overrides_file)
    seeds = find_pressure_seeds(
        args.seed_start,
        args.episodes,
        args.max_scan,
        args.action_mode,
        state_mode=state_mode,
        env_overrides=env_overrides,
    )
    rows: list[dict[str, Any]] = []
    videos: list[Path] = []
    for index, seed in enumerate(seeds):
        row, trace = evaluate_one(model, seed, args.action_mode, state_mode, env_overrides=env_overrides)
        rows.append(row)
        print(
            "pressure_eval index={index} seed={seed} success={success} steps={steps} "
            "angle={angle:.2f} recovery_rate={recovery:.3f} collisions={collisions}".format(
                index=index + 1,
                seed=seed,
                success=row["success"],
                steps=row["steps"],
                angle=float(row["avg_abs_angle_error_deg"]),
                recovery=float(row["recovery_action_rate"]),
                collisions=row["collisions"],
            ),
            flush=True,
        )
        if index < args.video_episodes:
            video = out_dir / f"pressure_eval_{index + 1:02d}.mp4"
            render_episode(trace, video, fps=args.fps)
            videos.append(video)
    write_csv(out_dir / "pressure_eval_results.csv", rows)
    concat_path: Path | None = None
    if videos:
        concat_path = out_dir / "pressure_eval_concat.mp4"
        concat_videos(videos, concat_path)
    write_summary(out_dir / "pressure_eval_summary.csv", rows, concat_path)


if __name__ == "__main__":
    main()
