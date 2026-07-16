from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any

import numpy as np
from stable_baselines3.common.base_class import BaseAlgorithm

from benchmark_state_dims_10k import (
    EXPERIMENTS,
    Experiment,
    NO_POLICY_JUMP_STATE_MODES,
    load_env_overrides,
    load_model,
    make_env,
    make_state_env,
    predict_action,
    wrap_policy_env,
)
from render_target8_eval import EpisodeTrace, concat_videos, render_episode


INFO_RECOVERY_ACTION_MODES = {
    "target_relative_macro_recovery",
    "target_relative_macro_trigger_recovery",
    "heading_relative_two_phase_macro_recovery",
    "heading_relative_macro_library_recovery",
}


def find_experiment(name: str) -> Experiment:
    for exp in EXPERIMENTS:
        if exp.name == name:
            return exp
    raise ValueError(f"Unknown experiment: {name}")


def angle_error_to_target(point: np.ndarray, target: np.ndarray, heading: float) -> float:
    delta = target - point
    target_angle = math.atan2(float(delta[1]), float(delta[0]))
    return float(math.atan2(math.sin(target_angle - heading), math.cos(target_angle - heading)))


def path_metrics(path: np.ndarray, headings: np.ndarray, actions: np.ndarray, target: np.ndarray) -> dict[str, float]:
    if len(path) < 2:
        return {
            "path_length": 0.0,
            "straight_distance": 0.0,
            "path_efficiency": 0.0,
            "avg_abs_angle_error_deg": 0.0,
            "final_abs_angle_error_deg": 0.0,
            "last5_abs_angle_error_deg": 0.0,
            "max_abs_angle_error_deg": 0.0,
            "turn_change_deg": 0.0,
            "action_flip_count": 0.0,
            "avg_abs_action_angle": 0.0,
            "recovery_action_count": 0.0,
            "recovery_action_rate": 0.0,
        }
    deltas = np.diff(path, axis=0)
    step_lengths = np.linalg.norm(deltas, axis=1)
    path_length = float(step_lengths.sum())
    straight_distance = float(np.linalg.norm(path[-1] - path[0]))
    path_efficiency = straight_distance / max(path_length, 1e-6)
    angle_errors = np.asarray([angle_error_to_target(path[i], target, float(headings[i])) for i in range(len(path))], dtype=np.float32)
    abs_angle_errors = np.abs(angle_errors)
    heading_deltas = np.diff(np.unwrap(headings.astype(np.float64))) if len(headings) >= 2 else np.asarray([], dtype=np.float64)
    action_angles = actions[:, 0] if len(actions) else np.asarray([], dtype=np.float32)
    if len(actions) and actions.shape[1] >= 4:
        recovery_actions = actions[:, 3]
        recovery_count = float(np.count_nonzero((recovery_actions < 0.0) | (recovery_actions >= 0.5)))
    else:
        recovery_count = 0.0
    if len(action_angles) >= 2:
        signs = np.sign(action_angles)
        active = np.abs(action_angles) > 0.06
        prev_active = active[:-1] & active[1:]
        flips = (signs[:-1] * signs[1:] < 0.0) & prev_active
        action_flip_count = float(np.count_nonzero(flips))
    else:
        action_flip_count = 0.0
    return {
        "path_length": path_length,
        "straight_distance": straight_distance,
        "path_efficiency": float(np.clip(path_efficiency, 0.0, 1.2)),
        "avg_abs_angle_error_deg": float(np.degrees(np.mean(abs_angle_errors))),
        "final_abs_angle_error_deg": float(np.degrees(abs_angle_errors[-1])),
        "last5_abs_angle_error_deg": float(np.degrees(np.mean(abs_angle_errors[-5:]))),
        "max_abs_angle_error_deg": float(np.degrees(np.max(abs_angle_errors))),
        "turn_change_deg": float(np.degrees(np.sum(np.abs(heading_deltas)))) if len(heading_deltas) else 0.0,
        "action_flip_count": action_flip_count,
        "avg_abs_action_angle": float(np.mean(np.abs(action_angles))) if len(action_angles) else 0.0,
        "recovery_action_count": recovery_count,
        "recovery_action_rate": recovery_count / max(float(len(actions)), 1.0),
    }


def stall_metrics(
    path: np.ndarray,
    collision_flags: np.ndarray,
    *,
    still_distance_epsilon: float = 1e-4,
) -> dict[str, float]:
    if len(path) < 2:
        return {
            "still_step_count": 0.0,
            "still_step_rate": 0.0,
            "collision_still_step_count": 0.0,
            "collision_still_step_rate": 0.0,
            "max_still_run": 0.0,
        }
    deltas = np.diff(path, axis=0)
    step_lengths = np.linalg.norm(deltas, axis=1)
    still_mask = step_lengths <= still_distance_epsilon
    collision_mask = np.asarray(collision_flags, dtype=bool)
    if len(collision_mask) != len(still_mask):
        raise ValueError("collision_flags length must match step count")
    collision_still_mask = still_mask & collision_mask
    max_still_run = 0
    current_run = 0
    for is_still in still_mask:
        if is_still:
            current_run += 1
            max_still_run = max(max_still_run, current_run)
        else:
            current_run = 0
    step_count = max(float(len(still_mask)), 1.0)
    still_step_count = float(np.count_nonzero(still_mask))
    collision_still_step_count = float(np.count_nonzero(collision_still_mask))
    return {
        "still_step_count": still_step_count,
        "still_step_rate": still_step_count / step_count,
        "collision_still_step_count": collision_still_step_count,
        "collision_still_step_rate": collision_still_step_count / step_count,
        "max_still_run": float(max_still_run),
    }


def evaluate_one(
    model: BaseAlgorithm,
    exp: Experiment,
    out_dir: Path,
    *,
    seed_start: int,
    episodes: int,
    video_episodes: int,
    fps: int,
    env_overrides: dict[str, dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], Path | None]:
    rows: list[dict[str, Any]] = []
    videos: list[Path] = []
    for index in range(episodes):
        seed = seed_start + index
        env = make_state_env(seed, exp.state_mode, env_overrides=env_overrides)
        policy_env = wrap_policy_env(env, exp.state_mode, env_overrides=env_overrides)
        obs, _ = policy_env.reset(seed=seed)
        path = [env.pos.copy()]
        headings = [env.heading]
        actions: list[np.ndarray] = []
        total_reward = 0.0
        terminated = False
        truncated = False
        state = None
        episode_start = True
        info: dict[str, Any] = {}
        collision_count = 0
        collision_flags: list[bool] = []
        jump_attempt_count = 0
        jump_trigger_count = 0
        effective_recovery_count = 0
        while not (terminated or truncated):
            action, state = predict_action(model, obs, episode_start, state)
            action_arr = np.asarray(action, dtype=np.float32).reshape(-1)
            if exp.state_mode in NO_POLICY_JUMP_STATE_MODES:
                jump_attempt = 0
            else:
                jump_attempt = int(float(action_arr[2]) > env.jump_threshold)
            jump_trigger = int(jump_attempt and env.jump_cooldown <= 0.0)
            obs, reward, terminated, truncated, info = policy_env.step(action)
            episode_start = False
            collided = float(info.get("time_since_collision", 1.0)) <= 1e-8
            collision_flags.append(collided)
            if collided:
                collision_count += 1
            if int(info.get("recovery_mode", 0) or 0) != 0:
                effective_recovery_count += 1
            jump_attempt_count += jump_attempt
            jump_trigger_count += jump_trigger
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
        if env.action_mode in INFO_RECOVERY_ACTION_MODES:
            metrics["recovery_action_count"] = float(effective_recovery_count)
            metrics["recovery_action_rate"] = effective_recovery_count / max(float(len(action_arr)), 1.0)
        row = {
            "episode": index + 1,
            "seed": seed,
            "state_mode": exp.state_mode,
            "action_mode": env.action_mode,
            "steps": len(path_arr) - 1,
            "reward": total_reward,
            "final_distance": final_distance,
            "success": int(bool(info.get("is_success", final_distance <= env.target_radius))),
            "collisions": collision_count,
            "jump_attempts": jump_attempt_count,
            "jump_triggers": jump_trigger_count,
            **metrics,
        }
        rows.append(row)
        if index < video_episodes:
            video = out_dir / f"target8_eval_{index + 1:02d}.mp4"
            trace = EpisodeTrace(env=env, path=path_arr, headings=heading_arr, total_reward=total_reward, row=row)
            render_episode(trace, video, fps=fps)
            videos.append(video)
        policy_env.close()
    concat_path: Path | None = None
    if videos:
        concat_path = out_dir / "target8_eval_concat.mp4"
        concat_videos(videos, concat_path)
    return rows, concat_path


def summarize(exp_name: str, rows: list[dict[str, Any]], param_trainable: int, train_seconds: float, model_path: Path, video_path: Path | None) -> dict[str, Any]:
    def avg(key: str) -> float:
        return float(np.mean([float(row[key]) for row in rows])) if rows else 0.0

    success_rate = avg("success")
    path_efficiency = avg("path_efficiency")
    avg_angle = avg("avg_abs_angle_error_deg")
    avg_flips = avg("action_flip_count")
    avg_steps = avg("steps")
    score = success_rate * 100.0 + path_efficiency * 35.0 - avg_angle * 0.45 - avg_flips * 1.8 - avg_steps * 0.04 - avg("collisions") * 1.5
    return {
        "name": exp_name,
        "param_trainable": param_trainable,
        "train_seconds": train_seconds,
        "success_rate": success_rate,
        "avg_steps": avg_steps,
        "avg_final_distance": avg("final_distance"),
        "avg_reward": avg("reward"),
        "avg_path_efficiency": path_efficiency,
        "avg_abs_angle_error_deg": avg_angle,
        "avg_final_abs_angle_error_deg": avg("final_abs_angle_error_deg"),
        "avg_last5_abs_angle_error_deg": avg("last5_abs_angle_error_deg"),
        "max_abs_angle_error_deg": avg("max_abs_angle_error_deg"),
        "avg_turn_change_deg": avg("turn_change_deg"),
        "avg_action_flip_count": avg_flips,
        "avg_recovery_action_count": avg("recovery_action_count"),
        "avg_recovery_action_rate": avg("recovery_action_rate"),
        "avg_collision_count": avg("collisions"),
        "avg_still_step_count": avg("still_step_count"),
        "avg_still_step_rate": avg("still_step_rate"),
        "avg_collision_still_step_count": avg("collision_still_step_count"),
        "avg_collision_still_step_rate": avg("collision_still_step_rate"),
        "avg_max_still_run": avg("max_still_run"),
        "avg_jump_attempts": avg("jump_attempts"),
        "score": score,
        "model_path": str(model_path),
        "video_path": str(video_path or ""),
    }


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


def write_summary_md(path: Path, rows: list[dict[str, Any]], best_name: str) -> None:
    lines = [
        "| model | params | train_s | success | avg_steps | path_eff | avg_angle_deg | flips | recovery_rate | score | recommended | video |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in sorted(rows, key=lambda item: float(item["score"]), reverse=True):
        lines.append(
            "| {name} | {params} | {train_s:.1f} | {success:.2f} | {steps:.1f} | {eff:.3f} | {angle:.1f} | {flips:.1f} | {recovery:.3f} | {score:.2f} | {rec} | {video} |".format(
                name=row["name"],
                params=int(row["param_trainable"]),
                train_s=float(row["train_seconds"]),
                success=float(row["success_rate"]),
                steps=float(row["avg_steps"]),
                eff=float(row["avg_path_efficiency"]),
                angle=float(row["avg_abs_angle_error_deg"]),
                flips=float(row["avg_action_flip_count"]),
                recovery=float(row.get("avg_recovery_action_rate", 0.0) or 0.0),
                score=float(row["score"]),
                rec="yes" if row["name"] == best_name else "",
                video=row.get("video_path", ""),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def find_model_path(train_dirs: list[Path], exp_name: str) -> Path:
    for root in train_dirs:
        candidates = [
            root / exp_name / f"{exp_name}.zip",
            root / exp_name / exp_name / f"{exp_name}.zip",
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
    raise FileNotFoundError(f"model for {exp_name} under {', '.join(str(path) for path in train_dirs)}")


def load_train_rows(train_dirs: list[Path]) -> dict[str, dict[str, str]]:
    rows: dict[str, dict[str, str]] = {}
    for root in train_dirs:
        for csv_path in [root / "state_dim_results.csv", *root.glob("*/state_dim_results.csv")]:
            if not csv_path.exists():
                continue
            with csv_path.open("r", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    if row.get("name"):
                        rows[row["name"]] = row
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-output-dir", default=None)
    parser.add_argument("--train-output-dirs", nargs="*", default=None)
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--experiments", nargs="+", required=True)
    parser.add_argument("--seed-start", type=int, default=1_800_000)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--video-episodes", type=int, default=20)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--env-overrides-file", default=None)
    args = parser.parse_args()

    train_dirs = [Path(path) for path in (args.train_output_dirs or [])]
    if args.train_output_dir:
        train_dirs.append(Path(args.train_output_dir))
    if not train_dirs:
        raise ValueError("Provide --train-output-dir or --train-output-dirs")
    env_overrides_file = args.env_overrides_file
    if env_overrides_file is None:
        for root in train_dirs:
            candidate = root / "env_overrides.json"
            if candidate.exists():
                env_overrides_file = str(candidate)
                break
    env_overrides = load_env_overrides(env_overrides_file)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_rows = load_train_rows(train_dirs)

    summaries: list[dict[str, Any]] = []
    for exp_name in args.experiments:
        exp = find_experiment(exp_name)
        model_path = Path(args.model_path) if args.model_path else find_model_path(train_dirs, exp.name)
        env = __import__("stable_baselines3.common.vec_env", fromlist=["SubprocVecEnv"]).SubprocVecEnv(
            [make_env(args.seed_start, 0, exp.state_mode, env_overrides=env_overrides)],
            start_method="fork",
        )
        model = load_model(exp, env, model_path, args.device)
        param_trainable = sum(p.numel() for p in model.policy.parameters() if p.requires_grad)
        env.close()

        exp_out = out_dir / exp.name
        exp_out.mkdir(parents=True, exist_ok=True)
        rows, video_path = evaluate_one(
            model,
            exp,
            exp_out,
            seed_start=args.seed_start,
            episodes=args.episodes,
            video_episodes=args.video_episodes,
            fps=args.fps,
            env_overrides=env_overrides,
        )
        write_csv(exp_out / "eval_results.csv", rows)
        train_seconds = float(train_rows.get(exp.name, {}).get("train_seconds", 0.0) or 0.0)
        summaries.append(summarize(exp.name, rows, param_trainable, train_seconds, model_path, video_path))
        print(f"evaluated {exp.name}", flush=True)

    best = max(summaries, key=lambda row: float(row["score"]))
    write_csv(out_dir / "target8_sweep_summary.csv", summaries)
    write_summary_md(out_dir / "target8_sweep_summary.md", summaries, str(best["name"]))
    (out_dir / "best_model.txt").write_text(str(best["model_path"]) + "\n", encoding="utf-8")
    print(f"best_model={best['name']} score={float(best['score']):.3f}", flush=True)


if __name__ == "__main__":
    main()
