from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import numpy as np
from sb3_contrib import RecurrentPPO

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark_state_dims_10k import load_env_overrides, make_state_env, predict_action, wrap_policy_env
from eval_target8_pressure import make_pressure_env


START = np.asarray([-520.0, -520.0], dtype=np.float32)
TARGET = np.asarray([520.0, 520.0], dtype=np.float32)


def make_diagonal_env(seed: int, state_mode: str, env_overrides):
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


def make_trace_env(seed: int, state_mode: str, env_overrides, scenario: str, action_mode: str):
    if scenario == "pressure":
        return make_pressure_env(seed, action_mode=action_mode, state_mode=state_mode, env_overrides=env_overrides)
    return make_diagonal_env(seed, state_mode, env_overrides=env_overrides)


def write_csv(path: Path, rows: list[dict[str, float | int]]) -> None:
    if not rows:
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--state-mode", default="observable12_target8_macro_library_v11rx_stage2")
    parser.add_argument("--scenario", choices=["diagonal", "pressure"], default="diagonal")
    parser.add_argument("--action-mode", default="v11")
    parser.add_argument("--env-overrides-file", default=None)
    parser.add_argument("--out", required=True)
    parser.add_argument("--summary-out", default=None)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    env_overrides = load_env_overrides(args.env_overrides_file)
    env = make_trace_env(
        args.seed,
        args.state_mode,
        env_overrides=env_overrides,
        scenario=args.scenario,
        action_mode=args.action_mode,
    )
    policy_env = wrap_policy_env(env, args.state_mode, env_overrides=env_overrides)
    model = RecurrentPPO.load(args.model, device=args.device)

    obs, _ = policy_env.reset(seed=args.seed)
    state = None
    episode_start = True
    terminated = False
    truncated = False
    rows: list[dict[str, float | int]] = []
    recovery_segments: list[dict[str, float | int]] = []
    current_segment: dict[str, float | int] | None = None

    while not (terminated or truncated):
        action, state = predict_action(model, obs, episode_start, state)
        action_arr = np.asarray(action, dtype=np.int64).reshape(-1)
        raw_action_arr = np.asarray(action, dtype=np.float32).reshape(-1)
        base = env
        delta = base.target - base.pos
        distance = float(np.linalg.norm(delta))
        target_angle = float(np.arctan2(delta[1], delta[0]))
        angle_error = float(np.arctan2(np.sin(target_angle - base.heading), np.cos(target_angle - base.heading)))
        obs, reward, terminated, truncated, info = policy_env.step(action)
        episode_start = False
        recovery_mode = int(info.get("recovery_mode", 0))
        macro_steps_left = int(info.get("macro_recovery_steps_left", 0))
        macro_total_steps = int(info.get("macro_recovery_total_steps", 0))
        policy_macro_bin = int(info.get("policy_macro_bin", action_arr[2] if action_arr.size > 2 else -1))
        effective_macro_bin = int(info.get("effective_macro_bin", -1))
        policy_macro_suppressed = int(bool(info.get("policy_macro_suppressed", False)))
        policy_macro_armed = int(bool(info.get("policy_macro_armed", False)))
        policy_macro_clear_rearm_triggered = int(bool(info.get("policy_macro_clear_rearm_triggered", False)))
        policy_macro_rearm_count = int(info.get("policy_macro_rearm_count", 0))
        policy_macro_rearm_pending = int(bool(info.get("policy_macro_rearm_pending", False)))
        stuck_macro_mid_trigger_count = int(info.get("stuck_macro_mid_trigger_count", 0))
        targeted_triggered = int(bool(info.get("targeted_stuck_macro_triggered", False)))
        generic_triggered = int(bool(info.get("stuck_macro_triggered", False)))
        suppressed_override_triggered = int(bool(info.get("suppressed_policy_macro_override_triggered", False)))
        suppressed_generic_redirect_triggered = int(
            bool(info.get("suppressed_policy_generic_redirect_triggered", False))
        )
        suppressed_policy_macro_passthrough_triggered = int(
            bool(info.get("suppressed_policy_macro_passthrough_triggered", False))
        )
        no_progress_time = float(info.get("no_progress_time", 0.0))
        stuck_time = float(info.get("stuck_time", 0.0))
        time_since_collision = float(info.get("time_since_collision", 0.0))
        if recovery_mode != 0:
            if current_segment is None:
                current_segment = {
                    "segment_index": len(recovery_segments) + 1,
                    "start_step": int(info.get("step_count", len(rows) + 1)),
                    "macro_bin": effective_macro_bin,
                    "policy_macro_bin_first": policy_macro_bin,
                    "duration_steps": 0,
                    "targeted_trigger_count": 0,
                    "generic_trigger_count": 0,
                    "collision_steps": 0,
                    "max_no_progress_time": 0.0,
                    "max_stuck_time": 0.0,
                }
            current_segment["duration_steps"] = int(current_segment["duration_steps"]) + 1
            current_segment["targeted_trigger_count"] = int(current_segment["targeted_trigger_count"]) + targeted_triggered
            current_segment["generic_trigger_count"] = int(current_segment["generic_trigger_count"]) + generic_triggered
            current_segment["collision_steps"] = int(current_segment["collision_steps"]) + int(time_since_collision <= 1e-8)
            current_segment["max_no_progress_time"] = max(float(current_segment["max_no_progress_time"]), no_progress_time)
            current_segment["max_stuck_time"] = max(float(current_segment["max_stuck_time"]), stuck_time)
        elif current_segment is not None:
            current_segment["end_step"] = int(info.get("step_count", len(rows) + 1)) - 1
            recovery_segments.append(current_segment)
            current_segment = None
        rows.append(
            {
                "step": int(info.get("step_count", len(rows) + 1)),
                "policy_angle_bin": int(action_arr[0]) if action_arr.size > 0 else -1,
                "policy_speed_bin": int(action_arr[1]) if action_arr.size > 1 else -1,
                "policy_macro_bin": policy_macro_bin,
                "policy_macro_raw": float(raw_action_arr[3]) if raw_action_arr.size > 3 else float("nan"),
                "policy_duration_raw": float(raw_action_arr[4]) if raw_action_arr.size > 4 else float("nan"),
                "effective_macro_bin": effective_macro_bin,
                "policy_macro_suppressed": policy_macro_suppressed,
                "policy_macro_armed": policy_macro_armed,
                "policy_macro_clear_rearm_triggered": policy_macro_clear_rearm_triggered,
                "policy_macro_rearm_count": policy_macro_rearm_count,
                "policy_macro_rearm_pending": policy_macro_rearm_pending,
                "stuck_macro_mid_trigger_count": stuck_macro_mid_trigger_count,
                "targeted_stuck_macro_triggered": targeted_triggered,
                "stuck_macro_triggered": generic_triggered,
                "suppressed_policy_macro_override_triggered": suppressed_override_triggered,
                "suppressed_policy_generic_redirect_triggered": suppressed_generic_redirect_triggered,
                "suppressed_policy_macro_passthrough_triggered": suppressed_policy_macro_passthrough_triggered,
                "targeted_stuck_macro_context": int(bool(info.get("targeted_stuck_macro_context", False))),
                "stuck_macro_context": int(bool(info.get("stuck_macro_context", False))),
                "reward": float(reward),
                "distance": float(info.get("distance", distance)),
                "angle_error": float(info.get("angle_error", angle_error)),
                "wrapper_angle_error": float(info.get("wrapper_angle_error", angle_error)),
                "wrapper_distance": float(info.get("wrapper_distance", distance)),
                "no_progress_time": no_progress_time,
                "stuck_time": stuck_time,
                "time_since_collision": time_since_collision,
                "wrapper_recent_collision_norm": float(info.get("wrapper_recent_collision_norm", 0.0)),
                "recovery_mode": recovery_mode,
                "macro_recovery_steps_left": macro_steps_left,
                "macro_recovery_total_steps": macro_total_steps,
                "success_flag": int(bool(info.get("is_success", False))),
                "x": float(base.pos[0]),
                "y": float(base.pos[1]),
                "heading": float(base.heading),
            }
        )

    if current_segment is not None:
        current_segment["end_step"] = len(rows)
        recovery_segments.append(current_segment)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_csv(out_path, rows)
    if args.summary_out:
        summary_rows = [
            {
                "seed": args.seed,
                "steps": len(rows),
                "success_flag": int(bool(rows[-1]["success_flag"])) if rows else 0,
                "collision_steps": sum(1 for row in rows if float(row["time_since_collision"]) <= 1e-8),
                "recovery_segments": len(recovery_segments),
                "recovery_steps": sum(int(row["duration_steps"]) for row in recovery_segments),
                "targeted_trigger_steps": sum(int(row["targeted_trigger_count"]) for row in recovery_segments),
                "generic_trigger_steps": sum(int(row["generic_trigger_count"]) for row in recovery_segments),
                "suppressed_override_steps": sum(
                    int(row["suppressed_policy_macro_override_triggered"]) for row in rows
                ),
                "suppressed_generic_redirect_steps": sum(
                    int(row["suppressed_policy_generic_redirect_triggered"]) for row in rows
                ),
                "suppressed_policy_macro_passthrough_steps": sum(
                    int(row["suppressed_policy_macro_passthrough_triggered"]) for row in rows
                ),
                "max_segment_duration": max((int(row["duration_steps"]) for row in recovery_segments), default=0),
                "policy_macro_suppressed_steps": sum(1 for row in rows if int(row["policy_macro_suppressed"]) != 0),
                "policy_macro_clear_rearm_steps": sum(
                    1 for row in rows if int(row["policy_macro_clear_rearm_triggered"]) != 0
                ),
                "max_mid_trigger_count": max((int(row["stuck_macro_mid_trigger_count"]) for row in rows), default=0),
            }
        ]
        summary_path = Path(args.summary_out)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        write_csv(summary_path, summary_rows)
        segment_path = summary_path.with_name(summary_path.stem + "_segments.csv")
        write_csv(segment_path, recovery_segments)
    print(
        "trace_done seed={seed} steps={steps} success={success} collisions={collisions}".format(
            seed=args.seed,
            steps=len(rows),
            success=int(bool(rows[-1]["success_flag"])) if rows else 0,
            collisions=sum(1 for row in rows if float(row["time_since_collision"]) <= 1e-8),
        ),
        flush=True,
    )
    policy_env.close()


if __name__ == "__main__":
    main()
