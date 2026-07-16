from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from sb3_contrib import RecurrentPPO

from benchmark_state_dims_10k import load_env_overrides, make_state_env, predict_action, wrap_policy_env
from eval_target8_bad_seeds import make_diagonal_env, parse_seed_list
from eval_target8_pressure import find_pressure_seeds, make_pressure_env


DEFAULT_STATE_MODE = "observable12_target8_macro_library_v11re_stage2"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed-list", default=None)
    parser.add_argument("--seed-file", default=None)
    parser.add_argument("--seed-start", type=int, default=None)
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--pressure-scan", type=int, default=10_000)
    parser.add_argument("--state-mode", default=DEFAULT_STATE_MODE)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--env-overrides-file", default=None)
    parser.add_argument("--scenario", choices=["diagonal", "pressure", "normal"], default="diagonal")
    parser.add_argument(
        "--action-label-source",
        choices=["policy", "effective_macro", "effective_macro_speed"],
        default="policy",
    )
    args = parser.parse_args()

    model = RecurrentPPO.load(args.model, device=args.device)
    env_overrides = load_env_overrides(args.env_overrides_file)
    if args.seed_start is not None:
        if args.episodes is None or int(args.episodes) <= 0:
            raise ValueError("--episodes must be positive when --seed-start is provided")
        if args.scenario == "pressure":
            seeds = find_pressure_seeds(
                int(args.seed_start),
                int(args.episodes),
                int(args.pressure_scan),
                "v11",
                state_mode=args.state_mode,
                env_overrides=env_overrides,
            )
        else:
            seeds = [int(args.seed_start) + index for index in range(int(args.episodes))]
    else:
        seeds = parse_seed_list(args.seed_list, args.seed_file)

    obs_rows: list[np.ndarray] = []
    action_rows: list[np.ndarray] = []
    policy_action_rows: list[np.ndarray] = []
    effective_action_rows: list[np.ndarray] = []
    episode_start_rows: list[bool] = []
    seed_rows: list[int] = []
    step_rows: list[int] = []
    success_rows: list[bool] = []
    no_progress_rows: list[float] = []
    stuck_time_rows: list[float] = []
    collision_rows: list[bool] = []
    recovery_mode_rows: list[int] = []

    for seed in seeds:
        if args.scenario == "normal":
            env = make_state_env(seed, args.state_mode, env_overrides=env_overrides)
        elif args.scenario == "pressure":
            env = make_pressure_env(seed, action_mode="v11", state_mode=args.state_mode, env_overrides=env_overrides)
        else:
            env = make_diagonal_env(seed, args.state_mode, env_overrides=env_overrides)
        policy_env = wrap_policy_env(env, args.state_mode, env_overrides=env_overrides)
        obs, _ = policy_env.reset(seed=seed)
        state = None
        episode_start = True
        terminated = False
        truncated = False
        step_index = 0
        info: dict[str, object] = {}
        while not (terminated or truncated):
            action, state = predict_action(model, obs, episode_start, state)
            action_arr = np.asarray(action, dtype=np.int64).reshape(-1)
            obs_rows.append(np.asarray(obs, dtype=np.float32).reshape(-1))
            episode_start_rows.append(bool(episode_start))
            seed_rows.append(int(seed))
            step_rows.append(int(step_index))
            no_progress_rows.append(float(getattr(env, "no_progress_time", 0.0)))
            stuck_time_rows.append(float(getattr(env, "stuck_time", 0.0)))
            collision_rows.append(float(getattr(env, "time_since_collision", 1.0)) <= 1e-8)
            recovery_mode_rows.append(int(getattr(env, "recovery_mode", 0)))

            obs, _reward, terminated, truncated, info = policy_env.step(action)
            policy_action = action_arr.copy()
            effective_action = action_arr.copy()
            if effective_action.shape == (3,):
                effective_macro_bin = info.get("effective_macro_bin", None)
                if effective_macro_bin is not None:
                    effective_action[2] = int(np.clip(int(effective_macro_bin), 0, 7))
                if args.action_label_source == "effective_macro_speed" and int(effective_action[2]) != 0:
                    effective_action[1] = 1
            if args.action_label_source == "policy":
                action_rows.append(policy_action)
            else:
                action_rows.append(effective_action.copy())
            policy_action_rows.append(policy_action)
            effective_action_rows.append(effective_action)
            episode_start = False
            step_index += 1

        success = bool(info.get("is_success", False))
        success_rows.extend([success] * step_index)
        policy_env.close()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        obs=np.asarray(obs_rows, dtype=np.float32),
        actions=np.asarray(action_rows, dtype=np.int64),
        policy_actions=np.asarray(policy_action_rows, dtype=np.int64),
        effective_actions=np.asarray(effective_action_rows, dtype=np.int64),
        episode_starts=np.asarray(episode_start_rows, dtype=bool),
        seeds=np.asarray(seed_rows, dtype=np.int64),
        step_index=np.asarray(step_rows, dtype=np.int32),
        teacher_success=np.asarray(success_rows, dtype=bool),
        no_progress_time=np.asarray(no_progress_rows, dtype=np.float32),
        stuck_time=np.asarray(stuck_time_rows, dtype=np.float32),
        collided=np.asarray(collision_rows, dtype=bool),
        recovery_mode=np.asarray(recovery_mode_rows, dtype=np.int8),
    )
    print(f"saved_dataset={out_path}", flush=True)
    print(f"samples={len(obs_rows)}", flush=True)
    print(f"episodes={len(seeds)}", flush=True)
    print(f"scenario={args.scenario}", flush=True)
    print(f"action_label_source={args.action_label_source}", flush=True)
    print(f"seed_count={len(seeds)}", flush=True)
    if seeds:
        print(f"seed_preview={','.join(str(seed) for seed in seeds[:8])}", flush=True)


if __name__ == "__main__":
    main()
