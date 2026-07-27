"""Fast, render-free evaluation in the exact 2D GPU training environment."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))

from run_pipeline import GPUBlindNavEnvV11b, RecurrentActorCritic  # noqa: E402


def load_model(path: Path, device: str):
    state = torch.load(path, map_location=device)
    hidden_dim = state["lstm.weight_hh_l0"].shape[1]
    actor_width = state["actor_fc.0.weight"].shape[0]
    macro = "macro_head.weight" in state
    model = RecurrentActorCritic(
        state_dim=10, hidden_dim=hidden_dim, actor_width=actor_width, macro=macro
    ).to(device)
    model.load_state_dict(state)
    model.eval()
    return model, hidden_dim, macro


def percentile(values: torch.Tensor, q: float) -> float | None:
    if values.numel() == 0:
        return None
    return float(torch.quantile(values.float(), q).item())


def evaluate(args) -> dict:
    device = args.device
    torch.manual_seed(args.seed)
    model, hidden_dim, macro = load_model(Path(args.weights), device)
    env = GPUBlindNavEnvV11b(
        num_envs=args.episodes,
        device=device,
        auto_reset=False,
        collision_mode=args.collision_mode,
        action_mode=args.action_mode,
        macro=macro,
        hybrid_free_max_deg=args.hybrid_free_max_deg,
        obstacle_signal_mode=args.obstacle_signal_mode,
    )
    obs = env.reset(seed=args.seed)
    start_dist = torch.norm(env.target - env.pos, dim=1)
    min_dist = start_dist.clone()
    active = torch.ones(args.episodes, dtype=torch.bool, device=device)
    success = torch.zeros_like(active)
    success_steps = torch.zeros(args.episodes, device=device)
    path_length = torch.zeros(args.episodes, device=device)
    collision_count = torch.zeros(args.episodes, device=device)
    total_turn_deg = torch.zeros(args.episodes, device=device)
    turn_reversals = torch.zeros(args.episodes, device=device)
    previous_turn_sign = torch.zeros(args.episodes, device=device)
    steer_entropy_sum = torch.zeros(args.episodes, device=device)
    steer_confidence_sum = torch.zeros(args.episodes, device=device)
    policy_steps = torch.zeros(args.episodes, device=device)
    jump_actions = torch.zeros(args.episodes, device=device)
    jump_clearances = torch.zeros(args.episodes, device=device)
    commit_left = torch.zeros(args.episodes, dtype=torch.long, device=device)
    commit_bin = torch.full((args.episodes,), 3, dtype=torch.long, device=device)
    jump_cooldown = torch.zeros(args.episodes, dtype=torch.long, device=device)

    h = torch.zeros(1, args.episodes, hidden_dim, device=device)
    c = torch.zeros_like(h)
    done_mask = torch.zeros(args.episodes, device=device)

    for step in range(1, args.max_steps + 1):
        with torch.no_grad():
            lo, (h, c) = model.get_states(obs.unsqueeze(0), (h, c), done_mask.unsqueeze(0))
            features = model.actor_fc(lo.squeeze(0))
            logits = [
                model.steer_head(features),
                model.speed_head(features),
                model.jump_head(features),
            ]
            if macro:
                logits.append(model.macro_head(features))
            steer_probs = torch.softmax(logits[0], dim=-1)
            steer_entropy = torch.distributions.Categorical(logits=logits[0]).entropy()
            steer_entropy_sum += torch.where(active, steer_entropy, 0.0)
            steer_confidence_sum += torch.where(active, steer_probs.max(dim=-1).values, 0.0)
            policy_steps += active.float()
            if args.sample:
                actions = [
                    torch.distributions.Categorical(logits=head / args.temperature).sample()
                    for head in logits
                ]
            else:
                actions = [head.argmax(dim=-1) for head in logits]
            action = torch.stack(actions, dim=1)
            if args.jump_controller == "probe":
                action[:, 2] = ((obs[:, 9] > 0.5) & (jump_cooldown == 0)).long()

            locked = commit_left > 0
            action[locked, 0] = commit_bin[locked]
            commit_left = torch.clamp(commit_left - locked.long(), min=0)

        old_pos = env.pos.clone()
        old_heading = env.heading.clone()
        obs, _, _, reached = env.step(action)
        displacement = torch.norm(env.pos - old_pos, dim=1)
        heading_delta = torch.atan2(
            torch.sin(env.heading - old_heading), torch.cos(env.heading - old_heading)
        )
        turn_deg = heading_delta.abs() * (180.0 / math.pi)
        turn_sign = torch.sign(heading_delta) * (turn_deg > 2.0)
        collided = env.time_since_collision == 0.0
        if args.jump_controller == "probe":
            jump_failed = collided & (action[:, 2] == 1)
            jump_cooldown = torch.clamp(jump_cooldown - 1, min=0)
            jump_cooldown[jump_failed] = args.jump_cooldown
            jump_cooldown[~collided] = 0

        path_length += torch.where(active, displacement, 0.0)
        jump_actions += (active & (action[:, 2] == 1)).float()
        jump_clearances += (active & env.last_jump_pass_low).float()
        collision_count += (active & collided).float()
        total_turn_deg += torch.where(active, turn_deg, 0.0)
        reversal = active & (turn_sign != 0) & (previous_turn_sign != 0) & (
            turn_sign != previous_turn_sign
        )
        turn_reversals += reversal.float()
        previous_turn_sign = torch.where(turn_sign != 0, turn_sign, previous_turn_sign)
        min_dist = torch.minimum(min_dist, torch.norm(env.target - env.pos, dim=1))

        if args.commit > 0:
            start_escape = active & collided & (commit_left == 0)
            if start_escape.any():
                escape_bins = torch.tensor([0, 1, 5, 6], device=device)
                steer_logits = logits[0][start_escape][:, escape_bins]
                if args.sample:
                    pick = torch.distributions.Categorical(
                        logits=steer_logits / args.temperature
                    ).sample()
                else:
                    pick = steer_logits.argmax(dim=-1)
                commit_bin[start_escape] = escape_bins[pick]
                commit_left[start_escape] = args.commit
            commit_left[active & ~collided] = 0

        newly_reached = active & reached
        success[newly_reached] = True
        success_steps[newly_reached] = step
        active &= ~newly_reached
        if not active.any():
            break

    success_path_ratio = path_length[success] / torch.clamp(start_dist[success], min=1.0)
    result = {
        "weights": str(Path(args.weights).resolve()),
        "action_mode": args.action_mode,
        "policy": "sample" if args.sample else "argmax",
        "temperature": args.temperature if args.sample else None,
        "episodes": args.episodes,
        "max_steps": args.max_steps,
        "successes": int(success.sum().item()),
        "success_rate": float(success.float().mean().item()),
        "success_steps_median": percentile(success_steps[success], 0.5),
        "success_steps_p90": percentile(success_steps[success], 0.9),
        "success_path_ratio_median": percentile(success_path_ratio, 0.5),
        "success_path_ratio_p90": percentile(success_path_ratio, 0.9),
        "turn_deg_per_step": float(
            (total_turn_deg / torch.clamp(torch.where(success, success_steps, args.max_steps), min=1))
            .mean()
            .item()
        ),
        "turn_reversals_mean": float(turn_reversals.mean().item()),
        "collisions_mean": float(collision_count.mean().item()),
        "jump_actions_mean": float(jump_actions.mean().item()),
        "jump_clearances_mean": float(jump_clearances.mean().item()),
        "min_distance_median": percentile(min_dist, 0.5),
        "steer_entropy_mean": float(
            (steer_entropy_sum.sum() / torch.clamp(policy_steps.sum(), min=1.0)).item()
        ),
        "steer_max_probability_mean": float(
            (steer_confidence_sum.sum() / torch.clamp(policy_steps.sum(), min=1.0)).item()
        ),
    }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument(
        "--action-mode",
        choices=["target_relative", "heading_relative", "hybrid"],
        required=True,
    )
    parser.add_argument("--episodes", type=int, default=256)
    parser.add_argument("--max-steps", type=int, default=240)
    parser.add_argument("--seed", type=int, default=20260727)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--collision-mode", choices=["slide", "weak", "hard"], default="hard")
    parser.add_argument("--sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--commit", type=int, default=0)
    parser.add_argument("--jump-controller", choices=["policy", "probe"], default="probe")
    parser.add_argument("--jump-cooldown", type=int, default=8)
    parser.add_argument("--hybrid-free-max-deg", type=float, default=10.0)
    parser.add_argument(
        "--obstacle-signal-mode",
        choices=["proximity", "jump_probe"],
        default="jump_probe",
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    result = evaluate(args)
    if args.json:
        print(json.dumps(result, ensure_ascii=False))
    else:
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
