"""Render a trained ProprioNav policy navigating the image-derived maze.

Example:
    python -m map_env.render_policy_rollout \
        --weights pipeline_out/maze_smoke.pth \
        --grid map_env/maze_grid.npz \
        --steps 200 \
        --output map_env/maze_policy_rollout.mp4
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from run_pipeline import OBS_DIM, RecurrentActorCritic, RenderShim
from map_env.maze_env import GPUImageMazeNavEnv
from map_env.render_maze import save_rollout_png


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--grid", default=str(Path(__file__).resolve().parent / "maze_grid.npz"))
    parser.add_argument("--steps", type=int, default=240)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sample", action="store_true", help="sample actions instead of argmax")
    parser.add_argument("--output", default=str(Path(__file__).resolve().parent / "maze_policy_rollout.mp4"))
    parser.add_argument("--png", default=None, help="optional still image output path")
    args = parser.parse_args()

    device = "cpu"
    agent = RecurrentActorCritic(state_dim=OBS_DIM, hidden_dim=96, actor_width=48, macro=False)
    agent.load_state_dict(torch.load(args.weights, map_location="cpu"))
    agent.eval()

    env = GPUImageMazeNavEnv(
        num_envs=1,
        grid_path=args.grid,
        device=device,
        auto_reset=False,
        collision_mode="hard",
    )
    obs = env.reset(seed=args.seed)
    shim = RenderShim(env)

    h = torch.zeros(1, 1, agent.lstm.hidden_size, device=device)
    c = torch.zeros_like(h)
    path = [env.pos[0].cpu().numpy().copy()]
    headings = [float(env.heading[0].item())]
    total_reward = 0.0

    for _ in range(args.steps):
        with torch.inference_mode():
            out, (h, c) = agent.get_states(obs.unsqueeze(0), (h, c), torch.zeros(1, 1, device=device))
            features = agent.actor_fc(out.squeeze(0))
            heads = [agent.steer_head(features), agent.speed_head(features), agent.jump_head(features)]
            if args.sample:
                action = torch.stack(
                    [torch.distributions.Categorical(logits=logit).sample() for logit in heads], dim=1
                )
            else:
                action = torch.stack([logit.argmax(dim=-1) for logit in heads], dim=1)
        action = env.apply_calibration(action)
        obs, reward, dones, _ = env.step(action)
        total_reward += float(reward[0].item())
        path.append(env.pos[0].cpu().numpy().copy())
        headings.append(float(env.heading[0].item()))
        if bool(dones[0].item()):
            break

    path = np.asarray(path)
    free = env.grid_free.cpu().numpy()
    png = args.png or str(Path(args.output).with_suffix(".png"))
    save_rollout_png(
        free,
        path,
        env.target[0].cpu().numpy(),
        png,
        headings=np.asarray(headings),
        scale=2,
        title=f"policy rollout steps={len(path) - 1} reward={total_reward:.1f}",
    )

    from render_eval10_concat import render_episode_video

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    render_episode_video(shim, path, total_reward, output, fps=20, headings=np.asarray(headings))

    dist = float(np.linalg.norm(env.target[0].cpu().numpy() - path[-1]))
    print(f"steps={len(path) - 1} reward={total_reward:.3f} final_dist={dist:.2f}")
    print(f"png: {Path(png).resolve()}")
    print(f"mp4: {output.resolve()}")


if __name__ == "__main__":
    main()
