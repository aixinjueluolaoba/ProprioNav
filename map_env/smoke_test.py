"""Sanity checks + random-rollout visualization for the maze environment."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from map_env.maze_env import GPUImageMazeNavEnv
from map_env.render_maze import save_rollout_png


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid", default=str(Path(__file__).resolve().parent / "maze_grid.npz"))
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument(
        "--output",
        default=str(Path(__file__).resolve().parent / "random_rollout.png"),
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    env = GPUImageMazeNavEnv(
        num_envs=1,
        grid_path=args.grid,
        device=device,
        auto_reset=False,
        collision_mode="hard",
    )
    env.reset(seed=7)

    path = [env.pos[0].cpu().numpy().copy()]
    headings = [float(env.heading[0].item())]
    penetrations = 0
    for _ in range(args.steps):
        action = torch.tensor(
            [[int(torch.randint(0, 7, (1,)).item()), int(torch.randint(0, 2, (1,)).item()), 0]],
            dtype=torch.long,
            device=device,
        )
        action = env.apply_calibration(action)
        _, _, done, _ = env.step(action)
        path.append(env.pos[0].cpu().numpy().copy())
        headings.append(float(env.heading[0].item()))
        clearance = env._bilinear_clearance(env.pos)[0].item()
        if clearance < env.agent_radius - 1e-3:
            penetrations += 1
        if bool(done[0].item()):
            break

    free = np.load(args.grid)["free"]
    out = save_rollout_png(
        free,
        np.asarray(path),
        env.target[0].cpu().numpy(),
        args.output,
        headings=np.asarray(headings),
        title="random rollout (hard collision)",
    )
    dist = float(np.linalg.norm(env.target[0].cpu().numpy() - path[-1]))
    print(f"steps={len(path) - 1} wall_penetrations={penetrations} final_dist={dist:.1f}")
    print(f"rollout image: {out.resolve()}")


if __name__ == "__main__":
    main()
