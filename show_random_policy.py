from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from blind_nav_rl import BlindNavEnv


def main() -> None:
    env = BlindNavEnv(seed=11, tree_count=45, mountain_count=8)
    obs, _info = env.reset(seed=11)

    path = [env.pos.copy()]
    total_reward = 0.0
    for _ in range(350):
        del obs
        action = env.action_space.sample()
        obs, reward, terminated, truncated, _info = env.step(action)
        path.append(env.pos.copy())
        total_reward += reward
        if terminated or truncated:
            break

    points = np.asarray(path)
    fig, ax = plt.subplots(figsize=(8, 8), dpi=140)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(f"Random policy, steps={len(path) - 1}, reward={total_reward:.2f}")

    for obstacle in env.obstacles:
        if hasattr(obstacle, "vertices"):
            ax.add_patch(
                plt.Polygon(
                    obstacle.vertices,
                    closed=True,
                    color="#7a7468",
                    alpha=0.38,
                    linewidth=1.0,
                    fill=True,
                )
            )
        else:
            ax.add_patch(
                plt.Circle(
                    (obstacle.x, obstacle.y),
                    obstacle.radius,
                    color="#3d7f3d",
                    alpha=0.42,
                    linewidth=1.0,
                    fill=True,
                )
            )

    ax.plot(points[:, 0], points[:, 1], color="#1f77b4", linewidth=1.4, label="random path")
    ax.scatter(points[0, 0], points[0, 1], color="#2ca02c", s=70, label="start", zorder=3)
    ax.scatter(env.target[0], env.target[1], color="#d62728", s=90, marker="*", label="target", zorder=3)
    ax.scatter(points[-1, 0], points[-1, 1], color="#111111", s=45, label="end", zorder=3)

    arrow_len = 80.0
    ax.annotate(
        "",
        xy=(points[-1, 0] + arrow_len * np.cos(env.heading), points[-1, 1] + arrow_len * np.sin(env.heading)),
        xytext=(points[-1, 0], points[-1, 1]),
        arrowprops=dict(arrowstyle="->", color="#ff7f0e", lw=2.0),
        zorder=4,
    )
    ax.add_patch(
        plt.Circle(
            (env.target[0], env.target[1]),
            env.target_radius,
            color="#d62728",
            fill=False,
            linestyle="--",
            linewidth=1.5,
        )
    )

    margin = 240.0
    min_x = min(points[:, 0].min(), env.target[0]) - margin
    max_x = max(points[:, 0].max(), env.target[0]) + margin
    min_y = min(points[:, 1].min(), env.target[1]) - margin
    max_y = max(points[:, 1].max(), env.target[1]) + margin
    ax.set_xlim(min_x, max_x)
    ax.set_ylim(min_y, max_y)
    ax.grid(True, color="#dddddd", linewidth=0.7)
    ax.legend(loc="upper right")

    output = Path("random_policy_demo.png")
    fig.tight_layout()
    fig.savefig(output)
    print(f"saved={output.resolve()}")
    print(env.render())


if __name__ == "__main__":
    main()
