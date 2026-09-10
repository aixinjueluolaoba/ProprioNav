"""Interactive, no-time-limit MiniWorld test for the V4 blind navigator."""

from __future__ import annotations

import argparse
import math
import sys
import time

import gymnasium as gym
import miniworld  # noqa: F401
import numpy as np
import pyglet
import torch
from miniworld.envs.maze import MazeS3, MazeS3Fast
from pyglet.window import key

BASE = __import__("pathlib").Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))

from test_miniworld_v4 import DelayedPositionBelief, position_of, target_of  # noqa: E402
from run_pipeline import OBS_DIM, RecurrentActorCritic  # noqa: E402

TURN_OFFSETS = np.deg2rad([-45.0, -25.0, -10.0, 0.0, 10.0, 25.0, 45.0])
AGE_MS = 250.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--weights",
        default=str(BASE / "pipeline_out" / "policy_weights_v4_replay.pth"),
    )
    parser.add_argument("--env-id", choices=["MazeS3", "MazeS3Fast"], default="MazeS3")
    parser.add_argument("--seed", type=int, default=20260826)
    parser.add_argument("--age-ms", type=float, default=AGE_MS)
    parser.add_argument("--frame-delay", type=float, default=0.08)
    args = parser.parse_args()

    state = torch.load(args.weights, map_location="cpu")
    model = RecurrentActorCritic(state_dim=OBS_DIM, hidden_dim=96, actor_width=48, macro=False)
    model.load_state_dict(state)
    model.eval()

    env_class = MazeS3Fast if args.env_id == "MazeS3Fast" else MazeS3
    env = env_class(render_mode="human", max_episode_steps=math.inf)
    env.reset(seed=args.seed)
    env.render()
    world = env.unwrapped
    stop_requested = False

    @world.window.event
    def on_key_press(symbol, modifiers):
        nonlocal stop_requested
        if symbol in (key.ESCAPE, key.Q):
            stop_requested = True
            world.close()

    @world.window.event
    def on_close():
        nonlocal stop_requested
        stop_requested = True
        pyglet.app.exit()

    print("MiniWorld V4 实时测试已启动", flush=True)
    print("关闭窗口或按 Esc/Q 结束；没有动作数和时间上限。", flush=True)
    print(f"地图={args.env_id} seed={args.seed} 定位延迟={args.age_ms:.0f}ms", flush=True)

    position = position_of(env)
    target = target_of(env)
    belief = DelayedPositionBelief(position, target)
    observation = belief.update(position, target, args.age_ms, False, 0)
    hidden = torch.zeros(1, 1, 96)
    cell = torch.zeros_like(hidden)
    primitive_steps = 0
    decisions = 0
    collisions = 0

    try:
        while not stop_requested:
            decisions += 1
            with torch.no_grad():
                x = torch.from_numpy(observation).view(1, 1, OBS_DIM)
                output, (hidden, cell) = model.get_states(
                    x, (hidden, cell), torch.zeros(1, 1)
                )
                features = model.actor_fc(output.squeeze(0))
                steer_bin = int(model.steer_head(features).argmax(dim=-1).item())
                speed_bin = int(model.speed_head(features).argmax(dim=-1).item())
                jump_bin = int(model.jump_head(features).argmax(dim=-1).item())

            steer_bin, speed_bin, jump_bin = belief.calibration_action(
                steer_bin, speed_bin, jump_bin
            )
            turn_delta = float(TURN_OFFSETS[steer_bin])
            turn_count = int(round(abs(math.degrees(turn_delta)) / 15.0))
            turn_action = (
                world.actions.turn_left
                if turn_delta >= 0.0
                else world.actions.turn_right
            )
            actions = [turn_action] * turn_count
            actions.extend(
                [world.actions.move_forward] * (2 if speed_bin == 1 else 1)
            )
            if not actions:
                actions = [world.actions.move_forward]

            old_position = position.copy()
            terminated = False
            truncated = False
            for action in actions:
                _, reward, terminated, truncated, _ = env.step(int(action))
                primitive_steps += 1
                env.render()
                if args.frame_delay > 0:
                    time.sleep(args.frame_delay)
                if stop_requested or terminated or truncated:
                    break

            position = position_of(env)
            displacement = float(np.linalg.norm(position - old_position))
            expected = 100.0 * 0.15 * (2 if speed_bin == 1 else 1)
            collided = expected > 0.0 and displacement < max(expected * 0.2, 2.0)
            collisions += int(collided)
            belief.last_turn_delta = turn_delta
            observation = belief.update(
                position, target, args.age_ms, collided, jump_bin
            )

            if decisions % 20 == 0 or collided:
                distance = float(np.linalg.norm(target - position))
                print(
                    f"decision={decisions} primitive={primitive_steps} "
                    f"distance={distance:.1f} collision={int(collided)} "
                    f"confidence={belief.heading_confidence:.2f}",
                    flush=True,
                )
            if terminated:
                print(
                    f"到达目标，结束。decision={decisions} "
                    f"primitive={primitive_steps} collisions={collisions}",
                    flush=True,
                )
                break
            if truncated:
                print("环境被截断，但本脚本已配置无限步数。", flush=True)
                break
    finally:
        env.close()
        print("MiniWorld V4 实时测试已结束。", flush=True)


if __name__ == "__main__":
    main()
