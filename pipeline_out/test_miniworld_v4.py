"""Run the V4 policy in a MiniWorld 3D maze without exposing agent heading."""

from __future__ import annotations

import argparse
import math
import subprocess
import sys
from pathlib import Path

import gymnasium as gym
import imageio.v2 as imageio
import miniworld  # noqa: F401  # registers MiniWorld environments
import numpy as np
import torch
from miniworld.envs.maze import MazeS3, MazeS3Fast

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
WORLD_SIZE = 2250.0
DT = 0.30
AGE_MS = 250.0
MAX_TURN = math.pi / 4.0
TURN_OFFSETS = np.deg2rad([-45.0, -25.0, -10.0, 0.0, 10.0, 25.0, 45.0])
MAX_REASONABLE_SPEED = 500.0


def normalize_angle(value: float) -> float:
    return math.atan2(math.sin(value), math.cos(value))


def freshness_scale(age_ms: float) -> float:
    if age_ms <= 200.0:
        return 1.0
    if age_ms <= 500.0:
        return 1.0 - 0.5 * ((age_ms - 200.0) / 300.0)
    if age_ms <= 1000.0:
        return 0.5 * ((1000.0 - age_ms) / 500.0)
    return 0.0


class DelayedPositionBelief:
    """Scalar counterpart of the V4 delayed-position state estimator."""

    def __init__(self, position: np.ndarray, target: np.ndarray):
        self.history = [position.copy()]
        self.observed_pos = position.copy()
        self.predicted_pos = position.copy()
        self.target = target.copy()
        self.velocity = np.zeros(2, dtype=np.float32)
        self.estimated_heading = 0.0
        self.heading_valid = False
        self.heading_confidence = 0.0
        self.calibration_samples = 0
        self.no_valid_steps = 0
        self.probe_phase = 0
        self.last_turn_delta = 0.0
        self.prev_distance = float(np.linalg.norm(target - position))
        self.time_since_collision = 10.0
        self.stuck_time = 0.0
        self.no_progress_time = 0.0
        self.prev_jump = 0
        self.jump_probe = 0.0
        self.last_expected_move = 0.0
        self.initialized = False

    def _delayed_position(self, age_ms: float) -> np.ndarray:
        age = max(0.0, min(age_ms / 1000.0, 0.5))
        if len(self.history) == 1:
            return self.history[-1].copy()
        current = self.history[-1]
        previous = self.history[-2]
        if age <= DT:
            fraction = age / DT
            return current * (1.0 - fraction) + previous * fraction
        older = self.history[-3] if len(self.history) >= 3 else previous
        fraction = min(1.0, max(0.0, (age - DT) / DT))
        return previous * (1.0 - fraction) + older * fraction

    def update(
        self,
        position: np.ndarray,
        target: np.ndarray,
        age_ms: float,
        collided: bool,
        jump: int,
    ) -> np.ndarray:
        self.target = target.copy()
        self.history.append(position.copy())
        if len(self.history) > 4:
            self.history.pop(0)

        delayed = self._delayed_position(age_ms)
        displacement = float(np.linalg.norm(delayed - self.observed_pos))
        sample_dt = DT
        sample_speed = displacement / sample_dt
        valid = self.initialized and displacement >= 2.0 and sample_speed <= MAX_REASONABLE_SPEED

        if valid:
            velocity_sample = (delayed - self.observed_pos) / sample_dt
            sample_norm = float(np.linalg.norm(velocity_sample))
            if sample_norm > MAX_REASONABLE_SPEED:
                velocity_sample *= MAX_REASONABLE_SPEED / sample_norm

            measured_heading = math.atan2(float(velocity_sample[1]), float(velocity_sample[0]))
            predicted_heading = normalize_angle(self.estimated_heading + self.last_turn_delta)
            heading_error = abs(normalize_angle(measured_heading - predicted_heading))
            consistent = heading_error <= math.radians(35.0)
            first_measurement = not self.heading_valid
            corrected_heading = measured_heading
            if first_measurement:
                self.estimated_heading = corrected_heading
            else:
                self.estimated_heading = normalize_angle(
                    predicted_heading
                    + normalize_angle(corrected_heading - predicted_heading) * 0.35
                )
            self.heading_valid = True
            self.calibration_samples = min(self.calibration_samples + 1, 3)
            if first_measurement or consistent:
                self.heading_confidence = min(1.0, self.heading_confidence + 0.25)
            else:
                self.heading_confidence = max(0.0, self.heading_confidence - 0.15)
            self.velocity = self.velocity * 0.5 + velocity_sample * 0.5
            self.no_valid_steps = 0
        else:
            self.estimated_heading = normalize_angle(self.estimated_heading + self.last_turn_delta)
            self.heading_confidence *= 0.98
            self.no_valid_steps += 1

        prediction_age = min(age_ms, 500.0) / 1000.0
        self.observed_pos = delayed
        self.predicted_pos = np.clip(
            delayed + self.velocity * prediction_age,
            -1100.0,
            1100.0,
        )
        delta = self.target - self.predicted_pos
        distance = float(np.linalg.norm(delta))
        progress = self.prev_distance - distance
        self.time_since_collision = 0.0 if collided else self.time_since_collision + DT
        if self.initialized:
            self.stuck_time = self.stuck_time + DT if displacement < 2.0 else 0.0
            self.no_progress_time = self.no_progress_time + DT if progress < 0.2 else 0.0
        self.prev_distance = distance
        self.jump_probe = (-1.0 if self.prev_jump else 1.0) if collided else 0.0
        self.prev_jump = int(jump)
        self.initialized = True

        angle_error = normalize_angle(
            math.atan2(float(delta[1]), float(delta[0])) - self.estimated_heading
        )
        collision_touch = 1.0 if self.time_since_collision < 0.6 else 0.0
        return np.asarray(
            [
                np.clip(delta[0] / WORLD_SIZE, -1.0, 1.0),
                np.clip(delta[1] / WORLD_SIZE, -1.0, 1.0),
                math.sin(angle_error),
                math.cos(angle_error),
                np.clip(self.velocity[0] / MAX_REASONABLE_SPEED, -1.0, 1.0),
                np.clip(self.velocity[1] / MAX_REASONABLE_SPEED, -1.0, 1.0),
                np.clip(distance / WORLD_SIZE, 0.0, 1.0),
                np.clip(self.stuck_time / 3.0, 0.0, 1.0),
                collision_touch,
                self.jump_probe,
                np.clip(age_ms / 500.0, 0.0, 1.0),
                self.heading_confidence,
                np.clip(self.last_turn_delta / MAX_TURN, -1.0, 1.0),
            ],
            dtype=np.float32,
        )

    def calibration_action(self, turn_bin: int, speed_bin: int, jump: int):
        calibrating = self.calibration_samples < 2
        if not calibrating:
            return turn_bin, speed_bin, jump
        speed_bin = 0
        jump = 0
        if self.no_valid_steps >= 2:
            turn_bin = 5 if self.probe_phase == 0 else 1
            self.probe_phase = 1 - self.probe_phase
        else:
            turn_bin = 3
        return turn_bin, speed_bin, jump


def position_of(env) -> np.ndarray:
    return np.asarray([env.unwrapped.agent.pos[0], env.unwrapped.agent.pos[2]], dtype=np.float32) * 100.0


def target_of(env) -> np.ndarray:
    return np.asarray([env.unwrapped.box.pos[0], env.unwrapped.box.pos[2]], dtype=np.float32) * 100.0


@torch.no_grad()
def run_episode(
    model,
    seed: int,
    max_decisions: int,
    age_ms: float,
    env_id: str,
    max_episode_steps: int,
    writer=None,
) -> dict:
    env_class = {
        "MiniWorld-MazeS3-v0": MazeS3,
        "MiniWorld-MazeS3Fast-v0": MazeS3Fast,
    }.get(env_id)
    if env_class is None:
        raise ValueError(f"Unsupported MiniWorld maze: {env_id}")
    env = env_class(render_mode="rgb_array", max_episode_steps=max_episode_steps)
    image, _ = env.reset(seed=seed)
    target = target_of(env)
    position = position_of(env)
    belief = DelayedPositionBelief(position, target)
    observation = belief.update(position, target, age_ms, False, 0)
    hidden = torch.zeros(1, 1, 96)
    cell = torch.zeros_like(hidden)
    total_reward = 0.0
    primitive_steps = 0
    collisions = 0
    terminated = False
    truncated = False
    if writer is not None:
        writer.append_data(image)

    decision_count = 0
    for decision_count in range(1, max_decisions + 1):
        x = torch.from_numpy(observation).view(1, 1, -1)
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
        turn_action = env.unwrapped.actions.turn_left if turn_delta >= 0.0 else env.unwrapped.actions.turn_right
        actions = [turn_action] * turn_count
        actions.extend([env.unwrapped.actions.move_forward] * (2 if speed_bin == 1 else 1))
        if not actions:
            actions = [env.unwrapped.actions.move_forward]

        for action in actions:
            image, reward, terminated, truncated, _ = env.step(int(action))
            total_reward += float(reward)
            primitive_steps += 1
            if writer is not None:
                writer.append_data(image)
            if terminated or truncated:
                break

        old_position = position
        position = position_of(env)
        actual_displacement = float(np.linalg.norm(position - old_position))
        expected_displacement = 100.0 * 0.15 * (2 if speed_bin == 1 else 1)
        collided = expected_displacement > 0.0 and actual_displacement < max(expected_displacement * 0.2, 2.0)
        collisions += int(collided)
        belief.last_turn_delta = turn_delta
        observation = belief.update(position, target, age_ms, collided, jump_bin)
        if terminated or truncated:
            break

    distance = float(np.linalg.norm(target - position))
    env.close()
    return {
        "seed": seed,
        "success": bool(terminated),
        "decisions": decision_count,
        "primitive_steps": primitive_steps,
        "distance": distance,
        "collisions": collisions,
        "reward": total_reward,
        "calibration_samples": belief.calibration_samples,
        "heading_confidence": belief.heading_confidence,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--weights",
        default=str(BASE / "pipeline_out" / "policy_weights_v4_replay.pth"),
    )
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--max-decisions", type=int, default=80)
    parser.add_argument("--max-episode-steps", type=int, default=1000)
    parser.add_argument("--age-ms", type=float, default=AGE_MS)
    parser.add_argument("--env-id", default="MiniWorld-MazeS3-v0")
    parser.add_argument("--seed-start", type=int, default=42000)
    parser.add_argument("--video-episode", type=int, default=0)
    parser.add_argument(
        "--output",
        default="/tmp/file/盲人寻路测试/MiniWorld_V4_最新.mp4",
    )
    parser.add_argument("--no-video", action="store_true")
    args = parser.parse_args()

    from run_pipeline import OBS_DIM, RecurrentActorCritic

    state = torch.load(args.weights, map_location="cpu")
    model = RecurrentActorCritic(state_dim=OBS_DIM, hidden_dim=96, actor_width=48, macro=False)
    model.load_state_dict(state)
    model.eval()

    output = Path(args.output)
    writer = None
    if not args.no_video:
        output.parent.mkdir(parents=True, exist_ok=True)
        writer = imageio.get_writer(
            output,
            fps=20,
            codec="libx264",
            quality=7,
            ffmpeg_log_level="error",
        )

    results = []
    for episode in range(args.episodes):
        episode_writer = writer if episode == args.video_episode else None
        result = run_episode(
            model,
            args.seed_start + episode,
            args.max_decisions,
            args.age_ms,
            args.env_id,
            args.max_episode_steps,
            episode_writer,
        )
        results.append(result)
        print(result, flush=True)
    if writer is not None:
        writer.close()
        try:
            subprocess.Popen(
                ["mpv", "--force-window=yes", "--no-terminal", str(output)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
        except FileNotFoundError:
            pass

    successes = sum(item["success"] for item in results)
    print(
        {
            "episodes": args.episodes,
            "successes": successes,
            "success_rate": successes / max(args.episodes, 1),
            "mean_distance": sum(item["distance"] for item in results) / max(args.episodes, 1),
            "video": str(output.resolve()) if writer is not None else None,
        }
    )


if __name__ == "__main__":
    main()
