"""延迟位置信念估计（从旧的 MiniWorld V4 测试脚本中提取，供 3D viewer 等复用）。

标量版 delayed-position 状态估计: 维护位置历史、外推延迟观测、
由位移估计朝向/速度与校准。不依赖 MiniWorld。
"""

from __future__ import annotations

import math

import numpy as np

WORLD_SIZE = 2250.0
DT = 0.30
MAX_TURN = math.pi / 4.0
MAX_REASONABLE_SPEED = 500.0


def normalize_angle(value: float) -> float:
    return math.atan2(math.sin(value), math.cos(value))


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


