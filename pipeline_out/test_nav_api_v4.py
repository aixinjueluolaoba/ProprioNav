"""High-level parity test for the V4 unknown-heading navigation API."""

from __future__ import annotations

import ctypes
import math
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent
WORLD_SIZE = 2250.0
DT = 0.3
MAX_TURN = math.pi / 4.0
MAX_POSITION_AGE_MS = 500.0
STALE_POSITION_AGE_MS = 1000.0
MIN_SAMPLE_DT = 0.05
MAX_SAMPLE_DT = 1.5
MAX_REASONABLE_SPEED = 500.0
TURN_OFFSETS = np.deg2rad([-45.0, -25.0, -10.0, 0.0, 10.0, 25.0, 45.0])


def norm_angle(value: float) -> float:
    return math.atan2(math.sin(value), math.cos(value))


def freshness_scale(age_ms: float) -> float:
    if age_ms <= 200.0:
        return 1.0
    if age_ms <= MAX_POSITION_AGE_MS:
        return 1.0 - 0.5 * ((age_ms - 200.0) / 300.0)
    if age_ms <= STALE_POSITION_AGE_MS:
        return 0.5 * ((STALE_POSITION_AGE_MS - age_ms) / 500.0)
    return 0.0


class PythonV4:
    def __init__(self, model):
        self.model = model
        self.h = np.zeros(96, dtype=np.float32)
        self.c = np.zeros(96, dtype=np.float32)
        self.first_step = True
        self.old_pos = np.zeros(2, dtype=np.float32)
        self.old_age_ms = 0.0
        self.velocity = np.zeros(2, dtype=np.float32)
        self.estimated_heading = 0.0
        self.heading_valid = False
        self.heading_confidence = 0.0
        self.calibration_samples = 0
        self.no_valid_observation_steps = 0
        self.probe_phase = 0
        self.last_turn_delta = 0.0
        self.prev_dist = 0.0
        self.prev_speed = 0.0
        self.prev_jump = 0
        self.time_since_collision = 10.0
        self.stuck_time = 0.0
        self.no_progress_time = 0.0
        self.jump_cooldown = 0
        self.recovery_commit_left = 0
        self.recovery_phase = 0
        self.recovery_bin = 6

    def step(self, pos, target, position_age_ms):
        age_ms = min(max(float(position_age_ms), 0.0), 2000.0)
        displacement = 0.0
        sample_dt = DT
        collided = False
        collision_measurement_valid = False

        if not self.first_step:
            sample_dt = DT - (age_ms - self.old_age_ms) / 1000.0
            displacement = float(np.linalg.norm(pos - self.old_pos))
            sample_time_valid = MIN_SAMPLE_DT <= sample_dt <= MAX_SAMPLE_DT
            collision_measurement_valid = sample_time_valid and age_ms <= STALE_POSITION_AGE_MS
            expected = self.prev_speed * max(sample_dt, MIN_SAMPLE_DT)
            collided = (
                expected > 0.0
                and collision_measurement_valid
                and displacement < max(expected * 0.2, 2.0)
            )

        sample_time_valid = MIN_SAMPLE_DT <= sample_dt <= MAX_SAMPLE_DT
        sample_speed = displacement / max(sample_dt, MIN_SAMPLE_DT)
        velocity_valid = (
            sample_time_valid
            and displacement >= 2.0
            and sample_speed <= MAX_REASONABLE_SPEED
        )
        velocity_sample = np.zeros(2, dtype=np.float32)
        if velocity_valid:
            velocity_sample = (pos - self.old_pos) / sample_dt
            sample_norm = float(np.linalg.norm(velocity_sample))
            if sample_norm > MAX_REASONABLE_SPEED:
                velocity_sample *= MAX_REASONABLE_SPEED / sample_norm
            self.velocity = self.velocity * 0.5 + velocity_sample * 0.5

            measured_heading = math.atan2(float(velocity_sample[1]), float(velocity_sample[0]))
            predicted_heading = norm_angle(self.estimated_heading + self.last_turn_delta)
            heading_error = abs(norm_angle(measured_heading - predicted_heading))
            first_measurement = not self.heading_valid
            consistent = heading_error <= math.radians(35.0)
            self.estimated_heading = measured_heading
            self.heading_valid = True
            if first_measurement or consistent:
                self.calibration_samples = min(self.calibration_samples + 1, 3)
                self.heading_confidence = min(self.heading_confidence + 0.25, 1.0)
            else:
                self.heading_confidence = max(self.heading_confidence - 0.15, 0.0)
            self.no_valid_observation_steps = 0
        else:
            self.estimated_heading = norm_angle(self.estimated_heading + self.last_turn_delta)
            self.heading_confidence *= 0.98
            self.no_valid_observation_steps += 1

        if not self.first_step:
            prediction_age = min(age_ms, MAX_POSITION_AGE_MS) / 1000.0
            predicted_pos = pos + self.velocity * prediction_age
            predicted_pos = np.clip(predicted_pos, -1100.0, 1100.0)
            distance = float(np.linalg.norm(target - predicted_pos))
            progress = self.prev_dist - distance
            self.time_since_collision = 0.0 if collided else self.time_since_collision + DT
            if collision_measurement_valid:
                self.stuck_time = self.stuck_time + DT if displacement < 2.0 else 0.0
                self.no_progress_time = self.no_progress_time + DT if progress < 0.2 else 0.0
        prediction_age = min(age_ms, MAX_POSITION_AGE_MS) / 1000.0
        predicted_pos = np.clip(pos + self.velocity * prediction_age, -1100.0, 1100.0)
        delta = target - predicted_pos
        distance = float(np.linalg.norm(delta))
        target_angle = math.atan2(float(delta[1]), float(delta[0]))
        angle_error = norm_angle(target_angle - self.estimated_heading)
        collision_touch = 1.0 if self.time_since_collision < 0.6 else 0.0
        jump_probe = (-1.0 if self.prev_jump else 1.0) if collided else 0.0
        obs = np.array(
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
                jump_probe,
                np.clip(age_ms / MAX_POSITION_AGE_MS, 0.0, 1.0),
                self.heading_confidence,
                np.clip(self.last_turn_delta / MAX_TURN, -1.0, 1.0),
            ],
            dtype=np.float32,
        )
        steer_logits, speed_logits, self.h, self.c = self.model(obs, self.h, self.c)
        steer_bin = int(steer_logits.argmax())
        speed_bin = int(speed_logits.argmax())
        calibrating = self.calibration_samples < 2
        if calibrating:
            speed_bin = 0
            if self.no_valid_observation_steps >= 2:
                steer_bin = 5 if self.probe_phase == 0 else 1
                self.probe_phase = 1 - self.probe_phase
            else:
                steer_bin = 3
        recovery_signal = (
            collided
            or self.time_since_collision < 0.9
            or self.no_progress_time > 0.6
        )
        if not calibrating and recovery_signal and self.recovery_commit_left <= 0:
            self.recovery_bin = 6 if self.recovery_phase == 0 else 0
            self.recovery_phase = 1 - self.recovery_phase
            self.recovery_commit_left = 4
        if not calibrating and self.recovery_commit_left > 0:
            steer_bin = self.recovery_bin
            self.recovery_commit_left -= 1
        turn_delta = float(TURN_OFFSETS[steer_bin])
        speed = (100.0 if speed_bin == 1 else 50.0) * freshness_scale(age_ms)

        if not collided:
            self.jump_cooldown = 0
        elif self.prev_jump:
            self.jump_cooldown = 8
        elif self.jump_cooldown > 0:
            self.jump_cooldown -= 1
        jump = int(not calibrating and collided and not self.prev_jump and self.jump_cooldown == 0)

        self.old_pos = pos.copy()
        self.old_age_ms = age_ms
        self.prev_dist = distance
        self.prev_speed = speed
        self.prev_jump = jump
        self.last_turn_delta = turn_delta
        self.first_step = False
        return turn_delta, speed, jump


def main():
    float_ptr = ctypes.POINTER(ctypes.c_float)
    library = ctypes.CDLL(str(BASE / "libncnn_rust.so"))
    library.init_net.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
    library.init_net.restype = ctypes.c_void_p
    library.free_net.argtypes = [ctypes.c_void_p]
    library.free_net.restype = None
    library.run_inference.argtypes = [
        ctypes.c_void_p,
        float_ptr,
        float_ptr,
        float_ptr,
        float_ptr,
        float_ptr,
        float_ptr,
        float_ptr,
    ]
    library.run_inference.restype = ctypes.c_int
    library.nav_init.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
    library.nav_init.restype = ctypes.c_void_p
    library.nav_step.argtypes = [
        ctypes.c_void_p,
        ctypes.c_float,
        ctypes.c_float,
        ctypes.c_float,
        ctypes.c_float,
        ctypes.c_float,
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_int),
    ]
    library.nav_step.restype = ctypes.c_int
    library.nav_free.argtypes = [ctypes.c_void_p]
    library.nav_free.restype = None

    param = str(BASE / "policy_v4.param").encode()
    model_bin = str(BASE / "policy_v4.bin").encode()
    low_handle = library.init_net(param, model_bin)
    handle = library.nav_init(param, model_bin)
    if not low_handle or not handle:
        raise RuntimeError("nav_init failed")

    def ncnn_model(obs, h, c):
        steer = np.zeros(7, dtype=np.float32)
        speed = np.zeros(2, dtype=np.float32)
        h_out = np.zeros(96, dtype=np.float32)
        c_out = np.zeros(96, dtype=np.float32)
        arrays = [obs, h, c, steer, speed, h_out, c_out]
        pointers = [array.ctypes.data_as(float_ptr) for array in arrays]
        status = library.run_inference(low_handle, *pointers)
        if status != 0:
            raise RuntimeError(f"run_inference failed: {status}")
        return steer, speed, h_out, c_out

    reference = PythonV4(ncnn_model)
    target = np.array([750.0, 500.0], dtype=np.float32)
    pos = np.array([0.0, 0.0], dtype=np.float32)
    max_turn_diff = 0.0
    jumps = 0
    for step in range(40):
        if step < 5 or step > 8:
            pos += np.array([30.0, 0.0], dtype=np.float32)
        age = 100.0 if step < 3 else 250.0 if step < 8 else 500.0
        expected = reference.step(pos, target, age)
        turn_delta = ctypes.c_float()
        speed = ctypes.c_float()
        jump = ctypes.c_int()
        status = library.nav_step(
            handle,
            float(pos[0]),
            float(pos[1]),
            float(target[0]),
            float(target[1]),
            age,
            ctypes.byref(turn_delta),
            ctypes.byref(speed),
            ctypes.byref(jump),
        )
        if status != 0:
            raise RuntimeError(f"nav_step failed at {step}: {status}")
        turn_diff = abs(turn_delta.value - expected[0])
        max_turn_diff = max(max_turn_diff, turn_diff)
        if turn_diff >= 0.02 or abs(speed.value - expected[1]) >= 0.5 or jump.value != expected[2]:
            raise AssertionError(
                f"step={step} py={expected} rust={(turn_delta.value, speed.value, jump.value)}"
            )
        if not -MAX_TURN <= turn_delta.value <= MAX_TURN:
            raise AssertionError(f"turn_delta out of range: {turn_delta.value}")
        if not 0.0 <= speed.value <= 100.0:
            raise AssertionError(f"speed out of range: {speed.value}")
        jumps += jump.value

    library.nav_free(handle)
    library.free_net(low_handle)
    print(f"40 steps matched; max_turn_diff={max_turn_diff:.6f}; jumps={jumps}")


if __name__ == "__main__":
    main()
