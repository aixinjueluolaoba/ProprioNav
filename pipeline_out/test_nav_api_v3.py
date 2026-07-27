"""High-level parity test for the simplified V3 nav API."""

from __future__ import annotations

import ctypes
import math
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent
WORLD_SIZE = 2250.0
DT = 0.3
MAX_TURN = math.pi / 4.0
HEADING_OFFSETS = np.deg2rad([-45.0, -25.0, -10.0, 0.0, 10.0, 25.0, 45.0])
GOAL_OFFSETS = np.deg2rad([-10.0, -6.0, -2.5, 0.0, 2.5, 6.0, 10.0])


def norm_angle(value: float) -> float:
    return math.atan2(math.sin(value), math.cos(value))


class PythonNav:
    def __init__(self, model):
        self.model = model
        self.h = np.zeros(96, dtype=np.float32)
        self.c = np.zeros(96, dtype=np.float32)
        self.first = True
        self.old_pos = np.zeros(2, dtype=np.float32)
        self.prev_dist = np.float32(0.0)
        self.prev_speed = np.float32(0.0)
        self.prev_jump = 0
        self.time_since_collision = np.float32(10.0)
        self.stuck_time = np.float32(0.0)
        self.no_progress_time = np.float32(0.0)
        self.jump_cooldown = 0

    def step(self, pos, target, heading):
        delta = target - pos
        distance = float(np.linalg.norm(delta))
        target_angle = math.atan2(float(delta[1]), float(delta[0]))
        angle_error = norm_angle(target_angle - heading)
        collided = False
        if not self.first:
            displacement = float(np.linalg.norm(pos - self.old_pos))
            progress = self.prev_dist - distance
            expected = self.prev_speed * DT
            collided = expected > 0.0 and displacement < max(expected * 0.2, 2.0)
            self.time_since_collision = np.float32(
                0.0 if collided else self.time_since_collision + np.float32(DT)
            )
            self.stuck_time = np.float32(
                self.stuck_time + np.float32(DT) if displacement < 2.0 else 0.0
            )
            self.no_progress_time = np.float32(
                self.no_progress_time + np.float32(DT) if progress < 0.2 else 0.0
            )

        collision_touch = float(self.time_since_collision < 0.6)
        jump_probe = (-1.0 if self.prev_jump else 1.0) if collided else 0.0
        obs = np.array(
            [
                np.clip(delta[0] / WORLD_SIZE, -1.0, 1.0),
                np.clip(delta[1] / WORLD_SIZE, -1.0, 1.0),
                math.sin(heading),
                math.cos(heading),
                math.sin(angle_error),
                math.cos(angle_error),
                np.clip(distance / WORLD_SIZE, 0.0, 1.0),
                np.clip(self.stuck_time / 3.0, 0.0, 1.0),
                collision_touch,
                jump_probe,
            ],
            dtype=np.float32,
        )
        steer, speed_logits, self.h, self.c = self.model(obs, self.h, self.c)
        steer_bin = int(steer.argmax())
        speed_bin = int(speed_logits.argmax())
        recovery = (
            self.time_since_collision < 0.9
            or self.no_progress_time > 0.5
            or self.stuck_time > 0.5
        )
        requested = (
            heading + HEADING_OFFSETS[steer_bin]
            if recovery
            else target_angle + GOAL_OFFSETS[steer_bin]
        )
        turn = np.clip(norm_angle(requested - heading), -MAX_TURN, MAX_TURN)
        direction = norm_angle(heading + float(turn))
        speed = 100.0 if speed_bin == 1 else 50.0

        if not collided:
            self.jump_cooldown = 0
        elif self.prev_jump:
            self.jump_cooldown = 8
        elif self.jump_cooldown > 0:
            self.jump_cooldown -= 1
        jump = int(collided and not self.prev_jump and self.jump_cooldown == 0)

        self.old_pos = pos.copy()
        self.prev_dist = np.float32(distance)
        self.prev_speed = np.float32(speed)
        self.prev_jump = jump
        self.first = False
        return direction, speed, jump


def main():
    library = ctypes.CDLL(str(BASE / "libncnn_rust.so"))
    float_ptr = ctypes.POINTER(ctypes.c_float)
    library.init_net.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
    library.init_net.restype = ctypes.c_void_p
    library.free_net.argtypes = [ctypes.c_void_p]
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

    param = str(BASE / "policy.param").encode()
    model_bin = str(BASE / "policy.bin").encode()
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

    reference = PythonNav(ncnn_model)

    pos = np.array([-600.0, -350.0], dtype=np.float32)
    target = np.array([750.0, 500.0], dtype=np.float32)
    heading = -2.2
    max_angle_diff = 0.0
    jumps = 0
    for step in range(40):
        expected = reference.step(pos, target, heading)
        direction = ctypes.c_float()
        speed = ctypes.c_float()
        jump = ctypes.c_int()
        status = library.nav_step(
            handle,
            float(pos[0]),
            float(pos[1]),
            float(target[0]),
            float(target[1]),
            heading,
            ctypes.byref(direction),
            ctypes.byref(speed),
            ctypes.byref(jump),
        )
        if status != 0:
            raise RuntimeError(f"nav_step failed at {step}: {status}")
        angle_diff = abs(norm_angle(direction.value - expected[0]))
        max_angle_diff = max(max_angle_diff, angle_diff)
        if angle_diff >= 0.02 or abs(speed.value - expected[1]) >= 0.5 or jump.value != expected[2]:
            raise AssertionError(
                f"step={step} py={expected} rust={(direction.value, speed.value, jump.value)}"
            )
        jumps += jump.value

        # Tree-like obstacle: jumping does not help for several steps.
        tree_blocked = 5 <= step <= 8
        # Low obstacle: the first jump clears it.
        low_blocked = 18 <= step <= 22 and jump.value == 0
        if not tree_blocked and not low_blocked:
            pos += np.array(
                [math.cos(expected[0]), math.sin(expected[0])], dtype=np.float32
            ) * expected[1] * DT
            heading = expected[0]

    library.nav_free(handle)
    library.free_net(low_handle)
    print(f"40 steps matched; max_angle_diff={max_angle_diff:.6f}; jumps={jumps}")
    if jumps == 0:
        raise SystemExit("jump probe controller was not exercised")


if __name__ == "__main__":
    main()
