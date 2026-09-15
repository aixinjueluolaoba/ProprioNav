"""Lightweight real-time 3D viewer for the V5 native blind navigator."""

from __future__ import annotations

import argparse
import ctypes
import math
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyglet
import torch
from pyglet import gl
from pyglet.window import key, mouse

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(BASE / "pipeline_out"))
from run_pipeline import RecurrentActorCritic  # noqa: E402
from delayed_belief import DelayedPositionBelief  # noqa: E402

WORLD_SIZE = 2250.0
WORLD_SCALE = 10.0
PHYSICAL_WORLD_SIZE = WORLD_SIZE * WORLD_SCALE
LOGICAL_WORLD_HALF = 1100.0
PHYSICAL_WORLD_HALF = LOGICAL_WORLD_HALF * WORLD_SCALE
WAYPOINT_MAX_LEG = 650.0
WAYPOINT_REACH_RADIUS = 55.0
GOAL_REACH_RADIUS = 25.0
PLAYER_RADIUS = 5.0
DECISION_INTERVAL = 0.30
GRAVITY = -45.0
JUMP_VELOCITY = 30.0


@dataclass(frozen=True)
class BoxObstacle:
    name: str
    x: float
    z: float
    width: float
    depth: float
    height: float
    color: tuple[float, float, float]


OBSTACLES = [
    BoxObstacle("Building_A", -480, -220, 180, 150, 48, (0.36, 0.40, 0.44)),
    BoxObstacle("Building_B", 40, -410, 220, 170, 68, (0.42, 0.43, 0.46)),
    BoxObstacle("Building_C", 430, 180, 160, 210, 54, (0.38, 0.42, 0.45)),
    BoxObstacle("Building_D", -220, 460, 240, 150, 58, (0.40, 0.43, 0.45)),
    BoxObstacle("Wall_1", -80, 20, 300, 18, 10, (0.58, 0.46, 0.30)),
    BoxObstacle("Wall_2", 520, -230, 18, 290, 10, (0.58, 0.46, 0.30)),
    BoxObstacle("Wall_3", -610, 280, 320, 18, 10, (0.58, 0.46, 0.30)),
    BoxObstacle("Rock_1", -720, -480, 60, 48, 18, (0.30, 0.32, 0.31)),
    BoxObstacle("Rock_2", -590, -360, 48, 54, 14, (0.32, 0.33, 0.32)),
    BoxObstacle("Rock_3", 270, -80, 66, 58, 16, (0.30, 0.32, 0.31)),
    BoxObstacle("Rock_4", 620, 420, 58, 72, 15, (0.30, 0.32, 0.31)),
    BoxObstacle("Rock_5", 280, 570, 74, 62, 18, (0.30, 0.32, 0.31)),
    BoxObstacle("Rock_6", -480, 650, 62, 72, 16, (0.30, 0.32, 0.31)),
]
OBSTACLES = [
    BoxObstacle(
        item.name,
        item.x * WORLD_SCALE,
        item.z * WORLD_SCALE,
        item.width * WORLD_SCALE,
        item.depth * WORLD_SCALE,
        item.height * WORLD_SCALE,
        item.color,
    )
    for item in OBSTACLES
]
PLAYER_RADIUS *= WORLD_SCALE
GRAVITY *= WORLD_SCALE
JUMP_VELOCITY *= WORLD_SCALE


class WaypointRoute:
    """Keep each model target inside the distance/coordinate range it was trained on."""

    def __init__(self, start: np.ndarray, final_target: np.ndarray):
        self.final_target = final_target.copy()
        self.points = []
        self.index = 0
        self.reset(start, final_target)

    def reset(self, start: np.ndarray, final_target: np.ndarray):
        self.final_target = final_target.copy()
        start_logical = start / WORLD_SCALE
        target_logical = final_target / WORLD_SCALE
        delta = target_logical - start_logical
        distance = float(np.linalg.norm(delta))
        legs = max(1, int(math.ceil(distance / WAYPOINT_MAX_LEG)))
        self.points = [
            (start_logical + delta * (step / legs)) * WORLD_SCALE
            for step in range(1, legs + 1)
        ]
        self.index = 0

    @property
    def current(self) -> np.ndarray:
        return self.points[min(self.index, len(self.points) - 1)]

    @property
    def reached_final(self) -> bool:
        return self.index >= len(self.points) - 1

    def update(self, position: np.ndarray) -> bool:
        switched = False
        while self.index < len(self.points) - 1:
            distance = float(np.linalg.norm(self.points[self.index] - position))
            if distance > WAYPOINT_REACH_RADIUS * WORLD_SCALE:
                break
            self.index += 1
            switched = True
        return switched

    def distance_to_current(self, position: np.ndarray) -> float:
        return float(np.linalg.norm(self.current - position)) / WORLD_SCALE


class NativeNavigator:
    def __init__(self, library_path: Path, param_path: Path, bin_path: Path):
        self.library = ctypes.CDLL(str(library_path))
        self.library.nav_init.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
        self.library.nav_init.restype = ctypes.c_void_p
        self.library.nav_step.argtypes = [
            ctypes.c_void_p,
            ctypes.c_float,
            ctypes.c_float,
            ctypes.c_float,
            ctypes.c_float,
            ctypes.c_float,
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_float),
        ]
        self.library.nav_step.restype = ctypes.c_int
        self.library.nav_step_feedback.argtypes = [
            ctypes.c_void_p,
            ctypes.c_float,
            ctypes.c_float,
            ctypes.c_float,
            ctypes.c_float,
            ctypes.c_float,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_float),
        ]
        self.library.nav_step_feedback.restype = ctypes.c_int
        self.library.nav_free.argtypes = [ctypes.c_void_p]
        self.state = self.library.nav_init(str(param_path).encode(), str(bin_path).encode())
        if not self.state:
            raise RuntimeError("nav_init failed")

    def step(self, pos_x: float, pos_z: float, target_x: float, target_z: float, age_ms: float, collided: bool = False):
        turn = ctypes.c_float()
        speed = ctypes.c_float()
        jump = ctypes.c_int()
        macro = ctypes.c_int()
        abs_angle = ctypes.c_float()
        result = self.library.nav_step_feedback(
            self.state,
            pos_x,
            pos_z,
            target_x,
            target_z,
            age_ms,
            int(collided),
            ctypes.byref(turn),
            ctypes.byref(speed),
            ctypes.byref(jump),
            ctypes.byref(macro),
            ctypes.byref(abs_angle),
        )
        if result != 0:
            raise RuntimeError(f"nav_step failed: {result}")
        return (
            float(turn.value),
            float(speed.value),
            int(jump.value),
            int(macro.value),
            float(abs_angle.value),
        )

    def close(self):
        if self.state:
            self.library.nav_free(self.state)
            self.state = None


class TorchNavigator:
    """Exact V4 PyTorch inference with the same delayed-position belief state."""

    def __init__(self, weights_path: Path):
        state = torch.load(weights_path, map_location="cpu")
        self.model = RecurrentActorCritic(state_dim=13, hidden_dim=96, actor_width=48, macro=True)
        self.model.load_state_dict(state)
        self.model.eval()
        self.hidden = torch.zeros(1, 1, 96)
        self.cell = torch.zeros_like(self.hidden)
        self.belief = None
        self.previous_jump = 0
        self.recovery_commit_left = 0
        self.recovery_phase = 0
        self.recovery_bin = 6

    def step(self, pos_x: float, pos_z: float, target_x: float, target_z: float, age_ms: float, collided: bool = False):
        position = np.asarray([pos_x, pos_z], dtype=np.float32)
        target = np.asarray([target_x, target_z], dtype=np.float32)
        if self.belief is None:
            self.belief = DelayedPositionBelief(position, target)
        observation = self.belief.update(position, target, age_ms, collided, self.previous_jump)
        with torch.no_grad():
            output, (self.hidden, self.cell) = self.model.get_states(
                torch.from_numpy(observation).view(1, 1, 13),
                (self.hidden, self.cell),
                torch.zeros(1, 1),
            )
            features = self.model.actor_fc(output.squeeze(0))
            steer_bin = int(self.model.steer_head(features).argmax(dim=-1).item())
            speed_bin = int(self.model.speed_head(features).argmax(dim=-1).item())
            jump_bin = int(self.model.jump_head(features).argmax(dim=-1).item())
        steer_bin, speed_bin, jump_bin = self.belief.calibration_action(
            steer_bin, speed_bin, jump_bin
        )
        recovery_signal = (
            collided
            or self.belief.time_since_collision < 0.9
            or self.belief.no_progress_time > 0.6
        )
        calibrating = self.belief.calibration_samples < 2
        if not calibrating and recovery_signal and self.recovery_commit_left <= 0:
            self.recovery_bin = 6 if self.recovery_phase == 0 else 0
            self.recovery_phase = 1 - self.recovery_phase
            self.recovery_commit_left = 4
        if not calibrating and self.recovery_commit_left > 0:
            steer_bin = self.recovery_bin
            self.recovery_commit_left -= 1
        turn_offsets = np.deg2rad([-45.0, -25.0, -10.0, 0.0, 10.0, 25.0, 45.0])
        turn_delta = float(turn_offsets[steer_bin])
        speed = 100.0 if speed_bin == 1 else 50.0
        if age_ms > 200.0:
            speed *= max(0.5, 1.0 - 0.5 * min(age_ms - 200.0, 300.0) / 300.0)
        self.belief.last_turn_delta = turn_delta
        self.previous_jump = jump_bin
        return turn_delta, speed, jump_bin, 0

    def close(self):
        pass


def cube_vertices(x: float, y: float, z: float, width: float, height: float, depth: float):
    hx, hz = width * 0.5, depth * 0.5
    return (
        (x - hx, y, z - hz), (x + hx, y, z - hz),
        (x + hx, y, z + hz), (x - hx, y, z + hz),
        (x - hx, y + height, z - hz), (x + hx, y + height, z - hz),
        (x + hx, y + height, z + hz), (x - hx, y + height, z + hz),
    )


def draw_cube(obstacle: BoxObstacle):
    vertices = cube_vertices(obstacle.x, 0.0, obstacle.z, obstacle.width, obstacle.height, obstacle.depth)
    faces = ((0, 1, 2, 3), (4, 7, 6, 5), (0, 4, 5, 1), (1, 5, 6, 2), (2, 6, 7, 3), (4, 0, 3, 7))
    gl.glColor3f(*obstacle.color)
    gl.glBegin(gl.GL_QUADS)
    for face in faces:
        for index in face:
            gl.glVertex3f(*vertices[index])
    gl.glEnd()


class Viewer(pyglet.window.Window):
    def __init__(self, navigator: NativeNavigator, delay_ms: float, **kwargs):
        super().__init__(width=1280, height=720, caption="ProprioNav V5 - Lightweight 3D Test", resizable=True, **kwargs)
        self.navigator = navigator
        self.delay_ms = float(delay_ms)
        self.position = np.array([-8500.0, -8500.0], dtype=np.float32)
        self.target = np.array([8500.0, 8500.0], dtype=np.float32)
        self.route = WaypointRoute(self.position, self.target)
        self.heading = math.radians(37.0)
        self.vertical = 0.0
        self.vertical_velocity = 0.0
        self.samples = deque(maxlen=120)
        self.samples.append((time.monotonic(), self.position.copy()))
        self.decision_timer = 0.0
        self.turn = 0.0
        self.speed = 0.0
        self.jump = 0
        self.current_macro = 0
        self.decisions = 0
        self.logged_decisions = 0
        self.collisions = 0
        self.status = "V5 原生模型已加载"
        self.last_age = self.delay_ms
        self.last_distance = float(np.linalg.norm(self.target - self.position)) / WORLD_SCALE
        self.last_waypoint_distance = self.route.distance_to_current(self.position)
        self.last_collision = False
        self.start_time = time.monotonic()
        self.free_camera = True
        self.camera_position = np.array([0.0, 2800.0, -9000.0], dtype=np.float32)
        self.camera_yaw = math.pi * 0.5
        self.camera_pitch = math.radians(-12.0)
        self.camera_speed = 3200.0
        self.label = pyglet.text.Label(
            "", x=18, y=self.height - 18, color=(240, 245, 250, 255),
            font_size=14, multiline=True, width=700,
        )
        self.help_label = pyglet.text.Label(
            "", x=18, y=16, color=(220, 225, 230, 255), font_size=12,
        )
        self.keys = key.KeyStateHandler()
        self.push_handlers(self.keys)
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glClearColor(0.18, 0.25, 0.31, 1.0)
        pyglet.clock.schedule_interval(self.update, 1.0 / 60.0)

    def on_resize(self, width, height):
        super().on_resize(width, height)
        self.label.y = height - 18
        gl.glViewport(0, 0, width, height)
        return pyglet.event.EVENT_HANDLED

    def delayed_position(self, now: float):
        wanted = now - self.delay_ms / 1000.0
        delayed = self.samples[0][1]
        for sample_time, sample_position in self.samples:
            if sample_time <= wanted:
                delayed = sample_position
            else:
                break
        oldest_age = max(0.0, (now - self.samples[0][0]) * 1000.0)
        return delayed.copy(), min(self.delay_ms, oldest_age if oldest_age else self.delay_ms)

    @staticmethod
    def collides(position: np.ndarray, obstacle: BoxObstacle, vertical: float) -> bool:
        if vertical > obstacle.height + 1.5:
            return False
        closest_x = max(obstacle.x - obstacle.width * 0.5, min(float(position[0]), obstacle.x + obstacle.width * 0.5))
        closest_z = max(obstacle.z - obstacle.depth * 0.5, min(float(position[1]), obstacle.z + obstacle.depth * 0.5))
        dx = float(position[0]) - closest_x
        dz = float(position[1]) - closest_z
        return dx * dx + dz * dz < PLAYER_RADIUS * PLAYER_RADIUS

    def try_move(self, displacement: np.ndarray):
        candidate = self.position + displacement
        for obstacle in OBSTACLES:
            if self.collides(candidate, obstacle, self.vertical):
                return False
        self.position = np.clip(candidate, -PHYSICAL_WORLD_HALF, PHYSICAL_WORLD_HALF)
        return True

    def update(self, dt: float):
        dt = min(float(dt), 0.05)
        if self.free_camera:
            self.move_free_camera(dt)
        now = time.monotonic()
        self.samples.append((now, self.position.copy()))
        self.vertical_velocity += GRAVITY * dt
        self.vertical = max(0.0, self.vertical + self.vertical_velocity * dt)
        if self.vertical == 0.0 and self.vertical_velocity < 0.0:
            self.vertical_velocity = 0.0

        self.decision_timer -= dt
        if self.decision_timer <= 0.0:
            self.decision_timer += DECISION_INTERVAL
            switched = self.route.update(self.position)
            if switched:
                self.status = f"已切换到 waypoint {self.route.index + 1}/{len(self.route.points)}"
            observed, age = self.delayed_position(now)
            self.last_age = age
            observed_logical = observed / WORLD_SCALE
            waypoint_logical = self.route.current / WORLD_SCALE
            try:
                self.turn, self.speed, self.jump, self.current_macro, abs_angle = (
                    self.navigator.step(
                        float(observed_logical[0]), float(observed_logical[1]),
                        float(waypoint_logical[0]), float(waypoint_logical[1]),
                        age, self.last_collision,
                    )
                )
                # 库内部维护的绝对摇杆角, 直接用它, 不再自己累加 turn。
                self.heading = abs_angle
                self.decisions += 1
                if self.jump and self.vertical == 0.0:
                    self.vertical_velocity = JUMP_VELOCITY
            except Exception as error:
                self.status = f"model error: {error}"
                self.speed = 0.0

        forward = np.array([math.cos(self.heading), math.sin(self.heading)], dtype=np.float32)
        moved = self.try_move(forward * self.speed * WORLD_SCALE * dt)
        self.last_collision = not moved and self.speed > 0.0
        if self.last_collision:
            self.collisions += 1
        self.last_distance = float(np.linalg.norm(self.target - self.position)) / WORLD_SCALE
        self.last_waypoint_distance = self.route.distance_to_current(self.position)
        if self.last_distance < GOAL_REACH_RADIUS:
            self.status = "已到达目标，关闭窗口结束测试"
            self.speed = 0.0

        if self.decisions and self.decisions % 20 == 0 and self.logged_decisions != self.decisions:
            self.logged_decisions = self.decisions
            print(
                f"decision={self.decisions} pos=({self.position[0]:.1f},{self.position[1]:.1f}) "
                f"distance={self.last_distance:.1f} turn={math.degrees(self.turn):.1f} "
                f"speed={self.speed:.1f} collision={int(self.last_collision)}",
                flush=True,
            )

        self.label.text = (
            f"{self.status}\n"
            f"目标距离 {self.last_distance:7.1f}   waypoint距离 {self.last_waypoint_distance:6.1f}   "
            f"段 {self.route.index + 1}/{len(self.route.points)}\n"
            f"延迟 {self.delay_ms:4.0f}ms / age {self.last_age:4.0f}ms   "
            f"转向 {math.degrees(self.turn):6.1f}°   速度 {self.speed:5.1f}   跳跃 {self.jump}   宏 {self.current_macro}   "
            f"decisions {self.decisions}   collisions {self.collisions}\n"
            f"camera={'FREE' if self.free_camera else 'FOLLOW'}"
        )
        self.help_label.text = "WASD 移动视角   Q/E 升降   鼠标左键/右键拖动转视角   F 跟随/自由   +/- 调延迟   R 重置   Esc/Q 退出"

    def on_key_press(self, symbol, modifiers):
        if symbol in (key.ESCAPE, key.Q):
            self.close()
        elif symbol in (key.PLUS, key.NUM_ADD, key.EQUAL):
            self.delay_ms = min(500.0, self.delay_ms + 25.0)
        elif symbol in (key.MINUS, key.NUM_SUBTRACT):
            self.delay_ms = max(0.0, self.delay_ms - 25.0)
        elif symbol == key.R:
            self.position[:] = (-8500.0, -8500.0)
            self.route.reset(self.position, self.target)
            self.heading = math.radians(37.0)
            self.vertical = 0.0
            self.vertical_velocity = 0.0
            self.samples.clear()
            self.samples.append((time.monotonic(), self.position.copy()))
            self.status = "V5 原生模型已重置，waypoint 路线已重建"
        elif symbol == key.F:
            self.free_camera = not self.free_camera

    def on_mouse_drag(self, x, y, dx, dy, buttons, modifiers):
        if buttons & (mouse.LEFT | mouse.RIGHT) and self.free_camera:
            self.camera_yaw += dx * 0.005
            self.camera_pitch = max(-1.45, min(1.45, self.camera_pitch + dy * 0.005))

    def move_free_camera(self, dt: float):
        forward = np.array([math.cos(self.camera_yaw), 0.0, math.sin(self.camera_yaw)], dtype=np.float32)
        right = np.array([-math.sin(self.camera_yaw), 0.0, math.cos(self.camera_yaw)], dtype=np.float32)
        movement = np.zeros(3, dtype=np.float32)
        if self.keys[key.W]:
            movement += forward
        if self.keys[key.S]:
            movement -= forward
        if self.keys[key.D]:
            movement += right
        if self.keys[key.A]:
            movement -= right
        if self.keys[key.E]:
            movement[1] += 1.0
        if self.keys[key.Q]:
            movement[1] -= 1.0
        norm = float(np.linalg.norm(movement))
        if norm > 0.0:
            self.camera_position += movement / norm * self.camera_speed * dt
            self.camera_position[0] = np.clip(self.camera_position[0], -PHYSICAL_WORLD_HALF * 1.25, PHYSICAL_WORLD_HALF * 1.25)
            self.camera_position[1] = np.clip(self.camera_position[1], 50.0, PHYSICAL_WORLD_HALF * 1.25)
            self.camera_position[2] = np.clip(self.camera_position[2], -PHYSICAL_WORLD_HALF * 1.25, PHYSICAL_WORLD_HALF * 1.25)

    def set_camera(self):
        if self.free_camera:
            camera = self.camera_position
            direction = np.array([
                math.cos(self.camera_pitch) * math.cos(self.camera_yaw),
                math.sin(self.camera_pitch),
                math.cos(self.camera_pitch) * math.sin(self.camera_yaw),
            ], dtype=np.float32)
            look = camera + direction * 100.0
        else:
            forward = np.array([math.cos(self.heading), math.sin(self.heading)], dtype=np.float32)
            camera = np.array([self.position[0] - forward[0] * 550.0, 350.0, self.position[1] - forward[1] * 550.0])
            look = np.array([self.position[0] + forward[0] * 350.0, 80.0 + self.vertical, self.position[1] + forward[1] * 350.0])
        gl.glMatrixMode(gl.GL_MODELVIEW)
        gl.glLoadIdentity()
        gl.gluLookAt(camera[0], camera[1], camera[2], look[0], look[1], look[2], 0.0, 1.0, 0.0)

    def on_draw(self):
        self.clear()
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glMatrixMode(gl.GL_PROJECTION)
        gl.glLoadIdentity()
        aspect = max(0.1, self.width / max(1.0, float(self.height)))
        gl.gluPerspective(68.0, aspect, 1.0, PHYSICAL_WORLD_SIZE * 2.0)
        self.set_camera()

        gl.glColor3f(0.20, 0.27, 0.18)
        gl.glBegin(gl.GL_QUADS)
        gl.glVertex3f(-PHYSICAL_WORLD_HALF, 0, -PHYSICAL_WORLD_HALF)
        gl.glVertex3f(PHYSICAL_WORLD_HALF, 0, -PHYSICAL_WORLD_HALF)
        gl.glVertex3f(PHYSICAL_WORLD_HALF, 0, PHYSICAL_WORLD_HALF)
        gl.glVertex3f(-PHYSICAL_WORLD_HALF, 0, PHYSICAL_WORLD_HALF)
        gl.glEnd()

        for obstacle in OBSTACLES:
            draw_cube(obstacle)

        gl.glColor3f(0.10, 0.35, 0.90)
        gl.glPushMatrix()
        gl.glTranslatef(float(self.position[0]), float(self.vertical + 5.0), float(self.position[1]))
        gl.glRotatef(-math.degrees(self.heading), 0.0, 1.0, 0.0)
        gl.glScalef(50.0, 100.0, 50.0)
        gl.glPointSize(10.0)
        pyglet.graphics.draw(1, gl.GL_POINTS, ("v3f", (0.0, 0.0, 0.0)))
        gl.glPopMatrix()

        # The active waypoint is green; the final goal remains yellow.
        waypoint = self.route.current
        gl.glColor3f(0.20, 0.95, 0.35)
        gl.glBegin(gl.GL_LINES)
        gl.glVertex3f(float(waypoint[0]), 0.0, float(waypoint[1]))
        gl.glVertex3f(float(waypoint[0]), 500.0, float(waypoint[1]))
        gl.glEnd()

        gl.glColor3f(0.45, 0.85, 0.35)
        gl.glBegin(gl.GL_LINE_STRIP)
        gl.glVertex3f(float(self.position[0]), 2.0, float(self.position[1]))
        for point in self.route.points[self.route.index:]:
            gl.glVertex3f(float(point[0]), 2.0, float(point[1]))
        gl.glEnd()

        gl.glColor3f(1.0, 0.75, 0.05)
        gl.glBegin(gl.GL_LINES)
        gl.glVertex3f(float(self.target[0]), 0.0, float(self.target[1]))
        gl.glVertex3f(float(self.target[0]), 800.0, float(self.target[1]))
        gl.glEnd()

        gl.glDisable(gl.GL_DEPTH_TEST)
        gl.glMatrixMode(gl.GL_PROJECTION)
        gl.glLoadIdentity()
        gl.glOrtho(0.0, float(self.width), 0.0, float(self.height), -1.0, 1.0)
        gl.glMatrixMode(gl.GL_MODELVIEW)
        gl.glLoadIdentity()
        self.label.draw()
        self.help_label.draw()

    def close(self):
        pyglet.clock.unschedule(self.update)
        self.navigator.close()
        super().close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--delay-ms", type=float, default=250.0)
    parser.add_argument("--backend", choices=["torch", "native"], default="native")
    parser.add_argument("--weights", default=str(BASE / "pipeline_out" / "policy_weights_v5_macro_supervised.pth"))
    parser.add_argument("--library", default=str(BASE / "pipeline_out" / "libncnn_rust_v5_macro_supervised.so"))
    parser.add_argument("--param", default=str(BASE / "pipeline_out" / "policy_v5_macro_supervised.param"))
    parser.add_argument("--bin", dest="bin_path", default=str(BASE / "pipeline_out" / "policy_v5_macro_supervised.bin"))
    args = parser.parse_args()

    if args.backend == "torch":
        navigator = TorchNavigator(Path(args.weights))
    else:
        navigator = NativeNavigator(Path(args.library), Path(args.param), Path(args.bin_path))
    viewer = Viewer(navigator, args.delay_ms, config=pyglet.gl.Config(double_buffer=True, depth_size=24))
    print("V5 轻量 3D 测试已启动，关闭窗口或按 Esc/Q 结束。", flush=True)
    print(
        f"逻辑地图=2250x2250 物理地图={PHYSICAL_WORLD_SIZE:.0f}x{PHYSICAL_WORLD_SIZE:.0f} "
        f"障碍物={len(OBSTACLES)} waypoint最大段={WAYPOINT_MAX_LEG:.0f} 定位延迟={args.delay_ms:.0f}ms",
        flush=True,
    )
    try:
        pyglet.app.run()
    finally:
        if getattr(viewer.navigator, "state", None):
            viewer.navigator.close()


if __name__ == "__main__":
    main()
