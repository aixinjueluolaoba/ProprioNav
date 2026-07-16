from __future__ import annotations

import math
from dataclasses import dataclass, field

import gymnasium as gym
import numpy as np
from gymnasium import spaces

JUMP_TRIGGER_THRESHOLD = 0.5
DEFAULT_DELTA_ANGLE_LIMIT = math.radians(90.0)


@dataclass(frozen=True)
class TreeObstacle:
    x: float
    y: float
    radius: float
    center: np.ndarray = field(init=False, repr=False)
    aabb_min: np.ndarray = field(init=False, repr=False)
    aabb_max: np.ndarray = field(init=False, repr=False)
    radius_sq: float = field(init=False, repr=False)

    def __post_init__(self) -> None:
        center = np.array([self.x, self.y], dtype=np.float32)
        radius = float(self.radius)
        object.__setattr__(self, "center", center)
        object.__setattr__(self, "aabb_min", center - radius)
        object.__setattr__(self, "aabb_max", center + radius)
        object.__setattr__(self, "radius_sq", radius * radius)


@dataclass(frozen=True)
class MountainObstacle:
    vertices: np.ndarray
    aabb_min: np.ndarray = field(init=False, repr=False)
    aabb_max: np.ndarray = field(init=False, repr=False)
    edge_starts: np.ndarray = field(init=False, repr=False)
    edge_ends: np.ndarray = field(init=False, repr=False)
    edges: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        vertices = np.asarray(self.vertices, dtype=np.float32)
        edge_ends = np.roll(vertices, shift=-1, axis=0)
        object.__setattr__(self, "vertices", vertices)
        object.__setattr__(self, "aabb_min", vertices.min(axis=0))
        object.__setattr__(self, "aabb_max", vertices.max(axis=0))
        object.__setattr__(self, "edge_starts", vertices)
        object.__setattr__(self, "edge_ends", edge_ends)
        object.__setattr__(self, "edges", edge_ends - vertices)


class BlindNavEnv(gym.Env):
    """Blind target navigation with hidden circular obstacles.

    The policy observes target-relative motion history, not obstacle positions.
    Hidden obstacles simulate trees/mountains that can trap the player until the
    policy changes direction or triggers jump.
    """

    metadata = {"render_modes": ["ansi"], "render_fps": 20}

    def __init__(
        self,
        *,
        world_size: float = 2400.0,
        target_radius: float = 35.0,
        max_steps: int = 700,
        dt: float = 0.3,
        tree_count: int = 34,
        mountain_count: int = 8,
        tree_radius: float = 28.0,
        tree_radius_range: tuple[float, float] = (18.0, 42.0),
        mountain_radius_range: tuple[float, float] = (85.0, 220.0),
        concave_mountain_probability: float = 0.65,
        deep_concave_mountain_probability: float = 0.0,
        max_turn_rate: float = math.radians(160.0),
        action_mode: str = "target_offset",
        delta_angle_limit: float = DEFAULT_DELTA_ANGLE_LIMIT,
        tracking_residual_limit: float = math.radians(20.0),
        macro_recovery_steps: int = 4,
        fixed_recovery_back_duration: float = 2.0,
        fixed_recovery_side_duration: float = 2.0,
        fixed_recovery_trigger_threshold: float = 0.1,
        fixed_recovery_cooldown_duration: float = 0.0,
        fixed_recovery_policy_side_control: bool = False,
        fixed_recovery_auto_trigger: bool = False,
        fixed_recovery_auto_no_progress_threshold: float = 0.8,
        fixed_recovery_auto_collision_threshold: float = 0.8,
        allow_zero_speed: bool = False,
        binary_speed: bool = False,
        random_initial_heading: bool = False,
        stop_distance_range: tuple[float, float] | None = None,
        start_target_distance_range: tuple[float, float] | None = None,
        stop_distance_tolerance: float = 3.0,
        align_angle_tolerance: float = math.radians(10.0),
        stop_speed_threshold: float = 12.0,
        reward_profile: str = "default",
        position_obs_dt_range: tuple[float, float] | None = None,
        coord_noise_std: float = 0.0,
        trapped_start_probability: float = 0.0,
        opposite_heading_probability: float = 0.0,
        use_spatial_grid: bool = True,
        use_collision_aabb_filter: bool = True,
        collision_slide_enabled: bool = True,
        grid_cell_size: float = 256.0,
        collect_collision_stats: bool = False,
        reward_angle_penalty_scale: float = 1.0,
        reward_heading_change_penalty_scale: float = 1.0,
        reward_action_change_penalty_scale: float = 1.0,
        reward_close_align_bonus_scale: float = 1.0,
        reward_terminal_align_bonus_scale: float = 1.0,
        reward_recovery_bonus_scale: float = 1.0,
        reward_recovery_collision_bonus_scale: float = 1.0,
        reward_recovery_backward_bonus_scale: float = 1.0,
        reward_recovery_stuck_bonus_scale: float = 1.0,
        reward_recovery_progress_bonus_scale: float = 1.0,
        reward_recovery_clear_bonus_scale: float = 1.0,
        reward_recovery_negative_progress_penalty_scale: float = 1.0,
        reward_recovery_switch_penalty_scale: float = 1.0,
        reward_recovery_premature_penalty_scale: float = 1.0,
        reward_recovery_idle_penalty_scale: float = 1.0,
        reward_recovery_clear_penalty_scale: float = 1.0,
        reward_non_recovery_angle_penalty_scale: float = 1.0,
        reward_non_recovery_heading_penalty_scale: float = 1.0,
        reward_non_recovery_action_change_penalty_scale: float = 1.0,
        macro_recovery_clear_time_threshold: float = 0.65,
        macro_recovery_clear_no_progress_threshold: float = 0.20,
        macro_recovery_progress_gain_threshold: float = 5.0,
        macro_recovery_progress_step_threshold: float = 0.8,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        self.world_size = float(world_size)
        self.target_radius = float(target_radius)
        self.max_steps = int(max_steps)
        self.dt = float(dt)
        self.tree_count = int(tree_count)
        self.mountain_count = int(mountain_count)
        self.tree_radius = float(tree_radius)
        self.tree_radius_range = tree_radius_range
        self.mountain_radius_range = mountain_radius_range
        self.concave_mountain_probability = float(concave_mountain_probability)
        self.deep_concave_mountain_probability = float(deep_concave_mountain_probability)
        self.max_turn_rate = float(max_turn_rate)
        self.action_mode = str(action_mode)
        self.delta_angle_limit = float(delta_angle_limit)
        self.tracking_residual_limit = float(tracking_residual_limit)
        self.macro_recovery_steps = max(1, int(macro_recovery_steps))
        self.fixed_recovery_back_duration = float(fixed_recovery_back_duration)
        self.fixed_recovery_side_duration = float(fixed_recovery_side_duration)
        self.fixed_recovery_trigger_threshold = float(fixed_recovery_trigger_threshold)
        self.fixed_recovery_cooldown_duration = float(fixed_recovery_cooldown_duration)
        self.fixed_recovery_policy_side_control = bool(fixed_recovery_policy_side_control)
        self.fixed_recovery_auto_trigger = bool(fixed_recovery_auto_trigger)
        self.fixed_recovery_auto_no_progress_threshold = float(fixed_recovery_auto_no_progress_threshold)
        self.fixed_recovery_auto_collision_threshold = float(fixed_recovery_auto_collision_threshold)
        self.fixed_recovery_back_steps = max(1, int(math.ceil(self.fixed_recovery_back_duration / max(self.dt, 1e-6))))
        self.fixed_recovery_side_steps = max(1, int(math.ceil(self.fixed_recovery_side_duration / max(self.dt, 1e-6))))
        self.fixed_recovery_total_steps = self.fixed_recovery_back_steps + self.fixed_recovery_side_steps
        self.allow_zero_speed = bool(allow_zero_speed)
        self.binary_speed = bool(binary_speed)
        self.random_initial_heading = bool(random_initial_heading)
        self.stop_distance_range = stop_distance_range
        self.start_target_distance_range = start_target_distance_range
        self.stop_distance_tolerance = float(stop_distance_tolerance)
        self.align_angle_tolerance = float(align_angle_tolerance)
        self.stop_speed_threshold = float(stop_speed_threshold)
        self.reward_profile = str(reward_profile)
        self.position_obs_dt_range = position_obs_dt_range
        self.coord_noise_std = float(coord_noise_std)
        self.trapped_start_probability = float(trapped_start_probability)
        self.opposite_heading_probability = float(opposite_heading_probability)
        self.use_spatial_grid = bool(use_spatial_grid)
        self.use_collision_aabb_filter = bool(use_collision_aabb_filter)
        self.collision_slide_enabled = bool(collision_slide_enabled)
        self.grid_cell_size = float(grid_cell_size)
        self.collect_collision_stats = bool(collect_collision_stats)
        self.reward_angle_penalty_scale = float(reward_angle_penalty_scale)
        self.reward_heading_change_penalty_scale = float(reward_heading_change_penalty_scale)
        self.reward_action_change_penalty_scale = float(reward_action_change_penalty_scale)
        self.reward_close_align_bonus_scale = float(reward_close_align_bonus_scale)
        self.reward_terminal_align_bonus_scale = float(reward_terminal_align_bonus_scale)
        self.reward_recovery_bonus_scale = float(reward_recovery_bonus_scale)
        self.reward_recovery_collision_bonus_scale = float(reward_recovery_collision_bonus_scale)
        self.reward_recovery_backward_bonus_scale = float(reward_recovery_backward_bonus_scale)
        self.reward_recovery_stuck_bonus_scale = float(reward_recovery_stuck_bonus_scale)
        self.reward_recovery_progress_bonus_scale = float(reward_recovery_progress_bonus_scale)
        self.reward_recovery_clear_bonus_scale = float(reward_recovery_clear_bonus_scale)
        self.reward_recovery_negative_progress_penalty_scale = float(reward_recovery_negative_progress_penalty_scale)
        self.reward_recovery_switch_penalty_scale = float(reward_recovery_switch_penalty_scale)
        self.reward_recovery_premature_penalty_scale = float(reward_recovery_premature_penalty_scale)
        self.reward_recovery_idle_penalty_scale = float(reward_recovery_idle_penalty_scale)
        self.reward_recovery_clear_penalty_scale = float(reward_recovery_clear_penalty_scale)
        self.reward_non_recovery_angle_penalty_scale = float(reward_non_recovery_angle_penalty_scale)
        self.reward_non_recovery_heading_penalty_scale = float(reward_non_recovery_heading_penalty_scale)
        self.reward_non_recovery_action_change_penalty_scale = float(reward_non_recovery_action_change_penalty_scale)
        self.macro_recovery_clear_time_threshold = float(macro_recovery_clear_time_threshold)
        self.macro_recovery_clear_no_progress_threshold = float(macro_recovery_clear_no_progress_threshold)
        self.macro_recovery_progress_gain_threshold = float(macro_recovery_progress_gain_threshold)
        self.macro_recovery_progress_step_threshold = float(macro_recovery_progress_step_threshold)

        # action = [angle_offset_norm, speed_norm, jump_norm]
        # angle offset is relative to target direction: [-1, 1] -> [-pi, pi]
        # speed is [0, 1] -> [50, 100]
        # jump is [-1, 1], values above threshold trigger jump/down action.
        # target_relative_recovery adds action[3] as a recovery mode selector:
        # back-left, back, normal, or back-right relative to the target line.
        # target_relative_macro_recovery uses the same selector, but holds
        # non-normal recovery choices for several steps to make exploration easier.
        # target_relative_macro_trigger_recovery keeps normal movement as the
        # default and only enters macro recovery when action[3] is high.
        # heading_relative_two_phase_macro_recovery anchors recovery to the
        # current heading, then runs a back-first two-phase escape macro.
        # heading_relative_macro_library_recovery lets a discrete wrapper pick
        # one of several fixed escape templates with different durations.
        # continuous_trigger_fixed_recovery keeps normal movement continuous and
        # only lets the policy trigger a fixed jump/back/side recovery template.
        action_low = [-1.0, 0.0, -1.0]
        action_high = [1.0, 1.0, 1.0]
        if self.action_mode in {
            "target_relative_recovery",
            "target_relative_macro_recovery",
            "target_relative_macro_trigger_recovery",
            "heading_relative_two_phase_macro_recovery",
            "continuous_trigger_fixed_recovery",
        }:
            action_low.append(-1.0)
            action_high.append(1.0)
        if self.action_mode == "heading_relative_macro_library_recovery":
            action_low.extend([-1.0, -1.0])
            action_high.extend([1.0, 1.0])
        self.action_space = spaces.Box(
            low=np.array(action_low, dtype=np.float32),
            high=np.array(action_high, dtype=np.float32),
            dtype=np.float32,
        )

        # No obstacle/ray data here: this is intentionally blind.
        # Values are normalized to approximately [-1, 1].
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(17,), dtype=np.float32)

        self.np_random: np.random.Generator
        self.obstacles: list[TreeObstacle | MountainObstacle] = []
        self._obstacle_grid: dict[tuple[int, int], list[int]] = {}
        self.collision_stats = {
            "collision_steps": 0,
            "broadphase_candidates": 0,
            "tree_precise_checks": 0,
            "mountain_precise_checks": 0,
        }
        self._forced_start_mountain: MountainObstacle | None = None
        self.pos = np.zeros(2, dtype=np.float32)
        self.prev_pos = np.zeros(2, dtype=np.float32)
        self.target = np.zeros(2, dtype=np.float32)
        self.velocity = np.zeros(2, dtype=np.float32)
        self.last_action = np.zeros(self.action_space.shape, dtype=np.float32)
        self.heading = 0.0
        self.stick_angle = 0.0
        self.desired_stop_distance = self.target_radius
        self.step_count = 0
        self.stuck_time = 0.0
        self.no_progress_time = 0.0
        self.jump_threshold = JUMP_TRIGGER_THRESHOLD
        self.jump_cooldown = 0.0
        self.time_since_collision = 5.0
        self._prev_distance = 0.0
        self._last_progress = 0.0
        self._was_stuck = False
        self.recovery_mode = 0
        self.macro_recovery_mode = 0
        self.macro_recovery_steps_left = 0
        self.macro_recovery_total_steps = 0
        self.macro_recovery_anchor_heading = 0.0
        self.macro_recovery_started_stuck = False
        self.macro_recovery_progress_gain = 0.0
        self.fixed_recovery_cooldown_left = 0.0
        self.reset(seed=seed)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        del options
        self.step_count = 0
        self.stuck_time = 0.0
        self.no_progress_time = 0.0
        self.jump_cooldown = 0.0
        self.time_since_collision = 5.0
        self.velocity.fill(0.0)
        self.last_action.fill(0.0)
        self._last_progress = 0.0
        self._was_stuck = False
        self.recovery_mode = 0
        self.macro_recovery_mode = 0
        self.macro_recovery_steps_left = 0
        self.macro_recovery_total_steps = 0
        self.macro_recovery_anchor_heading = self.heading
        self.macro_recovery_started_stuck = False
        self.macro_recovery_progress_gain = 0.0
        self.fixed_recovery_cooldown_left = 0.0

        if self.stop_distance_range is None:
            self.desired_stop_distance = self.target_radius
        else:
            low, high = self.stop_distance_range
            self.desired_stop_distance = float(self.np_random.uniform(low, high))
        self.pos, self.target = self._sample_start_and_target()
        if self.random_initial_heading:
            target_heading = math.atan2(float(self.target[1] - self.pos[1]), float(self.target[0] - self.pos[0]))
            if float(self.np_random.random()) < self.opposite_heading_probability:
                self.heading = self._wrap_angle(target_heading + math.pi + float(self.np_random.uniform(-0.35, 0.35)))
            else:
                self.heading = float(self.np_random.uniform(-math.pi, math.pi))
        else:
            self.heading = math.atan2(float(self.target[1] - self.pos[1]), float(self.target[0] - self.pos[0]))
        self.stick_angle = self.heading
        self.prev_pos = self.pos.copy()
        self.obstacles = self._sample_obstacles()
        self._build_obstacle_grid()
        self._prev_distance = self._distance_to_target()
        return self._get_obs(), self._info()

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        action = np.asarray(action, dtype=np.float32)
        if action.shape != self.action_space.shape:
            if self.action_mode == "heading_relative_macro_library_recovery" and action.shape == (4,) and self.action_space.shape == (5,):
                action = np.concatenate([action, np.array([0.0], dtype=np.float32)], axis=0)
            else:
                raise ValueError(f"expected action shape {self.action_space.shape}, got {action.shape}")
        action = np.clip(action, self.action_space.low, self.action_space.high)
        prev_action = self.last_action.copy()
        self.last_action = action
        self.step_count += 1

        prev_distance = self._distance_to_target()
        prev_stuck = self._is_stuck()
        prev_heading = self.heading
        self.prev_pos = self.pos.copy()

        target_angle = math.atan2(self.target[1] - self.pos[1], self.target[0] - self.pos[0])
        prev_recovery_mode = self.recovery_mode
        effective_recovery_mode = 0
        forced_jump = False
        recovery_started_this_step = False

        if self.action_mode == "target_offset":
            desired_angle = target_angle + float(action[0]) * math.pi
        elif self.action_mode == "target_relative_offset":
            desired_angle = target_angle + float(action[0]) * self.delta_angle_limit
        elif self.action_mode == "target_relative_full":
            desired_angle = target_angle + float(action[0]) * math.pi
        elif self.action_mode == "target_relative_recovery":
            effective_recovery_mode = self._recovery_signal_to_mode(float(action[3]))
            if effective_recovery_mode == -2:
                desired_angle = target_angle + math.radians(135.0)
            elif effective_recovery_mode == -1:
                desired_angle = target_angle + math.pi
            elif effective_recovery_mode == 0:
                desired_angle = target_angle + float(action[0]) * self.delta_angle_limit
            else:
                desired_angle = target_angle - math.radians(135.0)
        elif self.action_mode == "target_relative_macro_recovery":
            requested_recovery_mode = self._recovery_signal_to_mode(float(action[3]))
            if self.macro_recovery_steps_left <= 0 and requested_recovery_mode != 0:
                self.macro_recovery_mode = requested_recovery_mode
                self.macro_recovery_steps_left = self.macro_recovery_steps
                self.macro_recovery_anchor_heading = self.heading
                self.macro_recovery_started_stuck = prev_stuck
                self.macro_recovery_progress_gain = 0.0
            if self.macro_recovery_steps_left > 0:
                effective_recovery_mode = self.macro_recovery_mode
            else:
                effective_recovery_mode = 0
                self.macro_recovery_mode = 0
            if effective_recovery_mode == -2:
                desired_angle = target_angle + math.radians(135.0)
            elif effective_recovery_mode == -1:
                desired_angle = target_angle + math.pi
            elif effective_recovery_mode == 2:
                desired_angle = target_angle - math.radians(135.0)
            else:
                desired_angle = target_angle + float(action[0]) * self.delta_angle_limit
        elif self.action_mode == "target_relative_macro_trigger_recovery":
            requested_recovery_mode = self._macro_trigger_to_mode(float(action[3]), float(action[0]))
            if self.macro_recovery_steps_left <= 0 and requested_recovery_mode != 0:
                self.macro_recovery_mode = requested_recovery_mode
                self.macro_recovery_steps_left = self.macro_recovery_steps
                self.macro_recovery_anchor_heading = self.heading
                self.macro_recovery_started_stuck = prev_stuck
                self.macro_recovery_progress_gain = 0.0
            if self.macro_recovery_steps_left > 0:
                effective_recovery_mode = self.macro_recovery_mode
            else:
                effective_recovery_mode = 0
                self.macro_recovery_mode = 0
            if effective_recovery_mode == -2:
                desired_angle = target_angle + math.radians(135.0)
            elif effective_recovery_mode == -1:
                desired_angle = target_angle + math.pi
            elif effective_recovery_mode == 2:
                desired_angle = target_angle - math.radians(135.0)
            else:
                desired_angle = target_angle + float(action[0]) * self.delta_angle_limit
        elif self.action_mode == "heading_relative_two_phase_macro_recovery":
            requested_recovery_mode = self._recovery_signal_to_mode(float(action[3]))
            if self.macro_recovery_steps_left <= 0 and requested_recovery_mode != 0:
                self.macro_recovery_mode = requested_recovery_mode
                self.macro_recovery_steps_left = self.macro_recovery_steps
                self.macro_recovery_anchor_heading = self.heading
                self.macro_recovery_started_stuck = prev_stuck
                self.macro_recovery_progress_gain = 0.0
            if self.macro_recovery_steps_left > 0:
                effective_recovery_mode = self.macro_recovery_mode
                desired_angle = self._heading_relative_macro_desired_angle(effective_recovery_mode)
            else:
                effective_recovery_mode = 0
                self.macro_recovery_mode = 0
                desired_angle = target_angle + float(action[0]) * self.delta_angle_limit
        elif self.action_mode == "heading_relative_macro_library_recovery":
            requested_recovery_mode = self._macro_library_signal_to_mode(float(action[3]))
            if self.macro_recovery_steps_left <= 0 and requested_recovery_mode != 0:
                self.macro_recovery_mode = requested_recovery_mode
                self.macro_recovery_total_steps = self._macro_library_duration(
                    requested_recovery_mode,
                    requested_duration_scale=float(action[4]) if len(action) > 4 else None,
                )
                self.macro_recovery_steps_left = self.macro_recovery_total_steps
                self.macro_recovery_anchor_heading = self.heading
                self.macro_recovery_started_stuck = prev_stuck
                self.macro_recovery_progress_gain = 0.0
            if self.macro_recovery_steps_left > 0:
                effective_recovery_mode = self.macro_recovery_mode
                desired_angle = self._macro_library_desired_angle(effective_recovery_mode, target_angle)
            else:
                effective_recovery_mode = 0
                self.macro_recovery_mode = 0
                desired_angle = target_angle + float(action[0]) * self.delta_angle_limit
        elif self.action_mode == "continuous_trigger_fixed_recovery":
            recent_collision_norm = 1.0 - np.clip(self.time_since_collision / 3.0, 0.0, 1.0)
            if self.fixed_recovery_auto_trigger:
                trigger_recovery = (
                    self.fixed_recovery_cooldown_left <= 0.0
                    and (
                        self.no_progress_time >= self.fixed_recovery_auto_no_progress_threshold
                        or recent_collision_norm >= self.fixed_recovery_auto_collision_threshold
                    )
                )
            else:
                trigger_recovery = (
                    float(action[3]) >= self.fixed_recovery_trigger_threshold and self.fixed_recovery_cooldown_left <= 0.0
                )
            if self.macro_recovery_steps_left <= 0 and trigger_recovery:
                if self.fixed_recovery_policy_side_control:
                    self.macro_recovery_mode = -2 if float(action[2]) >= 0.0 else 2
                else:
                    self.macro_recovery_mode = -2 if float(self.np_random.random()) < 0.5 else 2
                self.macro_recovery_steps_left = self.fixed_recovery_total_steps
                self.macro_recovery_anchor_heading = self.heading
                self.macro_recovery_started_stuck = prev_stuck
                self.macro_recovery_progress_gain = 0.0
                recovery_started_this_step = True
            if self.macro_recovery_steps_left > 0:
                effective_recovery_mode = self.macro_recovery_mode
                phase_progress = self.fixed_recovery_total_steps - self.macro_recovery_steps_left
                anchor = self.macro_recovery_anchor_heading
                if phase_progress < self.fixed_recovery_back_steps:
                    desired_angle = anchor + math.pi
                    forced_jump = phase_progress == 0
                elif effective_recovery_mode == -2:
                    desired_angle = anchor + math.radians(90.0)
                else:
                    desired_angle = anchor - math.radians(90.0)
            else:
                effective_recovery_mode = 0
                self.macro_recovery_mode = 0
                desired_angle = target_angle + float(action[0]) * self.delta_angle_limit
        elif self.action_mode == "stick_delta":
            desired_angle = self.heading + float(action[0]) * self.delta_angle_limit
        elif self.action_mode == "stick_angle_delta":
            self.stick_angle = self._wrap_angle(self.stick_angle + float(action[0]) * self.delta_angle_limit)
            desired_angle = self.stick_angle
        elif self.action_mode == "target_tracking":
            target_delta = math.atan2(math.sin(target_angle - self.heading), math.cos(target_angle - self.heading))
            tracking_delta = float(np.clip(target_delta, -self.delta_angle_limit, self.delta_angle_limit))
            residual_delta = float(action[0]) * self.tracking_residual_limit
            desired_delta = float(np.clip(tracking_delta + residual_delta, -self.delta_angle_limit, self.delta_angle_limit))
            desired_angle = self.heading + desired_delta
        else:
            raise ValueError(f"Unsupported action_mode: {self.action_mode}")
        self.recovery_mode = effective_recovery_mode
        if self.action_mode in {
            "target_relative_macro_recovery",
            "target_relative_macro_trigger_recovery",
            "heading_relative_two_phase_macro_recovery",
            "heading_relative_macro_library_recovery",
            "continuous_trigger_fixed_recovery",
        } and self.macro_recovery_steps_left > 0:
            self.macro_recovery_steps_left -= 1
            if self.macro_recovery_steps_left <= 0:
                self.macro_recovery_mode = 0
                self.macro_recovery_total_steps = 0
                self.macro_recovery_anchor_heading = self.heading
                self.macro_recovery_started_stuck = False
                self.macro_recovery_progress_gain = 0.0
                if self.action_mode == "continuous_trigger_fixed_recovery":
                    self.fixed_recovery_cooldown_left = self.fixed_recovery_cooldown_duration
        max_turn = self.max_turn_rate * self.dt
        self.heading = self._move_angle_towards(self.heading, desired_angle, max_turn)
        if self.allow_zero_speed:
            speed = float(action[1]) * 100.0
        elif self.binary_speed:
            speed = 100.0 if float(action[1]) >= 0.5 else 50.0
        else:
            speed = 50.0 + float(action[1]) * 50.0
        policy_jump_attempted = bool(action[2] > self.jump_threshold)
        jump_attempted = bool(policy_jump_attempted or forced_jump)
        jump = bool(forced_jump or (policy_jump_attempted and self.jump_cooldown <= 0.0))

        intended_delta = np.array(
            [math.cos(self.heading), math.sin(self.heading)],
            dtype=np.float32,
        ) * speed * self.dt

        if jump:
            self.jump_cooldown = 0.8
            new_pos = self.pos + intended_delta * 1.35
        else:
            new_pos = self.pos + intended_delta

        new_pos = np.clip(new_pos, -self.world_size, self.world_size)
        resolved_pos, collided = self._resolve_collision(new_pos, intended_delta, jump)
        self.pos = resolved_pos.astype(np.float32)
        if self.stop_distance_range is None:
            stop_point = self._first_target_circle_point(self.prev_pos, self.pos)
            if stop_point is not None:
                self.pos = stop_point
        else:
            stop_point = self._first_stop_region_point(self.prev_pos, self.pos)
            if stop_point is not None:
                self.pos = stop_point
        self.velocity = (self.pos - self.prev_pos) / self.dt
        if collided:
            self.time_since_collision = 0.0
        else:
            self.time_since_collision = min(self.time_since_collision + self.dt, 10.0)

        displacement = float(np.linalg.norm(self.pos - self.prev_pos))
        displacement_vec = self.pos - self.prev_pos
        current_distance = self._distance_to_target()
        progress = prev_distance - current_distance
        self._last_progress = progress
        if effective_recovery_mode != 0:
            self.macro_recovery_progress_gain = max(0.0, self.macro_recovery_progress_gain + progress)
        if prev_distance > 1e-6:
            target_unit = (self.target - self.prev_pos) / prev_distance
            forward_distance = float(np.dot(displacement_vec, target_unit))
            lateral_vec = displacement_vec - target_unit * forward_distance
            lateral_distance = float(np.linalg.norm(lateral_vec))
        else:
            forward_distance = 0.0
            lateral_distance = 0.0

        if displacement < 1.0 or progress < 0.2:
            self.no_progress_time += self.dt
        else:
            self.no_progress_time = max(0.0, self.no_progress_time - self.dt * 2.0)

        if collided or displacement < 0.6:
            self.stuck_time += self.dt
        else:
            self.stuck_time = max(0.0, self.stuck_time - self.dt * 3.0)

        if (
            self.action_mode
            in {
                "heading_relative_two_phase_macro_recovery",
                "heading_relative_macro_library_recovery",
            }
            and self.macro_recovery_steps_left > 0
            and effective_recovery_mode != 0
        ):
            recovered_clear = (
                self.time_since_collision >= self.macro_recovery_clear_time_threshold
                and self.no_progress_time <= self.macro_recovery_clear_no_progress_threshold
            )
            recovered_progress = (
                self.macro_recovery_progress_gain >= self.macro_recovery_progress_gain_threshold
                and progress > self.macro_recovery_progress_step_threshold
            )
            if self.macro_recovery_started_stuck and (recovered_clear or recovered_progress):
                self.macro_recovery_steps_left = 0
                self.macro_recovery_mode = 0
                self.macro_recovery_anchor_heading = self.heading
                self.macro_recovery_started_stuck = False
                self.macro_recovery_progress_gain = 0.0

        self.jump_cooldown = max(0.0, self.jump_cooldown - self.dt)
        self.fixed_recovery_cooldown_left = max(0.0, self.fixed_recovery_cooldown_left - self.dt)

        angle_error = self._angle_error_to_target()
        heading_change = abs(self._angle_delta(prev_heading, self.heading))
        if self.stop_distance_range is None:
            reached = current_distance <= self.target_radius + 1e-4
        else:
            distance_error = abs(current_distance - self.desired_stop_distance)
            reached = distance_error <= self.stop_distance_tolerance
        truncated = self.step_count >= self.max_steps
        terminated = reached

        if self.action_mode in {
            "target_relative_recovery",
            "target_relative_macro_recovery",
            "target_relative_macro_trigger_recovery",
            "heading_relative_two_phase_macro_recovery",
            "continuous_trigger_fixed_recovery",
        }:
            recovery_mode = effective_recovery_mode
        elif self.action_mode == "target_relative_full":
            action_value_for_recovery = float(action[0])
            prev_action_value_for_recovery = float(prev_action[0])
            if action_value_for_recovery < -0.67:
                recovery_mode = -1
            elif action_value_for_recovery > 0.67:
                recovery_mode = 1
            else:
                recovery_mode = 0
            if prev_action_value_for_recovery < -0.67:
                prev_recovery_mode = -1
            elif prev_action_value_for_recovery > 0.67:
                prev_recovery_mode = 1
            else:
                prev_recovery_mode = 0
        else:
            recovery_mode = 0
            prev_recovery_mode = 0

        if self.stop_distance_range is None and self.reward_profile in {
            "target_line_recovery_angle",
            "target_line_recovery_pressure",
            "target_line_full180_pressure",
            "target_line_macro_recovery",
            "target_line_macro_recovery_v6",
            "target_line_macro_recovery_v7",
            "target_line_macro_recovery_v10_stage1",
            "target_line_macro_recovery_v10_stage2",
            "target_line_macro_recovery_v10",
            "target_line_v2_fixed_trigger",
        }:
            target_error = max(current_distance - self.target_radius, 0.0)
            angle_penalty = min(abs(angle_error) / math.pi, 1.0)
            far_weight = float(np.clip(target_error / 420.0, 0.0, 1.0))
            action_value = float(action[0])
            prev_action_value = float(prev_action[0])
            action_change = abs(action_value - prev_action_value)
            action_flip = 1.0 if abs(action_value) > 0.06 and abs(prev_action_value) > 0.06 and action_value * prev_action_value < 0.0 else 0.0
            recovery_active = recovery_mode != 0
            recovery_switch = 1.0 if recovery_mode != prev_recovery_mode else 0.0
            progress_efficiency = progress / max(displacement, 1e-6) if displacement > 0.4 else 0.0
            progress_efficiency = float(np.clip(progress_efficiency, -1.0, 1.0))
            inefficiency = max(1.0 - progress_efficiency, 0.0)
            pressure_profile = self.reward_profile in {
                "target_line_recovery_pressure",
                "target_line_full180_pressure",
                "target_line_macro_recovery",
                "target_line_macro_recovery_v6",
                "target_line_macro_recovery_v7",
                "target_line_macro_recovery_v10_stage1",
                "target_line_macro_recovery_v10_stage2",
                "target_line_macro_recovery_v10",
            }
            v6_profile = self.reward_profile == "target_line_macro_recovery_v6"
            v7_profile = self.reward_profile == "target_line_macro_recovery_v7"
            v10_stage1_profile = self.reward_profile == "target_line_macro_recovery_v10_stage1"
            v10_stage2_profile = self.reward_profile == "target_line_macro_recovery_v10_stage2"
            v10_profile = self.reward_profile == "target_line_macro_recovery_v10"
            fixed_trigger_profile = self.action_mode == "continuous_trigger_fixed_recovery"

            reward = float(np.clip(progress, -40.0, 40.0)) * 0.18 - 0.14 * self.dt
            reward += progress_efficiency * 0.34
            reward -= inefficiency * (0.10 + 0.20 * far_weight)
            reward -= lateral_distance * (0.095 + 0.085 * far_weight)
            if forward_distance < 0.0:
                reward += forward_distance * (0.070 if recovery_active else 0.135)
            angle_penalty_scale = self.reward_angle_penalty_scale * (
                self.reward_non_recovery_angle_penalty_scale if not recovery_active else 1.0
            )
            heading_penalty_scale = self.reward_heading_change_penalty_scale * (
                self.reward_non_recovery_heading_penalty_scale if not recovery_active else 1.0
            )
            action_change_penalty_scale = self.reward_action_change_penalty_scale * (
                self.reward_non_recovery_action_change_penalty_scale if not recovery_active else 1.0
            )
            reward -= angle_penalty * (1.05 + 2.05 * far_weight) * angle_penalty_scale
            reward -= min(heading_change / math.pi, 1.0) * (0.26 + 0.38 * far_weight) * heading_penalty_scale
            reward -= abs(action_value) * (0.06 + 0.20 * far_weight)
            reward -= action_change * (0.07 + 0.12 * far_weight) * action_change_penalty_scale
            reward -= action_flip * (0.22 + 0.40 * far_weight)
            reward -= recovery_switch * 0.08
            if pressure_profile and collided and not recovery_active:
                reward -= 0.34
            if pressure_profile and collided and recovery_active:
                reward += 0.10
            if recovery_active and self.no_progress_time < 0.6 and progress > 0.2:
                reward -= 0.03 if pressure_profile else 0.06
            if recovery_active and progress > 1.0 and prev_stuck:
                reward += 0.75 if pressure_profile else 0.35
            if recovery_active and self.no_progress_time > 0.9 and progress > 0.2:
                reward += 0.35 if pressure_profile else 0.12
            if not recovery_active and self.no_progress_time > 1.2 and progress < 0.2:
                reward -= (0.34 if pressure_profile else 0.16) * min(self.no_progress_time, 4.0)
            if current_distance > self.target_radius * 5.0 and angle_penalty < 0.08 and not recovery_active:
                reward += (speed / 100.0) * 0.10
            if target_error <= self.target_radius * 5.0:
                reward += (1.0 - angle_penalty) * 0.28 * self.reward_close_align_bonus_scale
            if self.reward_profile == "target_line_v2_fixed_trigger":
                if collided and not recovery_active:
                    reward -= 0.24
                if recovery_started_this_step:
                    recent_collision_norm = 1.0 - np.clip(self.time_since_collision / 1.5, 0.0, 1.0)
                    soft_trigger_context = self.no_progress_time > 0.45 or recent_collision_norm > 0.55
                    stuck_trigger_context = prev_stuck or self.no_progress_time > 0.75 or recent_collision_norm > 0.85
                    if stuck_trigger_context:
                        reward += 0.42
                    elif soft_trigger_context:
                        reward += 0.04
                    else:
                        reward -= 0.18
                if recovery_active and self.time_since_collision > 0.45 and progress > 0.5:
                    reward += 0.82
                if recovery_active and self.no_progress_time > 1.2 and progress < 0.1:
                    reward -= 0.08
                if recovery_active and self.time_since_collision > 0.8 and self.no_progress_time < 0.2:
                    reward -= 0.12
            if v10_stage1_profile:
                stuck_pressure = min(max(self.no_progress_time - 1.1, 0.0), 2.0)
                if recovery_active:
                    reward -= 0.32
                    if not prev_stuck:
                        reward -= 0.24
                    if self.no_progress_time < 0.9:
                        reward -= 0.26
                if recovery_switch > 0.0:
                    reward -= 0.18
                if prev_stuck and recovery_active and stuck_pressure > 0.0 and progress > 0.8:
                    reward += 0.28
            if v10_stage2_profile:
                stuck_pressure = min(max(self.no_progress_time - 0.85, 0.0), 2.5)
                collision_pressure = 1.0 if collided or self.time_since_collision < 0.55 else 0.0
                if stuck_pressure > 0.0 and not recovery_active:
                    reward -= 0.30 * stuck_pressure
                    reward -= 0.14 * collision_pressure
                if recovery_active and not prev_stuck:
                    reward -= 0.18
                if recovery_active and self.no_progress_time < 0.7:
                    reward -= 0.12
                if recovery_switch > 0.0:
                    reward -= 0.08
                if recovery_active and progress > 0.8 and prev_stuck:
                    reward += 0.40
                if recovery_active and forward_distance < -0.2 and prev_stuck:
                    reward += min(-forward_distance, speed * self.dt) * 0.018
            if v7_profile:
                stuck_pressure = min(max(self.no_progress_time - 0.7, 0.0), 3.0)
                collision_pressure = 1.0 if collided or self.time_since_collision < 0.6 else 0.0
                backward_escape = recovery_active and forward_distance < -0.4
                progress_restart = progress > max(0.6, displacement * 0.12)
                if stuck_pressure > 0.0 and not recovery_active:
                    reward -= 0.52 * stuck_pressure
                    reward -= 0.30 * collision_pressure
                if recovery_active and collision_pressure > 0.0:
                    reward += 0.20
                if backward_escape and stuck_pressure > 0.0:
                    reward += min(-forward_distance, speed * self.dt) * 0.020
                if recovery_active and stuck_pressure > 0.0 and progress > -2.5:
                    reward += 0.18 * min(stuck_pressure, 2.0)
                if recovery_active and progress_restart:
                    reward += 0.70 + 0.22 * min(stuck_pressure, 2.0)
                if prev_stuck and recovery_active and self.time_since_collision > 0.45 and progress > 1.6:
                    reward += 0.85
                if recovery_active and progress < -3.0:
                    reward -= 0.14
            if v6_profile:
                stuck_pressure = min(max(self.no_progress_time - 0.9, 0.0), 3.0)
                collision_pressure = 1.0 if collided or self.time_since_collision < 0.45 else 0.0
                progress_restart = progress > max(0.8, displacement * 0.16)
                if stuck_pressure > 0.0 and not recovery_active:
                    reward -= 0.42 * stuck_pressure
                    reward -= 0.22 * collision_pressure
                if stuck_pressure > 0.0 and recovery_active:
                    reward += 0.18 * min(stuck_pressure, 1.5)
                if recovery_active and progress_restart:
                    reward += 0.55 + 0.18 * min(stuck_pressure, 2.0)
                if prev_stuck and recovery_active and progress > 2.0:
                    reward += 0.65
                if recovery_active and progress < -0.2:
                    reward -= 0.18
                if not recovery_active and collision_pressure > 0.0 and progress < 0.15:
                    reward -= 0.28
            if v10_profile:
                stuck_pressure = min(max(self.no_progress_time - 0.55, 0.0), 3.5)
                collision_pressure = 1.0 if collided or self.time_since_collision < 0.55 else 0.0
                backward_escape = recovery_active and forward_distance < -0.25
                progress_restart = progress > max(0.7, displacement * 0.14)
                if stuck_pressure > 0.0 and not recovery_active:
                    reward -= 0.62 * stuck_pressure
                    reward -= 0.34 * collision_pressure
                if recovery_active and collision_pressure > 0.0:
                    reward += (
                        0.24
                        * self.reward_recovery_bonus_scale
                        * self.reward_recovery_collision_bonus_scale
                    )
                if backward_escape:
                    reward += (
                        min(-forward_distance, speed * self.dt)
                        * 0.032
                        * self.reward_recovery_bonus_scale
                        * self.reward_recovery_backward_bonus_scale
                    )
                if recovery_active and stuck_pressure > 0.0:
                    reward += (
                        0.22
                        * min(stuck_pressure, 2.5)
                        * self.reward_recovery_bonus_scale
                        * self.reward_recovery_stuck_bonus_scale
                    )
                if recovery_active and progress_restart:
                    reward += (
                        (0.95 + 0.28 * min(stuck_pressure, 2.5))
                        * self.reward_recovery_bonus_scale
                        * self.reward_recovery_progress_bonus_scale
                    )
                if prev_stuck and recovery_active and self.time_since_collision > 0.45 and progress > 1.4:
                    reward += (
                        1.05
                        * self.reward_recovery_bonus_scale
                        * self.reward_recovery_clear_bonus_scale
                    )
                if recovery_active and progress < -2.5:
                    reward -= 0.18 * self.reward_recovery_negative_progress_penalty_scale
                if recovery_switch > 0.0:
                    reward -= 0.12 * self.reward_recovery_switch_penalty_scale
                if recovery_active and self.no_progress_time < 0.5 and progress > 0.1:
                    reward -= 0.08 * self.reward_recovery_premature_penalty_scale
                if recovery_active and stuck_pressure <= 0.0 and collision_pressure <= 0.0:
                    reward -= 0.24 * self.reward_recovery_idle_penalty_scale
                if recovery_active and self.time_since_collision > 0.80 and self.no_progress_time < 0.25:
                    reward -= 0.18 * self.reward_recovery_clear_penalty_scale
            if fixed_trigger_profile:
                recent_collision_norm = 1.0 - np.clip(self.time_since_collision / 1.5, 0.0, 1.0)
                soft_trigger_context = self.no_progress_time > 0.45 or recent_collision_norm > 0.55
                stuck_trigger_context = prev_stuck or self.no_progress_time > 0.75 or recent_collision_norm > 0.85
                if stuck_trigger_context and not recovery_active and progress < 0.2:
                    reward -= 0.10 + 0.05 * min(max(self.no_progress_time - 0.6, 0.0), 2.0)
                if recovery_started_this_step:
                    if stuck_trigger_context:
                        reward += 0.65
                    elif soft_trigger_context:
                        reward += 0.08
                    else:
                        reward -= 0.22
                if recovery_active and recent_collision_norm > 0.0:
                    reward += 0.10
                if recovery_active and self.time_since_collision > 0.45 and progress > 0.5:
                    reward += 1.20
                if recovery_active and self.no_progress_time > 1.2 and progress < 0.1:
                    reward -= 0.08
                if recovery_active and not stuck_trigger_context and self.time_since_collision > 0.8 and self.no_progress_time < 0.2:
                    reward -= 0.12
        elif self.stop_distance_range is None and self.reward_profile == "target_line_v2":
            target_error = max(current_distance - self.target_radius, 0.0)
            angle_penalty = min(abs(angle_error) / math.pi, 1.0)
            far_weight = float(np.clip(target_error / 420.0, 0.0, 1.0))
            action_value = float(action[0])
            prev_action_value = float(prev_action[0])
            action_change = abs(action_value - prev_action_value)
            action_flip = 1.0 if abs(action_value) > 0.06 and abs(prev_action_value) > 0.06 and action_value * prev_action_value < 0.0 else 0.0
            progress_efficiency = progress / max(displacement, 1e-6) if displacement > 0.4 else 0.0
            progress_efficiency = float(np.clip(progress_efficiency, -1.0, 1.0))
            inefficiency = max(1.0 - progress_efficiency, 0.0)

            reward = float(np.clip(progress, -40.0, 40.0)) * 0.18 - 0.14 * self.dt
            reward += progress_efficiency * 0.32
            reward -= inefficiency * (0.08 + 0.18 * far_weight)
            reward -= lateral_distance * (0.080 + 0.070 * far_weight)
            if forward_distance < 0.0:
                reward += forward_distance * 0.120
            reward -= angle_penalty * (0.75 + 1.55 * far_weight)
            reward -= min(heading_change / math.pi, 1.0) * (0.28 + 0.42 * far_weight)
            reward -= abs(action_value) * (0.08 + 0.24 * far_weight)
            reward -= action_change * (0.08 + 0.14 * far_weight)
            reward -= action_flip * (0.20 + 0.38 * far_weight)
            if current_distance > self.target_radius * 5.0 and angle_penalty < 0.10:
                reward += (speed / 100.0) * 0.08
            if target_error <= self.target_radius * 5.0:
                reward += (1.0 - angle_penalty) * 0.22
        elif self.stop_distance_range is None and self.reward_profile == "target_line":
            target_error = max(current_distance - self.target_radius, 0.0)
            angle_penalty = min(abs(angle_error) / math.pi, 1.0)
            far_weight = float(np.clip(target_error / 420.0, 0.0, 1.0))
            action_change = abs(float(action[0]) - float(prev_action[0]))

            reward = float(np.clip(progress, -40.0, 40.0)) * 0.13 - 0.18 * self.dt
            reward += forward_distance * 0.040
            reward -= lateral_distance * (0.045 + 0.040 * far_weight)
            if forward_distance < 0.0:
                reward += forward_distance * 0.080
            reward -= angle_penalty * (0.18 + 0.95 * far_weight)
            reward -= action_change * (0.025 + 0.075 * far_weight)
            if current_distance > self.target_radius * 4.0:
                reward += (speed / 100.0) * 0.045
            if target_error <= self.target_radius * 5.0:
                reward += (1.0 - angle_penalty) * 0.12
        elif self.stop_distance_range is None:
            reward = progress * 0.08 - 0.2 * self.dt
        elif self.reward_profile == "quality_stop":
            prev_error = abs(prev_distance - self.desired_stop_distance)
            current_error = abs(current_distance - self.desired_stop_distance)
            ring_progress = prev_error - current_error
            clipped_error = min(current_error, 180.0)
            angle_penalty = min(abs(angle_error) / math.pi, 1.0)
            enforce_alignment = self.align_angle_tolerance < math.pi - 1e-6
            enforce_stop_speed = self.stop_speed_threshold < 99.5
            align_score = 1.0 - angle_penalty
            near_ring = current_error <= self.stop_distance_tolerance * 2.0
            close_ring = current_error <= self.stop_distance_tolerance

            reward = float(np.clip(ring_progress, -40.0, 40.0)) * 0.14 - 0.22 * self.dt
            reward -= clipped_error * 0.008
            if enforce_alignment:
                reward -= angle_penalty * 0.12
            if current_distance < self.desired_stop_distance - self.stop_distance_tolerance:
                reward -= (self.desired_stop_distance - current_distance) * 0.07
            if current_error <= self.stop_distance_tolerance * 4.0:
                reward += 0.10
            if near_ring:
                reward += align_score * 0.35 if enforce_alignment else 0.24
                if enforce_stop_speed:
                    reward -= (speed / 100.0) * 0.40
            aligned_enough = abs(angle_error) <= self.align_angle_tolerance if enforce_alignment else True
            if close_ring and aligned_enough:
                reward += 0.90
                if enforce_stop_speed:
                    reward -= (speed / 100.0) * 0.75
            elif close_ring and enforce_alignment:
                reward += 0.20
                reward -= angle_penalty * 0.25
        elif self.reward_profile == "line_stop":
            prev_error = abs(prev_distance - self.desired_stop_distance)
            current_error = abs(current_distance - self.desired_stop_distance)
            ring_progress = prev_error - current_error
            clipped_error = min(current_error, 180.0)
            angle_penalty = min(abs(angle_error) / math.pi, 1.0)
            enforce_alignment = self.align_angle_tolerance < math.pi - 1e-6
            enforce_stop_speed = self.stop_speed_threshold < 99.5
            align_score = 1.0 - angle_penalty
            near_ring = current_error <= self.stop_distance_tolerance * 2.0
            close_ring = current_error <= self.stop_distance_tolerance
            distance_outside_ring = max(current_distance - self.desired_stop_distance - self.stop_distance_tolerance * 2.0, 0.0)
            far_weight = float(np.clip(distance_outside_ring / 420.0, 0.0, 1.0))
            action_change = abs(float(action[0]) - float(prev_action[0]))

            reward = float(np.clip(ring_progress, -40.0, 40.0)) * 0.10 - 0.20 * self.dt
            reward += forward_distance * 0.035
            reward -= lateral_distance * (0.050 + 0.040 * far_weight)
            if forward_distance < 0.0:
                reward += forward_distance * 0.060
            reward -= clipped_error * 0.006
            if enforce_alignment:
                reward -= angle_penalty * (0.10 + 0.90 * far_weight)
            reward -= abs(float(action[0])) * (0.04 + 0.18 * far_weight)
            reward -= action_change * (0.03 + 0.10 * far_weight)
            if distance_outside_ring > 120.0:
                reward += (speed / 100.0) * 0.08
            if current_distance < self.desired_stop_distance - self.stop_distance_tolerance:
                reward -= (self.desired_stop_distance - current_distance) * 0.08
            if current_error <= self.stop_distance_tolerance * 4.0:
                reward += 0.10
            if near_ring:
                reward += align_score * 0.35 if enforce_alignment else 0.24
                if enforce_stop_speed:
                    reward -= (speed / 100.0) * 0.45
            aligned_enough = abs(angle_error) <= self.align_angle_tolerance if enforce_alignment else True
            if close_ring and aligned_enough:
                reward += 0.95
                if enforce_stop_speed:
                    reward -= (speed / 100.0) * 0.80
            elif close_ring and enforce_alignment:
                reward += 0.20
                reward -= angle_penalty * 0.30
        else:
            prev_error = abs(prev_distance - self.desired_stop_distance)
            current_error = abs(current_distance - self.desired_stop_distance)
            ring_progress = prev_error - current_error
            clipped_error = min(current_error, 160.0)
            angle_penalty = min(abs(angle_error) / math.pi, 1.0)
            enforce_alignment = self.align_angle_tolerance < math.pi - 1e-6
            enforce_stop_speed = self.stop_speed_threshold < 99.5
            align_score = 1.0 - angle_penalty
            near_ring = current_error <= self.stop_distance_tolerance * 2.0
            close_ring = current_error <= self.stop_distance_tolerance
            reward = float(np.clip(ring_progress, -35.0, 35.0)) * 0.10 - 0.2 * self.dt
            reward -= clipped_error * 0.004
            if enforce_alignment:
                reward -= angle_penalty * 0.05
            if current_distance < self.desired_stop_distance - self.stop_distance_tolerance:
                reward -= (self.desired_stop_distance - current_distance) * 0.03
            if current_error <= self.stop_distance_tolerance * 3.0:
                reward += 0.12
            if near_ring:
                reward += align_score * 0.25 if enforce_alignment else 0.20
                if enforce_stop_speed:
                    reward -= (speed / 100.0) * 0.25
            aligned_enough = abs(angle_error) <= self.align_angle_tolerance if enforce_alignment else True
            if close_ring and aligned_enough:
                reward += 0.35
                if enforce_stop_speed:
                    reward -= (speed / 100.0) * 0.45
        if self.reward_profile in {
            "quality_stop",
            "line_stop",
            "target_line",
            "target_line_v2",
            "target_line_recovery_angle",
            "target_line_recovery_pressure",
            "target_line_full180_pressure",
            "target_line_macro_recovery",
            "target_line_macro_recovery_v6",
            "target_line_macro_recovery_v7",
            "target_line_macro_recovery_v10_stage1",
            "target_line_macro_recovery_v10_stage2",
            "target_line_macro_recovery_v10",
        }:
            if collided:
                reward -= 0.22
            if self.no_progress_time > 1.0:
                reward -= 0.08 * min(self.no_progress_time, 5.0)
            if jump_attempted and not prev_stuck:
                reward -= 0.04
            if jump_attempted and not jump:
                reward -= 0.03
            if jump and not prev_stuck:
                reward -= 0.18
            if jump and prev_stuck:
                reward += 0.25
            if prev_stuck and not self._is_stuck() and progress > 0.0:
                reward += 0.80
        else:
            if collided:
                reward -= 0.25
            if self.no_progress_time > 1.0:
                reward -= 0.05 * min(self.no_progress_time, 5.0)
            if jump and not prev_stuck:
                reward -= 0.08
            if jump and prev_stuck:
                reward += 0.5
            if prev_stuck and not self._is_stuck() and progress > 0.0:
                reward += 1.0
        if reached:
            if self.reward_profile in {"quality_stop", "line_stop"} and self.stop_distance_range is not None:
                reward += 35.0
            elif self.reward_profile in {
                "target_line",
                "target_line_v2",
                "target_line_recovery_angle",
                "target_line_recovery_pressure",
                "target_line_full180_pressure",
                "target_line_macro_recovery",
                "target_line_macro_recovery_v6",
                "target_line_macro_recovery_v7",
                "target_line_macro_recovery_v10_stage1",
                "target_line_macro_recovery_v10_stage2",
                "target_line_macro_recovery_v10",
            } and self.stop_distance_range is None:
                align_bonus = 10.0 if self.reward_profile in {
                    "target_line_recovery_angle",
                    "target_line_recovery_pressure",
                    "target_line_full180_pressure",
                    "target_line_macro_recovery",
                    "target_line_macro_recovery_v6",
                    "target_line_macro_recovery_v7",
                    "target_line_macro_recovery_v10_stage1",
                    "target_line_macro_recovery_v10_stage2",
                    "target_line_macro_recovery_v10",
                } else (8.0 if self.reward_profile == "target_line_v2" else 3.0)
                reward += (
                    30.0
                    + max(0.0, 1.0 - min(abs(angle_error) / math.pi, 1.0))
                    * align_bonus
                    * self.reward_terminal_align_bonus_scale
                )
            else:
                reward += 25.0 if self.stop_distance_range is not None else 20.0
        if truncated and not reached:
            if self.stop_distance_range is None and self.reward_profile in {
                "target_line",
                "target_line_v2",
                "target_line_recovery_angle",
                "target_line_recovery_pressure",
                "target_line_full180_pressure",
                "target_line_macro_recovery",
                "target_line_macro_recovery_v6",
                "target_line_macro_recovery_v7",
                "target_line_macro_recovery_v10_stage1",
                "target_line_macro_recovery_v10_stage2",
                "target_line_macro_recovery_v10",
            }:
                reward -= 12.0 + min(max(current_distance - self.target_radius, 0.0), 180.0) * 0.05
                angle_timeout_weight = 5.5 if self.reward_profile in {
                    "target_line_recovery_angle",
                    "target_line_recovery_pressure",
                    "target_line_full180_pressure",
                    "target_line_macro_recovery",
                    "target_line_macro_recovery_v6",
                    "target_line_macro_recovery_v7",
                    "target_line_macro_recovery_v10_stage1",
                    "target_line_macro_recovery_v10_stage2",
                    "target_line_macro_recovery_v10",
                } else (4.0 if self.reward_profile == "target_line_v2" else 2.5)
                reward -= min(abs(angle_error) / math.pi, 1.0) * angle_timeout_weight
            elif self.stop_distance_range is None:
                reward -= 5.0
            else:
                final_error = abs(current_distance - self.desired_stop_distance)
                if self.reward_profile in {"quality_stop", "line_stop"}:
                    reward -= 14.0 + min(final_error, 140.0) * 0.08
                    if self.align_angle_tolerance < math.pi - 1e-6:
                        reward -= min(abs(angle_error) / math.pi, 1.0) * 3.0
                else:
                    reward -= 10.0 + min(final_error, 120.0) * 0.05
                    if self.align_angle_tolerance < math.pi - 1e-6:
                        reward -= min(abs(angle_error) / math.pi, 1.0) * 2.0

        self._prev_distance = current_distance
        self._was_stuck = self._is_stuck()
        return self._get_obs(), float(reward), terminated, truncated, self._info()

    def render(self) -> str:
        return (
            f"step={self.step_count} pos=({self.pos[0]:.1f},{self.pos[1]:.1f}) "
            f"target=({self.target[0]:.1f},{self.target[1]:.1f}) "
            f"dist={self._distance_to_target():.1f} stuck={self.stuck_time:.1f}"
        )

    def _sample_start_and_target(self) -> tuple[np.ndarray, np.ndarray]:
        self._forced_start_mountain = None
        if self.start_target_distance_range is not None:
            min_distance, max_distance = self.start_target_distance_range
        else:
            max_travel = self.max_steps * self.dt * 100.0
            min_distance = max(self.desired_stop_distance + 40.0, min(180.0, max_travel * 0.08))
            max_distance = min(self.world_size * 0.75, max(min_distance + 80.0, max_travel * 0.18))
        if self.trapped_start_probability > 0.0 and float(self.np_random.random()) < self.trapped_start_probability:
            trapped = self._sample_trapped_start_and_target(min_distance, max_distance)
            if trapped is not None:
                pos, target, mountain = trapped
                self._forced_start_mountain = mountain
                return pos, target
        while True:
            pos = self.np_random.uniform(-self.world_size * 0.55, self.world_size * 0.55, size=2)
            target = self.np_random.uniform(-self.world_size * 0.55, self.world_size * 0.55, size=2)
            distance = float(np.linalg.norm(target - pos))
            if min_distance <= distance <= max_distance:
                return pos.astype(np.float32), target.astype(np.float32)

    def _sample_trapped_start_and_target(
        self,
        min_distance: float,
        max_distance: float,
    ) -> tuple[np.ndarray, np.ndarray, MountainObstacle] | None:
        for _ in range(160):
            center = self.np_random.uniform(-self.world_size * 0.45, self.world_size * 0.45, size=2)
            radius = float(self.np_random.uniform(self.mountain_radius_range[0], self.mountain_radius_range[1]))
            mountain = self._make_mountain(center, radius)
            inner_vertex = mountain.vertices[int(self.np_random.integers(0, len(mountain.vertices)))]
            start_dir = inner_vertex - center
            start_norm = float(np.linalg.norm(start_dir))
            if start_norm < 1e-6:
                continue
            start_dir = start_dir / start_norm
            start = center + start_dir * max(radius * 0.34, 14.0)
            if self._point_in_polygon(start.astype(np.float32), mountain.vertices):
                start = center + start_dir * max(radius * 0.44, 18.0)
            if np.any(np.abs(start) > self.world_size * 0.78):
                continue

            escape_dir = start - center
            escape_norm = float(np.linalg.norm(escape_dir))
            if escape_norm < 1e-6:
                continue
            escape_dir = escape_dir / escape_norm
            side = 1.0 if float(self.np_random.random()) < 0.5 else -1.0
            rot = np.array(
                [
                    escape_dir[0] * math.cos(side * math.radians(78.0)) - escape_dir[1] * math.sin(side * math.radians(78.0)),
                    escape_dir[0] * math.sin(side * math.radians(78.0)) + escape_dir[1] * math.cos(side * math.radians(78.0)),
                ],
                dtype=np.float32,
            )
            target_distance = float(self.np_random.uniform(min_distance, max_distance))
            target = start + rot * target_distance
            if np.any(np.abs(target) > self.world_size * 0.82):
                continue
            target_vec = target - start
            if not (min_distance <= float(np.linalg.norm(target_vec)) <= max_distance):
                continue
            line_hit, _ = self._segment_polygon_hit(start.astype(np.float32), target.astype(np.float32), mountain)
            if line_hit is None:
                continue
            return start.astype(np.float32), target.astype(np.float32), mountain
        return None

    def _sample_obstacles(self) -> list[TreeObstacle | MountainObstacle]:
        obstacles: list[TreeObstacle | MountainObstacle] = []
        mountain_disks: list[tuple[np.ndarray, float]] = []
        if self._forced_start_mountain is not None:
            mountain = self._forced_start_mountain
            center = np.mean(mountain.vertices, axis=0)
            radius = float(np.max(np.linalg.norm(mountain.vertices - center, axis=1)))
            obstacles.append(mountain)
            mountain_disks.append((center.astype(np.float32), radius * 1.45))
        # Trees use fixed radius by default. Keeping tree_radius_range only as fallback.
        tree_low, tree_high = self.tree_radius_range
        for _ in range(self.tree_count):
            for _attempt in range(80):
                center = self.np_random.uniform(-self.world_size * 0.9, self.world_size * 0.9, size=2)
                radius = self.tree_radius if self.tree_radius > 0.0 else float(self.np_random.uniform(tree_low, tree_high))
                if np.linalg.norm(center - self.pos) < radius + 140.0:
                    continue
                if np.linalg.norm(center - self.target) < radius + self.target_radius + 80.0:
                    continue
                if self._forced_start_mountain is not None:
                    forced_center = np.mean(self._forced_start_mountain.vertices, axis=0)
                    forced_radius = float(np.max(np.linalg.norm(self._forced_start_mountain.vertices - forced_center, axis=1)))
                    if np.linalg.norm(center - forced_center) < forced_radius + radius + 24.0:
                        continue
                obstacles.append(TreeObstacle(float(center[0]), float(center[1]), radius))
                break
        mountain_low, mountain_high = self.mountain_radius_range
        for _ in range(self.mountain_count):
            for _attempt in range(120):
                center = self.np_random.uniform(-self.world_size * 0.9, self.world_size * 0.9, size=2)
                radius = float(self.np_random.uniform(mountain_low, mountain_high))
                if np.linalg.norm(center - self.pos) < radius + 180.0:
                    continue
                if np.linalg.norm(center - self.target) < radius + self.target_radius + 120.0:
                    continue
                # Mountains cannot overlap. Use a conservative effective radius.
                effective_radius = radius * 1.45
                overlap = False
                for existing_center, existing_effective in mountain_disks:
                    if np.linalg.norm(center - existing_center) < existing_effective + effective_radius + 18.0:
                        overlap = True
                        break
                if overlap:
                    continue
                obstacles.append(self._make_mountain(center, radius))
                mountain_disks.append((center.astype(np.float32), effective_radius))
                break
        return obstacles

    def _make_mountain(self, center: np.ndarray, radius: float) -> MountainObstacle:
        vertex_count = int(self.np_random.integers(6, 11))
        base = self.np_random.uniform(0.0, math.tau)
        angles = np.linspace(0.0, math.tau, vertex_count, endpoint=False) + base
        angles += self.np_random.uniform(-0.18, 0.18, size=vertex_count)
        radii = radius * self.np_random.uniform(0.62, 1.18, size=vertex_count)
        concavity_roll = float(self.np_random.random())
        if concavity_roll < self.deep_concave_mountain_probability:
            notch_start = int(self.np_random.integers(0, vertex_count))
            notch_width = int(self.np_random.integers(2, min(4, vertex_count - 1)))
            for offset in range(notch_width):
                radii[(notch_start + offset) % vertex_count] *= float(self.np_random.uniform(0.12, 0.30))
            shoulder_left = (notch_start - 1) % vertex_count
            shoulder_right = (notch_start + notch_width) % vertex_count
            radii[shoulder_left] *= float(self.np_random.uniform(1.10, 1.35))
            radii[shoulder_right] *= float(self.np_random.uniform(1.10, 1.35))
        # Force concavity often enough so escape behavior is actually trained.
        elif concavity_roll < self.concave_mountain_probability:
            notch_count = int(self.np_random.integers(1, 3))
            notch_indices = self.np_random.choice(vertex_count, size=notch_count, replace=False)
            radii[notch_indices] *= self.np_random.uniform(0.22, 0.45, size=notch_count)
        vertices = np.stack(
            [
                center[0] + np.cos(angles) * radii,
                center[1] + np.sin(angles) * radii,
            ],
            axis=1,
        ).astype(np.float32)
        return MountainObstacle(vertices=vertices)

    def _build_obstacle_grid(self) -> None:
        self._obstacle_grid = {}
        if not self.use_spatial_grid or not self.obstacles:
            return
        for index, obstacle in enumerate(self.obstacles):
            min_cell_x, min_cell_y = self._point_to_cell(obstacle.aabb_min)
            max_cell_x, max_cell_y = self._point_to_cell(obstacle.aabb_max)
            for cell_x in range(min_cell_x, max_cell_x + 1):
                for cell_y in range(min_cell_y, max_cell_y + 1):
                    self._obstacle_grid.setdefault((cell_x, cell_y), []).append(index)

    def _iter_collision_candidates(self, sweep_min: np.ndarray, sweep_max: np.ndarray):
        if not self.use_spatial_grid:
            return range(len(self.obstacles))
        min_cell_x, min_cell_y = self._point_to_cell(sweep_min)
        max_cell_x, max_cell_y = self._point_to_cell(sweep_max)
        seen: set[int] = set()
        ordered: list[int] = []
        for cell_x in range(min_cell_x, max_cell_x + 1):
            for cell_y in range(min_cell_y, max_cell_y + 1):
                for index in self._obstacle_grid.get((cell_x, cell_y), ()):
                    if index not in seen:
                        seen.add(index)
                        ordered.append(index)
        ordered.sort()
        return ordered

    def _point_to_cell(self, point: np.ndarray) -> tuple[int, int]:
        return (
            math.floor(float(point[0]) / self.grid_cell_size),
            math.floor(float(point[1]) / self.grid_cell_size),
        )

    def _resolve_collision(
        self,
        candidate: np.ndarray,
        intended_delta: np.ndarray,
        jump: bool,
    ) -> tuple[np.ndarray, bool]:
        travel_margin = float(np.linalg.norm(intended_delta)) + 12.0
        sweep_min = np.minimum(self.pos, candidate) - travel_margin
        sweep_max = np.maximum(self.pos, candidate) + travel_margin
        candidate_indices = self._iter_collision_candidates(sweep_min, sweep_max)
        if self.collect_collision_stats:
            self.collision_stats["collision_steps"] += 1
            self.collision_stats["broadphase_candidates"] += len(candidate_indices)
        for index in candidate_indices:
            obstacle = self.obstacles[index]
            if self.use_collision_aabb_filter and not self._aabb_intersects(sweep_min, sweep_max, obstacle.aabb_min, obstacle.aabb_max):
                continue
            if isinstance(obstacle, TreeObstacle):
                if self.collect_collision_stats:
                    self.collision_stats["tree_precise_checks"] += 1
                resolved = self._resolve_tree_collision(obstacle, candidate, intended_delta, jump)
            else:
                if self.collect_collision_stats:
                    self.collision_stats["mountain_precise_checks"] += 1
                resolved = self._resolve_mountain_collision(obstacle, candidate, intended_delta, jump)
            if resolved is not None:
                return resolved, True
        return candidate.astype(np.float32), False

    def _resolve_tree_collision(
        self,
        obstacle: TreeObstacle,
        candidate: np.ndarray,
        intended_delta: np.ndarray,
        jump: bool,
    ) -> np.ndarray | None:
        center = obstacle.center
        dx = float(candidate[0] - center[0])
        dy = float(candidate[1] - center[1])
        distance_sq = dx * dx + dy * dy
        segment_hit, hit_point = self._segment_circle_hit(self.pos, candidate, center, obstacle.radius)
        if distance_sq >= obstacle.radius_sq and not segment_hit:
            return None
        contact = candidate if distance_sq < obstacle.radius_sq else hit_point
        direction = contact - center
        direction = direction / max(float(np.linalg.norm(direction)), 1e-6)
        if jump:
            return center + direction * (obstacle.radius + 3.0)
        if not self.collision_slide_enabled:
            return self.pos.copy()
        tangent = np.array([-direction[1], direction[0]], dtype=np.float32)
        tangent_sign = 1.0 if float(np.dot(tangent, intended_delta)) >= 0.0 else -1.0
        slide = tangent * tangent_sign * float(np.linalg.norm(intended_delta)) * 0.45
        resolved = (self.pos + slide).astype(np.float32)
        if np.linalg.norm(resolved - center) < obstacle.radius:
            return self.pos.copy()
        return resolved

    def _resolve_mountain_collision(
        self,
        obstacle: MountainObstacle,
        candidate: np.ndarray,
        intended_delta: np.ndarray,
        jump: bool,
    ) -> np.ndarray | None:
        if self.use_collision_aabb_filter and not self._point_in_aabb(candidate, obstacle.aabb_min, obstacle.aabb_max):
            hit_possible = self._segment_aabb_intersects(self.pos, candidate, obstacle.aabb_min, obstacle.aabb_max)
            if not hit_possible:
                return None
        hit_point, tangent = self._segment_polygon_hit(self.pos, candidate, obstacle)
        candidate_inside = self._point_in_polygon(candidate, obstacle.vertices)
        if hit_point is None and not candidate_inside:
            return None
        nearest, nearest_tangent = self._nearest_polygon_edge(candidate, obstacle.vertices)
        if hit_point is None:
            hit_point = nearest
            tangent = nearest_tangent
        if jump:
            # Jump acts as a strong breakout when trapped in concave shapes.
            return self._push_outside_polygon(candidate, obstacle.vertices, clearance=12.0)
        if not self.collision_slide_enabled:
            return self.pos.copy()
        tangent_norm = tangent / max(float(np.linalg.norm(tangent)), 1e-6)
        tangent_sign = 1.0 if float(np.dot(tangent_norm, intended_delta)) >= 0.0 else -1.0
        slide = tangent_norm * tangent_sign * float(np.linalg.norm(intended_delta)) * 0.38
        resolved = (self.pos + slide).astype(np.float32)
        if self._point_in_polygon(resolved, obstacle.vertices):
            if self.stuck_time > 1.2:
                return self._push_outside_polygon(self.pos, obstacle.vertices, clearance=6.0)
            return self.pos.copy()
        return resolved

    @classmethod
    def _push_outside_polygon(cls, point: np.ndarray, vertices: np.ndarray, clearance: float) -> np.ndarray:
        projection, edge = cls._nearest_polygon_edge(point, vertices)
        normal = np.array([-edge[1], edge[0]], dtype=np.float32)
        normal_norm = float(np.linalg.norm(normal))
        if normal_norm < 1e-6:
            return point.astype(np.float32)
        normal = normal / normal_norm
        candidate_a = projection + normal * clearance
        candidate_b = projection - normal * clearance
        if not cls._point_in_polygon(candidate_a, vertices):
            return candidate_a.astype(np.float32)
        if not cls._point_in_polygon(candidate_b, vertices):
            return candidate_b.astype(np.float32)
        return (projection + normal * (clearance * 2.0)).astype(np.float32)

    @staticmethod
    def _segment_circle_hit(
        start: np.ndarray,
        end: np.ndarray,
        center: np.ndarray,
        radius: float,
    ) -> tuple[bool, np.ndarray]:
        segment = end - start
        length_sq = float(np.dot(segment, segment))
        if length_sq < 1e-6:
            return float(np.linalg.norm(start - center)) < radius, start.astype(np.float32)
        t = float(np.dot(center - start, segment) / length_sq)
        t = float(np.clip(t, 0.0, 1.0))
        closest = start + segment * t
        dx = float(closest[0] - center[0])
        dy = float(closest[1] - center[1])
        return dx * dx + dy * dy < radius * radius, closest.astype(np.float32)

    @staticmethod
    def _segment_polygon_hit(
        start: np.ndarray,
        end: np.ndarray,
        obstacle: MountainObstacle | np.ndarray,
    ) -> tuple[np.ndarray | None, np.ndarray]:
        if isinstance(obstacle, MountainObstacle):
            edge_starts = obstacle.edge_starts
            edge_ends = obstacle.edge_ends
            edges = obstacle.edges
        else:
            edge_starts = obstacle
            edge_ends = np.roll(obstacle, shift=-1, axis=0)
            edges = edge_ends - obstacle
        best_t = float("inf")
        best_point: np.ndarray | None = None
        best_edge = edges[0]
        for i in range(len(edge_starts)):
            a = edge_starts[i]
            b = edge_ends[i]
            hit, t, point = BlindNavEnv._segment_intersection(start, end, a, b)
            if hit and t < best_t:
                best_t = t
                best_point = point
                best_edge = edges[i]
        return best_point, best_edge.astype(np.float32)

    @staticmethod
    def _aabb_intersects(
        a_min: np.ndarray,
        a_max: np.ndarray,
        b_min: np.ndarray,
        b_max: np.ndarray,
    ) -> bool:
        return not (
            float(a_max[0]) < float(b_min[0])
            or float(a_min[0]) > float(b_max[0])
            or float(a_max[1]) < float(b_min[1])
            or float(a_min[1]) > float(b_max[1])
        )

    @staticmethod
    def _point_in_aabb(point: np.ndarray, aabb_min: np.ndarray, aabb_max: np.ndarray) -> bool:
        return (
            float(aabb_min[0]) <= float(point[0]) <= float(aabb_max[0])
            and float(aabb_min[1]) <= float(point[1]) <= float(aabb_max[1])
        )

    @staticmethod
    def _segment_aabb_intersects(
        start: np.ndarray,
        end: np.ndarray,
        aabb_min: np.ndarray,
        aabb_max: np.ndarray,
    ) -> bool:
        x0, y0 = float(start[0]), float(start[1])
        x1, y1 = float(end[0]), float(end[1])
        dx = x1 - x0
        dy = y1 - y0
        t_min = 0.0
        t_max = 1.0
        for origin, delta, box_min, box_max in (
            (x0, dx, float(aabb_min[0]), float(aabb_max[0])),
            (y0, dy, float(aabb_min[1]), float(aabb_max[1])),
        ):
            if abs(delta) < 1e-8:
                if origin < box_min or origin > box_max:
                    return False
                continue
            inv_delta = 1.0 / delta
            t1 = (box_min - origin) * inv_delta
            t2 = (box_max - origin) * inv_delta
            if t1 > t2:
                t1, t2 = t2, t1
            t_min = max(t_min, t1)
            t_max = min(t_max, t2)
            if t_min > t_max:
                return False
        return True

    @staticmethod
    def _segment_intersection(
        p: np.ndarray,
        p2: np.ndarray,
        q: np.ndarray,
        q2: np.ndarray,
    ) -> tuple[bool, float, np.ndarray]:
        r = p2 - p
        s = q2 - q
        denom = float(r[0] * s[1] - r[1] * s[0])
        if abs(denom) < 1e-6:
            return False, 0.0, p.astype(np.float32)
        qp = q - p
        t = float((qp[0] * s[1] - qp[1] * s[0]) / denom)
        u = float((qp[0] * r[1] - qp[1] * r[0]) / denom)
        if 0.0 <= t <= 1.0 and 0.0 <= u <= 1.0:
            return True, t, (p + t * r).astype(np.float32)
        return False, t, p.astype(np.float32)

    @staticmethod
    def _point_in_polygon(point: np.ndarray, vertices: np.ndarray) -> bool:
        x, y = float(point[0]), float(point[1])
        inside = False
        j = len(vertices) - 1
        for i in range(len(vertices)):
            xi, yi = float(vertices[i, 0]), float(vertices[i, 1])
            xj, yj = float(vertices[j, 0]), float(vertices[j, 1])
            denom = yj - yi
            intersects = (yi > y) != (yj > y) and abs(denom) > 1e-6 and x < (xj - xi) * (y - yi) / denom + xi
            if intersects:
                inside = not inside
            j = i
        return inside

    @staticmethod
    def _nearest_polygon_edge(point: np.ndarray, vertices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        best_projection = vertices[0]
        best_edge = vertices[1] - vertices[0]
        best_distance = float("inf")
        for i in range(len(vertices)):
            a = vertices[i]
            b = vertices[(i + 1) % len(vertices)]
            edge = b - a
            t = float(np.dot(point - a, edge) / max(float(np.dot(edge, edge)), 1e-6))
            projection = a + np.clip(t, 0.0, 1.0) * edge
            distance = float(np.linalg.norm(point - projection))
            if distance < best_distance:
                best_distance = distance
                best_projection = projection
                best_edge = edge
        return best_projection.astype(np.float32), best_edge.astype(np.float32)

    def _get_obs(self) -> np.ndarray:
        delta = self.target - self.pos
        distance = float(np.linalg.norm(delta))
        target_angle = math.atan2(float(delta[1]), float(delta[0]))
        speed = float(np.linalg.norm(self.velocity))

        obs = np.array(
            [
                np.clip(delta[0] / self.world_size, -1.0, 1.0),
                np.clip(delta[1] / self.world_size, -1.0, 1.0),
                np.clip(distance / (self.world_size * 2.0), 0.0, 1.0),
                math.sin(target_angle),
                math.cos(target_angle),
                math.sin(self.heading),
                math.cos(self.heading),
                np.clip(speed / 120.0, 0.0, 1.0),
                np.clip(self._last_progress / 30.0, -1.0, 1.0),
                np.clip(self.stuck_time / 5.0, 0.0, 1.0),
                np.clip(self.no_progress_time / 5.0, 0.0, 1.0),
                np.clip(self.time_since_collision / 5.0, 0.0, 1.0),
                1.0 if self._is_stuck() else 0.0,
                np.clip(self.jump_cooldown / 0.8, 0.0, 1.0),
                float(self.last_action[0]),
                float(self.last_action[1] * 2.0 - 1.0),
                float(self.last_action[2]),
            ],
            dtype=np.float32,
        )
        return obs

    def _distance_to_target(self) -> float:
        return float(np.linalg.norm(self.target - self.pos))

    def _is_stuck(self) -> bool:
        return self.stuck_time > 0.4 or self.no_progress_time > 0.9

    @staticmethod
    def _wrap_angle(angle: float) -> float:
        return (angle + math.pi) % math.tau - math.pi

    @staticmethod
    def _recovery_signal_to_mode(recovery_signal: float) -> int:
        if recovery_signal < -0.5:
            return -2
        if recovery_signal < 0.0:
            return -1
        if recovery_signal < 0.5:
            return 0
        return 2

    @staticmethod
    def _macro_trigger_to_mode(trigger_signal: float, direction_signal: float) -> int:
        if trigger_signal < 0.6:
            return 0
        if direction_signal < -0.33:
            return -2
        if direction_signal < 0.33:
            return -1
        return 2

    @staticmethod
    def _macro_library_signal_to_mode(recovery_signal: float) -> int:
        return int(np.clip(round((float(recovery_signal) + 1.0) * 3.5), 0, 7))

    @staticmethod
    def _angle_delta(start: float, end: float) -> float:
        return (end - start + math.pi) % math.tau - math.pi

    @staticmethod
    def _move_angle_towards(current: float, desired: float, max_delta: float) -> float:
        delta = (desired - current + math.pi) % math.tau - math.pi
        delta = float(np.clip(delta, -max_delta, max_delta))
        return (current + delta + math.pi) % math.tau - math.pi

    def _heading_relative_macro_desired_angle(self, recovery_mode: int) -> float:
        anchor = self.macro_recovery_anchor_heading
        phase_progress = self.macro_recovery_steps - self.macro_recovery_steps_left
        phase_split = max(1, self.macro_recovery_steps // 2)
        if recovery_mode == -1:
            return anchor + math.pi
        if phase_progress < phase_split:
            return anchor + math.pi
        if recovery_mode == -2:
            return anchor + math.radians(90.0)
        if recovery_mode == 2:
            return anchor - math.radians(90.0)
        return anchor + math.pi

    @staticmethod
    def _macro_library_base_duration(recovery_mode: int) -> int:
        durations = {
            1: 2,  # short_back
            2: 4,  # back_left
            3: 4,  # back_right
            4: 6,  # wide_left
            5: 6,  # wide_right
            6: 4,  # back_then_realign
            7: 3,  # short_wide_left_then_target
        }
        return durations.get(int(recovery_mode), 0)

    @classmethod
    def _macro_library_duration(cls, recovery_mode: int, requested_duration_scale: float | None = None) -> int:
        base_steps = cls._macro_library_base_duration(recovery_mode)
        if base_steps <= 0:
            return 0
        if requested_duration_scale is None:
            return base_steps
        scale = float(np.clip(requested_duration_scale, -1.0, 1.0))
        if scale < -0.33:
            factor = 0.75
        elif scale > 0.33:
            factor = 1.5
        else:
            factor = 1.0
        return max(1, int(round(base_steps * factor)))

    def _macro_library_desired_angle(self, recovery_mode: int, target_angle: float) -> float:
        mode = int(recovery_mode)
        total_steps = max(1, self.macro_recovery_total_steps or self._macro_library_base_duration(mode))
        phase_progress = total_steps - self.macro_recovery_steps_left
        anchor = self.macro_recovery_anchor_heading
        if mode == 1:
            return anchor + math.pi
        if mode in {2, 3}:
            if phase_progress < 2:
                return anchor + math.pi
            return anchor + (math.radians(90.0) if mode == 2 else -math.radians(90.0))
        if mode in {4, 5}:
            if phase_progress < 2:
                return anchor + math.pi
            return anchor + (math.radians(90.0) if mode == 4 else -math.radians(90.0))
        if mode == 6:
            if phase_progress < 2:
                return anchor + math.pi
            return target_angle
        if mode == 7:
            if phase_progress < 1:
                return anchor + math.pi
            if phase_progress < 3:
                return anchor + math.radians(90.0)
            return target_angle
        return target_angle

    def _info(self) -> dict:
        distance = self._distance_to_target()
        if self.stop_distance_range is None:
            stop_distance_error = max(distance - self.target_radius, 0.0)
        else:
            stop_distance_error = abs(distance - self.desired_stop_distance)
        return {
            "distance": distance,
            "desired_stop_distance": self.desired_stop_distance,
            "stop_distance_error": stop_distance_error,
            "angle_error": self._angle_error_to_target(),
            "is_success": self._is_success(),
            "stuck_time": self.stuck_time,
            "no_progress_time": self.no_progress_time,
            "time_since_collision": self.time_since_collision,
            "step_count": self.step_count,
            "recovery_mode": self.recovery_mode,
            "macro_recovery_steps_left": self.macro_recovery_steps_left,
            "macro_recovery_total_steps": self.macro_recovery_total_steps,
        }

    def _angle_error_to_target(self) -> float:
        delta = self.target - self.pos
        target_angle = math.atan2(float(delta[1]), float(delta[0]))
        return float(math.atan2(math.sin(target_angle - self.heading), math.cos(target_angle - self.heading)))

    def _angle_error_to_target_at(self, point: np.ndarray) -> float:
        delta = self.target - point
        target_angle = math.atan2(float(delta[1]), float(delta[0]))
        return float(math.atan2(math.sin(target_angle - self.heading), math.cos(target_angle - self.heading)))

    def _first_stop_region_point(self, start: np.ndarray, end: np.ndarray) -> np.ndarray | None:
        tolerance = self.stop_distance_tolerance
        inner = max(self.desired_stop_distance - tolerance, 0.0)
        outer = self.desired_stop_distance + tolerance
        segment = end - start
        segment_len_sq = float(np.dot(segment, segment))
        if segment_len_sq < 1e-8:
            distance = float(np.linalg.norm(self.target - end))
            if inner <= distance <= outer:
                return end.astype(np.float32)
            return None

        candidates: list[tuple[float, np.ndarray]] = []

        def add_candidate(t: float) -> None:
            if -1e-6 <= t <= 1.0 + 1e-6:
                t_clamped = float(np.clip(t, 0.0, 1.0))
                point = start + segment * t_clamped
                distance = float(np.linalg.norm(self.target - point))
                if inner - 1e-4 <= distance <= outer + 1e-4:
                    candidates.append((t_clamped, point.astype(np.float32)))

        start_distance = float(np.linalg.norm(self.target - start))
        end_distance = float(np.linalg.norm(self.target - end))
        if inner <= start_distance <= outer:
            add_candidate(0.0)
        if inner <= end_distance <= outer:
            add_candidate(1.0)

        offset = start - self.target
        a = segment_len_sq
        b = 2.0 * float(np.dot(offset, segment))
        for radius in (inner, outer):
            if radius <= 0.0:
                continue
            c = float(np.dot(offset, offset)) - radius * radius
            discriminant = b * b - 4.0 * a * c
            if discriminant < -1e-6:
                continue
            sqrt_disc = math.sqrt(max(discriminant, 0.0))
            add_candidate((-b - sqrt_disc) / (2.0 * a))
            add_candidate((-b + sqrt_disc) / (2.0 * a))

        for _, point in sorted(candidates, key=lambda item: item[0]):
            return point
        return None

    def _first_target_circle_point(self, start: np.ndarray, end: np.ndarray) -> np.ndarray | None:
        radius = float(self.target_radius)
        segment = end - start
        segment_len_sq = float(np.dot(segment, segment))
        if segment_len_sq < 1e-8:
            if float(np.linalg.norm(self.target - end)) <= radius:
                return end.astype(np.float32)
            return None

        start_distance = float(np.linalg.norm(self.target - start))
        end_distance = float(np.linalg.norm(self.target - end))
        if start_distance <= radius:
            return start.astype(np.float32)
        offset = start - self.target
        a = segment_len_sq
        b = 2.0 * float(np.dot(offset, segment))
        c = float(np.dot(offset, offset)) - radius * radius
        discriminant = b * b - 4.0 * a * c
        if discriminant >= -1e-6:
            sqrt_disc = math.sqrt(max(discriminant, 0.0))
            roots = [(-b - sqrt_disc) / (2.0 * a), (-b + sqrt_disc) / (2.0 * a)]
            roots = [float(np.clip(t, 0.0, 1.0)) for t in roots if -1e-6 <= t <= 1.0 + 1e-6]
            if roots:
                return (start + segment * min(roots)).astype(np.float32)
        if end_distance <= radius:
            return end.astype(np.float32)
        return None

    def _is_success(self) -> bool:
        distance = self._distance_to_target()
        if self.stop_distance_range is None:
            return distance <= self.target_radius + 1e-4
        return abs(distance - self.desired_stop_distance) <= self.stop_distance_tolerance
