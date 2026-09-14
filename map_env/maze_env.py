"""GPU-vectorized blind-navigation environment on a top-down maze occupancy grid.

This is a drop-in replacement for ``GPUUnknownHeadingNavEnv`` (a.k.a.
``GPUBlindNavEnvV11b`` from ``run_pipeline.py``): same 13-dim observation,
same 7-way steering / 2-way speed / jump action heads, same PPO interface.
The only differences are:

* the world is a static occupancy grid extracted from ``map_env/maze_grid.npz``
  instead of randomly scattered trees / mountains / low walls;
* start and target are randomized every episode among walkable cells;
* collision is a swept clearance probe against the grid, and the ``jump`` head
  has no effect (there is nothing jumpable in the maze).
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch

from run_pipeline import GPUUnknownHeadingNavEnv, MAX_TURN_RAD


class GPUImageMazeNavEnv(GPUUnknownHeadingNavEnv):
    def __init__(
        self,
        num_envs=8192,
        grid_path=None,
        device="cuda",
        dt=0.30,
        agent_radius=2.0,
        reach_radius=18.0,
        max_episode_steps=800,
        collision_substeps=8,
        min_start_target_dist=120.0,
        speed_scale=0.3,
        fresh_localization=False,
        disable_calibration=True,
        guide_lookahead=8,
        curriculum_min_dist=60.0,
        spawn_clearance=4.5,
        maze_step_cost=0.3,
        maze_collision_penalty=0.5,
        maze_idle_penalty=0.5,
        maze_new_cell_bonus=0.4,
        maze_same_cell_penalty=0.15,
        map_guidance=True,
        target_range_frac=0.25,
        position_age_spike_prob=0.05,
        position_age_spike_ms=1500.0,
        maze_decel_age_ms=350.0,
        maze_decel_penalty=0.25,
        maze_speed_match_coef=0.0,
        maze_approach_coef=0.0,
        maze_approach_radius=140.0,
        maze_overshoot_coef=0.0,
        learn_deceleration=False,
        **kwargs,
    ):
        if grid_path is None:
            grid_path = Path(__file__).resolve().parent / "maze_grid.npz"
        grid_path = Path(grid_path)
        if not grid_path.exists():
            raise FileNotFoundError(
                f"maze grid not found: {grid_path}; run map_env/build_maze_grid.py first"
            )

        data = np.load(grid_path)
        free = torch.as_tensor(np.ascontiguousarray(data["free"]), dtype=torch.float32, device=device)
        clearance = torch.as_tensor(
            np.ascontiguousarray(data["clearance"]), dtype=torch.float32, device=device
        )
        self.grid_h, self.grid_w = free.shape
        # Border is treated as wall so out-of-range samples collide instead of escaping.
        clearance[0, :] = 0.0
        clearance[-1, :] = 0.0
        clearance[:, 0] = 0.0
        clearance[:, -1] = 0.0
        self.grid_free = free
        self.grid_clearance = clearance
        self.grid_path = grid_path

        self.agent_radius = float(agent_radius)
        self.reach_radius = float(reach_radius)
        self.max_episode_steps = int(max_episode_steps)
        self.collision_substeps = int(collision_substeps)
        self.min_start_target_dist = float(min_start_target_dist)
        self.maze_speed_scale = float(speed_scale)
        self.maze_fresh_localization = bool(fresh_localization)
        self.maze_disable_calibration = bool(disable_calibration)
        self.guide_lookahead = max(1, int(guide_lookahead))
        self.curriculum_min_dist = float(curriculum_min_dist)
        self.curriculum_full_dist = float(np.hypot(self.grid_w, self.grid_h))
        self.curriculum_max_dist = float("inf")
        self.spawn_clearance = float(spawn_clearance)
        self.maze_step_cost = float(maze_step_cost)
        self.maze_collision_penalty = float(maze_collision_penalty)
        self.maze_idle_penalty = float(maze_idle_penalty)
        # map_guidance=False => strictly coordinate-only (no map/compass), the
        # general mode that transfers across games. Targets stay within
        # target_range_frac of the map's longer side (short multi-segment hops).
        self.map_guidance = bool(map_guidance)
        self.target_range_frac = float(target_range_frac)
        self.maze_new_cell_bonus = float(maze_new_cell_bonus)
        self.maze_same_cell_penalty = float(maze_same_cell_penalty)
        self.maze_decel_age_ms = float(maze_decel_age_ms)
        self.maze_decel_penalty = float(maze_decel_penalty)
        self.maze_speed_match_coef = float(maze_speed_match_coef)
        self.maze_approach_coef = float(maze_approach_coef)
        self.maze_approach_radius = float(maze_approach_radius)
        self.maze_overshoot_coef = float(maze_overshoot_coef)
        self.maze_spike_prob = float(position_age_spike_prob)
        self.maze_spike_ms = float(position_age_spike_ms)
        self._guide_cache = None
        self._last_cell = torch.full((num_envs,), -1, dtype=torch.long, device=device)
        self._same_cell_time = torch.zeros(num_envs, dtype=torch.float32, device=device)

        # Walkable cells with enough clearance to spawn, in centered world coords
        # (x right, y up, matching the math convention used by the base env).
        spawn = (clearance >= self.spawn_clearance) & (free > 0)
        rows, cols = torch.where(spawn)
        cells = torch.stack(
            [
                cols.float() - self.grid_w / 2.0 + 0.5,
                self.grid_h / 2.0 - 0.5 - rows.float(),
            ],
            dim=1,
        )
        self.spawn_cells = cells.to(device)

        # Coarse geodesic fields: one distance-to-goal map per candidate goal.
        # These remove the local minima that Euclidean progress creates in a maze.
        if "coarse_geo" in data:
            self.coarse_factor = int(data["coarse_factor"])
            coarse_geo = np.ascontiguousarray(data["coarse_geo"])
            self.coarse_geo = torch.as_tensor(coarse_geo, dtype=torch.float32, device=device)
            self.coarse_h, self.coarse_w = self.coarse_geo.shape[1:]
            goal_pixel = np.ascontiguousarray(data["goal_pixel"]).astype(np.float64)
            goal_x = goal_pixel[:, 1] - self.grid_w / 2.0 + 0.5
            goal_y = self.grid_h / 2.0 - 0.5 - goal_pixel[:, 0]
            self.goal_world = torch.as_tensor(
                np.stack([goal_x, goal_y], axis=1), dtype=torch.float32, device=device
            )
            self.num_goals = self.goal_world.shape[0]
        else:
            self.coarse_factor = 1
            self.coarse_geo = None
            self.goal_world = self.spawn_cells
            self.num_goals = self.goal_world.shape[0]

        self.goal_id = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.prev_geo = torch.zeros(num_envs, dtype=torch.float32, device=device)
        self._next_goal_ids = torch.zeros(num_envs, dtype=torch.long, device=device)

        world_size = float(max(self.grid_h, self.grid_w))
        super().__init__(
            num_envs=num_envs,
            world_size=world_size,
            dt=dt,
            device=device,
            **kwargs,
        )
        self._sync_goal_state(torch.arange(num_envs, device=device))
        # Maze-specific: turning in place (speed 0) is required to follow
        # corridors, so the open-world anti-dawdling penalties must be off.
        self.free_turn_penalty = 0.0
        self.far_slow_penalty = 0.0
        self.obs_speed_scale = 100.0 * self.maze_speed_scale
        self.position_age_spike_prob = self.maze_spike_prob
        self.position_age_spike_ms = self.maze_spike_ms
        if learn_deceleration:
            # Hand deceleration to the policy: no env-enforced freshness cap,
            # and a symmetric "speed should match freshness" reward.
            self.apply_freshness_speed = False
            self.maze_decel_penalty = 0.0
            self.decel_penalty = 0.0
            self.decel_match_coef = 0.0
            self.maze_speed_match_coef = 0.0
            self.decel_age_ms = self.maze_decel_age_ms
        if self.maze_fresh_localization:
            # Maze corridors need reliable steering; report a fresh position so
            # the heading estimator can lock onto the true motion direction.
            self.fixed_position_age_ms = 0.0
            self._refresh_observation()

    def apply_calibration(self, actions):
        if not self.maze_disable_calibration:
            return super().apply_calibration(actions)
        # Maze: skip the stop-and-calibrate sequence, but keep the collision
        # recovery commit so the agent commits to a large turn after a hit
        # instead of grinding against the wall.
        forced = actions.clone()
        recovery_signal = (
            (self.time_since_collision < self.recovery_signal_window)
            | (self.stuck_time > 0.5)
        ) & (not self.macro)
        start_commit = recovery_signal & (self.recovery_commit_left <= 0)
        commit_bin = torch.where(self.recovery_phase == 0, 6, 0)
        self.recovery_commit_left = torch.where(
            start_commit,
            torch.full_like(self.recovery_commit_left, self.recovery_commit_steps),
            self.recovery_commit_left,
        )
        in_commit = (self.recovery_commit_left > 0) | start_commit
        forced[in_commit, 0] = commit_bin[in_commit]
        self.recovery_commit_left = torch.clamp(
            self.recovery_commit_left - in_commit.int(), min=0
        )
        self.recovery_phase = torch.where(start_commit, 1 - self.recovery_phase, self.recovery_phase)
        return forced

    def reset(self, seed=None):
        self._guide_cache = None
        obs = super().reset(seed)
        self._sync_goal_state(torch.arange(self.num_envs, device=self.device))
        self._last_cell = self._cell_index(self.pos)
        self._same_cell_time.zero_()
        return obs

    # ------------------------------------------------------------------
    # Geodesic potential
    # ------------------------------------------------------------------
    def _potential(self, points, goal_ids, targets=None):
        """Distance-to-goal used by reward/obs.

        With map guidance this is the precomputed geodesic (along the maze);
        without it this is the plain straight-line distance (coordinate-only).
        ``targets`` overrides ``self.target`` for subset (auto-reset) calls.
        """
        if (not self.map_guidance) or self.coarse_geo is None:
            tgt = self.target if targets is None else targets
            return torch.norm(tgt - points, dim=1)
        col = torch.clamp(
            ((points[:, 0] + self.grid_w / 2.0) / self.coarse_factor).long(),
            0, self.coarse_w - 1,
        )
        row = torch.clamp(
            ((self.grid_h / 2.0 - points[:, 1]) / self.coarse_factor).long(),
            0, self.coarse_h - 1,
        )
        values = self.coarse_geo[goal_ids, row, col]
        return torch.clamp(values, max=1.0e5) * self.coarse_factor

    def _sync_goal_state(self, indices):
        self.goal_id[indices] = self._next_goal_ids
        self.prev_geo[indices] = self._potential(
            self.pos[indices], self.goal_id[indices], targets=self.target[indices]
        )

    def _cell_index(self, points):
        """Coarse-grid cell index of world points, for the exploration bonus."""
        col = torch.clamp(
            ((points[:, 0] + self.grid_w / 2.0) / self.coarse_factor).long(),
            0, self.coarse_w - 1,
        )
        row = torch.clamp(
            ((self.grid_h / 2.0 - points[:, 1]) / self.coarse_factor).long(),
            0, self.coarse_h - 1,
        )
        return row * self.coarse_w + col

    def _guide_vector(self, points=None):
        """Unit world direction toward a short-horizon point on the geodesic.

        The plain target bearing points through walls in a maze. Instead, walk a
        few cells downhill on the geodesic field and aim at that cell: a small
        lookahead smooths corners so the heading does not deadlock at them.
        Defaults to the delayed (believed) position, matching what the policy
        actually observes.
        """
        if points is None:
            points = getattr(self, "predicted_pos", self.pos)

        if (not self.map_guidance) or self.coarse_geo is None:
            delta = self.target - points
            return delta / torch.clamp(torch.norm(delta, dim=1, keepdim=True), min=1e-6)

        row = torch.clamp(
            ((self.grid_h / 2.0 - points[:, 1]) / self.coarse_factor).long(),
            0, self.coarse_h - 1,
        )
        col = torch.clamp(
            ((points[:, 0] + self.grid_w / 2.0) / self.coarse_factor).long(),
            0, self.coarse_w - 1,
        )
        goal = self.goal_id

        # Descent on 8 neighbours: diagonals give ~45-degree directions instead
        # of a 90-degree staircase, but only when both orthogonal cells are
        # traversable (the fields are clearance>=4, so corners are safe).
        # With a clearance>=4 field, diagonal cells cannot share a wall corner,
        # so 8-neighbour descent is safe and needs no extra corner check.
        dirs = ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1))
        for _ in range(self.guide_lookahead):
            current = self.coarse_geo[goal, row, col]
            best_val = current.clone()
            best_dr = torch.zeros_like(row)
            best_dc = torch.zeros_like(col)
            for dr, dc in dirs:
                rr = torch.clamp(row + dr, 0, self.coarse_h - 1)
                cc = torch.clamp(col + dc, 0, self.coarse_w - 1)
                value = self.coarse_geo[goal, rr, cc]
                better = value < best_val
                best_val = torch.where(better, value, best_val)
                best_dr = torch.where(better, torch.full_like(row, dr), best_dr)
                best_dc = torch.where(better, torch.full_like(col, dc), best_dc)
            moved = best_val < current
            row = torch.where(moved, row + best_dr, row)
            col = torch.where(moved, col + best_dc, col)

        aim_x = col.float() * self.coarse_factor - self.grid_w / 2.0 + 0.5
        aim_y = self.grid_h / 2.0 - 0.5 - row.float() * self.coarse_factor
        direct = torch.stack([aim_x, aim_y], dim=1) - points
        fallback = self.target - points
        no_move = torch.norm(direct, dim=1, keepdim=True) < 1e-6
        direct = torch.where(no_move, fallback, direct)
        return direct / torch.clamp(torch.norm(direct, dim=1, keepdim=True), min=1e-6)

    def _get_obs(self):
        if getattr(self, "_v4_initializing", False):
            return super()._get_obs()
        if not self.map_guidance:
            # Coordinate-only mode reuses the V4 observation layout, so the same
            # policy can be trained across maze and open-world environments.
            return super()._get_obs()

        guide = getattr(self, "_guide_cache", None)
        if guide is None:
            guide = self._guide_vector()
        guide_angle = torch.atan2(guide[:, 1], guide[:, 0])
        angle_error = self._normalize_angle(guide_angle - self.estimated_heading)
        geo = self._potential(self.predicted_pos, self.goal_id)

        low_obs_signal = self._get_low_proximity_signal()
        collision_touch = torch.where(self.time_since_collision < 0.6, 1.0, 0.0)

        obs = torch.stack([
            guide[:, 0],
            guide[:, 1],
            torch.sin(angle_error),
            torch.cos(angle_error),
            torch.clamp(self.velocity[:, 0] / self.MAX_REASONABLE_SPEED, -1.0, 1.0),
            torch.clamp(self.velocity[:, 1] / self.MAX_REASONABLE_SPEED, -1.0, 1.0),
            torch.clamp(geo / 2000.0, 0.0, 1.0),
            torch.clamp(self.stuck_time / 3.0, 0.0, 1.0),
            collision_touch,
            low_obs_signal,
            torch.clamp(self.observed_age_ms / self.position_age_max_ms, 0.0, 1.0),
            self.heading_confidence,
            torch.clamp(self.last_turn_delta / MAX_TURN_RAD, -1.0, 1.0),
        ], dim=1)
        return obs

    # ------------------------------------------------------------------
    # Grid sampling / obstacle hooks
    # ------------------------------------------------------------------
    def set_curriculum_progress(self, fraction):
        """Scale the max start->target distance from near to full as training proceeds."""
        fraction = float(max(0.0, min(1.0, fraction)))
        if fraction >= 1.0:
            self.curriculum_max_dist = float("inf")
        else:
            self.curriculum_max_dist = self.curriculum_min_dist + (
                self.curriculum_full_dist - self.curriculum_min_dist
            ) * fraction

    def _sample_start_and_target(self, num_samples):
        num_spawn = self.spawn_cells.shape[0]
        start_idx = torch.randint(0, num_spawn, (num_samples,), device=self.device)
        pos = self.spawn_cells[start_idx]

        # Random goal, biased far but capped by the curriculum distance and by
        # the target_range_frac short-hop limit (fraction of the longer side).
        max_dist = self.curriculum_max_dist
        if self.target_range_frac > 0.0:
            max_dist = min(max_dist, self.target_range_frac * max(self.grid_w, self.grid_h))
        dist = torch.cdist(pos, self.goal_world)  # (N, K)
        allowed = dist <= max_dist
        noise = torch.rand_like(dist) * (0.5 * min(max_dist, self.curriculum_full_dist))
        score = torch.where(allowed, dist - noise, torch.full_like(dist, -1.0e9))
        goal_id = score.argmax(dim=1)
        no_allowed = ~allowed.any(dim=1)
        goal_id = torch.where(no_allowed, dist.argmin(dim=1), goal_id)
        target = self.goal_world[goal_id]
        self._next_goal_ids = goal_id
        return pos, target

    def _generate_obstacles_for_envs(self, env_indices, player_pos, target_pos):
        return

    def set_obstacle_density(self, density, refresh=True):
        self.obstacle_density = float(max(0.0, min(1.0, density)))

    def _get_low_proximity_signal(self):
        return torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)

    def _speed_for_action(self, speed_bin):
        # Base speeds target a 2250-unit world; the maze grid is ~512 units, so
        # scale them down to keep per-step motion comparable to corridor width.
        return super()._speed_for_action(speed_bin) * self.maze_speed_scale

    # ------------------------------------------------------------------
    # Collision
    # ------------------------------------------------------------------
    def _bilinear_clearance(self, points):
        """Sample the clearance field at world points of shape (..., 2)."""
        col = points[..., 0] + self.grid_w / 2.0 - 0.5
        row = self.grid_h / 2.0 - 0.5 - points[..., 1]
        col = torch.clamp(col, 0.0, self.grid_w - 1.001)
        row = torch.clamp(row, 0.0, self.grid_h - 1.001)
        c0 = col.floor().long()
        r0 = row.floor().long()
        c1 = torch.clamp(c0 + 1, max=self.grid_w - 1)
        r1 = torch.clamp(r0 + 1, max=self.grid_h - 1)
        wx = (col - c0.float()).unsqueeze(-1)
        wy = (row - r0.float()).unsqueeze(-1)
        g = self.grid_clearance
        v00 = g[r0, c0].unsqueeze(-1)
        v01 = g[r0, c1].unsqueeze(-1)
        v10 = g[r1, c0].unsqueeze(-1)
        v11 = g[r1, c1].unsqueeze(-1)
        return (
            v00 * (1.0 - wx) * (1.0 - wy)
            + v01 * wx * (1.0 - wy)
            + v10 * (1.0 - wx) * wy
            + v11 * wx * wy
        ).squeeze(-1)

    def _grid_collide(self, old_pos, candidate):
        steps = self.collision_substeps
        t = torch.linspace(0.0, 1.0, steps, device=self.device).view(1, steps, 1)
        points = old_pos.unsqueeze(1) + (candidate - old_pos).unsqueeze(1) * t
        clearance = self._bilinear_clearance(points)
        free = clearance >= self.agent_radius
        collided = ~free.all(dim=1)

        if self.collision_mode == "hard":
            new_pos = torch.where(collided.unsqueeze(1), old_pos, candidate)
        else:
            prefix_free = free.to(torch.int32).cumprod(dim=1)
            last_free = (prefix_free.sum(dim=1) - 1).clamp(min=0)
            t_last = torch.linspace(0.0, 1.0, steps, device=self.device)[last_free]
            slide_pos = old_pos + (candidate - old_pos) * t_last.unsqueeze(1)
            if self.collision_mode == "weak":
                slide_pos = old_pos + (slide_pos - old_pos) * 0.25
            new_pos = torch.where(collided.unsqueeze(1), slide_pos, candidate)
        return collided, new_pos

    # ------------------------------------------------------------------
    # Physics + reward
    # ------------------------------------------------------------------
    def step(self, actions):
        old_pos = self.pos.clone()
        angle_bin = actions[:, 0].clamp(0, 6)
        self.last_turn_delta = self.turn_offsets[angle_bin]
        self.turn_history[:, 1] = self.turn_history[:, 0]
        self.turn_history[:, 0] = self.last_turn_delta

        self.step_count += 1
        self.time_since_collision += self.dt

        speed_bin = actions[:, 1].clamp(0, getattr(self, "speed_bins", 2) - 1)
        start_macro = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        delta = self.target - self.pos
        target_angle = torch.atan2(delta[:, 1], delta[:, 0])
        recovery_mode = (
            (self.time_since_collision < 0.9)
            | (self.no_progress_time > 0.5)
            | (self.stuck_time > 0.5)
        )
        if self.action_mode == "heading_relative":
            desired_angle = self.heading + self.ANGLE_OFFSETS_HEADING[angle_bin] * math.pi
        elif self.action_mode == "hybrid":
            goal_angle = target_angle + self.ANGLE_OFFSETS_HYBRID[angle_bin] * math.pi
            recovery_angle = self.heading + self.ANGLE_OFFSETS_HEADING[angle_bin] * math.pi
            desired_angle = torch.where(recovery_mode, recovery_angle, goal_angle)
        else:
            desired_angle = target_angle + self.ANGLE_OFFSETS[angle_bin] * math.pi

        if self.macro:
            macro_bin = actions[:, 3].clamp(0, 7)
            start_macro = (macro_bin > 0) & (self.macro_left <= 0)
            new_heading = self.heading + self.MACRO_TURN[macro_bin]
            self.macro_heading = torch.where(start_macro, new_heading, self.macro_heading)
            self.macro_left = torch.where(start_macro, self.MACRO_HOLD[macro_bin], self.macro_left)
            in_macro = self.macro_left > 0
            desired_angle = torch.where(in_macro, self.macro_heading, desired_angle)
            self.macro_left = torch.clamp(self.macro_left - in_macro.int(), min=0)

        speed = self._terminal_speed_limit(self._speed_for_action(speed_bin))

        max_turn = math.radians(45.0)
        ang_diff = desired_angle - self.heading
        ang_diff = torch.atan2(torch.sin(ang_diff), torch.cos(ang_diff))
        ang_diff_clamped = torch.clamp(ang_diff, -max_turn, max_turn)
        turn_fraction = ang_diff_clamped.abs() / max_turn
        self.heading = torch.atan2(
            torch.sin(self.heading + ang_diff_clamped), torch.cos(self.heading + ang_diff_clamped)
        )

        intended_delta = (
            torch.stack([torch.cos(self.heading), torch.sin(self.heading)], dim=1)
            * speed.unsqueeze(1)
            * self.dt
        )
        candidate = self.pos + intended_delta

        any_collided, new_pos = self._grid_collide(old_pos, candidate)
        self.pos = new_pos
        self.pos = torch.clamp(self.pos, -1100.0, 1100.0)

        displacement = torch.norm(self.pos - old_pos, dim=1)
        self.stuck_time = torch.where(
            displacement < 2.0, self.stuck_time + self.dt, torch.zeros_like(self.stuck_time)
        )
        self.time_since_collision = torch.where(
            any_collided, torch.zeros_like(self.time_since_collision), self.time_since_collision
        )
        self.jump_probe_signal = torch.zeros_like(self.jump_probe_signal)
        self.last_jump_pass_low = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.last_block_low_collided = torch.zeros_like(self.last_jump_pass_low)

        curr_dist = torch.norm(self.target - self.pos, dim=1)
        # Geodesic progress (along the maze) drives the reward instead of
        # straight-line progress, which would reward walking into dead ends.
        curr_geo = self._potential(self.pos, self.goal_id)
        progress = self.prev_geo - curr_geo
        self.prev_geo = curr_geo
        self.no_progress_time = torch.where(
            progress < 0.2, self.no_progress_time + self.dt, torch.zeros_like(self.no_progress_time)
        )
        self.prev_dist = curr_dist

        # Align the heading with the geodesic compass, not the straight line to
        # the goal (which points through walls in a maze).
        guide = self._guide_vector()
        self._guide_cache = guide
        guide_angle = torch.atan2(guide[:, 1], guide[:, 0])
        angle_err = self._normalize_angle(guide_angle - self.heading)
        angle_penalty = 1.0 - torch.cos(angle_err)

        reward = torch.clamp(progress, -40.0, 40.0) * 0.25 - self.maze_step_cost
        reward = reward - angle_penalty * self.angle_penalty_coef

        # Note: intentionally NO "unstick" bonus (it is farmable). Instead we
        # make standing still strictly worse than pressing forward: a large step
        # cost plus an explicit idle penalty, with a mild collision penalty so
        # exploring is cheaper than freezing.
        is_stuck = (self.time_since_collision < 0.6) | (self.no_progress_time > 0.5)
        collision_penalty = self.maze_collision_penalty * torch.where(
            any_collided, 1.0, 0.0
        )
        head_on_stuck = is_stuck & (angle_bin == 3) & any_collided
        head_on_penalty = torch.where(head_on_stuck, 1.0, 0.0)
        stuck_pressure = torch.clamp(self.no_progress_time - 0.5, 0.0, 3.0)
        idle_penalty = torch.where(
            displacement < 1.0, torch.full_like(reward, self.maze_idle_penalty), torch.zeros_like(reward)
        )

        reward = (
            reward
            - 0.40 * stuck_pressure
            - collision_penalty
            - head_on_penalty
            - idle_penalty
        )
        reward = reward - torch.where(
            ~is_stuck, turn_fraction * self.free_turn_penalty, torch.zeros_like(reward)
        )
        far_slow = (curr_dist >= 120.0) & (speed_bin == 0)
        reward = reward - torch.where(
            far_slow, torch.full_like(reward, self.far_slow_penalty), torch.zeros_like(reward)
        )

        # Deceleration: symmetric reward for matching speed to localisation age
        # (fast when fresh, slow when stale) so the policy learns conditional
        # deceleration instead of collapsing to always-slow.
        if self.maze_speed_match_coef > 0.0:
            age_norm = torch.clamp(
                self.observed_age_ms / self.position_age_max_ms, 0.0, 1.0
            )
            fast = (speed_bin == 1).float()
            reward = reward + self.maze_speed_match_coef * (
                fast * (1.0 - age_norm) + (1.0 - fast) * age_norm
            )
        if self.maze_decel_penalty > 0.0:
            stale = self.observed_age_ms > self.maze_decel_age_ms
            reward = reward - torch.where(
                stale & (speed_bin == 1),
                torch.full_like(reward, self.maze_decel_penalty),
                torch.zeros_like(reward),
            )

        # Approach slowdown: prefer fast while far from the goal and slow while
        # closing in, so the policy brakes before the target instead of
        # overshooting (especially with delayed localisation).
        if self.maze_approach_coef > 0.0:
            near = curr_dist < self.maze_approach_radius
            fast = (speed_bin == getattr(self, "speed_bins", 2) - 1).float()
            desired_fast = (~near).float()
            match = 1.0 - (fast - desired_fast).abs()
            reward = reward + self.maze_approach_coef * (2.0 * match - 1.0)

        # Outcome-driven overshoot penalty: inside the approach zone, penalise
        # the target distance increasing (overshoot) instead of preferring a
        # speed bin. This cannot collapse to always-slow.
        if self.maze_overshoot_coef > 0.0:
            near = curr_dist < self.maze_approach_radius
            overshoot = near & (progress < 0.0)
            penalty = self.maze_overshoot_coef * torch.clamp(-progress, 0.0, 12.0)
            reward = reward - torch.where(overshoot, penalty, torch.zeros_like(reward))

        # Exploration: reward entering a new coarse cell, penalise loitering in
        # one place (encourages trying/搜索 instead of grinding against a wall).
        cell = self._cell_index(self.pos)
        new_cell = cell != self._last_cell
        reward = reward + torch.where(
            new_cell, torch.full_like(reward, self.maze_new_cell_bonus), torch.zeros_like(reward)
        )
        self._same_cell_time = torch.where(
            new_cell, torch.zeros_like(self._same_cell_time), self._same_cell_time + self.dt
        )
        reward = reward - torch.where(
            ~new_cell,
            self.maze_same_cell_penalty * torch.clamp(self._same_cell_time, 0.0, 3.0),
            torch.zeros_like(reward),
        )
        self._last_cell = cell

        if self.macro:
            fired = start_macro
            reward = reward + torch.where(fired & is_stuck, 3.0, 0.0)
            reward = reward - torch.where(fired & ~is_stuck, 0.5, 0.0)

        near_target = curr_dist < 120.0
        zeros_r = torch.zeros_like(reward)
        reward = reward - torch.where(near_target, angle_penalty * 0.8, zeros_r)
        if self.action_mode in ("heading_relative", "hybrid"):
            off_line = near_target & (angle_err.abs() > math.radians(15.0))
        else:
            off_line = near_target & (angle_bin != 3)
        reward = torch.where(off_line, reward - 0.6, reward)
        near_fast = near_target & (speed_bin == 1)
        reward = torch.where(near_fast, reward - 0.4, reward)

        reached = curr_dist < self.reach_radius
        truncated = self.step_count >= self.max_episode_steps
        dones = reached | truncated

        reward = torch.where(reached, reward + 20.0, reward)
        reward = torch.where(truncated & ~reached, reward - 5.0, reward)

        if self.auto_reset and dones.any():
            done_indices = torch.where(dones)[0]
            num_dones = len(done_indices)
            r_pos, r_target = self._sample_start_and_target(num_dones)
            self.pos[done_indices] = r_pos
            self.target[done_indices] = r_target
            self.goal_id[done_indices] = self._next_goal_ids
            self.prev_geo[done_indices] = self._potential(
                self.pos[done_indices], self.goal_id[done_indices],
                targets=self.target[done_indices],
            )
            self.heading[done_indices] = (
                torch.rand(num_dones, device=self.device) * 2.0 - 1.0
            ) * math.pi
            self.macro_left[done_indices] = 0
            self.jump_probe_signal[done_indices] = 0.0
            self.step_count[done_indices] = 0
            self.stuck_time[done_indices] = 0.0
            self.no_progress_time[done_indices] = 0.0
            self.time_since_collision[done_indices] = 10.0
            self.prev_dist[done_indices] = torch.norm(
                self.target[done_indices] - self.pos[done_indices], dim=1
            )
            self.prev_angle_error[done_indices] = self._angle_error()[done_indices]
            self._last_cell[done_indices] = self._cell_index(self.pos[done_indices])
            self._same_cell_time[done_indices] = 0.0

        # V4 belief-state bookkeeping (mirrors GPUUnknownHeadingNavEnv.step).
        active = ~dones
        if active.any():
            self.prev_prev_actual_pos[active] = self.prev_actual_pos[active]
            self.prev_actual_pos[active] = old_pos[active]
        if dones.any():
            self._reset_belief_state(torch.where(dones)[0])
        self._refresh_observation(active=active)
        return self._get_obs(), reward, dones, reached
