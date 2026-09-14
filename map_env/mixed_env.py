"""Mixed training env: half maze, half open-world, sharing one coordinate-only policy.

Both sub-environments expose the same 13-dim observation (coordinate-only V4
layout) and the same 3-head action, so a single recurrent policy is trained on
both layouts at once. This is the practical way to make one policy transfer
across different games without assuming any map or obstacle sensing.
"""
from __future__ import annotations

import torch

from run_pipeline import CONFIG, GPUBlindNavEnvV11b
from map_env.maze_env import GPUImageMazeNavEnv


class MixedNavEnv:
    _FORWARD = (
        "fixed_position_age_ms",
        "position_age_min_ms",
        "stale_target_assist",
        "collision_penalty_scale",
        "low_block_penalty_scale",
        "position_age_max_ms",
        "position_stale_ms",
        "position_age_extreme_prob",
        "apply_freshness_speed",
        "decel_penalty",
        "decel_match_coef",
        "decel_age_ms",
        "maze_decel_penalty",
        "maze_speed_match_coef",
        "maze_overshoot_coef",
        "maze_approach_radius",
        "overshoot_coef",
        "overshoot_radius",
        "terminal_slow_radius",
        "terminal_slow_cap_frac",
        "speed_bins",
    )

    def __init__(self, num_envs, grid_path, device="cuda", **kwargs):
        self.n_maze = num_envs // 2
        self.n_open = num_envs - self.n_maze
        maze_kwargs = dict(kwargs)
        maze_kwargs.setdefault("map_guidance", CONFIG.get("maze_map_guidance", False))
        maze_kwargs.setdefault("target_range_frac", CONFIG.get("maze_target_range", 0.25))
        self.maze_env = GPUImageMazeNavEnv(
            num_envs=self.n_maze, grid_path=grid_path, device=device, **maze_kwargs
        )
        self.open_env = GPUBlindNavEnvV11b(
            num_envs=self.n_open, device=device, **kwargs
        )
        # Latency spikes on the open-world half too.
        self.open_env.position_age_spike_prob = self.maze_env.maze_spike_prob
        self.open_env.position_age_spike_ms = self.maze_env.maze_spike_ms
        self.open_env.obs_speed_scale = 100.0

    def __setattr__(self, name, value):
        if name in MixedNavEnv._FORWARD and hasattr(self, "maze_env"):
            setattr(self.maze_env, name, value)
            setattr(self.open_env, name, value)
        else:
            object.__setattr__(self, name, value)

    @property
    def turn_offsets(self):
        return self.maze_env.turn_offsets

    @property
    def heading(self):
        return torch.cat([self.maze_env.heading, self.open_env.heading])

    @property
    def estimated_heading(self):
        return torch.cat([self.maze_env.estimated_heading, self.open_env.estimated_heading])

    @property
    def time_since_collision(self):
        return torch.cat([self.maze_env.time_since_collision, self.open_env.time_since_collision])

    def reset(self, seed=None):
        o1 = self.maze_env.reset(seed)
        o2 = self.open_env.reset(seed)
        return torch.cat([o1, o2], dim=0)

    def step(self, actions):
        a1 = actions[: self.n_maze]
        a2 = actions[self.n_maze :]
        o1, r1, d1, re1 = self.maze_env.step(a1)
        o2, r2, d2, re2 = self.open_env.step(a2)
        return (
            torch.cat([o1, o2], dim=0),
            torch.cat([r1, r2], dim=0),
            torch.cat([d1, d2], dim=0),
            torch.cat([re1, re2], dim=0),
        )

    def apply_calibration(self, actions):
        a1 = self.maze_env.apply_calibration(actions[: self.n_maze])
        a2 = self.open_env.apply_calibration(actions[self.n_maze :])
        return torch.cat([a1, a2], dim=0)

    def set_obstacle_density(self, density, refresh=True):
        self.open_env.set_obstacle_density(density, refresh=refresh)

    def set_curriculum_progress(self, fraction):
        self.maze_env.set_curriculum_progress(fraction)

    def _get_obs(self):
        return torch.cat([self.maze_env._get_obs(), self.open_env._get_obs()], dim=0)
