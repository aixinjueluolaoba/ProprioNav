import os
import math
import time
import argparse
import subprocess
import multiprocessing
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

# =====================================================================
# 🛠️ 全局配置参数 (纯端到端纯物理 RL 寻路)
# =====================================================================
CONFIG = {
    # --- 强化学习训练参数 (GPU Vectorized) ---
    "num_envs": 8192,            # 8192路高频极速并发 (每 2 秒实时刷新迭代日志)
    "total_episodes": 1000000,   # 快速验证默认 100 万轮
    "rollout_steps": 128,        # 每次收集的步数
    "minibatch_envs": 8192,      # 单 minibatch，避免切分环境后重复序列推理
    "ppo_epochs": 1,             # 当前模型用一次 PPO 更新，吞吐优先
    "lr": 1.5e-4,                # 学习率
    "gamma": 0.99,               # 折扣因子
    "gae_lambda": 0.95,          # GAE 优势估计系数
    "ent_coef": -0.010,          # 压低策略熵，避免 argmax 蛇形抖动
    "vf_coef": 0.5,              # 价值损失权重
    "max_grad_norm": 0.5,        # 梯度裁剪阈值
    "checkpoint_every": 15,      # 自动保存间隔

    # --- 测试与渲染参数 ---
    "eval_episodes": 10,         # 评估测试次数
    "seed_start": 140001,        # 评估起点种子
    "fps": 20,                   # 视频帧率
    "eval_sample": False,        # 按策略分布采样；高熵策略用 argmax 会退化成直冲
    "eval_commit": 0,            # 碰撞后锁定大角度转向档若干步

    # --- 目录和文件保存 ---
    "out_dir": "pipeline_out",                         # 输出主目录
    "weights_name": "policy_weights_v4_unknown_heading.pth",
    "state_dim": 13,
    "hidden_dim": 96,
    "actor_width": 48,
    "collision_mode": "hard",    # hard=硬停，迫使策略主动转向
    "angle_penalty_coef": 1.00,  # 偏离目标方向惩罚
    "step_cost": 0.12,           # 每步固定成本，抑制无效绕路
    "free_turn_penalty": 0.18,   # 非脱困状态的实际转向惩罚
    "far_slow_penalty": 0.15,    # 远离终点时选择低速的惩罚
    "hybrid_free_max_deg": 10.0, # 保留旧配置字段，V4 使用相对转向
    "obstacle_signal_mode": "jump_probe",  # 仅使用位置历史可推断的反馈
    "false_jump_penalty": 0.05,
    "reset_jump_head": False,
    "train_jump_only": False,

    # --- V4 未知朝向配置 ---
    # target_relative : desired_angle = 目标方向 + 偏移 (原版)。每步重锚到目标方向,
    #                   叠加 45°/步 的转向上限 → 无法完成持续掉头, 顶墙时会冻结。
    # heading_relative: desired_angle = 当前朝向 + 偏移。转向可累积, 能持续绕行。
    # hybrid         : 正常行走时锚定目标、碰撞后切到朝向相对转向，兼顾直线效率和脱困。
    "action_mode": "unknown_heading",
    "position_age_max_ms": 500.0,
    "position_stale_ms": 1000.0,
    "position_age_extreme_prob": 0.8,
    "teacher_coef": 0.25,
    "teacher_decay_steps": 30000000,
    "macro_teacher_coef": 1.0,
    "macro_teacher_decay_steps": 150000000,
    "macro_teacher_loss_coef": 1.0,
    "steer_pretrain_steps": 300,
    "steer_pretrain_batch_envs": 2048,
    "steer_pretrain_seq_len": 32,
    "steer_pretrain_lr": 1.0e-3,
    "obstacle_density_start": 0.0,
    "obstacle_density_end": 1.0,
    "obstacle_curriculum_steps": 30000000,
    "validation_every_updates": 50,
    "validation_episodes": 64,
    "validation_max_steps": 240,
    "post_training_video": True,
    "training_video_fps": 20,
    "training_video_steps": 120,
    "training_video_age_ms": 250.0,
    "training_video_dir": "/tmp/file/盲人寻路训练",
    "recovery_commit_steps": 4,
    "recovery_signal_window": 0.6,
    "fixed_training_age_ms": None,
    "stale_target_assist": False,
    "recovery_reward_scale": 1.0,
    "collision_penalty_scale": 1.0,
    "low_block_penalty_scale": 1.0,
    # 显式脱困宏: 加第 4 个动作头(8 档), 0=正常导航, 1-7=锁定若干步的大角度脱困机动。
    "macro": False,
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OBS_DIM = 13
MAX_TURN_RAD = math.radians(45.0)
TURN_OFFSETS_RAD = torch.tensor([
    -math.radians(45.0), -math.radians(25.0), -math.radians(10.0), 0.0,
    math.radians(10.0), math.radians(25.0), math.radians(45.0),
], dtype=torch.float32)

# =====================================================================
# 1. 向量化 GPU 仿真环境 (纯正统盲人寻路)
# =====================================================================
class GPUVectorizedBlindNavEnv:
    """GPU 高频极速向量化 2.25 倍巨型大世界纯正统盲人寻路环境 (2250x2250)
    障碍密度增加 50%：38 棵树 + 9 座山 + 24 挡长条形起跳墙 (Long Strip Jump Barriers)
    """
    def __init__(self, num_envs=8192, world_size=2250.0, dt=0.30, device='cuda', auto_reset=True,
                 collision_mode="hard", action_mode="hybrid", macro=False,
                 angle_penalty_coef=1.00, step_cost=0.12, free_turn_penalty=0.18,
                 far_slow_penalty=0.15, hybrid_free_max_deg=10.0,
                 obstacle_signal_mode="jump_probe", false_jump_penalty=0.05,
                 obstacle_density=1.0, position_age_extreme_prob=0.4,
                 recovery_commit_steps=4, recovery_reward_scale=1.0):
        self.num_envs = num_envs
        self.auto_reset = auto_reset
        self.collision_mode = collision_mode  # slide=完整滑动 / weak=削弱滑动 / hard=硬停回退
        self.action_mode = action_mode        # target_relative / heading_relative
        self.macro = macro                    # 是否启用显式脱困宏 (第 4 动作头)
        self.angle_penalty_coef = angle_penalty_coef
        self.step_cost = step_cost
        self.free_turn_penalty = free_turn_penalty
        self.far_slow_penalty = far_slow_penalty
        self.hybrid_free_max_deg = hybrid_free_max_deg
        self.obstacle_signal_mode = obstacle_signal_mode
        self.false_jump_penalty = false_jump_penalty
        self.world_size = world_size
        self.dt = dt
        self.device = device
        self.obstacle_density = float(obstacle_density)
        self.position_age_extreme_prob = float(max(0.0, min(1.0, position_age_extreme_prob)))
        self.recovery_commit_steps = max(0, int(recovery_commit_steps))
        self.recovery_reward_scale = float(recovery_reward_scale)
        self.collision_penalty_scale = 1.0
        self.low_block_penalty_scale = 1.0

        # 极高障碍密度配置 (密集修罗场)
        self.tree_count = 60
        self.mountain_count = 15
        self.low_count = 40

        # 长条形矮墙参数 (半长 40.0, 半宽 24.0)
        self.low_length = 80.0
        self.low_width = 24.0
        self.low_radii = torch.full((num_envs, self.low_count), 24.0, device=device)
        self.low_centers = torch.zeros((num_envs, self.low_count, 2), dtype=torch.float32, device=device)
        self.low_angles = torch.zeros((num_envs, self.low_count), dtype=torch.float32, device=device)

        self.tree_radii = torch.full((num_envs, self.tree_count), 28.0, device=device)
        self.mountain_radii = torch.rand((num_envs, self.mountain_count), device=device) * (90.0 - 40.0) + 40.0

        self.pos = torch.zeros((self.num_envs, 2), dtype=torch.float32, device=device)
        self.target = torch.zeros((self.num_envs, 2), dtype=torch.float32, device=device)
        self.heading = torch.zeros(self.num_envs, dtype=torch.float32, device=device)
        self.tree_centers = torch.zeros((self.num_envs, self.tree_count, 2), dtype=torch.float32, device=device)
        self.mountain_centers = torch.zeros((self.num_envs, self.mountain_count, 2), dtype=torch.float32, device=device)
        self.mountain_vertices = torch.zeros((self.num_envs, self.mountain_count, 8, 2), dtype=torch.float32, device=device)

        self.step_count = torch.zeros(self.num_envs, dtype=torch.int32, device=device)
        self.stuck_time = torch.zeros(self.num_envs, dtype=torch.float32, device=device)
        self.no_progress_time = torch.zeros(self.num_envs, dtype=torch.float32, device=device)
        self.time_since_collision = torch.full((self.num_envs,), 10.0, dtype=torch.float32, device=device)

        self.prev_dist = torch.zeros(self.num_envs, dtype=torch.float32, device=device)
        self.prev_angle_error = torch.zeros(self.num_envs, dtype=torch.float32, device=device)

        # 预计算几何缓存张量
        self.mountain_vertices_roll = torch.zeros((self.num_envs, self.mountain_count, 8, 2), dtype=torch.float32, device=device)
        self.mountain_edges = torch.zeros((self.num_envs, self.mountain_count, 8, 2), dtype=torch.float32, device=device)
        self.mountain_edge_dx = torch.zeros((self.num_envs, self.mountain_count, 8), dtype=torch.float32, device=device)
        self.mountain_edge_dy = torch.zeros((self.num_envs, self.mountain_count, 8), dtype=torch.float32, device=device)

        # 7 档全向转向偏角：[-180°, -90°, -45°, 0°, +45°, +90°, +180°] (新增 ±45° 精细进站微调)
        self.ANGLE_OFFSETS = torch.tensor([-1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0], device=device)

        # 朝向相对模式的偏角: [-45°, -25°, -10°, 0°, +10°, +25°, +45°]。
        # 必须落在 45°/步 的转向上限内, 否则 ±90°/±180° 会被 clamp 成同一个 45°, 7 档退化成 3 档。
        # 转向可累积: 连续 4 步 -45° 即完成一次掉头, 这正是原版做不到的。
        self.ANGLE_OFFSETS_HEADING = torch.tensor(
            [-0.25, -25.0 / 180.0, -10.0 / 180.0, 0.0, 10.0 / 180.0, 25.0 / 180.0, 0.25], device=device)
        hybrid_scale = hybrid_free_max_deg / 180.0
        self.ANGLE_OFFSETS_HYBRID = torch.tensor(
            [-1.0, -0.60, -0.25, 0.0, 0.25, 0.60, 1.0], device=device) * hybrid_scale

        # 脱困宏库: 7 种恢复机动 = (相对当前朝向的转角, 锁定步数)。
        # 锁定期内忽略转向头输出、朝一个固定绝对朝向走, 让 45°/步 的限制走得完大角度机动。
        self.MACRO_TURN = torch.tensor(
            [0.0, -0.25, 0.25, -0.5, 0.5, -0.75, 0.75, 1.0], device=device) * math.pi
        self.MACRO_HOLD = torch.tensor([0, 4, 4, 6, 6, 6, 6, 8], device=device, dtype=torch.int32)
        self.macro_left = torch.zeros(num_envs, dtype=torch.int32, device=device)
        self.macro_heading = torch.zeros(num_envs, dtype=torch.float32, device=device)
        self.jump_probe_signal = torch.zeros(num_envs, dtype=torch.float32, device=device)

        self.reset()

    def _sample_start_and_target(self, num_samples):
        # 起点固定在地图绝对中心 (0, 0)
        pos = torch.zeros((num_samples, 2), device=self.device)

        # 目标点全随机采样在四个远距离对角象限 (Corner Quadrants)
        corners = torch.tensor([
            [1.0, 1.0],   # 东北对角
            [-1.0, 1.0],  # 西北对角
            [-1.0, -1.0], # 西南对角
            [1.0, -1.0]   # 东南对角
        ], device=self.device)

        corner_idx = torch.randint(0, 4, (num_samples,), device=self.device)
        chosen_corners = corners[corner_idx]

        # 加上对角区域微小抖动
        jitter = (torch.rand((num_samples, 2), device=self.device) - 0.5) * 160.0
        target = chosen_corners * 950.0 + jitter
        target = torch.clamp(target, -1050.0, 1050.0)
        return pos, target

    def _generate_obstacles_for_envs(self, env_indices, player_pos, target_pos):
        num_envs = len(env_indices)
        if num_envs == 0:
            return

        # 在 2250 大地图 [-1050, +1050] 范围内全随机撒落海量树木、山体与长条矮墙
        t_centers = (torch.rand((num_envs, self.tree_count, 2), device=self.device) - 0.5) * 2100.0
        m_centers = (torch.rand((num_envs, self.mountain_count, 2), device=self.device) - 0.5) * 2100.0
        l_centers = (torch.rand((num_envs, self.low_count, 2), device=self.device) - 0.5) * 2100.0
        if self.obstacle_density < 1.0:
            density = max(0.0, min(1.0, self.obstacle_density))
            active_trees = torch.rand((num_envs, self.tree_count), device=self.device) < density
            active_mountains = torch.rand((num_envs, self.mountain_count), device=self.device) < density
            active_low = torch.rand((num_envs, self.low_count), device=self.device) < density
            t_centers = torch.where(
                active_trees.unsqueeze(2),
                t_centers,
                torch.full_like(t_centers, 1e6),
            )
            m_centers = torch.where(
                active_mountains.unsqueeze(2),
                m_centers,
                torch.full_like(m_centers, 1e6),
            )
            l_centers = torch.where(
                active_low.unsqueeze(2),
                l_centers,
                torch.full_like(l_centers, 1e6),
            )
        l_angles = torch.rand((num_envs, self.low_count), device=self.device) * math.pi
        m_radii = self.mountain_radii[env_indices]

        # 仅保留起点 40.0 安全避让缓冲
        dist_to_pos_t = torch.norm(t_centers - player_pos.unsqueeze(1), dim=2)
        t_centers = torch.where((dist_to_pos_t < 40.0).unsqueeze(2), t_centers + 50.0, t_centers)

        dist_to_pos_m = torch.norm(m_centers - player_pos.unsqueeze(1), dim=2)
        m_centers = torch.where((dist_to_pos_m < 60.0).unsqueeze(2), m_centers + 80.0, m_centers)

        num_vertices = 8
        angles = torch.linspace(0.0, 2.0 * math.pi, num_vertices + 1, device=self.device)[:-1]
        angles = angles.unsqueeze(0).unsqueeze(0).expand(num_envs, self.mountain_count, num_vertices)
        angles = angles + (torch.rand_like(angles) * 0.36 - 0.18)
        r_vert = m_radii.unsqueeze(2) * (0.6 + torch.rand_like(angles) * 0.4)

        v_x = r_vert * torch.cos(angles)
        v_y = r_vert * torch.sin(angles)
        m_vertices = m_centers.unsqueeze(2) + torch.stack([v_x, v_y], dim=3)

        self.low_centers[env_indices] = l_centers
        self.low_angles[env_indices] = l_angles
        self.tree_centers[env_indices] = t_centers
        self.mountain_centers[env_indices] = m_centers
        self.mountain_vertices[env_indices] = m_vertices

        m_vertices_roll = torch.roll(m_vertices, shifts=-1, dims=2)
        self.mountain_vertices_roll[env_indices] = m_vertices_roll
        self.mountain_edges[env_indices] = m_vertices_roll - m_vertices
        self.mountain_edge_dx[env_indices] = m_vertices_roll[..., 0] - m_vertices[..., 0]
        self.mountain_edge_dy[env_indices] = m_vertices_roll[..., 1] - m_vertices[..., 1]

    def set_obstacle_density(self, density, refresh=True):
        self.obstacle_density = float(max(0.0, min(1.0, density)))
        if refresh:
            env_indices = torch.arange(self.num_envs, device=self.device)
            self._generate_obstacles_for_envs(env_indices, self.pos, self.target)

    def reset(self, seed=None):
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        self.pos, self.target = self._sample_start_and_target(self.num_envs)
        self._generate_obstacles_for_envs(torch.arange(self.num_envs), self.pos, self.target)

        self.heading = (torch.rand(self.num_envs, dtype=torch.float32, device=self.device) * 2.0 - 1.0) * math.pi
        self.step_count.fill_(0)
        self.stuck_time.fill_(0.0)
        self.no_progress_time.fill_(0.0)
        self.time_since_collision.fill_(10.0)
        self.macro_left.fill_(0)
        self.jump_probe_signal.fill_(0.0)

        self.prev_dist = torch.norm(self.target - self.pos, dim=1)
        self.prev_angle_error = self._angle_error()
        return self._get_obs()

    def _angle_error(self):
        delta = self.target - self.pos
        target_angle = torch.atan2(delta[:, 1], delta[:, 0])
        err = target_angle - self.heading
        return torch.atan2(torch.sin(err), torch.cos(err))

    def _get_obs(self):
        delta = self.target - self.pos
        distance = torch.norm(delta, dim=1)
        angle_error = self._angle_error()

        low_proximity_signal = self._get_low_proximity_signal()
        low_obs_signal = (self.jump_probe_signal
                          if self.obstacle_signal_mode == "jump_probe"
                          else low_proximity_signal)
        collision_touch = torch.where(self.time_since_collision < 0.6, 1.0, 0.0)

        # 纯正统盲人寻路 (10 维纯物理 Proprioceptive 状态向量，无视觉/无雷达)
        obs = torch.stack([
            torch.clamp(delta[:, 0] / self.world_size, -1.0, 1.0),   # 1. 相对目标 X
            torch.clamp(delta[:, 1] / self.world_size, -1.0, 1.0),   # 2. 相对目标 Y
            torch.sin(self.heading),                                 # 3. 自身朝向 sin
            torch.cos(self.heading),                                 # 4. 自身朝向 cos
            torch.sin(angle_error),                                  # 5. 目标夹角 error sin
            torch.cos(angle_error),                                  # 6. 目标夹角 error cos
            torch.clamp(distance / self.world_size, 0.0, 1.0),       # 7. 归一化目标距离
            torch.clamp(self.stuck_time / 3.0, 0.0, 1.0),           # 8. 物理阻力卡顿感知
            collision_touch,                                         # 9. 碰撞体感触觉
            low_obs_signal                                           # 10. 矮障碍接近 / 自维护跳跃探测
        ], dim=1)
        return obs

    def _get_low_proximity_signal(self):
        """仅用于仿真奖励或旧 proximity 观测；部署 jump_probe 不依赖障碍物真值。"""
        # 前方矮桩触觉感知 (前方 ±45° 视角，80 单位内最近矮桩)
        forward_dir = torch.stack([torch.cos(self.heading), torch.sin(self.heading)], dim=1)
        diff_low = self.low_centers - self.pos.unsqueeze(1)
        dist_low = torch.norm(diff_low, dim=2)
        dir_low = diff_low / torch.clamp(dist_low.unsqueeze(2), min=1e-6)
        dot_low = (dir_low * forward_dir.unsqueeze(1)).sum(dim=2)
        in_cone = (dot_low > 0.707) & (dist_low < 80.0)
        dist_front = torch.where(in_cone, dist_low, torch.full_like(dist_low, 1e5))
        min_dist_front, _ = dist_front.min(dim=1)
        return torch.where(
            min_dist_front < 80.0,
            1.0 - (min_dist_front / 80.0),
            torch.zeros_like(min_dist_front),
        )

    def _speed_for_action(self, speed_bin):
        """Map the policy speed head to a physical speed.

        V4 overrides this hook to apply a freshness limit when localization is
        old. Keeping it here lets the existing collision solver stay shared.
        """
        return torch.where(speed_bin == 1, 100.0, 50.0)

    def step(self, actions):
        self.step_count += 1
        self.time_since_collision += 0.3

        angle_bin = actions[:, 0].clamp(0, 6)
        speed_bin = actions[:, 1].clamp(0, 1)
        jump_bin = actions[:, 2].clamp(0, 1)
        start_macro = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        delta = self.target - self.pos
        target_angle = torch.atan2(delta[:, 1], delta[:, 0])

        recovery_mode = ((self.time_since_collision < 0.9) |
                         (self.no_progress_time > 0.5) |
                         (self.stuck_time > 0.5))
        if self.action_mode == "heading_relative":
            # 相对当前朝向: 转向可累积, 能持续绕行/掉头
            desired_angle = self.heading + self.ANGLE_OFFSETS_HEADING[angle_bin] * math.pi
        elif self.action_mode == "hybrid":
            # 空旷区始终围绕目标方向做小偏角，避免累计成蛇形；碰撞/停滞后改为
            # 朝向相对转向，使同一档动作可连续累积并完成大角度脱困。
            goal_angle = target_angle + self.ANGLE_OFFSETS_HYBRID[angle_bin] * math.pi
            recovery_angle = self.heading + self.ANGLE_OFFSETS_HEADING[angle_bin] * math.pi
            desired_angle = torch.where(recovery_mode, recovery_angle, goal_angle)
        else:
            desired_angle = target_angle + self.ANGLE_OFFSETS[angle_bin] * math.pi

        if self.macro:
            # 宏状态机: 仅在"未处于锁定中"时才允许起新宏; 锁定期内朝固定绝对朝向走
            macro_bin = actions[:, 3].clamp(0, 7)
            start_macro = (macro_bin > 0) & (self.macro_left <= 0)
            new_heading = self.heading + self.MACRO_TURN[macro_bin]
            self.macro_heading = torch.where(start_macro, new_heading, self.macro_heading)
            self.macro_left = torch.where(start_macro, self.MACRO_HOLD[macro_bin], self.macro_left)

            in_macro = self.macro_left > 0
            desired_angle = torch.where(in_macro, self.macro_heading, desired_angle)
            self.macro_left = torch.clamp(self.macro_left - in_macro.int(), min=0)

        speed = self._speed_for_action(speed_bin)
        jump_active = (jump_bin == 1)

        max_turn = math.radians(45.0)  # 每步允许高达 45° 的极速切弯避障
        ang_diff = desired_angle - self.heading
        ang_diff = torch.atan2(torch.sin(ang_diff), torch.cos(ang_diff))
        ang_diff_clamped = torch.clamp(ang_diff, -max_turn, max_turn)
        turn_fraction = ang_diff_clamped.abs() / max_turn
        self.heading = torch.atan2(torch.sin(self.heading + ang_diff_clamped), torch.cos(self.heading + ang_diff_clamped))

        intended_delta = torch.stack([torch.cos(self.heading), torch.sin(self.heading)], dim=1) * speed.unsqueeze(1) * self.dt
        candidate = self.pos + intended_delta

        old_pos = self.pos.clone()

        # 树木碰撞
        diff_t = candidate.unsqueeze(1) - self.tree_centers
        dist_t = torch.norm(diff_t, dim=2)
        tree_collided = dist_t < self.tree_radii
        any_tree_collided = tree_collided.any(dim=1)

        # 山体碰撞
        P_cand = candidate.unsqueeze(1).unsqueeze(2)
        C = self.mountain_vertices
        D = self.mountain_vertices_roll
        c_x, c_y = C[..., 0], C[..., 1]
        d_x, d_y = D[..., 0], D[..., 1]
        p_x, p_y = P_cand[..., 0], P_cand[..., 1]

        cross_prod = (d_x - c_x) * (p_y - c_y) - (d_y - c_y) * (p_x - c_x)
        inside_all_edges = (cross_prod >= -1e-4).all(dim=2)
        any_mountain_collided = inside_all_edges.any(dim=1)

        # 矮桩碰撞 (跃过可通行)
        diff_l = candidate.unsqueeze(1) - self.low_centers
        dist_l = torch.norm(diff_l, dim=2)
        low_collided = dist_l < self.low_radii
        any_low_collided = low_collided.any(dim=1)

        jump_pass_low = jump_active & any_low_collided
        block_low_collided = any_low_collided & ~jump_pass_low
        self.last_jump_pass_low = jump_pass_low
        self.last_block_low_collided = block_low_collided

        any_collided = any_tree_collided | any_mountain_collided | block_low_collided

        # 100% GPU 向量化物理碰撞解算 (消除 Python 循环，吞吐量提升 60 倍)
        batch_inds = torch.arange(self.num_envs, device=self.device)

        # 树木碰撞向量化滑动
        dist_t_masked = torch.where(tree_collided, dist_t, torch.full_like(dist_t, 1e5))
        min_tree_idx = torch.argmin(dist_t_masked, dim=1)
        center_t = self.tree_centers[batch_inds, min_tree_idx]
        radius_t = self.tree_radii[batch_inds, min_tree_idx]
        dir_t = candidate - center_t
        norm_t = torch.norm(dir_t, dim=1, keepdim=True)
        resolved_tree = center_t + (dir_t / torch.clamp(norm_t, min=1e-6)) * (radius_t.unsqueeze(1) + 0.1)

        # 矮桩碰撞向量化滑动
        dist_l_masked = torch.where(block_low_collided.unsqueeze(1), dist_l, torch.full_like(dist_l, 1e5))
        min_low_idx = torch.argmin(dist_l_masked, dim=1)
        center_l = self.low_centers[batch_inds, min_low_idx]
        radius_l = self.low_radii[batch_inds, min_low_idx]
        dir_l = candidate - center_l
        norm_l = torch.norm(dir_l, dim=1, keepdim=True)
        resolved_low = center_l + (dir_l / torch.clamp(norm_l, min=1e-6)) * (radius_l.unsqueeze(1) + 0.1)

        # 山体碰撞向量化滑动 (找最近边法线推出)
        mountain_hit = inside_all_edges  # (num_envs, mountain_count) already reduced
        hit_mountain_idx = mountain_hit.float().argmax(dim=1)  # (num_envs,)
        C_hit = self.mountain_vertices[batch_inds, hit_mountain_idx]  # (num_envs, 8, 2)
        D_hit = self.mountain_vertices_roll[batch_inds, hit_mountain_idx]  # (num_envs, 8, 2)
        edge_vec = D_hit - C_hit  # (num_envs, 8, 2)
        point_vec = candidate.unsqueeze(1) - C_hit  # (num_envs, 8, 2)
        edge_len_sq = (edge_vec ** 2).sum(dim=2)  # (num_envs, 8)
        t_proj = torch.clamp((point_vec * edge_vec).sum(dim=2) / torch.clamp(edge_len_sq, min=1e-6), 0.0, 1.0)
        closest_on_edge = C_hit + t_proj.unsqueeze(2) * edge_vec  # (num_envs, 8, 2)
        dist_to_edge = torch.norm(candidate.unsqueeze(1) - closest_on_edge, dim=2)  # (num_envs, 8)
        min_edge_idx = dist_to_edge.argmin(dim=1)  # (num_envs,)
        nearest_pt = closest_on_edge[batch_inds, min_edge_idx]  # (num_envs, 2)
        edge_dir = edge_vec[batch_inds, min_edge_idx]  # (num_envs, 2)
        normal = torch.stack([-edge_dir[:, 1], edge_dir[:, 0]], dim=1)
        normal = normal / torch.clamp(torch.norm(normal, dim=1, keepdim=True), min=1e-6)
        # 确保法线朝外 (远离山体中心)
        m_center = C_hit.mean(dim=1)  # (num_envs, 2)
        outward = nearest_pt - m_center
        flip_mask = (normal * outward).sum(dim=1, keepdim=True) < 0
        normal = torch.where(flip_mask, -normal, normal)
        resolved_mountain = nearest_pt + normal * 0.5

        resolved = torch.where(any_tree_collided.unsqueeze(1), resolved_tree, resolved_low)
        resolved = torch.where(any_mountain_collided.unsqueeze(1), resolved_mountain, resolved)

        # 碰撞模式: slide=完整切线滑动(旧) / weak=削弱滑动(向old_pos插值) / hard=硬停回退原位
        if self.collision_mode == "hard":
            # 硬停: 撞到就回退原位, 逼网络自己转向 (最贴近无滑动游戏, worst-case sim2real)
            resolved = old_pos
        elif self.collision_mode == "weak":
            # 削弱滑动: 只保留 25% 滑动位移, 大部分被挡回, 顶墙几乎不前进
            resolved = old_pos + (resolved - old_pos) * 0.25

        new_pos = torch.where(any_collided.unsqueeze(1), resolved, candidate)
        self.pos = new_pos
        self.pos = torch.clamp(self.pos, -1100.0, 1100.0)

        # 运动学统计与阻力
        displacement = torch.norm(self.pos - old_pos, dim=1)
        self.stuck_time = torch.where(displacement < 2.0, self.stuck_time + self.dt, torch.zeros_like(self.stuck_time))
        self.time_since_collision = torch.where(any_collided, torch.zeros_like(self.time_since_collision), self.time_since_collision)
        self.jump_probe_signal = torch.where(
            any_collided,
            torch.where(
                jump_active,
                -torch.ones_like(self.jump_probe_signal),
                torch.ones_like(self.jump_probe_signal),
            ),
            torch.zeros_like(self.jump_probe_signal),
        )

        curr_dist = torch.norm(self.target - self.pos, dim=1)
        progress = self.prev_dist - curr_dist
        self.no_progress_time = torch.where(progress < 0.2, self.no_progress_time + self.dt, torch.zeros_like(self.no_progress_time))
        self.prev_dist = curr_dist

        # 奖励计算 (兼顾主导航推进与主动避障)
        angle_err = self._angle_error()
        angle_penalty = 1.0 - torch.cos(angle_err)

        reward = torch.clamp(progress, -40.0, 40.0) * 0.25 - self.step_cost
        reward = reward - angle_penalty * self.angle_penalty_coef

        # 1. 盲人纯触觉碰撞与打湾切角脱困 (Proprioceptive Wall-Unsticking)
        current_obs = self._get_obs()
        collision_touch = current_obs[:, 8]
        low_proximity_signal = self._get_low_proximity_signal()

        # 任何非正对目标的转向都算主动避障 (bin 0,1,2,4,5,6 = 掉头/大切/精调；bin 3=0°直顶)
        is_steering = (angle_bin != 3)          # 只要不是 0° 直顶就算尝试转向
        is_big_turn = (angle_bin == 1) | (angle_bin == 5) | (angle_bin == 0) | (angle_bin == 6)  # 大角度绕行
        is_stuck = (collision_touch > 0.5) | (self.no_progress_time > 0.5)

        # 当体感到碰撞/卡死阻力时，盲人主动打角拐弯给予脱困重奖 (大角度更高)
        steer_unstick_reward = self.recovery_reward_scale * torch.where(
            is_stuck & is_big_turn,
            3.5,
            torch.where(is_stuck & is_steering, 1.5, 0.0),
        )
        collision_penalty = self.collision_penalty_scale * torch.where(
            any_tree_collided | any_mountain_collided, 1.5, 0.0
        )
        low_block_penalty = self.low_block_penalty_scale * torch.where(
            block_low_collided, 1.0, 0.0
        )

        # 顶墙不转向重罚: 撞墙/卡死时仍选 0° 直顶 → 持续加压惩罚 (逼其放弃直顶、主动绕行)
        head_on_stuck = is_stuck & (angle_bin == 3) & (collision_touch > 0.5)
        head_on_penalty = torch.where(head_on_stuck, 2.0, 0.0)

        stuck_pressure = torch.clamp(self.no_progress_time - 0.5, 0.0, 3.0)
        reward = reward - 0.40 * stuck_pressure - collision_penalty - low_block_penalty + steer_unstick_reward - head_on_penalty
        reward = reward - torch.where(
            ~is_stuck, turn_fraction * self.free_turn_penalty, torch.zeros_like(reward))
        far_slow = (curr_dist >= 120.0) & (speed_bin == 0)
        reward = reward - torch.where(
            far_slow, torch.full_like(reward, self.far_slow_penalty), torch.zeros_like(reward))

        # 1b. 脱困宏: 卡住时起宏给奖(鼓励用大角度机动逃离), 没卡时起宏给小罚(防止无脑刷宏)
        if self.macro:
            fired = start_macro
            reward = reward + torch.where(fired & is_stuck, 3.0, 0.0)
            reward = reward - torch.where(fired & ~is_stuck, 0.5, 0.0)

        # 2. 矮桩触觉起跳过栏
        near_low = (low_proximity_signal > 0.25)
        reward = torch.where(jump_pass_low, reward + 5.0, reward)
        reward = torch.where(near_low & ~jump_active, reward - 1.5, reward)
        reward = torch.where(~near_low & jump_active, reward - self.false_jump_penalty, reward)

        # 3. 接近目标精准进站 (走自身-终点连线 + 降速，消除近终点转圈震荡)
        near_target = curr_dist < 120.0
        zeros_r = torch.zeros_like(reward)
        # 接近时加大连线对准奖励 (angle_penalty=1-cos(err)，越对准越小)
        reward = reward - torch.where(near_target, angle_penalty * 0.8, zeros_r)
        # 接近时偏离连线 → 惩罚，逼其走连线。
        # 注意: target_relative 下 bin3 就是"正对目标"; heading_relative 下 bin3 只是"不转向",
        # 与是否对准目标无关, 必须改用真实夹角判定, 否则这一项会教出错误行为。
        if self.action_mode in ("heading_relative", "hybrid"):
            off_line = near_target & (angle_err.abs() > math.radians(15.0))
        else:
            off_line = near_target & (angle_bin != 3)
        reward = torch.where(off_line, reward - 0.6, reward)
        # 接近时仍全速冲刺 → 惩罚，鼓励降速稳入圈
        near_fast = near_target & (speed_bin == 1)
        reward = torch.where(near_fast, reward - 0.4, reward)

        reached = curr_dist < 20.0
        truncated = self.step_count >= 800
        dones = reached | truncated

        reward = torch.where(reached, reward + 20.0, reward)
        reward = torch.where(truncated & ~reached, reward - 5.0, reward)

        if self.auto_reset and dones.any():
            done_indices = torch.where(dones)[0]
            num_dones = len(done_indices)

            r_pos, r_target = self._sample_start_and_target(num_dones)
            self.pos[done_indices] = r_pos
            self.target[done_indices] = r_target
            self._generate_obstacles_for_envs(done_indices, r_pos, r_target)

            self.heading[done_indices] = (torch.rand(num_dones, device=self.device) * 2.0 - 1.0) * math.pi
            self.macro_left[done_indices] = 0      # 宏状态必须随 episode 重置, 否则会串到下一局
            self.jump_probe_signal[done_indices] = 0.0
            self.step_count[done_indices] = 0
            self.stuck_time[done_indices] = 0.0
            self.no_progress_time[done_indices] = 0.0
            self.time_since_collision[done_indices] = 10.0
            self.prev_dist[done_indices] = torch.norm(self.target[done_indices] - self.pos[done_indices], dim=1)
            self.prev_angle_error[done_indices] = self._angle_error()[done_indices]

        return self._get_obs(), reward, dones, reached


class GPUUnknownHeadingNavEnv(GPUVectorizedBlindNavEnv):
    """V4 environment with delayed localization and no heading observation.

    The inherited environment keeps the true heading for hidden physics and
    reward calculation. This layer exposes only delayed position feedback and
    estimates motion direction from timestamp-aligned displacement.
    """

    OBS_DIM = OBS_DIM
    MIN_VALID_SAMPLE_DT = 0.05
    MAX_VALID_SAMPLE_DT = 1.5
    MAX_REASONABLE_SPEED = 500.0
    CALIBRATION_SAMPLES_REQUIRED = 2

    def __init__(self, num_envs=8192, world_size=2250.0, dt=0.30, device="cuda",
                 auto_reset=True, collision_mode="hard", macro=False,
                 angle_penalty_coef=1.00, step_cost=0.12, free_turn_penalty=0.18,
                 far_slow_penalty=0.15, hybrid_free_max_deg=10.0,
                 obstacle_signal_mode="jump_probe", false_jump_penalty=0.05,
                 position_age_max_ms=500.0, position_stale_ms=1000.0,
                 action_mode="unknown_heading", obstacle_density=1.0,
                 position_age_extreme_prob=0.4, recovery_commit_steps=4,
                 recovery_reward_scale=1.0):
        self._v4_initializing = True
        super().__init__(
            num_envs=num_envs,
            world_size=world_size,
            dt=dt,
            device=device,
            auto_reset=auto_reset,
            collision_mode=collision_mode,
            action_mode="heading_relative",
            macro=macro,
            angle_penalty_coef=angle_penalty_coef,
            step_cost=step_cost,
            free_turn_penalty=free_turn_penalty,
            far_slow_penalty=far_slow_penalty,
            hybrid_free_max_deg=hybrid_free_max_deg,
            obstacle_signal_mode=obstacle_signal_mode,
            false_jump_penalty=false_jump_penalty,
            obstacle_density=obstacle_density,
            position_age_extreme_prob=position_age_extreme_prob,
            recovery_commit_steps=recovery_commit_steps,
            recovery_reward_scale=recovery_reward_scale,
        )
        self._v4_initializing = False
        self.position_age_max_ms = float(position_age_max_ms)
        self.position_stale_ms = float(position_stale_ms)
        self.fixed_position_age_ms = None
        self.stale_target_assist = False
        self.recovery_signal_window = 0.6
        self.turn_offsets = torch.tensor(
            [-math.radians(45.0), -math.radians(25.0), -math.radians(10.0), 0.0,
             math.radians(10.0), math.radians(25.0), math.radians(45.0)],
            dtype=torch.float32,
            device=device,
        )
        self._allocate_belief_state()
        GPUVectorizedBlindNavEnv.reset(self)
        self._reset_belief_state()
        self._refresh_observation()

    def _allocate_belief_state(self):
        shape = (self.num_envs, 2)
        self.prev_actual_pos = torch.zeros(shape, dtype=torch.float32, device=self.device)
        self.prev_prev_actual_pos = torch.zeros_like(self.prev_actual_pos)
        self.observed_pos = torch.zeros_like(self.prev_actual_pos)
        self.predicted_pos = torch.zeros_like(self.prev_actual_pos)
        self.observed_age_ms = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.velocity = torch.zeros_like(self.prev_actual_pos)
        self.estimated_heading = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.heading_valid = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.heading_confidence = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.calibration_samples = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
        self.no_valid_observation_steps = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
        self.probe_phase = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
        self.last_turn_delta = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.recovery_commit_left = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
        self.recovery_phase = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
        self.turn_history = torch.zeros((self.num_envs, 2), dtype=torch.float32, device=self.device)

    def _reset_belief_state(self, indices=None):
        if indices is None:
            indices = torch.arange(self.num_envs, device=self.device)
        self.prev_actual_pos[indices] = self.pos[indices]
        self.prev_prev_actual_pos[indices] = self.pos[indices]
        self.observed_pos[indices] = self.pos[indices]
        self.predicted_pos[indices] = self.pos[indices]
        self.observed_age_ms[indices] = 0.0
        self.velocity[indices] = 0.0
        self.estimated_heading[indices] = 0.0
        self.heading_valid[indices] = False
        self.heading_confidence[indices] = 0.0
        self.calibration_samples[indices] = 0
        self.no_valid_observation_steps[indices] = 0
        self.probe_phase[indices] = 0
        self.last_turn_delta[indices] = 0.0
        self.recovery_commit_left[indices] = 0
        self.recovery_phase[indices] = 0
        self.turn_history[indices] = 0.0

    def reset(self, seed=None):
        if getattr(self, "_v4_initializing", False):
            return GPUVectorizedBlindNavEnv.reset(self, seed)
        result = GPUVectorizedBlindNavEnv.reset(self, seed)
        self._reset_belief_state()
        self._refresh_observation()
        return self._get_obs()

    @staticmethod
    def _normalize_angle(value):
        return torch.atan2(torch.sin(value), torch.cos(value))

    def _sample_delayed_position(self, age_ms):
        age = torch.clamp(age_ms / 1000.0, min=0.0, max=self.position_age_max_ms / 1000.0)
        recent_fraction = torch.clamp(age / self.dt, 0.0, 1.0).unsqueeze(1)
        recent = (
            self.pos * (1.0 - recent_fraction)
            + self.prev_actual_pos * recent_fraction
        )
        older_fraction = torch.clamp((age - self.dt) / self.dt, 0.0, 1.0).unsqueeze(1)
        older = (
            self.prev_actual_pos * (1.0 - older_fraction)
            + self.prev_prev_actual_pos * older_fraction
        )
        return torch.where((age <= self.dt).unsqueeze(1), recent, older)

    def _refresh_observation(self, active=None):
        if active is None:
            active = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)

        if self.fixed_position_age_ms is None:
            sampled_age = torch.rand(self.num_envs, device=self.device) * self.position_age_max_ms
            extreme_roll = torch.rand(self.num_envs, device=self.device)
            extreme_bucket = torch.floor(extreme_roll * 3.0).long().clamp(max=2)
            extreme_age = extreme_bucket.float() * (self.position_age_max_ms / 2.0)
            sampled_age = torch.where(
                extreme_roll < self.position_age_extreme_prob,
                extreme_age,
                sampled_age,
            )
        else:
            sampled_age = torch.full(
                (self.num_envs,),
                float(self.fixed_position_age_ms),
                dtype=torch.float32,
                device=self.device,
            )
        age_ms = torch.where(active, sampled_age, torch.zeros_like(sampled_age))
        delayed_pos = self._sample_delayed_position(age_ms)

        age_delta = (age_ms - self.observed_age_ms) / 1000.0
        sample_dt = self.dt - age_delta
        observed_delta = delayed_pos - self.observed_pos
        displacement = torch.norm(observed_delta, dim=1)
        sample_speed = displacement / torch.clamp(sample_dt, min=self.MIN_VALID_SAMPLE_DT)
        valid = (
            active
            & (sample_dt >= self.MIN_VALID_SAMPLE_DT)
            & (sample_dt <= self.MAX_VALID_SAMPLE_DT)
            & (displacement >= 2.0)
            & (sample_speed <= self.MAX_REASONABLE_SPEED)
        )

        velocity_sample = observed_delta / torch.clamp(sample_dt.unsqueeze(1), min=self.MIN_VALID_SAMPLE_DT)
        sample_norm = torch.norm(velocity_sample, dim=1, keepdim=True)
        velocity_sample = velocity_sample * torch.clamp(
            self.MAX_REASONABLE_SPEED / torch.clamp(sample_norm, min=1e-6),
            max=1.0,
        )
        replay_turn = torch.where(
            (age_ms > self.dt * 1000.0),
            self.turn_history[:, 0],
            torch.zeros_like(age_ms),
        )
        replay_turn = replay_turn + torch.where(
            (age_ms > self.dt * 2000.0),
            self.turn_history[:, 1],
            torch.zeros_like(age_ms),
        )
        velocity_x = velocity_sample[:, 0] * torch.cos(replay_turn) - velocity_sample[:, 1] * torch.sin(replay_turn)
        velocity_y = velocity_sample[:, 0] * torch.sin(replay_turn) + velocity_sample[:, 1] * torch.cos(replay_turn)
        velocity_current = torch.stack([velocity_x, velocity_y], dim=1)
        self.velocity = torch.where(
            valid.unsqueeze(1),
            self.velocity * 0.5 + velocity_current * 0.5,
            self.velocity,
        )

        predicted_heading = self._normalize_angle(self.estimated_heading + self.last_turn_delta)
        measured_heading = torch.atan2(velocity_sample[:, 1], velocity_sample[:, 0])
        corrected_heading = self._normalize_angle(measured_heading + replay_turn)
        first_measurement = valid & ~self.heading_valid
        has_previous_heading = valid & self.heading_valid
        heading_error = self._normalize_angle(corrected_heading - predicted_heading).abs()
        consistent = heading_error <= math.radians(35.0)
        accepted_measurement = valid

        self.estimated_heading = torch.where(
            first_measurement,
            corrected_heading,
            torch.where(
                valid,
                self._normalize_angle(
                    predicted_heading
                    + self._normalize_angle(corrected_heading - predicted_heading) * 0.35
                ),
                predicted_heading,
            ),
        )
        self.heading_valid = self.heading_valid | valid
        self.calibration_samples = torch.where(
            accepted_measurement,
            torch.clamp(self.calibration_samples + 1, max=3),
            self.calibration_samples,
        )
        confidence_up = torch.where(consistent | first_measurement, 0.25, -0.15)
        self.heading_confidence = torch.where(
            valid,
            torch.clamp(self.heading_confidence + confidence_up, 0.0, 1.0),
            self.heading_confidence * 0.98,
        )
        self.no_valid_observation_steps = torch.where(
            valid,
            torch.zeros_like(self.no_valid_observation_steps),
            self.no_valid_observation_steps + 1,
        )
        self.probe_phase = torch.where(
            valid,
            torch.zeros_like(self.probe_phase),
            self.probe_phase,
        )

        prediction_age = torch.clamp(age_ms, min=0.0, max=self.position_age_max_ms) / 1000.0
        self.observed_pos = delayed_pos
        self.predicted_pos = torch.clamp(
            delayed_pos + self.velocity * prediction_age.unsqueeze(1),
            -1100.0,
            1100.0,
        )
        self.observed_age_ms = age_ms

    def _freshness_scale(self):
        age = self.observed_age_ms
        scale = torch.where(
            age <= 200.0,
            torch.ones_like(age),
            torch.where(
                age <= 500.0,
                1.0 - 0.5 * ((age - 200.0) / 300.0),
                torch.where(
                    age <= self.position_stale_ms,
                    0.5 * ((self.position_stale_ms - age) / (self.position_stale_ms - 500.0)),
                    torch.zeros_like(age),
                ),
            ),
        )
        return torch.clamp(scale, 0.0, 1.0)

    def _speed_for_action(self, speed_bin):
        speed = super()._speed_for_action(speed_bin)
        if getattr(self, "_v4_initializing", False):
            return speed
        return speed * self._freshness_scale()

    def apply_calibration(self, actions):
        """Force the short deterministic calibration sequence when needed."""
        calibrating = self.calibration_samples < self.CALIBRATION_SAMPLES_REQUIRED
        forced = actions.clone()
        forced[calibrating, 0] = 3
        forced[calibrating, 1] = 0
        forced[calibrating, 2] = 0

        probing = calibrating & (self.no_valid_observation_steps >= 2)
        probe_bin = torch.where(self.probe_phase == 0, 5, 1)
        forced[probing, 0] = probe_bin[probing]
        self.probe_phase = torch.where(
            probing,
            1 - self.probe_phase,
            self.probe_phase,
        )

        # Keep one large relative turn committed for a few frames after a
        # hard collision. Without this, the recurrent policy can alternate
        # opposite bins while its physical position remains unchanged.
        recovery_signal = (
            (self.time_since_collision < self.recovery_signal_window)
            | (self.stuck_time > 0.5)
        ) & ~calibrating & (not self.macro)
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
        self.recovery_phase = torch.where(
            start_commit,
            1 - self.recovery_phase,
            self.recovery_phase,
        )

        if self.stale_target_assist:
            stale = (
                (self.observed_age_ms >= 400.0)
                & (self.heading_confidence >= 0.5)
                & ~recovery_signal
                & ~calibrating
            )
            delta = self.target - self.predicted_pos
            target_angle = torch.atan2(delta[:, 1], delta[:, 0])
            target_error = self._normalize_angle(target_angle - self.estimated_heading)
            target_distance = self._normalize_angle(
                target_error.unsqueeze(1) - self.turn_offsets.unsqueeze(0)
            ).abs()
            target_bin = target_distance.argmin(dim=1)
            forced[stale, 0] = target_bin[stale]
        return forced

    def step(self, actions):
        old_pos = self.pos.clone()
        angle_bin = actions[:, 0].clamp(0, 6)
        self.last_turn_delta = self.turn_offsets[angle_bin]
        self.turn_history[:, 1] = self.turn_history[:, 0]
        self.turn_history[:, 0] = self.last_turn_delta
        result = GPUVectorizedBlindNavEnv.step(self, actions)
        _, reward, dones, reached = result

        active = ~dones
        if active.any():
            self.prev_prev_actual_pos[active] = self.prev_actual_pos[active]
            self.prev_actual_pos[active] = old_pos[active]
        if dones.any():
            self._reset_belief_state(torch.where(dones)[0])
        self._refresh_observation(active=active)
        return self._get_obs(), reward, dones, reached

    def _get_obs(self):
        if getattr(self, "_v4_initializing", False):
            return GPUVectorizedBlindNavEnv._get_obs(self)

        delta = self.target - self.predicted_pos
        distance = torch.norm(delta, dim=1)
        target_angle = torch.atan2(delta[:, 1], delta[:, 0])
        angle_error = self._normalize_angle(target_angle - self.estimated_heading)

        low_proximity_signal = self._get_low_proximity_signal()
        low_obs_signal = (
            self.jump_probe_signal
            if self.obstacle_signal_mode == "jump_probe"
            else low_proximity_signal
        )
        collision_touch = torch.where(self.time_since_collision < 0.6, 1.0, 0.0)

        # 保持 collision_touch 在第 9 个位置，便于复用父类奖励逻辑。
        obs = torch.stack([
            torch.clamp(delta[:, 0] / self.world_size, -1.0, 1.0),
            torch.clamp(delta[:, 1] / self.world_size, -1.0, 1.0),
            torch.sin(angle_error),
            torch.cos(angle_error),
            torch.clamp(self.velocity[:, 0] / self.MAX_REASONABLE_SPEED, -1.0, 1.0),
            torch.clamp(self.velocity[:, 1] / self.MAX_REASONABLE_SPEED, -1.0, 1.0),
            torch.clamp(distance / self.world_size, 0.0, 1.0),
            torch.clamp(self.stuck_time / 3.0, 0.0, 1.0),
            collision_touch,
            low_obs_signal,
            torch.clamp(self.observed_age_ms / self.position_age_max_ms, 0.0, 1.0),
            self.heading_confidence,
            torch.clamp(self.last_turn_delta / MAX_TURN_RAD, -1.0, 1.0),
        ], dim=1)
        return obs


GPUBlindNavEnvV11b = GPUUnknownHeadingNavEnv

# =====================================================================
# 2. PPO Recurrent (LSTM) 网络架构
# =====================================================================
class RecurrentActorCritic(nn.Module):
    def __init__(self, state_dim=OBS_DIM, hidden_dim=96, actor_width=48, macro=False):
        super().__init__()
        self.lstm = nn.LSTM(state_dim, hidden_dim, num_layers=1)
        self.macro = macro

        self.actor_fc = nn.Sequential(
            nn.Linear(hidden_dim, actor_width),
            nn.Tanh()
        )
        self.steer_head = nn.Linear(actor_width, 7)  # 相对运动方向: -45/-25/-10/0/+10/+25/+45
        self.speed_head = nn.Linear(actor_width, 2)  # 2档基础速度 (50/100)，再由新鲜度限速
        self.jump_head = nn.Linear(actor_width, 2)   # 2档跳跃 (0/1)
        # 8档脱困宏 (0=正常导航, 1-7=锁定若干步的大角度恢复机动)
        self.macro_head = nn.Linear(actor_width, 8) if macro else None

        self.critic_fc = nn.Sequential(
            nn.Linear(actor_width, actor_width),
            nn.Tanh(),
            nn.Linear(actor_width, 1)
        )

    def get_states(self, x, lstm_state, dones):
        seq_len = x.size(0)
        h, c = lstm_state
        if seq_len == 1 or not bool(torch.any(dones > 0.5).item()):
            return self.lstm(x, (h, c))

        # Reset masks vary by environment. Run each maximal interval without
        # a reset as one cuDNN LSTM call instead of launching one call per
        # timestep; this preserves the exact hidden-state reset semantics.
        boundary_ends = (torch.where(torch.any(dones[:-1] > 0.5, dim=1))[0] + 1).tolist()
        boundaries = [0] + boundary_ends + [seq_len]
        output_parts = []
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            out, (h, c) = self.lstm(x[start:end], (h, c))
            output_parts.append(out)
            if end < seq_len:
                reset_mask = (dones[end - 1] > 0.5).view(1, -1, 1)
                h = torch.where(reset_mask, torch.zeros_like(h), h)
                c = torch.where(reset_mask, torch.zeros_like(c), c)
        return torch.cat(output_parts, dim=0), (h, c)

    @staticmethod
    def _categorical_stats(logits, actions):
        log_probs = F.log_softmax(logits, dim=-1)
        probs = log_probs.exp()
        selected = log_probs.gather(-1, actions.unsqueeze(-1)).squeeze(-1)
        entropy = -(probs * log_probs).sum(dim=-1)
        return selected, entropy

    @staticmethod
    def _sample_categorical(logits):
        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, 1).squeeze(-1)

    def evaluate_actions(self, x, lstm_state, dones, actions):
        seq_len = x.size(0)
        num_envs = x.size(1)

        lstm_outputs, _ = self.get_states(x, lstm_state, dones)
        features = self.actor_fc(lstm_outputs.view(-1, lstm_outputs.size(-1)))

        logits_steer = self.steer_head(features)
        logits_speed = self.speed_head(features)
        logits_jump = self.jump_head(features)

        n_act = 4 if self.macro_head is not None else 3
        actions_flat = actions.view(-1, n_act)
        steer_log_prob, steer_entropy = self._categorical_stats(logits_steer, actions_flat[:, 0])
        speed_log_prob, speed_entropy = self._categorical_stats(logits_speed, actions_flat[:, 1])
        jump_log_prob, jump_entropy = self._categorical_stats(logits_jump, actions_flat[:, 2])
        log_prob = steer_log_prob + speed_log_prob + jump_log_prob
        entropy = steer_entropy + speed_entropy + jump_entropy

        if self.macro_head is not None:
            macro_logits = self.macro_head(features)
            macro_log_prob, macro_entropy = self._categorical_stats(macro_logits, actions_flat[:, 3])
            log_prob = log_prob + macro_log_prob
            entropy = entropy + macro_entropy

        log_prob = log_prob.view(seq_len, num_envs)
        entropy = entropy.view(seq_len, num_envs)

        values = self.critic_fc(features).view(seq_len, num_envs)
        macro_logits_out = None
        if self.macro_head is not None:
            macro_logits_out = macro_logits.view(seq_len, num_envs, 8)
        return log_prob, entropy, values, logits_steer.view(seq_len, num_envs, 7), macro_logits_out

    def steer_logits_for_sequence(self, x, lstm_state, dones):
        """Return steering logits for the auxiliary angle-to-turn loss."""
        lstm_outputs, _ = self.get_states(x, lstm_state, dones)
        features = self.actor_fc(lstm_outputs.view(-1, lstm_outputs.size(-1)))
        return self.steer_head(features).view(x.size(0), x.size(1), 7)


def pretrain_steering(agent, device, steps=300, batch_envs=2048, seq_len=32, lr=1e-3):
    """Teach the recurrent policy the local angle-to-turn mapping before PPO.

    The labels use only the same angle-error channels exposed at inference;
    no hidden simulator heading is used in this phase.
    """
    if steps <= 0:
        return
    optimizer = optim.Adam(
        list(agent.lstm.parameters())
        + list(agent.actor_fc.parameters())
        + list(agent.steer_head.parameters()),
        lr=lr,
    )
    turn_offsets = torch.tensor(
        [-math.radians(45.0), -math.radians(25.0), -math.radians(10.0), 0.0,
         math.radians(10.0), math.radians(25.0), math.radians(45.0)],
        dtype=torch.float32,
        device=device,
    )
    agent.train()
    for step in range(steps):
        obs = torch.zeros((seq_len, batch_envs, OBS_DIM), device=device)
        angle_error = (torch.rand((seq_len, batch_envs), device=device) * 2.0 - 1.0) * math.pi
        obs[:, :, 2] = torch.sin(angle_error)
        obs[:, :, 3] = torch.cos(angle_error)
        obs[:, :, 4:6] = torch.randn((seq_len, batch_envs, 2), device=device) * 0.2
        obs[:, :, 6] = torch.rand((seq_len, batch_envs), device=device)
        obs[:, :, 7] = torch.rand((seq_len, batch_envs), device=device)
        obs[:, :, 10] = torch.rand((seq_len, batch_envs), device=device)
        obs[:, :, 11] = 0.5 + 0.5 * torch.rand((seq_len, batch_envs), device=device)
        obs[:, :, 12] = torch.randn((seq_len, batch_envs), device=device) * 0.5
        labels = torch.atan2(
            torch.sin(angle_error.unsqueeze(2) - turn_offsets.view(1, 1, -1)),
            torch.cos(angle_error.unsqueeze(2) - turn_offsets.view(1, 1, -1)),
        ).abs().argmin(dim=2)

        h = torch.zeros(1, batch_envs, agent.lstm.hidden_size, device=device)
        c = torch.zeros_like(h)
        dones = torch.zeros((seq_len, batch_envs), device=device)
        logits = agent.steer_logits_for_sequence((obs), (h, c), dones)
        loss = F.cross_entropy(logits.reshape(-1, 7), labels.reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
        optimizer.step()
        if step == 0 or (step + 1) % 50 == 0 or step + 1 == steps:
            with torch.inference_mode():
                accuracy = (logits.argmax(dim=2) == labels).float().mean().item()
            print(
                f"[转向预训练] {step + 1}/{steps} loss={loss.item():.4f} accuracy={accuracy:.3f}",
                flush=True,
            )


@torch.no_grad()
def evaluate_training_checkpoint(agent, device, episodes=64, max_steps=240, seed=20260825):
    """Run held-out simulator checks without affecting PPO rollout state."""
    was_training = agent.training
    agent.eval()
    results = []
    for age_ms in (0.0, 250.0, 500.0):
        env = GPUBlindNavEnvV11b(
            num_envs=episodes,
            device=device,
            auto_reset=False,
            collision_mode="hard",
            action_mode="unknown_heading",
            macro=agent.macro_head is not None,
            obstacle_signal_mode="jump_probe",
            position_age_max_ms=500.0,
            position_stale_ms=1000.0,
            position_age_extreme_prob=0.8,
            obstacle_density=1.0,
        )
        env.fixed_position_age_ms = age_ms
        obs = env.reset(seed=seed + int(age_ms))
        h = torch.zeros(1, episodes, agent.lstm.hidden_size, device=device)
        c = torch.zeros_like(h)
        active = torch.ones(episodes, dtype=torch.bool, device=device)
        success = torch.zeros(episodes, dtype=torch.bool, device=device)
        heading_error_sum = torch.zeros(episodes, device=device)
        heading_error_count = torch.zeros(episodes, device=device)
        calibration_steps = torch.full((episodes,), -1, dtype=torch.long, device=device)
        jump_cooldown = torch.zeros(episodes, dtype=torch.long, device=device)

        for step in range(1, max_steps + 1):
            with torch.no_grad():
                out, (h, c) = agent.get_states(
                    obs.unsqueeze(0), (h, c), torch.zeros(1, episodes, device=device)
                )
                features = agent.actor_fc(out.squeeze(0))
                action_heads = [
                    agent.steer_head(features).argmax(dim=-1),
                    agent.speed_head(features).argmax(dim=-1),
                    agent.jump_head(features).argmax(dim=-1),
                ]
                if agent.macro_head is not None:
                    macro_action = agent.macro_head(features).argmax(dim=-1)
                    macro_trigger = (obs[:, 8] > 0.5) | (obs[:, 7] > 0.25)
                    macro_action = torch.where(
                        macro_trigger, macro_action, torch.zeros_like(macro_action)
                    )
                    action_heads.append(macro_action)
                action = torch.stack(action_heads, dim=1)
                action = env.apply_calibration(action)
                action[:, 2] = ((obs[:, 9] > 0.5) & (jump_cooldown == 0)).long()
                action[~active] = torch.zeros(action.shape[1], dtype=torch.long, device=device)

            obs, _, _, reached = env.step(action)
            newly_reached = active & reached
            success |= newly_reached
            active &= ~newly_reached
            calibration_steps[newly_reached] = step
            estimate_error = torch.atan2(
                torch.sin(env.estimated_heading - env.heading),
                torch.cos(env.estimated_heading - env.heading),
            ).abs()
            heading_error_sum += torch.where(active, estimate_error, 0.0)
            heading_error_count += active.float()

            collided = env.time_since_collision == 0.0
            jump_failed = collided & (action[:, 2] == 1)
            jump_cooldown = torch.clamp(jump_cooldown - 1, min=0)
            jump_cooldown[jump_failed] = 8
            jump_cooldown[~collided] = 0

        valid_errors = torch.clamp(heading_error_count.sum(), min=1.0)
        results.append({
            "age_ms": age_ms,
            "success_rate": float(success.float().mean().item()),
            "heading_error_deg": float(
                (heading_error_sum.sum() / valid_errors * (180.0 / math.pi)).item()
            ),
            "calibration_steps_median": (
                float(torch.quantile(calibration_steps[calibration_steps >= 0].float(), 0.5).item())
                if (calibration_steps >= 0).any()
                else None
            ),
        })
    if was_training:
        agent.train()
    print(f"[独立仿真验证] {results}", flush=True)
    return results

# =====================================================================
# 3. 评估 Worker (CPU 渲染 10 评估样本)
# =====================================================================
class RenderShim:
    def __init__(self, gpu_env):
        self.gpu_env = gpu_env
        self.world_size = float(gpu_env.world_size)
        self.dt = float(gpu_env.dt)
        self.target_radius = 20.0
        self.tree_radius = 28.0
        self.mountain_min_radius = 35.0
        self.mountain_max_radius = 70.0
        self.low_radius = 16.0
        self.desired_stop_distance = 0.0
        self.stop_distance_tolerance = 8.0
        self.align_angle_tolerance = math.radians(180.0)

    @property
    def pos(self):
        return self.gpu_env.pos[0].cpu().numpy()

    @property
    def target(self):
        return self.gpu_env.target[0].cpu().numpy()

    @property
    def heading(self):
        return float(self.gpu_env.heading[0].item())

    @property
    def tree_centers(self):
        return self.gpu_env.tree_centers[0].cpu().numpy()

    @property
    def mountain_centers(self):
        return self.gpu_env.mountain_centers[0].cpu().numpy()

    @property
    def mountain_vertices(self):
        return self.gpu_env.mountain_vertices[0].cpu().numpy()

    @property
    def low_centers(self):
        return self.gpu_env.low_centers[0].cpu().numpy()

    @property
    def obstacles(self):
        return []


_training_video_player = None


@torch.no_grad()
def render_training_episode(
    agent,
    device,
    iteration,
    output_dir,
    *,
    age_ms=250.0,
    max_steps=240,
    fps=20,
    seed=20260825,
):
    """Render one fixed-map progress episode and open it in mpv asynchronously."""
    from render_eval10_concat import render_episode_video

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_video = output_dir / "最新迭代.mp4"
    temporary_video = output_dir / "最新迭代.写入中.mp4"

    env = GPUBlindNavEnvV11b(
        num_envs=1,
        device="cpu",
        auto_reset=False,
        collision_mode="hard",
        action_mode="unknown_heading",
        macro=False,
        obstacle_signal_mode="jump_probe",
        position_age_max_ms=500.0,
        position_stale_ms=1000.0,
        position_age_extreme_prob=0.8,
        obstacle_density=1.0,
    )
    env.fixed_position_age_ms = float(age_ms)
    obs = env.reset(seed=seed)
    shim = RenderShim(env)
    path = [env.pos[0].cpu().numpy().copy()]
    headings = [float(env.heading[0].item())]
    total_reward = 0.0
    hidden_dim = agent.lstm.hidden_size
    h = torch.zeros(1, 1, hidden_dim, device=device)
    c = torch.zeros_like(h)

    was_training = agent.training
    agent.eval()
    for _ in range(max_steps):
        obs_device = obs.to(device)
        with torch.inference_mode():
            out, (h, c) = agent.get_states(
                obs_device.unsqueeze(0),
                (h, c),
                torch.zeros(1, 1, device=device),
            )
            features = agent.actor_fc(out.squeeze(0))
            action = torch.stack([
                agent.steer_head(features).argmax(dim=-1),
                agent.speed_head(features).argmax(dim=-1),
                agent.jump_head(features).argmax(dim=-1),
            ], dim=1).cpu()
        action = env.apply_calibration(action)
        obs, reward, dones, _ = env.step(action)
        total_reward += float(reward[0].item())
        path.append(env.pos[0].cpu().numpy().copy())
        headings.append(float(env.heading[0].item()))
        if bool(dones[0].item()):
            break

    if was_training:
        agent.train()
    render_episode_video(
        shim,
        np.asarray(path),
        total_reward,
        temporary_video,
        fps=fps,
        headings=np.asarray(headings),
    )
    temporary_video.replace(output_video)

    global _training_video_player
    try:
        if _training_video_player is not None and _training_video_player.poll() is None:
            _training_video_player.terminate()
        _training_video_player = subprocess.Popen(
            [
                "mpv",
                "--force-window=yes",
                "--no-terminal",
                f"--title=ProprioNav V4 迭代 {iteration}",
                str(output_video),
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        player_status = "mpv-started"
    except FileNotFoundError:
        player_status = "mpv-not-found"
    print(
        f"[训练视频] iteration={iteration} age={age_ms:.0f}ms "
        f"steps={len(path) - 1} reward={total_reward:.3f} "
        f"status={player_status} file={output_video.resolve()}",
        flush=True,
    )
    return output_video


def run_and_render_seed_worker(seed, index, weights_path, fps, out_dir, hidden_dim=96, actor_width=48,
                               action_mode="unknown_heading", macro=False, collision_mode="hard",
                               sample=False, commit=0, hybrid_free_max_deg=10.0,
                               obstacle_signal_mode="jump_probe"):
    try:
        from render_eval10_concat import render_episode_video

        model = RecurrentActorCritic(state_dim=OBS_DIM, hidden_dim=hidden_dim, actor_width=actor_width, macro=macro)
        model.load_state_dict(torch.load(weights_path, map_location="cpu"))
        model.eval()
        if sample:
            torch.manual_seed(seed)
        h = torch.zeros(1, 1, hidden_dim, dtype=torch.float32)
        c = torch.zeros(1, 1, hidden_dim, dtype=torch.float32)

        env = GPUBlindNavEnvV11b(num_envs=1, device="cpu", auto_reset=False,
                                 collision_mode=collision_mode, action_mode=action_mode, macro=macro,
                                 hybrid_free_max_deg=hybrid_free_max_deg,
                                 obstacle_signal_mode=obstacle_signal_mode,
                                 position_age_max_ms=CONFIG["position_age_max_ms"],
                                 position_stale_ms=CONFIG["position_stale_ms"],
                                 position_age_extreme_prob=CONFIG["position_age_extreme_prob"],
                                 recovery_commit_steps=CONFIG["recovery_commit_steps"])
        env.reset(seed=seed)
        obs = env._get_obs()
        shim = RenderShim(env)

        path = [env.pos[0].cpu().numpy().copy()]
        headings = [float(env.heading[0].item())]
        total_reward = 0.0
        done = False
        steps = 0
        commit_left, commit_bin = 0, 3
        while not done and steps < 240:
            with torch.no_grad():
                lo, (h, c) = model.get_states(obs.view(1, 1, OBS_DIM), (h, c), torch.zeros(1, 1))
                f = model.actor_fc(lo.view(-1, lo.size(-1)))
                steer_logits = model.steer_head(f)
                action_logits = [steer_logits, model.speed_head(f), model.jump_head(f)]
                mh = model.macro_head
                if mh is not None:
                    action_logits.append(mh(f))
                if sample:
                    heads = [torch.distributions.Categorical(logits=logits).sample().item()
                             for logits in action_logits]
                else:
                    heads = [logits.argmax(-1).item() for logits in action_logits]
                if commit_left > 0:
                    heads[0] = commit_bin
                    commit_left -= 1
                action = torch.tensor([heads], dtype=torch.long)
                action = env.apply_calibration(action)
            obs, reward, dones, reached = env.step(action)
            collided = bool(env.time_since_collision[0].item() == 0.0)
            if commit > 0:
                if collided and commit_left == 0:
                    escape_bins = torch.tensor([0, 1, 5, 6])
                    escape_probs = torch.softmax(steer_logits.view(-1)[escape_bins], dim=0)
                    commit_bin = int(escape_bins[torch.multinomial(escape_probs, 1)].item())
                    commit_left = commit
                elif not collided:
                    commit_left = 0
            done = bool(dones[0].item())
            steps += 1
            path.append(env.pos[0].cpu().numpy().copy())
            headings.append(float(env.heading[0].item()))
            total_reward += float(reward[0].item())

        output_video = Path(out_dir) / f"eval_{index:02d}.mp4"
        render_episode_video(shim, np.array(path), total_reward, output_video, fps=fps, headings=headings)
        dist = float(np.linalg.norm(shim.target - path[-1]))
        is_success = dist <= shim.target_radius
        print(f"[Worker-{index}] seed={seed} steps={steps} reward={total_reward:.3f} dist={dist:.1f} success={is_success} video={output_video}", flush=True)
        return str(output_video), is_success
    except Exception as e:
        print(f"[Worker-{index}] Error processing seed {seed}: {e}", flush=True)
        import traceback
        traceback.print_exc()
        return None, False

# =====================================================================
# 4. 主执行管线 (GPU 训练 -> 并行评估 -> 拼接视频)
# =====================================================================
def train(out_dir, weights_path):
    print("==================================================")
    print("STAGE 1: 启动极速 GPU 向量化训练...")
    print("==================================================")

    num_envs = CONFIG["num_envs"]
    num_steps = CONFIG["rollout_steps"]
    total_episodes = CONFIG["total_episodes"]

    env = GPUBlindNavEnvV11b(num_envs=num_envs, device=DEVICE, collision_mode=CONFIG["collision_mode"],
                             action_mode=CONFIG["action_mode"], macro=CONFIG["macro"],
                             angle_penalty_coef=CONFIG["angle_penalty_coef"],
                             step_cost=CONFIG["step_cost"],
                             free_turn_penalty=CONFIG["free_turn_penalty"],
                             far_slow_penalty=CONFIG["far_slow_penalty"],
                             hybrid_free_max_deg=CONFIG["hybrid_free_max_deg"],
                             obstacle_signal_mode=CONFIG["obstacle_signal_mode"],
                             false_jump_penalty=CONFIG["false_jump_penalty"],
                             position_age_max_ms=CONFIG["position_age_max_ms"],
                             position_stale_ms=CONFIG["position_stale_ms"],
                             obstacle_density=CONFIG["obstacle_density_start"],
                             position_age_extreme_prob=CONFIG["position_age_extreme_prob"],
                             recovery_commit_steps=CONFIG["recovery_commit_steps"],
                             recovery_reward_scale=CONFIG["recovery_reward_scale"])
    env.fixed_position_age_ms = CONFIG["fixed_training_age_ms"]
    env.stale_target_assist = CONFIG["stale_target_assist"]
    env.collision_penalty_scale = CONFIG["collision_penalty_scale"]
    env.low_block_penalty_scale = CONFIG["low_block_penalty_scale"]
    agent = RecurrentActorCritic(state_dim=OBS_DIM, hidden_dim=CONFIG["hidden_dim"],
                                 actor_width=CONFIG["actor_width"], macro=CONFIG["macro"]).to(DEVICE)
    fresh_weights = True
    macro_head_needs_init = False
    load_path = weights_path
    if not load_path.exists() and CONFIG["macro"]:
        warm_start = out_dir / "policy_weights_v4_replay.pth"
        if warm_start.exists():
            load_path = warm_start
    if load_path.exists():
        try:
            state = torch.load(load_path, map_location=DEVICE)
            missing, unexpected = agent.load_state_dict(state, strict=False)
            base_missing = [key for key in missing if not key.startswith("macro_head.")]
            base_unexpected = [key for key in unexpected if not key.startswith("macro_head.")]
            if base_missing or base_unexpected:
                raise RuntimeError(
                    f"base keys incompatible missing={base_missing} unexpected={base_unexpected}"
                )
            fresh_weights = False
            macro_head_needs_init = CONFIG["macro"] and bool(missing)
            print(f"[warm-start] 从 {load_path.resolve()} 加载基础 V4，缺失宏头={missing}", flush=True)
        except Exception as e:
            fresh_weights = True
            print(f"[新架构初始化] 基础权重不兼容 ({e})，将初始化全新网络", flush=True)
    if fresh_weights and not CONFIG["train_jump_only"]:
        pretrain_steering(
            agent,
            DEVICE,
            steps=CONFIG["steer_pretrain_steps"],
            batch_envs=CONFIG["steer_pretrain_batch_envs"],
            seq_len=CONFIG["steer_pretrain_seq_len"],
            lr=CONFIG["steer_pretrain_lr"],
        )
    if (fresh_weights or macro_head_needs_init) and CONFIG["macro"] and agent.macro_head is not None:
        nn.init.zeros_(agent.macro_head.weight)
        nn.init.zeros_(agent.macro_head.bias)
        print("[训练模式] macro_head 初始化为 macro=0，只有碰撞/停滞观测允许触发宏", flush=True)
    if CONFIG["reset_jump_head"]:
        nn.init.zeros_(agent.jump_head.weight)
        nn.init.zeros_(agent.jump_head.bias)
        print("[训练模式] 已重置 jump_head，初始跳/不跳概率均为 50%", flush=True)
    if CONFIG["train_jump_only"]:
        for parameter in agent.parameters():
            parameter.requires_grad_(False)
        for parameter in agent.jump_head.parameters():
            parameter.requires_grad_(True)
        print("[训练模式] 冻结导航网络，仅训练 jump_head", flush=True)
    optimizer = optim.Adam(
        [parameter for parameter in agent.parameters() if parameter.requires_grad],
        lr=CONFIG["lr"],
        eps=1e-5,
    )
    print(f"[自检] 未知朝向盲人寻路({OBS_DIM}维延迟位置State) | 地图=2250x2250 | 7档相对转向 | 2档速度 | 2档跳跃 | hidden={CONFIG['hidden_dim']} actor={CONFIG['actor_width']} | 碰撞={CONFIG['collision_mode']} | 最大定位年龄={CONFIG['position_age_max_ms']}ms | 脱困宏={'开' if CONFIG['macro'] else '关'}", flush=True)
    print(f"[进度] 正在初始化 GPU 向量化环境张量 (8,192 路高频极速并发)...", flush=True)

    obs_batch = torch.zeros((num_steps, num_envs, OBS_DIM), device=DEVICE)
    actions_batch = torch.zeros((num_steps, num_envs, 4 if CONFIG["macro"] else 3), dtype=torch.long, device=DEVICE)
    teacher_steer_batch = torch.zeros((num_steps, num_envs), dtype=torch.long, device=DEVICE)
    teacher_mask_batch = torch.zeros((num_steps, num_envs), dtype=torch.bool, device=DEVICE)
    macro_teacher_batch = torch.zeros((num_steps, num_envs), dtype=torch.long, device=DEVICE)
    macro_teacher_mask_batch = torch.zeros((num_steps, num_envs), dtype=torch.bool, device=DEVICE)
    logprobs_batch = torch.zeros((num_steps, num_envs), device=DEVICE)
    rewards_batch = torch.zeros((num_steps, num_envs), device=DEVICE)
    dones_batch = torch.zeros((num_steps, num_envs), device=DEVICE)
    values_batch = torch.zeros((num_steps, num_envs), device=DEVICE)

    next_obs = env.reset()
    print(f"[进度] GPU 环境就绪！进行 1,048,576 Steps 极速并行采集中 (约 2 秒/轮)...", flush=True)
    next_done = torch.zeros(num_envs, device=DEVICE)
    next_lstm_state_h = torch.zeros(1, num_envs, CONFIG["hidden_dim"], device=DEVICE)
    next_lstm_state_c = torch.zeros(1, num_envs, CONFIG["hidden_dim"], device=DEVICE)
    macro_teacher_phase = torch.zeros(num_envs, dtype=torch.long, device=DEVICE)

    total_steps_collected = 0
    iteration = 0
    smooth_success = 0.1
    start_time = time.time()

    while total_steps_collected < (total_episodes * 150):
        iteration += 1
        curriculum_fraction = min(
            1.0,
            total_steps_collected / max(CONFIG["obstacle_curriculum_steps"], 1),
        )
        obstacle_density = (
            CONFIG["obstacle_density_start"]
            + (CONFIG["obstacle_density_end"] - CONFIG["obstacle_density_start"])
            * curriculum_fraction
        )
        env.set_obstacle_density(obstacle_density)
        teacher_coef = CONFIG["teacher_coef"] * max(
            0.0,
            1.0 - total_steps_collected / max(CONFIG["teacher_decay_steps"], 1),
        )
        macro_teacher_coef = CONFIG["macro_teacher_coef"] * max(
            0.0,
            1.0 - total_steps_collected / max(CONFIG["macro_teacher_decay_steps"], 1),
        )
        lstm_start_h = next_lstm_state_h.clone()
        lstm_start_c = next_lstm_state_c.clone()

        reached_accum = torch.zeros((), device=DEVICE)
        for step in range(num_steps):
            total_steps_collected += num_envs
            obs_batch[step] = next_obs
            dones_batch[step] = next_done

            observed_angle_error = torch.atan2(next_obs[:, 2], next_obs[:, 3])
            teacher_distance = torch.atan2(
                torch.sin(observed_angle_error.unsqueeze(1) - env.turn_offsets.unsqueeze(0)),
                torch.cos(observed_angle_error.unsqueeze(1) - env.turn_offsets.unsqueeze(0)),
            ).abs()
            teacher_steer_batch[step] = teacher_distance.argmin(dim=1)
            teacher_mask_batch[step] = (
                (next_obs[:, 11] >= 0.5)
                & (next_obs[:, 8] < 0.5)
                & (next_done < 0.5)
            )

            reset_mask = (next_done > 0.5).view(1, -1, 1)
            next_lstm_state_h = torch.where(reset_mask, torch.zeros_like(next_lstm_state_h), next_lstm_state_h)
            next_lstm_state_c = torch.where(reset_mask, torch.zeros_like(next_lstm_state_c), next_lstm_state_c)

            with torch.no_grad():
                lo, (next_lstm_state_h, next_lstm_state_c) = agent.get_states(
                    next_obs.unsqueeze(0), (next_lstm_state_h, next_lstm_state_c), next_done.unsqueeze(0)
                )
                features = agent.actor_fc(lo.squeeze(0))
                logits_steer = agent.steer_head(features)
                logits_speed = agent.speed_head(features)
                logits_jump = agent.jump_head(features)

                a_steer = RecurrentActorCritic._sample_categorical(logits_steer)
                a_speed = RecurrentActorCritic._sample_categorical(logits_speed)
                a_jump = RecurrentActorCritic._sample_categorical(logits_jump)

                heads = [a_steer, a_speed, a_jump]
                if agent.macro_head is not None:
                    macro_logits = agent.macro_head(features)
                    a_macro = RecurrentActorCritic._sample_categorical(macro_logits)
                    macro_trigger = (next_obs[:, 8] > 0.5) | (next_obs[:, 7] > 0.25)
                    macro_start = macro_trigger & (env.macro_left <= 0)
                    macro_teacher = torch.where(
                        macro_teacher_phase == 0,
                        torch.ones_like(a_macro),
                        torch.full_like(a_macro, 2),
                    )
                    macro_teacher_mask = macro_start & (
                        torch.rand(num_envs, device=DEVICE) < macro_teacher_coef
                    )
                    macro_teacher_batch[step] = macro_teacher
                    macro_teacher_mask_batch[step] = macro_teacher_mask
                    a_macro = torch.where(
                        macro_trigger, a_macro, torch.zeros_like(a_macro)
                    )
                    a_macro = torch.where(macro_teacher_mask, macro_teacher, a_macro)
                    macro_teacher_phase = torch.where(
                        macro_teacher_mask,
                        1 - macro_teacher_phase,
                        macro_teacher_phase,
                    )
                    heads.append(a_macro)
                action = torch.stack(heads, dim=1)
                action = env.apply_calibration(action)
                logprob = (
                    F.log_softmax(logits_steer, dim=-1).gather(1, action[:, 0:1]).squeeze(1)
                    + F.log_softmax(logits_speed, dim=-1).gather(1, action[:, 1:2]).squeeze(1)
                    + F.log_softmax(logits_jump, dim=-1).gather(1, action[:, 2:3]).squeeze(1)
                )
                if agent.macro_head is not None:
                    logprob = logprob + F.log_softmax(macro_logits, dim=-1).gather(1, action[:, 3:4]).squeeze(1)
                value = agent.critic_fc(features).squeeze(1)

            next_obs, reward, dones, reached = env.step(action)
            next_done = dones.float()

            actions_batch[step] = action
            logprobs_batch[step] = logprob
            rewards_batch[step] = reward
            values_batch[step] = value

            reached_accum = reached_accum + reached.float().sum()

        reached_this_iter = float(reached_accum.item()) / (num_steps * num_envs)
        smooth_success = 0.99 * smooth_success + 0.01 * reached_this_iter

        with torch.no_grad():
            reset_mask = (next_done > 0.5).view(1, -1, 1)
            next_lstm_state_h = torch.where(reset_mask, torch.zeros_like(next_lstm_state_h), next_lstm_state_h)
            next_lstm_state_c = torch.where(reset_mask, torch.zeros_like(next_lstm_state_c), next_lstm_state_c)
            lo_next, (next_lstm_state_h, next_lstm_state_c) = agent.get_states(
                next_obs.unsqueeze(0), (next_lstm_state_h, next_lstm_state_c), next_done.unsqueeze(0)
            )
            features_next = agent.actor_fc(lo_next.squeeze(0))
            next_value = agent.critic_fc(features_next).squeeze(1)

        # GAE 优势计算
        advantages = torch.zeros_like(rewards_batch)
        lastgaelam = 0
        for t in reversed(range(num_steps)):
            if t == num_steps - 1:
                nextnonterminal = 1.0 - next_done
                nextvalues = next_value
            else:
                nextnonterminal = 1.0 - dones_batch[t + 1]
                nextvalues = values_batch[t + 1]
            delta = rewards_batch[t] + CONFIG["gamma"] * nextvalues * nextnonterminal - values_batch[t]
            advantages[t] = lastgaelam = delta + CONFIG["gamma"] * CONFIG["gae_lambda"] * nextnonterminal * lastgaelam
        returns = advantages + values_batch

        # PPO 策略与价值网络更新
        env_indices = np.arange(num_envs)
        mb_size = CONFIG["minibatch_envs"]

        for epoch in range(CONFIG["ppo_epochs"]):
            np.random.shuffle(env_indices)
            for start in range(0, num_envs, mb_size):
                end = start + mb_size
                mb_env_inds = env_indices[start:end]

                mb_obs = obs_batch[:, mb_env_inds]
                mb_actions = actions_batch[:, mb_env_inds]
                mb_logprobs = logprobs_batch[:, mb_env_inds]
                mb_advantages = advantages[:, mb_env_inds]
                mb_returns = returns[:, mb_env_inds]
                mb_dones = dones_batch[:, mb_env_inds]
                mb_teacher_steer = teacher_steer_batch[:, mb_env_inds]
                mb_teacher_mask = teacher_mask_batch[:, mb_env_inds]
                mb_macro_teacher = macro_teacher_batch[:, mb_env_inds]
                mb_macro_teacher_mask = macro_teacher_mask_batch[:, mb_env_inds]

                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                init_h = lstm_start_h[:, mb_env_inds].detach()
                init_c = lstm_start_c[:, mb_env_inds].detach()

                newlogprob, entropy, newvalue, teacher_logits, macro_logits = agent.evaluate_actions(
                    mb_obs, (init_h, init_c), mb_dones, mb_actions
                )

                logratio = newlogprob - mb_logprobs
                ratio = logratio.exp()

                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1.0 - 0.2, 1.0 + 0.2)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                v_loss = 0.5 * ((newvalue - mb_returns) ** 2).mean()
                entropy_loss = entropy.mean()

                teacher_logits = teacher_logits.reshape(-1, 7)
                teacher_labels = mb_teacher_steer.reshape(-1)
                teacher_mask = mb_teacher_mask.reshape(-1)
                if teacher_mask.any():
                    teacher_loss = F.cross_entropy(
                        teacher_logits[teacher_mask], teacher_labels[teacher_mask]
                    )
                else:
                    teacher_loss = torch.zeros((), device=DEVICE)

                if macro_logits is not None and mb_macro_teacher_mask.any():
                    macro_teacher_loss = F.cross_entropy(
                        macro_logits[mb_macro_teacher_mask],
                        mb_macro_teacher[mb_macro_teacher_mask],
                    )
                else:
                    macro_teacher_loss = torch.zeros((), device=DEVICE)

                loss = (
                    pg_loss
                    - CONFIG["ent_coef"] * entropy_loss
                    + CONFIG["vf_coef"] * v_loss
                    + teacher_coef * teacher_loss
                    + CONFIG["macro_teacher_loss_coef"] * macro_teacher_loss
                )

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), CONFIG["max_grad_norm"])
                optimizer.step()

        elapsed = time.time() - start_time
        sps = int(total_steps_collected / elapsed)
        avg_reward = float(rewards_batch.mean().item())
        current_episodes = int(total_steps_collected / 150)

        print(f"[{elapsed:5.1f}s] Ep: {current_episodes:6d}/{total_episodes} | 平滑成功率: {smooth_success:.3f} | 平均奖励: {avg_reward:6.3f} | 迭代次数: {iteration:4d} | 密度: {obstacle_density:.2f} | 教师系数: {teacher_coef:.3f} | 宏教师: {macro_teacher_coef:.3f} | 速度 (Steps/s): {sps}", flush=True)

        if (
            CONFIG["validation_every_updates"] > 0
            and iteration % CONFIG["validation_every_updates"] == 0
        ):
            evaluate_training_checkpoint(
                agent,
                DEVICE,
                episodes=CONFIG["validation_episodes"],
                max_steps=CONFIG["validation_max_steps"],
                seed=CONFIG["seed_start"] + iteration * 100,
            )

        if iteration % CONFIG["checkpoint_every"] == 0 or total_steps_collected >= (total_episodes * 150):
            torch.save(agent.state_dict(), weights_path)

    torch.save(agent.state_dict(), weights_path)
    print(f"STAGE 1 完成！模型权重固化于: {weights_path.resolve()}\n")
    if CONFIG["post_training_video"]:
        render_training_episode(
            agent,
            DEVICE,
            iteration,
            CONFIG["training_video_dir"],
            age_ms=CONFIG["training_video_age_ms"],
            max_steps=CONFIG["training_video_steps"],
            fps=CONFIG["training_video_fps"],
            seed=CONFIG["seed_start"],
        )


def evaluate(out_dir, weights_path, play=True):
    print("\n==================================================")
    print("STAGE 2: 并行评估与多进程视频渲染 (10次测试)...")
    print("==================================================")

    import random
    from datetime import datetime
    eval_timestamp = datetime.now().strftime("%H%M%S")
    random_seed_offset = random.randint(100, 9999)
    actual_seed_start = CONFIG["seed_start"] + random_seed_offset
    print(f"[验证] 本次测试随机起始种子: {actual_seed_start} (基于当前时间戳 {eval_timestamp} 刷新全新地图)")

    pool_args = []
    for i in range(CONFIG["eval_episodes"]):
        seed = actual_seed_start + i
        pool_args.append((
            seed,
            i + 1,
            str(weights_path),
            CONFIG["fps"],
            str(out_dir),
            CONFIG["hidden_dim"],
            CONFIG["actor_width"],
            CONFIG["action_mode"],
            CONFIG["macro"],
            CONFIG["collision_mode"],
            CONFIG["eval_sample"],
            CONFIG["eval_commit"],
            CONFIG["hybrid_free_max_deg"],
            CONFIG["obstacle_signal_mode"]
        ))

    eval_start_time = time.time()
    with multiprocessing.Pool(processes=CONFIG["eval_episodes"]) as pool:
        results = pool.starmap(run_and_render_seed_worker, pool_args)

    time.sleep(0.5)  # 确保所有子进程 MP4 文件磁盘写入与 close() 完毕
    eval_elapsed = time.time() - eval_start_time
    print(f"STAGE 2 完成！10 路并行视频渲染耗时: {eval_elapsed:.1f} 秒。")

    episode_videos = []
    success_count = 0
    for res in results:
        if res[0] is not None:
            episode_videos.append(Path(res[0]))
            if res[1]:
                success_count += 1

    print("\n==================================================")
    print("STAGE 3: 视频拼接与 FFMpeg 压缩...")
    print("==================================================")

    concat_mp4 = out_dir / f"eval10_concat_{eval_timestamp}.mp4"
    compressed_mp4 = out_dir / f"eval10_concat_compressed_{eval_timestamp}.mp4"

    from render_eval10_concat import concat_and_compress
    concat_and_compress(episode_videos, concat_mp4, compressed_mp4, fps=CONFIG["fps"])

    abs_compressed_mp4 = compressed_mp4.resolve()
    print("==================================================")
    print(f"最终成功率: {success_count}/{CONFIG['eval_episodes']} ({success_count/CONFIG['eval_episodes']*100:.1f}%)")
    print(f"全新带时间戳成片 (绝对路径): {abs_compressed_mp4}")
    print("==================================================\n")

    if not play:
        return
    print("==================================================")
    print(f"STAGE 4: 启动本地 mpv 播放最新生成视频 [{eval_timestamp}]...")
    print("==================================================")
    subprocess.Popen(["mpv", str(abs_compressed_mp4)])
    print("播放器已拉起！管道任务圆满结束。")


def run_pipeline(train_only=False, eval_only=False, play=True):
    out_dir = Path(CONFIG["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    weights_path = out_dir / CONFIG["weights_name"]
    if not eval_only:
        train(out_dir, weights_path)
    if train_only:
        return
    if not weights_path.exists():
        raise SystemExit(f"权重文件不存在: {weights_path.resolve()}, 请先跑 --train-only")
    evaluate(out_dir, weights_path, play=play)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ProprioNav 端到端 Reinforcement Learning")
    parser.add_argument("--train-only", action="store_true")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--no-play", action="store_true", help="评估完成后不启动本地 mpv")
    parser.add_argument("--eval-episodes", type=int, default=None)
    parser.add_argument("--eval-sample", action="store_true",
                        help="评估时按策略概率采样，避免高熵策略被 argmax 压成直冲")
    parser.add_argument("--eval-commit", type=int, default=None,
                        help="评估时碰撞后锁定大角度转向档 N 步")
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--hidden-dim", type=int, default=None)
    parser.add_argument("--actor-width", type=int, default=None)
    parser.add_argument("--weights-name", type=str, default=None)
    parser.add_argument("--collision-mode", type=str, default=None, choices=["slide", "weak", "hard"])
    parser.add_argument("--num-envs", type=int, default=None)
    parser.add_argument("--minibatch-envs", type=int, default=None)
    parser.add_argument("--ent-coef", type=float, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--rollout-steps", type=int, default=None)
    parser.add_argument("--ppo-epochs", type=int, default=None)
    parser.add_argument("--action-mode", type=str, default=None,
                        choices=["unknown_heading"])
    parser.add_argument("--angle-penalty-coef", type=float, default=None)
    parser.add_argument("--step-cost", type=float, default=None)
    parser.add_argument("--free-turn-penalty", type=float, default=None)
    parser.add_argument("--far-slow-penalty", type=float, default=None)
    parser.add_argument("--hybrid-free-max-deg", type=float, default=None)
    parser.add_argument("--obstacle-signal-mode", type=str, default=None,
                        choices=["proximity", "jump_probe"])
    parser.add_argument("--false-jump-penalty", type=float, default=None)
    parser.add_argument("--position-age-max-ms", type=float, default=None)
    parser.add_argument("--position-stale-ms", type=float, default=None)
    parser.add_argument("--position-age-extreme-prob", type=float, default=None)
    parser.add_argument("--validation-every-updates", type=int, default=None)
    parser.add_argument("--validation-episodes", type=int, default=None)
    parser.add_argument("--validation-max-steps", type=int, default=None)
    parser.add_argument("--no-training-video", action="store_true")
    parser.add_argument("--training-video-steps", type=int, default=None)
    parser.add_argument("--training-video-age-ms", type=float, default=None)
    parser.add_argument("--training-video-dir", type=str, default=None)
    parser.add_argument("--recovery-commit-steps", type=int, default=None)
    parser.add_argument("--fixed-training-age-ms", type=float, default=None)
    parser.add_argument("--stale-target-assist", action="store_true")
    parser.add_argument("--recovery-reward-scale", type=float, default=None)
    parser.add_argument("--collision-penalty-scale", type=float, default=None)
    parser.add_argument("--low-block-penalty-scale", type=float, default=None)
    parser.add_argument("--teacher-coef", type=float, default=None)
    parser.add_argument("--teacher-decay-steps", type=int, default=None)
    parser.add_argument("--steer-pretrain-steps", type=int, default=None)
    parser.add_argument("--steer-pretrain-batch-envs", type=int, default=None)
    parser.add_argument("--steer-pretrain-seq-len", type=int, default=None)
    parser.add_argument("--steer-pretrain-lr", type=float, default=None)
    parser.add_argument("--obstacle-density-start", type=float, default=None)
    parser.add_argument("--obstacle-density-end", type=float, default=None)
    parser.add_argument("--obstacle-curriculum-steps", type=int, default=None)
    parser.add_argument("--reset-jump-head", action="store_true")
    parser.add_argument("--train-jump-only", action="store_true")
    parser.add_argument("--macro", action="store_true", help="启用 8 档显式脱困宏动作头")
    args = parser.parse_args()
    if args.episodes is not None:
        CONFIG["total_episodes"] = args.episodes
    if args.eval_episodes is not None:
        CONFIG["eval_episodes"] = args.eval_episodes
    if args.eval_sample:
        CONFIG["eval_sample"] = True
    if args.eval_commit is not None:
        CONFIG["eval_commit"] = args.eval_commit
    if args.hidden_dim is not None:
        CONFIG["hidden_dim"] = args.hidden_dim
    if args.actor_width is not None:
        CONFIG["actor_width"] = args.actor_width
    if args.weights_name is not None:
        CONFIG["weights_name"] = args.weights_name
    if args.collision_mode is not None:
        CONFIG["collision_mode"] = args.collision_mode
    if args.num_envs is not None:
        CONFIG["num_envs"] = args.num_envs
    if args.minibatch_envs is not None:
        CONFIG["minibatch_envs"] = args.minibatch_envs
    if args.ent_coef is not None:
        CONFIG["ent_coef"] = args.ent_coef
    if args.lr is not None:
        CONFIG["lr"] = args.lr
    if args.rollout_steps is not None:
        CONFIG["rollout_steps"] = args.rollout_steps
    if args.ppo_epochs is not None:
        CONFIG["ppo_epochs"] = max(1, args.ppo_epochs)
    if args.angle_penalty_coef is not None:
        CONFIG["angle_penalty_coef"] = args.angle_penalty_coef
    if args.step_cost is not None:
        CONFIG["step_cost"] = args.step_cost
    if args.free_turn_penalty is not None:
        CONFIG["free_turn_penalty"] = args.free_turn_penalty
    if args.far_slow_penalty is not None:
        CONFIG["far_slow_penalty"] = args.far_slow_penalty
    if args.hybrid_free_max_deg is not None:
        CONFIG["hybrid_free_max_deg"] = args.hybrid_free_max_deg
    if args.obstacle_signal_mode is not None:
        CONFIG["obstacle_signal_mode"] = args.obstacle_signal_mode
    if args.false_jump_penalty is not None:
        CONFIG["false_jump_penalty"] = args.false_jump_penalty
    if args.position_age_max_ms is not None:
        CONFIG["position_age_max_ms"] = args.position_age_max_ms
    if args.position_stale_ms is not None:
        CONFIG["position_stale_ms"] = args.position_stale_ms
    if args.position_age_extreme_prob is not None:
        CONFIG["position_age_extreme_prob"] = args.position_age_extreme_prob
    if args.validation_every_updates is not None:
        CONFIG["validation_every_updates"] = args.validation_every_updates
    if args.validation_episodes is not None:
        CONFIG["validation_episodes"] = args.validation_episodes
    if args.validation_max_steps is not None:
        CONFIG["validation_max_steps"] = args.validation_max_steps
    if args.no_training_video:
        CONFIG["post_training_video"] = False
    if args.training_video_steps is not None:
        CONFIG["training_video_steps"] = args.training_video_steps
    if args.training_video_age_ms is not None:
        CONFIG["training_video_age_ms"] = args.training_video_age_ms
    if args.training_video_dir is not None:
        CONFIG["training_video_dir"] = args.training_video_dir
    if args.recovery_commit_steps is not None:
        CONFIG["recovery_commit_steps"] = max(0, args.recovery_commit_steps)
    if args.fixed_training_age_ms is not None:
        CONFIG["fixed_training_age_ms"] = args.fixed_training_age_ms
    if args.stale_target_assist:
        CONFIG["stale_target_assist"] = True
    if args.recovery_reward_scale is not None:
        CONFIG["recovery_reward_scale"] = args.recovery_reward_scale
    if args.collision_penalty_scale is not None:
        CONFIG["collision_penalty_scale"] = args.collision_penalty_scale
    if args.low_block_penalty_scale is not None:
        CONFIG["low_block_penalty_scale"] = args.low_block_penalty_scale
    if args.teacher_coef is not None:
        CONFIG["teacher_coef"] = args.teacher_coef
    if args.teacher_decay_steps is not None:
        CONFIG["teacher_decay_steps"] = args.teacher_decay_steps
    if args.steer_pretrain_steps is not None:
        CONFIG["steer_pretrain_steps"] = args.steer_pretrain_steps
    if args.steer_pretrain_batch_envs is not None:
        CONFIG["steer_pretrain_batch_envs"] = args.steer_pretrain_batch_envs
    if args.steer_pretrain_seq_len is not None:
        CONFIG["steer_pretrain_seq_len"] = args.steer_pretrain_seq_len
    if args.steer_pretrain_lr is not None:
        CONFIG["steer_pretrain_lr"] = args.steer_pretrain_lr
    if args.obstacle_density_start is not None:
        CONFIG["obstacle_density_start"] = args.obstacle_density_start
    if args.obstacle_density_end is not None:
        CONFIG["obstacle_density_end"] = args.obstacle_density_end
    if args.obstacle_curriculum_steps is not None:
        CONFIG["obstacle_curriculum_steps"] = args.obstacle_curriculum_steps
    if args.reset_jump_head:
        CONFIG["reset_jump_head"] = True
    if args.train_jump_only:
        CONFIG["train_jump_only"] = True
    if args.action_mode is not None:
        CONFIG["action_mode"] = args.action_mode
    if args.macro:
        CONFIG["macro"] = True
    multiprocessing.set_start_method('spawn', force=True)
    run_pipeline(train_only=args.train_only, eval_only=args.eval_only, play=not args.no_play)
