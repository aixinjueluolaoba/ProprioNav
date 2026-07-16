import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import os
import math
from pathlib import Path

# =====================================================================
# 1. GPU-accelerated Vectorized 2D Environment (v11b/v11rxc Specs)
# =====================================================================
class GPUBlindNavEnvV11b:
    def __init__(self, num_envs=4096, world_size=2400.0, dt=0.3, device="cuda"):
        self.num_envs = num_envs
        self.world_size = world_size
        self.dt = dt
        self.device = device
        
        # 障碍物设置
        self.tree_count = 90
        self.mountain_count = 24
        
        self.tree_radii = torch.full((num_envs, self.tree_count), 28.0, device=device) # [N, 90]
        # 山体包络半径 35.0 到 80.0
        self.mountain_radii = torch.rand((num_envs, self.mountain_count), device=device) * (80.0 - 35.0) + 35.0 # [N, 24]
        
        # 核心状态张量
        self.pos = torch.zeros((self.num_envs, 2), dtype=torch.float32, device=device)
        self.target = torch.zeros((self.num_envs, 2), dtype=torch.float32, device=device)
        self.heading = torch.zeros(self.num_envs, dtype=torch.float32, device=device)
        self.tree_centers = torch.zeros((self.num_envs, self.tree_count, 2), dtype=torch.float32, device=device)
        self.mountain_centers = torch.zeros((self.num_envs, self.mountain_count, 2), dtype=torch.float32, device=device)
        self.mountain_vertices = torch.zeros((self.num_envs, self.mountain_count, 8, 2), dtype=torch.float32, device=device)
        
        # 时间与进度跟踪器
        self.step_count = torch.zeros(self.num_envs, dtype=torch.int32, device=device)
        self.stuck_time = torch.zeros(self.num_envs, dtype=torch.float32, device=device)
        self.no_progress_time = torch.zeros(self.num_envs, dtype=torch.float32, device=device)
        self.time_since_collision = torch.full((self.num_envs,), 10.0, dtype=torch.float32, device=device)
        self.coord_age = torch.zeros(self.num_envs, dtype=torch.float32, device=device)
        
        # 上一步状态 (用于增量/惩罚计算)
        self.prev_dist = torch.zeros(self.num_envs, dtype=torch.float32, device=device)
        self.prev_angle_error = torch.zeros(self.num_envs, dtype=torch.float32, device=device)
        self.last_action_value = torch.zeros(self.num_envs, dtype=torch.float32, device=device)
        
        # 宏动作状态机张量
        self.macro_steps_left = torch.zeros(self.num_envs, dtype=torch.int32, device=device)
        self.macro_recovery_mode = torch.zeros(self.num_envs, dtype=torch.int32, device=device)
        self.macro_recovery_total_steps = torch.zeros(self.num_envs, dtype=torch.int32, device=device)
        self.macro_anchor_heading = torch.zeros(self.num_envs, dtype=torch.float32, device=device)
        
        # 动作空间参数映射
        self.ANGLE_OFFSETS = torch.tensor([-1.0, -0.33333334, 0.0, 0.33333334, 1.0], device=device)
        self.MACRO_DURATIONS = torch.tensor([0, 2, 4, 4, 6, 6, 4, 3], dtype=torch.int32, device=device) # 宏 ID (0-7) 对应的步数
        
        self.reset()
        
    def _sample_start_and_target(self, num_samples):
        # 默认起点终点采样
        pos = (torch.rand((num_samples, 2), device=self.device) - 0.5) * (self.world_size * 1.1)
        theta = torch.rand(num_samples, device=self.device) * 2.0 * math.pi
        d = torch.rand(num_samples, device=self.device) * (1296.0 - 180.0) + 180.0
        target = pos + torch.stack([torch.cos(theta), torch.sin(theta)], dim=1) * d.unsqueeze(1)
        target = torch.clamp(target, -self.world_size * 0.6, self.world_size * 0.6)
        return pos, target

    def _generate_obstacles_for_envs(self, env_indices, player_pos, target_pos):
        num_envs = len(env_indices)
        if num_envs == 0:
            return
            
        t_centers = torch.rand((num_envs, self.tree_count, 2), device=self.device) * (self.world_size * 1.8) - (self.world_size * 0.9)
        m_centers = torch.rand((num_envs, self.mountain_count, 2), device=self.device) * (self.world_size * 1.8) - (self.world_size * 0.9)
        m_radii = self.mountain_radii[env_indices]
        
        # 起点终点避让
        dist_to_pos_t = torch.norm(t_centers - player_pos.unsqueeze(1), dim=2)
        t_pos_too_close = dist_to_pos_t < (28.0 + 140.0)
        dir_pos_t = t_centers - player_pos.unsqueeze(1)
        dir_pos_t = dir_pos_t / torch.clamp(torch.norm(dir_pos_t, dim=2, keepdim=True), min=1e-6)
        t_centers = torch.where(t_pos_too_close.unsqueeze(2), player_pos.unsqueeze(1) + dir_pos_t * 168.0, t_centers)
        
        dist_to_target_t = torch.norm(t_centers - target_pos.unsqueeze(1), dim=2)
        t_target_too_close = dist_to_target_t < (28.0 + 35.0 + 80.0)
        dir_target_t = t_centers - target_pos.unsqueeze(1)
        dir_target_t = dir_target_t / torch.clamp(torch.norm(dir_target_t, dim=2, keepdim=True), min=1e-6)
        t_centers = torch.where(t_target_too_close.unsqueeze(2), target_pos.unsqueeze(1) + dir_target_t * 143.0, t_centers)
        
        dist_to_pos_m = torch.norm(m_centers - player_pos.unsqueeze(1), dim=2)
        m_pos_too_close = dist_to_pos_m < (80.0 + 180.0)
        dir_pos_m = m_centers - player_pos.unsqueeze(1)
        dir_pos_m = dir_pos_m / torch.clamp(torch.norm(dir_pos_m, dim=2, keepdim=True), min=1e-6)
        m_centers = torch.where(m_pos_too_close.unsqueeze(2), player_pos.unsqueeze(1) + dir_pos_m * 260.0, m_centers)
        
        dist_to_target_m = torch.norm(m_centers - target_pos.unsqueeze(1), dim=2)
        m_target_too_close = dist_to_target_m < (80.0 + 35.0 + 120.0)
        dir_target_m = m_centers - target_pos.unsqueeze(1)
        dir_target_m = dir_target_m / torch.clamp(torch.norm(dir_target_m, dim=2, keepdim=True), min=1e-6)
        m_centers = torch.where(m_target_too_close.unsqueeze(2), target_pos.unsqueeze(1) + dir_target_m * 235.0, m_centers)
        
        # 8边形山体生成
        num_vertices = 8
        angles = torch.linspace(0.0, 2.0 * math.pi, num_vertices + 1, device=self.device)[:-1]
        angles = angles.unsqueeze(0).unsqueeze(0).expand(num_envs, self.mountain_count, num_vertices)
        angles = angles + (torch.rand_like(angles) * 0.36 - 0.18)
        r_vert = m_radii.unsqueeze(2) * (0.6 + torch.rand_like(angles) * 0.4)
        
        v_x = r_vert * torch.cos(angles)
        v_y = r_vert * torch.sin(angles)
        m_vertices = m_centers.unsqueeze(2) + torch.stack([v_x, v_y], dim=3)
        
        self.tree_centers[env_indices] = t_centers
        self.mountain_centers[env_indices] = m_centers
        self.mountain_vertices[env_indices] = m_vertices

    def reset(self, seed=None):
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            
        # 默认起点与终点
        self.pos, self.target = self._sample_start_and_target(self.num_envs)
        
        # 为所有环境生成基本的树木与山体中心
        self._generate_obstacles_for_envs(torch.arange(self.num_envs), self.pos, self.target)
        
        # 概率性应用起点受困机制 (Trapped Start Probability = 0.55)
        trapped_mask = torch.rand(self.num_envs, device=self.device) < 0.55
        trapped_indices = torch.where(trapped_mask)[0]
        if len(trapped_indices) > 0:
            # 起点采样在山体 0 内部
            centers_0 = self.mountain_centers[trapped_indices, 0]
            radii_0 = self.mountain_radii[trapped_indices, 0]
            angles_0 = torch.rand(len(trapped_indices), device=self.device) * 2.0 * math.pi
            
            # 放到山体半径的 0.4 处，确保在里面
            offsets_pos = torch.stack([torch.cos(angles_0), torch.sin(angles_0)], dim=1) * radii_0.unsqueeze(1) * 0.4
            self.pos[trapped_indices] = centers_0 + offsets_pos
            
            # 目标点在相反方向旋转 78 度的远处
            side_rot = torch.where(torch.rand(len(trapped_indices), device=self.device) < 0.5, 78.0, -78.0) * (math.pi / 180.0)
            rot_angles = angles_0 + side_rot
            target_dists = torch.rand(len(trapped_indices), device=self.device) * (1296.0 - 180.0) + 180.0
            offsets_target = torch.stack([torch.cos(rot_angles), torch.sin(rot_angles)], dim=1) * target_dists.unsqueeze(1)
            self.target[trapped_indices] = torch.clamp(self.pos[trapped_indices] + offsets_target, -self.world_size * 0.6, self.world_size * 0.6)
            
            # 重新生成受困环境的障碍物 (确保山体 0 包裹了起点)
            self._generate_obstacles_for_envs(trapped_indices, self.pos[trapped_indices], self.target[trapped_indices])
            
        # 车头朝向初始化
        self.heading = (torch.rand(self.num_envs, dtype=torch.float32, device=self.device) * 2.0 - 1.0) * math.pi
        # 反向车头初始化对齐 (Opposite Heading Probability = 0.5)
        opposite_mask = torch.rand(self.num_envs, device=self.device) < 0.5
        delta_to_target = self.target - self.pos
        target_angles = torch.atan2(delta_to_target[:, 1], delta_to_target[:, 0])
        opposite_headings = target_angles + math.pi + (torch.rand(self.num_envs, device=self.device) * 0.4 - 0.2)
        self.heading = torch.where(opposite_mask, opposite_headings, self.heading)
        self.heading = torch.atan2(torch.sin(self.heading), torch.cos(self.heading))
        
        # 跟踪器清零
        self.step_count.fill_(0)
        self.stuck_time.fill_(0.0)
        self.no_progress_time.fill_(0.0)
        self.time_since_collision.fill_(10.0)
        self.coord_age.fill_(0.0)
        
        self.prev_dist = torch.norm(self.target - self.pos, dim=1)
        self.prev_angle_error = self._angle_error()
        self.last_action_value.fill_(0.0)
        
        self.macro_steps_left.fill_(0)
        self.macro_recovery_mode.fill_(0)
        self.macro_recovery_total_steps.fill_(0)
        self.macro_anchor_heading.fill_(0.0)
        
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
        
        # coord_age 模拟: 每个物理 step 增加 0.3，当超过 0.6 时重置为 0
        coord_age_scaled = torch.clamp(self.coord_age / 1.5, 0.0, 1.0)
        
        # no_progress_time_norm
        no_progress_time_norm = torch.clamp(self.no_progress_time / 3.0, 0.0, 1.0)
        
        # recent_collision_norm
        recent_collision_norm = 1.0 - torch.clamp(self.time_since_collision / 3.0, 0.0, 1.0)
        
        # targeted_context: no_progress_time >= 0.55*3 或 recent_collision_norm >= 0.82
        targeted_context = ((self.no_progress_time >= 1.65) | (recent_collision_norm >= 0.82)).float()
        
        # distance_pressure
        distance_pressure = torch.clamp(distance / 60.0, 0.0, 1.0)
        
        obs = torch.stack([
            torch.clamp(delta[:, 0] / self.world_size, -1.0, 1.0),
            torch.clamp(delta[:, 1] / self.world_size, -1.0, 1.0),
            torch.sin(self.heading),
            torch.cos(self.heading),
            torch.sin(angle_error),
            torch.cos(angle_error),
            coord_age_scaled,
            no_progress_time_norm,
            targeted_context,
            torch.clamp(distance / self.world_size, 0.0, 1.0),
            torch.clamp(self.macro_recovery_mode.float() / 7.0, -1.0, 1.0),
            distance_pressure
        ], dim=1)
        return obs
        
    def step(self, actions):
        self.step_count += 1
        self.coord_age = torch.where(self.coord_age >= 0.6, torch.zeros_like(self.coord_age), self.coord_age + 0.3)
        self.time_since_collision += 0.3
        
        # 1. 拆解网络输出动作
        angle_bin = actions[:, 0].clamp(0, 4)
        speed_bin = actions[:, 1].clamp(0, 1)
        macro_bin = actions[:, 2].clamp(0, 7)
        
        # 2. stuck explore 注入 (Wrapper level)
        recent_collision_norm = 1.0 - torch.clamp(self.time_since_collision / 3.0, 0.0, 1.0)
        stuck_macro_context = (self.no_progress_time >= 1.26) | (recent_collision_norm >= 0.72) # 0.42 * 3 = 1.26
        explore_mask = (macro_bin == 0) & stuck_macro_context & (torch.rand(self.num_envs, device=self.device) < 0.24)
        random_macro = torch.randint(1, 8, (self.num_envs,), device=self.device)
        macro_bin = torch.where(explore_mask, random_macro, macro_bin)
        
        # 3. 宏状态机转换与初始化
        init_macro_mask = (self.macro_steps_left <= 0) & (macro_bin != 0)
        self.macro_recovery_mode = torch.where(init_macro_mask, macro_bin, self.macro_recovery_mode)
        self.macro_recovery_total_steps = torch.where(init_macro_mask, self.MACRO_DURATIONS[macro_bin], self.macro_recovery_total_steps)
        self.macro_steps_left = torch.where(init_macro_mask, self.macro_recovery_total_steps, self.macro_steps_left)
        self.macro_anchor_heading = torch.where(init_macro_mask, self.heading, self.macro_anchor_heading)
        
        # 4. 决定本步的目标 desired_angle 与 speed
        delta = self.target - self.pos
        target_angle = torch.atan2(delta[:, 1], delta[:, 0])
        
        # 默认正常行为 (macro_steps_left <= 0)
        angle_offset = self.ANGLE_OFFSETS[angle_bin] * (math.pi / 4.0) # delta_angle_limit = 45度
        desired_angle_normal = target_angle + angle_offset
        speed_normal = torch.where(speed_bin == 1, 100.0, 50.0)
        
        # 宏动作覆盖行为
        macro_active = self.macro_steps_left > 0
        mode = self.macro_recovery_mode
        total_steps = self.macro_recovery_total_steps
        phase_progress = total_steps - self.macro_steps_left
        anchor = self.macro_anchor_heading
        
        # 定义各类宏的方向
        angle_mode1 = anchor + math.pi # short_back
        
        angle_mode2 = torch.where(phase_progress < 2, anchor + math.pi, anchor + math.radians(90.0)) # back_left
        angle_mode3 = torch.where(phase_progress < 2, anchor + math.pi, anchor - math.radians(90.0)) # back_right
        
        angle_mode4 = torch.where(phase_progress < 2, anchor + math.pi, anchor + math.radians(90.0)) # wide_left
        angle_mode5 = torch.where(phase_progress < 2, anchor + math.pi, anchor - math.radians(90.0)) # wide_right
        
        angle_mode6 = torch.where(phase_progress < 2, anchor + math.pi, target_angle) # back_then_realign
        
        # mode 7: short_wide_left_then_target
        angle_mode7 = torch.where(
            phase_progress < 1, 
            anchor + math.pi,
            torch.where(phase_progress < 3, anchor + math.radians(90.0), target_angle)
        )
        
        desired_angle_macro = target_angle # 兜底
        desired_angle_macro = torch.where(mode == 1, angle_mode1, desired_angle_macro)
        desired_angle_macro = torch.where(mode == 2, angle_mode2, desired_angle_macro)
        desired_angle_macro = torch.where(mode == 3, angle_mode3, desired_angle_macro)
        desired_angle_macro = torch.where(mode == 4, angle_mode4, desired_angle_macro)
        desired_angle_macro = torch.where(mode == 5, angle_mode5, desired_angle_macro)
        desired_angle_macro = torch.where(mode == 6, angle_mode6, desired_angle_macro)
        desired_angle_macro = torch.where(mode == 7, angle_mode7, desired_angle_macro)
        
        desired_angle = torch.where(macro_active, desired_angle_macro, desired_angle_normal)
        speed = torch.where(macro_active, torch.full_like(speed_normal, 100.0), speed_normal)
        
        # 5. 执行转向与前进物理模拟
        max_turn = math.radians(180.0) * self.dt # 0.94248
        # move angle towards desired_angle
        ang_diff = desired_angle - self.heading
        ang_diff = torch.atan2(torch.sin(ang_diff), torch.cos(ang_diff))
        ang_diff_clamped = torch.clamp(ang_diff, -max_turn, max_turn)
        prev_heading = self.heading.clone()
        self.heading = self.heading + ang_diff_clamped
        self.heading = torch.atan2(torch.sin(self.heading), torch.cos(self.heading))
        
        intended_delta = torch.stack([torch.cos(self.heading), torch.sin(self.heading)], dim=1) * speed.unsqueeze(1) * self.dt
        candidate = self.pos + intended_delta
        
        prev_stuck = (self.stuck_time > 0.4) | (self.no_progress_time > 0.9)
        old_pos = self.pos.clone()
        
        # ==========================================
        # Collision Detection & sliding (vectorized)
        # ==========================================
        # A. 树木碰撞 (Circles)
        diff_t = candidate.unsqueeze(1) - self.tree_centers
        dist_t = torch.norm(diff_t, dim=2)
        tree_collided = dist_t < self.tree_radii
        any_tree_collided = tree_collided.any(dim=1)
        
        # B. 山体碰撞 (8-sided Polygons)
        P_cand = candidate.unsqueeze(1).unsqueeze(2)
        C = self.mountain_vertices
        D = torch.roll(self.mountain_vertices, shifts=-1, dims=2)
        c_x, c_y = C[..., 0], C[..., 1]
        d_x, d_y = D[..., 0], D[..., 1]
        p_x, p_y = P_cand[..., 0], P_cand[..., 1]
        
        cond1 = (c_y > p_y) != (d_y > p_y)
        denom = d_y - c_y
        denom = torch.where(denom.abs() < 1e-6, torch.sign(denom) * 1e-6, denom)
        x_intersect = (d_x - c_x) * (p_y - c_y) / denom + c_x
        ray_cross = cond1 & (p_x < x_intersect)
        inside = (ray_cross.sum(dim=2) % 2 == 1)
        
        A_seg = self.pos.unsqueeze(1).unsqueeze(2)
        B_seg = candidate.unsqueeze(1).unsqueeze(2)
        cp1 = (B_seg[..., 0] - A_seg[..., 0]) * (C[..., 1] - A_seg[..., 1]) - (B_seg[..., 1] - A_seg[..., 1]) * (C[..., 0] - A_seg[..., 0])
        cp2 = (B_seg[..., 0] - A_seg[..., 0]) * (D[..., 1] - A_seg[..., 1]) - (B_seg[..., 1] - A_seg[..., 1]) * (D[..., 0] - A_seg[..., 0])
        cp3 = (D[..., 0] - C[..., 0]) * (A_seg[..., 1] - C[..., 1]) - (D[..., 1] - C[..., 1]) * (A_seg[..., 0] - C[..., 0])
        cp4 = (D[..., 0] - C[..., 0]) * (B_seg[..., 1] - C[..., 1]) - (D[..., 1] - C[..., 1]) * (B_seg[..., 0] - C[..., 0])
        edge_hit = (cp1 * cp2 < 0.0) & (cp3 * cp4 < 0.0)
        mountain_hit = edge_hit.any(dim=2)
        
        mountain_collided = inside | mountain_hit
        any_mountain_collided = mountain_collided.any(dim=1)
        
        any_collided = any_tree_collided | any_mountain_collided
        self.time_since_collision = torch.where(any_collided, torch.zeros_like(self.time_since_collision), self.time_since_collision)
        
        # C. 滑动解决
        # 1. 树木滑动
        penetration_t = 28.0 - dist_t
        max_pen_idx = torch.argmax(torch.where(tree_collided, penetration_t, -1e9), dim=1)
        coll_centers_t = self.tree_centers[torch.arange(self.num_envs), max_pen_idx]
        normal_t = (candidate - coll_centers_t)
        normal_t = normal_t / torch.clamp(torch.norm(normal_t, dim=1, keepdim=True), min=1e-6)
        tangent_t = torch.stack([-normal_t[:, 1], normal_t[:, 0]], dim=1)
        dot_t = tangent_t[:, 0] * intended_delta[:, 0] + tangent_t[:, 1] * intended_delta[:, 1]
        slide_t = tangent_t * torch.sign(dot_t).unsqueeze(1) * torch.norm(intended_delta, dim=1, keepdim=True) * 0.45
        resolved_tree = self.pos + slide_t
        
        # 2. 山体滑动
        P_close = candidate.unsqueeze(1).unsqueeze(2)
        segment_m = D - C
        length_sq_m = (segment_m ** 2).sum(dim=3)
        t_m = ((P_close - C) * segment_m).sum(dim=3) / torch.clamp(length_sq_m, min=1e-6)
        t_m = torch.clamp(t_m, 0.0, 1.0)
        closest_m = C + segment_m * t_m.unsqueeze(3)
        dist_sq_m = ((P_close - closest_m) ** 2).sum(dim=3)
        dist_sq_m = torch.where(mountain_collided.unsqueeze(2), dist_sq_m, 1e9)
        min_idx = torch.argmin(dist_sq_m.view(self.num_envs, -1), dim=1)
        m_idx = min_idx // 8
        e_idx = min_idx % 8
        
        C_close = self.mountain_vertices[torch.arange(self.num_envs), m_idx, e_idx]
        D_close = self.mountain_vertices[torch.arange(self.num_envs), m_idx, (e_idx + 1) % 8]
        edge_vec = D_close - C_close
        tangent_m = edge_vec / torch.norm(edge_vec, dim=1, keepdim=True).clamp(min=1e-6)
        dot_m = tangent_m[:, 0] * intended_delta[:, 0] + tangent_m[:, 1] * intended_delta[:, 1]
        slide_m = tangent_m * torch.sign(dot_m).unsqueeze(1) * torch.norm(intended_delta, dim=1, keepdim=True) * 0.38
        resolved_mountain = self.pos + slide_m
        
        # 结合滑动
        resolved = torch.where(any_mountain_collided.unsqueeze(1), resolved_mountain, resolved_tree)
        new_pos = torch.where(any_collided.unsqueeze(1), resolved, candidate)
        new_pos = torch.clamp(new_pos, -self.world_size * 0.55, self.world_size * 0.55)
        
        displacement_vec = new_pos - self.pos
        displacement = torch.norm(displacement_vec, dim=1)
        self.pos = new_pos
        
        # 减扣宏步数
        self.macro_steps_left = torch.where(macro_active, self.macro_steps_left - 1, self.macro_steps_left)
        # 宏动作跑完后清零 mode
        self.macro_recovery_mode = torch.where(self.macro_steps_left <= 0, torch.zeros_like(self.macro_recovery_mode), self.macro_recovery_mode)
        
        # ==========================================
        # Timer and reward calculation
        # ==========================================
        curr_dist = torch.norm(self.target - self.pos, dim=1)
        progress = self.prev_dist - curr_dist
        self.prev_dist = curr_dist
        
        # stuck 和 no_progress 计时
        self.no_progress_time = torch.where(
            (displacement < 1.0) | (progress < 0.2),
            self.no_progress_time + self.dt,
            torch.clamp(self.no_progress_time - self.dt * 2.0, min=0.0)
        )
        self.stuck_time = torch.where(
            any_collided | (displacement < 0.6),
            self.stuck_time + self.dt,
            torch.clamp(self.stuck_time - self.dt * 3.0, min=0.0)
        )
        stuck = (self.stuck_time > 0.4) | (self.no_progress_time > 0.9)
        
        target_unit = (self.target - old_pos) / torch.clamp(self.prev_dist.unsqueeze(1), min=1e-6)
        forward_distance = (displacement_vec * target_unit).sum(dim=1)
        lateral_vec = displacement_vec - target_unit * forward_distance.unsqueeze(1)
        lateral_distance = torch.norm(lateral_vec, dim=1)
        
        # 各种评估因子
        target_error = torch.clamp(curr_dist - 8.0, min=0.0)
        angle_err = self._angle_error()
        angle_penalty = torch.clamp(angle_err.abs() / math.pi, max=1.0)
        far_weight = torch.clamp(target_error / 420.0, 0.0, 1.0)
        
        action_value = self.ANGLE_OFFSETS[angle_bin]
        action_change = (action_value - self.last_action_value).abs()
        action_flip = ((self.last_action_value.abs() > 0.06) & (action_value.abs() > 0.06) & (self.last_action_value * action_value < 0.0)).float()
        self.last_action_value = action_value
        
        progress_efficiency = torch.where(displacement > 0.4, progress / torch.clamp(displacement, min=1e-6), torch.zeros_like(progress))
        progress_efficiency = torch.clamp(progress_efficiency, -1.0, 1.0)
        inefficiency = torch.clamp(1.0 - progress_efficiency, min=0.0)
        
        heading_change = (self.heading - prev_heading).abs()
        heading_change = torch.where(heading_change > math.pi, 2.0 * math.pi - heading_change, heading_change)
        
        # 基础奖励计算 (Target Line V2 Profile)
        reward = torch.clamp(progress, -40.0, 40.0) * 0.18 - 0.14 * self.dt
        reward = reward + progress_efficiency * 0.32
        reward = reward - inefficiency * (0.08 + 0.18 * far_weight)
        reward = reward - lateral_distance * (0.080 + 0.070 * far_weight)
        reward = torch.where(forward_distance < 0.0, reward + forward_distance * 0.120, reward)
        reward = reward - angle_penalty * (0.75 + 1.55 * far_weight)
        reward = reward - torch.clamp(heading_change / math.pi, max=1.0) * (0.28 + 0.42 * far_weight)
        reward = reward - action_value.abs() * (0.08 + 0.24 * far_weight)
        reward = reward - action_change * (0.08 + 0.14 * far_weight)
        reward = reward - action_flip * (0.20 + 0.38 * far_weight)
        
        # 速度和对齐额外奖励
        reward = torch.where((curr_dist > 40.0) & (angle_penalty < 0.10), reward + (speed / 100.0) * 0.08, reward)
        reward = torch.where(target_error <= 40.0, reward + (1.0 - angle_penalty) * 0.22, reward)
        
        # 额外惩罚与脱困释放奖励
        reward = torch.where(any_collided, reward - 0.22, reward)
        reward = torch.where(self.no_progress_time > 1.0, reward - 0.08 * torch.clamp(self.no_progress_time, max=5.0), reward)
        reward = torch.where(prev_stuck & ~stuck & (progress > 0.0), reward + 0.80, reward)
        
        # 判定成功与终止
        reached = curr_dist < 8.0
        truncated = self.step_count >= 240
        dones = reached | truncated
        
        reward = torch.where(reached, reward + 20.0, reward)
        reward = torch.where(truncated & ~reached, reward - 5.0, reward)
        
        # ==========================================
        # Auto-Reset for Done Environments
        # ==========================================
        if dones.any():
            done_indices = torch.where(dones)[0]
            num_dones = len(done_indices)
            
            # 重新采样起点终点
            r_pos, r_target = self._sample_start_and_target(num_dones)
            self.pos[done_indices] = r_pos
            self.target[done_indices] = r_target
            self._generate_obstacles_for_envs(done_indices, r_pos, r_target)
            
            # 起点受困重新处理
            r_trapped_mask = torch.rand(num_dones, device=self.device) < 0.55
            r_trapped_indices = done_indices[r_trapped_mask]
            if len(r_trapped_indices) > 0:
                centers_0 = self.mountain_centers[r_trapped_indices, 0]
                radii_0 = self.mountain_radii[r_trapped_indices, 0]
                angles_0 = torch.rand(len(r_trapped_indices), device=self.device) * 2.0 * math.pi
                
                offsets_pos = torch.stack([torch.cos(angles_0), torch.sin(angles_0)], dim=1) * radii_0.unsqueeze(1) * 0.4
                self.pos[r_trapped_indices] = centers_0 + offsets_pos
                
                side_rot = torch.where(torch.rand(len(r_trapped_indices), device=self.device) < 0.5, 78.0, -78.0) * (math.pi / 180.0)
                rot_angles = angles_0 + side_rot
                target_dists = torch.rand(len(r_trapped_indices), device=self.device) * (1296.0 - 180.0) + 180.0
                offsets_target = torch.stack([torch.cos(rot_angles), torch.sin(rot_angles)], dim=1) * target_dists.unsqueeze(1)
                self.target[r_trapped_indices] = torch.clamp(self.pos[r_trapped_indices] + offsets_target, -self.world_size * 0.6, self.world_size * 0.6)
                
                self._generate_obstacles_for_envs(r_trapped_indices, self.pos[r_trapped_indices], self.target[r_trapped_indices])
                
            # 车头初始化
            self.heading[done_indices] = (torch.rand(num_dones, device=self.device) * 2.0 - 1.0) * math.pi
            r_opposite_mask = torch.rand(num_dones, device=self.device) < 0.5
            r_delta_to_target = self.target[done_indices] - self.pos[done_indices]
            r_target_angles = torch.atan2(r_delta_to_target[:, 1], r_delta_to_target[:, 0])
            r_opposite_headings = r_target_angles + math.pi + (torch.rand(num_dones, device=self.device) * 0.4 - 0.2)
            self.heading[done_indices] = torch.where(r_opposite_mask, r_opposite_headings, self.heading[done_indices])
            self.heading[done_indices] = torch.atan2(torch.sin(self.heading[done_indices]), torch.cos(self.heading[done_indices]))
            
            # 清零
            self.step_count[done_indices] = 0
            self.stuck_time[done_indices] = 0.0
            self.no_progress_time[done_indices] = 0.0
            self.time_since_collision[done_indices] = 10.0
            self.coord_age[done_indices] = 0.0
            
            self.prev_dist[done_indices] = torch.norm(self.target[done_indices] - self.pos[done_indices], dim=1)
            self.prev_angle_error[done_indices] = self._angle_error()[done_indices]
            self.last_action_value[done_indices] = 0.0
            
            self.macro_steps_left[done_indices] = 0
            self.macro_recovery_mode[done_indices] = 0
            self.macro_recovery_total_steps[done_indices] = 0
            self.macro_anchor_heading[done_indices] = 0.0
            
        curr_angle_error = self._angle_error()
        self.prev_angle_error = curr_angle_error
        
        return self._get_obs(), reward, dones, reached

# =====================================================================
# 2. PPO Recurrent (LSTM) Network Architecture
# =====================================================================
class RecurrentActorCritic(nn.Module):
    def __init__(self, state_dim=12, hidden_dim=64):
        super().__init__()
        # LSTM 记忆单元
        self.lstm = nn.LSTM(state_dim, hidden_dim, num_layers=1)
        
        # Actor 输出头 (Steer: 5, Speed: 2, Macro: 8)
        self.actor_fc = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.Tanh()
        )
        self.steer_head = nn.Linear(32, 5)
        self.speed_head = nn.Linear(32, 2)
        self.macro_head = nn.Linear(32, 8)
        
        # Critic 输出头
        self.critic_fc = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )
        
    def get_states(self, x, lstm_state, dones):
        # x: [seq_len, batch_size, state_dim]
        # lstm_state: tuple of (h, c) each [1, batch_size, hidden_dim]
        # dones: [seq_len, batch_size]
        seq_len = x.size(0)
        h, c = lstm_state
        
        lstm_outputs = []
        for t in range(seq_len):
            if t > 0:
                reset_mask = (dones[t-1] > 0.5).unsqueeze(0).unsqueeze(2) # [1, batch_size, 1]
                h = torch.where(reset_mask, torch.zeros_like(h), h)
                c = torch.where(reset_mask, torch.zeros_like(c), c)
                
            out, (h, c) = self.lstm(x[t].unsqueeze(0), (h, c))
            lstm_outputs.append(out.squeeze(0))
            
        lstm_outputs = torch.stack(lstm_outputs, dim=0) # [seq_len, batch_size, hidden_dim]
        return lstm_outputs, (h, c)
        
    def get_action_and_value(self, x, lstm_state, dones, action=None):
        lstm_outputs, next_lstm_state = self.get_states(x, lstm_state, dones)
        flat_outputs = lstm_outputs.view(-1, lstm_outputs.size(-1))
        
        actor_features = self.actor_fc(flat_outputs)
        steer_logits = self.steer_head(actor_features)
        speed_logits = self.speed_head(actor_features)
        macro_logits = self.macro_head(actor_features)
        
        dist_steer = torch.distributions.Categorical(logits=steer_logits)
        dist_speed = torch.distributions.Categorical(logits=speed_logits)
        dist_macro = torch.distributions.Categorical(logits=macro_logits)
        
        if action is None:
            a_steer = dist_steer.sample()
            a_speed = dist_speed.sample()
            a_macro = dist_macro.sample()
            action = torch.stack([a_steer, a_speed, a_macro], dim=1) # [seq_len * batch_size, 3]
        else:
            a_steer = action[:, 0]
            a_speed = action[:, 1]
            a_macro = action[:, 2]
            
        logprob = dist_steer.log_prob(a_steer) + dist_speed.log_prob(a_speed) + dist_macro.log_prob(a_macro)
        entropy = dist_steer.entropy() + dist_speed.entropy() + dist_macro.entropy()
        value = self.critic_fc(flat_outputs)
        
        return action, logprob, entropy, value, next_lstm_state

    def get_value(self, x, lstm_state, dones):
        lstm_outputs, _ = self.get_states(x, lstm_state, dones)
        flat_outputs = lstm_outputs.view(-1, lstm_outputs.size(-1))
        return self.critic_fc(flat_outputs)

# =====================================================================
# 3. Training Loop (Extreme Parallel GPU Speed!)
# =====================================================================
def train_gpu_ppo(total_episodes=100000, num_envs=8192, num_steps=128):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"采用设备: {device} | 进程环境数 (Environments): {num_envs} | 单次收集步数 (Rollout Steps): {num_steps}")
    
    # 物理环境
    env = GPUBlindNavEnvV11b(num_envs=num_envs, device=device)
    
    # 网络 & 优化器
    agent = RecurrentActorCritic(state_dim=12, hidden_dim=64).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=4e-4, eps=1e-5)
    
    # 核心训练缓冲池
    obs_batch = torch.zeros((num_steps, num_envs, 12), device=device)
    actions_batch = torch.zeros((num_steps, num_envs, 3), dtype=torch.long, device=device)
    logprobs_batch = torch.zeros((num_steps, num_envs), device=device)
    rewards_batch = torch.zeros((num_steps, num_envs), device=device)
    dones_batch = torch.zeros((num_steps, num_envs), device=device)
    values_batch = torch.zeros((num_steps, num_envs), device=device)
    
    # 初始状态
    next_obs = env.reset()
    next_done = torch.zeros(num_envs, device=device)
    
    # LSTM 初始化隐状态
    next_lstm_state_h = torch.zeros(1, num_envs, 64, device=device)
    next_lstm_state_c = torch.zeros(1, num_envs, 64, device=device)
    
    # 评估指标
    total_steps_collected = 0
    episodes_finished = 0
    episodes_successes = 0
    success_rate_smooth = 0.0
    reward_smooth = 0.0
    
    # GAE 参数
    gamma = 0.99
    gae_lambda = 0.95
    ppo_epochs = 4
    clip_coef = 0.2
    ent_coef = 0.01
    vf_coef = 0.5
    max_grad_norm = 0.5
    
    start_time = time.time()
    
    # 循环收集与更新，直至跑满 100,000 个 Episode
    update_iter = 0
    while episodes_finished < total_episodes:
        update_iter += 1
        
        # 记录本轮 Rollout 开始时每个环境的初始 LSTM 隐状态
        initial_lstm_state_h = next_lstm_state_h.clone()
        initial_lstm_state_c = next_lstm_state_c.clone()
        
        # 1. 收集 Rollouts (全部向量化在 GPU 执行)
        current_rollout_successes = 0
        current_rollout_dones = 0
        
        for step in range(num_steps):
            obs_batch[step] = next_obs
            dones_batch[step] = next_done
            
            with torch.no_grad():
                action, logprob, _, value, next_lstm_state = agent.get_action_and_value(
                    next_obs.unsqueeze(0), 
                    (next_lstm_state_h, next_lstm_state_c), 
                    next_done.unsqueeze(0)
                )
                values_batch[step] = value.squeeze()
                next_lstm_state_h, next_lstm_state_c = next_lstm_state
                
            actions_batch[step] = action
            logprobs_batch[step] = logprob
            
            # 环境前进
            next_obs, reward, done, reached = env.step(action)
            rewards_batch[step] = reward
            next_done = done.float()
            
            current_rollout_successes += reached.sum().item()
            current_rollout_dones += done.sum().item()
            
        total_steps_collected += num_envs * num_steps
        episodes_finished += current_rollout_dones
        episodes_successes += current_rollout_successes
        
        # 增滑平稳数据
        if current_rollout_dones > 0:
            success_rate = current_rollout_successes / current_rollout_dones
            success_rate_smooth = 0.9 * success_rate_smooth + 0.1 * success_rate
        mean_reward = rewards_batch.mean().item()
        reward_smooth = 0.9 * reward_smooth + 0.1 * mean_reward
        
        # 2. 估计广义优势度 GAE
        with torch.no_grad():
            next_value = agent.get_value(
                next_obs.unsqueeze(0), 
                (next_lstm_state_h, next_lstm_state_c), 
                next_done.unsqueeze(0)
            ).squeeze()
            
            advantages = torch.zeros_like(rewards_batch)
            lastgaelam = 0
            for t in reversed(range(num_steps)):
                if t == num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones_batch[t + 1]
                    nextvalues = values_batch[t + 1]
                delta = rewards_batch[t] + gamma * nextvalues * nextnonterminal - values_batch[t]
                advantages[t] = lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values_batch
            
        # 3. 压扁环境维度准备 PPO 梯度更新
        # 对 recurrent 网络，必须在环境维度做 minibatch 划分，以保证 seq_len 时间维度的序列完整性
        b_obs = obs_batch
        b_actions = actions_batch
        b_logprobs = logprobs_batch
        b_dones = dones_batch
        b_advantages = advantages
        b_returns = returns
        
        # 开始更新 epochs
        for epoch in range(ppo_epochs):
            # 随机打乱环境索引
            env_permutation = torch.randperm(num_envs, device=device)
            # 以 4096 个环境作为一个小批次 (Minibatch size = 4096 envs)
            minibatch_envs = 4096
            
            for start in range(0, num_envs, minibatch_envs):
                end = start + minibatch_envs
                m_env_indices = env_permutation[start:end]
                
                # 切出切片 (seq_len, m_envs, ...)
                obs_mb = b_obs[:, m_env_indices]
                actions_mb = b_actions[:, m_env_indices]
                dones_mb = b_dones[:, m_env_indices]
                logprobs_mb = b_logprobs[:, m_env_indices].view(-1)
                advantages_mb = b_advantages[:, m_env_indices].view(-1)
                returns_mb = b_returns[:, m_env_indices].view(-1)
                
                init_h_mb = initial_lstm_state_h[:, m_env_indices]
                init_c_mb = initial_lstm_state_c[:, m_env_indices]
                
                # 前向推理
                _, newlogprob, entropy, newvalue, _ = agent.get_action_and_value(
                    obs_mb, 
                    (init_h_mb, init_c_mb), 
                    dones_mb, 
                    actions_mb.view(-1, 3)
                )
                
                logratio = newlogprob - logprobs_mb
                ratio = logratio.exp()
                
                # Policy loss
                pg_loss1 = -advantages_mb * ratio
                pg_loss2 = -advantages_mb * torch.clamp(ratio, 1.0 - clip_coef, 1.0 + clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                
                # Value loss
                v_loss = 0.5 * ((newvalue.squeeze() - returns_mb) ** 2).mean()
                
                # Entropy loss
                entropy_loss = entropy.mean()
                
                loss = pg_loss - ent_coef * entropy_loss + vf_coef * v_loss
                
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), max_grad_norm)
                optimizer.step()
                
        # 4. 打印指标与进展
        if update_iter % 5 == 0 or episodes_finished >= total_episodes:
            elapsed_time = time.time() - start_time
            steps_per_sec = int(total_steps_collected / elapsed_time)
            print(f"[{elapsed_time:5.1f}s] Ep: {episodes_finished:6d}/{total_episodes} | "
                  f"平滑成功率: {success_rate_smooth:.3f} | 平均奖励: {reward_smooth:.3f} | "
                  f"更新迭代: {update_iter:4d} | 速度 (Steps/s): {steps_per_sec}")
            
    # 5. 训练结束，保存模型权重
    save_path = Path("runs/v11b_gpu_finished")
    save_path.mkdir(parents=True, exist_ok=True)
    torch.save(agent.state_dict(), save_path / "policy_weights.pth")
    print(f"训练完成！总耗时 {time.time() - start_time:.1f} 秒。")
    print(f"模型权重已成功固化保存至: {save_path / 'policy_weights.pth'}")

if __name__ == "__main__":
    # 执行 10 万轮训练
    train_gpu_ppo(total_episodes=100000, num_envs=60000, num_steps=128)
