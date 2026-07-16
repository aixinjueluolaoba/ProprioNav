import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import numpy as np
import time
import os
import math
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon
from matplotlib.animation import FFMpegWriter
import subprocess

# ==========================================
# 1. GPU-accelerated 2D Environment in PyTorch (With Polygon Mountains!)
# ==========================================
class GPUBlindNavEnv:
    def __init__(self, num_envs=256, world_size=2400.0, dt=0.3, device="cuda"):
        self.num_envs = num_envs
        self.world_size = world_size
        self.dt = dt
        self.device = device
        
        # 障碍物数量
        self.tree_count = 90
        self.mountain_count = 16
        
        # 1. 树木半径和山体包络半径
        self.tree_radii = torch.full((num_envs, self.tree_count), 28.0, device=device) # [N, 90]
        self.mountain_radii = torch.rand((num_envs, self.mountain_count), device=device) * (220.0 - 85.0) + 85.0 # [N, 16]
        
        # 存储张量
        self.pos = torch.zeros((self.num_envs, 2), dtype=torch.float32, device=device)
        self.target = torch.zeros((self.num_envs, 2), dtype=torch.float32, device=device)
        self.heading = torch.zeros(self.num_envs, dtype=torch.float32, device=device)
        self.tree_centers = torch.zeros((self.num_envs, self.tree_count, 2), dtype=torch.float32, device=device)
        self.mountain_centers = torch.zeros((self.num_envs, self.mountain_count, 2), dtype=torch.float32, device=device)
        self.mountain_vertices = torch.zeros((self.num_envs, self.mountain_count, 8, 2), dtype=torch.float32, device=device) # 8边形山体
        
        self.step_count = torch.zeros(self.num_envs, dtype=torch.int32, device=device)
        self.stuck_time = torch.zeros(self.num_envs, dtype=torch.float32, device=device)
        self.no_progress_time = torch.zeros(self.num_envs, dtype=torch.float32, device=device)
        
        self.prev_dist = torch.zeros(self.num_envs, dtype=torch.float32, device=device)
        self.prev_angle_error = torch.zeros(self.num_envs, dtype=torch.float32, device=device)
        self.delta_angle_error = torch.zeros(self.num_envs, dtype=torch.float32, device=device)
        self.distance_progress = torch.zeros(self.num_envs, dtype=torch.float32, device=device)
        self.movement_dist = torch.zeros(self.num_envs, dtype=torch.float32, device=device)
        
        self.last_action_speed = torch.full((self.num_envs,), -1.0, device=device)
        self.last_action_jump = torch.zeros(self.num_envs, device=device)
        
        self.reset()
        
    def _sample_start_and_target(self, num_samples):
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
            
        # 1. 采样树木中心
        t_centers = torch.rand((num_envs, self.tree_count, 2), device=self.device) * (self.world_size * 1.8) - (self.world_size * 0.9)
        # 2. 采样山体中心与半径
        m_centers = torch.rand((num_envs, self.mountain_count, 2), device=self.device) * (self.world_size * 1.8) - (self.world_size * 0.9)
        m_radii = self.mountain_radii[env_indices] # [num_envs, 16]
        
        # 3. 避开起点和终点
        # 树木避开
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
        
        # 山体避开
        dist_to_pos_m = torch.norm(m_centers - player_pos.unsqueeze(1), dim=2)
        m_pos_too_close = dist_to_pos_m < (220.0 + 180.0) # 取山体最大包络半径
        dir_pos_m = m_centers - player_pos.unsqueeze(1)
        dir_pos_m = dir_pos_m / torch.clamp(torch.norm(dir_pos_m, dim=2, keepdim=True), min=1e-6)
        m_centers = torch.where(m_pos_too_close.unsqueeze(2), player_pos.unsqueeze(1) + dir_pos_m * 400.0, m_centers)
        
        dist_to_target_m = torch.norm(m_centers - target_pos.unsqueeze(1), dim=2)
        m_target_too_close = dist_to_target_m < (220.0 + 35.0 + 120.0)
        dir_target_m = m_centers - target_pos.unsqueeze(1)
        dir_target_m = dir_target_m / torch.clamp(torch.norm(dir_target_m, dim=2, keepdim=True), min=1e-6)
        m_centers = torch.where(m_target_too_close.unsqueeze(2), target_pos.unsqueeze(1) + dir_target_m * 375.0, m_centers)
        
        # 4. 生成多边形山体顶点 (8边形)
        num_vertices = 8
        angles = torch.linspace(0.0, 2.0 * math.pi, num_vertices + 1, device=self.device)[:-1] # [8]
        angles = angles.unsqueeze(0).unsqueeze(0).expand(num_envs, self.mountain_count, num_vertices) # [num_envs, 16, 8]
        # 添加角度扰动以创造不规则多边形
        angles = angles + (torch.rand_like(angles) * 0.36 - 0.18)
        # 每个顶点的放射长度
        r_vert = m_radii.unsqueeze(2) * (0.6 + torch.rand_like(angles) * 0.4) # [num_envs, 16, 8]
        
        v_x = r_vert * torch.cos(angles)
        v_y = r_vert * torch.sin(angles)
        m_vertices = m_centers.unsqueeze(2) + torch.stack([v_x, v_y], dim=3) # [num_envs, 16, 8, 2]
        
        # 5. 存回主数组
        self.tree_centers[env_indices] = t_centers
        self.mountain_centers[env_indices] = m_centers
        self.mountain_vertices[env_indices] = m_vertices
        
    def reset(self, seed=None):
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            
        self.pos, self.target = self._sample_start_and_target(self.num_envs)
        self.heading = (torch.rand(self.num_envs, dtype=torch.float32, device=self.device) * 2.0 - 1.0) * math.pi
        
        # 为所有环境生成树木与多边形山体
        self._generate_obstacles_for_envs(torch.arange(self.num_envs), self.pos, self.target)
        
        self.step_count.fill_(0)
        self.stuck_time.fill_(0.0)
        self.no_progress_time.fill_(0.0)
        
        self.prev_dist = torch.norm(self.target - self.pos, dim=1)
        self.prev_angle_error = self._angle_error()
        self.delta_angle_error.fill_(0.0)
        self.distance_progress.fill_(0.0)
        self.movement_dist.fill_(0.0)
        
        self.last_action_speed.fill_(-1.0)
        self.last_action_jump.fill_(0.0)
        
        return self._get_obs()

    def _angle_error(self):
        delta = self.target - self.pos
        target_angle = torch.atan2(delta[:, 1], delta[:, 0])
        err = target_angle - self.heading
        return torch.atan2(torch.sin(err), torch.cos(err))
        
    def _get_obs(self):
        delta = self.target - self.pos
        obs = torch.stack([
            torch.clamp(delta[:, 0] / self.world_size, -1.0, 1.0),
            torch.clamp(delta[:, 1] / self.world_size, -1.0, 1.0),
            torch.clamp(self.prev_angle_error / math.pi, -1.0, 1.0),
            torch.clamp(self.delta_angle_error / math.pi, -1.0, 1.0),
            torch.clamp(self.distance_progress / 30.0, -1.0, 1.0),
            torch.clamp(self.movement_dist / 40.0, 0.0, 1.0),
            torch.clamp(self.last_action_speed, -1.0, 1.0),
            torch.clamp(self.last_action_jump, -1.0, 1.0)
        ], dim=1)
        return obs
        
    def step(self, actions):
        self.step_count += 1
        
        # 动作提取与缩放
        delta_angle = actions[:, 0] * math.radians(45.0)
        speed = 50.0 + (actions[:, 1] + 1.0) / 2.0 * 50.0
        jump_active = actions[:, 2] > 0.0
        
        self.last_action_speed = actions[:, 1]
        self.last_action_jump = actions[:, 2]
        
        prev_stuck = (self.stuck_time > 0.4) | (self.no_progress_time > 0.9)
        old_pos = self.pos.clone()
        
        # 转向与移动计算
        self.heading = self.heading + delta_angle
        self.heading = torch.atan2(torch.sin(self.heading), torch.cos(self.heading))
        
        intended_delta = torch.stack([torch.cos(self.heading), torch.sin(self.heading)], dim=1) * speed.unsqueeze(1) * self.dt
        candidate = self.pos + intended_delta
        
        # ==========================================
        # A. 树木碰撞判定 (圆形障碍物)
        # ==========================================
        diff_t = candidate.unsqueeze(1) - self.tree_centers # [N, 90, 2]
        dist_t = torch.norm(diff_t, dim=2) # [N, 90]
        tree_collided = dist_t < self.tree_radii # [N, 90]
        any_tree_collided = tree_collided.any(dim=1)
        
        # ==========================================
        # B. 多边形山体碰撞判定 (点在多边形内 & 线段相交)
        # ==========================================
        # 1. 射线检测法判定 Candidate 是否位于多边形山体内 (Point-in-Polygon)
        P_cand = candidate.unsqueeze(1).unsqueeze(2) # [N, 1, 1, 2]
        C = self.mountain_vertices # [N, 16, 8, 2]
        D = torch.roll(self.mountain_vertices, shifts=-1, dims=2) # [N, 16, 8, 2]
        c_x, c_y = C[..., 0], C[..., 1]
        d_x, d_y = D[..., 0], D[..., 1]
        p_x, p_y = P_cand[..., 0], P_cand[..., 1]
        
        cond1 = (c_y > p_y) != (d_y > p_y)
        denom = d_y - c_y
        denom = torch.where(denom.abs() < 1e-6, torch.sign(denom) * 1e-6, denom)
        x_intersect = (d_x - c_x) * (p_y - c_y) / denom + c_x
        ray_cross = cond1 & (p_x < x_intersect)
        inside = (ray_cross.sum(dim=2) % 2 == 1) # [N, 16]
        
        # 2. 判定移动线段是否与山体的边相交 (Segment-Segment Intersection)
        A_seg = self.pos.unsqueeze(1).unsqueeze(2) # [N, 1, 1, 2]
        B_seg = candidate.unsqueeze(1).unsqueeze(2) # [N, 1, 1, 2]
        
        cp1 = (B_seg[..., 0] - A_seg[..., 0]) * (C[..., 1] - A_seg[..., 1]) - (B_seg[..., 1] - A_seg[..., 1]) * (C[..., 0] - A_seg[..., 0])
        cp2 = (B_seg[..., 0] - A_seg[..., 0]) * (D[..., 1] - A_seg[..., 1]) - (B_seg[..., 1] - A_seg[..., 1]) * (D[..., 0] - A_seg[..., 0])
        cp3 = (D[..., 0] - C[..., 0]) * (A_seg[..., 1] - C[..., 1]) - (D[..., 1] - C[..., 1]) * (A_seg[..., 0] - C[..., 0])
        cp4 = (D[..., 0] - C[..., 0]) * (B_seg[..., 1] - C[..., 1]) - (D[..., 1] - C[..., 1]) * (B_seg[..., 0] - C[..., 0])
        edge_hit = (cp1 * cp2 < 0.0) & (cp3 * cp4 < 0.0) # [N, 16, 8]
        mountain_hit = edge_hit.any(dim=2) # [N, 16]
        
        # 综合判定山体碰撞
        mountain_collided = inside | mountain_hit # [N, 16]
        any_mountain_collided = mountain_collided.any(dim=1) # [N]
        
        any_collided = any_tree_collided | any_mountain_collided
        
        # ==========================================
        # C. 碰撞解算 (滑动滑动滑动!)
        # ==========================================
        # 1. 树木滑动结算 (Tree Circle Slide)
        penetration_t = 28.0 - dist_t
        max_pen_idx = torch.argmax(torch.where(tree_collided, penetration_t, -1e9), dim=1)
        coll_centers_t = self.tree_centers[torch.arange(self.num_envs), max_pen_idx]
        normal_t = (candidate - coll_centers_t)
        normal_t = normal_t / torch.clamp(torch.norm(normal_t, dim=1, keepdim=True), min=1e-6)
        tangent_t = torch.stack([-normal_t[:, 1], normal_t[:, 0]], dim=1)
        dot_t = tangent_t[:, 0] * intended_delta[:, 0] + tangent_t[:, 1] * intended_delta[:, 1]
        slide_t = tangent_t * torch.sign(dot_t).unsqueeze(1) * torch.norm(intended_delta, dim=1, keepdim=True) * 0.45
        resolved_tree = self.pos + slide_t
        jump_pos_t = candidate + normal_t * 12.0
        resolved_tree = torch.where(jump_active.unsqueeze(1), jump_pos_t, resolved_tree)
        
        # 2. 多边形山体滑动结算 (Polygon Edge Slide)
        # 寻找距离 Candidate 最近的山体的边
        P_close = candidate.unsqueeze(1).unsqueeze(2) # [N, 1, 1, 2]
        segment_m = D - C # [N, 16, 8, 2]
        length_sq_m = (segment_m ** 2).sum(dim=3)
        t_m = ((P_close - C) * segment_m).sum(dim=3) / torch.clamp(length_sq_m, min=1e-6)
        t_m = torch.clamp(t_m, 0.0, 1.0)
        closest_m = C + segment_m * t_m.unsqueeze(3)
        dist_sq_m = ((P_close - closest_m) ** 2).sum(dim=3) # [N, 16, 8]
        # 仅考虑发生碰撞的山体
        dist_sq_m = torch.where(mountain_collided.unsqueeze(2), dist_sq_m, 1e9)
        min_idx = torch.argmin(dist_sq_m.view(self.num_envs, -1), dim=1) # [N]
        m_idx = min_idx // 8
        e_idx = min_idx % 8
        
        C_close = self.mountain_vertices[torch.arange(self.num_envs), m_idx, e_idx]
        D_close = self.mountain_vertices[torch.arange(self.num_envs), m_idx, (e_idx + 1) % 8]
        edge_vec = D_close - C_close
        tangent_m = edge_vec / torch.norm(edge_vec, dim=1, keepdim=True).clamp(min=1e-6)
        dot_m = tangent_m[:, 0] * intended_delta[:, 0] + tangent_m[:, 1] * intended_delta[:, 1]
        slide_m = tangent_m * torch.sign(dot_m).unsqueeze(1) * torch.norm(intended_delta, dim=1, keepdim=True) * 0.38
        resolved_mountain = self.pos + slide_m
        
        # 多边形跳跃挣脱 (计算切线垂直法线，通过射线检测法确定朝外的一侧)
        normal_m = torch.stack([-tangent_m[:, 1], tangent_m[:, 0]], dim=1)
        # 用 ray-cast 检测 candidate + normal_m * 12.0 是否落入多边形内部，如果是则反向
        P_jump = (candidate + normal_m * 12.0).unsqueeze(1).unsqueeze(2)
        p_jump_x, p_jump_y = P_jump[..., 0], P_jump[..., 1]
        cond1_j = (c_y > p_jump_y) != (d_y > p_jump_y)
        x_intersect_j = (d_x - c_x) * (p_jump_y - c_y) / denom + c_x
        ray_cross_j = cond1_j & (p_jump_x < x_intersect_j)
        inside_jump = (ray_cross_j.sum(dim=2) % 2 == 1)
        is_inside_jump = inside_jump[torch.arange(self.num_envs), m_idx]
        normal_m = torch.where(is_inside_jump.unsqueeze(1), -normal_m, normal_m)
        jump_pos_m = candidate + normal_m * 12.0
        resolved_mountain = torch.where(jump_active.unsqueeze(1), jump_pos_m, resolved_mountain)
        
        # 3. 融合树木与山体的滑动结算
        resolved = torch.where(any_mountain_collided.unsqueeze(1), resolved_mountain, resolved_tree)
        new_pos = torch.where(any_collided.unsqueeze(1), resolved, candidate)
        new_pos = torch.clamp(new_pos, -self.world_size * 0.55, self.world_size * 0.55)
        
        self.movement_dist = torch.norm(new_pos - self.pos, dim=1)
        self.pos = new_pos
        
        # ==========================================
        # D. 计时器更新 & 奖励结算
        # ==========================================
        curr_dist = torch.norm(self.target - self.pos, dim=1)
        self.distance_progress = self.prev_dist - curr_dist
        self.prev_dist = curr_dist
        
        self.no_progress_time = torch.where(
            (self.movement_dist < 1.0) | (self.distance_progress < 0.2),
            self.no_progress_time + self.dt,
            torch.clamp(self.no_progress_time - self.dt * 2.0, min=0.0)
        )
        
        self.stuck_time = torch.where(
            any_collided | (self.movement_dist < 0.6),
            self.stuck_time + self.dt,
            torch.clamp(self.stuck_time - self.dt * 3.0, min=0.0)
        )
        
        curr_stuck = (self.stuck_time > 0.4) | (self.no_progress_time > 0.9)
        
        # 基础移动奖励与进度效率 (实际缩短距离 / 单步物理位移，即 距离/步数) 以及朝向夹角惩罚 (防止走弯路)
        displacement = self.movement_dist
        progress = self.distance_progress
        efficiency = torch.where(displacement > 0.1, progress / displacement, torch.zeros_like(progress))
        efficiency = torch.clamp(efficiency, min=-1.0, max=1.0)
        
        angle_error = self._angle_error()
        angle_penalty = angle_error.abs() / math.pi
        
        rewards = progress * 0.08 + efficiency * 0.15 - angle_penalty * 0.25 - 0.4 * self.dt
        
        # 扣分/奖励
        rewards = torch.where(any_collided, rewards - 0.25, rewards)
        rewards = torch.where(self.no_progress_time > 1.0, rewards - 0.05 * torch.clamp(self.no_progress_time, max=5.0), rewards)
        rewards = torch.where(jump_active & ~prev_stuck, rewards - 0.08, rewards)
        rewards = torch.where(jump_active & prev_stuck, rewards + 0.5, rewards)
        rewards = torch.where(prev_stuck & ~curr_stuck & (self.distance_progress > 0.0), rewards + 1.0, rewards)
        
        reached = curr_dist < 35.0
        truncated = self.step_count >= 240
        dones = reached | truncated
        
        rewards = torch.where(reached, rewards + 20.0, rewards)
        rewards = torch.where(truncated & ~reached, rewards - 5.0, rewards)
        
        # ==========================================
        # E. 单环境自动重置 (Auto-Reset)
        # ==========================================
        if dones.any():
            done_indices = torch.where(dones)[0]
            num_dones = len(done_indices)
            
            r_pos, r_target = self._sample_start_and_target(num_dones)
            self.pos[done_indices] = r_pos
            self.target[done_indices] = r_target
            self.heading[done_indices] = (torch.rand(num_dones, device=self.device) * 2.0 - 1.0) * math.pi
            
            self.step_count[done_indices] = 0
            self.stuck_time[done_indices] = 0.0
            self.no_progress_time[done_indices] = 0.0
            self.prev_dist[done_indices] = torch.norm(r_target - r_pos, dim=1)
            self.prev_angle_error[done_indices] = self._angle_error()[done_indices]
            self.delta_angle_error[done_indices] = 0.0
            self.distance_progress[done_indices] = 0.0
            self.movement_dist[done_indices] = 0.0
            self.last_action_speed[done_indices] = -1.0
            self.last_action_jump[done_indices] = 0.0
            
            # 为重置环境重新生成树木与多边形山体
            self._generate_obstacles_for_envs(done_indices, r_pos, r_target)
            
        curr_angle_error = self._angle_error()
        self.delta_angle_error = self.prev_angle_error - curr_angle_error
        self.prev_angle_error = curr_angle_error
        
        return self._get_obs(), rewards, dones

# ==========================================
# 2. PPO Network and Trainer in PyTorch
# ==========================================
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.actor_mean = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, action_dim),
            nn.Tanh()
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, action_dim))
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        
    def get_value(self, x):
        return self.critic(x)
        
    def get_action_and_value(self, x, action=None):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)

def train_ppo(device="cuda", total_timesteps=100000, num_envs=4096, num_steps=128):
    batch_size = num_envs * num_steps
    minibatch_size = max(256, batch_size // 16)
    
    env = GPUBlindNavEnv(num_envs=num_envs, device=device)
    agent = ActorCritic(8, 3).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=3e-4, eps=1e-5)
    
    obs_batch = torch.zeros((num_steps, num_envs, 8), device=device)
    actions_batch = torch.zeros((num_steps, num_envs, 3), device=device)
    logprobs_batch = torch.zeros((num_steps, num_envs), device=device)
    rewards_batch = torch.zeros((num_steps, num_envs), device=device)
    dones_batch = torch.zeros((num_steps, num_envs), device=device)
    values_batch = torch.zeros((num_steps, num_envs), device=device)
    
    next_obs = env.reset()
    next_done = torch.zeros(num_envs, device=device)
    
    num_updates = max(1, total_timesteps // batch_size)
    print(f"开始训练: {num_updates} 轮 PPO 迭代 (每轮批大小: {batch_size})...")
    
    start_time = time.time()
    for update in range(1, num_updates + 1):
        for step in range(num_steps):
            obs_batch[step] = next_obs
            dones_batch[step] = next_done
            
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values_batch[step] = value.squeeze()
            
            actions_batch[step] = action
            logprobs_batch[step] = logprob
            
            clipped_action = torch.clamp(action, -1.0, 1.0)
            next_obs, reward, done = env.step(clipped_action)
            rewards_batch[step] = reward
            next_done = done.float()
            
        with torch.no_grad():
            next_value = agent.get_value(next_obs).squeeze()
            advantages = torch.zeros_like(rewards_batch)
            lastgaelam = 0
            for t in reversed(range(num_steps)):
                if t == num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones_batch[t + 1]
                    nextvalues = values_batch[t + 1]
                delta = rewards_batch[t] + 0.99 * nextvalues * nextnonterminal - values_batch[t]
                advantages[t] = lastgaelam = delta + 0.99 * 0.95 * nextnonterminal * lastgaelam
            returns = advantages + values_batch
            
        b_obs = obs_batch.reshape((-1, 8))
        b_actions = actions_batch.reshape((-1, 3))
        b_logprobs = logprobs_batch.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values_batch.reshape(-1)
        
        for epoch in range(4):
            b_inds = torch.randperm(batch_size, device=device)
            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb_inds = b_inds[start:end]
                
                _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    b_obs[mb_inds], b_actions[mb_inds]
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()
                
                mb_advantages = b_advantages[mb_inds]
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)
                
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1.0 - 0.2, 1.0 + 0.2)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                
                v_loss = 0.5 * ((newvalue.squeeze() - b_returns[mb_inds]) ** 2).mean()
                entropy_loss = entropy.mean()
                loss = pg_loss - 0.01 * entropy_loss + 0.5 * v_loss
                
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), 0.5)
                optimizer.step()
                
        if update % 1 == 0 or update == num_updates:
            elapsed = time.time() - start_time
            sps = (update * batch_size) / elapsed
            avg_reward = rewards_batch.mean().item()
            print(f"迭代 {update}/{num_updates} | 累计步数: {update * batch_size} | 平均单步奖励: {avg_reward:.4f} | 速度: {sps:,.0f} 步/秒")
            
    return agent

# ==========================================
# 3. Video Rendering and Concatenation
# ==========================================
def render_evaluation_video(agent, seed, output_path, device="cuda"):
    env = GPUBlindNavEnv(num_envs=1, device=device)
    env.reset(seed=seed)
    
    start_pos = env.pos[0].cpu().numpy().copy()
    target = env.target[0].cpu().numpy()
    
    # 导出树木中心和半径
    tree_centers = env.tree_centers[0].cpu().numpy()
    tree_radii = env.tree_radii[0].cpu().numpy()
    
    paths = []
    max_steps = 240
    obs = env._get_obs()
    
    for _ in range(max_steps):
        paths.append(env.pos[0].cpu().numpy().copy())
        with torch.no_grad():
            action, _, _, _ = agent.get_action_and_value(obs)
            action = torch.clamp(action, -1.0, 1.0)
        obs, reward, done = env.step(action)
        if done[0]:
            break
            
    paths = np.array(paths)
    
    fig, ax = plt.subplots(figsize=(6, 6))
    limit = env.world_size * 0.55
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_aspect('equal')
    ax.set_title(f"Episode Evaluation (Seed: {seed})")
    
    ax.plot(start_pos[0], start_pos[1], 'go', markersize=10, label="Start")
    ax.plot(target[0], target[1], 'ro', markersize=10, label="Target")
    
    # 1. 绘制圆形树木
    for c, r in zip(tree_centers, tree_radii):
        circle = Circle(c, r, facecolor='#22c55e', alpha=0.3, edgecolor='none')
        ax.add_patch(circle)
        
    # 2. 绘制多边形山体
    for m in range(env.mountain_count):
        vertices = env.mountain_vertices[0, m].cpu().numpy()
        poly = Polygon(vertices, facecolor='#ef4444', alpha=0.3, edgecolor='none')
        ax.add_patch(poly)
        
    line, = ax.plot([], [], 'b-', linewidth=2, label="Agent Path")
    ax.legend(loc="upper left")
    
    writer = FFMpegWriter(fps=20, metadata=dict(artist='LuoKeNav'), bitrate=1000)
    
    with writer.saving(fig, str(output_path), dpi=100):
        for i in range(len(paths)):
            line.set_data(paths[:i+1, 0], paths[:i+1, 1])
            writer.grab_frame()
            
    plt.close(fig)
    return len(paths)

def main():
    import argparse
    parser = argparse.ArgumentParser(description="GPU-accelerated BlindNav RL Trainer & Evaluator")
    parser.add_argument("-t", "--total-timesteps", type=int, default=100000, help="Total training steps (default: 100000)")
    parser.add_argument("-n", "--n-envs", type=int, default=4096, help="Number of parallel environments (default: 4096)")
    parser.add_argument("-s", "--num-steps", type=int, default=128, help="Rollout steps per environment (default: 128)")
    parser.add_argument("-v", "--video-episodes", type=int, default=10, help="Number of evaluation episodes to render as video (default: 10)")
    parser.add_argument("-o", "--out-dir", type=str, default="/home/diana/fishing/blind_nav_rl/runs/test_100k_10videos", help="Output directory for checkpoints and videos")
    parser.add_argument("-d", "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to run simulation and training (default: cuda if available)")
    args = parser.parse_args()

    device = args.device
    print(f"当前运行硬件设备: {device}")
    
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. 训练 PPO
    print("-" * 50)
    print(f"开始 GPU 仿真和 PPO 强化学习训练 (总步数: {args.total_timesteps})...")
    agent = train_ppo(device=device, total_timesteps=args.total_timesteps, num_envs=args.n_envs, num_steps=args.num_steps)
    print("GPU 训练完成！")
    print("-" * 50)
    
    model_path = out_dir / "gpu_ppo_policy.pt"
    torch.save(agent.state_dict(), model_path)
    print(f"已保存策略模型权重至: {model_path}")
    
    # 2. 仿真评估并渲染视频
    print(f"开始执行 {args.video_episodes} 局测试评估并录制视频...")
    video_files = []
    seed_start = 140001
    
    for i in range(args.video_episodes):
        seed = seed_start + i
        video_path = out_dir / f"eval_{i+1:02d}.mp4"
        steps_taken = render_evaluation_video(agent, seed, video_path, device=device)
        print(f"评估局 {i+1}/{args.video_episodes} | 种子: {seed} | 耗时步数: {steps_taken} | 视频已保存")
        video_files.append(video_path)
        
    if len(video_files) == 0:
        print("未指定渲染任何视频。")
        return
        
    # 3. 拼接视频
    concat_list_path = out_dir / "concat_list.txt"
    with open(concat_list_path, "w") as f:
        for vf in video_files:
            f.write(f"file '{vf.absolute()}'\n")
            
    concat_mp4 = out_dir / "gpu_eval_concat.mp4"
    compressed_mp4 = out_dir / "gpu_eval_concat_compressed.mp4"
    
    print("使用 ffmpeg 执行视频无损拼接...")
    subprocess.run([
        "ffmpeg", "-y", "-f", "concat", "-safe", "0",
        "-i", str(concat_list_path), "-c", "copy", str(concat_mp4)
    ], check=True)
    
    print("执行视频压缩转码...")
    subprocess.run([
        "ffmpeg", "-y", "-i", str(concat_mp4),
        "-vf", "fps=20", "-c:v", "libx264", "-crf", "28",
        "-preset", "fast", "-pix_fmt", "yuv420p", str(compressed_mp4)
    ], check=True)
    
    print("=" * 60)
    print("所有操作完成！")
    print(f"最终拼接视频绝对路径: {compressed_mp4}")
    print("=" * 60)

if __name__ == "__main__":
    main()
