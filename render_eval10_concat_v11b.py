import argparse
import math
import subprocess
import torch
import torch.nn as nn
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# 导入本项目中的依赖项
from train_gpu_v11b import RecurrentActorCritic
from benchmark_state_dims_10k import make_state_env, wrap_policy_env
from render_eval10_concat import render_episode_video, concat_and_compress

def run_eval_recurrent(model, seed, state_mode="observable12_target8_macro_library_v11rxc_stage2"):
    # 创建并包装对齐 v11b / v11rxc stage2 的 CPU 包装环境
    base_env = make_state_env(seed, state_mode)
    env = wrap_policy_env(base_env, state_mode)
    obs, _ = env.reset(seed=seed)
    
    # 初始化 LSTM 隐状态 (Batch size = 1)
    h = torch.zeros(1, 1, 64, dtype=torch.float32)
    c = torch.zeros(1, 1, 64, dtype=torch.float32)
    
    path = [env.base_env.pos.copy()]
    headings = [env.base_env.heading]
    total_reward = 0.0
    terminated = False
    truncated = False
    
    done = False
    while not done:
        # 转换为张量，增加 sequence_len=1 和 batch_size=1 维度
        obs_tensor = torch.tensor(obs, dtype=torch.float32).view(1, 1, 12)
        done_tensor = torch.tensor([float(done)], dtype=torch.float32).view(1, 1)
        
        with torch.no_grad():
            lstm_outputs, (h, c) = model.get_states(obs_tensor, (h, c), done_tensor)
            flat_outputs = lstm_outputs.view(-1, lstm_outputs.size(-1))
            actor_features = model.actor_fc(flat_outputs)
            
            # 确定性推理：取最大概率的 logits 对应的动作 bin
            a_steer = model.steer_head(actor_features).argmax(dim=-1).item()
            a_speed = model.speed_head(actor_features).argmax(dim=-1).item()
            a_macro = model.macro_head(actor_features).argmax(dim=-1).item()
            
            action = [a_steer, a_speed, a_macro]
            
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        path.append(env.base_env.pos.copy())
        headings.append(env.base_env.heading)
        total_reward += float(reward)
        
    return env.base_env, np.asarray(path), np.asarray(headings, dtype=np.float32), total_reward

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-weights", default="runs/v11b_gpu_finished/policy_weights.pth")
    parser.add_argument("--out-dir", default="eval_out")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed-start", type=int, default=140001)
    parser.add_argument("--fps", type=int, default=20)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 实例化 LSTM 神经网络模型
    model = RecurrentActorCritic(state_dim=12, hidden_dim=64)
    model.load_state_dict(torch.load(args.model_weights, map_location="cpu"))
    model.eval()
    print(f"成功载入 PyTorch 模型权重: {args.model_weights}")

    episode_videos = []
    success_count = 0
    
    for i in range(args.episodes):
        seed = args.seed_start + i
        print(f"正在进行测试 {i+1}/{args.episodes} (Seed: {seed})...", flush=True)
        
        env, path, headings, reward_sum = run_eval_recurrent(
            model,
            seed,
            state_mode="observable12_target8_macro_library_v11rxc_stage2"
        )
        
        output_video = out_dir / f"eval_{i+1:02d}.mp4"
        render_episode_video(env, path, reward_sum, output_video, fps=args.fps, headings=headings)
        
        dist = float(np.linalg.norm(env.target - path[-1]))
        is_success = dist <= env.target_radius
        if is_success:
            success_count += 1
            
        print(f"测试={i+1} 种子={seed} 步数={len(path)-1} 奖励={reward_sum:.3f} 终点距离={dist:.3f} 成功={'是' if is_success else '否'} 视频已生成", flush=True)
        episode_videos.append(output_video)

    # 视频拼接合并
    concat_mp4 = out_dir / "eval10_concat.mp4"
    compressed_mp4 = out_dir / "eval10_concat_compressed.mp4"
    concat_and_compress(episode_videos, concat_mp4, compressed_mp4, fps=args.fps)
    
    print("\n==================================================")
    print(f"测试完毕！成功率: {success_count}/{args.episodes} ({success_count/args.episodes*100:.1f}%)")
    print(f"未压缩拼接视频已输出至: {concat_mp4}")
    print(f"已压缩拼接视频已输出至: {compressed_mp4}")
    print("==================================================\n")

if __name__ == "__main__":
    main()
