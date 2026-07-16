# blind_nav_rl 交接文档

本文档给后续 LLM / 工程师快速接手当前任务用，目标是避免重复翻历史会话。

## 1. 项目目标

这是一个 2D `gymnasium` 虚拟环境，用来训练“盲人寻路”策略：

- 不给地图输入
- 不给障碍几何信息
- 只依赖坐标类信息和人物朝向
- 最终希望迁移到真实游戏中的摇杆控制场景

当前重点不是做通用地图导航，而是验证：

1. 在只知道目标方向、人物方向、坐标延迟的情况下，RL 能否学出比“直线+脱困脚本”更稳的策略
2. 在凹形山体、反向起步、坐标延迟下，是否能学会合理脱困

## 2. 当前代码基线

当前默认接手候选已经升级为 `v11b stage2`，旧 `AE500` / 旧 `v11` 作为历史对照保留。

- experiment: `rppo_medium_lstm128x2_observable12_target8_macro_library_v11_stage2`
- state mode: `observable12_target8_macro_library_v11_stage2`
- action mode: `v11`
- env overrides: `configs/v11/方案B_卡住宏动作探索.json`
- 算法: `RecurrentPPO + MlpLstmPolicy`

当前判断依据：

- `v11b stage2` 在 100 局 pressure 复评里成功率 `1.00`，优于 `AE500` 的 `0.91`
- `v11b stage2` 在 100 局对角高障碍复评里成功率 `0.97`，优于 `AE500` 的 `0.77`
- 在固定 20 局对角高障碍视频场景里，`v11b stage2` 达到 `20/20` 成功，并显著降低碰撞和连续原地卡住

当前最优可直接用 checkpoint：

- stage: `stage2`
- checkpoint: `checkpoint_ep_01000.zip`
- 远端模型路径：
  `/home/diana/fishing/blind_nav_rl/runs/目标8宏动作库v11b卡住探索短训_v1002/rppo_medium_lstm128x2_observable12_target8_macro_library_v11_stage2/models/checkpoint_ep_01000.zip`
- 说明：
  - `01000 / 01500 / 02000` 的 watcher 指标基本持平
  - `01000` 是最早达到该水平的 checkpoint，应优先作为交付和后续 resume 起点

当前最佳成果预览：

- ASCII 主入口预览：
  `http://47.99.153.111:12346/file/target8-best/preview.html`
- ASCII 主入口清单：
  `http://47.99.153.111:12346/file/target8-best/manifest.json`
- 本地预览：
  `http://127.0.0.1:6000/file/%E7%9B%AE%E6%A0%878%E5%AE%8F%E5%8A%A8%E4%BD%9C%E5%BA%93v11b%E5%BD%93%E5%89%8D%E6%9C%80%E4%BD%B3%E6%88%90%E6%9E%9C/%E6%88%90%E6%9E%9C%E9%A2%84%E8%A7%88.html`
- 稳定短链接预览：
  `http://47.99.153.111:12346/file/%E7%9B%AE%E6%A0%878%E5%BD%93%E5%89%8D%E6%9C%80%E4%BD%B3%E6%88%90%E6%9E%9C/%E6%88%90%E6%9E%9C%E9%A2%84%E8%A7%88.html`
- 稳定短链接清单：
  `http://47.99.153.111:12346/file/%E7%9B%AE%E6%A0%878%E5%BD%93%E5%89%8D%E6%9C%80%E4%BD%B3%E6%88%90%E6%9E%9C/%E5%88%86%E4%BA%AB%E6%B8%85%E5%8D%95.json`
- 已验证外网直连：
  `http://47.99.153.111:12346/file/%E7%9B%AE%E6%A0%878%E5%AE%8F%E5%8A%A8%E4%BD%9C%E5%BA%93v11b%E5%BD%93%E5%89%8D%E6%9C%80%E4%BD%B3%E6%88%90%E6%9E%9C/%E6%88%90%E6%9E%9C%E9%A2%84%E8%A7%88.html`
- 代理候选链接：
  `http://ai.jiaran.icu:12346/file/%E7%9B%AE%E6%A0%878%E5%AE%8F%E5%8A%A8%E4%BD%9C%E5%BA%93v11b%E5%BD%93%E5%89%8D%E6%9C%80%E4%BD%B3%E6%88%90%E6%9E%9C/%E6%88%90%E6%9E%9C%E9%A2%84%E8%A7%88.html`
- 服务自报公开基址：
  `http://47.99.153.111:12346/file`
- 分享目录：
  `/home/diana/screencap/file/目标8宏动作库v11b当前最佳成果`
- 稳定短目录：
  `/home/diana/screencap/file/目标8当前最佳成果`
- ASCII 稳定短目录：
  `/home/diana/screencap/file/target8-best`
- 说明：
  - 本地 `127.0.0.1:6000/file/...` 已验证可访问
  - `47.99.153.111:12346/file/target8-best/...` 已验证 `preview.html / manifest.json / normal.mp4 / pressure.mp4 / diagonal.mp4` 都可访问
  - 远端归档脚本和本地同步脚本现在都会生成稳定短目录 `目标8当前最佳成果`，其中包含 `成果预览.html`、`链接清单.md`、`分享清单.json`
  - 远端归档脚本和本地同步脚本现在也会生成 ASCII 短目录 `target8-best`，其中包含 `preview.html`、`links.md`、`manifest.json`、`normal.mp4`、`pressure.mp4`、`diagonal.mp4`
  - 代理候选 `ai.jiaran.icu:12346/file/...` 在 shell 与 Playwright 浏览器上下文中都返回 `403`
  - 该 `403` 页面标题为 `Non-compliance ICP Filing`
  - 直接访问 `http://47.99.153.111:12346/` 会命中当前本机 `uvicorn` 服务，说明 `frpc` 的 `6000 -> 12346` 隧道本身是通的
  - `screencap-dataset-ui` 已改为支持 `PUBLIC_FILE_BASE` 覆盖，当前 `run.sh` 默认导出 `http://47.99.153.111:12346/file`
  - 本地请求 `/api/files` 时，当前自报 `public_base = http://47.99.153.111:12346/file`
  - 结论：当前可用的公网文件出口是 `47.99.153.111:12346/file/...`；问题只剩域名 `ai.jiaran.icu:12346` 的 Host/边缘策略拦截

相关文件：

- 环境实现: [blind_nav_rl/env.py](/home/diana/fishing/blind_nav_rl/blind_nav_rl/env.py)
- 实验配置: [benchmark_state_dims_10k.py](/home/diana/fishing/blind_nav_rl/benchmark_state_dims_10k.py)
- 总说明: [README.md](/home/diana/fishing/blind_nav_rl/README.md)
- `v11b` 训练脚本: [tools/启动_v11b卡住宏动作探索短训_v1002.sh](/home/diana/fishing/blind_nav_rl/tools/启动_v11b卡住宏动作探索短训_v1002.sh)
- `v11b` watcher 脚本: [tools/启动_v11b卡住宏动作探索评估_v1002.sh](/home/diana/fishing/blind_nav_rl/tools/启动_v11b卡住宏动作探索评估_v1002.sh)
- `v11b` 当前最佳打印脚本: [tools/打印_v11b当前最佳checkpoint_v1002.sh](/home/diana/fishing/blind_nav_rl/tools/打印_v11b当前最佳checkpoint_v1002.sh)
- `v11b` 当前最佳归档脚本: [tools/归档_v11b当前最佳_v1002.sh](/home/diana/fishing/blind_nav_rl/tools/归档_v11b当前最佳_v1002.sh)
- `v11b` 同步并归档脚本: [tools/同步并归档_v11b当前最佳_v1002.sh](/home/diana/fishing/blind_nav_rl/tools/同步并归档_v11b当前最佳_v1002.sh)
- `v11b` 一键发布脚本: [tools/发布_v11b当前最佳成果.sh](/home/diana/fishing/blind_nav_rl/tools/发布_v11b当前最佳成果.sh)
- `v11b` 分享链路诊断脚本: [tools/诊断_v11b分享链路.sh](/home/diana/fishing/blind_nav_rl/tools/诊断_v11b分享链路.sh)

补充：

- 以后若只想重跑当前最佳归档并重新发布到本机分享目录，优先执行：
  `bash tools/发布_v11b当前最佳成果.sh`
- `打印_v11b当前最佳checkpoint_v1002.sh` 和 `归档_v11b当前最佳_v1002.sh` 已改为从 watcher 汇总自动选当前最佳，不再硬编码 `stage2 01000`
- 默认规则是：先筛 `normal_success=1.0 && pressure_success=1.0`，再按 `normal_avg_steps + pressure_avg_steps` 最小，之后按 `pressure_avg_collisions` 和更早 checkpoint 破同分

## 3. 当前状态空间

当前默认 12 维状态：

| # | 名称 | 中文解释 |
|---:|---|---|
| 1 | `dx_norm` | 目标相对旧坐标的 x 偏移 |
| 2 | `dy_norm` | 目标相对旧坐标的 y 偏移 |
| 3 | `sin_heading` | 当前人物方向正弦 |
| 4 | `cos_heading` | 当前人物方向余弦 |
| 5 | `sin_angle_error` | 目标方向减当前人物方向的角差正弦 |
| 6 | `cos_angle_error` | 目标方向减当前人物方向的角差余弦 |
| 7 | `coord_age_norm` | 当前坐标观测年龄，用于模拟坐标延迟 |
| 8 | `distance_progress_norm` | 最近一次有效坐标刷新后的距离进展 |
| 9 | `no_progress_time_norm` | 连续无进展时间 |
| 10 | `distance_norm` | 当前观测点到目标的归一化距离 |
| 11 | `recent_collision_norm` | 最近碰撞时近信号 |
| 12 | `stuck_time_norm` | 连续卡住时间 |

说明：

- 不输入地图
- 不输入障碍物位置
- 不输入碰撞法线、边端点、圆心等几何信息
- 这条线的假设是：让 LSTM 从时间序列里判断“是否卡住”和“何时切宏脱困动作”

## 4. 当前动作空间

当前默认动作是 `v11` 宏动作库离散组合：

| # | 名称 | 说明 |
|---:|---|---|
| 1 | `angle_bin` | 相对目标方向的离散偏角档位 |
| 2 | `speed_bin` | 速度档位 |
| 3 | `macro_bin` | 宏脱困动作选择，`0` 表示不触发，其余档位对应不同宏动作 |

重点：

- 这条线里，模型直接决定“直行控制 + 是否进入哪种宏脱困”
- `v11b` 在卡住/碰撞上下文下，对 `macro_bin=0` 增加了受控探索，避免学成长期顶障碍不动

## 5. 当前最重要的默认入口

- 普通评估：`eval_target8_sweep.py`
- 压力评估默认：`eval_target8_pressure.py --action-mode v11 --state-mode observable12_target8_macro_library_v11_stage2`
- 对角高障碍视频默认：`eval_target8_diagonal_dense_video.py`
- checkpoint watcher：`watch_v11_checkpoints.py`
- `v1002` watcher 启动脚本：`tools/启动_v11b卡住宏动作探索评估_v1002.sh`

## 6. 当前环境设定

`observable12_target8_macro_library_v11_stage2` 当前关键配置：

- `world_size = 1200`
- `max_steps = 260`
- `dt = 0.3` 秒/步
- `tree_count = 45`
- `mountain_count = 24`
- `tree_radius = 18`
- `mountain_radius_range = (35, 85)`
- `concave_mountain_probability = 0.92`
- `deep_concave_mountain_probability = 0.45`
- `target_radius = 8`
- `start_target_distance_range = (260, 680)`
- `position_obs_dt_range = (0.40, 0.80)`
- `coord_noise_std = 1.6`
- `trapped_start_probability = 0.30`
- `opposite_heading_probability = 0.45`
- `macro_recovery_steps = 6`

含义：

- 模拟真实环境中坐标更新慢、人物方向更新快
- 允许一开始朝向和目标相反
- stage2 比 stage3 更强调“中高密度障碍 + 可稳定学会宏脱困”，是当前更稳的默认候选

## 7. 停止规则

停止不是模型自己输出“停”，而是环境外部条件控制：

- 当 `distance_to_target <= 8px` 时，直接判定到达

这点是用户明确倾向：

- 停止半径由环境负责
- 模型主要学“每步保持较小角度误差”和“卡住时何时脱困”

## 8. 奖励设计现状

当前主奖励线使用 `target_line_macro_recovery_v10_stage2`，核心组成包括：

- 朝目标推进奖励
- 单步效率奖励 `progress / movement`
- 横向偏移惩罚
- 角度误差惩罚
- 频繁改向惩罚
- 左右来回翻转惩罚
- 卡住但不脱困时惩罚
- 脱困期间恢复进展时奖励

需要注意：

- `v11b` 不是新 reward 体系，而是在现有 reward 下修正“卡住时宏动作探索不足”
- 旧 `v11` 的关键坏解不是纯粹摆烂，而是碰撞后连续原地，现已通过 stall 指标和 `v11b` 对照确认

## 9. 已确认的问题

### 9.1 固定脱困 v1 没有可确认的好模型

目前没有确认到一个“可用”的 `observable9_target8_continuous_fixed_recovery_v1` 训练好模型。

已经明确见到的一个 smoke 结果是失败的：

- 路径: [runs/本地直线优先固定脱困_smoke/state_dim_results.csv](/home/diana/fishing/blind_nav_rl/runs/本地直线优先固定脱困_smoke/state_dim_results.csv)
- 现象:
  - `success_rate = 0.0`
  - `avg_steps = 280`
  - `avg_final_distance = 596.88`
  - `avg_collision_count = 279.5`

结论：不能把这个当成可迁移基线。

### 9.2 v1 主要风险

- 模型可能从头到尾不触发 `recovery_trigger`
- 只靠 9 维盲状态，学会“什么时候该触发模板”比较困难
- 随机左右模板可能在凹陷地形中不稳定

### 9.3 真实环境迁移难点

用户明确说明真实环境特点：

- 坐标更新大约 `500ms` 一次
- 人物方向/指针识别很快
- 摇杆角度和人物方向在第一帧不直接对应

因此模拟时要尽量保留：

- 坐标延迟
- 朝向快速更新
- 起步反向概率

## 10. 之前走过的主要路线

以下路线都做过，不要再从零重复讨论：

1. 纯 SAC / MLP 小网络
2. RecurrentPPO + LSTM
3. 8 维、9 维、10 维、12 维状态
4. 连续动作
5. 离散宏动作
6. 目标相对偏角输出
7. 摇杆增量角输出
8. 直线优先奖励
9. 目标圈停止任务
10. 学习型脱困
11. 固定模板脱困
12. 凹形山体高密度压力环境

经验结论：

- 只加轮数，不一定解决“不会触发脱困”
- 只靠奖励，不一定解决蛇形和目标周围反复穿插
- LSTM 比纯 MLP 更符合这个任务，但也不能自动解决所有部分可观测问题

## 11. 当前建议的接手方向

如果下一个 LLM 要继续推进，建议顺序如下：

1. 先确认本地、253、v1002 是否已有 `observable9_target8_continuous_fixed_recovery_v1` 或 `v2/v3/v4` 的可用 checkpoint
2. 如果没有，不要盲目长训 `v1`
3. 优先转到 `observable10_target8_continuous_fixed_recovery_v2/v3/v4` 这几条“显式强化 recovery trigger 学习信号”的分支
4. 评估时重点看：
   - 是否真的触发过 recovery
   - 触发后是否脱离碰撞
   - 是否仍出现长时间蛇形
   - 是否在目标附近反复穿过
5. 如果继续坚持“不输入地图”，那就把状态增量加得非常克制，只补最必要的时序提示

## 12. 设备和运行规范

用户已经明确给过设备规范，后续必须遵守。

### 本地

- AI 开发环境：`conda activate ML`
- 适合技术验证和轻任务
- CPU 不能超过 `85%`
- 内存不能超过 `80%`

后续用户又要求一键调度时尽量把本机控制在：

- CPU 使用率 `< 85%`
- 内存 `< 80%`

### ssh 253

- 工作目录：`/home/weiaokang`
- 项目目录：`/home/weiaokang/fishing/blind_nav_rl`
- Python 环境：`source /home/weiaokang/ML/bin/activate`
- 常规限制：
  - CPU `< 70%`
  - 内存 `< 70%`

### ssh v10032

- GPU 机器
- 工作目录：`/data`
- 环境：`conda activate ML`
- 需要留出显存，不要吃满

### ssh v1002

- 长期部署/任务调度机器
- 环境：`conda activate ML`

### 额外规则

- 下载依赖尽量用镜像源
- 如果能用多 agent 加速，尽量使用
- `/home/weiaokang/screencap/file` 和 `/tmp/file` 下的目录名、文件名都要求中文
- 253 上做完需要把产物拉回本地保存，再删除共享目录里的文件，避免占空间

## 13. 文件分享规范

用户已经明确指出：

- 展示文件优先放本地 `/home/diana/screencap/file/...`
- 然后提供 URL

不要把“展示给用户”的结果只留在远端机器上。

## 14. 建议先检查的内容

接手后第一轮建议检查：

1. `README.md` 顶部基线是否仍指向 9 维固定脱困
2. `benchmark_state_dims_10k.py` 中 `observable9_target8_continuous_fixed_recovery_v1` / `v2/v3/v4` 配置
3. `env.py` 中 `continuous_trigger_fixed_recovery` 分支逻辑
4. `runs/` 下是否存在真实有效 checkpoint 和评估结果
5. `tools/启动_连续9维固定脱困_253.sh` / `v1002.sh` 是否仍指向正确 experiment

## 15. 当前最重要的事实

当前仓库默认候选已经不是“连续动作 + 固定脱困模板”。

- 当前主候选：`v11b stage2`
- 历史对照：`AE500 v10 stage3`、旧 `v11`
- 当前结论来自已完成的 20 局视频评估和 100 局大样本复评，不只是代码基线切换
- 唯一保留意见是：`v11b stage2` 的 `still_step_rate / max_still_run` 仍可能高于 `AE500`，所以 `AE500` 还值得保留作“更激进 recovery 风格”的参考线

## 16. 备注

在一次本地查找中，`rtk find` 对中文路径出现过字符边界 panic。后续如果批量扫描中文目录，优先用：

- `rtk rg`
- `rtk ls`
- `rtk find ... | perl/python 处理` 之外的更稳妥方式

避免再次踩这个工具问题。
