# blind_nav_rl

2D gymnasium blind-navigation environment and RL training scripts for a joystick-driven game character. The current default candidate is `v11b stage2` macro-library recovery; the older continuous fixed-recovery line is now a historical rollback baseline:

- current default candidate: `rppo_medium_lstm128x2_observable12_target8_macro_library_v11_stage2`
- state mode: `observable12_target8_macro_library_v11_stage2`
- action mode: `v11`
- env overrides: `configs/v11/方案B_卡住宏动作探索.json`
- earliest equally-best checkpoint: `stage2 checkpoint_ep_01000.zip`
- final archive dir: `/home/diana/fishing/blind_nav_rl/runs/目标8宏动作库v11b当前最佳归档_v1002`
- canonical ascii preview: `http://47.99.153.111:12346/file/target8-best/preview.html`
- canonical ascii manifest: `http://47.99.153.111:12346/file/target8-best/manifest.json`
- local preview: `http://127.0.0.1:6000/file/%E7%9B%AE%E6%A0%878%E5%AE%8F%E5%8A%A8%E4%BD%9C%E5%BA%93v11b%E5%BD%93%E5%89%8D%E6%9C%80%E4%BD%B3%E6%88%90%E6%9E%9C/%E6%88%90%E6%9E%9C%E9%A2%84%E8%A7%88.html`
- stable short preview: `http://47.99.153.111:12346/file/%E7%9B%AE%E6%A0%878%E5%BD%93%E5%89%8D%E6%9C%80%E4%BD%B3%E6%88%90%E6%9E%9C/%E6%88%90%E6%9E%9C%E9%A2%84%E8%A7%88.html`
- stable short manifest: `http://47.99.153.111:12346/file/%E7%9B%AE%E6%A0%878%E5%BD%93%E5%89%8D%E6%9C%80%E4%BD%B3%E6%88%90%E6%9E%9C/%E5%88%86%E4%BA%AB%E6%B8%85%E5%8D%95.json`
- verified external preview: `http://47.99.153.111:12346/file/%E7%9B%AE%E6%A0%878%E5%AE%8F%E5%8A%A8%E4%BD%9C%E5%BA%93v11b%E5%BD%93%E5%89%8D%E6%9C%80%E4%BD%B3%E6%88%90%E6%9E%9C/%E6%88%90%E6%9E%9C%E9%A2%84%E8%A7%88.html`
- proxy preview candidate: `http://ai.jiaran.icu:12346/file/%E7%9B%AE%E6%A0%878%E5%AE%8F%E5%8A%A8%E4%BD%9C%E5%BA%93v11b%E5%BD%93%E5%89%8D%E6%9C%80%E4%BD%B3%E6%88%90%E6%9E%9C/%E6%88%90%E6%9E%9C%E9%A2%84%E8%A7%88.html`
- service public base: `http://47.99.153.111:12346/file`
- current external status: domain proxy `403 Non-compliance ICP Filing`, direct IP `200`
- current conclusion: the FRP tunnel is healthy and externally usable through `47.99.153.111:12346`; prefer the canonical ASCII alias `target8-best`, while `ai.jiaran.icu:12346` remains blocked at the remote edge
- 100-episode pressure re-eval: success `1.00`, avg steps `32.07`, avg collisions `9.34`
- 100-episode diagonal-dense re-eval: success `0.97`, avg steps `82.51`, avg collisions `20.76`

Current strongest non-default contender:

- reward/exit candidate: `rppo_medium_lstm128x2_observable12_target8_macro_library_v11re_stage2`
- env overrides: `configs/v11/方案H_脱困退出与收角强化.json`
- source checkpoint: `runs/v11re_reward_exit_probe/.../checkpoint_ep_01000.zip`
- pressure 100: success `1.00`, avg steps `29.17`, avg collisions `8.97`
- diagonal-dense 100: success `0.95`, avg steps `85.59`, avg collisions `27.16`
- interpretation:
  - clearly better than default on pressure speed and slightly better on pressure collisions
  - clearly worse than default on diagonal success/collisions, and slightly slower on diagonal
  - keep as a serious branch, but not a replacement for the default candidate yet

## v11ds Duration-Only Probe

Minimal next-step branch for the current `v11b stage2` line:

- experiment: `rppo_medium_lstm128x2_observable12_target8_macro_library_v11ds_stage2`
- state mode: `observable12_target8_macro_library_v11ds_stage2`
- action space: `MultiDiscrete([5, 2, 7, 3])`
- semantics: keep the existing v11 macro library, add one duration bin for `short / medium / long`

Entrypoints:

```bash
rtk bash -lc 'source ~/miniconda3/etc/profile.d/conda.sh && conda activate ML && python train_target8_v11ds_duration_curriculum.py --output-dir runs/v11ds_duration_probe --stage1-episodes 2000 --eval-episodes 20 --video-episodes 0 --device cpu --torch-threads 1 --env-overrides-file configs/v11/方案B_卡住宏动作探索_duration_only.json'
```

```bash
rtk bash -lc 'source ~/miniconda3/etc/profile.d/conda.sh && conda activate ML && python watch_v11ds_checkpoints.py --run-dir runs/v11ds_duration_probe --out-dir runs/v11ds_duration_probe_watch --curriculum --max-checkpoint 2000 --checkpoint-step 500 --stage2-max-checkpoint 2000'
```

New pressure / diagonal summary fields:

- `avg_max_stuck_run_steps`
- `avg_recovery_resolve_steps`
- `recovery_escape_rate`

## v11b Lite-Speed Probe

Conservative mainline continuation for `v11b stage2`, aimed at reducing pressure steps
without changing the macro-action interface:

- keep experiment/state family on `v11 stage2`
- resume from current best `stage2 checkpoint_ep_01000.zip`
- slightly reduce early macro exploration
- slightly strengthen non-recovery straightness / terminal alignment pressure

Entrypoints:

```bash
rtk bash -lc 'source ~/miniconda3/etc/profile.d/conda.sh && conda activate ML && python train_target8_v11_macro_library_curriculum.py --output-dir runs/v11b_lite_speed_probe --resume-model runs/目标8宏动作库v11b卡住探索短训_v1002/rppo_medium_lstm128x2_observable12_target8_macro_library_v11_stage2/models/checkpoint_ep_01000.zip --start-stage rppo_medium_lstm128x2_observable12_target8_macro_library_v11_stage2 --end-stage rppo_medium_lstm128x2_observable12_target8_macro_library_v11_stage2 --stage2-episodes 2000 --eval-episodes 20 --video-episodes 0 --device cpu --torch-threads 1 --env-overrides-file configs/v11/方案D_主线小幅压步数.json'
```

```bash
rtk bash -lc 'source ~/miniconda3/etc/profile.d/conda.sh && conda activate ML && python watch_v11b_lite_speed_checkpoints.py --run-dir runs/v11b_lite_speed_probe --out-dir runs/v11b_lite_speed_probe_watch --curriculum --max-checkpoint 2000 --checkpoint-step 500 --stage2-max-checkpoint 2000'
```

## v11b Weighted Quick Probe

当前最值得继续快筛的是 `方案F`，但先压短，不再直接跑 `2000 + 双20局 watcher`：

- resume from: `stage2 checkpoint_ep_01000.zip`
- env overrides: `configs/v11/方案F_偏置短脱困探索.json`
- quick-train target: `1000` episodes
- watcher checkpoints: `500 / 1000`

Entrypoints:

```bash
rtk bash -lc 'source ~/miniconda3/etc/profile.d/conda.sh && conda activate ML && python train_target8_v11_macro_library_curriculum.py --output-dir runs/v11b_weighted_quick_probe --resume-model runs/目标8宏动作库v11b卡住探索短训_v1002/rppo_medium_lstm128x2_observable12_target8_macro_library_v11_stage2/models/checkpoint_ep_01000.zip --start-stage rppo_medium_lstm128x2_observable12_target8_macro_library_v11_stage2 --end-stage rppo_medium_lstm128x2_observable12_target8_macro_library_v11_stage2 --stage2-episodes 1000 --eval-episodes 8 --video-episodes 0 --checkpoint-interval 500 --progress-interval 250 --device cpu --torch-threads 1 --env-overrides-file configs/v11/方案F_偏置短脱困探索.json'
```

```bash
rtk bash -lc 'source ~/miniconda3/etc/profile.d/conda.sh && conda activate ML && python watch_v11b_weighted_checkpoints.py --run-dir runs/v11b_weighted_quick_probe --out-dir runs/v11b_weighted_quick_probe_watch --max-checkpoint 1000 --checkpoint-step 500'
```

用途：

- 先判断 `方案F` 在更短 continuation 下是否还能稳定复现
- 快筛通过后，再补 `pressure 100 / diagonal 100`
- 快筛不过，就不再浪费到大样本评估

## v11tm Trigger-Macro Probe

下一条结构分支不再调 recovery 权重，而是把“是否触发脱困”和“选择哪种宏模板”拆开：

- experiment: `rppo_medium_lstm128x2_observable12_target8_macro_library_v11tm_stage2`
- state mode: `observable12_target8_macro_library_v11tm_stage2`
- action space: `MultiDiscrete([5, 2, 2, 6])`
- semantics:
  - `trigger_bin`: 先决定是否进入脱困
  - `macro_id`: 仅在触发脱困时选择 `6` 种 escape template

Entrypoints:

```bash
rtk bash -lc 'source ~/miniconda3/etc/profile.d/conda.sh && conda activate ML && python train_target8_v11tm_trigger_macro_curriculum.py --output-dir runs/v11tm_trigger_macro_probe --resume-model runs/目标8宏动作库v11b卡住探索短训_v1002/rppo_medium_lstm128x2_observable12_target8_macro_library_v11_stage2/models/checkpoint_ep_01000.zip --stage-episodes 1000 --eval-episodes 8 --video-episodes 0 --checkpoint-interval 500 --progress-interval 250 --device cpu --torch-threads 1 --env-overrides-file configs/v11/方案G_显式触发宏模板.json'
```

```bash
rtk bash -lc 'source ~/miniconda3/etc/profile.d/conda.sh && conda activate ML && python watch_v11tm_checkpoints.py --run-dir runs/v11tm_trigger_macro_probe --out-dir runs/v11tm_trigger_macro_probe_watch --max-checkpoint 1000 --checkpoint-step 500'
```

## v11re Reward-Exit Probe

这条分支不改动作头，直接强化“脱困后继续收角/重新贴线”的学习信号，并把宏脱困提前退出阈值改得更保守：

- experiment: `rppo_medium_lstm128x2_observable12_target8_macro_library_v11re_stage2`
- state mode: `observable12_target8_macro_library_v11re_stage2`
- action space: 保持 `MultiDiscrete([5, 2, 7])`
- 主要变化:
  - 提高非脱困阶段角度/转向稳定性压力
  - 提高 close align / terminal align bonus
  - 提高 recovery 后真正脱困再给奖励的权重
  - 把宏脱困提前退出条件变严格，减少“刚松一点就退出又转圈”

Entrypoints:

```bash
rtk bash -lc 'source ~/miniconda3/etc/profile.d/conda.sh && conda activate ML && python train_target8_v11re_reward_exit_curriculum.py --output-dir runs/v11re_reward_exit_probe --resume-model runs/目标8宏动作库v11b卡住探索短训_v1002/rppo_medium_lstm128x2_observable12_target8_macro_library_v11_stage2/models/checkpoint_ep_01000.zip --stage-episodes 1000 --eval-episodes 8 --video-episodes 0 --checkpoint-interval 500 --progress-interval 250 --device cpu --torch-threads 1 --env-overrides-file configs/v11/方案H_脱困退出与收角强化.json'
```

```bash
rtk bash -lc 'source ~/miniconda3/etc/profile.d/conda.sh && conda activate ML && python watch_v11re_checkpoints.py --run-dir runs/v11re_reward_exit_probe --out-dir runs/v11re_reward_exit_probe_watch --max-checkpoint 1000 --checkpoint-step 500'
```

Fast screen:

```bash
rtk bash -lc 'source ~/miniconda3/etc/profile.d/conda.sh && conda activate ML && python train_target8_v11re_reward_exit_curriculum.py --quick --output-dir runs/v11re_reward_exit_probe_quick --resume-model runs/目标8宏动作库v11b卡住探索短训_v1002/rppo_medium_lstm128x2_observable12_target8_macro_library_v11_stage2/models/checkpoint_ep_01000.zip --video-episodes 0 --device cpu --torch-threads 1 --env-overrides-file configs/v11/方案H_脱困退出与收角强化.json'
```

```bash
rtk bash -lc 'source ~/miniconda3/etc/profile.d/conda.sh && conda activate ML && python watch_v11re_checkpoints.py --quick --run-dir runs/v11re_reward_exit_probe_quick --out-dir runs/v11re_reward_exit_probe_quick_watch'
```

## v11re Balanced Follow-Up

如果 `方案H` 在 pressure 很强、但 diagonal 碰撞偏高，下一条最小 follow-up 直接复用 `v11re` 训练脚本，只换更平衡的 env overrides：

- env overrides: `configs/v11/方案I_脱困退出与收角平衡.json`
- intent:
  - 保留更好的收角和 pressure speed
  - 收一点 recovery bonus，避免在 diagonal 里过度滞留/反复磨障碍
  - 放松一点宏脱困退出阈值，减少长时间 recovery 挂起

Entrypoints:

```bash
rtk bash -lc 'source ~/miniconda3/etc/profile.d/conda.sh && conda activate ML && python train_target8_v11re_reward_exit_curriculum.py --output-dir runs/v11re_balanced_probe --resume-model runs/目标8宏动作库v11b卡住探索短训_v1002/rppo_medium_lstm128x2_observable12_target8_macro_library_v11_stage2/models/checkpoint_ep_01000.zip --stage-episodes 1000 --eval-episodes 8 --video-episodes 0 --checkpoint-interval 500 --progress-interval 250 --device cpu --torch-threads 1 --env-overrides-file configs/v11/方案I_脱困退出与收角平衡.json'
```

```bash
rtk bash -lc 'source ~/miniconda3/etc/profile.d/conda.sh && conda activate ML && python watch_v11re_checkpoints.py --run-dir runs/v11re_balanced_probe --out-dir runs/v11re_balanced_probe_watch --max-checkpoint 1000 --checkpoint-step 500 --experiment rppo_medium_lstm128x2_observable12_target8_macro_library_v11re_stage2 --state-mode observable12_target8_macro_library_v11re_stage2'
```

Observed short-run outcome:

- `checkpoint_ep_01000` quick re-eval:
  - normal 20: success `1.00`, avg steps `21.0`
  - pressure 20: success `0.95`, avg steps `39.35`, avg collisions `16.5`
- interpretation:
  - pressure side regressed sharply versus `方案H`
  - no evidence yet that it fixes the diagonal degradation enough to justify the loss
  - do not promote this balanced follow-up; keep `方案H` as the stronger `v11re` branch

## v11re Diagonal-Focused Follow-Up

如果继续沿 `方案H` 往前，优先做更窄的 wrapper 级 follow-up，不碰 env reward，只压 recovery 探索强度和触发门槛：

- env overrides: `configs/v11/方案J_对角定向抑碰撞.json`
- intent:
  - 保留 `方案H` 的 pressure 优势
  - 提高 selective recovery 门槛
  - 降低 stuck macro explore prob
  - 把宏动作分布再往短/保守 escape 偏一点

Entrypoints:

```bash
rtk bash -lc 'source ~/miniconda3/etc/profile.d/conda.sh && conda activate ML && python train_target8_v11re_reward_exit_curriculum.py --output-dir runs/v11re_diagonal_tune_probe --resume-model runs/目标8宏动作库v11b卡住探索短训_v1002/rppo_medium_lstm128x2_observable12_target8_macro_library_v11_stage2/models/checkpoint_ep_01000.zip --stage-episodes 1000 --eval-episodes 8 --video-episodes 0 --checkpoint-interval 500 --progress-interval 250 --device cpu --torch-threads 1 --env-overrides-file configs/v11/方案J_对角定向抑碰撞.json'
```

```bash
rtk bash -lc 'source ~/miniconda3/etc/profile.d/conda.sh && conda activate ML && python watch_v11re_checkpoints.py --run-dir runs/v11re_diagonal_tune_probe --out-dir runs/v11re_diagonal_tune_probe_watch --max-checkpoint 1000 --checkpoint-step 500 --experiment rppo_medium_lstm128x2_observable12_target8_macro_library_v11re_stage2 --state-mode observable12_target8_macro_library_v11re_stage2'
```

Observed formal outcome:

- pressure 100: success `1.00`, avg steps `29.91`, avg collisions `10.63`
- diagonal-dense 100: success `0.91`, avg steps `96.56`, avg collisions `41.32`
- interpretation:
  - recovery became too conservative
  - pressure side lost some of `方案H`'s advantage
  - diagonal side regressed badly
  - do not promote `方案J`; keep `方案H` as the strongest `v11re` branch

## v11re Diagonal Long-Tail Curriculum

前两条 `v11re` follow-up 已经说明，只在 reward / wrapper 上微调，不足以修掉对角 dense 的长尾极端卡死。下一条直接保留 `方案H` 的 reward / exit 设定，但把训练分布往“更长对角距离 + 更多 concave 山体 + 更高反向朝向/困住开局”推：

- env overrides: `configs/v11/方案K_对角长尾课程.json`
- intent:
  - 保留 `方案H` 的 pressure 强项
  - 增加 diagonal dense 失败样本更接近的训练覆盖
  - 优先压缩少数极端 seed 的超长 stuck run

Fast screen:

```bash
rtk bash -lc 'source ~/miniconda3/etc/profile.d/conda.sh && conda activate ML && python train_target8_v11re_reward_exit_curriculum.py --quick --output-dir runs/v11re_diag_curriculum_quick --resume-model runs/目标8宏动作库v11b卡住探索短训_v1002/rppo_medium_lstm128x2_observable12_target8_macro_library_v11_stage2/models/checkpoint_ep_01000.zip --video-episodes 0 --device cpu --torch-threads 1 --env-overrides-file configs/v11/方案K_对角长尾课程.json'
```

```bash
rtk bash -lc 'source ~/miniconda3/etc/profile.d/conda.sh && conda activate ML && python watch_v11re_checkpoints.py --quick --run-dir runs/v11re_diag_curriculum_quick --out-dir runs/v11re_diag_curriculum_quick_watch'
```

Observed outcome:

- quick screen:
  - `checkpoint_ep_00250`: normal 20 `0.95 / 38.3`, pressure 20 `0.95 / 45.7`
  - `checkpoint_ep_00500`: normal 20 `1.00 / 28.55`, pressure 20 `1.00 / 32.55`
- formal `checkpoint_ep_00500`:
  - pressure 100: success `0.99`, avg steps `33.4`, avg collisions `10.94`
  - diagonal-dense 100: success `0.95`, avg steps `85.59`, avg collisions `27.16`
- interpretation:
  - pressure side is worse than `方案H`
  - diagonal side did not improve at all versus `方案H`; the same extreme long-tail seeds still dominate
  - do not continue this diagonal-curriculum branch as-is
  - next useful move should target the persistent bad seeds more directly instead of only shifting the broad training distribution

## v11re Bad-Seed Curriculum

既然 broad curriculum 不能修掉 `6300058 / 6300069 / 6300025 / 6300031 / 6300057` 这批 persistent bad seeds，下一条直接把训练分布切成“固定对角 dense 坏种子池循环重放”：

- trainer: `train_target8_v11re_bad_seed_curriculum.py`
- env overrides: `configs/v11/方案L_坏种子定向课程.json`
- default bad seed pool:
  - `6300058,6300069,6300025,6300031,6300057,6300067,6300089,6300006`
- intent:
  - 不再泛化采样
  - 直接让 `v11re` 对最坏长尾局反复学习
  - 快速验证“定向复训”是否能压掉极端 stuck-run / 爆碰撞 seed

Fast screen:

```bash
rtk bash -lc 'source ~/miniconda3/etc/profile.d/conda.sh && conda activate ML && python train_target8_v11re_bad_seed_curriculum.py --quick --output-dir runs/v11re_bad_seed_quick --resume-model runs/目标8宏动作库v11b卡住探索短训_v1002/rppo_medium_lstm128x2_observable12_target8_macro_library_v11_stage2/models/checkpoint_ep_01000.zip --video-episodes 0 --device cpu --torch-threads 1 --env-overrides-file configs/v11/方案L_坏种子定向课程.json'
```

```bash
rtk bash -lc 'source ~/miniconda3/etc/profile.d/conda.sh && conda activate ML && python watch_v11re_checkpoints.py --quick --run-dir runs/v11re_bad_seed_quick --out-dir runs/v11re_bad_seed_quick_watch'
```

Micro specialization:

```bash
rtk bash -lc 'source ~/miniconda3/etc/profile.d/conda.sh && conda activate ML && python train_target8_v11re_bad_seed_curriculum.py --micro --output-dir runs/v11re_bad_seed_micro --resume-model runs/目标8宏动作库v11b卡住探索短训_v1002/rppo_medium_lstm128x2_observable12_target8_macro_library_v11_stage2/models/checkpoint_ep_01000.zip --video-episodes 0 --device cpu --torch-threads 1 --env-overrides-file configs/v11/方案L_坏种子定向课程.json'
```

```bash
rtk bash -lc 'source ~/miniconda3/etc/profile.d/conda.sh && conda activate ML && python watch_v11re_checkpoints.py --micro --run-dir runs/v11re_bad_seed_micro --out-dir runs/v11re_bad_seed_micro_watch'
```

Observed micro outcome:

- `checkpoint_ep_00125` bad-seed target eval:
  - success `0.75` on the 8-seed pool
  - failed seeds: `6300057,6300089`
- `checkpoint_ep_00125` quick re-eval:
  - normal 20: success `0.95`, avg steps `31.25`
  - pressure 20: success `1.00`, avg steps `33.6`, avg collisions `7.15`
- formal `checkpoint_ep_00125`:
  - pressure 100: success `0.99`, avg steps `34.96`, avg collisions `7.76`
  - diagonal-dense 100: success `0.93`, avg steps `91.58`, avg collisions `11.79`
- interpretation:
  - this is the strongest targeted bad-seed improvement seen so far
  - it cuts diagonal collisions sharply, but loses too much diagonal success and speed
  - the branch changes which seeds fail instead of cleanly dominating the baseline
  - do not promote this as a general replacement for the default line or for `方案H`
  - the useful remaining signal is that a very short bad-seed specialization stage can reshape failure modes, but it still needs a better merge-back strategy if we want global improvement

## v11re Bad-Seed Merge-Back

既然 `micro-125` 已经证明“极短坏 seed 特化”有效，但单独停在那里会伤 diagonal success / steps，下一条直接做两段式：

- phase1: bad-seed micro specialization
- phase2: mixed merge-back
  - a configurable slice of envs keeps replaying bad seeds
  - the rest go back to normal `v11re`
- trainer: `train_target8_v11re_bad_seed_mergeback_curriculum.py`

Minimal run:

```bash
rtk bash -lc 'source ~/miniconda3/etc/profile.d/conda.sh && conda activate ML && python train_target8_v11re_bad_seed_mergeback_curriculum.py --micro --phase2-bad-seed-ratio 0.25 --output-dir runs/v11re_bad_seed_mergeback_micro --resume-model runs/目标8宏动作库v11b卡住探索短训_v1002/rppo_medium_lstm128x2_observable12_target8_macro_library_v11_stage2/models/checkpoint_ep_01000.zip --bad-seed-env-overrides-file configs/v11/方案L_坏种子定向课程.json --mergeback-env-overrides-file configs/v11/方案H_脱困退出与收角强化.json --video-episodes 0 --device cpu --torch-threads 1'
```

Observed outcome:

- `phase1 checkpoint_ep_00125` keeps the same bad-seed gain as standalone `micro-125`
- but the final merged model does not materially move away from the standalone micro result:
  - pressure 100: success `0.99`, avg steps `34.96`, avg collisions `7.76`
  - diagonal-dense 100: success `0.93`, avg steps `91.58`, avg collisions `11.79`
- interpretation:
  - this minimal merge-back is too weak to recover the lost diagonal success / speed
  - the run behaves like “micro specialization preserved” rather than “specialize then merge back”
  - do not promote this minimal `125 + 125` merge-back recipe as a usable final flow

Observed outcome:

- bad-seed target eval:
  - `checkpoint_ep_00250`: success `0.50` on the 8-seed pool, failed seeds `6300069,6300031,6300067,6300006`
  - `checkpoint_ep_00500`: success `0.375` on the 8-seed pool, failed seeds `6300058,6300069,6300025,6300031,6300057`
- normal / pressure quick re-eval:
  - `checkpoint_ep_00250`: normal 20 `1.00 / 21.2`, pressure 20 `0.95 / 53.2`
  - `checkpoint_ep_00500`: normal 20 `1.00 / 20.5`, pressure 20 `1.00 / 29.55`
- interpretation:
  - direct bad-seed replay does produce real early gains on the target hard cases
  - but the gain is unstable: by `00500` the hard-seed improvement collapses
  - the better bad-seed checkpoint (`00250`) hurts pressure badly, so this branch is not promotable as a general replacement
  - the useful signal is narrower: bad-seed replay plus early stopping may be viable as a short specialization stage, but not as a longer standalone continuation

## v11re Hybrid Anchor Probe

`merge-back` 暴露出的核心问题不是“坏 seed 特化无效”，而是它缺少一个显式的 base-behavior 约束。下一条 probe 直接把这个约束写进训练流：

- online PPO:
  - 保留 `方案H` 的 `v11re` wrapper
  - 继续让 `25%` env 重放坏 seed
- anchor BC:
  - 每一小轮 PPO 之后，插入几轮行为克隆
  - 用 `normal + pressure` 成功轨迹做 `base` anchor
  - 用对角坏 seed 轨迹做 `specialist` anchor
  - `specialist` teacher 默认不是 `方案H` 本身，而是 `v11re_bad_seed_micro checkpoint 00125`
- multi-teacher note:
  - `方案H 01000` 和 `bad_seed_micro 00125` 在坏 seed 上是互补的
  - 当前可用的拼接脚本是 `merge_v11_teacher_datasets.py`
  - 已验证的坏 seed 选优分配是 `base:3 / micro125:5`
- trainer: `train_target8_v11re_hybrid_anchor_finetune.py`

The trainer now keeps cumulative checkpoint numbering across rounds, so watcher can follow `00032 / 00064 / 00096 / 00128` instead of later rounds overwriting earlier checkpoints.

Remote micro probe on `v1002`:

```bash
rtk bash tools/启动_v11re_hybrid_anchor短训_v1002.sh
```

That script:

- resumes from `runs/v11re_reward_exit_probe/.../checkpoint_ep_01000.zip`
- exports three anchor datasets if they are missing:
  - normal sequential seeds: `1810000 + 24`
  - pressure auto-selected seeds from scan start `10651000`
  - default diagonal bad-seed pool, exported from `v11re_bad_seed_micro/.../checkpoint_ep_00125.zip`
- launches `runs/v11re_hybrid_anchor_micro`

Watcher:

```bash
rtk bash tools/启动_v11re_hybrid_anchor评估_v1002.sh
```

Watcher defaults:

- out dir: `runs/v11re_hybrid_anchor_micro_watch`
- checkpoint step: `32`
- max checkpoint: `128`

- no map input
- no obstacle geometry in observation
- learned macro-library recovery on top of blind state
- `distance_to_target <= 8px` is still an external/environment-owned stop condition

Historical comparison kept for reference:

- `AE500` remains the old `v10` reference baseline
- old `v11 stage2/stage3` remain diagnosis baselines for the obstacle-stall failure mode

## Continuous 9D fixed-recovery branch

Current rollback baseline:

- experiment: `rppo_small_lstm128x1_observable9_target8_continuous_fixed_recovery_v1`
- state mode: `observable9_target8_continuous_fixed_recovery_v1`
- action mode: `continuous_trigger_fixed_recovery`
- action space: continuous `Box(4,)`
- stop rule remains external/environment-owned: `distance_to_target <= 8px`

9D observation:

| # | State | 中文解释 |
|---:|---|---|
| 1 | `dx_norm` | 目标相对旧坐标的 x 偏移 |
| 2 | `dy_norm` | 目标相对旧坐标的 y 偏移 |
| 3 | `sin_heading` | 当前人物方向正弦 |
| 4 | `cos_heading` | 当前人物方向余弦 |
| 5 | `sin_angle_error` | 目标方向减当前人物方向的角差正弦 |
| 6 | `cos_angle_error` | 目标方向减当前人物方向的角差余弦 |
| 7 | `coord_age_norm` | 当前坐标观测年龄，模拟坐标延迟 |
| 8 | `distance_progress_norm` | 最近一次有效坐标刷新后的距离进展 |
| 9 | `no_progress_time_norm` | 连续无进展时间 |

Continuous action semantics:

| 维度 | 含义 | 说明 |
|---:|---|---|
| 1 | `angle_offset` | 相对目标方向的连续偏角，范围 `[-60deg, +60deg]` |
| 2 | `speed` | 映射到二值速度 `50 / 100` |
| 3 | `jump` | 对该分支固定关闭，由 wrapper 写死为 `-1` |
| 4 | `recovery_trigger` | 仅决定是否触发固定脱困模板 |

Fixed recovery template:

- 第 1 步自动触发一次 `jump`
- 朝当前锚定朝向后退 `2.0s`
- 然后随机左移或右移 `2.0s`
- 左/右在每次触发时随机一次，并在整个模板期间锁定
- 碰撞去掉沿边滑移，改成纯阻挡；`jump` 仍可脱出树/山

`v1002` 启动命令：

```bash
rtk ssh v1002 'cd /home/diana/fishing/blind_nav_rl && source ~/miniconda3/etc/profile.d/conda.sh && conda activate ML && bash tools/启动_连续9维固定脱困_v1002.sh'
```

`253` 启动命令：

```bash
rtk ssh 253 'cd /home/weiaokang/fishing/blind_nav_rl && source /home/weiaokang/ML/bin/activate && bash tools/启动_连续9维固定脱困_253.sh'
```

## Continuous fixed-recovery v2

`v1` 的问题已经验证清楚：模型一次都没有触发脱困模板。`v2` 不是继续堆轮数，而是直接改触发学习信号。

- experiment: `rppo_small_lstm128x1_observable10_target8_continuous_fixed_recovery_v2`
- state mode: `observable10_target8_continuous_fixed_recovery_v2`
- 10D state: `v1` 的 9 维 + `recent_collision_norm`
- fixed recovery trigger threshold: `0.1`
- fixed recovery reward shaping:
  - 卡住/刚碰撞时，不触发会持续扣分
  - 在卡住上下文里触发 recovery 会立即加分
  - recovery 触发后重新恢复 progress/脱离碰撞会再给一次明显奖励
- `v1002` 默认并行调高到：
  - `N_ENVS=256`
  - `THREADS=2`

目标不是学更多直线局，而是先把“卡住时按 trigger”这件事学出来。

## Legacy 8D target-circle task

Observation from `Observable8TargetWrapper`:

| # | State | 中文解释 | 来源 |
|---:|---|---|---|
| 1 | `dx_norm` | 目标相对上次坐标观测点的 x 偏移，按地图尺度归一化 | 自身坐标 + 目标坐标 |
| 2 | `dy_norm` | 目标相对上次坐标观测点的 y 偏移，按地图尺度归一化 | 自身坐标 + 目标坐标 |
| 3 | `sin_heading` | 当前人物/指针方向的正弦 | 指针识别 |
| 4 | `cos_heading` | 当前人物/指针方向的余弦 | 指针识别 |
| 5 | `sin_angle_error` | 目标方向减当前人物方向的角度差正弦 | 坐标 + 指针 |
| 6 | `cos_angle_error` | 目标方向减当前人物方向的角度差余弦 | 坐标 + 指针 |
| 7 | `coord_age_norm` | 上次坐标刷新到现在经过多久，模拟真实坐标延迟 | 坐标时间戳 |
| 8 | `distance_progress_norm` | 最近一次观测距离目标减少了多少，正数表示靠近 | 前后两次坐标 |

The model does not receive `desired_stop_distance`. The stop rule is:

```text
if distance_to_target <= 8px:
    success / release joystick
```

Current experiment:

- `rppo_lstm256x2_observable8_target8_line_delta90`
- state mode `observable8_target8_line_delta90`
- action mode `stick_angle_delta`
- action angle means changing the persisted joystick angle by `[-90deg, +90deg]`
- speed is binary: action speed `< 0.5 -> 50`, `>= 0.5 -> 100`
- no speed `0`; stopping is handled outside the policy
- jump is still controlled by RL
- target radius `8px`
- smaller generated map: `world_size=1200`, start-target distance `280px` to `700px`
- reward profile `target_line`

## Target-circle v2 sweep

The current anti-snake experiment is `observable8_target8_v2_offset60`.

- stop rule stays hard: `distance_to_target <= 8px`
- action mode: `target_relative_offset`
- action angle means offset from current target direction, limited to `[-60deg, +60deg]`
- speed remains binary `50 / 100`, with no speed `0`
- jump remains controlled by RL
- state stays the same 8D target-circle state
- reward profile: `target_line_v2`

The v2 reward adds stronger path-quality pressure:

- `progress / movement` efficiency reward
- stronger angle-error penalty
- stronger lateral-distance penalty
- heading-change / curvature penalty
- action magnitude penalty
- left-right action flip penalty

## v7 Two-Phase Heading-Relative Recovery

`v6` proved that simply increasing reward pressure was not enough to make the policy use recovery consistently. The next additive branch is `v7`, which keeps the same blind 12D observation and discrete action size, but changes the recovery primitive itself.

- experiment: `rppo_medium_lstm128x2_observable12_target8_discrete_macro_v7`
- state mode: `observable12_target8_discrete_macro_v7`
- action space: `MultiDiscrete([5, 5, 2])`
- action mode: `heading_relative_two_phase_macro_recovery`
- reward profile: `target_line_macro_recovery_v7`
- stop rule remains external: `distance_to_target <= 8px`

What changes versus `v6`:

- recovery is anchored to the current heading, not the target line
- recovery macro becomes two-phase:
  `back for the first half -> side exit for the second half`
- steering bins are denser near zero:
  `[-60deg, -20deg, 0deg, 20deg, 60deg]`
- concave pressure is stronger:
  `mountain_count=32`, `concave=0.98`, `deep_concave=0.72`

What stays unchanged:

- no map input
- no obstacle geometry
- no ray features
- no collision normal
- stale coordinates + live heading only

The intention is to make the trap-escape behavior easier for PPO to discover without adding obstacle hints to the observation.

Current `253` long run:

```bash
rtk ssh 253 'cd /home/weiaokang/fishing/blind_nav_rl && tail -f runs/目标8离散宏动作v7_100k/训练.log'
rtk ssh 253 'cd /home/weiaokang/fishing/blind_nav_rl && tail -f runs/目标8离散宏动作v7_checkpoint评估/watcher.log'
rtk ssh 253 'cd /home/weiaokang/fishing/blind_nav_rl && tail -f runs/目标8离散宏动作v7_最终归档/finalizer.log'
rtk python3 distributed_v7_progress.py --hosts 253 --include-main
```

Run directories on `253`:

- training: `/home/weiaokang/fishing/blind_nav_rl/runs/目标8离散宏动作v7_100k`
- checkpoint eval: `/home/weiaokang/fishing/blind_nav_rl/runs/目标8离散宏动作v7_checkpoint评估`
- final archive: `/home/weiaokang/fishing/blind_nav_rl/runs/目标8离散宏动作v7_最终归档`

Current parallel `v1002` run:

- training: `/home/diana/fishing/blind_nav_rl/runs/目标8离散宏动作v7_v1002_100k`
- checkpoint eval: `/home/diana/fishing/blind_nav_rl/runs/目标8离散宏动作v7_v1002_checkpoint评估`
- final archive: `/home/diana/fishing/blind_nav_rl/runs/目标8离散宏动作v7_v1002_最终归档`

## v8 Trapped-Start Distribution

`v7` plateaued through `10k` checkpoint. The next branch, `v8`, keeps the same blind 12D observation and the same heading-relative two-phase recovery action, but changes the start-state distribution itself.

- experiment: `rppo_medium_lstm128x2_observable12_target8_discrete_macro_v8`
- state mode: `observable12_target8_discrete_macro_v8`
- action mode: `heading_relative_two_phase_macro_recovery`
- reward profile: still `target_line_macro_recovery_v7`
- new environment parameter: `trapped_start_probability`

The key idea is simple:

- some episodes now start near or inside a concave mountain pocket
- the target is placed so that the direct line still crosses the mountain
- the model sees no extra obstacle features; only the training distribution changes

Current local validation:

- local audit passed: `runs/目标8离散宏动作v8_本地审计/audit.csv`
- local smoke passed: `runs/目标8离散宏动作v8_本地smoke2/`
- in the local audit sample, `3 / 8` seeds were already near a mountain trap at reset

Current `v1002` long run:

- training on `253`: `/home/weiaokang/fishing/blind_nav_rl/runs/目标8离散宏动作v8_100k`
- training: `/home/diana/fishing/blind_nav_rl/runs/目标8离散宏动作v8_v1002_100k`
- checkpoint eval: `/home/diana/fishing/blind_nav_rl/runs/目标8离散宏动作v8_v1002_checkpoint评估`
- final archive: `/home/diana/fishing/blind_nav_rl/runs/目标8离散宏动作v8_v1002_最终归档`

Progress:

```bash
rtk python3 distributed_v8_progress.py
```

## v9 Trigger-Recovery Curriculum

`v9` 是在 `v8` 基础上的新主线，目标不是继续堆轮数，而是降低盲策略的探索难度。

改动原则：

- 仍然不输入地图
- 仍然不输入障碍几何
- 仍然不输入射线
- 仍然不输入碰撞法线
- 保留盲状态，只改动作接口、训练分布和课程顺序

### v9 状态

仍然使用 12 维盲状态，和 `v7/v8` 同量级：

| # | State | 中文解释 |
|---:|---|---|
| 1 | `dx_norm` | 目标相对旧坐标的 x 偏移 |
| 2 | `dy_norm` | 目标相对旧坐标的 y 偏移 |
| 3 | `sin_heading` | 当前人物方向正弦 |
| 4 | `cos_heading` | 当前人物方向余弦 |
| 5 | `sin_angle_error` | 目标方向减当前人物方向的角差正弦 |
| 6 | `cos_angle_error` | 目标方向减当前人物方向的角差余弦 |
| 7 | `coord_age_norm` | 坐标年龄，模拟坐标延迟 |
| 8 | `distance_progress_norm` | 最近一次观测距离进展 |
| 9 | `no_progress_time_norm` | 无进展时间 |
| 10 | `distance_norm` | 当前距离归一化 |
| 11 | `last_move_mode` | 上一步是否处于 recovery 宏动作 |
| 12 | `macro_phase_norm` | 当前 recovery 宏动作剩余阶段 |

### v9 动作

`v9` 去掉了策略对 `jump` 的控制。动作空间改为 `MultiDiscrete([5, 2, 2])`：

| 维度 | 含义 | 说明 |
|---:|---|---|
| 1 | `direction_bin` | 相对目标方向的偏角 bin，`[-60, -20, 0, 20, 60]` |
| 2 | `speed_bin` | 二值速度，`50 / 100` |
| 3 | `recovery_trigger_bin` | 是否触发 6 步宏脱困；触发后由环境按方向 bin 决定后退左/后退/后退右 |

也就是说：

- 模型只负责“是否触发脱困”
- `jump` 固定关闭
- 脱困由环境保持若干步，减少逐帧探索难度

### v9 两阶段课程

为了让同一个网络先学直线，再学 trapped-start 脱困，`v9` 分成两个环境阶段，但状态维度和动作空间保持一致，可以直接续训：

1. `observable12_target8_discrete_trigger_v9_stage1`
   - 较少山体
   - 中等凹陷概率
   - 无 trapped start
   - 加入部分反向初始朝向
   - 目标：先学稳定直线追踪和方向修正
2. `observable12_target8_discrete_trigger_v9_stage2`
   - 高凹陷山体
   - `trapped_start_probability`
   - 更强坐标延迟与噪声
   - 更高反向初始朝向概率
   - 目标：在保留直线能力的前提下学脱困

启动：

```bash
rtk bash launch_target8_v9_curriculum.sh
```

核心脚本：

- [train_target8_v9_curriculum.py](/home/diana/fishing/blind_nav_rl/train_target8_v9_curriculum.py)
- [launch_target8_v9_curriculum.sh](/home/diana/fishing/blind_nav_rl/launch_target8_v9_curriculum.sh)

压力评估新增：

- `eval_target8_pressure.py --action-mode v9`

## v10 显式脱困课程

`v10` 是在 `v9` 的基础上继续收敛的一条加法分支，重点不是再堆轮数，而是让策略在“触发前”就拿到更直接的卡住信号，并把 recovery 动作语义拆清楚。

改动原则：

- 仍然不输入地图
- 仍然不输入障碍几何
- 仍然不输入射线
- 仍然不输入碰撞法线
- 仍然保持盲策略，只优化状态语义、动作语义和课程分布

### v10 状态

仍然是 12 维盲状态，但把 `v9` 中“触发后的动作记忆”换成“触发前可用的卡住信号”：

| # | State | 中文解释 |
|---:|---|---|
| 1 | `dx_norm` | 目标相对旧坐标的 x 偏移 |
| 2 | `dy_norm` | 目标相对旧坐标的 y 偏移 |
| 3 | `sin_heading` | 当前人物方向正弦 |
| 4 | `cos_heading` | 当前人物方向余弦 |
| 5 | `sin_angle_error` | 目标方向减当前人物方向的角差正弦 |
| 6 | `cos_angle_error` | 目标方向减当前人物方向的角差余弦 |
| 7 | `coord_age_norm` | 坐标年龄，模拟坐标延迟 |
| 8 | `distance_progress_norm` | 最近一次观测距离进展 |
| 9 | `no_progress_time_norm` | 无进展时间 |
| 10 | `distance_norm` | 当前距离归一化 |
| 11 | `recent_collision_norm` | 最近是否刚撞过，越接近 1 表示越近期发生碰撞 |
| 12 | `stuck_time_norm` | 当前连续卡住时间 |

### v10 动作

`v10` 的动作空间是 `MultiDiscrete([5, 2, 4])`：

| 维度 | 含义 | 说明 |
|---:|---|---|
| 1 | `direction_bin` | 相对目标方向的偏角 bin，`[-60, -20, 0, 20, 60]` |
| 2 | `speed_bin` | 二值速度，`50 / 100` |
| 3 | `recovery_mode_bin` | `none / back-left / back / back-right` |

和 `v9` 的区别是：

- 不再让方向 bin 兼任 recovery 侧向选择
- 策略直接输出 recovery 模式
- recovery 仍由环境保持 6 步，降低逐帧探索难度

### v10 四阶段课程

1. `observable12_target8_discrete_recovery_v10_stage1`
   - 低密度障碍
   - 轻度反向初始朝向
   - 目标：先学直线追目标
2. `observable12_target8_discrete_recovery_v10_stage2a`
   - softer trapped-start 过渡阶段
   - recovery primitive 改成相对当前 heading 的两段式宏动作
   - 目标：先学“后退再侧出”，避免 `stage1 -> hard trap` 直接塌缩
3. `observable12_target8_discrete_recovery_v10_stage2`
   - 强 trapped-start
   - 高密度强内凹山体
   - 目标距离更近，让“先后退再侧出”成为必需动作
4. `observable12_target8_discrete_recovery_v10_stage3`
   - 普通分布和陷阱分布混合
   - 目标：保留直线能力，同时保留脱困能力

从这版开始，`stage2/3` 不再使用 `target_relative_macro_recovery`，而是改成已有的 `heading_relative_two_phase_macro_recovery`：

- 先按当前朝向后退
- 再按当前朝向左/右侧移
- 比“相对目标方向后退”更符合内凹山体脱困几何

另外做了两个执行层约束：

- `stage2a/2/3` 的 recovery 只在 `no_progress_time` 或近期碰撞达到阈值时才允许生效
- 一旦进入 recovery，速度固定为高速，避免策略在 recovery 内还要同时探索速度分支

2026-05-20 的这一版又补了三处约束，用来解决 `stage3` 把普通追目标也推成“过度脱困”的问题：

- `stage3` 分布降难：
  - `mountain_count: 38 -> 30`
  - `deep_concave: 0.88 -> 0.60`
  - `trapped_start_probability: 0.75 -> 0.50`
  - `opposite_heading_probability: 0.55 -> 0.42`
- selective recovery gate 收紧：
  - `no_progress_time >= 1.10`
  - 或 `recent_collision_norm >= 0.90`
- `heading_relative_two_phase_macro_recovery` 增加提前退出：
  - 只有在“这次 recovery 确实是从 stuck 状态启动”的情况下才允许提前退出
  - 一旦碰撞间隔恢复、`no_progress_time` 回落，或者 recovery 已经带来明确正进展，就提前结束剩余宏动作

目标很明确：

- `stage2` 学出的 recovery 不要被洗掉
- 但普通 episode 也不要被 `stage3` 推成频繁后退/侧退
- 让 `stage3` 重新回到“保留直线能力，同时保留脱困能力”的角色

当前这条优化版 `stage3` 的经验也已经明确了：

- `stage3 01000` 有改善
- `stage3 02000` 已经再次退化成“几乎不用 recovery，靠高碰撞贴障滑移”

所以这版的执行策略不是“把优化版 `stage3` 跑很久”，而是：

- `stage3` 只做短训微调
- 当前默认把 `1000 ~ 1500` 当成有效窗口
- 超过这个范围要重新评估，不能默认继续训练更久

对应的正式入口已经补齐：

- 远端短训启动：
  [tools/启动_v10阶段3短训选优_v1002.sh](/home/diana/fishing/blind_nav_rl/tools/启动_v10阶段3短训选优_v1002.sh)
- 远端最终归档：
  [tools/归档_v10阶段3短训选优_v1002.sh](/home/diana/fishing/blind_nav_rl/tools/归档_v10阶段3短训选优_v1002.sh)
- 三方案对照启动：
  [tools/启动_v10阶段3三方案对照_v1002.sh](/home/diana/fishing/blind_nav_rl/tools/启动_v10阶段3三方案对照_v1002.sh)
- 三方案对照归档：
  [tools/归档_v10阶段3三方案对照_v1002.sh](/home/diana/fishing/blind_nav_rl/tools/归档_v10阶段3三方案对照_v1002.sh)
- 方案 A 保留线启动：
  [tools/启动_v10阶段3方案A保留线_v1002.sh](/home/diana/fishing/blind_nav_rl/tools/启动_v10阶段3方案A保留线_v1002.sh)
- 方案 A 保留线归档：
  [tools/归档_v10阶段3方案A保留线_v1002.sh](/home/diana/fishing/blind_nav_rl/tools/归档_v10阶段3方案A保留线_v1002.sh)

这条线的设计点是：

- resume from `stage2 checkpoint 03000`
- `stage3` 总轮数默认 `1500`
- checkpoint 间隔 `500`
- 直接在 `500 / 1000 / 1500` 三个点里选优

在短训选优之后，当前又加了一层更小的 `stage3` 微调对照：

- 方案 A：收紧 wrapper gate
  - `configs/v10_stage3/方案A_收紧gate.json`
- 方案 C：降低 `trapped_start_probability / opposite_heading_probability`
  - `configs/v10_stage3/方案C_降低trapped与反向朝向.json`
- 方案 B：降低深凹比例和陷阱密度
  - `configs/v10_stage3/方案B_降低深凹与陷阱密度.json`

批量汇总脚本：

- [summarize_v10_variant_batch.py](/home/diana/fishing/blind_nav_rl/summarize_v10_variant_batch.py)

当前 `pressure_priority` 评分下的结果已经明确：

| 排名 | 方案 | 选中 checkpoint | 分数 | 普通成功率 | 普通步数 | 压力成功率 | 压力步数 | 压力脱困成功率 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | A 收紧 gate | 500 | 347.227 | 1.00 | 36.5 | 0.95 | 46.6 | 0.90 |
| 2 | C 降 trapped | 500 | 344.378 | 1.00 | 45.2 | 0.95 | 51.3 | 0.90 |
| 3 | B 降深凹 | 500 | 334.423 | 1.00 | 38.2 | 0.90 | 62.9 | 0.85 |

所以当前 `stage3` 微调的默认保留方案是：

1. `方案A收紧gate`
2. 保留 `方案C降trapped` 作为次优回退

为了避免后续每次都从三方案批处理脚本里回推，现在已经补了 `A-only` 正式入口，默认就是：

- `stage2 checkpoint 03000`
- `stage3_episodes = 1500`
- `checkpoint_interval = 500`
- `n_envs = 96`
- `env_overrides = configs/v10_stage3/方案A_收紧gate.json`

当前这条保留线已经完成一轮正式 run：

- 训练目录：
  `/home/diana/fishing/blind_nav_rl/runs/目标8显式脱困v10阶段3方案A保留线_v1002`
- 最终归档目录：
  `/home/diana/fishing/blind_nav_rl/runs/目标8显式脱困v10阶段3方案A保留线_最终归档_v1002`

最终选择仍然稳定在：

- `checkpoint = 00500`
- `score = 347.227`
- 普通：`1.00 / 36.45步`
- 压力：`0.95 / 46.65步 / 脱困成功率 0.90`

### v10 奖励

`reward_profile = target_line_macro_recovery_v10`

相对于 `v7/v9`，额外强调：

- 卡住后不做 recovery 的惩罚更强
- recovery 期间产生明确后退位移会加分
- recovery 后恢复正向进展会给更高奖励
- recovery 来回切模式会额外扣分
- 非卡住状态乱用 recovery 会被压制

当前版本还把 gate 和 reward 的节奏拉近了：

- recovery gate 起点约 `0.8s`
- `stage2` 的 stuck pressure 也是在这一段附近开始明显加压

目的是减少“reward 在叫你脱困，但 wrapper 还不让 recovery 生效”这种冲突。

### v10 重点评估指标

除了原来的：

- `pressure_recovery_rate`
- `pressure_avg_collisions`
- `pressure_success`

现在额外记录：

- `pressure_avg_recovery_steps_to_trigger`：平均到第几步才第一次进入 recovery
- `pressure_recovery_escape_rate`：进入 recovery 后，是否在短时间内把 `stuck/no_progress` 压回低位

### v10 启动

```bash
rtk bash launch_target8_v10_curriculum.sh
```

核心脚本：

- [train_target8_v10_curriculum.py](/home/diana/fishing/blind_nav_rl/train_target8_v10_curriculum.py)
- [launch_target8_v10_curriculum.sh](/home/diana/fishing/blind_nav_rl/launch_target8_v10_curriculum.sh)

压力评估：

- `eval_target8_pressure.py --action-mode v10`

## Tiny Recovery Angle Experiment

This is the planned fix for concave mountain traps while keeping the same deployable 8D observation. It does not expose obstacle geometry, collision flags, stuck flags, rays, maps, or mountain vertices.

Experiment:

- `rppo_tiny_lstm64x1_observable8_target8_recovery_angle`
- state mode `observable8_target8_recovery_angle`
- observation: same 8D `Observable8TargetWrapper`
- policy: `RecurrentPPO + MlpLstmPolicy`
- LSTM: `64x1`
- MLP heads: `[64]`
- normal movement: `target_angle + offset[-60deg, +60deg]`
- speed: binary `50 / 100`
- jump: still controlled by RL
- stop: environment hard success when `distance_to_target <= 8px`

Action space for this experiment is 4D:

| # | Action | Meaning |
|---:|---|---|
| 1 | `angle_offset_norm` | normal-mode offset around target direction |
| 2 | `speed_norm` | binary speed selector, `<0.5 -> 50`, `>=0.5 -> 100` |
| 3 | `jump_norm` | jump/down trigger |
| 4 | `recovery_mode_norm` | recovery selector |

`recovery_mode_norm` is interpreted as:

| Range | Movement direction |
|---|---|
| `< -0.5` | `target_angle + 135deg` back-left |
| `[-0.5, 0.0)` | `target_angle + 180deg` back |
| `[0.0, 0.5)` | normal target-relative movement |
| `>= 0.5` | `target_angle - 135deg` back-right |

Environment changes for this experiment:

- mountain count `10`
- tree count `45`
- fixed tree radius `18px`
- concave mountain probability `0.78`
- deep concave / U-like notch probability `0.25`
- mountain overlap prevention remains enabled
- world size remains `1200`
- target radius remains `8px`
- coordinate refresh delay remains randomized in `[0.35s, 0.75s]`

Reward changes versus v2:

- stronger far-field angle-error penalty
- stronger lateral movement penalty
- stronger timeout angle penalty
- light recovery switch penalty
- light penalty for using recovery while already making progress
- bonus when recovery restores progress after stuck/no-progress behavior
- extra penalty for continuing normal movement during sustained no-progress

Run on `ssh 253`:

```bash
cd /home/weiaokang/fishing/blind_nav_rl
bash launch_target8_recovery_angle_tiny_bg.sh
```

Progress:

```bash
tail -f /home/weiaokang/screencap/file/blind_nav_target8_recovery_angle_tiny_100k/rppo_tiny_lstm64x1_observable8_target8_recovery_angle/train.log
ls -lh /home/weiaokang/screencap/file/blind_nav_target8_recovery_angle_tiny_100k/rppo_tiny_lstm64x1_observable8_target8_recovery_angle/models/
```

Audit checklist:

| Requirement | Evidence |
|---|---|
| State remains 8D | `Observable8TargetWrapper`, `observation_dim(...) == 8` |
| No obstacle info is observed | observation only uses stale position, target, heading, coord age, distance progress |
| Model can move backward | `target_relative_recovery` supports `target_angle + 180deg` |
| Model can side-exit concavity | recovery supports `target_angle +/- 135deg` |
| Normal straight motion remains possible | recovery mode `[0.0, 0.5)` uses target-relative offset, and offset `0` points at target |
| Stop remains external | `distance_to_target <= 8px` terminates the episode |
| More concave mountains are generated | recovery mode sets `concave_mountain_probability=0.78`, `deep_concave_mountain_probability=0.25` |
| Angle-error pressure is stronger | reward profile `target_line_recovery_angle` |
| Recovery misuse is measurable | `eval_target8_sweep.py` reports `recovery_action_rate` |
| Existing v2 results remain comparable | new mode/name are additive; old `observable8_target8_v2_offset60` is unchanged |

Training result on `ssh 253`:

| Variant | Episodes | Params | Normal success | Normal avg angle | Pressure success | Pressure avg angle | Recovery rate | Result |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `recovery_angle` 4D selector | 100k | 46,537 | 1.00 | 11.0deg | 0.85 | 7.2deg | 0.000 | Reaches often, but does not learn explicit back/side recovery |
| `recovery_pressure` 4D selector | 32k checkpoint | 46,537 | not completed | not completed | 0.75 | 5.4deg | 0.000 | More pressure made success worse and still did not use recovery |
| `full180_pressure` 3D full-angle | 50k | 46,471 | 0.95 | not measured by target8 sweep | 0.90 | 8.8deg | 0.000 | Wider angle helps pressure success, but still no explicit backward moves |

Artifacts:

- recovery 100k model: `/home/weiaokang/screencap/file/blind_nav_target8_recovery_angle_tiny_100k/rppo_tiny_lstm64x1_observable8_target8_recovery_angle/rppo_tiny_lstm64x1_observable8_target8_recovery_angle.zip`
- recovery 20-episode normal video: `/home/weiaokang/screencap/file/blind_nav_target8_recovery_angle_tiny_eval20/rppo_tiny_lstm64x1_observable8_target8_recovery_angle/target8_eval_concat.mp4`
- recovery pressure video: `/home/weiaokang/screencap/file/blind_nav_target8_recovery_angle_tiny_pressure20/pressure_eval_concat.mp4`
- full180 50k model: `/home/weiaokang/screencap/file/blind_nav_target8_full180_pressure_50k/rppo_tiny_lstm64x1_observable8_target8_full180_pressure/rppo_tiny_lstm64x1_observable8_target8_full180_pressure.zip`
- full180 pressure video: `/home/weiaokang/screencap/file/blind_nav_target8_full180_pressure_eval20/pressure_eval_concat.mp4`

Conclusion from the audit:

- The 8D observation constraint is preserved.
- More concave mountains are generated.
- Backward/side-exit action capability exists in the environment.
- Angle-error pressure was strengthened.
- The trained Tiny policy did not actually use explicit recovery actions in the tested seeds.

Do not treat this as solved by simply adding more episodes. The likely next useful change is to make recovery a more learnable action primitive, for example discrete `normal/back/back-left/back-right` actions or a temporary macro-action that holds the recovery direction for several env steps. That still keeps the input 8D and avoids map/collision preprocessing, but gives PPO a cleaner exploration target.

## Macro Recovery Dense Half-Mountain Experiment

This is the current follow-up to the recovery audit. It keeps the `target-circle v2` anti-snake control shape, but adds a short macro recovery action.

- state mode: `observable8_target8_macro_recovery_dense_halfmountain`
- fixed trigger state mode: `observable8_target8_macro_trigger_dense_halfmountain`
- observation: same 8D `Observable8TargetWrapper`
- action modes: `target_relative_macro_recovery`, `target_relative_macro_trigger_recovery`
- normal movement: `target_angle + offset[-60deg, +60deg]`
- recovery choices: back-left, back, back-right; the trigger version keeps normal movement as default and only enters recovery when action[3] crosses the trigger
- macro duration: 4 environment steps
- mountain count: `16` (double v2)
- mountain radius range: `35px` to `80px` (half v2)
- tree count/radius: `45`, `18px`
- stop rule: `distance_to_target <= 8px`

10k smoke runs on `ssh 253`:

```bash
rtk ssh 253 'cd /home/weiaokang/fishing/blind_nav_rl && tail -f runs/目标8宏脱困山体减半密度翻倍10k/训练.log'
rtk ssh 253 'cd /home/weiaokang/fishing/blind_nav_rl && tail -f runs/目标8触发宏脱困山体减半密度翻倍10k/训练.log'
```

Launchers:

```bash
rtk ssh 253 'cd /home/weiaokang/fishing/blind_nav_rl && RUN_DIR=/home/weiaokang/fishing/blind_nav_rl/runs/目标8宏脱困山体减半密度翻倍10k EPISODES=10000 N_ENVS=256 bash launch_target8_macro_recovery_dense_halfmountain.sh'
rtk ssh 253 'cd /home/weiaokang/fishing/blind_nav_rl && RUN_DIR=/home/weiaokang/fishing/blind_nav_rl/runs/目标8触发宏脱困山体减半密度翻倍10k EPISODES=10000 N_ENVS=256 bash launch_target8_macro_trigger_dense_halfmountain.sh'
```

10k result:

| Variant | Normal success | Pressure success | Avg angle error | Recovery rate | Conclusion |
|---|---:|---:|---:|---:|---|
| `macro_recovery_dense_halfmountain` Tiny | 0.00 | 0.00 | pressure about 135deg | 1.00 | Bad action definition; policy learned persistent diagonal backward movement |
| `macro_trigger_dense_halfmountain` Tiny | 0.95 | 0.75 | normal 6.3deg / pressure 4.35deg | 0.00 | Preserves straight-line behavior; 10k did not learn active recovery yet |
| `macro_trigger_dense_halfmountain` Medium `128x2` | 0.95 | 0.80 | normal 5.98deg / pressure 3.82deg | 0.00 | Best 10k pressure result among the trigger runs |
| `macro_trigger_dense_halfmountain` Current `256x2` | 0.95 | 0.75 | normal 7.22deg / pressure 5.70deg | 0.00 | More parameters did not improve this 10k run |
| `macro_trigger_dense_halfmountain` Medium `128x2` 30k | 1.00 | 0.80 | normal 6.94deg / pressure 4.60deg | 0.00 | Longer training improved normal success to 100%, but did not teach active recovery |

Shared trigger artifacts:

- model: `http://43.242.75.250:12346/file/目标8触发宏脱困山体减半密度翻倍结果/模型.zip`
- normal video: `http://43.242.75.250:12346/file/目标8触发宏脱困山体减半密度翻倍结果/普通测试视频.mp4`
- pressure video: `http://43.242.75.250:12346/file/目标8触发宏脱困山体减半密度翻倍结果/压力测试视频.mp4`
- train result CSV: `http://43.242.75.250:12346/file/目标8触发宏脱困山体减半密度翻倍结果/训练结果.csv`
- normal metrics CSV: `http://43.242.75.250:12346/file/目标8触发宏脱困山体减半密度翻倍结果/普通指标.csv`
- pressure metrics CSV: `http://43.242.75.250:12346/file/目标8触发宏脱困山体减半密度翻倍结果/压力指标.csv`

Shared Medium/Current comparison artifacts:

- comparison CSV: `http://43.242.75.250:12346/file/目标8触发宏脱困模型对比10k/普通详细指标.csv`
- Medium model: `http://43.242.75.250:12346/file/目标8触发宏脱困模型对比10k/中型模型.zip`
- Medium normal video: `http://43.242.75.250:12346/file/目标8触发宏脱困模型对比10k/中型普通详细视频.mp4`
- Medium pressure video: `http://43.242.75.250:12346/file/目标8触发宏脱困模型对比10k/中型压力测试视频.mp4`
- Current model: `http://43.242.75.250:12346/file/目标8触发宏脱困模型对比10k/当前模型.zip`
- Current normal video: `http://43.242.75.250:12346/file/目标8触发宏脱困模型对比10k/当前普通详细视频.mp4`
- Current pressure video: `http://43.242.75.250:12346/file/目标8触发宏脱困模型对比10k/当前压力测试视频.mp4`

Shared Medium 30k artifacts:

- Medium 30k model: `http://43.242.75.250:12346/file/目标8触发宏脱困中型扩大训练30k/中型模型30k.zip`
- Medium 30k normal video: `http://43.242.75.250:12346/file/目标8触发宏脱困中型扩大训练30k/普通测试视频.mp4`
- Medium 30k pressure video: `http://43.242.75.250:12346/file/目标8触发宏脱困中型扩大训练30k/压力测试视频.mp4`
- Medium 30k normal metrics CSV: `http://43.242.75.250:12346/file/目标8触发宏脱困中型扩大训练30k/普通详细指标.csv`
- Medium 30k pressure metrics CSV: `http://43.242.75.250:12346/file/目标8触发宏脱困中型扩大训练30k/压力指标.csv`

## Target-circle v4 12D Recovery Result

v4 is now implemented and evaluated as a 12D extension of the deployable target-circle state.

- experiment: `rppo_medium_lstm128x2_observable12_target8_recovery_v4`
- state mode: `observable12_target8_recovery_v4`
- policy: `RecurrentPPO + MlpLstmPolicy`
- LSTM: `128x2`
- MLP heads: `[128, 128]`
- parameters: `476,297`
- model size: `5.51 MB`
- train episodes: `10000`
- train time on `ssh 253`: `860.5 s`
- action mode: `target_relative_macro_recovery`
- constraint remains unchanged: no map, no rays, no obstacle geometry, no collision flag, no stuck flag

Actual 12D state slots:

| # | State | Meaning |
|---:|---|---|
| 1 | `dx_norm` | target x offset from the stale observed position |
| 2 | `dy_norm` | target y offset from the stale observed position |
| 3 | `sin_heading` | current live heading sine |
| 4 | `cos_heading` | current live heading cosine |
| 5 | `sin_angle_error` | sine of target-angle minus heading |
| 6 | `cos_angle_error` | cosine of target-angle minus heading |
| 7 | `coord_age_norm` | age of the last coordinate observation |
| 8 | `distance_progress_norm` | recent distance reduction toward target |
| 9 | `no_progress_time_norm` | accumulated no-progress time estimated from observed progress |
| 10 | `distance_norm` | current target distance normalized by world size |
| 11 | `last_recovery_mode` | previous effective recovery mode code |
| 12 | `last_action_angle` | previous action angle channel |

Action contract:

| Action | Meaning |
|---|---|
| `angle_offset_norm` | normal target-relative offset |
| `speed_norm` | binary speed selector |
| `jump_norm` | jump/down trigger |
| `recovery_selector_norm` | maps to normal / back / back-left / back-right |

The v4 wrapper maps the fourth action channel into macro recovery buckets:

- `< -0.5`: normal
- `[-0.5, 0.0)`: back
- `[0.0, 0.5)`: back-left
- `>= 0.5`: back-right

Normal 20-episode evaluation result:

| Model | Params | Train s | Success | Avg steps | Path eff | Avg angle | Avg collisions | Recovery rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Medium `128x2` v4 | 476,297 | 860.5 | 0.95 | 43.75 | 0.974 | 7.57deg | 9.35 | 0.000 |

Pressure 20-episode evaluation result:

| Model | Success | Avg steps | Avg final dist | Path eff | Avg angle | Avg collisions | Recovery rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| Medium `128x2` v4 | 0.75 | 105.4 | 94.25 | 0.810 | 5.04deg | 66.0 | 0.000 |

Conclusion:

- Normal-map behavior is still usable and does not collapse into the earlier persistent zig-zag pattern.
- Pressure-map performance is weaker than the better trigger-macro baselines.
- `recovery_action_rate` stayed at `0.000` in both normal and pressure evaluation, so the added 12D memory signals did not make explicit recovery usage emerge in this 10k run.
- The useful part of v4 is better angle quality on normal maps; the unsolved part is still active concavity escape.

Artifacts are stored locally under:

- `/home/diana/screencap/file/目标8十二维宏脱困v4结果/模型.zip`
- `/home/diana/screencap/file/目标8十二维宏脱困v4结果/普通测试视频.mp4`
- `/home/diana/screencap/file/目标8十二维宏脱困v4结果/压力测试视频.mp4`
- `/home/diana/screencap/file/目标8十二维宏脱困v4结果/普通汇总指标.csv`
- `/home/diana/screencap/file/目标8十二维宏脱困v4结果/普通详细指标.csv`
- `/home/diana/screencap/file/目标8十二维宏脱困v4结果/压力汇总指标.csv`
- `/home/diana/screencap/file/目标8十二维宏脱困v4结果/压力详细指标.csv`

Note:

- the original `benchmark_state_dims_10k.py` run wrote the final model successfully, but the benchmark summary row ended with `BrokenPipeError` before filling aggregate eval columns
- the numbers above come from the separate completed reruns:
  - `eval_target8_sweep.py` to `runs/目标8十二维宏脱困v4_普通评估20`
  - `eval_target8_pressure.py` to `runs/目标8十二维宏脱困v4_压力评估20`

## Target-circle v5 Discrete Macro Recovery

v5 is the current execution branch for the one-pass audit plan. It keeps the no-map/no-ray/no-obstacle-state constraint, but changes recovery from a continuous selector into a discrete macro-action interface.

- experiment: `rppo_medium_lstm128x2_observable12_target8_discrete_macro_v5`
- state mode: `observable12_target8_discrete_macro_v5`
- policy: `RecurrentPPO + MlpLstmPolicy`
- LSTM: `128x2`
- MLP heads: `[128, 128]`
- action space: `MultiDiscrete([5, 5, 2])`
- macro recovery duration: `6` env steps
- target radius: `8px`
- training host: `v1002`
- status: 100k training complete; final archive pulled to local shared folder

Discrete action contract:

| Dimension | Meaning | Values |
|---:|---|---|
| 1 | `move_mode` | `0 normal`, `1 back`, `2 back-left`, `3 back-right`, `4 normal fine adjust` |
| 2 | `angle_offset_bin` | `-60, -30, 0, +30, +60` degrees |
| 3 | `speed_bin` | `50 / 100` |

Current audit evidence:

| Check | Status | Evidence |
|---|---|---|
| 12D observation shape | passed | `runs/目标8离散宏动作v5_本地审计/audit.csv` |
| no obstacle/map/ray observation | passed by wrapper design | `Observable12DiscreteMacroWrapper` |
| discrete macro action space | passed | `MultiDiscrete([5 5 2])` in audit CSV |
| macro recovery holds across steps | passed | recovery mode repeats while macro steps decrease |
| non-overlapping mountains in smoke seeds | passed | audit CSV overlap count is `0` |
| pressure seeds have straight-line mountain hits | passed | audit CSV pressure seed list |
| v1002 smoke training | passed | `/home/diana/fishing/blind_nav_rl/runs/目标8离散宏动作v5_远端smoke/` |
| 100k main training | passed | `runs/目标8离散宏动作v5_100k/训练.log`, checkpoint `100000` |
| checkpoint evaluation | passed | `runs/目标8离散宏动作v5_checkpoint评估/checkpoint_eval_summary.csv` contains `100000` |
| final videos and archive | passed | `/home/diana/screencap/file/目标8离散宏动作v5最终结果/` |

Final v5 results:

| Metric | Value |
|---|---:|
| selected checkpoint | `100000` |
| trainable parameters | `477325` |
| normal success rate | `0.90` |
| normal average steps | `44.5` |
| normal path efficiency | `0.951` |
| normal average angle error | `7.71 deg` |
| pressure success rate | `0.75` |
| pressure average steps | `90.2` |
| pressure path efficiency | `0.761` |
| pressure average angle error | `7.41 deg` |
| pressure average collisions | `69.8` |
| pressure recovery action rate | `0.000` |

Final local artifacts:

- `/home/diana/screencap/file/目标8离散宏动作v5最终结果/模型.zip`
- `/home/diana/screencap/file/目标8离散宏动作v5最终结果/checkpoint评估汇总.csv`
- `/home/diana/screencap/file/目标8离散宏动作v5最终结果/普通汇总指标.csv`
- `/home/diana/screencap/file/目标8离散宏动作v5最终结果/普通详细指标.csv`
- `/home/diana/screencap/file/目标8离散宏动作v5最终结果/普通测试视频.mp4`
- `/home/diana/screencap/file/目标8离散宏动作v5最终结果/压力汇总指标.csv`
- `/home/diana/screencap/file/目标8离散宏动作v5最终结果/压力详细指标.csv`
- `/home/diana/screencap/file/目标8离散宏动作v5最终结果/压力测试视频.mp4`
- `/home/diana/screencap/file/目标8离散宏动作v5最终结果/最终选择说明.md`

Verified local URLs:

- `http://127.0.0.1:6000/file/%E7%9B%AE%E6%A0%878%E7%A6%BB%E6%95%A3%E5%AE%8F%E5%8A%A8%E4%BD%9Cv5%E6%9C%80%E7%BB%88%E7%BB%93%E6%9E%9C/%E6%99%AE%E9%80%9A%E6%B5%8B%E8%AF%95%E8%A7%86%E9%A2%91.mp4`
- `http://127.0.0.1:6000/file/%E7%9B%AE%E6%A0%878%E7%A6%BB%E6%95%A3%E5%AE%8F%E5%8A%A8%E4%BD%9Cv5%E6%9C%80%E7%BB%88%E7%BB%93%E6%9E%9C/%E5%8E%8B%E5%8A%9B%E6%B5%8B%E8%AF%95%E8%A7%86%E9%A2%91.mp4`
- `http://127.0.0.1:6000/file/%E7%9B%AE%E6%A0%878%E7%A6%BB%E6%95%A3%E5%AE%8F%E5%8A%A8%E4%BD%9Cv5%E6%9C%80%E7%BB%88%E7%BB%93%E6%9E%9C/%E6%9C%80%E7%BB%88%E9%80%89%E6%8B%A9%E8%AF%B4%E6%98%8E.md`

Public route status:

- `http://ai.jiaran.icu:12346/file/目标8离散宏动作v5最终结果/普通测试视频.mp4`
- `http://ai.jiaran.icu:12346/file/目标8离散宏动作v5最终结果/压力测试视频.mp4`
- `http://ai.jiaran.icu:12346/file/目标8离散宏动作v5最终结果/最终选择说明.md`

The local file route was verified through `127.0.0.1:6000`. At audit time the external `ai.jiaran.icu:12346` proxy returned `403` even for `/file/health.txt`, so the public proxy was not treated as proof of availability.

Video validation:

- normal video: H.264, `960x960`, `20fps`, `293.0s`
- pressure video: H.264, `960x960`, `20fps`, `567.2s`

Final archive command after the 100k checkpoint has been evaluated:

```bash
ssh v1002 'cd /home/diana/fishing/blind_nav_rl && source ~/miniconda3/etc/profile.d/conda.sh && conda activate ML && python finalize_v5_audit.py --min-checkpoint 100000'
```

It selects the best eligible checkpoint from `runs/目标8离散宏动作v5_checkpoint评估/checkpoint_eval_summary.csv`, generates normal 20-episode and pressure 20-episode concatenated videos, and writes Chinese-named artifacts to `runs/目标8离散宏动作v5_最终归档/`.

Distributed task launcher:

```bash
python distributed_v5_launcher.py --profile light --tasks 2
python distributed_v5_launcher.py --profile normal --tasks 4 --hosts v1002 253 local
python distributed_v5_progress.py --manifest runs/分布式任务_<batch>.json
python distributed_v5_progress.py --include-main
```

Default per-host rules:

- `local`: raw gate `cpu<85%`, `mem<80%`; effective launch gate `cpu<80%`, `mem<77%`
- `v1002`: raw gate `cpu<90%`, `mem<80%`; effective launch gate `cpu<85%`, `mem<77%`
- `253`: raw gate `cpu<80%`, `mem<80%`; effective launch gate `cpu<75%`, `mem<77%`
- `light` prefers `local -> v1002 -> 253`
- `normal/heavy` prefer `v1002 -> 253 -> local`
- each task re-checks host resources before launch
- task directories and files use Chinese names under `runs/`

Parameter sweep models:

| Model | Policy | LSTM | MLP heads | Purpose |
|---|---|---:|---|---|
| Tiny | `MlpLstmPolicy` | `64x1` | `[64]` | minimum viable recurrent model |
| Small | `MlpLstmPolicy` | `128x1` | `[64,64]` | light deployment candidate |
| Medium | `MlpLstmPolicy` | `128x2` | `[128,128]` | balanced candidate |
| Current | `MlpLstmPolicy` | `256x2` | `[128,128]` | previous capacity baseline |
| Large | `MlpLstmPolicy` | `384x2` | `[192,192]` | capacity stress test |

Final v2 sweep result, 100k episodes per model on `ssh 253`, 20 shared eval seeds:

| Model | Params | Train s | Success | Avg steps | Path eff | Avg angle | Flips | Recommendation |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Large `384x2` | 3,798,535 | 5160.2 | 1.00 | 16.6 | 0.974 | 10.8deg | 0.0 | Best score |
| Tiny `64x1` | 46,471 | 1106.0 | 1.00 | 16.6 | 0.973 | 11.4deg | 0.0 | Best lightweight |
| Medium `128x2` | 472,071 | 3020.3 | 1.00 | 16.6 | 0.971 | 11.5deg | 0.0 | Balanced |
| Current `256x2` | 1,696,775 | 4280.7 | 1.00 | 16.4 | 0.972 | 12.1deg | 0.0 | Not best |
| Small `128x1` | 166,407 | 1530.6 | 1.00 | 17.3 | 0.972 | 11.9deg | 0.0 | Not best |

Public artifacts:

- summary CSV: `http://43.242.75.250:12346/file/blind_nav_target8_v2_final_eval/target8_sweep_summary.csv`
- summary Markdown: `http://43.242.75.250:12346/file/blind_nav_target8_v2_final_eval/target8_sweep_summary.md`
- best Large model: `http://43.242.75.250:12346/file/blind_nav_target8_v2_final_eval/best_rppo_large_lstm384x2_observable8_target8_v2_offset60.zip`
- best Large 20-episode video: `http://43.242.75.250:12346/file/blind_nav_target8_v2_final_eval/rppo_large_lstm384x2_observable8_target8_v2_offset60/target8_eval_concat.mp4`

## Legacy 7D stop task

Observation from `Observable7StopWrapper`:

1. `dx_norm`: observed x offset from stale observed position to target, normalized by world size
2. `dy_norm`: observed y offset from stale observed position to target, normalized by world size
3. `sin_heading`: sine of current character heading
4. `cos_heading`: cosine of current character heading
5. `desired_stop_distance_norm`: desired stop radius mapped from `[8, 20]` into `[-1, 1]`
6. `coord_age_norm`: age of last coordinate observation, normalized to `[0, 1]`
7. `last_speed_cmd`: previous speed action mapped to `[-1, 1]`

This keeps the deploy signal path minimal. No `angle_error`, `distance_error`, collision normals, rays, or map features are exposed to the policy in this branch.

## 11D small-state experiment

`Observable11StopWrapper` is the current small-state upgrade path. It keeps the same deployment constraint as 7D: no map, no rays, no obstacle geometry, no collision flag, and no stuck flag. The added features are derived only from stale coordinates, target coordinates, live heading, stop distance config, and the previous observed distance.

| # | State | 中文解释 | 来源 |
|---:|---|---|---|
| 1 | `dx_norm` | 目标相对上次坐标观测点的 x 偏移，按地图尺度归一化 | 自身坐标 + 目标坐标 |
| 2 | `dy_norm` | 目标相对上次坐标观测点的 y 偏移，按地图尺度归一化 | 自身坐标 + 目标坐标 |
| 3 | `sin_heading` | 当前人物/指针方向的正弦 | 指针识别 |
| 4 | `cos_heading` | 当前人物/指针方向的余弦 | 指针识别 |
| 5 | `desired_stop_distance_norm` | 本局希望停在目标外多远，按当前阶段 stop range 归一化 | 配置参数 |
| 6 | `coord_age_norm` | 上次坐标刷新到现在经过多久，模拟真实坐标 500ms 延迟 | 坐标时间戳 |
| 7 | `last_speed_cmd` | 上一次速度动作，映射到 `[-1, 1]` | 控制器记录 |
| 8 | `sin_angle_error` | 目标方向减当前 heading 的角度差正弦 | 坐标 + 指针 |
| 9 | `cos_angle_error` | 目标方向减当前 heading 的角度差余弦 | 坐标 + 指针 |
| 10 | `stop_error_norm` | 当前距离减希望停止距离，正数表示还在停止圈外，负数表示太近 | 坐标 + 停止距离 |
| 11 | `distance_progress_norm` | 最近一次坐标观测中距离目标减少了多少，正数表示靠近 | 前后两次观测距离 |

The purpose is not to give the model obstacle information. It only exposes simple, deployable quantities that reduce the burden on LSTM memory.

## Environment

Core environment: [blind_nav_rl/env.py](/home/diana/fishing/blind_nav_rl/blind_nav_rl/env.py)

- World: large continuous 2D plane
- Agent motion:
  - `dt = 0.3s`
  - heading has turn-rate limit
  - action mode for current experiments is `stick_delta`
  - commanded stick angle is a delta relative to current heading
- Obstacles:
  - fixed-radius trees
  - non-overlapping polygon mountains
  - many mountains are concave
  - collision resolves as sliding along tree tangent or mountain edge
  - jump provides breakout from traps / concave pockets
- Goal for current stop task:
  - approach target
  - stop at configurable distance
  - face target direction

## Current task variants

Configured in [benchmark_state_dims_10k.py](/home/diana/fishing/blind_nav_rl/benchmark_state_dims_10k.py).

- `observable7_stop_stage1_delta45` / `observable11_stop_stage1_delta45`
  - short-range reaching stage
  - start-target distance `160px` to `360px`
  - very light obstacle density: `16` trees, `3` mountains
  - target radius `55px`
  - no stop-distance requirement in this stage
- `observable7_stop_stage2_delta45` / `observable11_stop_stage2_delta45`
  - medium curriculum stage
  - start-target distance `260px` to `560px`
  - medium obstacle density: `48` trees, `8` mountains
  - stop radius target `35px` to `55px`
  - stop tolerance `12px`
  - align tolerance effectively disabled
  - stop speed threshold effectively disabled
- `observable7_stop_stage3_delta45` / `observable11_stop_stage3_delta45`
  - pre-final curriculum stage
  - start-target distance `320px` to `680px`
  - heavier obstacle density: `72` trees, `12` mountains
  - stop radius target `28px` to `42px`
  - stop tolerance `10px`
  - align tolerance effectively disabled
  - stop speed threshold effectively disabled
- `observable7_stop_delta45`
  - strict target task
  - start-target distance `800px` to `1400px`
  - stop tolerance `3px`
  - align tolerance `10deg`
  - stop speed threshold `12`
- `observable7_stop_quality_delta60` / `observable11_stop_quality_delta60`
  - strict target task with quality-oriented reward
  - stick delta angle limit `60deg`
  - start-target distance `800px` to `1400px`
  - stop distance target `8px` to `20px`
  - stop tolerance `3px`
  - align tolerance `10deg`
  - stop speed threshold `12`
  - coordinate refresh interval randomized from `0.35s` to `0.75s`
  - coordinate observation noise `2px`
  - best model selected by a quality score, not success rate alone
- `observable7_stop_line_delta45` / `observable11_stop_line_delta45`
  - straight-line-first control mode
  - action mode `target_tracking`: action angle is only a small residual around target direction
  - model output `angle=0` means move directly toward target
  - target tracking turn limit `45deg`
  - residual limit `12deg`
  - line reward penalizes lateral movement, action-angle oscillation, and large far-field heading error
  - intended to avoid late correction and snake-like paths when no obstacle blocks the route

All stages use the same 7D observation and same recurrent architecture.

## Reward design

For the 8px target-circle task:

- main shaping: reduction in distance to target
- per-step time cost
- per-step angle-error penalty
- lateral movement penalty to discourage snake-like paths
- penalty for moving backward relative to the target
- action-change penalty to reduce rapid left-right oscillation
- small speed bonus while far from target
- collision and no-progress penalties
- jump penalty unless it helps recover from stuck behavior
- terminal success reward when entering the target-centered `8px` circle
- terminal failure penalty on timeout

For the stop-distance task:

- main shaping: reduction in stop-distance error
- per-step time cost
- distance-error penalty
- angle penalty
- bonus near the target ring
- stronger bonus when inside ring and aligned
- penalty for approaching too close inside the stop ring
- collision penalty
- no-progress penalty
- small jump penalty when used unnecessarily
- breakout bonus when jump helps escape stuck state
- large terminal success reward
- terminal failure penalty on truncation

The reward logic is in [env.py](/home/diana/fishing/blind_nav_rl/blind_nav_rl/env.py:220).

The quality profile `reward_profile="quality_stop"` tightens this objective:

- stronger penalty for stop-distance error
- stronger penalty for being inside the stop ring
- larger bonus only when stop distance and heading are both correct
- lower reward for unnecessary jump attempts
- lower reward for repeated no-progress behavior
- harsher terminal failure penalty when the final stop error is large

The continuation runner uses this model-selection score:

```text
quality_score =
  success_rate * 100
  - avg_stop_error * 8
  - avg_abs_angle_error * 12
  - avg_jump_attempts * 0.08
  - avg_collisions * 0.60
  - avg_steps * 0.02
```

This avoids choosing a checkpoint that has high success but poor stop precision.

## Training entry points

Main benchmark / single experiment runner:

```bash
rtk python benchmark_state_dims_10k.py --output-dir <out> --only rppo_lstm256x2_observable8_target8_line_delta90 --episodes 10000 --n-envs 256
```

Target-circle v2 model sweep:

```bash
rtk python benchmark_state_dims_10k.py \
  --output-dir <out> \
  --only \
    rppo_tiny_lstm64x1_observable8_target8_v2_offset60 \
    rppo_small_lstm128x1_observable8_target8_v2_offset60 \
    rppo_medium_lstm128x2_observable8_target8_v2_offset60 \
    rppo_current_lstm256x2_observable8_target8_v2_offset60 \
    rppo_large_lstm384x2_observable8_target8_v2_offset60 \
  --episodes 100000 \
  --n-envs 256
```

Target-circle v2 final comparison/eval:

```bash
rtk python eval_target8_sweep.py \
  --train-output-dir <train_out> \
  --out-dir <eval_out> \
  --experiments \
    rppo_tiny_lstm64x1_observable8_target8_v2_offset60 \
    rppo_small_lstm128x1_observable8_target8_v2_offset60 \
    rppo_medium_lstm128x2_observable8_target8_v2_offset60 \
    rppo_current_lstm256x2_observable8_target8_v2_offset60 \
    rppo_large_lstm384x2_observable8_target8_v2_offset60 \
  --episodes 20 \
  --video-episodes 20
```

Clean target-circle eval renderer:

```bash
rtk python render_target8_eval.py --model <model.zip> --out-dir <out> --episodes 5 --fps 20
```

Sequential curriculum runner:

```bash
rtk python train_curriculum_sequence.py --output-dir <out> --episodes-per-stage 25000 --n-envs 128
```

This runs:

1. `stage1`
2. `stage2`
3. `stage3`
4. `strict`

and resumes weights between stages.

11D small-state curriculum runner:

```bash
rtk python train_observable11_sequence.py --output-dir <out> --episodes-per-stage 6000 --n-envs 64 --torch-threads 4
```

This runs:

1. `observable11_stop_stage2_delta45`
2. `observable11_stop_stage3_delta45`
3. `observable11_stop_stage3_heading_delta45`
4. `observable11_stop_final_warm1_delta45`
5. `observable11_stop_final_warm2_delta45`
6. `observable11_stop_delta45`

Do not resume a 7D checkpoint into the 11D policy because the observation input layer shape is different. `--resume-model` is only for continuing a previous 11D run.

## Environment speed benchmark

The environment now has a broad-phase collision path enabled by default:

- obstacle AABB prefilter
- spatial grid candidate lookup
- cached mountain AABB/edges
- `use_spatial_grid=False` and `use_collision_aabb_filter=False` are kept for legacy speed/correctness comparisons

Run the local benchmark:

```bash
rtk python benchmark_env_speed.py --episodes 80 --profile
```

Recent strict-task local result:

| mode | step/s | ms/step | speedup |
|---|---:|---:|---:|
| legacy full scan | 895 | 1.118 | 1.0x |
| AABB full scan | 11046 | 0.091 | 12.3x |
| spatial grid + AABB | 13891 | 0.072 | 15.5x |

The optimized path was checked against the legacy full scan with fixed seeds/actions for 20 episodes: same step counts, final positions, rewards, collision counts, and success flags.

## Evaluation

Single-model evaluation:

```bash
rtk python eval_observable7_stop.py --model <model.zip> --out-dir <eval_dir> --state-mode observable11_stop_delta45 --episodes 20 --video-episodes 3
```

Outputs:

- per-episode log
- `eval_results.csv`
- optional concatenated MP4

## Remote run convention

For `ssh 253`:

```bash
rtk ssh 253 'cd /home/weiaokang/fishing/blind_nav_rl && source /home/weiaokang/ML/bin/activate && python train_curriculum_sequence.py --output-dir /home/weiaokang/screencap/file/<run_name> --episodes-per-stage 25000 --n-envs 128'
```

Use `/home/weiaokang/screencap/file` for persistent remote artifacts.

11D launch script for `ssh 253`:

```bash
rtk ssh 253 'cd /home/weiaokang/fishing/blind_nav_rl && bash launch_observable11_sequence.sh'
```

It writes to `/home/weiaokang/screencap/file/blind_nav_observable11_smooth_final`, checks `load<20`, memory `<80%`, CPU `<80%`, and saves checkpoints every 1000 completed episodes per stage.

Quality continuation from the current best 11D model:

```bash
rtk ssh 253 'cd /home/weiaokang/fishing/blind_nav_rl && bash launch_quality_continue.sh'
```

Defaults:

- resume model: `/home/weiaokang/screencap/file/blind_nav_observable11_continue_until_good_opt/best_model.zip`
- output: `/home/weiaokang/screencap/file/blind_nav_observable11_quality_continue`
- state mode: `observable11_stop_quality_delta60`
- chunks: `10000` episodes, up to `30000`
- workers: `256`

Override from the command line when needed:

```bash
rtk ssh 253 'cd /home/weiaokang/fishing/blind_nav_rl && OUT_DIR=/home/weiaokang/screencap/file/my_run N_ENVS=128 bash launch_quality_continue.sh'
```

Straight-line-first continuation:

```bash
rtk ssh 253 'cd /home/weiaokang/fishing/blind_nav_rl && bash launch_line_continue.sh'
```

This resumes from the quality model by default and writes to:

`/home/weiaokang/screencap/file/blind_nav_observable11_line_continue`

## v6 Recovery-Pressure Follow-up

This branch adds a new blind macro-recovery variant without changing v5 behavior.

- experiment: `rppo_medium_lstm128x2_observable12_target8_discrete_macro_v6`
- state mode: `observable12_target8_discrete_macro_v6`
- observation: same 12D blind state as v5
- action space: `MultiDiscrete([5, 5, 2])`
- reward profile: `target_line_macro_recovery_v6`
- macro recovery duration: `6` env steps
- params: `477325`

Relative to v5, v6 changes two things:

- harder concave-trap preset: more mountains and higher deep-concavity probability
- stronger reward shaping around recovery usefulness:
  - punish staying in normal movement during sustained no-progress / fresh collision pressure
  - reward recovery actions that restart progress after stuck pressure

Validation completed on 2026-05-19:

| Check | Result | Evidence |
|---|---|---|
| compile check | passed | local `py_compile` |
| local structure audit | passed | `runs/目标8离散宏动作v6_本地审计/audit.csv` |
| remote structure audit | passed | `/home/weiaokang/fishing/blind_nav_rl/runs/目标8离散宏动作v6_远端审计/audit.csv` |
| remote smoke train | passed | `/home/weiaokang/fishing/blind_nav_rl/runs/目标8离散宏动作v6_远端smoke/` |
| remote smoke pressure eval | passed | `/home/weiaokang/fishing/blind_nav_rl/runs/目标8离散宏动作v6_远端smoke_pressure/` |

Current smoke metrics from `ssh 253`:

| Metric | Value |
|---|---:|
| train episodes | `256` |
| train time | `41.19s` |
| normal success rate | `1.00` |
| normal avg steps | `42.5` |
| pressure episodes | `4` |
| pressure success rate | `0.50` |
| pressure avg steps | `150.75` |
| pressure avg final distance | `298.95` |
| pressure avg angle error | `3.19 deg` |
| pressure avg collisions | `126.25` |
| pressure recovery action rate | `0.000` |

Interpretation:

- v6 wiring is complete end-to-end
- short smoke runs still do not produce explicit macro recovery usage
- the next meaningful step is long remote training with checkpoint tracking

Current long run on `ssh 253`:

- run dir: `/home/weiaokang/fishing/blind_nav_rl/runs/目标8离散宏动作v6_100k`
- checkpoint eval dir: `/home/weiaokang/fishing/blind_nav_rl/runs/目标8离散宏动作v6_checkpoint评估`
- episodes: `100000`
- n_envs: `128`
- seed base: `9400000`
- eval seed base: `9500000`

Progress commands:

```bash
rtk ssh 253 'cd /home/weiaokang/fishing/blind_nav_rl && tail -f runs/目标8离散宏动作v6_100k/训练.log'
rtk ssh 253 'cd /home/weiaokang/fishing/blind_nav_rl && ps -p $(cat runs/目标8离散宏动作v6_100k/训练.pid) -o pid,stat,etime,cmd'
rtk ssh 253 'cd /home/weiaokang/fishing/blind_nav_rl && ps -p $(cat runs/目标8离散宏动作v6_checkpoint评估/watcher.pid) -o pid,stat,etime,cmd'
rtk ssh 253 'cd /home/weiaokang/fishing/blind_nav_rl && cat runs/目标8离散宏动作v6_checkpoint评估/checkpoint_eval_summary.csv 2>/dev/null || true'
rtk python3 distributed_v6_progress.py --hosts 253 v1002 --include-main
```

Supporting scripts:

- `watch_v6_checkpoints.py`
- `finalize_v6_audit.py`
- `launch_v6_finalizer.sh`
- `distributed_v6_launcher.py`
- `distributed_v6_progress.py`
- `checkpoint_summary.py`
- `append_manual_checkpoint_eval.py`

First checkpoint evidence on `253`:

| Checkpoint | Normal success | Normal avg steps | Normal path eff | Normal avg angle | Pressure success | Pressure avg steps | Pressure path eff | Pressure avg angle | Pressure recovery rate | Pressure avg collisions |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `10000` | `0.80` | `68.6` | `0.923` | `6.49 deg` | `0.80` | `78.15` | `0.893` | `6.96 deg` | `0.000` | `56.45` |

Manual 20000-checkpoint verification on `253`:

| Checkpoint | Normal success | Normal avg steps | Normal path eff | Normal avg angle | Pressure success | Pressure avg steps | Pressure path eff | Pressure avg angle | Pressure recovery rate | Pressure avg collisions |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `20000` | `0.80` | `68.6` | `0.923` | `6.49 deg` | `0.80` | `78.15` | `0.893` | `6.96 deg` | `0.000` | `56.45` |

This manual `20000` evaluation matched the `10000` checkpoint exactly on the fixed 20-episode eval seed set.

Manual 30000-checkpoint verification on `253`:

| Checkpoint | Normal success | Normal avg steps | Normal path eff | Normal avg angle | Pressure success | Pressure avg steps | Pressure path eff | Pressure avg angle | Pressure recovery rate | Pressure avg collisions |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `30000` | `0.80` | `68.6` | `0.923` | `6.49 deg` | `0.80` | `78.15` | `0.893` | `6.96 deg` | `0.000` | `56.45` |

This manual `30000` evaluation also matched the `10000` and `20000` checkpoints exactly on the fixed 20-episode eval seed set.

Current summary from `checkpoint_summary.py` on `253`:

- `first=10000`
- `latest=30000`
- `best=10000`
- delta from `10000 -> 30000`: `pressure_success +0.000`, `pressure_recovery +0.000`, `pressure_collisions +0.00`

Practical interpretation:

- by `30000` episodes, this v6 branch is still behaviorally identical to the `10000` checkpoint on the fixed evaluation seed set
- the model is not improving its explicit recovery usage through longer training alone, at least not in the first `30000` episodes

Parallel v6 run on `ssh v1002`:

- run dir: `/home/diana/fishing/blind_nav_rl/runs/目标8离散宏动作v6_v1002_100k`
- checkpoint eval dir: `/home/diana/fishing/blind_nav_rl/runs/目标8离散宏动作v6_v1002_checkpoint评估`
- final archive dir: `/home/diana/fishing/blind_nav_rl/runs/目标8离散宏动作v6_v1002_最终归档`
- episodes: `100000`
- n_envs: `128`
- seed base: `9600000`
- eval seed base: `9700000`

`v1002` progress commands:

```bash
rtk ssh v1002 'cd /home/diana/fishing/blind_nav_rl && tail -f runs/目标8离散宏动作v6_v1002_100k/训练.log'
rtk ssh v1002 'cd /home/diana/fishing/blind_nav_rl && ps -p $(cat runs/目标8离散宏动作v6_v1002_100k/训练.pid) -o pid,stat,etime,cmd'
rtk ssh v1002 'cd /home/diana/fishing/blind_nav_rl && cat runs/目标8离散宏动作v6_v1002_checkpoint评估/checkpoint_eval_summary.csv 2>/dev/null || true'
```

Checkpoint summary helper:

```bash
rtk ssh 253 'cd /home/weiaokang/fishing/blind_nav_rl && source /home/weiaokang/ML/bin/activate && python checkpoint_summary.py --csv runs/目标8离散宏动作v6_checkpoint评估/checkpoint_eval_summary.csv'
rtk ssh v1002 'cd /home/diana/fishing/blind_nav_rl && source ~/miniconda3/etc/profile.d/conda.sh && conda activate ML && python checkpoint_summary.py --csv runs/目标8离散宏动作v6_v1002_checkpoint评估/checkpoint_eval_summary.csv'
```

At the time of writing, the `253` summary shows:

- `first=10000`
- `latest=20000`
- `best=10000`
- `pressure_success` delta `0.000`
- `pressure_recovery` delta `0.000`
