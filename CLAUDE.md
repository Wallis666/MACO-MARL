# CLAUDE.md — AI 助手项目指南

## 项目简介

本科毕设项目。构建一个**基于世界模型的多任务多智能体强化学习（MT-MARL）框架，支持 few-shot 快速适应**。核心目标：在多个已知任务上联合训练后，仅凭几条新任务的演示轨迹（无需梯度更新）即可适应未见过的新任务。

本项目参考五个代码库：
- **DIMA**（`references/DIMA/`）：基于扩散模型的多智能体世界模型（NeurIPS 2025）
- **M3W-MARL**（`references/m3w-marl/`）：基于 MoE 的多任务世界模型（NeurIPS 2025）— 参考世界模型架构和训练流程
- **HiSSD**（`references/HiSSD/`）：分层可分离技能发现的离线多任务 MARL（ICLR 2025）— 参考技能分解与多任务迁移
- **TD-MPC2**（`references/tdmpc2/`）：可扩展的多任务世界模型（ICLR 2024）— 主要参考，任务嵌入 + Dense 多任务架构
- **dm_control**（`references/dm_control/`）：DeepMind Control Suite — 参考任务定义和奖励设计

目标环境：**MA-MuJoCo**（多智能体 MuJoCo）。

### 开发路线图

| 阶段 | 说明 | 状态 |
|------|------|------|
| 阶段 0 | 单任务验证（Dense 世界模型 + Dreamer 风格训练） | 已完成 |
| 阶段 1 | 多任务基础设施（任务注册表、向量化环境、per-task 日志） | 已完成 |
| 阶段 2 | 任务条件化世界模型（可学习任务嵌入 + Concatenation 条件化） | 已完成 |
| 阶段 3 | 上下文编码器（轨迹 → 任务嵌入，对齐训练嵌入空间） | 未开始 |
| 阶段 4 | Few-shot 适应评估（冻结模型，上下文编码器推断任务） | 未开始 |
| 阶段 5 | 实验评估与消融实验 | 未开始 |

### 上次尝试的已知问题
- 多任务训练未能收敛（回报不到理想值的 10%）
- 疑似原因：奖励尺度未对齐、观测/动作空间对齐 bug、或世界模型预测误差过大
- 已放弃 MoE 路线，改用 Dense 模型 + 任务条件化（可学习嵌入 + FiLM + 上下文编码器）
- 参考：TD-MPC2 用纯 Dense 模型成功训练 80 个多样化任务，Newt 扩展到 200 个任务

---

## 技术栈

| 组件 | 版本 / 详情 |
|------|-------------|
| Python | 3.11（服务器）/ 3.10.19（本地） |
| Conda 环境 | **maco** |
| PyTorch | 2.10.0（CUDA） |
| MuJoCo | 3.1.6 |
| Gymnasium | 1.1.1 |
| Gymnasium-Robotics | 1.3.1（提供 MaMuJoCo 环境） |
| PettingZoo | 1.25.0 |
| einops | 0.8.1 |
| tensorboard | 日志记录 |
| 操作系统 | Windows 11 Pro |

---

## 关键文档

- `docs/coding-standard.md` — 编码规范（必读）
- `docs/changelog.md` — 变更日志
- `docs/requirements.md` — 需求文档（项目目标、技术路线、性能指标）
- `docs/paper-readings/` — 论文阅读笔记

### 参考论文

| 文件 | 说明 |
|------|------|
| `pdfs/DIMA.pdf` | DIMA 论文 — 基于扩散模型的多智能体世界模型（NeurIPS 2025） |
| `pdfs/M3W-MARL.pdf` | M3W 论文 — 基于 MoE 的多任务世界模型（NeurIPS 2025） |
| `pdfs/HiSSD.pdf` | HiSSD 论文 — 分层可分离技能发现的离线多任务 MARL（ICLR 2025） |
| `pdfs/TD-MPC2.pdf` | TD-MPC2 论文 — 可扩展的多任务世界模型（ICLR 2024） |

### 参考代码库关键路径

**DIMA**（`references/DIMA/`，只读）：
- `train.py` — 训练入口
- `agent/world_models/diffusion/` — 基于扩散的状态预测
- `agent/world_models/vq.py` — VQ/FSQ 自编码器
- `agent/world_models/rew_end_model.py` — 奖励与终止模型（Transformer 架构）
- `agent/learners/DreamerLearner.py` — 主学习器
- `agent/optim/loss.py` — 损失函数与想象 rollout

**M3W-MARL**（`references/m3w-marl/`，只读）：
- `examples/train.py` — 训练入口
- `m3w/models/world_models.py` — 核心模型（SoftMoE 动力学、SparseMoE 奖励、编码器）
- `m3w/runners/world_model_runner.py` — 训练循环与 MPPI 规划器
- `m3w/algorithms/actors/world_model_actor.py` — Actor 网络
- `m3w/algorithms/critics/world_model_critic.py` — Critic 网络
- `m3w/common/buffers/world_model_buffer.py` — 经验回放缓冲区
- `m3w/envs/mujoco/` — MA-MuJoCo 环境封装

**HiSSD**（`references/HiSSD/`，只读）：
- `src/main.py` — 训练入口
- `src/modules/agents/multi_task/hissd_agent.py` — HiSSD 智能体（通用技能 + 任务特定技能）
- `src/modules/agents/multi_task/vq_skill.py` — VQ-VAE 技能编码器
- `src/learners/multi_task/hissd_learner.py` — HiSSD 学习器
- `src/controllers/multi_task/mt_hissd_controller.py` — 多任务控制器
- `src/runners/multi_task/episode_ada_runner.py` — 适应阶段 Runner

**TD-MPC2**（`references/tdmpc2/`，只读）：
- `tdmpc2/train.py` — 训练入口
- `tdmpc2/tdmpc2.py` — 核心算法（MPPI 规划、模型更新）
- `tdmpc2/common/world_model.py` — 世界模型（可学习任务嵌入、Encoder、Dynamics、Reward、Policy）
- `tdmpc2/common/layers.py` — 网络层（SimNorm、NormedLinear 等）
- `tdmpc2/common/scale.py` — RunningScale（Q 值自适应缩放）
- `tdmpc2/trainer/online_trainer.py` — 在线训练循环
- `tdmpc2/envs/wrappers/multitask.py` — 多任务环境封装

**dm_control**（`references/dm_control/`，只读）：
- `dm_control/suite/` — 各类 MuJoCo 任务定义
- `dm_control/suite/humanoid.py` — Humanoid 任务奖励函数参考

---

## 编码规范（摘要）

完整规范见 `docs/coding-standard.md`，以下为必须遵守的核心规则：

- 遵循 PEP 8，行宽 88 字符
- 导入分三组：标准库 → 第三方库 → 项目模块，组内字典序，组间空行
- 函数参数 >= 2 个（含 self）时逐行书写，尾部加逗号
- 所有参数和返回值必须有类型注解
- Docstring 除参数名外使用中文
- 张量变形使用 `einops`
- `references/` 下的参考代码库只读，新代码放入独立项目目录

---

## 变更记录规则

每次修改代码后，在 `docs/changelog.md` 顶部追加一条记录：
- 时间戳格式：`YYYYMMDD-HHMMSS`（年月日-时分秒）
- 包含类型（feat / fix / refactor / docs / test / chore / exp）、涉及文件、中文摘要

### Git 提交信息格式

**重要规则**：
- **禁止**在提交信息中添加 `Co-Authored-By` 行，提交记录只显示用户本人
- 提交信息必须使用中文描述
- 格式：`<类型>(<范围>): <中文简要说明>`
- 类型限定为：feat / fix / refactor / docs / test / chore / exp

示例：
```
feat(dynamics): 添加可配置专家数量的 SoftMoE 动力学模型
fix(reward): 修复多任务奖励尺度未归一化的问题
exp(phase0): 单任务 HalfCheetah-2x3 基线实验结果
chore(deps): 更新 PyTorch 至 2.10.0
refactor(buffer): 重构经验回放缓冲区结构
```

### 分支命名规范
- `phase-0/single-task-sanity` — 阶段 0 工作
- `phase-1/multi-task-infra` — 阶段 1 工作
- `feat/<功能名>` — 具体功能开发
- `fix/<问题描述>` — Bug 修复
- `exp/<实验名>` — 实验分支

---

## Shell / 环境规则

**关键要求**：所有 Python 命令必须在 `maco` conda 环境中执行。

```bash
# 使用 conda run 执行单次命令
conda run -n maco python script.py

# 先激活再运行（适用于交互式会话）
conda activate maco && python script.py

# 多行脚本先写入 .py 文件，再执行
conda run -n maco python path/to/script.py

# 安装新包
conda run -n maco pip install <包名>
```

不得在 maco 环境外执行任何 `python` 或 `pip install` 命令。

---

## Compaction 保护

当进行上下文压缩时，始终保留以下信息：

### 关键上下文
1. **Conda 环境**：始终使用 `maco`（Python 3.10.19，PyTorch 2.10.0）
2. **项目目标**：基于世界模型的 MT-MARL，支持对未见任务的 few-shot 适应
3. **当前阶段**：参见上方路线图表格中的最新状态
4. **上次失败经验**：多任务训练收敛到理想值的不到 10% — 大概率是工程问题（奖励尺度、观测对齐），而非算法问题
5. **开发策略**：Dense + 任务条件化（FiLM）→ 上下文编码器 → few-shot 适应
6. **参考代码库只读**：`references/` 下的 DIMA、m3w-marl、HiSSD、tdmpc2、dm_control 不得修改
7. **目标环境**：MA-MuJoCo（连续动作空间，多智能体）

### 技术决策
- 已放弃 MoE 路线，改用 Dense 世界模型 + 任务条件化（参考 TD-MPC2/Newt）
- 多任务泛化方案：可学习任务嵌入（训练）+ 上下文编码器（适应）+ FiLM 条件化（注入）
- Few-shot = 基于上下文的适应（几条轨迹通过编码器 → 任务向量，无需梯度更新）
- 策略学习：想象空间训练（Dreamer 风格），MPPI 保留供评估
- 遵循 CTDE 范式：集中式训练，分散式执行

### 持续保留项
- 已修改的文件完整列表
- 当前正在进行的任务目标
- 未完成的 TODO 项
- 最近一次测试的通过 / 失败状态

### 文件位置
- 项目根目录：`C:\Users\11242\Desktop\temp`
- 参考代码库：`C:\Users\11242\Desktop\temp\references\`（DIMA、m3w-marl、HiSSD、tdmpc2、dm_control）
- 论文文件：`C:\Users\11242\Desktop\temp\pdfs\`（DIMA.pdf、M3W-MARL.pdf、HiSSD.pdf、TD-MPC2.pdf）
- 项目文档：`C:\Users\11242\Desktop\temp\docs\`
