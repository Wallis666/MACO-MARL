# 项目需求文档

## 1. 项目背景

本项目为本科毕业设计，研究方向为**多任务多智能体强化学习（Multi-Task Multi-Agent Reinforcement Learning, MT-MARL）**。

## 2. 核心目标

构建一个基于世界模型的 MT-MARL 框架，使多智能体系统在多个已知任务上联合训练后，能够在**未见过的新任务**上实现 **few-shot 快速适应**（仅凭几条轨迹，无需梯度更新）。

## 3. 问题定义

### 3.1 环境与智能体

- 仿真环境：**MA-MuJoCo**（Multi-Agent MuJoCo）。
- 智能体拆分：机器人的关节被拆分为多个智能体协同控制。
- 环境接口：连续观测空间、连续动作空间。

### 3.2 任务体系

| 类别 | 任务示例 | 说明 |
|------|---------|------|
| 训练任务（meta-train） | cheetah-run, cheetah-run-backward, walker-run, walker-walk, ... | 联合训练使用 |
| 评估任务（meta-test） | 待定（hold-out 的未见任务） | few-shot 适应评估 |

### 3.3 训练阶段

- 在多个已知任务上进行**并行多任务训练**。
- 世界模型学习跨任务共享的环境动力学和奖励预测。
- 在世界模型想象空间中训练策略（imagination-based）或进行规划（MPPI）。
- 多任务联合训练性能达到单任务专家的 70%+ 水平。

### 3.4 适应阶段（few-shot）

- 在未见过的任务上，利用**少量轨迹样本**（如 5-10 条 episode）进行快速适应。
- 无需梯度更新，通过上下文推断实现适应。
- 适应后性能显著优于随机策略和直接迁移。

## 4. 技术方案

### 4.1 世界模型

- 采用 MoE 架构（参考 M3W-MARL），SoftMoE 动力学 + SparseMoE 奖励。
- 先用 Dense 基线验证 pipeline 正确性，再升级为 MoE。
- 世界模型在潜空间中运行，不含解码器。

### 4.2 策略优化 / 规划

- 可选方案 A：想象空间训练（Dreamer 风格）+ SAC / MAPPO
- 可选方案 B：MPPI 规划器（TD-MPC 风格），Actor 仅作为候选动作生成器
- 遵循 CTDE 范式：集中式训练，分散式执行

### 4.3 Few-shot 适应（核心创新）

- 方案 A：上下文编码器 — 将 few-shot 轨迹编码为 task embedding，注入世界模型各组件
- 方案 B：MoE 路由适应 — 用 few-shot 轨迹推断新任务的路由权重，不更新专家参数
- 两种方案可组合使用

## 5. 参考代码库

| 代码库 | 位置 | 说明 |
|--------|------|------|
| DIMA | `references/DIMA/` | 基于扩散模型的多智能体世界模型（NeurIPS 2025） |
| M3W-MARL | `references/m3w-marl/` | 基于 MoE 的多任务世界模型（NeurIPS 2025） |
| dm_control | `references/dm_control/` | DeepMind Control Suite，参考任务定义和奖励设计 |

## 6. 评估指标

| 指标 | 说明 |
|------|------|
| 多任务收敛性 | 每个任务的 return 不低于单任务专家的 60-70% |
| 世界模型精度 | dynamics / reward 预测误差持续下降 |
| 适应效率 | 在 unseen 任务上达到阈值性能所需的轨迹数量 |
| 适应后表现 | few-shot 适应后在 unseen 任务上的累计奖励 |
| 样本效率 | 优于 model-free 基线（MAPPO、HAPPO 等） |

## 7. 约束

- 目标环境仅限连续动作空间（受 MPPI 规划器限制）。
- 参考代码库（`references/` 下）只读，不得修改。
- 所有 Python 命令在 `maco` conda 环境中执行。

## 8. 扩展性规划

当前仅使用 MA-MuJoCo，后续可能扩展到：

- **Bi-DexHands**：灵巧手操作环境
- 其他多智能体协作环境

代码架构应预留环境接口的抽象层，使环境切换不需大幅改动上层逻辑。
