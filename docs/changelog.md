# 变更日志（Changelog）

记录项目中每次有意义的代码修改。每条记录按 `YYYYMMDD-HHMMSS` 格式标注时间戳（年月日-时分秒）。

## 格式规范

每条变更记录包含以下字段：

- **时间戳**：`YYYYMMDD-HHMMSS`
- **类型**：`feat`（新功能）| `fix`（修复）| `refactor`（重构）| `docs`（文档）| `test`（测试）| `chore`（杂项）| `exp`（实验）
- **涉及文件**：列出修改的文件路径
- **摘要**：用一两句中文描述改了什么、为什么改

---

<!-- 新记录插入到此行下方，最新的在最上面 -->

### 20260403-220000
- **类型**: fix
- **涉及文件**: `src/runner/trainer.py`, `scripts/evaluate_fewshot.py`, `tests/test_context_encoder.py`
- **摘要**: 修复上下文编码器的 moving target 问题。为任务嵌入表添加 EMA 副本作为上下文编码器的稳定训练目标（类似 Critic 的 target network），防止因在线嵌入不断变化导致 ctx_loss 无法收敛。评估脚本加载 EMA 嵌入用于余弦相似度比较。新增 3 个 EMA 相关测试，总计 75 个测试通过。

### 20260404-030000
- **类型**: feat
- **涉及文件**: `scripts/evaluate_fewshot.py`（新建）, `src/config/tasks.py`, `tests/test_evaluate_fewshot.py`（新建）
- **摘要**: 阶段 4 — Few-shot 适应评估。新增 held-out 评估任务（`cheetah_run_slow` 5m/s、`cheetah_walk` 2m/s）。创建评估脚本支持 4 种模式：few-shot（随机 demo → 上下文编码器 → Actor 推理）、few-shot-mppi（MPPI 规划器）、oracle（训练任务嵌入查表，上限基线）、random（随机策略，下限基线）。输出含 per-episode 回报、demo 奖励统计、与训练任务嵌入的余弦相似度。72 个测试全部通过。

### 20260404-010000
- **类型**: feat
- **涉及文件**: `src/models/context_encoder.py`（新建）, `src/buffer/replay_buffer.py`, `src/runner/trainer.py`, `src/config/multitask.json`, `tests/test_context_encoder.py`（新建）
- **摘要**: 阶段 3 — 上下文编码器。实现 PEARL 风格上下文编码器（MLP + 均值池化 + L2 归一化），从 K 条演示 transitions 推断任务嵌入向量，训练目标为 MSE 对齐已学会的任务嵌入（detach 目标）。Buffer 新增 `sample_context()` 按任务分组采样。Trainer 集成独立优化器和 `_context_encoder_train()` 训练步。检查点保存 context_encoder 和 ctx_optimizer 状态。`multitask.json` 新增 `context_encoder` 配置段。64 个测试全部通过。

### 20260403-230000
- **类型**: fix
- **涉及文件**: `scripts/evaluate.py`
- **摘要**: 修复评估脚本兼容阶段 2 任务条件化架构。去掉 one-hot 观测拼接逻辑，改为从检查点加载 `TaskEmbeddingTable`，通过 `task_emb` 参数传递给 Encoder 和 Actor。`load_models` 新增 `task_dim` 感知，正确构造 `MLPEncoder(task_dim=32)`。

### 20260403-220000
- **类型**: feat
- **涉及文件**: `src/models/task_embedding.py`（新建）, `src/models/encoder.py`, `src/models/dynamics.py`, `src/models/reward.py`, `src/algorithms/actor.py`, `src/algorithms/critic.py`, `src/algorithms/planner.py`, `src/buffer/replay_buffer.py`, `src/envs/mamujoco.py`, `src/runner/trainer.py`, `src/config/multitask.json`, `tests/test_shapes.py`, `tests/test_multitask.py`
- **摘要**: 阶段 2 — 任务条件化世界模型。参考 TD-MPC2，实现可学习任务嵌入（`nn.Embedding` + `max_norm=1`）通过拼接（concatenation）注入所有模型组件（Encoder/Dynamics/Reward/Actor/Critic/Planner）。去掉环境层 one-hot 观测拼接，改为模型层嵌入拼接。Buffer 新增 `task_idx` 存储。Trainer 集成 TaskEmbeddingTable，训练时查表获取嵌入传递给所有组件。`multitask.json` 新增 `task_dim=32`。单任务模式向后兼容（`task_dim=0` 时跳过拼接）。55 个测试全部通过。

### 20260403-180000
- **类型**: docs
- **涉及文件**: `CLAUDE.md`, `docs/requirements.md`, `docs/changelog.md`
- **摘要**: 新增 TD-MPC2 参考代码库（`references/tdmpc2/`，ICLR 2024 可扩展多任务世界模型）和对应论文。所有 PDF 迁移至 `pdfs/` 目录，更新论文路径。更新参考代码库列表和关键路径（world_model.py、layers.py、multitask.py 等）。

### 20260403-160000
- **类型**: docs
- **涉及文件**: `CLAUDE.md`, `docs/requirements.md`, `docs/changelog.md`
- **摘要**: 技术路线重大调整：放弃 MoE 路线，改用 Dense 世界模型 + 任务条件化（可学习任务嵌入 + FiLM 条件化 + 上下文编码器）。新增 HiSSD 参考代码库和论文。更新开发路线图（阶段 0-1 已完成，阶段 2-5 按新方案调整）。更新需求文档技术方案章节，详细描述多任务泛化机制和 few-shot 适应流程。

### 20260403-120000
- **类型**: feat
- **涉及文件**: `scripts/evaluate.py`
- **摘要**: 评估脚本添加 `--task` 参数支持多任务配置。新增 `_resolve_env_config` 函数从多任务配置和 TASK_REGISTRY 解析环境参数（含 task_idx/n_tasks）。推理时自动拼接 one-hot 任务编码到观测，确保与多任务训练时的 obs_dim 一致。单任务配置向后兼容。

### 20260403-010000
- **类型**: feat
- **涉及文件**: `src/config/tasks.py`(新建), `src/envs/mamujoco.py`, `src/runner/trainer.py`, `src/config/multitask.json`(新建), `tests/test_multitask.py`(新建)
- **摘要**: 阶段 1a 多任务基础设施。新建任务注册表（tolerance 函数 + HalfCheetah run/run_backwards 奖励函数，姿态要求融入 reward_fn 作为乘法因子）。新建 MultiTaskVectorMAMuJoCoEnv 多任务向量化环境（one-hot 任务编码嵌入观测、per-worker 任务分配）。改造 Trainer 支持多任务配置检测、per-task episode tracking 和 TensorBoard 日志。单任务向后兼容。22 项多任务测试全部通过。

### 20260402-233000
- **类型**: feat
- **涉及文件**: `src/envs/mamujoco.py`, `src/runner/trainer.py`, `src/config/default.json`
- **摘要**: 添加乘法式高度奖励缩放（healthy_height），防止 HalfCheetah 跪地/趴下跑步。参考 dm_control Walker 的设计，使用高斯 tolerance 函数：torso z >= 0.4m 时奖励不变，低于阈值时奖励平滑衰减（0.3m→56%，0.2m→10%，0.1m→0.6%）。跪着跑基本拿不到分，迫使策略学习正常站立跑步。

### 20260402-230000
- **类型**: feat
- **涉及文件**: `src/envs/mamujoco.py`, `src/runner/trainer.py`, `src/config/default.json`
- **摘要**: 添加动作变化率惩罚（action rate penalty），惩罚相邻时间步之间动作的剧烈变化 `||a_t - a_{t-1}||²`，鼓励策略输出平滑连续的控制信号。参数 `action_rate_penalty` 默认 0.1，参考 Isaac Gym legged_gym 的做法。reset 后首步不施加惩罚。

### 20260402-220000
- **类型**: feat
- **涉及文件**: `src/envs/mamujoco.py`, `src/runner/trainer.py`, `src/config/default.json`
- **摘要**: 添加朝向惩罚机制，防止 HalfCheetah 翻转后以头着地跑步。通过 `cos(rooty)` 计算身体朝向，翻转时施加惩罚。参数 `orientation_penalty` 可通过配置控制，默认 1.0。贯穿 MAMuJoCoEnv、SubprocVectorMAMuJoCoEnv、VectorMAMuJoCoEnv 三层环境封装。

### 20260402-190000
- **类型**: refactor
- **涉及文件**: `src/runner/trainer.py`, `src/envs/mamujoco.py`, `src/config/default.json`, `train.py`
- **摘要**: 切换为路线 2（Dreamer 风格）：训练时用 Actor 采样替代 MPPI 规划，大幅降低每步推理开销（2304 次→1 次）。添加 `_act` 方法、`SubprocVectorMAMuJoCoEnv` 多进程环境、tqdm 进度条。MPPI 保留供评估使用。配置更新：batch_size=4096, n_rollout_threads=16, update_per_train=4, log_interval=250。

### 20260402-150000
- **类型**: feat
- **涉及文件**: `src/` 全部新建文件, `train.py`, `tests/test_shapes.py`, `src/config/default.json`
- **摘要**: 阶段 0 完整框架搭建。实现了 Dense 世界模型（Encoder、Dynamics、Reward）、高斯 Actor、分布式 Twin-Q Critic、MPPI 规划器、经验回放 Buffer、主训练循环。包含 TwoHotProcessor/SimNorm/RunningScale 等基础工具。19 项单元测试全部通过。
