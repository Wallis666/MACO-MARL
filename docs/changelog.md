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
