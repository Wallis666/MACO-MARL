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

### 20260402-190000
- **类型**: refactor
- **涉及文件**: `src/runner/trainer.py`, `src/envs/mamujoco.py`, `src/config/default.json`, `train.py`
- **摘要**: 切换为路线 2（Dreamer 风格）：训练时用 Actor 采样替代 MPPI 规划，大幅降低每步推理开销（2304 次→1 次）。添加 `_act` 方法、`SubprocVectorMAMuJoCoEnv` 多进程环境、tqdm 进度条。MPPI 保留供评估使用。配置更新：batch_size=4096, n_rollout_threads=16, update_per_train=4, log_interval=250。

### 20260402-150000
- **类型**: feat
- **涉及文件**: `src/` 全部新建文件, `train.py`, `tests/test_shapes.py`, `src/config/default.json`
- **摘要**: 阶段 0 完整框架搭建。实现了 Dense 世界模型（Encoder、Dynamics、Reward）、高斯 Actor、分布式 Twin-Q Critic、MPPI 规划器、经验回放 Buffer、主训练循环。包含 TwoHotProcessor/SimNorm/RunningScale 等基础工具。19 项单元测试全部通过。
