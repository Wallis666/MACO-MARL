"""经验回放缓冲区。

支持 n-step 回报计算、horizon 序列采样，
用于世界模型训练和 MPPI 规划。
"""
import numpy as np


class ReplayBuffer:
    """多智能体经验回放缓冲区。

    存储多环境并行收集的 transitions，支持：
    - n-step 折扣回报采样（用于 Critic 训练）
    - horizon 步序列采样（用于世界模型训练）

    :param n_agents: 智能体数量
    :param obs_dims: 各智能体观测维度列表
    :param act_dims: 各智能体动作维度列表
    :param share_obs_dim: 全局共享观测维度
    :param buffer_size: 缓冲区容量
    :param batch_size: 采样批次大小
    :param n_step: n-step 回报步数
    :param gamma: 折扣因子
    :param n_rollout_threads: 并行环境数量
    """

    def __init__(
        self,
        n_agents: int,
        obs_dims: list[int],
        act_dims: list[int],
        share_obs_dim: int,
        buffer_size: int = 1000000,
        batch_size: int = 1000,
        n_step: int = 20,
        gamma: float = 0.995,
        n_rollout_threads: int = 4,
    ) -> None:
        self.n_agents = n_agents
        self.obs_dims = obs_dims
        self.act_dims = act_dims
        self.share_obs_dim = share_obs_dim
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.n_step = n_step
        self.gamma = gamma
        self.n_rollout_threads = n_rollout_threads

        self.obs = [
            np.zeros((buffer_size, dim), dtype=np.float32)
            for dim in obs_dims
        ]
        self.next_obs = [
            np.zeros((buffer_size, dim), dtype=np.float32)
            for dim in obs_dims
        ]
        self.actions = [
            np.zeros((buffer_size, dim), dtype=np.float32)
            for dim in act_dims
        ]
        self.share_obs = np.zeros(
            (buffer_size, share_obs_dim), dtype=np.float32,
        )
        self.next_share_obs = np.zeros(
            (buffer_size, share_obs_dim), dtype=np.float32,
        )
        self.rewards = np.zeros((buffer_size, 1), dtype=np.float32)
        self.dones = np.zeros((buffer_size, 1), dtype=np.float32)
        self.terms = np.zeros((buffer_size, 1), dtype=np.float32)
        self.valid = [
            np.ones((buffer_size, 1), dtype=np.float32)
            for _ in range(n_agents)
        ]
        self.task_idx = np.zeros(buffer_size, dtype=np.int32)
        self.end_flag = np.zeros(buffer_size, dtype=np.float32)

        self.idx = 0
        self.cur_size = 0
        self._task_pool: dict[int, np.ndarray] = {}
        self._task_pool_dirty = True
        self._inserts_since_rebuild = 0

        self.gamma_buffer = np.zeros(n_step + 1, dtype=np.float32)
        self.gamma_buffer[0] = 1.0
        for i in range(1, n_step + 1):
            self.gamma_buffer[i] = self.gamma_buffer[i - 1] * gamma

    def insert(
        self,
        share_obs: np.ndarray,
        obs: list[np.ndarray],
        actions: list[np.ndarray],
        rewards: np.ndarray,
        dones: np.ndarray,
        terms: np.ndarray,
        valid: list[np.ndarray],
        next_share_obs: np.ndarray,
        next_obs: list[np.ndarray],
        task_idx: np.ndarray | None = None,
    ) -> None:
        """插入一步 transition（来自多个并行环境）。

        :param share_obs: 全局观测，形状 (n_threads, share_obs_dim)
        :param obs: 各智能体观测，列表中每项 (n_threads, obs_dim_i)
        :param actions: 各智能体动作，列表中每项 (n_threads, act_dim_i)
        :param rewards: 平均奖励，形状 (n_threads, 1)
        :param dones: 终止标志，形状 (n_threads, 1)
        :param terms: 真终止标志，形状 (n_threads, 1)
        :param valid: 有效 transition 标志，列表中每项 (n_threads, 1)
        :param next_share_obs: 下一步全局观测
        :param next_obs: 下一步各智能体观测
        :param task_idx: 每个环境的任务索引，形状 (n_threads,) 或 None
        """
        length = share_obs.shape[0]
        indices = np.arange(self.idx, self.idx + length) % self.buffer_size

        self.share_obs[indices] = share_obs
        self.next_share_obs[indices] = next_share_obs
        self.rewards[indices] = rewards
        self.dones[indices] = dones
        self.terms[indices] = terms

        if task_idx is not None:
            self.task_idx[indices] = task_idx

        for i in range(self.n_agents):
            self.obs[i][indices] = obs[i]
            self.next_obs[i][indices] = next_obs[i]
            self.actions[i][indices] = actions[i]
            self.valid[i][indices] = valid[i]

        self.idx = (self.idx + length) % self.buffer_size
        self.cur_size = min(self.cur_size + length, self.buffer_size)
        self._inserts_since_rebuild += 1
        if self._inserts_since_rebuild >= 1000:
            self._task_pool_dirty = True
            self._inserts_since_rebuild = 0
        self._update_end_flag()

    def _update_end_flag(self) -> None:
        """更新 episode 边界标记。"""
        self.end_flag = self.dones.squeeze(-1).copy()
        unfinished = (
            np.arange(self.idx - self.n_rollout_threads, self.idx)
            % self.buffer_size
        )
        self.end_flag[unfinished] = 1.0

    def _next(self, indices: np.ndarray) -> np.ndarray:
        """获取下一步索引（跳过 episode 边界）。

        :param indices: 当前索引
        :return: 下一步索引
        """
        step = (1 - self.end_flag[indices]) * self.n_rollout_threads
        return (indices + step.astype(int)) % self.cur_size

    def sample(self) -> dict[str, np.ndarray | list[np.ndarray]]:
        """采样一个 batch（含 n-step 回报）。

        :return: 字典格式的采样数据
        """
        indices = np.random.randint(0, self.cur_size, self.batch_size)

        n_step_indices = [indices]
        for _ in range(self.n_step - 1):
            n_step_indices.append(self._next(n_step_indices[-1]))

        nstep_reward = np.zeros(
            (self.batch_size, 1), dtype=np.float32,
        )
        gammas = np.full(self.batch_size, self.n_step, dtype=np.int32)

        for n in range(self.n_step - 1, -1, -1):
            now = n_step_indices[n]
            at_end = self.end_flag[now] > 0
            gammas[at_end] = n + 1
            nstep_reward[at_end] = 0.0
            nstep_reward = (
                self.rewards[now] + self.gamma * nstep_reward
            )

        nstep_gamma = self.gamma_buffer[gammas].reshape(-1, 1)
        final_idx = n_step_indices[-1]

        return {
            "share_obs": self.share_obs[indices],
            "obs": [self.obs[i][indices] for i in range(self.n_agents)],
            "actions": [
                self.actions[i][indices] for i in range(self.n_agents)
            ],
            "rewards": self.rewards[indices],
            "dones": self.dones[indices],
            "nstep_reward": nstep_reward,
            "nstep_gamma": nstep_gamma,
            "nstep_term": self.terms[final_idx],
            "nstep_next_share_obs": self.next_share_obs[final_idx],
            "nstep_next_obs": [
                self.next_obs[i][final_idx]
                for i in range(self.n_agents)
            ],
            "next_obs": [
                self.next_obs[i][indices] for i in range(self.n_agents)
            ],
            "task_idx": self.task_idx[indices],
        }

    def sample_horizon(
        self,
        horizon: int,
    ) -> dict[str, np.ndarray | list[np.ndarray]]:
        """采样 horizon 步连续序列（用于世界模型训练）。

        :param horizon: 序列长度
        :return: 字典格式的序列数据
        """
        t0 = np.random.randint(0, self.cur_size, self.batch_size)

        indices = [t0]
        for _ in range(horizon - 1):
            indices.append(self._next(indices[-1]))

        # 向量化：将 indices 列表转为 (horizon, batch) 数组后一次性索引
        idx_arr = np.array(indices)  # (horizon, batch_size)
        h_obs = [self.obs[i][idx_arr] for i in range(self.n_agents)]
        h_actions = [
            self.actions[i][idx_arr] for i in range(self.n_agents)
        ]
        h_next_obs = [
            self.next_obs[i][idx_arr] for i in range(self.n_agents)
        ]
        h_rewards = self.rewards[idx_arr]

        return {
            "obs": h_obs,
            "actions": h_actions,
            "next_obs": h_next_obs,
            "rewards": h_rewards,
            "task_idx": self.task_idx[t0],
        }

    def sample_context(
        self,
        n_context: int,
        batch_size: int | None = None,
    ) -> dict[str, np.ndarray | list[np.ndarray]]:
        """采样上下文 transitions 用于上下文编码器训练。

        对每个 batch 样本，从同一任务中随机采样 n_context 条 transition。

        :param n_context: 每个样本的上下文 transition 数量 K
        :param batch_size: 批次大小，默认使用 self.batch_size
        :return: 字典，包含上下文观测、动作、奖励、下一步观测、task_idx
        """
        if batch_size is None:
            batch_size = self.batch_size

        # 缓存 per-task 索引映射，每 1000 次 insert 或首次调用时重建
        # （新插入的少量数据不影响采样分布，无需每步重建）
        if self._task_pool_dirty or not self._task_pool:
            valid_idx = np.arange(self.cur_size)
            task_ids_all = self.task_idx[:self.cur_size]
            self._task_pool = {
                int(tid): valid_idx[task_ids_all == tid]
                for tid in np.unique(task_ids_all)
            }
            self._task_pool_dirty = False
        task_pool = self._task_pool
        unique_tasks = np.array(list(task_pool.keys()))

        # 随机为每个样本分配任务
        sample_tasks = np.random.choice(unique_tasks, size=batch_size)

        # 向量化：按任务分组批量采样，避免 Python for 循环
        ctx_indices = np.empty(
            (batch_size, n_context), dtype=np.int64,
        )
        for tid in unique_tasks:
            mask = sample_tasks == tid
            count = mask.sum()
            if count == 0:
                continue
            pool = task_pool[int(tid)]
            # 一次性生成该任务所有样本的随机索引
            rand_idx = np.random.randint(
                0, len(pool), size=(int(count), n_context),
            )
            ctx_indices[mask] = pool[rand_idx]

        # 收集数据
        ctx_obs = [
            self.obs[agent][ctx_indices]
            for agent in range(self.n_agents)
        ]
        ctx_actions = [
            self.actions[agent][ctx_indices]
            for agent in range(self.n_agents)
        ]
        ctx_rewards = self.rewards[ctx_indices]
        ctx_next_obs = [
            self.next_obs[agent][ctx_indices]
            for agent in range(self.n_agents)
        ]

        return {
            "ctx_obs": ctx_obs,
            "ctx_actions": ctx_actions,
            "ctx_rewards": ctx_rewards,
            "ctx_next_obs": ctx_next_obs,
            "task_idx": sample_tasks,
        }

    def can_sample(self) -> bool:
        """是否可以采样。

        :return: 缓冲区中数据是否足够
        """
        return self.cur_size >= self.batch_size

    @property
    def size(self) -> int:
        """当前缓冲区已使用量。"""
        return self.cur_size
