"""MPPI 规划器。

基于 CEM（Cross-Entropy Method）的模型预测路径积分规划，
在世界模型的潜在空间中搜索最优动作序列。
"""
import torch
import torch.nn.functional as F
from einops import rearrange


class MPPIPlanner:
    """MPPI 规划器。

    使用 CEM 风格迭代优化在世界模型想象空间中搜索最优动作。

    :param n_agents: 智能体数量
    :param act_dims: 各智能体动作维度列表
    :param latent_dim: 潜在状态维度
    :param horizon: 规划步长
    :param iterations: CEM 迭代次数
    :param num_samples: 采样数量
    :param num_pi_trajs: 策略轨迹数量
    :param num_elites: 精英样本数量
    :param max_std: 动作标准差上界
    :param min_std: 动作标准差下界
    :param temperature: softmax 温度系数
    :param gamma: 折扣因子
    :param device: 设备
    """

    def __init__(
        self,
        n_agents: int,
        act_dims: list[int],
        latent_dim: int,
        horizon: int = 3,
        iterations: int = 6,
        num_samples: int = 128,
        num_pi_trajs: int = 8,
        num_elites: int = 16,
        max_std: float = 1.0,
        min_std: float = 0.05,
        temperature: float = 0.5,
        gamma: float = 0.995,
        device: str = "cpu",
    ) -> None:
        self.n_agents = n_agents
        self.act_dims = act_dims
        self.latent_dim = latent_dim
        self.horizon = horizon
        self.iterations = iterations
        self.num_samples = num_samples
        self.num_pi_trajs = num_pi_trajs
        self.num_elites = num_elites
        self.max_std = max_std
        self.min_std = min_std
        self.temperature = temperature
        self.gamma = gamma
        self.device = device

        self.running_mean: list[torch.Tensor | None] = [
            None for _ in range(n_agents)
        ]

    @torch.no_grad()
    def plan(
        self,
        zs: list[torch.Tensor],
        t0: list[bool],
        dynamics_model: torch.nn.Module,
        reward_model: torch.nn.Module,
        reward_processor: object,
        actors: list[object],
        critic: object,
    ) -> list[torch.Tensor]:
        """执行 MPPI 规划。

        :param zs: 各智能体的潜在状态，列表中每项 (n_threads, latent_dim)
        :param t0: 每个环境是否为 episode 开头
        :param dynamics_model: 动力学模型
        :param reward_model: 奖励模型
        :param reward_processor: TwoHot 编解码器
        :param actors: Actor 列表
        :param critic: Critic
        :return: 各智能体的动作列表，每项 (n_threads, act_dim)
        """
        n_threads = zs[0].shape[0]

        act_mean = [
            torch.zeros(
                self.horizon, n_threads, self.act_dims[i],
                device=self.device,
            )
            for i in range(self.n_agents)
        ]
        act_std = [
            torch.full(
                (self.horizon, n_threads, self.act_dims[i]),
                self.max_std,
                device=self.device,
            )
            for i in range(self.n_agents)
        ]

        for i in range(self.n_agents):
            if self.running_mean[i] is not None:
                for env_idx in range(n_threads):
                    if not t0[env_idx]:
                        act_mean[i][:-1, env_idx] = (
                            self.running_mean[i][1:, env_idx]
                        )

        pi_actions = None
        if self.num_pi_trajs > 0:
            pi_actions = self._generate_pi_trajs(
                zs, dynamics_model, actors,
            )

        actions = [
            torch.zeros(
                self.horizon,
                n_threads,
                self.num_samples,
                self.act_dims[i],
                device=self.device,
            )
            for i in range(self.n_agents)
        ]

        out_actions = [
            torch.zeros(n_threads, self.act_dims[i], device=self.device)
            for i in range(self.n_agents)
        ]

        for iteration in range(self.iterations):
            for i in range(self.n_agents):
                n_rand = self.num_samples - self.num_pi_trajs
                rand_actions = (
                    act_mean[i].unsqueeze(2)
                    + act_std[i].unsqueeze(2)
                    * torch.randn(
                        self.horizon,
                        n_threads,
                        n_rand,
                        self.act_dims[i],
                        device=self.device,
                    )
                ).clamp(-1, 1)

                if pi_actions is not None:
                    actions[i] = torch.cat(
                        [pi_actions[i], rand_actions], dim=2,
                    )
                else:
                    actions[i] = rand_actions

            g_returns = self._estimate_value(
                zs, actions, dynamics_model, reward_model,
                reward_processor, actors, critic,
            )

            for i in range(self.n_agents):
                value = g_returns.squeeze(-1)

                _, elite_idx = torch.topk(
                    value, self.num_elites, dim=-1,
                )

                elite_idx_expand = elite_idx.unsqueeze(0).unsqueeze(
                    -1,
                ).expand(
                    self.horizon, -1, -1, self.act_dims[i],
                )
                elite_actions = torch.gather(
                    actions[i], 2, elite_idx_expand,
                )

                elite_values = torch.gather(value, -1, elite_idx)
                max_val = elite_values.max(dim=-1, keepdim=True)[0]
                score = torch.exp(
                    self.temperature * (elite_values - max_val),
                )
                score = score / (score.sum(dim=-1, keepdim=True) + 1e-8)
                score_w = score.unsqueeze(0).unsqueeze(-1)

                act_mean[i] = (score_w * elite_actions).sum(dim=2)
                diff = elite_actions - act_mean[i].unsqueeze(2)
                act_std[i] = torch.sqrt(
                    (score_w * diff.pow(2)).sum(dim=2) + 1e-6,
                ).clamp(self.min_std, self.max_std)

                if iteration == self.iterations - 1:
                    for env_idx in range(n_threads):
                        idx = torch.multinomial(
                            score[env_idx], 1,
                        ).item()
                        out_actions[i][env_idx] = (
                            elite_actions[0, env_idx, idx]
                        )

        for i in range(self.n_agents):
            self.running_mean[i] = act_mean[i].clone()

        return out_actions

    def _generate_pi_trajs(
        self,
        zs: list[torch.Tensor],
        dynamics_model: torch.nn.Module,
        actors: list[object],
    ) -> list[torch.Tensor]:
        """用当前策略生成轨迹样本。

        :param zs: 各智能体当前潜在状态
        :param dynamics_model: 动力学模型
        :param actors: Actor 列表
        :return: 策略动作序列
        """
        n_threads = zs[0].shape[0]
        pi_actions = [
            torch.zeros(
                self.horizon,
                n_threads,
                self.num_pi_trajs,
                self.act_dims[i],
                device=self.device,
            )
            for i in range(self.n_agents)
        ]

        cur_zs = [
            z.unsqueeze(1).expand(-1, self.num_pi_trajs, -1)
            for z in zs
        ]

        for t in range(self.horizon):
            for i in range(self.n_agents):
                z_flat = cur_zs[i].reshape(-1, self.latent_dim)
                a_flat = actors[i].get_actions(z_flat, stochastic=True)
                pi_actions[i][t] = a_flat.reshape(
                    n_threads, self.num_pi_trajs, self.act_dims[i],
                )

            if t < self.horizon - 1:
                z_joint = torch.stack(cur_zs, dim=2)
                a_joint = torch.stack(
                    [
                        pi_actions[i][t]
                        for i in range(self.n_agents)
                    ],
                    dim=2,
                )
                batch = n_threads * self.num_pi_trajs
                z_in = z_joint.reshape(batch, self.n_agents, -1)
                a_in = a_joint.reshape(batch, self.n_agents, -1)
                z_next = dynamics_model.predict(z_in, a_in)
                for i in range(self.n_agents):
                    cur_zs[i] = z_next[:, i].reshape(
                        n_threads, self.num_pi_trajs, -1,
                    )

        return pi_actions

    def _estimate_value(
        self,
        zs: list[torch.Tensor],
        actions: list[torch.Tensor],
        dynamics_model: torch.nn.Module,
        reward_model: torch.nn.Module,
        reward_processor: object,
        actors: list[object],
        critic: object,
    ) -> torch.Tensor:
        """估计动作序列的累计回报。

        :param zs: 各智能体初始潜在状态
        :param actions: 动作序列
        :param dynamics_model: 动力学模型
        :param reward_model: 奖励模型
        :param reward_processor: TwoHot 编解码器
        :param actors: Actor 列表
        :param critic: Critic
        :return: 累计回报，形状 (n_threads, num_samples, 1)
        """
        n_threads = zs[0].shape[0]
        num_samples = actions[0].shape[2]

        cur_zs = [
            z.unsqueeze(1).expand(-1, num_samples, -1)
            for z in zs
        ]

        g_returns = torch.zeros(
            n_threads, num_samples, 1, device=self.device,
        )

        for t in range(self.horizon):
            z_joint = torch.stack(cur_zs, dim=2)
            a_joint = torch.stack(
                [actions[i][t] for i in range(self.n_agents)], dim=2,
            )

            batch = n_threads * num_samples
            z_in = z_joint.reshape(batch, self.n_agents, -1)
            a_in = a_joint.reshape(batch, self.n_agents, -1)

            r_logits = reward_model.predict(z_in, a_in)
            r_value = reward_processor.logits_to_scalar(r_logits)
            r_value = r_value.reshape(n_threads, num_samples, 1)

            g_returns = g_returns + (self.gamma ** t) * r_value

            z_next = dynamics_model.predict(z_in, a_in)
            for i in range(self.n_agents):
                cur_zs[i] = z_next[:, i].reshape(
                    n_threads, num_samples, -1,
                )

        joint_z = torch.cat(cur_zs, dim=-1).reshape(
            batch, -1,
        )
        joint_a_list = []
        for i in range(self.n_agents):
            z_flat = cur_zs[i].reshape(batch, -1)
            a_flat = actors[i].get_actions(z_flat, stochastic=True)
            joint_a_list.append(a_flat)
        joint_a = torch.cat(joint_a_list, dim=-1)

        horizon_q = critic.get_target_values(joint_z, joint_a)
        horizon_q = horizon_q.reshape(n_threads, num_samples, 1)

        g_returns = g_returns + (self.gamma ** self.horizon) * horizon_q

        return g_returns
