"""Actor 策略网络。

高斯策略 Actor，每个智能体独立维护一个策略网络。
输入潜在状态，输出 tanh 压缩后的连续动作。
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class GaussianPolicy(nn.Module):
    """高斯策略网络（Squashed Gaussian）。

    输出均值和对数标准差，通过重参数化采样并用 tanh 压缩到动作范围。

    :param latent_dim: 潜在状态维度（输入维度）
    :param action_dim: 动作维度
    :param hidden_sizes: 隐藏层维度列表
    :param log_std_min: 对数标准差下界
    :param log_std_max: 对数标准差上界
    :param device: 设备
    """

    def __init__(
        self,
        latent_dim: int,
        action_dim: int,
        hidden_sizes: list[int] | None = None,
        log_std_min: float = -10.0,
        log_std_max: float = 2.0,
        device: str = "cpu",
    ) -> None:
        super().__init__()
        if hidden_sizes is None:
            hidden_sizes = [128, 128]
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.action_dim = action_dim

        layers: list[nn.Module] = []
        dims = [latent_dim] + hidden_sizes
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers).to(device)
        self.mu_layer = nn.Linear(hidden_sizes[-1], action_dim).to(device)
        self.log_std_layer = nn.Linear(
            hidden_sizes[-1], action_dim,
        ).to(device)

    def forward(
        self,
        z: torch.Tensor,
        stochastic: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """前向传播，生成动作和对数概率。

        :param z: 潜在状态，形状 (batch, latent_dim)
        :param stochastic: 是否随机采样
        :return: (action, log_prob)
            - action: 形状 (batch, action_dim)，范围 [-1, 1]
            - log_prob: 形状 (batch, 1)，随机模式下返回
        """
        h = self.net(z)
        mu = self.mu_layer(h)
        log_std = self.log_std_layer(h)
        log_std = self.log_std_min + 0.5 * (
            self.log_std_max - self.log_std_min
        ) * (torch.tanh(log_std) + 1)

        if stochastic:
            std = log_std.exp()
            eps = torch.randn_like(mu)
            pre_tanh = mu + eps * std
        else:
            pre_tanh = mu
            eps = torch.zeros_like(mu)

        action = torch.tanh(pre_tanh)

        if stochastic:
            log_prob = self._log_prob(eps, log_std, action)
        else:
            log_prob = None

        return action, log_prob

    def _log_prob(
        self,
        eps: torch.Tensor,
        log_std: torch.Tensor,
        squashed_action: torch.Tensor,
    ) -> torch.Tensor:
        """计算对数概率（含 tanh 雅可比修正）。

        :param eps: 标准正态采样噪声
        :param log_std: 对数标准差
        :param squashed_action: tanh 后的动作
        :return: 对数概率，形状 (batch, 1)
        """
        gaussian_log_prob = (
            -0.5 * eps.pow(2) - log_std - 0.5 * math.log(2 * math.pi)
        ).sum(dim=-1, keepdim=True)
        squash_correction = torch.log(
            torch.clamp(1 - squashed_action.pow(2), min=1e-6),
        ).sum(dim=-1, keepdim=True)
        return gaussian_log_prob - squash_correction


class WorldModelActor:
    """世界模型 Actor 封装。

    管理单个智能体的策略网络及其优化器。

    :param latent_dim: 潜在状态维度
    :param action_dim: 动作维度
    :param hidden_sizes: 隐藏层维度
    :param lr: 学习率
    :param log_std_min: 对数标准差下界
    :param log_std_max: 对数标准差上界
    :param device: 设备
    """

    def __init__(
        self,
        latent_dim: int,
        action_dim: int,
        hidden_sizes: list[int] | None = None,
        lr: float = 3e-4,
        log_std_min: float = -10.0,
        log_std_max: float = 2.0,
        device: str = "cpu",
    ) -> None:
        self.policy = GaussianPolicy(
            latent_dim=latent_dim,
            action_dim=action_dim,
            hidden_sizes=hidden_sizes,
            log_std_min=log_std_min,
            log_std_max=log_std_max,
            device=device,
        )
        self.optimizer = torch.optim.Adam(
            self.policy.parameters(), lr=lr,
        )
        self.device = device

    def get_actions(
        self,
        z: torch.Tensor,
        stochastic: bool = True,
    ) -> torch.Tensor:
        """获取动作（不带梯度）。

        :param z: 潜在状态
        :param stochastic: 是否随机
        :return: 动作
        """
        with torch.no_grad():
            action, _ = self.policy(z, stochastic=stochastic)
        return action

    def get_actions_with_logprobs(
        self,
        z: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """获取动作和对数概率（带梯度）。

        :param z: 潜在状态
        :return: (action, log_prob)
        """
        return self.policy(z, stochastic=True)

    def turn_on_grad(self) -> None:
        """启用策略参数梯度。"""
        for p in self.policy.parameters():
            p.requires_grad_(True)

    def turn_off_grad(self) -> None:
        """禁用策略参数梯度。"""
        for p in self.policy.parameters():
            p.requires_grad_(False)
