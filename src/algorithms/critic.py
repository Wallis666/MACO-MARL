"""Critic 价值网络。

基于分布式 Q-Network（DisRegQNet），输出 Q 值的 two-hot 分布 logits。
使用 Twin Q 架构避免过估计。
多任务模式下通过拼接任务嵌入实现任务条件化。
"""
import copy

import torch
import torch.nn as nn

from src.models.task_embedding import cat_task_emb
from src.models.utils import RunningScale, TwoHotProcessor, create_mlp


class DisRegQNet(nn.Module):
    """分布式 Q-Network。

    输出 Q 值的分布 logits，配合 TwoHotProcessor 使用。
    多任务时输入末尾拼接任务嵌入。

    :param input_dim: 输入维度（联合潜在状态 + 联合动作 + task_dim）
    :param num_bins: Q 值离散化 bin 数量
    :param hidden_sizes: 隐藏层维度列表
    :param device: 设备
    """

    def __init__(
        self,
        input_dim: int,
        num_bins: int = 101,
        hidden_sizes: list[int] | None = None,
        device: str = "cpu",
    ) -> None:
        super().__init__()
        if hidden_sizes is None:
            hidden_sizes = [512, 512]
        self.mlp = create_mlp(
            in_dim=input_dim,
            mlp_dims=hidden_sizes,
            out_dim=num_bins,
            device=device,
        )
        self.mlp[-1].weight.data.fill_(0)
        self.mlp[-1].bias.data.fill_(0)

    def forward(
        self,
        joint_z: torch.Tensor,
        joint_a: torch.Tensor,
        task_emb: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """前向传播。

        :param joint_z: 联合潜在状态，形状 (batch, n_agents * latent_dim)
        :param joint_a: 联合动作，形状 (batch, n_agents * action_dim)
        :param task_emb: 任务嵌入，形状 (batch, task_dim) 或 None
        :return: Q 值 logits，形状 (batch, num_bins)
        """
        x = torch.cat([joint_z, joint_a], dim=-1)
        x = cat_task_emb(x, task_emb)
        return self.mlp(x)


class WorldModelCritic:
    """世界模型 Critic 封装。

    管理 Twin Q 网络、目标网络、TwoHot 编解码器和 Q 值缩放。

    :param joint_latent_dim: 联合潜在维度 (n_agents * latent_dim)
    :param joint_action_dim: 联合动作维度 (sum of action_dims)
    :param task_dim: 任务嵌入维度，0 表示单任务
    :param num_bins: Q 值离散化 bin 数量
    :param reward_min: 奖励最小值
    :param reward_max: 奖励最大值
    :param hidden_sizes: 隐藏层维度
    :param polyak: 目标网络软更新系数
    :param scale_tau: RunningScale 更新率
    :param device: 设备
    """

    def __init__(
        self,
        joint_latent_dim: int,
        joint_action_dim: int,
        task_dim: int = 0,
        num_bins: int = 101,
        reward_min: float = -10.0,
        reward_max: float = 10.0,
        hidden_sizes: list[int] | None = None,
        polyak: float = 0.01,
        scale_tau: float = 0.01,
        device: str = "cpu",
    ) -> None:
        input_dim = joint_latent_dim + joint_action_dim + task_dim

        self.critic = DisRegQNet(
            input_dim=input_dim,
            num_bins=num_bins,
            hidden_sizes=hidden_sizes,
            device=device,
        )
        self.critic2 = DisRegQNet(
            input_dim=input_dim,
            num_bins=num_bins,
            hidden_sizes=hidden_sizes,
            device=device,
        )
        self.target_critic = copy.deepcopy(self.critic)
        self.target_critic2 = copy.deepcopy(self.critic2)

        for p in self.target_critic.parameters():
            p.requires_grad_(False)
        for p in self.target_critic2.parameters():
            p.requires_grad_(False)

        self.processor = TwoHotProcessor(
            num_bins=num_bins,
            vmin=reward_min,
            vmax=reward_max,
            device=device,
        )
        self.scale = RunningScale(tau=scale_tau, device=device)
        self.polyak = polyak
        self.device = device

    def get_values(
        self,
        joint_z: torch.Tensor,
        joint_a: torch.Tensor,
        task_emb: torch.Tensor | None = None,
        mode: str = "min",
    ) -> torch.Tensor:
        """获取 Q 值。

        :param joint_z: 联合潜在状态
        :param joint_a: 联合动作
        :param task_emb: 任务嵌入
        :param mode: "min" 取两 Q 最小值，"mean" 取均值
        :return: Q 值，形状 (batch, 1)
        """
        q1 = self.processor.logits_to_scalar(
            self.critic(joint_z, joint_a, task_emb),
        )
        q2 = self.processor.logits_to_scalar(
            self.critic2(joint_z, joint_a, task_emb),
        )
        if mode == "min":
            return torch.min(q1, q2)
        return (q1 + q2) / 2

    def get_target_values(
        self,
        joint_z: torch.Tensor,
        joint_a: torch.Tensor,
        task_emb: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """获取目标 Q 值（用目标网络，取最小值）。

        :param joint_z: 联合潜在状态
        :param joint_a: 联合动作
        :param task_emb: 任务嵌入
        :return: 目标 Q 值，形状 (batch, 1)
        """
        q1 = self.processor.logits_to_scalar(
            self.target_critic(joint_z, joint_a, task_emb),
        )
        q2 = self.processor.logits_to_scalar(
            self.target_critic2(joint_z, joint_a, task_emb),
        )
        return torch.min(q1, q2)

    def soft_update(self) -> None:
        """软更新目标网络。"""
        for p, tp in zip(
            self.critic.parameters(),
            self.target_critic.parameters(),
        ):
            tp.data.lerp_(p.data, self.polyak)
        for p, tp in zip(
            self.critic2.parameters(),
            self.target_critic2.parameters(),
        ):
            tp.data.lerp_(p.data, self.polyak)

    def parameters(self) -> list[torch.Tensor]:
        """返回所有可训练参数。

        :return: 参数列表
        """
        return (
            list(self.critic.parameters())
            + list(self.critic2.parameters())
        )
