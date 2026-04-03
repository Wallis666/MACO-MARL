"""Dense 奖励模型。

集中式预测：输入所有智能体的联合状态和动作，
预测奖励的 two-hot 分布 logits。多任务通过拼接任务嵌入实现条件化。
"""
import torch
import torch.nn as nn
from einops import rearrange

from src.models.task_embedding import cat_task_emb
from src.models.utils import create_mlp


class DenseReward(nn.Module):
    """基于 Dense MLP 的集中式奖励模型。

    输入所有智能体的潜在状态和动作，
    输出奖励分布 logits（用于 TwoHot 编码）。
    多任务时输入末尾拼接任务嵌入。

    :param latent_dim: 单个智能体的潜在状态维度
    :param action_dim: 单个智能体的动作维度
    :param n_agents: 智能体数量
    :param task_dim: 任务嵌入维度，0 表示单任务
    :param num_bins: 奖励离散化 bin 数量
    :param hidden_dims: 隐藏层维度列表
    :param device: 设备
    """

    def __init__(
        self,
        latent_dim: int,
        action_dim: int,
        n_agents: int,
        task_dim: int = 0,
        num_bins: int = 101,
        hidden_dims: list[int] | None = None,
        device: str = "cpu",
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.n_agents = n_agents
        self.num_bins = num_bins

        if hidden_dims is None:
            hidden_dims = [512, 512]

        in_dim = n_agents * (latent_dim + action_dim) + task_dim

        self.mlp = create_mlp(
            in_dim=in_dim,
            mlp_dims=hidden_dims,
            out_dim=num_bins,
            device=device,
        )

    def predict(
        self,
        z: torch.Tensor,
        a: torch.Tensor,
        task_emb: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """预测奖励分布。

        :param z: 潜在状态，形状 (batch, n_agents, latent_dim)
        :param a: 动作，形状 (batch, n_agents, action_dim)
        :param task_emb: 任务嵌入，形状 (batch, task_dim) 或 None
        :return: 奖励 logits，形状 (batch, num_bins)
        """
        za = torch.cat([z, a], dim=-1)
        za_flat = rearrange(za, "b n d -> b (n d)")
        za_flat = cat_task_emb(za_flat, task_emb)
        return self.mlp(za_flat)

    def forward(
        self,
        z: torch.Tensor,
        a: torch.Tensor,
        task_emb: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """前向传播，等价于 predict。

        :param z: 潜在状态
        :param a: 动作
        :param task_emb: 任务嵌入
        :return: 奖励 logits
        """
        return self.predict(z, a, task_emb)
