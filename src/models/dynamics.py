"""Dense 动力学模型（阶段 0 基线）。

集中式预测：输入所有智能体的联合状态和动作，
预测下一时刻的联合潜在状态。阶段 3 将替换为 SoftMoE 版本。
"""
import torch
import torch.nn as nn
from einops import rearrange

from src.models.utils import SimNorm, create_mlp


class DenseDynamics(nn.Module):
    """基于 Dense MLP 的集中式动力学模型。

    输入所有智能体的潜在状态和动作的拼接，
    预测每个智能体下一时刻的潜在状态。

    :param latent_dim: 单个智能体的潜在状态维度
    :param action_dim: 单个智能体的动作维度
    :param n_agents: 智能体数量
    :param hidden_dims: 隐藏层维度列表
    :param simnorm_dim: SimNorm 分组大小
    :param device: 设备
    """

    def __init__(
        self,
        latent_dim: int,
        action_dim: int,
        n_agents: int,
        hidden_dims: list[int] | None = None,
        simnorm_dim: int = 8,
        device: str = "cpu",
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.n_agents = n_agents

        if hidden_dims is None:
            hidden_dims = [512, 512]

        in_dim = n_agents * (latent_dim + action_dim)
        out_dim = n_agents * latent_dim

        self.mlp = create_mlp(
            in_dim=in_dim,
            mlp_dims=hidden_dims,
            out_dim=out_dim,
            act=SimNorm(simnorm_dim),
            device=device,
        )

    def predict(
        self,
        z: torch.Tensor,
        a: torch.Tensor,
    ) -> torch.Tensor:
        """预测下一时刻的潜在状态。

        :param z: 当前潜在状态，形状 (batch, n_agents, latent_dim)
        :param a: 当前动作，形状 (batch, n_agents, action_dim)
        :return: 下一时刻潜在状态，形状 (batch, n_agents, latent_dim)
        """
        batch_size = z.shape[0]
        za = torch.cat([z, a], dim=-1)
        za_flat = rearrange(za, "b n d -> b (n d)")
        out_flat = self.mlp(za_flat)
        return rearrange(
            out_flat,
            "b (n d) -> b n d",
            n=self.n_agents,
            d=self.latent_dim,
        )

    def forward(
        self,
        z: torch.Tensor,
        a: torch.Tensor,
    ) -> torch.Tensor:
        """前向传播，等价于 predict。

        :param z: 当前潜在状态
        :param a: 当前动作
        :return: 预测的下一时刻潜在状态
        """
        return self.predict(z, a)
