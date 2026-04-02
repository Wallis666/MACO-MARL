"""观测编码器。

将各智能体的原始观测映射到共享潜在空间，
使用 NormedLinear + SimNorm 结构。
"""
import torch
import torch.nn as nn

from src.models.utils import SimNorm, create_mlp


class MLPEncoder(nn.Module):
    """基于 MLP 的观测编码器。

    将单个智能体的局部观测编码为潜在向量。
    使用 NormedLinear 隐藏层 + SimNorm 末层激活。

    :param obs_dim: 观测维度
    :param latent_dim: 潜在状态维度
    :param hidden_dims: 隐藏层维度列表
    :param simnorm_dim: SimNorm 分组大小
    :param dropout: Dropout 概率
    :param device: 设备
    """

    def __init__(
        self,
        obs_dim: int,
        latent_dim: int,
        hidden_dims: list[int] | None = None,
        simnorm_dim: int = 8,
        dropout: float = 0.0,
        device: str = "cpu",
    ) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [512, 512]
        self.mlp = create_mlp(
            in_dim=obs_dim,
            mlp_dims=hidden_dims,
            out_dim=latent_dim,
            act=SimNorm(simnorm_dim),
            dropout=dropout,
            device=device,
        )

    def encode(self, obs: torch.Tensor) -> torch.Tensor:
        """编码观测到潜在空间。

        :param obs: 智能体局部观测，形状 (batch, obs_dim)
        :return: 潜在状态，形状 (batch, latent_dim)
        """
        return self.mlp(obs)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """前向传播，等价于 encode。

        :param obs: 智能体局部观测
        :return: 潜在状态
        """
        return self.encode(obs)
