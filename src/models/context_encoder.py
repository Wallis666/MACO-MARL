"""上下文编码器。

PEARL 风格：MLP 独立编码每条 transition，均值池化聚合，
L2 归一化输出任务嵌入向量。用于 few-shot 适应时从演示轨迹推断任务。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from src.models.utils import create_mlp


class ContextEncoder(nn.Module):
    """基于 MLP + 均值池化的上下文编码器。

    将 K 条演示 transitions 编码为一个任务嵌入向量，
    输出与 TaskEmbeddingTable 对齐（L2 归一化，task_dim 维）。

    :param obs_dim: 单个智能体的观测维度（使用 agent 0）
    :param action_dim: 单个智能体的动作维度（使用 agent 0）
    :param task_dim: 任务嵌入输出维度
    :param hidden_dims: MLP 隐藏层维度列表
    :param device: 设备
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        task_dim: int,
        hidden_dims: list[int] | None = None,
        device: str = "cpu",
    ) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 256]

        # 输入: concat(obs, action, reward, next_obs)
        in_dim = obs_dim + action_dim + 1 + obs_dim

        self.mlp = create_mlp(
            in_dim=in_dim,
            mlp_dims=hidden_dims,
            out_dim=task_dim,
            device=device,
        )

    def encode_transitions(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_obs: torch.Tensor,
    ) -> torch.Tensor:
        """将 K 条 transitions 编码为任务嵌入。

        :param obs: 观测，形状 (batch, K, obs_dim)
        :param actions: 动作，形状 (batch, K, action_dim)
        :param rewards: 奖励，形状 (batch, K, 1)
        :param next_obs: 下一步观测，形状 (batch, K, obs_dim)
        :return: 任务嵌入，形状 (batch, task_dim)，L2 归一化
        """
        batch, k, _ = obs.shape

        # 拼接每条 transition 的特征
        x = torch.cat([obs, actions, rewards, next_obs], dim=-1)

        # 展平为 (batch*K, in_dim) 通过 MLP
        x_flat = rearrange(x, "b k d -> (b k) d")
        h_flat = self.mlp(x_flat)

        # 恢复为 (batch, K, task_dim)，均值池化
        h = rearrange(h_flat, "(b k) d -> b k d", b=batch, k=k)
        z_raw = h.mean(dim=1)

        # L2 归一化匹配 TaskEmbeddingTable 的 max_norm=1
        return F.normalize(z_raw, dim=-1)

    def forward(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_obs: torch.Tensor,
    ) -> torch.Tensor:
        """前向传播，等价于 encode_transitions。

        :param obs: 观测，形状 (batch, K, obs_dim)
        :param actions: 动作，形状 (batch, K, action_dim)
        :param rewards: 奖励，形状 (batch, K, 1)
        :param next_obs: 下一步观测，形状 (batch, K, obs_dim)
        :return: 任务嵌入，形状 (batch, task_dim)
        """
        return self.encode_transitions(obs, actions, rewards, next_obs)
