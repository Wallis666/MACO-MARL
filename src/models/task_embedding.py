"""任务嵌入模块。

为每个已知任务维护一个可学习的嵌入向量，
通过 task_id 查表得到任务向量，用于条件化世界模型各组件。
参考 TD-MPC2: nn.Embedding(n_tasks, task_dim, max_norm=1)。
"""
import torch
import torch.nn as nn


class TaskEmbeddingTable(nn.Module):
    """可学习任务嵌入表。

    训练时通过 task_id 查表获取任务嵌入向量，
    推理时可由上下文编码器替代。

    :param n_tasks: 已知任务数量
    :param task_dim: 任务嵌入维度
    """

    def __init__(
        self,
        n_tasks: int,
        task_dim: int,
    ) -> None:
        super().__init__()
        self.n_tasks = n_tasks
        self.task_dim = task_dim
        self._emb = nn.Embedding(n_tasks, task_dim, max_norm=1)

    def forward(self, task_id: torch.Tensor) -> torch.Tensor:
        """查表获取任务嵌入。

        :param task_id: 任务索引，形状 (batch,) 或标量
        :return: 任务嵌入向量，形状 (batch, task_dim)
        """
        if task_id.dim() == 0:
            task_id = task_id.unsqueeze(0)
        return self._emb(task_id.long())


def cat_task_emb(
    x: torch.Tensor,
    task_emb: torch.Tensor | None,
) -> torch.Tensor:
    """将任务嵌入拼接到输入张量末尾。

    单任务模式下 task_emb 为 None，直接返回 x。

    :param x: 输入张量，形状 (batch, feat_dim)
    :param task_emb: 任务嵌入，形状 (batch, task_dim) 或 None
    :return: 拼接后的张量，形状 (batch, feat_dim + task_dim)
    """
    if task_emb is None:
        return x
    return torch.cat([x, task_emb], dim=-1)
