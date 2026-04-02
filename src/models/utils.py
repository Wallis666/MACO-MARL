"""世界模型通用工具组件。

包括 SimNorm、NormedLinear、TwoHotProcessor、RunningScale 等基础模块，
供 Encoder、Dynamics、Reward、Critic 等组件使用。
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimNorm(nn.Module):
    """SimNorm 归一化层。

    将输入张量最后一维按分组大小拆分，对每组做 softmax 归一化。

    :param dim: 分组大小
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播。

        :param x: 输入张量，最后一维须为 dim 的整数倍
        :return: 归一化后的张量，形状不变
        """
        shp = x.shape
        x = x.view(*shp[:-1], -1, self.dim)
        x = F.softmax(x, dim=-1)
        return x.view(*shp)


class NormedLinear(nn.Linear):
    """带 LayerNorm 和激活函数的线性层。

    执行顺序：Linear -> Dropout -> LayerNorm -> Activation。

    :param in_features: 输入特征维度
    :param out_features: 输出特征维度
    :param dropout: Dropout 概率
    :param act: 激活函数，默认 Mish
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        dropout: float = 0.0,
        act: nn.Module | None = None,
    ) -> None:
        super().__init__(in_features, out_features)
        if act is None:
            act = nn.Mish(inplace=True)
        self.ln = nn.LayerNorm(out_features)
        self.act = act
        self.dropout = nn.Dropout(dropout, inplace=True) if dropout else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播。

        :param x: 输入张量
        :return: 变换后的张量
        """
        x = super().forward(x)
        if self.dropout is not None:
            x = self.dropout(x)
        return self.act(self.ln(x))


def create_mlp(
    in_dim: int,
    mlp_dims: list[int],
    out_dim: int,
    act: nn.Module | None = None,
    dropout: float = 0.0,
    device: str = "cpu",
) -> nn.Sequential:
    """构建标准 MLP（NormedLinear 隐藏层 + 可选末层激活）。

    :param in_dim: 输入维度
    :param mlp_dims: 隐藏层维度列表
    :param out_dim: 输出维度
    :param act: 末层激活函数，None 则末层为普通 Linear
    :param dropout: 第一层的 Dropout 概率
    :param device: 设备
    :return: nn.Sequential 模型
    """
    if isinstance(mlp_dims, int):
        mlp_dims = [mlp_dims]
    dims = [in_dim] + mlp_dims + [out_dim]
    layers: list[nn.Module] = []
    for i in range(len(dims) - 2):
        layers.append(
            NormedLinear(
                dims[i],
                dims[i + 1],
                dropout=dropout * (i == 0),
            )
        )
    if act is not None:
        layers.append(NormedLinear(dims[-2], dims[-1], act=act))
    else:
        layers.append(nn.Linear(dims[-2], dims[-1]))
    return nn.Sequential(*layers).to(device)


class TwoHotProcessor:
    """Two-Hot 分布式奖励编解码器。

    将标量奖励编码为 two-hot 向量用于分类训练，
    将预测 logits 解码回标量值。使用 symlog 变换压缩极端值。

    :param num_bins: 离散 bin 数量
    :param vmin: 最小值
    :param vmax: 最大值
    :param device: 设备
    """

    def __init__(
        self,
        num_bins: int,
        vmin: float,
        vmax: float,
        device: str = "cpu",
    ) -> None:
        self.num_bins = num_bins
        self.vmin = vmin
        self.vmax = vmax
        if num_bins > 1:
            self.bin_size = (vmax - vmin) / (num_bins - 1)
            self.bins = torch.linspace(
                vmin, vmax, num_bins, device=device,
            )
        else:
            self.bin_size = 0.0
            self.bins = None

    @staticmethod
    def symlog(x: torch.Tensor) -> torch.Tensor:
        """对称对数变换：sign(x) * log(1 + |x|)。

        :param x: 输入标量张量
        :return: 变换后的张量
        """
        return torch.sign(x) * torch.log(1 + torch.abs(x))

    @staticmethod
    def symexp(x: torch.Tensor) -> torch.Tensor:
        """对称指数变换（symlog 的逆运算）。

        :param x: 输入张量
        :return: 还原后的张量
        """
        return torch.sign(x) * (torch.exp(torch.abs(x)) - 1)

    def scalar_to_twohot(self, x: torch.Tensor) -> torch.Tensor:
        """将标量编码为 two-hot 目标向量。

        :param x: 标量奖励，形状 (batch, 1)
        :return: two-hot 向量，形状 (batch, num_bins)
        """
        if self.num_bins <= 1:
            return self.symlog(x)
        x_log = self.symlog(x)
        x_clamped = torch.clamp(x_log, self.vmin, self.vmax).squeeze(-1)
        bin_idx = torch.floor(
            (x_clamped - self.vmin) / self.bin_size,
        ).long()
        bin_idx = torch.clamp(bin_idx, 0, self.num_bins - 2)
        offset = (
            (x_clamped - self.vmin) / self.bin_size - bin_idx.float()
        ).unsqueeze(-1)
        twohot = torch.zeros(
            x.size(0), self.num_bins, device=x.device,
        )
        twohot.scatter_(1, bin_idx.unsqueeze(1), 1 - offset)
        twohot.scatter_(1, (bin_idx + 1).unsqueeze(1), offset)
        return twohot

    def logits_to_scalar(self, logits: torch.Tensor) -> torch.Tensor:
        """将预测 logits 解码为标量值。

        :param logits: 预测分布，形状 (*, num_bins)
        :return: 标量值，形状 (*, 1)
        """
        if self.num_bins <= 1:
            return self.symexp(logits)
        probs = F.softmax(logits, dim=-1)
        weighted = torch.sum(probs * self.bins, dim=-1, keepdim=True)
        return self.symexp(weighted)

    def loss(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """计算分布式回归损失（交叉熵）。

        :param logits: 预测 logits，形状 (batch, num_bins)
        :param target: 目标标量，形状 (batch, 1)
        :return: 每样本损失，形状 (batch, 1)
        """
        if self.num_bins <= 1:
            return F.mse_loss(
                self.logits_to_scalar(logits), target, reduction="none",
            )
        log_pred = F.log_softmax(logits, dim=-1)
        target_twohot = self.scalar_to_twohot(target)
        return -(target_twohot * log_pred).sum(dim=-1, keepdim=True)

    def to(self, device: str) -> "TwoHotProcessor":
        """迁移到指定设备。

        :param device: 目标设备
        :return: self
        """
        if self.bins is not None:
            self.bins = self.bins.to(device)
        return self


class RunningScale(nn.Module):
    """基于百分位的自适应缩放。

    追踪 Q 值的第 5 和第 95 百分位数，用其差值作为缩放系数。
    使用指数移动平均（EMA）更新。

    :param tau: EMA 更新率
    :param device: 设备
    """

    def __init__(
        self,
        tau: float = 0.01,
        device: str = "cpu",
    ) -> None:
        super().__init__()
        self.register_buffer(
            "_value", torch.ones(1, device=device),
        )
        self.register_buffer(
            "_percentiles",
            torch.tensor([5.0, 95.0], device=device),
        )
        self.tau = tau

    @property
    def value(self) -> torch.Tensor:
        """当前缩放值。"""
        return self._value

    def _percentile(self, x: torch.Tensor) -> torch.Tensor:
        """计算指定百分位数。

        :param x: 输入数据
        :return: 百分位值
        """
        x_flat = x.view(x.shape[0], -1)
        in_sorted, _ = torch.sort(x_flat, dim=0)
        positions = self._percentiles * (x_flat.shape[0] - 1) / 100.0
        floored = torch.floor(positions).long()
        ceiled = torch.clamp(floored + 1, max=x_flat.shape[0] - 1)
        weight_ceil = (positions - floored.float()).unsqueeze(-1)
        weight_floor = 1.0 - weight_ceil
        d0 = in_sorted[floored] * weight_floor
        d1 = in_sorted[ceiled] * weight_ceil
        return (d0 + d1).view(-1, *x.shape[1:])

    def update(self, x: torch.Tensor) -> None:
        """用新数据更新缩放系数。

        :param x: 输入数据
        """
        percentiles = self._percentile(x.detach())
        value = torch.clamp(percentiles[1] - percentiles[0], min=1.0)
        self._value.data.lerp_(value.mean(), self.tau)

    def forward(
        self,
        x: torch.Tensor,
        update: bool = False,
    ) -> torch.Tensor:
        """缩放输入。

        :param x: 输入张量
        :param update: 是否同时更新缩放系数
        :return: 缩放后的张量
        """
        if update:
            self.update(x)
        return x / self._value
