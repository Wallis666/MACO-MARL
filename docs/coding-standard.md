# Python 编码规范

本项目遵循 PEP 8 风格，并在此基础上做以下强制约定。

## 1. 文件结构

每个 `.py` 文件顶部须包含**模块级文档字符串**（三引号），简要说明该文件的职责。

```python
"""SoftMoE 动力学模型。

将多智能体的观测-动作对作为 token 序列，通过软路由分配给多个专家，
预测下一时刻的潜在状态。
"""
```

## 2. 导入顺序

按以下三组排列，组内按字典序排列，**组间空一行**：

1. **标准库**（`abc`、`typing`、`os` 等）
2. **第三方库**（`gymnasium_robotics`、`mujoco`、`torch` 等）
3. **项目内部模块**

```python
import os
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange, repeat

from models.dynamics import SoftMoEDynamics
from utils.normalization import RunningMeanStd
```

## 3. 函数 / 方法签名格式

当参数（含 `self` / `cls`）**>= 2 个**时，每个参数必须独占一行，右括号与 `def` 对齐：

```python
def compute_loss(
    self,
    obs: torch.Tensor,
    action: torch.Tensor,
    task_embed: torch.Tensor | None = None,
) -> torch.Tensor:
```

单参数可写在同一行：

```python
def reset(self) -> None:
```

## 4. 类型注解

- 所有函数 / 方法的**参数**和**返回值**必须标注类型。
- 优先使用内建泛型（`dict[str, Any]`、`list[int]`）而非 `typing.Dict` / `typing.List`（Python >= 3.9）。

## 5. 文档字符串（Docstring）

- 类和公开方法必须有 docstring。
- **语言**：除参数名、类型名外，一律使用**中文**。
- 参数说明使用 `:param xxx:` 格式，返回值使用 `:return:` 格式。

```python
class SoftMoEDynamics(nn.Module):
    """基于 SoftMoE 的多智能体动力学模型。

    将多智能体的观测-动作对作为 token 序列，通过软路由分配给多个专家，
    预测下一时刻的潜在状态。

    :param latent_dim: 潜在状态维度
    :param action_dim: 动作维度
    :param num_experts: 专家数量
    :param hidden_dim: 专家 MLP 隐藏层维度
    """
```

## 6. 类的结构

类内元素按以下顺序排列，各组之间空一行：

1. 类级文档字符串
2. `__init__`
3. 属性（`@property`）
4. 公开方法
5. 私有方法（`_` 前缀）

## 7. 其他约定

- 行宽上限：**88 字符**（与 `black` 默认一致）。
- 字符串统一使用**双引号** `"`。
- 尾部逗号（trailing comma）：多行参数列表、字典、列表等末尾元素后**加逗号**。
- 空行规则：顶层定义之间空 **2 行**，类内方法之间空 **1 行**。
- 张量变形使用 `einops` 而非原生 `view`/`reshape`/`permute` 链式调用。
- 所有神经网络模块继承自 `torch.nn.Module`。
- 新代码的配置采用 JSON 文件（遵循 M3W 模式）。

## 8. 项目专用规则

- 所有新代码放入**独立的项目目录**（不在 `references/` 内，参考代码库保持只读）。
- 世界模型各组件应模块化且可独立测试。
- 每个组件（encoder、dynamics、reward、actor、critic）应有独立的单元测试验证前向传播的张量形状。

## 9. 参考示例

```python
"""SoftMoE 动力学模型。

将多智能体的观测-动作对作为 token 序列，通过软路由分配给多个专家，
预测下一时刻的潜在状态。
"""
from abc import ABC, abstractmethod
from typing import Any

import torch
import torch.nn as nn
from einops import rearrange


class BaseDynamics(ABC):
    """动力学模型基类，所有具体实现必须继承此类。

    子类需要实现：
        - forward(): 给定当前状态和动作，预测下一时刻的潜在状态
    """

    def __init__(self) -> None:
        self._step_count = 0

    @abstractmethod
    def forward(
        self,
        latent: torch.Tensor,
        action: torch.Tensor,
        task_embed: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """前向预测。

        :param latent: 潜在状态，形状 (batch, n_agents, latent_dim)
        :param action: 动作，形状 (batch, n_agents, action_dim)
        :param task_embed: 任务嵌入，形状 (batch, embed_dim)，可选
        :return: 预测的下一时刻潜在状态
        """
        ...

    def reset(self) -> None:
        """重置内部状态。"""
        self._step_count = 0
```
