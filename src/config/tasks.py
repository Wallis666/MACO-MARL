"""多任务注册表。

定义可用任务、自定义奖励函数及 tolerance 工具。
奖励函数输出归一化到 [0, 1]，姿态要求以乘法因子融入。
"""
from dataclasses import dataclass
from typing import Callable

import numpy as np


# ---------------------------------------------------------------------------
# tolerance / sigmoid 工具（移植自 dm_control / M3W custom_suites/utils.py）
# ---------------------------------------------------------------------------

def _sigmoids(
    x: np.ndarray | float,
    value_at_1: float,
    sigmoid: str,
) -> np.ndarray | float:
    """当 x == 0 返回 1，其余在 0~1 之间。

    :param x: 输入标量或数组
    :param value_at_1: x == 1 时的输出值
    :param sigmoid: sigmoid 类型
    :return: [0, 1] 之间的值
    """
    if sigmoid in ("cosine", "linear", "quadratic"):
        if not 0 <= value_at_1 < 1:
            raise ValueError(
                f"`value_at_1` 须非负且小于 1，得到 {value_at_1}",
            )
    else:
        if not 0 < value_at_1 < 1:
            raise ValueError(
                f"`value_at_1` 须严格在 0 和 1 之间，得到 {value_at_1}",
            )

    if sigmoid == "gaussian":
        scale = np.sqrt(-2 * np.log(value_at_1))
        return np.exp(-0.5 * (x * scale) ** 2)

    elif sigmoid == "linear":
        scale = 1 - value_at_1
        scaled_x = x * scale
        return np.where(abs(scaled_x) < 1, 1 - scaled_x, 0.0)

    elif sigmoid == "quadratic":
        scale = np.sqrt(1 - value_at_1)
        scaled_x = x * scale
        return np.where(abs(scaled_x) < 1, 1 - scaled_x ** 2, 0.0)

    elif sigmoid == "hyperbolic":
        scale = np.arccosh(1 / value_at_1)
        return 1 / np.cosh(x * scale)

    elif sigmoid == "long_tail":
        scale = np.sqrt(1 / value_at_1 - 1)
        return 1 / ((x * scale) ** 2 + 1)

    elif sigmoid == "cosine":
        scale = np.arccos(2 * value_at_1 - 1) / np.pi
        scaled_x = x * scale
        cos_pi_scaled_x = np.cos(np.pi * scaled_x)
        return np.where(
            abs(scaled_x) < 1, (1 + cos_pi_scaled_x) / 2, 0.0,
        )

    elif sigmoid == "tanh_squared":
        scale = np.arctanh(np.sqrt(1 - value_at_1))
        return 1 - np.tanh(x * scale) ** 2

    else:
        raise ValueError(f"未知 sigmoid 类型 {sigmoid!r}")


_DEFAULT_VALUE_AT_MARGIN = 0.1


def tolerance(
    x: float | np.ndarray,
    bounds: tuple[float, float] = (0.0, 0.0),
    margin: float = 0.0,
    sigmoid: str = "gaussian",
    value_at_margin: float = _DEFAULT_VALUE_AT_MARGIN,
) -> float:
    """x 在 bounds 内返回 1，超出时按 sigmoid 衰减。

    :param x: 输入值
    :param bounds: (lower, upper) 目标区间
    :param margin: 衰减区间宽度
    :param sigmoid: sigmoid 类型
    :param value_at_margin: margin 处的输出值
    :return: [0, 1] 之间的标量
    """
    lower, upper = bounds
    if lower > upper:
        raise ValueError("下界须 <= 上界")
    if margin < 0:
        raise ValueError("`margin` 须非负")

    in_bounds = np.logical_and(lower <= x, x <= upper)
    if margin == 0:
        value = np.where(in_bounds, 1.0, 0.0)
    else:
        d = np.where(x < lower, lower - x, x - upper) / margin
        value = np.where(
            in_bounds, 1.0, _sigmoids(d, value_at_margin, sigmoid),
        )

    return float(value) if np.isscalar(x) else value


# ---------------------------------------------------------------------------
# HalfCheetah 奖励函数
# ---------------------------------------------------------------------------

_CHEETAH_RUN_SPEED = 10
_CHEETAH_RUN_BACKWARDS_SPEED = 8
_CHEETAH_RUN_SLOW_SPEED = 5
_CHEETAH_WALK_SPEED = 2
_CHEETAH_HEALTHY_HEIGHT = 0.6
_CHEETAH_HEIGHT_MARGIN = 0.2


def cheetah_run(info: dict) -> float:
    """HalfCheetah 前进奔跑任务奖励。

    speed_reward × height_reward × upright_reward，输出 [0, 1]。

    :param info: 物理信息字典
    :return: 归一化奖励
    """
    speed_reward = tolerance(
        info["forward_velocity"],
        bounds=(_CHEETAH_RUN_SPEED, float("inf")),
        margin=_CHEETAH_RUN_SPEED,
        sigmoid="linear",
        value_at_margin=0,
    )
    height_reward = tolerance(
        info["torso_z"],
        bounds=(_CHEETAH_HEALTHY_HEIGHT, float("inf")),
        margin=_CHEETAH_HEIGHT_MARGIN,
        sigmoid="gaussian",
    )
    upright_reward = (1.0 + np.cos(info["rooty"])) / 2.0
    return float(speed_reward * height_reward * upright_reward)


def cheetah_run_backwards(info: dict) -> float:
    """HalfCheetah 后退奔跑任务奖励。

    speed_reward × height_reward × upright_reward，输出 [0, 1]。

    :param info: 物理信息字典
    :return: 归一化奖励
    """
    speed_reward = tolerance(
        -info["forward_velocity"],
        bounds=(_CHEETAH_RUN_BACKWARDS_SPEED, float("inf")),
        margin=_CHEETAH_RUN_BACKWARDS_SPEED,
        sigmoid="linear",
        value_at_margin=0,
    )
    height_reward = tolerance(
        info["torso_z"],
        bounds=(_CHEETAH_HEALTHY_HEIGHT, float("inf")),
        margin=_CHEETAH_HEIGHT_MARGIN,
        sigmoid="gaussian",
    )
    upright_reward = (1.0 + np.cos(info["rooty"])) / 2.0
    return float(speed_reward * height_reward * upright_reward)


def cheetah_run_slow(info: dict) -> float:
    """HalfCheetah 慢速前进任务奖励（held-out 评估任务）。

    目标速度 5 m/s，输出 [0, 1]。

    :param info: 物理信息字典
    :return: 归一化奖励
    """
    speed_reward = tolerance(
        info["forward_velocity"],
        bounds=(_CHEETAH_RUN_SLOW_SPEED, float("inf")),
        margin=_CHEETAH_RUN_SLOW_SPEED,
        sigmoid="linear",
        value_at_margin=0,
    )
    height_reward = tolerance(
        info["torso_z"],
        bounds=(_CHEETAH_HEALTHY_HEIGHT, float("inf")),
        margin=_CHEETAH_HEIGHT_MARGIN,
        sigmoid="gaussian",
    )
    upright_reward = (1.0 + np.cos(info["rooty"])) / 2.0
    return float(speed_reward * height_reward * upright_reward)


def cheetah_walk(info: dict) -> float:
    """HalfCheetah 行走任务奖励（held-out 评估任务）。

    目标速度 2 m/s，输出 [0, 1]。

    :param info: 物理信息字典
    :return: 归一化奖励
    """
    speed_reward = tolerance(
        info["forward_velocity"],
        bounds=(_CHEETAH_WALK_SPEED, float("inf")),
        margin=_CHEETAH_WALK_SPEED,
        sigmoid="linear",
        value_at_margin=0,
    )
    height_reward = tolerance(
        info["torso_z"],
        bounds=(_CHEETAH_HEALTHY_HEIGHT, float("inf")),
        margin=_CHEETAH_HEIGHT_MARGIN,
        sigmoid="gaussian",
    )
    upright_reward = (1.0 + np.cos(info["rooty"])) / 2.0
    return float(speed_reward * height_reward * upright_reward)


# ---------------------------------------------------------------------------
# TaskDef 与注册表
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TaskDef:
    """任务定义。

    :param name: 任务名称
    :param scenario: MA-MuJoCo 场景名
    :param agent_conf: 智能体配置
    :param reward_fn: 自定义奖励函数，签名 (info: dict) -> float
    """

    name: str
    scenario: str
    agent_conf: str
    reward_fn: Callable[[dict], float]


TASK_REGISTRY: dict[str, TaskDef] = {
    "cheetah_run": TaskDef(
        name="cheetah_run",
        scenario="HalfCheetah",
        agent_conf="2x3",
        reward_fn=cheetah_run,
    ),
    "cheetah_run_backwards": TaskDef(
        name="cheetah_run_backwards",
        scenario="HalfCheetah",
        agent_conf="2x3",
        reward_fn=cheetah_run_backwards,
    ),
    "cheetah_run_slow": TaskDef(
        name="cheetah_run_slow",
        scenario="HalfCheetah",
        agent_conf="2x3",
        reward_fn=cheetah_run_slow,
    ),
    "cheetah_walk": TaskDef(
        name="cheetah_walk",
        scenario="HalfCheetah",
        agent_conf="2x3",
        reward_fn=cheetah_walk,
    ),
}


def get_tasks(names: list[str]) -> list[TaskDef]:
    """根据名称列表获取任务定义。

    :param names: 任务名称列表
    :return: TaskDef 列表
    :raises KeyError: 任务名不存在
    """
    tasks = []
    for name in names:
        if name not in TASK_REGISTRY:
            available = ", ".join(sorted(TASK_REGISTRY.keys()))
            raise KeyError(
                f"未知任务 {name!r}，可用任务：{available}",
            )
        tasks.append(TASK_REGISTRY[name])
    return tasks
