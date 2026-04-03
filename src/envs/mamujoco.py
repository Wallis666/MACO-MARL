"""MA-MuJoCo 环境封装。

封装 gymnasium_robotics 提供的 MaMuJoCo 环境，
输出统一的多智能体接口格式。支持多进程并行及多任务训练。
"""
import multiprocessing as mp
from typing import Any, Callable

import gymnasium
import numpy as np
from gymnasium_robotics import mamujoco_v1


def _gaussian_tolerance(
    x: float,
    lower: float,
    margin: float,
    value_at_margin: float = 0.1,
) -> float:
    """高斯型 tolerance 函数（参考 dm_control rewards.tolerance）。

    x >= lower 时返回 1.0；
    x < lower 时按高斯曲线平滑衰减，在 lower - margin 处返回 value_at_margin。

    :param x: 输入值
    :param lower: 下界阈值
    :param margin: 衰减区间宽度
    :param value_at_margin: margin 处的值
    :return: [value_at_margin, 1.0] 之间的标量
    """
    if x >= lower:
        return 1.0
    if margin <= 0:
        return value_at_margin
    d = (x - lower) / margin
    # 高斯核：exp(-0.5 * (d / sigma)^2)，令 sigma 使得 d=-1 时值为 value_at_margin
    sigma = 1.0 / np.sqrt(-2.0 * np.log(value_at_margin))
    return float(np.exp(-0.5 * (d / sigma) ** 2))


class MAMuJoCoEnv:
    """MA-MuJoCo 多智能体环境封装。

    将 gymnasium_robotics 的 MaMuJoCo 环境包装为统一接口，
    提供 per-agent 的观测和动作空间。

    :param scenario: 机器人名称，如 "HalfCheetah"
    :param agent_conf: 智能体拆分方式，如 "2x3"
    :param episode_limit: 最大步数
    :param seed: 随机种子
    :param reward_fn: 自定义奖励函数，签名 (info: dict) -> float。
        提供时替换原始奖励及所有惩罚逻辑。
    """

    def __init__(
        self,
        scenario: str = "HalfCheetah",
        agent_conf: str = "2x3",
        episode_limit: int = 1000,
        seed: int | None = None,
        orientation_penalty: float = 0.0,
        action_rate_penalty: float = 0.0,
        healthy_height: float = 0.0,
        reward_fn: Callable[[dict], float] | None = None,
    ) -> None:
        self.env = mamujoco_v1.parallel_env(
            scenario=scenario,
            agent_conf=agent_conf,
            max_episode_steps=episode_limit,
        )
        self.n_agents = len(self.env.possible_agents)
        self.episode_limit = episode_limit
        self._step_count = 0
        self._orientation_penalty = orientation_penalty
        self._action_rate_penalty = action_rate_penalty
        self._healthy_height = healthy_height
        self._prev_actions: np.ndarray | None = None
        self._reward_fn = reward_fn

        self.env.reset(seed=seed)

        self._obs_spaces: list[gymnasium.spaces.Box] = []
        self._act_spaces: list[gymnasium.spaces.Box] = []
        for agent in self.env.possible_agents:
            self._obs_spaces.append(self.env.observation_space(agent))
            self._act_spaces.append(self.env.action_space(agent))

    @property
    def obs_spaces(self) -> list[gymnasium.spaces.Box]:
        """各智能体的观测空间列表。"""
        return self._obs_spaces

    @property
    def act_spaces(self) -> list[gymnasium.spaces.Box]:
        """各智能体的动作空间列表。"""
        return self._act_spaces

    @property
    def obs_dims(self) -> list[int]:
        """各智能体的观测维度列表。"""
        return [s.shape[0] for s in self._obs_spaces]

    @property
    def act_dims(self) -> list[int]:
        """各智能体的动作维度列表。"""
        return [s.shape[0] for s in self._act_spaces]

    @property
    def share_obs_dim(self) -> int:
        """全局共享观测维度（所有智能体观测拼接）。"""
        return sum(self.obs_dims)

    def reset(self) -> tuple[list[np.ndarray], np.ndarray]:
        """重置环境。

        :return: (obs, share_obs)
        """
        raw_obs, _ = self.env.reset()
        self._step_count = 0
        self._prev_actions = None
        return self._process_obs(raw_obs)

    def step(
        self,
        actions: np.ndarray,
    ) -> tuple[list[np.ndarray], np.ndarray, np.ndarray, bool, bool, dict[str, Any]]:
        """执行一步。

        :param actions: 各智能体动作，形状 (n_agents, act_dim_i)
        :return: (obs, share_obs, rewards, terminated, truncated, info)
        """
        action_dict = {}
        for i, agent in enumerate(self.env.possible_agents):
            action_dict[agent] = actions[i]

        raw_obs, raw_rewards, raw_terms, raw_truncs, raw_infos = (
            self.env.step(action_dict)
        )
        self._step_count += 1

        obs, share_obs = self._process_obs(raw_obs)

        any_term = any(raw_terms[a] for a in self.env.possible_agents)
        any_trunc = any(raw_truncs[a] for a in self.env.possible_agents)

        if self._reward_fn is not None:
            # 多任务模式：使用自定义奖励函数（姿态要求已融入）
            physics_info = {
                "forward_velocity": self._get_forward_velocity(),
                "torso_z": self._get_torso_z(),
                "rooty": self._get_rooty_angle(),
            }
            reward_scalar = self._reward_fn(physics_info)
            rewards = np.full(
                self.n_agents, reward_scalar, dtype=np.float32,
            )
            self._prev_actions = actions.copy()
            info = {
                "per_agent_rewards": rewards,
                "bad_transition": any_trunc and not any_term,
            }
        else:
            # 单任务模式：原始奖励 + 外挂惩罚（向后兼容）
            rewards = np.array(
                [raw_rewards[a] for a in self.env.possible_agents],
                dtype=np.float32,
            )

            orientation_pen = 0.0
            if self._orientation_penalty > 0:
                rooty = self._get_rooty_angle()
                upright = (1.0 + np.cos(rooty)) / 2.0
                orientation_pen = self._orientation_penalty * (1.0 - upright)
                rewards -= orientation_pen

            action_rate_pen = 0.0
            if self._action_rate_penalty > 0 and self._prev_actions is not None:
                delta = actions - self._prev_actions
                action_rate_pen = self._action_rate_penalty * float(
                    np.sum(delta ** 2) / self.n_agents,
                )
                rewards -= action_rate_pen
            self._prev_actions = actions.copy()

            height_scale = 1.0
            if self._healthy_height > 0:
                torso_z = self._get_torso_z()
                height_scale = _gaussian_tolerance(
                    torso_z,
                    lower=self._healthy_height,
                    margin=self._healthy_height * 0.5,
                )
                rewards *= height_scale

            info = {
                "per_agent_rewards": rewards,
                "bad_transition": any_trunc and not any_term,
                "orientation_penalty": orientation_pen,
                "action_rate_penalty": action_rate_pen,
                "height_scale": height_scale,
                "torso_z": self._get_torso_z() if self._healthy_height > 0 else 0.0,
            }

        return obs, share_obs, rewards, any_term, any_trunc, info

    def _get_forward_velocity(self) -> float:
        """获取根关节的前进速度（rootx 方向的 qvel）。

        :return: 前进速度（m/s）
        """
        base_env = self.env.unwrapped.single_agent_env.unwrapped
        return float(base_env.data.qvel[0])

    def _get_torso_z(self) -> float:
        """获取 torso 在世界坐标系下的 z 高度。

        :return: torso z 坐标（米）
        """
        base_env = self.env.unwrapped.single_agent_env.unwrapped
        return float(base_env.data.body("torso").xpos[2])

    def _get_rooty_angle(self) -> float:
        """获取身体绕 Y 轴旋转角度（rooty）。

        通过底层单智能体环境的 qpos[2] 获取。
        HalfCheetah qpos 布局: [rootx, rootz, rooty, bthigh, bshin, bfoot, fthigh, fshin, ffoot]

        :return: rooty 角度（弧度）
        """
        base_env = self.env.unwrapped.single_agent_env.unwrapped
        return float(base_env.data.qpos[2])

    def _process_obs(
        self,
        raw_obs: dict[str, np.ndarray],
    ) -> tuple[list[np.ndarray], np.ndarray]:
        """处理原始观测为统一格式。

        :param raw_obs: 字典形式的观测
        :return: (obs_list, share_obs)
        """
        obs_list = []
        for agent in self.env.possible_agents:
            obs_list.append(
                raw_obs[agent].astype(np.float32),
            )

        share_obs = np.concatenate(obs_list, axis=-1)
        return obs_list, share_obs

    def close(self) -> None:
        """关闭环境。"""
        self.env.close()


def _worker_fn(
    remote: mp.connection.Connection,
    scenario: str,
    agent_conf: str,
    episode_limit: int,
    seed: int,
    orientation_penalty: float = 0.0,
    action_rate_penalty: float = 0.0,
    healthy_height: float = 0.0,
) -> None:
    """子进程工作函数。

    :param remote: 通信管道
    :param scenario: 机器人名称
    :param agent_conf: 智能体拆分方式
    :param episode_limit: 最大步数
    :param seed: 随机种子
    :param orientation_penalty: 朝向惩罚权重
    :param action_rate_penalty: 动作变化率惩罚权重
    :param healthy_height: 健康高度阈值
    """
    env = MAMuJoCoEnv(
        scenario=scenario,
        agent_conf=agent_conf,
        episode_limit=episode_limit,
        seed=seed,
        orientation_penalty=orientation_penalty,
        action_rate_penalty=action_rate_penalty,
        healthy_height=healthy_height,
    )
    while True:
        cmd, data = remote.recv()
        if cmd == "step":
            obs, share_obs, rewards, term, trunc, info = env.step(data)
            done = term or trunc
            if done:
                new_obs, new_share = env.reset()
                info["terminal_obs"] = obs
                info["terminal_share_obs"] = share_obs
                obs = new_obs
                share_obs = new_share
            remote.send((obs, share_obs, rewards.mean(), done, trunc and not term, info))
        elif cmd == "reset":
            obs, share_obs = env.reset()
            remote.send((obs, share_obs))
        elif cmd == "get_spaces":
            remote.send((env.obs_spaces, env.act_spaces, env.n_agents,
                         env.obs_dims, env.act_dims, env.share_obs_dim))
        elif cmd == "sample_actions":
            actions = np.stack(
                [env.act_spaces[i].sample() for i in range(env.n_agents)],
                axis=0,
            )
            remote.send(actions)
        elif cmd == "close":
            env.close()
            remote.close()
            break


class SubprocVectorMAMuJoCoEnv:
    """多进程向量化 MA-MuJoCo 环境。

    每个环境在独立子进程中运行，通过管道通信，
    实现真正的并行环境步进。

    :param n_envs: 并行环境数量
    :param scenario: 机器人名称
    :param agent_conf: 智能体拆分方式
    :param episode_limit: 最大步数
    :param seed: 基础随机种子
    """

    def __init__(
        self,
        n_envs: int,
        scenario: str = "HalfCheetah",
        agent_conf: str = "2x3",
        episode_limit: int = 1000,
        seed: int = 0,
        orientation_penalty: float = 0.0,
        action_rate_penalty: float = 0.0,
        healthy_height: float = 0.0,
    ) -> None:
        self.n_envs = n_envs
        self.remotes: list[mp.connection.Connection] = []
        self.processes: list[mp.Process] = []

        ctx = mp.get_context("spawn")
        for i in range(n_envs):
            parent_conn, child_conn = ctx.Pipe()
            proc = ctx.Process(
                target=_worker_fn,
                args=(
                    child_conn, scenario, agent_conf,
                    episode_limit, seed + i,
                    orientation_penalty, action_rate_penalty,
                    healthy_height,
                ),
                daemon=True,
            )
            proc.start()
            child_conn.close()
            self.remotes.append(parent_conn)
            self.processes.append(proc)

        self.remotes[0].send(("get_spaces", None))
        spaces = self.remotes[0].recv()
        self.obs_spaces = spaces[0]
        self.act_spaces = spaces[1]
        self.n_agents = spaces[2]
        self.obs_dims = spaces[3]
        self.act_dims = spaces[4]
        self.share_obs_dim = spaces[5]

    def reset(self) -> tuple[list[list[np.ndarray]], np.ndarray]:
        """重置所有环境。

        :return: (obs, share_obs)
        """
        for remote in self.remotes:
            remote.send(("reset", None))

        all_obs = []
        all_share = []
        for remote in self.remotes:
            obs, share_obs = remote.recv()
            all_obs.append(obs)
            all_share.append(share_obs)
        return all_obs, np.stack(all_share)

    def step(
        self,
        actions: np.ndarray,
    ) -> tuple[
        list[list[np.ndarray]],
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        list[dict],
    ]:
        """所有环境并行执行一步。

        :param actions: 形状 (n_envs, n_agents, act_dim)
        :return: (obs, share_obs, rewards, dones, truncs, infos)
        """
        for i, remote in enumerate(self.remotes):
            remote.send(("step", actions[i]))

        all_obs = []
        all_share = []
        all_rewards = []
        all_dones = []
        all_truncs = []
        all_infos = []

        for remote in self.remotes:
            obs, share_obs, reward, done, trunc, info = remote.recv()
            all_obs.append(obs)
            all_share.append(share_obs)
            all_rewards.append(reward)
            all_dones.append(done)
            all_truncs.append(trunc)
            all_infos.append(info)

        return (
            all_obs,
            np.stack(all_share),
            np.array(all_rewards, dtype=np.float32),
            np.array(all_dones, dtype=np.bool_),
            np.array(all_truncs, dtype=np.bool_),
            all_infos,
        )

    def sample_random_actions(self) -> np.ndarray:
        """并行采样随机动作。

        :return: 形状 (n_envs, n_agents, act_dim)
        """
        for remote in self.remotes:
            remote.send(("sample_actions", None))
        actions = [remote.recv() for remote in self.remotes]
        return np.stack(actions, axis=0)

    def close(self) -> None:
        """关闭所有子进程。"""
        for remote in self.remotes:
            remote.send(("close", None))
        for proc in self.processes:
            proc.join(timeout=5)


class VectorMAMuJoCoEnv:
    """向量化 MA-MuJoCo 环境（串行，用于调试）。

    :param n_envs: 并行环境数量
    :param scenario: 机器人名称
    :param agent_conf: 智能体拆分方式
    :param episode_limit: 最大步数
    :param seed: 基础随机种子
    :param orientation_penalty: 朝向惩罚权重
    """

    def __init__(
        self,
        n_envs: int,
        scenario: str = "HalfCheetah",
        agent_conf: str = "2x3",
        episode_limit: int = 1000,
        seed: int = 0,
        orientation_penalty: float = 0.0,
        action_rate_penalty: float = 0.0,
        healthy_height: float = 0.0,
    ) -> None:
        self.envs = [
            MAMuJoCoEnv(
                scenario=scenario,
                agent_conf=agent_conf,
                episode_limit=episode_limit,
                seed=seed + i,
                orientation_penalty=orientation_penalty,
                action_rate_penalty=action_rate_penalty,
                healthy_height=healthy_height,
            )
            for i in range(n_envs)
        ]
        self.n_envs = n_envs
        self.n_agents = self.envs[0].n_agents
        self.obs_spaces = self.envs[0].obs_spaces
        self.act_spaces = self.envs[0].act_spaces
        self.obs_dims = self.envs[0].obs_dims
        self.act_dims = self.envs[0].act_dims
        self.share_obs_dim = self.envs[0].share_obs_dim

    def reset(self) -> tuple[list[list[np.ndarray]], np.ndarray]:
        """重置所有环境。

        :return: (obs, share_obs)
        """
        all_obs = []
        all_share = []
        for env in self.envs:
            obs, share_obs = env.reset()
            all_obs.append(obs)
            all_share.append(share_obs)
        return all_obs, np.stack(all_share)

    def step(
        self,
        actions: np.ndarray,
    ) -> tuple[
        list[list[np.ndarray]],
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        list[dict],
    ]:
        """所有环境执行一步。

        :param actions: 形状 (n_envs, n_agents, act_dim)
        :return: (obs, share_obs, rewards, dones, truncs, infos)
        """
        all_obs = []
        all_share = []
        all_rewards = []
        all_dones = []
        all_truncs = []
        all_infos = []

        for i, env in enumerate(self.envs):
            obs, share_obs, rewards, term, trunc, info = env.step(
                actions[i],
            )
            done = term or trunc

            if done:
                new_obs, new_share = env.reset()
                info["terminal_obs"] = obs
                info["terminal_share_obs"] = share_obs
                obs = new_obs
                share_obs = new_share

            all_obs.append(obs)
            all_share.append(share_obs)
            all_rewards.append(rewards.mean())
            all_dones.append(done)
            all_truncs.append(trunc and not term)
            all_infos.append(info)

        return (
            all_obs,
            np.stack(all_share),
            np.array(all_rewards, dtype=np.float32),
            np.array(all_dones, dtype=np.bool_),
            np.array(all_truncs, dtype=np.bool_),
            all_infos,
        )

    def close(self) -> None:
        """关闭所有环境。"""
        for env in self.envs:
            env.close()


# ---------------------------------------------------------------------------
# 多任务向量化环境
# ---------------------------------------------------------------------------


def _mt_worker_fn(
    remote: mp.connection.Connection,
    scenario: str,
    agent_conf: str,
    episode_limit: int,
    seed: int,
    reward_fn: Callable[[dict], float],
    task_idx: int,
    n_tasks: int,
) -> None:
    """多任务子进程工作函数。

    返回原始观测（不做 one-hot 拼接），任务信息通过 task_idx 传递。

    :param remote: 通信管道
    :param scenario: 机器人名称
    :param agent_conf: 智能体拆分方式
    :param episode_limit: 最大步数
    :param seed: 随机种子
    :param reward_fn: 自定义奖励函数
    :param task_idx: 本 worker 的任务索引
    :param n_tasks: 总任务数（保留参数以兼容接口）
    """
    env = MAMuJoCoEnv(
        scenario=scenario,
        agent_conf=agent_conf,
        episode_limit=episode_limit,
        seed=seed,
        reward_fn=reward_fn,
    )

    while True:
        cmd, data = remote.recv()
        if cmd == "step":
            obs, share_obs, rewards, term, trunc, info = env.step(data)
            done = term or trunc
            if done:
                new_obs, new_share = env.reset()
                info["terminal_obs"] = list(obs)
                info["terminal_share_obs"] = share_obs
                obs, share_obs = new_obs, new_share
            remote.send((
                obs, share_obs, rewards.mean(),
                done, trunc and not term, info,
            ))
        elif cmd == "reset":
            obs, share_obs = env.reset()
            remote.send((obs, share_obs))
        elif cmd == "get_spaces":
            remote.send((
                env.obs_spaces, env.act_spaces, env.n_agents,
                env.obs_dims, env.act_dims, env.share_obs_dim,
            ))
        elif cmd == "sample_actions":
            actions = np.stack(
                [env.act_spaces[i].sample() for i in range(env.n_agents)],
                axis=0,
            )
            remote.send(actions)
        elif cmd == "close":
            env.close()
            remote.close()
            break


class MultiTaskVectorMAMuJoCoEnv:
    """多任务多进程向量化 MA-MuJoCo 环境。

    每个 worker 绑定一个任务（含自定义 reward_fn），
    返回原始观测，任务信息通过 task_idxes 属性获取。

    :param n_envs: 并行环境数量，须能被 n_tasks 整除
    :param tasks: TaskDef 列表
    :param episode_limit: 最大步数
    :param seed: 基础随机种子
    """

    def __init__(
        self,
        n_envs: int,
        tasks: list,
        episode_limit: int = 1000,
        seed: int = 0,
    ) -> None:
        self.n_tasks = len(tasks)
        assert n_envs % self.n_tasks == 0, (
            f"n_envs({n_envs}) 须能被 n_tasks({self.n_tasks}) 整除"
        )
        self.n_envs = n_envs
        self.task_idxes = np.arange(
            self.n_tasks,
        ).repeat(n_envs // self.n_tasks)

        self.remotes: list[mp.connection.Connection] = []
        self.processes: list[mp.Process] = []

        ctx = mp.get_context("spawn")
        for i in range(n_envs):
            task_idx = int(self.task_idxes[i])
            task = tasks[task_idx]
            parent_conn, child_conn = ctx.Pipe()
            proc = ctx.Process(
                target=_mt_worker_fn,
                args=(
                    child_conn, task.scenario, task.agent_conf,
                    episode_limit, seed + i,
                    task.reward_fn, task_idx, self.n_tasks,
                ),
                daemon=True,
            )
            proc.start()
            child_conn.close()
            self.remotes.append(parent_conn)
            self.processes.append(proc)

        # 获取空间信息（从第一个 worker）
        self.remotes[0].send(("get_spaces", None))
        spaces = self.remotes[0].recv()
        self.n_agents = spaces[2]
        self.act_spaces = spaces[1]
        self.act_dims = spaces[4]

        # 使用原始观测维度（任务信息通过嵌入注入，不再拼接 one-hot）
        self.obs_dims = list(spaces[3])
        self.share_obs_dim = spaces[5]

    def reset(self) -> tuple[list[list[np.ndarray]], np.ndarray]:
        """重置所有环境。

        :return: (obs, share_obs)
        """
        for remote in self.remotes:
            remote.send(("reset", None))

        all_obs = []
        all_share = []
        for remote in self.remotes:
            obs, share_obs = remote.recv()
            all_obs.append(obs)
            all_share.append(share_obs)
        return all_obs, np.stack(all_share)

    def step(
        self,
        actions: np.ndarray,
    ) -> tuple[
        list[list[np.ndarray]],
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        list[dict],
    ]:
        """所有环境并行执行一步。

        :param actions: 形状 (n_envs, n_agents, act_dim)
        :return: (obs, share_obs, rewards, dones, truncs, infos)
        """
        for i, remote in enumerate(self.remotes):
            remote.send(("step", actions[i]))

        all_obs = []
        all_share = []
        all_rewards = []
        all_dones = []
        all_truncs = []
        all_infos = []

        for remote in self.remotes:
            obs, share_obs, reward, done, trunc, info = remote.recv()
            all_obs.append(obs)
            all_share.append(share_obs)
            all_rewards.append(reward)
            all_dones.append(done)
            all_truncs.append(trunc)
            all_infos.append(info)

        return (
            all_obs,
            np.stack(all_share),
            np.array(all_rewards, dtype=np.float32),
            np.array(all_dones, dtype=np.bool_),
            np.array(all_truncs, dtype=np.bool_),
            all_infos,
        )

    def sample_random_actions(self) -> np.ndarray:
        """并行采样随机动作。

        :return: 形状 (n_envs, n_agents, act_dim)
        """
        for remote in self.remotes:
            remote.send(("sample_actions", None))
        actions = [remote.recv() for remote in self.remotes]
        return np.stack(actions, axis=0)

    def close(self) -> None:
        """关闭所有子进程。"""
        for remote in self.remotes:
            remote.send(("close", None))
        for proc in self.processes:
            proc.join(timeout=5)


class MultiTaskVectorMAMuJoCoEnvDebug:
    """多任务串行向量化环境（调试用）。

    功能与 MultiTaskVectorMAMuJoCoEnv 相同，但在主进程串行执行。
    返回原始观测，任务信息通过 task_idxes 属性获取。

    :param n_envs: 环境数量，须能被 n_tasks 整除
    :param tasks: TaskDef 列表
    :param episode_limit: 最大步数
    :param seed: 基础随机种子
    """

    def __init__(
        self,
        n_envs: int,
        tasks: list,
        episode_limit: int = 1000,
        seed: int = 0,
    ) -> None:
        self.n_tasks = len(tasks)
        assert n_envs % self.n_tasks == 0, (
            f"n_envs({n_envs}) 须能被 n_tasks({self.n_tasks}) 整除"
        )
        self.n_envs = n_envs
        self.task_idxes = np.arange(
            self.n_tasks,
        ).repeat(n_envs // self.n_tasks)

        self.envs: list[MAMuJoCoEnv] = []
        for i in range(n_envs):
            task_idx = int(self.task_idxes[i])
            task = tasks[task_idx]
            env = MAMuJoCoEnv(
                scenario=task.scenario,
                agent_conf=task.agent_conf,
                episode_limit=episode_limit,
                seed=seed + i,
                reward_fn=task.reward_fn,
            )
            self.envs.append(env)

        self.n_agents = self.envs[0].n_agents
        self.act_spaces = self.envs[0].act_spaces
        self.act_dims = self.envs[0].act_dims
        # 使用原始观测维度（任务信息通过嵌入注入，不再拼接 one-hot）
        self.obs_dims = list(self.envs[0].obs_dims)
        self.share_obs_dim = self.envs[0].share_obs_dim

    def reset(self) -> tuple[list[list[np.ndarray]], np.ndarray]:
        """重置所有环境。

        :return: (obs, share_obs)
        """
        all_obs = []
        all_share = []
        for env in self.envs:
            obs, share_obs = env.reset()
            all_obs.append(obs)
            all_share.append(share_obs)
        return all_obs, np.stack(all_share)

    def step(
        self,
        actions: np.ndarray,
    ) -> tuple[
        list[list[np.ndarray]],
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        list[dict],
    ]:
        """所有环境执行一步。

        :param actions: 形状 (n_envs, n_agents, act_dim)
        :return: (obs, share_obs, rewards, dones, truncs, infos)
        """
        all_obs = []
        all_share = []
        all_rewards = []
        all_dones = []
        all_truncs = []
        all_infos = []

        for i, env in enumerate(self.envs):
            obs, share_obs, rewards, term, trunc, info = env.step(
                actions[i],
            )
            done = term or trunc

            if done:
                new_obs, new_share = env.reset()
                info["terminal_obs"] = list(obs)
                info["terminal_share_obs"] = share_obs
                obs, share_obs = new_obs, new_share

            all_obs.append(obs)
            all_share.append(share_obs)
            all_rewards.append(rewards.mean())
            all_dones.append(done)
            all_truncs.append(trunc and not term)
            all_infos.append(info)

        return (
            all_obs,
            np.stack(all_share),
            np.array(all_rewards, dtype=np.float32),
            np.array(all_dones, dtype=np.bool_),
            np.array(all_truncs, dtype=np.bool_),
            all_infos,
        )

    def sample_random_actions(self) -> np.ndarray:
        """采样随机动作。

        :return: 形状 (n_envs, n_agents, act_dim)
        """
        return np.stack(
            [
                np.stack(
                    [env.act_spaces[j].sample()
                     for j in range(env.n_agents)],
                )
                for env in self.envs
            ],
            axis=0,
        )

    def close(self) -> None:
        """关闭所有环境。"""
        for env in self.envs:
            env.close()
