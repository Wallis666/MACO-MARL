"""MA-MuJoCo 环境封装。

封装 gymnasium_robotics 提供的 MaMuJoCo 环境，
输出统一的多智能体接口格式。支持多进程并行。
"""
import multiprocessing as mp
from typing import Any

import gymnasium
import numpy as np
from gymnasium_robotics import mamujoco_v1


class MAMuJoCoEnv:
    """MA-MuJoCo 多智能体环境封装。

    将 gymnasium_robotics 的 MaMuJoCo 环境包装为统一接口，
    提供 per-agent 的观测和动作空间。

    :param scenario: 机器人名称，如 "HalfCheetah"
    :param agent_conf: 智能体拆分方式，如 "2x3"
    :param episode_limit: 最大步数
    :param seed: 随机种子
    """

    def __init__(
        self,
        scenario: str = "HalfCheetah",
        agent_conf: str = "2x3",
        episode_limit: int = 1000,
        seed: int | None = None,
        orientation_penalty: float = 0.0,
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

        rewards = np.array(
            [raw_rewards[a] for a in self.env.possible_agents],
            dtype=np.float32,
        )

        # 朝向惩罚：防止 HalfCheetah 翻转后以头着地跑步
        orientation_pen = 0.0
        if self._orientation_penalty > 0:
            rooty = self._get_rooty_angle()
            # 正立时 cos=1, upright=1, penalty=0
            # 翻转时 cos=-1, upright=0, penalty=weight
            upright = (1.0 + np.cos(rooty)) / 2.0
            orientation_pen = self._orientation_penalty * (1.0 - upright)
            rewards -= orientation_pen

        any_term = any(raw_terms[a] for a in self.env.possible_agents)
        any_trunc = any(raw_truncs[a] for a in self.env.possible_agents)

        info = {
            "per_agent_rewards": rewards,
            "bad_transition": any_trunc and not any_term,
            "orientation_penalty": orientation_pen,
        }

        return obs, share_obs, rewards, any_term, any_trunc, info

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
) -> None:
    """子进程工作函数。

    :param remote: 通信管道
    :param scenario: 机器人名称
    :param agent_conf: 智能体拆分方式
    :param episode_limit: 最大步数
    :param seed: 随机种子
    :param orientation_penalty: 朝向惩罚权重
    """
    env = MAMuJoCoEnv(
        scenario=scenario,
        agent_conf=agent_conf,
        episode_limit=episode_limit,
        seed=seed,
        orientation_penalty=orientation_penalty,
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
                    episode_limit, seed + i, orientation_penalty,
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
    ) -> None:
        self.envs = [
            MAMuJoCoEnv(
                scenario=scenario,
                agent_conf=agent_conf,
                episode_limit=episode_limit,
                seed=seed + i,
                orientation_penalty=orientation_penalty,
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
