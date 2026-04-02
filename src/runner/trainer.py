"""主训练循环。

整合环境、世界模型、Actor/Critic、Buffer 和 MPPI 规划器，
完成 warmup -> rollout -> model_train -> actor_train 的完整流程。
"""
import json
import os
import time
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from src.algorithms.actor import WorldModelActor
from src.algorithms.critic import WorldModelCritic
from src.algorithms.planner import MPPIPlanner
from src.buffer.replay_buffer import ReplayBuffer
from src.envs.mamujoco import SubprocVectorMAMuJoCoEnv, VectorMAMuJoCoEnv
from src.models.dynamics import DenseDynamics
from src.models.encoder import MLPEncoder
from src.models.reward import DenseReward
from src.models.utils import TwoHotProcessor


class Trainer:
    """阶段 0 单任务训练器。

    :param config_path: 配置文件路径
    :param device: 设备
    :param run_dir: 日志和模型保存目录
    :param use_subproc: 是否使用多进程环境
    """

    def __init__(
        self,
        config_path: str,
        device: str = "cpu",
        run_dir: str = "runs",
        use_subproc: bool = True,
    ) -> None:
        with open(config_path, "r") as f:
            self.config = json.load(f)

        self.device = device
        self.run_dir = run_dir
        self.use_subproc = use_subproc
        os.makedirs(run_dir, exist_ok=True)

        self._init_env()
        self._init_models()
        self._init_buffer()
        self._init_planner()
        self._init_optimizer()

        self.writer = SummaryWriter(log_dir=os.path.join(run_dir, "tb"))
        self.global_step = 0

    def _init_env(self) -> None:
        """初始化向量化环境。"""
        env_cfg = self.config["env"]
        train_cfg = self.config["train"]
        self.n_threads = train_cfg["n_rollout_threads"]

        EnvClass = (
            SubprocVectorMAMuJoCoEnv if self.use_subproc
            else VectorMAMuJoCoEnv
        )
        self.envs = EnvClass(
            n_envs=self.n_threads,
            scenario=env_cfg["scenario"],
            agent_conf=env_cfg["agent_conf"],
            episode_limit=env_cfg["episode_limit"],
        )
        self.n_agents = self.envs.n_agents
        self.obs_dims = self.envs.obs_dims
        self.act_dims = self.envs.act_dims
        self.share_obs_dim = self.envs.share_obs_dim

    def _init_models(self) -> None:
        """初始化世界模型组件、Actor 和 Critic。"""
        wm_cfg = self.config["world_model"]
        algo_cfg = self.config["algo"]
        actor_cfg = self.config["actor"]
        critic_cfg = self.config["critic"]
        latent_dim = wm_cfg["latent_dim"]

        self.encoders = [
            MLPEncoder(
                obs_dim=self.obs_dims[i],
                latent_dim=latent_dim,
                hidden_dims=wm_cfg["hidden_dims"],
                simnorm_dim=wm_cfg["simnorm_dim"],
                device=self.device,
            )
            for i in range(self.n_agents)
        ]

        self.dynamics = DenseDynamics(
            latent_dim=latent_dim,
            action_dim=self.act_dims[0],
            n_agents=self.n_agents,
            hidden_dims=wm_cfg["hidden_dims"],
            simnorm_dim=wm_cfg["simnorm_dim"],
            device=self.device,
        )

        self.reward_model = DenseReward(
            latent_dim=latent_dim,
            action_dim=self.act_dims[0],
            n_agents=self.n_agents,
            num_bins=wm_cfg["num_bins"],
            hidden_dims=wm_cfg["hidden_dims"],
            device=self.device,
        )

        self.reward_processor = TwoHotProcessor(
            num_bins=wm_cfg["num_bins"],
            vmin=wm_cfg["reward_min"],
            vmax=wm_cfg["reward_max"],
            device=self.device,
        )

        self.actors = [
            WorldModelActor(
                latent_dim=latent_dim,
                action_dim=self.act_dims[i],
                hidden_sizes=actor_cfg["hidden_sizes"],
                lr=actor_cfg["lr"],
                log_std_min=actor_cfg["log_std_min"],
                log_std_max=actor_cfg["log_std_max"],
                device=self.device,
            )
            for i in range(self.n_agents)
        ]

        total_act_dim = sum(self.act_dims)
        self.critic = WorldModelCritic(
            joint_latent_dim=latent_dim * self.n_agents,
            joint_action_dim=total_act_dim,
            num_bins=wm_cfg["num_bins"],
            reward_min=wm_cfg["reward_min"],
            reward_max=wm_cfg["reward_max"],
            hidden_sizes=critic_cfg["hidden_sizes"],
            polyak=algo_cfg["polyak"],
            scale_tau=critic_cfg["scale_tau"],
            device=self.device,
        )

    def _init_buffer(self) -> None:
        """初始化经验回放。"""
        algo_cfg = self.config["algo"]
        self.buffer = ReplayBuffer(
            n_agents=self.n_agents,
            obs_dims=self.obs_dims,
            act_dims=self.act_dims,
            share_obs_dim=self.share_obs_dim,
            buffer_size=algo_cfg["buffer_size"],
            batch_size=algo_cfg["batch_size"],
            n_step=algo_cfg["n_step"],
            gamma=algo_cfg["gamma"],
            n_rollout_threads=self.n_threads,
        )

    def _init_planner(self) -> None:
        """初始化 MPPI 规划器。"""
        plan_cfg = self.config["plan"]
        algo_cfg = self.config["algo"]
        self.planner = MPPIPlanner(
            n_agents=self.n_agents,
            act_dims=self.act_dims,
            latent_dim=self.config["world_model"]["latent_dim"],
            horizon=plan_cfg["horizon"],
            iterations=plan_cfg["iterations"],
            num_samples=plan_cfg["num_samples"],
            num_pi_trajs=plan_cfg["num_pi_trajs"],
            num_elites=plan_cfg["num_elites"],
            max_std=plan_cfg["max_std"],
            min_std=plan_cfg["min_std"],
            temperature=plan_cfg["temperature"],
            gamma=algo_cfg["gamma"],
            device=self.device,
        )

    def _init_optimizer(self) -> None:
        """初始化统一优化器（Encoder + Dynamics + Reward + Critic）。"""
        algo_cfg = self.config["algo"]
        lr = algo_cfg["lr"]
        enc_lr = lr * algo_cfg["enc_lr_scale"]

        param_groups = []
        for enc in self.encoders:
            param_groups.append(
                {"params": enc.parameters(), "lr": enc_lr},
            )
        param_groups.append(
            {"params": self.dynamics.parameters(), "lr": lr},
        )
        param_groups.append(
            {"params": self.reward_model.parameters(), "lr": lr},
        )
        param_groups.append(
            {"params": self.critic.parameters(), "lr": lr},
        )
        self.optimizer = torch.optim.Adam(param_groups, lr=lr)

    def run(self) -> None:
        """主训练循环。"""
        train_cfg = self.config["train"]
        algo_cfg = self.config["algo"]
        total_steps = (
            train_cfg["num_env_steps"] // self.n_threads
        )

        obs_list, share_obs = self.envs.reset()
        t0 = [True] * self.n_threads
        episode_rewards = np.zeros(self.n_threads, dtype=np.float32)
        recent_returns: deque[float] = deque(maxlen=20)
        total_episodes = 0
        train_info: dict[str, float] = {}
        step_timer = time.time()

        print("=== Warmup: 收集随机数据 ===")
        obs_list, share_obs, t0 = self._warmup(
            obs_list, share_obs, t0, episode_rewards, recent_returns,
        )

        if train_cfg["warmup_train"]:
            print("=== Warmup: 预训练世界模型 ===")
            for _ in range(train_cfg["warmup_train_steps"]):
                self._model_train()

        print("=== 开始主训练 ===")
        for step in range(1, total_steps + 1):
            self.global_step = step * self.n_threads

            actions = self._plan(obs_list, t0)

            actions_np = self._actions_to_numpy(actions)

            next_obs_list, next_share_obs, rewards, dones, truncs, infos = (
                self.envs.step(actions_np)
            )

            self._insert_buffer(
                obs_list, share_obs, actions_np,
                rewards, dones, truncs, infos,
                next_obs_list, next_share_obs,
            )

            episode_rewards += rewards
            for i in range(self.n_threads):
                if dones[i]:
                    recent_returns.append(float(episode_rewards[i]))
                    total_episodes += 1
                    episode_rewards[i] = 0.0
                    t0[i] = True
                else:
                    t0[i] = False

            obs_list = next_obs_list
            share_obs = next_share_obs

            if (
                step % algo_cfg["train_interval"] == 0
                and self.buffer.can_sample()
            ):
                for _ in range(algo_cfg["update_per_train"]):
                    train_info = self._model_train()
                    if step % algo_cfg["policy_freq"] == 0:
                        self._actor_train()

                self.critic.soft_update()

            if step % train_cfg["log_interval"] == 0:
                elapsed = time.time() - step_timer
                sps = train_cfg["log_interval"] * self.n_threads / max(elapsed, 1e-6)
                self._log(
                    step, total_steps, recent_returns,
                    total_episodes, train_info, sps,
                )
                step_timer = time.time()

            if step % train_cfg["save_interval"] == 0:
                self._save(step)

        self.envs.close()
        self.writer.close()
        print("=== 训练完成 ===")

    def _warmup(
        self,
        obs_list: list,
        share_obs: np.ndarray,
        t0: list[bool],
        episode_rewards: np.ndarray,
        recent_returns: deque,
    ) -> tuple[list, np.ndarray, list[bool]]:
        """收集随机动作数据用于预热。

        :param obs_list: 当前观测
        :param share_obs: 全局观测
        :param t0: episode 开始标志
        :param episode_rewards: 当前 episode 累计奖励
        :param recent_returns: 最近回报
        :return: 更新后的 (obs_list, share_obs, t0)
        """
        train_cfg = self.config["train"]
        warmup_steps = train_cfg["warmup_steps"] // self.n_threads

        for _ in range(warmup_steps):
            actions_np = np.stack(
                [
                    np.stack(
                        [
                            self.envs.act_spaces[i].sample()
                            for i in range(self.n_agents)
                        ],
                        axis=0,
                    )
                    for _ in range(self.n_threads)
                ],
                axis=0,
            )

            next_obs_list, next_share_obs, rewards, dones, truncs, infos = (
                self.envs.step(actions_np)
            )

            self._insert_buffer(
                obs_list, share_obs, actions_np,
                rewards, dones, truncs, infos,
                next_obs_list, next_share_obs,
            )

            episode_rewards += rewards
            for i in range(self.n_threads):
                if dones[i]:
                    recent_returns.append(float(episode_rewards[i]))
                    episode_rewards[i] = 0.0
                    t0[i] = True
                else:
                    t0[i] = False

            obs_list = next_obs_list
            share_obs = next_share_obs

        print(
            f"  Warmup 完成，Buffer 大小: {self.buffer.size}，"
            f"平均回报: {np.mean(recent_returns) if recent_returns else 0:.2f}",
        )
        recent_returns.clear()
        return obs_list, share_obs, t0

    def _actions_to_numpy(
        self,
        actions: list[torch.Tensor],
    ) -> np.ndarray:
        """将 GPU tensor 动作转为 numpy。

        :param actions: 各智能体动作列表
        :return: 形状 (n_threads, n_agents, act_dim) 的 numpy 数组
        """
        agent_actions = torch.stack(actions, dim=1)
        return agent_actions.cpu().numpy()

    def _plan(
        self,
        obs_list: list,
        t0: list[bool],
    ) -> list[torch.Tensor]:
        """用 MPPI 规划动作。

        :param obs_list: 当前观测
        :param t0: episode 开始标志
        :return: 各智能体动作列表
        """
        zs = []
        for i in range(self.n_agents):
            obs_i = np.stack(
                [obs_list[env_idx][i] for env_idx in range(self.n_threads)],
                axis=0,
            )
            obs_t = torch.as_tensor(
                obs_i, dtype=torch.float32, device=self.device,
            )
            z_i = self.encoders[i].encode(obs_t)
            zs.append(z_i)

        return self.planner.plan(
            zs=zs,
            t0=t0,
            dynamics_model=self.dynamics,
            reward_model=self.reward_model,
            reward_processor=self.reward_processor,
            actors=self.actors,
            critic=self.critic,
        )

    def _insert_buffer(
        self,
        obs_list: list,
        share_obs: np.ndarray,
        actions_np: np.ndarray,
        rewards: np.ndarray,
        dones: np.ndarray,
        truncs: np.ndarray,
        infos: list[dict],
        next_obs_list: list,
        next_share_obs: np.ndarray,
    ) -> None:
        """将 transition 插入缓冲区。

        :param obs_list: 当前观测
        :param share_obs: 全局观测
        :param actions_np: 动作
        :param rewards: 奖励
        :param dones: 终止标志
        :param truncs: 截断标志
        :param infos: 环境信息
        :param next_obs_list: 下一步观测
        :param next_share_obs: 下一步全局观测
        """
        obs_per_agent = [
            np.stack(
                [obs_list[env_idx][i] for env_idx in range(self.n_threads)],
                axis=0,
            )
            for i in range(self.n_agents)
        ]
        next_obs_per_agent = [
            np.stack(
                [next_obs_list[env_idx][i] for env_idx in range(self.n_threads)],
                axis=0,
            )
            for i in range(self.n_agents)
        ]
        actions_per_agent = [
            actions_np[:, i, :] for i in range(self.n_agents)
        ]

        terms = np.zeros((self.n_threads, 1), dtype=np.float32)
        for i in range(self.n_threads):
            if dones[i] and not truncs[i]:
                terms[i, 0] = 1.0

        valid = [
            np.ones((self.n_threads, 1), dtype=np.float32)
            for _ in range(self.n_agents)
        ]

        self.buffer.insert(
            share_obs=share_obs,
            obs=obs_per_agent,
            actions=actions_per_agent,
            rewards=rewards.reshape(-1, 1),
            dones=dones.astype(np.float32).reshape(-1, 1),
            terms=terms,
            valid=valid,
            next_share_obs=next_share_obs,
            next_obs=next_obs_per_agent,
        )

    def _model_train(self) -> dict[str, float]:
        """世界模型训练一步。

        :return: 训练指标字典
        """
        wm_cfg = self.config["world_model"]
        horizon = wm_cfg["horizon"]
        step_rho = wm_cfg["step_rho"]

        batch = self.buffer.sample_horizon(horizon)
        nstep_batch = self.buffer.sample()

        obs_h = [
            torch.as_tensor(batch["obs"][i], dtype=torch.float32, device=self.device)
            for i in range(self.n_agents)
        ]
        actions_h = [
            torch.as_tensor(batch["actions"][i], dtype=torch.float32, device=self.device)
            for i in range(self.n_agents)
        ]
        next_obs_h = [
            torch.as_tensor(batch["next_obs"][i], dtype=torch.float32, device=self.device)
            for i in range(self.n_agents)
        ]
        rewards_h = torch.as_tensor(
            batch["rewards"], dtype=torch.float32, device=self.device,
        )

        nstep_reward = torch.as_tensor(
            nstep_batch["nstep_reward"], dtype=torch.float32, device=self.device,
        )
        nstep_gamma = torch.as_tensor(
            nstep_batch["nstep_gamma"], dtype=torch.float32, device=self.device,
        )
        nstep_term = torch.as_tensor(
            nstep_batch["nstep_term"], dtype=torch.float32, device=self.device,
        )
        nstep_next_obs = [
            torch.as_tensor(
                nstep_batch["nstep_next_obs"][i],
                dtype=torch.float32, device=self.device,
            )
            for i in range(self.n_agents)
        ]
        batch_obs = [
            torch.as_tensor(
                nstep_batch["obs"][i],
                dtype=torch.float32, device=self.device,
            )
            for i in range(self.n_agents)
        ]
        batch_actions = [
            torch.as_tensor(
                nstep_batch["actions"][i],
                dtype=torch.float32, device=self.device,
            )
            for i in range(self.n_agents)
        ]

        with torch.no_grad():
            nstep_next_zs = [
                self.encoders[i].encode(nstep_next_obs[i])
                for i in range(self.n_agents)
            ]
            joint_next_z = torch.cat(nstep_next_zs, dim=-1)

            next_actions = [
                self.actors[i].get_actions(nstep_next_zs[i])
                for i in range(self.n_agents)
            ]
            joint_next_a = torch.cat(next_actions, dim=-1)

            q_target = self.critic.get_target_values(
                joint_next_z, joint_next_a,
            )
            q_targets = (
                nstep_reward
                + nstep_gamma * q_target * (1 - nstep_term)
            )

        dynamics_loss = torch.tensor(0.0, device=self.device)
        reward_loss = torch.tensor(0.0, device=self.device)

        zs = [
            self.encoders[i].encode(obs_h[i][0])
            for i in range(self.n_agents)
        ]

        all_zs_for_actor = [[] for _ in range(self.n_agents)]

        for t in range(horizon):
            z_joint = torch.stack(zs, dim=1)
            a_joint = torch.stack(
                [actions_h[i][t] for i in range(self.n_agents)], dim=1,
            )

            z_pred = self.dynamics.predict(z_joint, a_joint)
            r_logits = self.reward_model.predict(z_joint, a_joint)

            with torch.no_grad():
                z_true = [
                    self.encoders[i].encode(next_obs_h[i][t])
                    for i in range(self.n_agents)
                ]
                z_true_joint = torch.stack(z_true, dim=1)

            rho_t = step_rho ** t
            dynamics_loss = dynamics_loss + (
                F.mse_loss(z_pred, z_true_joint) * rho_t
            )
            reward_loss = reward_loss + (
                self.reward_processor.loss(
                    r_logits, rewards_h[t],
                ).mean() * rho_t
            )

            for i in range(self.n_agents):
                all_zs_for_actor[i].append(zs[i].detach())

            zs = [z_pred[:, i] for i in range(self.n_agents)]

        dynamics_loss = dynamics_loss / horizon
        reward_loss = reward_loss / horizon

        batch_z = [
            self.encoders[i].encode(batch_obs[i])
            for i in range(self.n_agents)
        ]
        joint_z = torch.cat(batch_z, dim=-1)
        joint_a = torch.cat(batch_actions, dim=-1)

        q1_logits = self.critic.critic(joint_z, joint_a)
        q2_logits = self.critic.critic2(joint_z, joint_a)
        q_loss = (
            self.critic.processor.loss(q1_logits, q_targets).mean()
            + self.critic.processor.loss(q2_logits, q_targets).mean()
        ) / 2

        total_loss = (
            wm_cfg["dynamics_coef"] * dynamics_loss
            + wm_cfg["reward_coef"] * reward_loss
            + wm_cfg["q_coef"] * q_loss
        )

        self.optimizer.zero_grad()
        total_loss.backward()
        for group in self.optimizer.param_groups:
            nn.utils.clip_grad_norm_(
                group["params"],
                self.config["algo"]["grad_clip"],
            )
        self.optimizer.step()

        self._actor_zs = all_zs_for_actor

        return {
            "dynamics_loss": dynamics_loss.item(),
            "reward_loss": reward_loss.item(),
            "q_loss": q_loss.item(),
            "total_loss": total_loss.item(),
        }

    def _actor_train(self) -> dict[str, float]:
        """Actor 训练一步。

        :return: 训练指标字典
        """
        algo_cfg = self.config["algo"]
        wm_cfg = self.config["world_model"]
        step_rho = wm_cfg["step_rho"]
        entropy_coef = algo_cfg["entropy_coef"]

        if not hasattr(self, "_actor_zs") or self._actor_zs is None:
            return {}

        zs_list = self._actor_zs

        agent_order = np.random.permutation(self.n_agents)
        if algo_cfg["fixed_order"]:
            agent_order = np.arange(self.n_agents)

        actor_losses = []

        current_actions = [None] * self.n_agents
        for i in range(self.n_agents):
            with torch.no_grad():
                a, _ = self.actors[i].policy(
                    zs_list[i][0], stochastic=True,
                )
                current_actions[i] = a

        for agent_idx in agent_order:
            self.actors[agent_idx].turn_on_grad()

            actor_loss = torch.tensor(0.0, device=self.device)
            for t in range(len(zs_list[agent_idx])):
                action, log_prob = self.actors[
                    agent_idx
                ].get_actions_with_logprobs(zs_list[agent_idx][t])
                current_actions[agent_idx] = action

                joint_z = torch.cat(
                    [zs_list[i][t] for i in range(self.n_agents)],
                    dim=-1,
                )
                joint_a = torch.cat(current_actions, dim=-1)

                q_value = self.critic.get_values(
                    joint_z, joint_a, mode="mean",
                )

                rho_t = step_rho ** t
                actor_loss = actor_loss + (
                    (entropy_coef * log_prob - q_value).mean() * rho_t
                )

            actor_loss = actor_loss / len(zs_list[agent_idx])

            self.actors[agent_idx].optimizer.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(
                self.actors[agent_idx].policy.parameters(),
                self.config["algo"]["grad_clip"],
            )
            self.actors[agent_idx].optimizer.step()
            self.actors[agent_idx].turn_off_grad()

            with torch.no_grad():
                a, _ = self.actors[agent_idx].policy(
                    zs_list[agent_idx][0], stochastic=True,
                )
                current_actions[agent_idx] = a

            actor_losses.append(actor_loss.item())

        return {"actor_loss": np.mean(actor_losses)}

    def _log(
        self,
        step: int,
        total_steps: int,
        recent_returns: deque,
        total_episodes: int,
        train_info: dict[str, float],
        sps: float,
    ) -> None:
        """记录日志。

        :param step: 当前步数
        :param total_steps: 总步数
        :param recent_returns: 最近完成的 episode 回报
        :param total_episodes: 总完成 episode 数
        :param train_info: 训练指标
        :param sps: 每秒环境步数
        """
        gs = self.global_step
        avg_return = np.mean(recent_returns) if recent_returns else 0.0

        self.writer.add_scalar("train/avg_return", avg_return, gs)
        self.writer.add_scalar("train/total_episodes", total_episodes, gs)
        self.writer.add_scalar("train/buffer_size", self.buffer.size, gs)
        self.writer.add_scalar("train/sps", sps, gs)

        if train_info:
            for k, v in train_info.items():
                self.writer.add_scalar(f"train/{k}", v, gs)

        pct = step / total_steps * 100
        print(
            f"[{step}/{total_steps} ({pct:.1f}%)] "
            f"EnvSteps={gs} | "
            f"Ep={total_episodes} | "
            f"Return={avg_return:.1f} | "
            f"SPS={sps:.0f}",
            end="",
        )

        if train_info:
            print(
                f" | DynL={train_info.get('dynamics_loss', 0):.4f}"
                f" | RewL={train_info.get('reward_loss', 0):.4f}"
                f" | QL={train_info.get('q_loss', 0):.4f}",
            )
        else:
            print()

    def _save(self, step: int) -> None:
        """保存模型检查点。

        :param step: 当前步数
        """
        ckpt_dir = os.path.join(self.run_dir, "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)
        path = os.path.join(ckpt_dir, f"step_{step}.pt")

        state = {
            "step": step,
            "encoders": [e.state_dict() for e in self.encoders],
            "dynamics": self.dynamics.state_dict(),
            "reward_model": self.reward_model.state_dict(),
            "actors": [
                a.policy.state_dict() for a in self.actors
            ],
            "critic": self.critic.critic.state_dict(),
            "critic2": self.critic.critic2.state_dict(),
            "target_critic": self.critic.target_critic.state_dict(),
            "target_critic2": self.critic.target_critic2.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }
        torch.save(state, path)
        print(f"  模型已保存至 {path}")
