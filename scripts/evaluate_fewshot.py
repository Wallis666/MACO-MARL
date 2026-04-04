"""Few-shot 适应评估脚本。

从训练检查点加载冻结模型，收集少量 demo transitions，
通过上下文编码器推断任务嵌入，评估在未见任务上的表现。

用法:
    # Few-shot 评估（held-out 任务）
    conda run -n maco python scripts/evaluate_fewshot.py \
        --checkpoint runs/phase3_ctx/checkpoints/step_60000.pt \
        --config src/config/multitask.json \
        --task cheetah_run_slow \
        --n_context 10 --n_episodes 10 --mode few-shot

    # Oracle 基线（训练任务）
    conda run -n maco python scripts/evaluate_fewshot.py \
        --checkpoint runs/phase3_ctx/checkpoints/step_60000.pt \
        --config src/config/multitask.json \
        --task cheetah_run \
        --n_episodes 10 --mode oracle

    # 随机策略基线
    conda run -n maco python scripts/evaluate_fewshot.py \
        --config src/config/multitask.json \
        --task cheetah_run_slow \
        --n_episodes 10 --mode random
"""
import argparse
import json
import sys

import numpy as np
import torch

sys.path.insert(0, ".")

from gymnasium_robotics import mamujoco_v1

from src.algorithms.actor import GaussianPolicy
from src.algorithms.critic import WorldModelCritic
from src.algorithms.planner import MPPIPlanner
from src.config.tasks import TASK_REGISTRY
from src.envs.mamujoco import MAMuJoCoEnv
from src.models.context_encoder import ContextEncoder
from src.models.dynamics import DenseDynamics
from src.models.encoder import MLPEncoder
from src.models.reward import DenseReward
from src.models.task_embedding import TaskEmbeddingTable
from src.models.utils import TwoHotProcessor


def collect_demos(
    task_name: str,
    n_context: int,
    seed: int = 0,
    models: dict | None = None,
    task_emb_for_demo: torch.Tensor | None = None,
    device: str = "cpu",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """收集 demo transitions。

    当提供 models 和 task_emb_for_demo 时使用训练好的策略收集
    （模拟专家演示），否则使用随机策略。

    :param task_name: 任务名（须在 TASK_REGISTRY 中注册）
    :param n_context: 收集的 transition 数量 K
    :param seed: 随机种子
    :param models: 加载的模型字典（trained 模式需要）
    :param task_emb_for_demo: demo 收集时使用的任务嵌入
    :param device: 设备
    :return: (obs, actions, rewards, next_obs)，各形状 (1, K, dim)
    """
    task_def = TASK_REGISTRY[task_name]
    env = MAMuJoCoEnv(
        scenario=task_def.scenario,
        agent_conf=task_def.agent_conf,
        episode_limit=1000,
        seed=seed,
        reward_fn=task_def.reward_fn,
    )

    use_trained = (
        models is not None and task_emb_for_demo is not None
    )

    obs_list, actions_list, rewards_list, next_obs_list = [], [], [], []
    obs_raw, _ = env.reset()

    collected = 0
    while collected < n_context:
        if use_trained:
            # 用训练好的策略生成动作
            action_parts = []
            for i in range(env.n_agents):
                obs_t = torch.as_tensor(
                    obs_raw[i],
                    dtype=torch.float32,
                    device=device,
                ).unsqueeze(0)
                with torch.no_grad():
                    z_i = models["encoders"][i].encode(
                        obs_t, task_emb_for_demo,
                    )
                    a_i, _ = models["policies"][i](
                        z_i, task_emb_for_demo, stochastic=True,
                    )
                action_parts.append(
                    a_i.squeeze(0).cpu().numpy(),
                )
            action = np.stack(action_parts, axis=0)
        else:
            # 随机动作
            action = np.stack(
                [env.act_spaces[i].sample()
                 for i in range(env.n_agents)],
                axis=0,
            )

        next_obs_raw, _, rewards, term, trunc, _ = env.step(action)

        # 记录 agent 0 的 transition
        obs_list.append(obs_raw[0].astype(np.float32))
        actions_list.append(action[0].astype(np.float32))
        rewards_list.append(
            np.array([rewards.mean()], dtype=np.float32),
        )
        next_obs_list.append(next_obs_raw[0].astype(np.float32))
        collected += 1

        if term or trunc:
            obs_raw, _ = env.reset()
        else:
            obs_raw = next_obs_raw

    env.close()

    # 堆叠为 (1, K, dim)
    return (
        np.stack(obs_list)[np.newaxis],
        np.stack(actions_list)[np.newaxis],
        np.stack(rewards_list)[np.newaxis],
        np.stack(next_obs_list)[np.newaxis],
    )


def load_all_models(
    config: dict,
    checkpoint_path: str,
    device: str,
) -> dict:
    """加载所有模型组件。

    :param config: 配置字典
    :param checkpoint_path: 检查点路径
    :param device: 设备
    :return: 包含所有模型的字典
    """
    ckpt = torch.load(
        checkpoint_path, map_location=device, weights_only=False,
    )
    print(f"加载检查点: step={ckpt['step']}")

    wm_cfg = config["world_model"]
    actor_cfg = config["actor"]
    critic_cfg = config["critic"]
    task_dim = wm_cfg.get("task_dim", 0)
    env_cfg = config["env"]
    n_tasks = len(env_cfg.get("tasks", []))

    # 用临时环境获取维度信息
    first_task = TASK_REGISTRY[env_cfg["tasks"][0]]
    env_tmp = mamujoco_v1.parallel_env(
        scenario=first_task.scenario,
        agent_conf=first_task.agent_conf,
        max_episode_steps=env_cfg.get("episode_limit", 1000),
    )
    env_tmp.reset()
    agents = env_tmp.possible_agents
    n_agents = len(agents)
    obs_dims = [
        env_tmp.observation_space(a).shape[0] for a in agents
    ]
    act_dims = [
        env_tmp.action_space(a).shape[0] for a in agents
    ]
    env_tmp.close()

    models = {
        "n_agents": n_agents,
        "obs_dims": obs_dims,
        "act_dims": act_dims,
        "task_dim": task_dim,
        "n_tasks": n_tasks,
    }

    # Encoders
    encoders = []
    for i in range(n_agents):
        enc = MLPEncoder(
            obs_dim=obs_dims[i],
            latent_dim=wm_cfg["latent_dim"],
            task_dim=task_dim,
            hidden_dims=wm_cfg["hidden_dims"],
            simnorm_dim=wm_cfg["simnorm_dim"],
            device=device,
        )
        enc.load_state_dict(ckpt["encoders"][i])
        enc.eval()
        encoders.append(enc)
    models["encoders"] = encoders

    # Actors
    policies = []
    for i in range(n_agents):
        policy = GaussianPolicy(
            latent_dim=wm_cfg["latent_dim"],
            action_dim=act_dims[i],
            task_dim=task_dim,
            hidden_sizes=actor_cfg["hidden_sizes"],
            log_std_min=actor_cfg["log_std_min"],
            log_std_max=actor_cfg["log_std_max"],
            device=device,
        )
        policy.load_state_dict(ckpt["actors"][i])
        policy.eval()
        policies.append(policy)
    models["policies"] = policies

    # TaskEmbeddingTable
    if task_dim > 0 and "task_embedding" in ckpt:
        task_emb_table = TaskEmbeddingTable(
            n_tasks=n_tasks,
            task_dim=task_dim,
        ).to(device)
        task_emb_table.load_state_dict(ckpt["task_embedding"])
        task_emb_table.eval()
        models["task_embedding"] = task_emb_table
    else:
        models["task_embedding"] = None

    # ContextEncoder
    if task_dim > 0 and "context_encoder" in ckpt:
        ctx_cfg = config.get("context_encoder", {})
        ctx_enc = ContextEncoder(
            obs_dim=obs_dims[0],
            action_dim=act_dims[0],
            task_dim=task_dim,
            hidden_dims=ctx_cfg.get("hidden_dims", [256, 256]),
            device=device,
        )
        ctx_enc.load_state_dict(ckpt["context_encoder"])
        ctx_enc.eval()
        models["context_encoder"] = ctx_enc
    else:
        models["context_encoder"] = None

    # Dynamics（MPPI 模式需要）
    dynamics = DenseDynamics(
        latent_dim=wm_cfg["latent_dim"],
        action_dim=act_dims[0],
        n_agents=n_agents,
        task_dim=task_dim,
        hidden_dims=wm_cfg["hidden_dims"],
        simnorm_dim=wm_cfg["simnorm_dim"],
        device=device,
    )
    dynamics.load_state_dict(ckpt["dynamics"])
    dynamics.eval()
    models["dynamics"] = dynamics

    # Reward（MPPI 模式需要）
    reward_model = DenseReward(
        latent_dim=wm_cfg["latent_dim"],
        action_dim=act_dims[0],
        n_agents=n_agents,
        task_dim=task_dim,
        num_bins=wm_cfg["num_bins"],
        hidden_dims=wm_cfg["hidden_dims"],
        device=device,
    )
    reward_model.load_state_dict(ckpt["reward_model"])
    reward_model.eval()
    models["reward_model"] = reward_model

    # Critic（MPPI 模式需要）
    total_act_dim = sum(act_dims)
    critic = WorldModelCritic(
        joint_latent_dim=wm_cfg["latent_dim"] * n_agents,
        joint_action_dim=total_act_dim,
        task_dim=task_dim,
        num_bins=wm_cfg["num_bins"],
        reward_min=wm_cfg["reward_min"],
        reward_max=wm_cfg["reward_max"],
        hidden_sizes=critic_cfg["hidden_sizes"],
        device=device,
    )
    critic.critic.load_state_dict(ckpt["critic"])
    critic.critic2.load_state_dict(ckpt["critic2"])
    critic.target_critic.load_state_dict(ckpt["target_critic"])
    critic.target_critic2.load_state_dict(ckpt["target_critic2"])
    models["critic"] = critic

    # TwoHotProcessor
    models["reward_processor"] = TwoHotProcessor(
        num_bins=wm_cfg["num_bins"],
        vmin=wm_cfg["reward_min"],
        vmax=wm_cfg["reward_max"],
        device=device,
    )

    return models


def infer_task_embedding(
    context_encoder: ContextEncoder,
    obs: np.ndarray,
    actions: np.ndarray,
    rewards: np.ndarray,
    next_obs: np.ndarray,
    device: str,
) -> torch.Tensor:
    """用上下文编码器从 demo transitions 推断任务嵌入。

    :param context_encoder: 上下文编码器
    :param obs: (1, K, obs_dim)
    :param actions: (1, K, act_dim)
    :param rewards: (1, K, 1)
    :param next_obs: (1, K, obs_dim)
    :param device: 设备
    :return: 任务嵌入 (1, task_dim)
    """
    with torch.no_grad():
        z_task = context_encoder.encode_transitions(
            torch.as_tensor(obs, dtype=torch.float32, device=device),
            torch.as_tensor(
                actions, dtype=torch.float32, device=device,
            ),
            torch.as_tensor(
                rewards, dtype=torch.float32, device=device,
            ),
            torch.as_tensor(
                next_obs, dtype=torch.float32, device=device,
            ),
        )
    return z_task


def run_episodes(
    models: dict,
    task_name: str,
    task_emb: torch.Tensor | None,
    n_episodes: int,
    device: str,
    deterministic: bool = True,
    use_mppi: bool = False,
    config: dict | None = None,
) -> list[float]:
    """运行评估 episodes。

    :param models: 模型字典
    :param task_name: 任务名
    :param task_emb: 任务嵌入 (1, task_dim) 或 None
    :param n_episodes: episode 数
    :param device: 设备
    :param deterministic: 是否确定性策略
    :param use_mppi: 是否使用 MPPI 规划器
    :param config: 配置字典（MPPI 模式需要）
    :return: 每 episode 的回报列表
    """
    task_def = TASK_REGISTRY[task_name]
    env = MAMuJoCoEnv(
        scenario=task_def.scenario,
        agent_conf=task_def.agent_conf,
        episode_limit=1000,
        reward_fn=task_def.reward_fn,
    )

    encoders = models["encoders"]
    policies = models["policies"]
    n_agents = models["n_agents"]

    # MPPI 规划器（可选）
    planner = None
    if use_mppi and config is not None:
        plan_cfg = config["plan"]
        planner = MPPIPlanner(
            n_agents=n_agents,
            act_dims=models["act_dims"],
            latent_dim=config["world_model"]["latent_dim"],
            horizon=plan_cfg["horizon"],
            iterations=plan_cfg["iterations"],
            num_samples=plan_cfg["num_samples"],
            num_pi_trajs=plan_cfg["num_pi_trajs"],
            num_elites=plan_cfg["num_elites"],
            max_std=plan_cfg["max_std"],
            min_std=plan_cfg["min_std"],
            temperature=plan_cfg["temperature"],
            gamma=config["algo"]["gamma"],
            device=device,
        )

    from src.algorithms.actor import WorldModelActor
    actors_for_mppi = None
    if planner is not None:
        actors_for_mppi = [
            WorldModelActor(
                latent_dim=config["world_model"]["latent_dim"],
                action_dim=models["act_dims"][i],
                task_dim=models["task_dim"],
                device=device,
            )
            for i in range(n_agents)
        ]
        for i in range(n_agents):
            actors_for_mppi[i].policy = policies[i]

    all_returns = []

    for ep in range(n_episodes):
        obs_list, _ = env.reset()
        ep_reward = 0.0
        step = 0
        t0 = [True]

        while True:
            # 编码观测
            zs = []
            for i in range(n_agents):
                obs_t = torch.as_tensor(
                    obs_list[i],
                    dtype=torch.float32,
                    device=device,
                ).unsqueeze(0)
                with torch.no_grad():
                    z_i = encoders[i].encode(obs_t, task_emb)
                zs.append(z_i)

            if planner is not None:
                # MPPI 规划
                with torch.no_grad():
                    actions_t = planner.plan(
                        zs=zs,
                        t0=t0,
                        dynamics_model=models["dynamics"],
                        reward_model=models["reward_model"],
                        reward_processor=models["reward_processor"],
                        actors=actors_for_mppi,
                        critic=models["critic"],
                        task_emb=task_emb,
                    )
                actions_np = np.stack(
                    [a.squeeze(0).cpu().numpy() for a in actions_t],
                    axis=0,
                )
            else:
                # Actor 直接推理
                actions_np_list = []
                for i in range(n_agents):
                    with torch.no_grad():
                        a, _ = policies[i](
                            zs[i], task_emb,
                            stochastic=not deterministic,
                        )
                    actions_np_list.append(
                        a.squeeze(0).cpu().numpy(),
                    )
                actions_np = np.stack(actions_np_list, axis=0)

            next_obs_list, _, rewards, term, trunc, _ = env.step(
                actions_np,
            )
            ep_reward += float(rewards.mean())
            step += 1
            t0 = [False]

            if term or trunc:
                break

            obs_list = next_obs_list

        all_returns.append(ep_reward)
        print(
            f"  Episode {ep + 1}/{n_episodes}: "
            f"Return={ep_reward:.2f}, Steps={step}",
        )

    env.close()
    return all_returns


def main() -> None:
    """主入口。"""
    parser = argparse.ArgumentParser(
        description="Few-shot 适应评估",
    )
    parser.add_argument(
        "--config", type=str, required=True,
        help="配置文件路径",
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="检查点路径（random 模式可不填）",
    )
    parser.add_argument(
        "--task", type=str, required=True,
        help="评估任务名",
    )
    parser.add_argument(
        "--n_context", type=int, default=10,
        help="demo transition 数量 K",
    )
    parser.add_argument(
        "--n_episodes", type=int, default=10,
        help="评估 episode 数",
    )
    parser.add_argument(
        "--mode", type=str, default="few-shot",
        choices=["few-shot", "few-shot-mppi", "oracle", "random"],
        help="评估模式",
    )
    parser.add_argument(
        "--device", type=str, default="cpu",
    )
    parser.add_argument(
        "--deterministic", action="store_true",
        help="使用确定性策略",
    )
    parser.add_argument(
        "--seed", type=int, default=0,
    )
    parser.add_argument(
        "--demo_policy", type=str, default="random",
        choices=["random", "trained"],
        help="demo 收集策略：random（随机）或 trained（用训练策略）",
    )
    parser.add_argument(
        "--demo_task", type=str, default=None,
        help="trained 模式下用哪个训练任务的嵌入收集 demo"
             "（默认用第一个训练任务）",
    )
    args = parser.parse_args()

    # 验证任务存在
    if args.task not in TASK_REGISTRY:
        available = ", ".join(sorted(TASK_REGISTRY.keys()))
        raise ValueError(f"未知任务 {args.task!r}，可用：{available}")

    with open(args.config, "r") as f:
        config = json.load(f)

    print(f"任务: {args.task}")
    print(f"模式: {args.mode}")

    # === random 模式：无需加载模型 ===
    if args.mode == "random":
        task_def = TASK_REGISTRY[args.task]
        env = MAMuJoCoEnv(
            scenario=task_def.scenario,
            agent_conf=task_def.agent_conf,
            episode_limit=1000,
            seed=args.seed,
            reward_fn=task_def.reward_fn,
        )
        all_returns = []
        for ep in range(args.n_episodes):
            obs_list, _ = env.reset()
            ep_reward = 0.0
            step = 0
            while True:
                action = np.stack(
                    [env.act_spaces[i].sample()
                     for i in range(env.n_agents)],
                    axis=0,
                )
                _, _, rewards, term, trunc, _ = env.step(action)
                ep_reward += float(rewards.mean())
                step += 1
                if term or trunc:
                    break
            all_returns.append(ep_reward)
            print(
                f"  Episode {ep + 1}/{args.n_episodes}: "
                f"Return={ep_reward:.2f}, Steps={step}",
            )
        env.close()
        avg = np.mean(all_returns)
        std = np.std(all_returns)
        print(f"\n=== Random 基线结果 ===")
        print(f"任务: {args.task}")
        print(f"平均回报: {avg:.2f} +/- {std:.2f}")
        return

    # === 需要检查点的模式 ===
    if args.checkpoint is None:
        raise ValueError(f"{args.mode} 模式需要 --checkpoint 参数")

    models = load_all_models(config, args.checkpoint, args.device)

    # === oracle 模式 ===
    if args.mode == "oracle":
        task_list = config["env"].get("tasks", [])
        if args.task not in task_list:
            raise ValueError(
                f"Oracle 模式仅限训练任务，{args.task!r} "
                f"不在训练列表 {task_list} 中",
            )
        task_idx = task_list.index(args.task)
        with torch.no_grad():
            task_emb = models["task_embedding"](
                torch.tensor([task_idx], device=args.device),
            )
        print(f"Oracle 嵌入: task_idx={task_idx}")

        all_returns = run_episodes(
            models, args.task, task_emb,
            args.n_episodes, args.device,
            deterministic=args.deterministic,
        )

    # === few-shot / few-shot-mppi 模式 ===
    else:
        if models["context_encoder"] is None:
            raise ValueError("检查点中没有 context_encoder")

        # 确定 demo 收集方式
        demo_models = None
        demo_task_emb = None
        if args.demo_policy == "trained":
            task_list = config["env"].get("tasks", [])
            demo_task = args.demo_task or task_list[0]
            if demo_task not in task_list:
                raise ValueError(
                    f"demo_task {demo_task!r} 不在训练列表中",
                )
            demo_idx = task_list.index(demo_task)
            with torch.no_grad():
                demo_task_emb = models["task_embedding"](
                    torch.tensor(
                        [demo_idx], device=args.device,
                    ),
                )
            demo_models = models
            print(
                f"使用训练策略收集 demo "
                f"(嵌入来自 {demo_task}, idx={demo_idx})",
            )

        print(f"收集 {args.n_context} 条 demo transitions...")
        obs, actions, rewards, next_obs = collect_demos(
            args.task, args.n_context, seed=args.seed,
            models=demo_models,
            task_emb_for_demo=demo_task_emb,
            device=args.device,
        )
        print(
            f"  Demo 奖励统计: "
            f"mean={rewards.mean():.4f}, "
            f"std={rewards.std():.4f}, "
            f"min={rewards.min():.4f}, "
            f"max={rewards.max():.4f}",
        )

        task_emb = infer_task_embedding(
            models["context_encoder"],
            obs, actions, rewards, next_obs,
            args.device,
        )
        print(f"推断嵌入范数: {task_emb.norm().item():.4f}")

        # 与训练任务嵌入的余弦相似度
        # 与训练任务嵌入的余弦相似度
        if models["task_embedding"] is not None:
            with torch.no_grad():
                for i, tname in enumerate(
                    config["env"].get("tasks", []),
                ):
                    train_emb = models["task_embedding"](
                        torch.tensor([i], device=args.device),
                    )
                    cos_sim = torch.nn.functional.cosine_similarity(
                        task_emb, train_emb,
                    ).item()
                    print(f"  与 {tname} 的余弦相似度: {cos_sim:.4f}")

        use_mppi = args.mode == "few-shot-mppi"
        all_returns = run_episodes(
            models, args.task, task_emb,
            args.n_episodes, args.device,
            deterministic=args.deterministic,
            use_mppi=use_mppi,
            config=config if use_mppi else None,
        )

    # 输出结果
    avg = np.mean(all_returns)
    std = np.std(all_returns)
    mode_desc = args.mode
    if "few-shot" in args.mode:
        mode_desc += f" (K={args.n_context})"

    print(f"\n=== Few-shot 评估结果 ===")
    print(f"任务: {args.task}")
    print(f"模式: {mode_desc}")
    print(f"每 episode 回报: {all_returns}")
    print(f"平均回报: {avg:.2f} +/- {std:.2f}")


if __name__ == "__main__":
    main()
