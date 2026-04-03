import os
os.environ.setdefault("MUJOCO_GL", "egl")

"""评估脚本：加载检查点并渲染 episode 视频。

用法:
    conda run -n maco python scripts/evaluate.py \
        --checkpoint runs/phase0/checkpoints/step_50000.pt \
        --config src/config/default.json \
        --episodes 3 \
        --output runs/phase0/eval_50000.mp4

    # 多任务配置下评估指定任务:
    conda run -n maco python scripts/evaluate.py \
        --checkpoint runs/phase0/checkpoints/step_60000.pt \
        --config src/config/multitask.json \
        --task cheetah_run_backwards
"""
import argparse
import json
import sys

import numpy as np
import torch

sys.path.insert(0, ".")

from gymnasium_robotics import mamujoco_v1

from src.algorithms.actor import GaussianPolicy
from src.config.tasks import TASK_REGISTRY
from src.models.encoder import MLPEncoder
from src.models.task_embedding import TaskEmbeddingTable


def _resolve_env_config(
    config: dict,
    task_name: str | None,
) -> dict:
    """从配置中解析出环境参数（scenario、agent_conf、episode_limit）。

    支持单任务配置（env 中直接含 scenario/agent_conf）和多任务配置
    （env.mode == "multitask"，需通过 --task 指定）。

    :param config: 完整配置字典
    :param task_name: 指定的任务名（多任务时必填）
    :return: 包含 scenario、agent_conf、episode_limit 的字典
    """
    env_cfg = config["env"]

    if env_cfg.get("mode") == "multitask":
        task_list = env_cfg.get("tasks", [])
        if task_name is None:
            available = ", ".join(task_list)
            raise ValueError(
                f"多任务配置需要 --task 参数，可用任务：{available}",
            )
        if task_name not in TASK_REGISTRY:
            available = ", ".join(sorted(TASK_REGISTRY.keys()))
            raise ValueError(
                f"未知任务 {task_name!r}，可用任务：{available}",
            )
        if task_name not in task_list:
            raise ValueError(
                f"任务 {task_name!r} 不在配置的 tasks 列表中: {task_list}",
            )
        task_def = TASK_REGISTRY[task_name]
        task_idx = task_list.index(task_name)
        return {
            "scenario": task_def.scenario,
            "agent_conf": task_def.agent_conf,
            "episode_limit": env_cfg.get("episode_limit", 1000),
            "n_tasks": len(task_list),
            "task_idx": task_idx,
        }

    if task_name is not None:
        if task_name not in TASK_REGISTRY:
            available = ", ".join(sorted(TASK_REGISTRY.keys()))
            raise ValueError(
                f"未知任务 {task_name!r}，可用任务：{available}",
            )
        task_def = TASK_REGISTRY[task_name]
        return {
            "scenario": task_def.scenario,
            "agent_conf": task_def.agent_conf,
            "episode_limit": env_cfg.get("episode_limit", 1000),
            "n_tasks": 0,
            "task_idx": -1,
        }

    return {
        "scenario": env_cfg["scenario"],
        "agent_conf": env_cfg["agent_conf"],
        "episode_limit": env_cfg.get("episode_limit", 1000),
        "n_tasks": 0,
        "task_idx": -1,
    }


def load_models(
    checkpoint_path: str,
    config: dict,
    device: str,
    resolved_env: dict | None = None,
) -> tuple[list[MLPEncoder], list[GaussianPolicy], TaskEmbeddingTable | None]:
    """加载编码器、策略网络和任务嵌入表。

    :param checkpoint_path: 检查点路径
    :param config: 配置字典
    :param device: 设备
    :param resolved_env: 已解析的环境参数
    :return: (encoders, policies, task_embedding)
    """
    ckpt = torch.load(checkpoint_path, map_location=device)
    print(f"加载检查点: step={ckpt['step']}")

    wm_cfg = config["world_model"]
    actor_cfg = config["actor"]
    env_cfg = resolved_env or config["env"]

    env_tmp = mamujoco_v1.parallel_env(
        scenario=env_cfg["scenario"],
        agent_conf=env_cfg["agent_conf"],
        max_episode_steps=env_cfg["episode_limit"],
    )
    env_tmp.reset()
    agents = env_tmp.possible_agents
    n_agents = len(agents)
    obs_dims = [env_tmp.observation_space(a).shape[0] for a in agents]
    act_dims = [env_tmp.action_space(a).shape[0] for a in agents]
    env_tmp.close()

    task_dim = wm_cfg.get("task_dim", 0)
    n_tasks = env_cfg.get("n_tasks", 0)

    # 加载任务嵌入表（阶段 2+）
    task_embedding = None
    if task_dim > 0 and n_tasks > 0 and "task_embedding" in ckpt:
        task_embedding = TaskEmbeddingTable(
            n_tasks=n_tasks,
            task_dim=task_dim,
        ).to(device)
        task_embedding.load_state_dict(ckpt["task_embedding"])
        task_embedding.eval()
        print(f"任务嵌入: n_tasks={n_tasks}, task_dim={task_dim}")

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

    return encoders, policies, task_embedding


def evaluate(
    config: dict,
    encoders: list[MLPEncoder],
    policies: list[GaussianPolicy],
    task_embedding: TaskEmbeddingTable | None,
    num_episodes: int,
    device: str,
    output_path: str | None,
    stochastic: bool,
    resolved_env: dict | None = None,
) -> None:
    """运行评估并保存视频。

    :param config: 配置字典
    :param encoders: 编码器列表
    :param policies: 策略网络列表
    :param task_embedding: 任务嵌入表（多任务时非 None）
    :param num_episodes: 评估 episode 数
    :param device: 设备
    :param output_path: 视频输出路径
    :param stochastic: 是否使用随机策略
    :param resolved_env: 已解析的环境参数
    """
    env_cfg = resolved_env or config["env"]

    render_mode = "rgb_array" if output_path else "human"
    env = mamujoco_v1.parallel_env(
        scenario=env_cfg["scenario"],
        agent_conf=env_cfg["agent_conf"],
        max_episode_steps=env_cfg["episode_limit"],
        render_mode=render_mode,
    )

    # 获取任务嵌入（阶段 2+）
    task_emb = None
    task_idx = env_cfg.get("task_idx", -1)
    if task_embedding is not None and task_idx >= 0:
        with torch.no_grad():
            task_id = torch.tensor([task_idx], device=device)
            task_emb = task_embedding(task_id)  # (1, task_dim)

    frames = []
    all_returns = []

    for ep in range(num_episodes):
        raw_obs, _ = env.reset()
        agents = env.possible_agents
        n_agents = len(agents)
        ep_reward = 0.0
        step = 0

        while True:
            if output_path:
                frame = env.render()
                if frame is not None:
                    frames.append(frame)

            actions = {}
            for i, agent in enumerate(agents):
                obs_t = torch.tensor(
                    raw_obs[agent],
                    dtype=torch.float32,
                    device=device,
                ).unsqueeze(0)
                with torch.no_grad():
                    z = encoders[i].encode(obs_t, task_emb)
                    action, _ = policies[i](
                        z, task_emb, stochastic=stochastic,
                    )
                actions[agent] = action.squeeze(0).cpu().numpy()

            raw_obs, raw_rewards, raw_terms, raw_truncs, raw_infos = (
                env.step(actions)
            )

            ep_reward += sum(
                raw_rewards[a] for a in agents
            ) / n_agents
            step += 1

            if any(raw_terms.values()) or any(raw_truncs.values()):
                break

        all_returns.append(ep_reward)
        print(
            f"Episode {ep + 1}/{num_episodes}: "
            f"Return={ep_reward:.2f}, Steps={step}",
        )

    env.close()

    print(f"\n平均回报: {np.mean(all_returns):.2f} +/- {np.std(all_returns):.2f}")

    if output_path and frames:
        save_video(frames, output_path)
        print(f"视频已保存至: {output_path}")


def save_video(
    frames: list[np.ndarray],
    output_path: str,
    fps: int = 30,
) -> None:
    """将帧序列保存为 MP4 视频。

    :param frames: RGB 帧列表
    :param output_path: 输出路径
    :param fps: 帧率
    """
    try:
        import imageio
        imageio.mimsave(output_path, frames, fps=fps)
    except ImportError:
        try:
            import cv2
            h, w = frames[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
            for frame in frames:
                writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            writer.release()
        except ImportError:
            gif_path = output_path.replace(".mp4", ".gif")
            print(
                f"未安装 imageio 或 cv2，尝试保存为 GIF: {gif_path}",
            )
            from PIL import Image
            imgs = [Image.fromarray(f) for f in frames]
            imgs[0].save(
                gif_path,
                save_all=True,
                append_images=imgs[1:],
                duration=1000 // fps,
                loop=0,
            )
            print(f"GIF 已保存至: {gif_path}")


def main() -> None:
    """主入口。"""
    parser = argparse.ArgumentParser(description="评估已训练模型")
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="检查点路径",
    )
    parser.add_argument(
        "--config", type=str, default="src/config/default.json",
        help="配置文件路径",
    )
    parser.add_argument(
        "--episodes", type=int, default=3,
        help="评估 episode 数",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="视频输出路径 (如 eval.mp4)，不指定则弹窗渲染",
    )
    parser.add_argument(
        "--deterministic", action="store_true",
        help="使用确定性策略",
    )
    parser.add_argument(
        "--device", type=str, default="cpu",
        help="设备",
    )
    parser.add_argument(
        "--task", type=str, default=None,
        help="指定评估的任务名（多任务配置时必填，如 cheetah_run_backwards）",
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = json.load(f)

    resolved_env = _resolve_env_config(config, args.task)
    if args.task:
        print(f"评估任务: {args.task}")

    encoders, policies, task_embedding = load_models(
        args.checkpoint, config, args.device,
        resolved_env=resolved_env,
    )

    evaluate(
        config=config,
        encoders=encoders,
        policies=policies,
        task_embedding=task_embedding,
        num_episodes=args.episodes,
        device=args.device,
        output_path=args.output,
        stochastic=not args.deterministic,
        resolved_env=resolved_env,
    )


if __name__ == "__main__":
    main()
