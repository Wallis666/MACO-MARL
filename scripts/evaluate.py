"""评估脚本：加载检查点并渲染 episode 视频。

用法:
    conda run -n maco python scripts/evaluate.py \
        --checkpoint runs/phase0/checkpoints/step_50000.pt \
        --config src/config/default.json \
        --episodes 3 \
        --output runs/phase0/eval_50000.mp4
"""
import argparse
import json
import sys

import numpy as np
import torch

sys.path.insert(0, ".")

from gymnasium_robotics import mamujoco_v1

from src.algorithms.actor import GaussianPolicy
from src.models.encoder import MLPEncoder


def load_models(
    checkpoint_path: str,
    config: dict,
    device: str,
) -> tuple[list[MLPEncoder], list[GaussianPolicy]]:
    """加载编码器和策略网络。

    :param checkpoint_path: 检查点路径
    :param config: 配置字典
    :param device: 设备
    :return: (encoders, policies)
    """
    ckpt = torch.load(checkpoint_path, map_location=device)
    print(f"加载检查点: step={ckpt['step']}")

    wm_cfg = config["world_model"]
    actor_cfg = config["actor"]
    env_cfg = config["env"]

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

    encoders = []
    for i in range(n_agents):
        enc = MLPEncoder(
            obs_dim=obs_dims[i],
            latent_dim=wm_cfg["latent_dim"],
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
            hidden_sizes=actor_cfg["hidden_sizes"],
            log_std_min=actor_cfg["log_std_min"],
            log_std_max=actor_cfg["log_std_max"],
            device=device,
        )
        policy.load_state_dict(ckpt["actors"][i])
        policy.eval()
        policies.append(policy)

    return encoders, policies


def evaluate(
    config: dict,
    encoders: list[MLPEncoder],
    policies: list[GaussianPolicy],
    num_episodes: int,
    device: str,
    output_path: str | None,
    stochastic: bool,
) -> None:
    """运行评估并保存视频。

    :param config: 配置字典
    :param encoders: 编码器列表
    :param policies: 策略网络列表
    :param num_episodes: 评估 episode 数
    :param device: 设备
    :param output_path: 视频输出路径
    :param stochastic: 是否使用随机策略
    """
    env_cfg = config["env"]

    render_mode = "rgb_array" if output_path else "human"
    env = mamujoco_v1.parallel_env(
        scenario=env_cfg["scenario"],
        agent_conf=env_cfg["agent_conf"],
        max_episode_steps=env_cfg["episode_limit"],
        render_mode=render_mode,
    )

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
                    z = encoders[i].encode(obs_t)
                    action, _ = policies[i](z, stochastic=stochastic)
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
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = json.load(f)

    encoders, policies = load_models(
        args.checkpoint, config, args.device,
    )

    evaluate(
        config=config,
        encoders=encoders,
        policies=policies,
        num_episodes=args.episodes,
        device=args.device,
        output_path=args.output,
        stochastic=not args.deterministic,
    )


if __name__ == "__main__":
    main()
