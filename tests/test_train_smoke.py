"""烟雾测试：验证训练流程的前几步能正常运行。"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(line_buffering=True)

import json
import numpy as np
import torch

# 使用小配置快速测试
config = {
    "algo": {
        "batch_size": 64,
        "buffer_size": 10000,
        "gamma": 0.995,
        "n_step": 5,
        "polyak": 0.01,
        "policy_freq": 1,
        "fixed_order": False,
        "entropy_coef": 1e-4,
        "grad_clip": 20.0,
        "lr": 3e-4,
        "enc_lr_scale": 0.3,
        "update_per_train": 1,
        "train_interval": 1,
    },
    "world_model": {
        "latent_dim": 64,
        "hidden_dims": [128, 128],
        "num_bins": 51,
        "reward_min": -10,
        "reward_max": 10,
        "simnorm_dim": 8,
        "step_rho": 0.5,
        "dynamics_coef": 20.0,
        "reward_coef": 0.1,
        "q_coef": 0.1,
        "horizon": 3,
    },
    "plan": {
        "horizon": 2,
        "iterations": 2,
        "num_samples": 16,
        "num_pi_trajs": 4,
        "num_elites": 4,
        "max_std": 1.0,
        "min_std": 0.05,
        "temperature": 0.5,
    },
    "train": {
        "n_rollout_threads": 2,
        "num_env_steps": 1000000,
        "warmup_steps": 200,
        "warmup_train": True,
        "warmup_train_steps": 10,
        "log_interval": 5,
        "eval_interval": 100000,
        "save_interval": 100000,
        "use_linear_lr_decay": False,
    },
    "actor": {
        "hidden_sizes": [64, 64],
        "log_std_min": -10,
        "log_std_max": 2,
        "lr": 3e-4,
    },
    "critic": {
        "hidden_sizes": [128, 128],
        "scale_tau": 0.01,
    },
    "env": {
        "scenario": "HalfCheetah",
        "agent_conf": "2x3",
        "episode_limit": 200,
    },
}

# 写入临时配置
config_path = "tests/test_config.json"
with open(config_path, "w") as f:
    json.dump(config, f)

print("=== 初始化 Trainer ===")
from src.runner.trainer import Trainer
trainer = Trainer(config_path=config_path, device="cpu", run_dir="runs/test")
print(f"  环境: {config['env']['scenario']}-{config['env']['agent_conf']}")
print(f"  智能体数: {trainer.n_agents}")
print(f"  观测维度: {trainer.obs_dims}")
print(f"  动作维度: {trainer.act_dims}")

print("\n=== 测试 Warmup ===")
obs_list, share_obs = trainer.envs.reset()
t0 = [True] * trainer.n_threads
episode_rewards = np.zeros(trainer.n_threads, dtype=np.float32)
done_rewards = []

obs_list, share_obs, t0 = trainer._warmup(
    obs_list, share_obs, t0, episode_rewards, done_rewards,
)
print(f"  Buffer 大小: {trainer.buffer.size}")

print("\n=== 测试 Warmup Train ===")
for i in range(5):
    info = trainer._model_train()
    print(f"  Step {i}: dyn_loss={info['dynamics_loss']:.4f} rew_loss={info['reward_loss']:.4f} q_loss={info['q_loss']:.4f}")

print("\n=== 测试 MPPI Planning ===")
actions = trainer._plan(obs_list, t0)
print(f"  动作数: {len(actions)}, 形状: {[a.shape for a in actions]}")

print("\n=== 测试完整 Rollout+Train 循环 ===")
for step in range(10):
    actions = trainer._plan(obs_list, t0)
    actions_np = np.stack(
        [
            np.stack(
                [actions[i][env_idx].cpu().numpy() for i in range(trainer.n_agents)],
                axis=0,
            )
            for env_idx in range(trainer.n_threads)
        ],
        axis=0,
    )

    next_obs_list, next_share_obs, rewards, dones, truncs, infos = trainer.envs.step(actions_np)

    trainer._insert_buffer(
        obs_list, share_obs, actions_np,
        rewards, dones, truncs, infos,
        next_obs_list, next_share_obs,
    )

    episode_rewards += rewards
    for i in range(trainer.n_threads):
        if dones[i]:
            done_rewards.append(episode_rewards[i])
            episode_rewards[i] = 0.0
            t0[i] = True
        else:
            t0[i] = False

    obs_list = next_obs_list
    share_obs = next_share_obs

    train_info = trainer._model_train()
    actor_info = trainer._actor_train()

    if step % 5 == 0:
        print(f"  Step {step}: dyn={train_info['dynamics_loss']:.4f} rew={train_info['reward_loss']:.4f} actor={actor_info.get('actor_loss', 0):.4f}")

trainer.envs.close()
print("\n=== 烟雾测试通过！ ===")
