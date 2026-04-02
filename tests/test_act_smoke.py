"""烟雾测试：验证 Actor 采样的训练流程。"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(line_buffering=True)

import json
import numpy as np

config = {
    "algo": {"batch_size": 64, "buffer_size": 10000, "gamma": 0.995,
             "n_step": 5, "polyak": 0.01, "policy_freq": 1,
             "fixed_order": False, "entropy_coef": 1e-4, "grad_clip": 20.0,
             "lr": 3e-4, "enc_lr_scale": 0.3, "update_per_train": 2,
             "train_interval": 1},
    "world_model": {"latent_dim": 64, "hidden_dims": [128, 128],
                    "num_bins": 51, "reward_min": -10, "reward_max": 10,
                    "simnorm_dim": 8, "step_rho": 0.5, "dynamics_coef": 20.0,
                    "reward_coef": 0.1, "q_coef": 0.1, "horizon": 3},
    "plan": {"horizon": 2, "iterations": 2, "num_samples": 16,
             "num_pi_trajs": 4, "num_elites": 4, "max_std": 1.0,
             "min_std": 0.05, "temperature": 0.5},
    "train": {"n_rollout_threads": 2, "num_env_steps": 1000000,
              "warmup_steps": 200, "warmup_train": True,
              "warmup_train_steps": 5, "log_interval": 5,
              "eval_interval": 100000, "save_interval": 100000,
              "use_linear_lr_decay": False},
    "actor": {"hidden_sizes": [64, 64], "log_std_min": -10,
              "log_std_max": 2, "lr": 3e-4},
    "critic": {"hidden_sizes": [128, 128], "scale_tau": 0.01},
    "env": {"scenario": "HalfCheetah", "agent_conf": "2x3",
            "episode_limit": 100},
}

cfg_path = "tests/_tmp_cfg.json"
with open(cfg_path, "w") as f:
    json.dump(config, f)

from src.runner.trainer import Trainer

print("=== 初始化 ===")
t = Trainer(cfg_path, device="cpu", run_dir="runs/_test", use_subproc=False)

obs, share = t.envs.reset()
ep_r = np.zeros(2, dtype=np.float32)

print("=== Warmup ===")
for _ in range(50):
    a = np.stack([np.stack([t.envs.act_spaces[i].sample()
                            for i in range(2)]) for _ in range(2)])
    no, ns, r, d, tr, inf = t.envs.step(a)
    t._insert_buffer(obs, share, a, r, d, tr, inf, no, ns)
    ep_r += r
    for i in range(2):
        if d[i]:
            ep_r[i] = 0
    obs, share = no, ns
print(f"  Buffer: {t.buffer.size}")

print("=== Actor 采样 + 训练 ===")
for s in range(10):
    actions = t._act(obs)
    anp = t._actions_to_numpy(actions)
    no, ns, r, d, tr, inf = t.envs.step(anp)
    t._insert_buffer(obs, share, anp, r, d, tr, inf, no, ns)
    obs, share = no, ns
    info = t._model_train()
    t._actor_train()
    if s % 5 == 0:
        print(f"  Step {s}: dyn={info['dynamics_loss']:.4f} "
              f"rew={info['reward_loss']:.4f} q={info['q_loss']:.4f}")

t.envs.close()
print("=== Actor 采样烟雾测试通过！===")
