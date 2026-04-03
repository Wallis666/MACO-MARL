"""生成烟雾测试用的极小配置。"""
import json

with open("src/config/multitask.json") as f:
    cfg = json.load(f)

cfg["train"]["n_rollout_threads"] = 4
cfg["train"]["num_env_steps"] = 200
cfg["train"]["warmup_steps"] = 100
cfg["train"]["warmup_train"] = True
cfg["train"]["warmup_train_steps"] = 5
cfg["train"]["log_interval"] = 50
cfg["train"]["save_interval"] = 99999
cfg["algo"]["batch_size"] = 64
cfg["algo"]["buffer_size"] = 2000

with open("src/config/multitask_smoke.json", "w") as f:
    json.dump(cfg, f, indent=4)
print("烟雾测试配置已生成")
