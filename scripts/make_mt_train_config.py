"""生成 CPU 友好的多任务训练配置。"""
import json

with open("src/config/multitask.json") as f:
    cfg = json.load(f)

# CPU 友好参数：降低并行度和 batch 大小加快迭代速度
cfg["train"]["n_rollout_threads"] = 8
cfg["train"]["num_env_steps"] = 500000
cfg["train"]["warmup_steps"] = 4000
cfg["train"]["warmup_train"] = True
cfg["train"]["warmup_train_steps"] = 2000
cfg["train"]["log_interval"] = 500
cfg["train"]["save_interval"] = 5000
cfg["algo"]["batch_size"] = 2048
cfg["algo"]["buffer_size"] = 500000

with open("src/config/multitask_cpu.json", "w") as f:
    json.dump(cfg, f, indent=4)
print("CPU 多任务训练配置已生成")
