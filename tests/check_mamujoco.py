from gymnasium_robotics import mamujoco_v1
import numpy as np

env = mamujoco_v1.parallel_env("HalfCheetah", "2x3", max_episode_steps=1000)
obs, info = env.reset(seed=42)

print("Agents:", env.possible_agents)
for a in env.possible_agents:
    print(f"  {a}: obs_shape={env.observation_space(a).shape}, act_shape={env.action_space(a).shape}")

actions = {a: env.action_space(a).sample() for a in env.possible_agents}
next_obs, rewards, terms, truncs, infos = env.step(actions)
print("Rewards:", rewards)
print("Terms:", terms)
print("Truncs:", truncs)
print("Step OK")
env.close()
