"""多任务基础设施测试。"""
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, ".")

from src.config.tasks import (
    TASK_REGISTRY,
    TaskDef,
    cheetah_run,
    cheetah_run_backwards,
    get_tasks,
    tolerance,
)


# ---------------------------------------------------------------------------
# 1. tolerance 函数测试
# ---------------------------------------------------------------------------

class TestTolerance:
    """tolerance 函数测试。"""

    def test_in_bounds(self) -> None:
        """目标区间内返回 1.0。"""
        assert tolerance(5.0, bounds=(3.0, 7.0)) == 1.0
        assert tolerance(10.0, bounds=(10.0, float("inf"))) == 1.0

    def test_linear_sigmoid(self) -> None:
        """线性 sigmoid 测试。"""
        r = tolerance(
            5.0,
            bounds=(10.0, float("inf")),
            margin=10.0,
            sigmoid="linear",
            value_at_margin=0,
        )
        assert abs(r - 0.5) < 1e-6

    def test_zero_margin(self) -> None:
        """margin=0 时区间外返回 0。"""
        assert tolerance(5.0, bounds=(10.0, 20.0), margin=0) == 0.0

    def test_gaussian_sigmoid(self) -> None:
        """高斯 sigmoid 在区间外平滑衰减。"""
        r = tolerance(
            0.3,
            bounds=(0.4, float("inf")),
            margin=0.2,
            sigmoid="gaussian",
        )
        assert 0.0 < r < 1.0


# ---------------------------------------------------------------------------
# 2. 任务注册表测试
# ---------------------------------------------------------------------------

class TestTaskRegistry:
    """任务注册表测试。"""

    def test_get_tasks_valid(self) -> None:
        """获取已注册任务。"""
        tasks = get_tasks(["cheetah_run", "cheetah_run_backwards"])
        assert len(tasks) == 2
        assert all(isinstance(t, TaskDef) for t in tasks)
        assert tasks[0].scenario == "HalfCheetah"
        assert tasks[1].agent_conf == "2x3"

    def test_get_tasks_invalid(self) -> None:
        """获取未注册任务抛出 KeyError。"""
        with pytest.raises(KeyError, match="未知任务"):
            get_tasks(["nonexistent_task"])

    def test_registry_not_empty(self) -> None:
        """注册表非空。"""
        assert len(TASK_REGISTRY) >= 2


# ---------------------------------------------------------------------------
# 3. 奖励函数测试
# ---------------------------------------------------------------------------

class TestRewardFunctions:
    """奖励函数测试。"""

    def test_cheetah_run_fast(self) -> None:
        """高速前进 + 正常姿态 → 接近 1.0。"""
        info = {
            "forward_velocity": 12.0,
            "torso_z": 0.7,
            "rooty": 0.0,
        }
        r = cheetah_run(info)
        assert 0.9 <= r <= 1.0

    def test_cheetah_run_zero_speed(self) -> None:
        """零速 → 接近 0。"""
        info = {
            "forward_velocity": 0.0,
            "torso_z": 0.7,
            "rooty": 0.0,
        }
        r = cheetah_run(info)
        assert r < 0.01

    def test_cheetah_run_flipped(self) -> None:
        """翻转（rooty=pi）→ 接近 0。"""
        info = {
            "forward_velocity": 12.0,
            "torso_z": 0.7,
            "rooty": np.pi,
        }
        r = cheetah_run(info)
        assert r < 0.01

    def test_cheetah_run_low_height(self) -> None:
        """低高度 → 奖励大幅衰减。"""
        info = {
            "forward_velocity": 12.0,
            "torso_z": 0.1,
            "rooty": 0.0,
        }
        r = cheetah_run(info)
        assert r < 0.1

    def test_cheetah_run_backwards_fast(self) -> None:
        """高速后退 → 接近 1.0。"""
        info = {
            "forward_velocity": -10.0,
            "torso_z": 0.7,
            "rooty": 0.0,
        }
        r = cheetah_run_backwards(info)
        assert r > 0.8

    def test_reward_range(self) -> None:
        """奖励始终在 [0, 1]。"""
        for fn in [cheetah_run, cheetah_run_backwards]:
            for vel in [-20, -5, 0, 5, 20]:
                for z in [0.0, 0.2, 0.5, 1.0]:
                    for rot in [0, np.pi / 2, np.pi]:
                        info = {
                            "forward_velocity": vel,
                            "torso_z": z,
                            "rooty": rot,
                        }
                        r = fn(info)
                        assert 0.0 <= r <= 1.0, (
                            f"{fn.__name__}({info}) = {r}"
                        )


# ---------------------------------------------------------------------------
# 4. 多任务环境测试（不含 one-hot）
# ---------------------------------------------------------------------------

class TestMultiTaskEnv:
    """多任务向量化环境测试（串行版本）。"""

    @pytest.fixture()
    def mt_env(self):
        """创建 2 任务 4 线程的多任务环境。"""
        from src.envs.mamujoco import MultiTaskVectorMAMuJoCoEnvDebug

        tasks = get_tasks(["cheetah_run", "cheetah_run_backwards"])
        env = MultiTaskVectorMAMuJoCoEnvDebug(
            n_envs=4, tasks=tasks, episode_limit=10, seed=42,
        )
        yield env
        env.close()

    def test_task_assignment(self, mt_env) -> None:
        """任务分配正确：前半 task 0，后半 task 1。"""
        assert list(mt_env.task_idxes) == [0, 0, 1, 1]

    def test_raw_obs_dims(self, mt_env) -> None:
        """观测维度为原始值（不含 one-hot）。"""
        for d in mt_env.obs_dims:
            assert d > 0

    def test_share_obs_shape(self, mt_env) -> None:
        """全局观测维度正确。"""
        _, share_obs = mt_env.reset()
        assert share_obs.shape == (4, mt_env.share_obs_dim)

    def test_step_reward_range(self, mt_env) -> None:
        """step 后奖励在 [0, 1]。"""
        mt_env.reset()
        actions = mt_env.sample_random_actions()
        _, _, rewards, _, _, _ = mt_env.step(actions)
        for r in rewards:
            assert 0.0 <= r <= 1.0

    def test_step_shapes(self, mt_env) -> None:
        """step 返回值形状正确。"""
        mt_env.reset()
        actions = mt_env.sample_random_actions()
        obs, share, rewards, dones, truncs, infos = mt_env.step(actions)
        assert len(obs) == 4
        assert share.shape[0] == 4
        assert rewards.shape == (4,)
        assert dones.shape == (4,)

    def test_n_envs_not_divisible(self) -> None:
        """n_envs 不能被 n_tasks 整除时抛出断言错误。"""
        from src.envs.mamujoco import MultiTaskVectorMAMuJoCoEnvDebug

        tasks = get_tasks(["cheetah_run", "cheetah_run_backwards"])
        with pytest.raises(AssertionError):
            MultiTaskVectorMAMuJoCoEnvDebug(
                n_envs=3, tasks=tasks, episode_limit=10,
            )


# ---------------------------------------------------------------------------
# 5. 任务嵌入 + 模型管线集成测试
# ---------------------------------------------------------------------------

class TestTaskEmbeddingIntegration:
    """验证任务嵌入在多任务模型管线中的传播。"""

    def test_encoder_with_task_emb(self) -> None:
        """Encoder 接受原始 obs + 任务嵌入。"""
        from src.models.encoder import MLPEncoder
        from src.models.task_embedding import TaskEmbeddingTable

        n_tasks = 2
        task_dim = 32
        raw_obs_dim = 12

        table = TaskEmbeddingTable(n_tasks=n_tasks, task_dim=task_dim)
        enc = MLPEncoder(
            obs_dim=raw_obs_dim,
            latent_dim=128,
            task_dim=task_dim,
            hidden_dims=[512, 512],
            simnorm_dim=8,
            device="cpu",
        )
        task_ids = torch.tensor([0, 0, 1, 1])
        task_emb = table(task_ids)

        x = torch.randn(4, raw_obs_dim)
        z = enc.encode(x, task_emb)
        assert z.shape == (4, 128)

    def test_full_pipeline_with_task_emb(self) -> None:
        """原始 obs + 任务嵌入 → encoder → dynamics → reward → actor → critic。"""
        from src.algorithms.actor import GaussianPolicy
        from src.algorithms.critic import WorldModelCritic
        from src.models.dynamics import DenseDynamics
        from src.models.encoder import MLPEncoder
        from src.models.reward import DenseReward
        from src.models.task_embedding import TaskEmbeddingTable

        n_tasks = 2
        task_dim = 32
        raw_obs_dim = 12
        latent_dim = 128
        act_dim = 3
        n_agents = 2
        batch = 16

        table = TaskEmbeddingTable(n_tasks=n_tasks, task_dim=task_dim)
        task_ids = torch.randint(0, n_tasks, (batch,))
        task_emb = table(task_ids)

        enc = MLPEncoder(
            obs_dim=raw_obs_dim, latent_dim=latent_dim,
            task_dim=task_dim,
            hidden_dims=[256, 256], simnorm_dim=8, device="cpu",
        )
        dyn = DenseDynamics(
            latent_dim=latent_dim, action_dim=act_dim,
            n_agents=n_agents, task_dim=task_dim,
            hidden_dims=[256, 256], simnorm_dim=8, device="cpu",
        )
        rew = DenseReward(
            latent_dim=latent_dim, action_dim=act_dim,
            n_agents=n_agents, task_dim=task_dim,
            num_bins=101, hidden_dims=[256, 256], device="cpu",
        )
        actor = GaussianPolicy(
            latent_dim=latent_dim, action_dim=act_dim,
            task_dim=task_dim,
            hidden_sizes=[128, 128], device="cpu",
        )
        critic = WorldModelCritic(
            joint_latent_dim=latent_dim * n_agents,
            joint_action_dim=act_dim * n_agents,
            task_dim=task_dim,
            num_bins=101, reward_min=-10, reward_max=10,
            hidden_sizes=[256, 256], device="cpu",
        )

        obs = torch.randn(batch, raw_obs_dim)
        z = enc.encode(obs, task_emb)
        assert z.shape == (batch, latent_dim)

        z_joint = z.unsqueeze(1).expand(-1, n_agents, -1)
        a = torch.randn(batch, n_agents, act_dim)
        z_next = dyn.predict(z_joint, a, task_emb)
        assert z_next.shape == (batch, n_agents, latent_dim)

        r_logits = rew.predict(z_joint, a, task_emb)
        assert r_logits.shape[0] == batch

        act, _ = actor(z, task_emb, stochastic=True)
        assert act.shape == (batch, act_dim)

        joint_z = z.unsqueeze(1).expand(
            -1, n_agents, -1,
        ).reshape(batch, -1)
        joint_a = a.reshape(batch, -1)
        q = critic.get_values(joint_z, joint_a, task_emb)
        assert q.shape == (batch, 1)
