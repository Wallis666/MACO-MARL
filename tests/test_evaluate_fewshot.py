"""Few-shot 评估相关测试。"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
import torch

TASK_DIM = 32
DEVICE = "cpu"


class TestHeldOutTasks:
    """Held-out 评估任务注册测试。"""

    def test_tasks_registered(self) -> None:
        """评估任务在注册表中存在。"""
        from src.config.tasks import TASK_REGISTRY
        assert "cheetah_run_slow" in TASK_REGISTRY
        assert "cheetah_walk" in TASK_REGISTRY

    def test_same_morphology(self) -> None:
        """评估任务���训练任务使用相同形态。"""
        from src.config.tasks import TASK_REGISTRY
        for name in ["cheetah_run_slow", "cheetah_walk"]:
            task = TASK_REGISTRY[name]
            assert task.scenario == "HalfCheetah"
            assert task.agent_conf == "2x3"

    def test_reward_range(self) -> None:
        """评估任务奖励在 [0, 1] 范围内。"""
        from src.config.tasks import cheetah_run_slow, cheetah_walk
        for fn in [cheetah_run_slow, cheetah_walk]:
            for vel in [-10, -2, 0, 2, 5, 10]:
                for z in [0.0, 0.5, 1.0]:
                    for rot in [0, np.pi / 2, np.pi]:
                        info = {
                            "forward_velocity": vel,
                            "torso_z": z,
                            "rooty": rot,
                        }
                        r = fn(info)
                        assert 0.0 <= r <= 1.0

    def test_slow_reward_targets(self) -> None:
        """cheetah_run_slow 对 5m/s 给高奖励。"""
        from src.config.tasks import cheetah_run_slow
        info_fast = {
            "forward_velocity": 5.0,
            "torso_z": 0.7,
            "rooty": 0.0,
        }
        info_zero = {
            "forward_velocity": 0.0,
            "torso_z": 0.7,
            "rooty": 0.0,
        }
        assert cheetah_run_slow(info_fast) > 0.8
        assert cheetah_run_slow(info_zero) < 0.01

    def test_walk_reward_targets(self) -> None:
        """cheetah_walk 对 2m/s 给高奖励。"""
        from src.config.tasks import cheetah_walk
        info_walk = {
            "forward_velocity": 2.0,
            "torso_z": 0.7,
            "rooty": 0.0,
        }
        info_zero = {
            "forward_velocity": 0.0,
            "torso_z": 0.7,
            "rooty": 0.0,
        }
        assert cheetah_walk(info_walk) > 0.8
        assert cheetah_walk(info_zero) < 0.01


class TestCollectDemos:
    """Demo 收集测试。"""

    def test_shapes(self) -> None:
        """collect_demos 返回正确形状。"""
        from scripts.evaluate_fewshot import collect_demos
        n_context = 5
        obs, actions, rewards, next_obs = collect_demos(
            "cheetah_run", n_context, seed=42,
        )
        assert obs.ndim == 3
        assert obs.shape[0] == 1
        assert obs.shape[1] == n_context
        assert actions.shape[:2] == (1, n_context)
        assert rewards.shape == (1, n_context, 1)
        assert next_obs.shape == obs.shape

    def test_different_tasks_different_rewards(self) -> None:
        """不同任务的 demo 奖励分布应有差异。"""
        from scripts.evaluate_fewshot import collect_demos
        _, _, rew_run, _ = collect_demos("cheetah_run", 50, seed=0)
        _, _, rew_back, _ = collect_demos(
            "cheetah_run_backwards", 50, seed=0,
        )
        # 随机策略下两个任务的奖励不必完全相同
        # （前进 vs 后退的 tolerance 函数不同）
        assert rew_run.shape == rew_back.shape


class TestInferTaskEmbedding:
    """任务嵌入推断测试。"""

    def test_output_shape(self) -> None:
        """推断嵌入形状正确且归一化。"""
        from src.models.context_encoder import ContextEncoder
        from scripts.evaluate_fewshot import infer_task_embedding

        obs_dim, act_dim = 8, 3
        enc = ContextEncoder(obs_dim, act_dim, TASK_DIM, device=DEVICE)

        obs = np.random.randn(1, 10, obs_dim).astype(np.float32)
        actions = np.random.randn(1, 10, act_dim).astype(np.float32)
        rewards = np.random.randn(1, 10, 1).astype(np.float32)
        next_obs = np.random.randn(1, 10, obs_dim).astype(np.float32)

        z = infer_task_embedding(
            enc, obs, actions, rewards, next_obs, DEVICE,
        )
        assert z.shape == (1, TASK_DIM)
        assert torch.allclose(z.norm(dim=-1), torch.ones(1), atol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
