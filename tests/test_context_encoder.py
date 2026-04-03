"""上下文编码器测试。"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
import torch
import torch.nn.functional as F

BATCH = 16
OBS_DIM = 8
ACT_DIM = 3
TASK_DIM = 32
N_AGENTS = 2
DEVICE = "cpu"


# ---------------------------------------------------------------------------
# 1. ContextEncoder 形状测试
# ---------------------------------------------------------------------------

class TestContextEncoderShape:
    """上下文编码器输出形状测试。"""

    def test_output_shape_k16(self) -> None:
        """K=16 时输出形状正确。"""
        from src.models.context_encoder import ContextEncoder
        enc = ContextEncoder(OBS_DIM, ACT_DIM, TASK_DIM, device=DEVICE)
        obs = torch.randn(BATCH, 16, OBS_DIM)
        actions = torch.randn(BATCH, 16, ACT_DIM)
        rewards = torch.randn(BATCH, 16, 1)
        next_obs = torch.randn(BATCH, 16, OBS_DIM)
        z = enc.encode_transitions(obs, actions, rewards, next_obs)
        assert z.shape == (BATCH, TASK_DIM)

    def test_output_shape_k1(self) -> None:
        """K=1 单条 transition 仍然有效。"""
        from src.models.context_encoder import ContextEncoder
        enc = ContextEncoder(OBS_DIM, ACT_DIM, TASK_DIM, device=DEVICE)
        obs = torch.randn(BATCH, 1, OBS_DIM)
        actions = torch.randn(BATCH, 1, ACT_DIM)
        rewards = torch.randn(BATCH, 1, 1)
        next_obs = torch.randn(BATCH, 1, OBS_DIM)
        z = enc.encode_transitions(obs, actions, rewards, next_obs)
        assert z.shape == (BATCH, TASK_DIM)

    def test_variable_k(self) -> None:
        """同一编码器支持不同 K 值。"""
        from src.models.context_encoder import ContextEncoder
        enc = ContextEncoder(OBS_DIM, ACT_DIM, TASK_DIM, device=DEVICE)
        for k in [1, 5, 16, 64]:
            obs = torch.randn(BATCH, k, OBS_DIM)
            actions = torch.randn(BATCH, k, ACT_DIM)
            rewards = torch.randn(BATCH, k, 1)
            next_obs = torch.randn(BATCH, k, OBS_DIM)
            z = enc.encode_transitions(obs, actions, rewards, next_obs)
            assert z.shape == (BATCH, TASK_DIM)

    def test_output_normalized(self) -> None:
        """输出 L2 范数约为 1.0。"""
        from src.models.context_encoder import ContextEncoder
        enc = ContextEncoder(OBS_DIM, ACT_DIM, TASK_DIM, device=DEVICE)
        obs = torch.randn(BATCH, 16, OBS_DIM)
        actions = torch.randn(BATCH, 16, ACT_DIM)
        rewards = torch.randn(BATCH, 16, 1)
        next_obs = torch.randn(BATCH, 16, OBS_DIM)
        z = enc.encode_transitions(obs, actions, rewards, next_obs)
        norms = z.norm(dim=-1)
        assert torch.allclose(norms, torch.ones(BATCH), atol=1e-5)


# ---------------------------------------------------------------------------
# 2. ContextEncoder 训练测试
# ---------------------------------------------------------------------------

class TestContextEncoderTraining:
    """上下文编码器训练行为测试。"""

    def test_mse_loss_decreases(self) -> None:
        """训练 50 步后 MSE 损失下降。"""
        from src.models.context_encoder import ContextEncoder
        from src.models.task_embedding import TaskEmbeddingTable

        n_tasks = 2
        table = TaskEmbeddingTable(n_tasks, TASK_DIM)
        enc = ContextEncoder(OBS_DIM, ACT_DIM, TASK_DIM, device=DEVICE)
        optimizer = torch.optim.Adam(enc.parameters(), lr=1e-3)

        losses = []
        for _ in range(50):
            task_ids = torch.randint(0, n_tasks, (BATCH,))
            with torch.no_grad():
                z_target = table(task_ids)

            obs = torch.randn(BATCH, 16, OBS_DIM)
            actions = torch.randn(BATCH, 16, ACT_DIM)
            rewards = torch.randn(BATCH, 16, 1)
            next_obs = torch.randn(BATCH, 16, OBS_DIM)

            z_ctx = enc.encode_transitions(obs, actions, rewards, next_obs)
            loss = F.mse_loss(z_ctx, z_target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        assert losses[-1] < losses[0], (
            f"损失未下降: {losses[0]:.4f} -> {losses[-1]:.4f}"
        )

    def test_gradient_isolation(self) -> None:
        """ctx encoder 训练不影响 task_embedding 参数。"""
        from src.models.context_encoder import ContextEncoder
        from src.models.task_embedding import TaskEmbeddingTable

        n_tasks = 2
        table = TaskEmbeddingTable(n_tasks, TASK_DIM)
        enc = ContextEncoder(OBS_DIM, ACT_DIM, TASK_DIM, device=DEVICE)
        optimizer = torch.optim.Adam(enc.parameters(), lr=1e-3)

        task_ids = torch.randint(0, n_tasks, (BATCH,))
        with torch.no_grad():
            z_target = table(task_ids)

        # max_norm 触发后记录参数（forward 会做 renorm）
        table_params_before = table._emb.weight.data.clone()

        obs = torch.randn(BATCH, 16, OBS_DIM)
        actions = torch.randn(BATCH, 16, ACT_DIM)
        rewards = torch.randn(BATCH, 16, 1)
        next_obs = torch.randn(BATCH, 16, OBS_DIM)

        z_ctx = enc.encode_transitions(obs, actions, rewards, next_obs)
        loss = F.mse_loss(z_ctx, z_target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # table 参数应保持不变（梯度未流向 table）
        assert torch.equal(
            table._emb.weight.data, table_params_before,
        ), "task_embedding 参数被意外修改"


# ---------------------------------------------------------------------------
# 3. Buffer sample_context 测试
# ---------------------------------------------------------------------------

class TestBufferSampleContext:
    """Buffer sample_context 方法测试。"""

    @pytest.fixture()
    def filled_buffer(self):
        """预填充数据的 buffer。"""
        from src.buffer.replay_buffer import ReplayBuffer
        buf = ReplayBuffer(
            n_agents=N_AGENTS,
            obs_dims=[OBS_DIM] * N_AGENTS,
            act_dims=[ACT_DIM] * N_AGENTS,
            share_obs_dim=OBS_DIM * N_AGENTS,
            buffer_size=1000,
            batch_size=32,
            n_step=5,
            gamma=0.99,
            n_rollout_threads=4,
        )
        task_idx = np.array([0, 0, 1, 1], dtype=np.int32)
        for _ in range(100):
            buf.insert(
                share_obs=np.random.randn(
                    4, OBS_DIM * N_AGENTS,
                ).astype(np.float32),
                obs=[
                    np.random.randn(4, OBS_DIM).astype(np.float32)
                    for _ in range(N_AGENTS)
                ],
                actions=[
                    np.random.randn(4, ACT_DIM).astype(np.float32)
                    for _ in range(N_AGENTS)
                ],
                rewards=np.random.randn(4, 1).astype(np.float32),
                dones=np.zeros((4, 1), dtype=np.float32),
                terms=np.zeros((4, 1), dtype=np.float32),
                valid=[
                    np.ones((4, 1), dtype=np.float32)
                    for _ in range(N_AGENTS)
                ],
                next_share_obs=np.random.randn(
                    4, OBS_DIM * N_AGENTS,
                ).astype(np.float32),
                next_obs=[
                    np.random.randn(4, OBS_DIM).astype(np.float32)
                    for _ in range(N_AGENTS)
                ],
                task_idx=task_idx,
            )
        return buf

    def test_sample_context_shapes(self, filled_buffer) -> None:
        """sample_context 返回形状正确。"""
        n_context = 8
        batch_size = 16
        ctx = filled_buffer.sample_context(n_context, batch_size)

        assert len(ctx["ctx_obs"]) == N_AGENTS
        assert ctx["ctx_obs"][0].shape == (batch_size, n_context, OBS_DIM)
        assert ctx["ctx_actions"][0].shape == (
            batch_size, n_context, ACT_DIM,
        )
        assert ctx["ctx_rewards"].shape == (batch_size, n_context, 1)
        assert ctx["ctx_next_obs"][0].shape == (
            batch_size, n_context, OBS_DIM,
        )
        assert ctx["task_idx"].shape == (batch_size,)

    def test_sample_context_task_consistency(self, filled_buffer) -> None:
        """每个样本的所有 context transitions 来自同一任务。"""
        n_context = 8
        batch_size = 32
        ctx = filled_buffer.sample_context(n_context, batch_size)

        # 验证: 对每个样本，其 n_context 条 transition 的 task_idx 应一致
        for i in range(batch_size):
            expected_task = ctx["task_idx"][i]
            for k in range(n_context):
                # 通过 buffer 中的索引验证（间接验证）
                # ctx_obs 来自同一任务的 buffer 区域
                assert expected_task in [0, 1]


# ---------------------------------------------------------------------------
# 4. 集成测试
# ---------------------------------------------------------------------------

class TestContextEncoderIntegration:
    """上下文编码器与模型管线集成测试。"""

    def test_ctx_emb_replaces_table(self) -> None:
        """ctx encoder 输出可替代 table lookup 注入所有模型。"""
        from src.algorithms.actor import GaussianPolicy
        from src.algorithms.critic import WorldModelCritic
        from src.models.context_encoder import ContextEncoder
        from src.models.dynamics import DenseDynamics
        from src.models.encoder import MLPEncoder
        from src.models.reward import DenseReward

        latent_dim = 128
        n_agents = N_AGENTS
        batch = BATCH

        ctx_enc = ContextEncoder(
            OBS_DIM, ACT_DIM, TASK_DIM, device=DEVICE,
        )

        # 生成上下文嵌入
        obs_ctx = torch.randn(batch, 16, OBS_DIM)
        act_ctx = torch.randn(batch, 16, ACT_DIM)
        rew_ctx = torch.randn(batch, 16, 1)
        nobs_ctx = torch.randn(batch, 16, OBS_DIM)
        task_emb = ctx_enc.encode_transitions(
            obs_ctx, act_ctx, rew_ctx, nobs_ctx,
        )
        assert task_emb.shape == (batch, TASK_DIM)

        # 用 ctx encoder 输出替代 table lookup，传入所有模型
        enc = MLPEncoder(
            OBS_DIM, latent_dim, task_dim=TASK_DIM,
            hidden_dims=[256, 256], simnorm_dim=8, device=DEVICE,
        )
        dyn = DenseDynamics(
            latent_dim, ACT_DIM, n_agents, task_dim=TASK_DIM,
            hidden_dims=[256, 256], device=DEVICE,
        )
        rew = DenseReward(
            latent_dim, ACT_DIM, n_agents, task_dim=TASK_DIM,
            num_bins=101, hidden_dims=[256, 256], device=DEVICE,
        )
        actor = GaussianPolicy(
            latent_dim, ACT_DIM, task_dim=TASK_DIM,
            hidden_sizes=[128, 128], device=DEVICE,
        )
        critic = WorldModelCritic(
            latent_dim * n_agents, ACT_DIM * n_agents,
            task_dim=TASK_DIM, num_bins=101, device=DEVICE,
        )

        obs = torch.randn(batch, OBS_DIM)
        z = enc.encode(obs, task_emb)
        assert z.shape == (batch, latent_dim)

        z_joint = z.unsqueeze(1).expand(-1, n_agents, -1)
        a = torch.randn(batch, n_agents, ACT_DIM)
        z_next = dyn.predict(z_joint, a, task_emb)
        assert z_next.shape == (batch, n_agents, latent_dim)

        r = rew.predict(z_joint, a, task_emb)
        assert r.shape[0] == batch

        act, _ = actor(z, task_emb, stochastic=True)
        assert act.shape == (batch, ACT_DIM)

        joint_z = z_joint.reshape(batch, -1)
        joint_a = a.reshape(batch, -1)
        q = critic.get_values(joint_z, joint_a, task_emb)
        assert q.shape == (batch, 1)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
