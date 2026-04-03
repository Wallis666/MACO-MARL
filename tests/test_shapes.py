"""单元测试：验证各组件前向传播的张量形状。"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import pytest

BATCH = 16
N_AGENTS = 2
OBS_DIM = 8
ACT_DIM = 3
LATENT_DIM = 128
NUM_BINS = 101
TASK_DIM = 32
DEVICE = "cpu"


class TestSimNorm:
    """SimNorm 张量形状测试。"""

    def test_shape_preserved(self) -> None:
        from src.models.utils import SimNorm
        norm = SimNorm(dim=8)
        x = torch.randn(BATCH, LATENT_DIM)
        out = norm(x)
        assert out.shape == (BATCH, LATENT_DIM)

    def test_softmax_groups(self) -> None:
        from src.models.utils import SimNorm
        norm = SimNorm(dim=8)
        x = torch.randn(BATCH, 16)
        out = norm(x)
        reshaped = out.view(BATCH, 2, 8)
        sums = reshaped.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)


class TestTwoHotProcessor:
    """TwoHot 编解码器测试。"""

    def test_encode_shape(self) -> None:
        from src.models.utils import TwoHotProcessor
        proc = TwoHotProcessor(NUM_BINS, -10, 10, DEVICE)
        x = torch.randn(BATCH, 1)
        twohot = proc.scalar_to_twohot(x)
        assert twohot.shape == (BATCH, NUM_BINS)

    def test_decode_shape(self) -> None:
        from src.models.utils import TwoHotProcessor
        proc = TwoHotProcessor(NUM_BINS, -10, 10, DEVICE)
        logits = torch.randn(BATCH, NUM_BINS)
        scalar = proc.logits_to_scalar(logits)
        assert scalar.shape == (BATCH, 1)

    def test_roundtrip(self) -> None:
        from src.models.utils import TwoHotProcessor
        proc = TwoHotProcessor(NUM_BINS, -10, 10, DEVICE)
        x = torch.tensor([[0.5], [-0.3], [2.0]])
        twohot = proc.scalar_to_twohot(x)
        recovered = proc.logits_to_scalar(
            torch.log(twohot + 1e-8),
        )
        assert torch.allclose(
            TwoHotProcessor.symlog(x),
            TwoHotProcessor.symlog(recovered),
            atol=0.1,
        )

    def test_loss_shape(self) -> None:
        from src.models.utils import TwoHotProcessor
        proc = TwoHotProcessor(NUM_BINS, -10, 10, DEVICE)
        logits = torch.randn(BATCH, NUM_BINS)
        target = torch.randn(BATCH, 1)
        loss = proc.loss(logits, target)
        assert loss.shape == (BATCH, 1)


class TestRunningScale:
    """RunningScale 测试。"""

    def test_shape(self) -> None:
        from src.models.utils import RunningScale
        scale = RunningScale(tau=0.01, device=DEVICE)
        x = torch.randn(BATCH, 1)
        out = scale(x, update=True)
        assert out.shape == (BATCH, 1)

    def test_initial_value(self) -> None:
        from src.models.utils import RunningScale
        scale = RunningScale(tau=0.01, device=DEVICE)
        assert scale.value.item() == pytest.approx(1.0)


class TestTaskEmbedding:
    """TaskEmbeddingTable 测试。"""

    def test_shape(self) -> None:
        from src.models.task_embedding import TaskEmbeddingTable
        table = TaskEmbeddingTable(n_tasks=4, task_dim=TASK_DIM)
        task_ids = torch.tensor([0, 1, 2, 3])
        emb = table(task_ids)
        assert emb.shape == (4, TASK_DIM)

    def test_max_norm(self) -> None:
        from src.models.task_embedding import TaskEmbeddingTable
        table = TaskEmbeddingTable(n_tasks=4, task_dim=TASK_DIM)
        task_ids = torch.tensor([0, 1, 2, 3])
        emb = table(task_ids)
        norms = emb.norm(dim=-1)
        assert (norms <= 1.0 + 1e-5).all()

    def test_scalar_input(self) -> None:
        from src.models.task_embedding import TaskEmbeddingTable
        table = TaskEmbeddingTable(n_tasks=4, task_dim=TASK_DIM)
        emb = table(torch.tensor(2))
        assert emb.shape == (1, TASK_DIM)


class TestCatTaskEmb:
    """cat_task_emb 工具函数测试。"""

    def test_with_emb(self) -> None:
        from src.models.task_embedding import cat_task_emb
        x = torch.randn(BATCH, OBS_DIM)
        emb = torch.randn(BATCH, TASK_DIM)
        out = cat_task_emb(x, emb)
        assert out.shape == (BATCH, OBS_DIM + TASK_DIM)

    def test_without_emb(self) -> None:
        from src.models.task_embedding import cat_task_emb
        x = torch.randn(BATCH, OBS_DIM)
        out = cat_task_emb(x, None)
        assert out.shape == (BATCH, OBS_DIM)


class TestEncoder:
    """MLPEncoder 测试。"""

    def test_shape_no_task(self) -> None:
        from src.models.encoder import MLPEncoder
        enc = MLPEncoder(
            obs_dim=OBS_DIM,
            latent_dim=LATENT_DIM,
            hidden_dims=[256, 256],
            simnorm_dim=8,
            device=DEVICE,
        )
        obs = torch.randn(BATCH, OBS_DIM)
        z = enc.encode(obs)
        assert z.shape == (BATCH, LATENT_DIM)

    def test_shape_with_task(self) -> None:
        from src.models.encoder import MLPEncoder
        enc = MLPEncoder(
            obs_dim=OBS_DIM,
            latent_dim=LATENT_DIM,
            task_dim=TASK_DIM,
            hidden_dims=[256, 256],
            simnorm_dim=8,
            device=DEVICE,
        )
        obs = torch.randn(BATCH, OBS_DIM)
        task_emb = torch.randn(BATCH, TASK_DIM)
        z = enc.encode(obs, task_emb)
        assert z.shape == (BATCH, LATENT_DIM)


class TestDynamics:
    """DenseDynamics 测试。"""

    def test_shape_no_task(self) -> None:
        from src.models.dynamics import DenseDynamics
        dyn = DenseDynamics(
            latent_dim=LATENT_DIM,
            action_dim=ACT_DIM,
            n_agents=N_AGENTS,
            hidden_dims=[256, 256],
            device=DEVICE,
        )
        z = torch.randn(BATCH, N_AGENTS, LATENT_DIM)
        a = torch.randn(BATCH, N_AGENTS, ACT_DIM)
        z_next = dyn.predict(z, a)
        assert z_next.shape == (BATCH, N_AGENTS, LATENT_DIM)

    def test_shape_with_task(self) -> None:
        from src.models.dynamics import DenseDynamics
        dyn = DenseDynamics(
            latent_dim=LATENT_DIM,
            action_dim=ACT_DIM,
            n_agents=N_AGENTS,
            task_dim=TASK_DIM,
            hidden_dims=[256, 256],
            device=DEVICE,
        )
        z = torch.randn(BATCH, N_AGENTS, LATENT_DIM)
        a = torch.randn(BATCH, N_AGENTS, ACT_DIM)
        task_emb = torch.randn(BATCH, TASK_DIM)
        z_next = dyn.predict(z, a, task_emb)
        assert z_next.shape == (BATCH, N_AGENTS, LATENT_DIM)


class TestReward:
    """DenseReward 测试。"""

    def test_shape_no_task(self) -> None:
        from src.models.reward import DenseReward
        rew = DenseReward(
            latent_dim=LATENT_DIM,
            action_dim=ACT_DIM,
            n_agents=N_AGENTS,
            num_bins=NUM_BINS,
            hidden_dims=[256, 256],
            device=DEVICE,
        )
        z = torch.randn(BATCH, N_AGENTS, LATENT_DIM)
        a = torch.randn(BATCH, N_AGENTS, ACT_DIM)
        logits = rew.predict(z, a)
        assert logits.shape == (BATCH, NUM_BINS)

    def test_shape_with_task(self) -> None:
        from src.models.reward import DenseReward
        rew = DenseReward(
            latent_dim=LATENT_DIM,
            action_dim=ACT_DIM,
            n_agents=N_AGENTS,
            task_dim=TASK_DIM,
            num_bins=NUM_BINS,
            hidden_dims=[256, 256],
            device=DEVICE,
        )
        z = torch.randn(BATCH, N_AGENTS, LATENT_DIM)
        a = torch.randn(BATCH, N_AGENTS, ACT_DIM)
        task_emb = torch.randn(BATCH, TASK_DIM)
        logits = rew.predict(z, a, task_emb)
        assert logits.shape == (BATCH, NUM_BINS)


class TestActor:
    """GaussianPolicy / WorldModelActor 测试。"""

    def test_policy_shape_no_task(self) -> None:
        from src.algorithms.actor import GaussianPolicy
        policy = GaussianPolicy(
            latent_dim=LATENT_DIM,
            action_dim=ACT_DIM,
            hidden_sizes=[64, 64],
            device=DEVICE,
        )
        z = torch.randn(BATCH, LATENT_DIM)
        action, log_prob = policy(z, stochastic=True)
        assert action.shape == (BATCH, ACT_DIM)
        assert log_prob.shape == (BATCH, 1)
        assert (action >= -1).all() and (action <= 1).all()

    def test_policy_shape_with_task(self) -> None:
        from src.algorithms.actor import GaussianPolicy
        policy = GaussianPolicy(
            latent_dim=LATENT_DIM,
            action_dim=ACT_DIM,
            task_dim=TASK_DIM,
            hidden_sizes=[64, 64],
            device=DEVICE,
        )
        z = torch.randn(BATCH, LATENT_DIM)
        task_emb = torch.randn(BATCH, TASK_DIM)
        action, log_prob = policy(z, task_emb, stochastic=True)
        assert action.shape == (BATCH, ACT_DIM)
        assert log_prob.shape == (BATCH, 1)

    def test_deterministic(self) -> None:
        from src.algorithms.actor import GaussianPolicy
        policy = GaussianPolicy(
            latent_dim=LATENT_DIM,
            action_dim=ACT_DIM,
            device=DEVICE,
        )
        z = torch.randn(BATCH, LATENT_DIM)
        action, log_prob = policy(z, stochastic=False)
        assert action.shape == (BATCH, ACT_DIM)
        assert log_prob is None

    def test_actor_wrapper(self) -> None:
        from src.algorithms.actor import WorldModelActor
        actor = WorldModelActor(
            latent_dim=LATENT_DIM,
            action_dim=ACT_DIM,
            device=DEVICE,
        )
        z = torch.randn(BATCH, LATENT_DIM)
        a = actor.get_actions(z)
        assert a.shape == (BATCH, ACT_DIM)

    def test_actor_wrapper_with_task(self) -> None:
        from src.algorithms.actor import WorldModelActor
        actor = WorldModelActor(
            latent_dim=LATENT_DIM,
            action_dim=ACT_DIM,
            task_dim=TASK_DIM,
            device=DEVICE,
        )
        z = torch.randn(BATCH, LATENT_DIM)
        task_emb = torch.randn(BATCH, TASK_DIM)
        a = actor.get_actions(z, task_emb)
        assert a.shape == (BATCH, ACT_DIM)


class TestCritic:
    """DisRegQNet / WorldModelCritic 测试。"""

    def test_qnet_shape_no_task(self) -> None:
        from src.algorithms.critic import DisRegQNet
        joint_dim = N_AGENTS * LATENT_DIM + N_AGENTS * ACT_DIM
        qnet = DisRegQNet(
            input_dim=joint_dim,
            num_bins=NUM_BINS,
            hidden_sizes=[256, 256],
            device=DEVICE,
        )
        joint_z = torch.randn(BATCH, N_AGENTS * LATENT_DIM)
        joint_a = torch.randn(BATCH, N_AGENTS * ACT_DIM)
        logits = qnet(joint_z, joint_a)
        assert logits.shape == (BATCH, NUM_BINS)

    def test_qnet_shape_with_task(self) -> None:
        from src.algorithms.critic import DisRegQNet
        joint_dim = N_AGENTS * LATENT_DIM + N_AGENTS * ACT_DIM + TASK_DIM
        qnet = DisRegQNet(
            input_dim=joint_dim,
            num_bins=NUM_BINS,
            hidden_sizes=[256, 256],
            device=DEVICE,
        )
        joint_z = torch.randn(BATCH, N_AGENTS * LATENT_DIM)
        joint_a = torch.randn(BATCH, N_AGENTS * ACT_DIM)
        task_emb = torch.randn(BATCH, TASK_DIM)
        logits = qnet(joint_z, joint_a, task_emb)
        assert logits.shape == (BATCH, NUM_BINS)

    def test_critic_values(self) -> None:
        from src.algorithms.critic import WorldModelCritic
        critic = WorldModelCritic(
            joint_latent_dim=N_AGENTS * LATENT_DIM,
            joint_action_dim=N_AGENTS * ACT_DIM,
            num_bins=NUM_BINS,
            device=DEVICE,
        )
        joint_z = torch.randn(BATCH, N_AGENTS * LATENT_DIM)
        joint_a = torch.randn(BATCH, N_AGENTS * ACT_DIM)

        q_min = critic.get_values(joint_z, joint_a, mode="min")
        assert q_min.shape == (BATCH, 1)

        q_mean = critic.get_values(joint_z, joint_a, mode="mean")
        assert q_mean.shape == (BATCH, 1)

        q_target = critic.get_target_values(joint_z, joint_a)
        assert q_target.shape == (BATCH, 1)

    def test_critic_values_with_task(self) -> None:
        from src.algorithms.critic import WorldModelCritic
        critic = WorldModelCritic(
            joint_latent_dim=N_AGENTS * LATENT_DIM,
            joint_action_dim=N_AGENTS * ACT_DIM,
            task_dim=TASK_DIM,
            num_bins=NUM_BINS,
            device=DEVICE,
        )
        joint_z = torch.randn(BATCH, N_AGENTS * LATENT_DIM)
        joint_a = torch.randn(BATCH, N_AGENTS * ACT_DIM)
        task_emb = torch.randn(BATCH, TASK_DIM)

        q_min = critic.get_values(joint_z, joint_a, task_emb, mode="min")
        assert q_min.shape == (BATCH, 1)

        q_target = critic.get_target_values(joint_z, joint_a, task_emb)
        assert q_target.shape == (BATCH, 1)


class TestBuffer:
    """ReplayBuffer 测试。"""

    def test_insert_and_sample(self) -> None:
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
                share_obs=np.random.randn(4, OBS_DIM * N_AGENTS).astype(np.float32),
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
                next_share_obs=np.random.randn(4, OBS_DIM * N_AGENTS).astype(np.float32),
                next_obs=[
                    np.random.randn(4, OBS_DIM).astype(np.float32)
                    for _ in range(N_AGENTS)
                ],
                task_idx=task_idx,
            )

        assert buf.can_sample()
        batch = buf.sample()
        assert batch["share_obs"].shape == (32, OBS_DIM * N_AGENTS)
        assert len(batch["obs"]) == N_AGENTS
        assert batch["obs"][0].shape == (32, OBS_DIM)
        assert batch["nstep_reward"].shape == (32, 1)
        assert batch["task_idx"].shape == (32,)
        assert set(batch["task_idx"]).issubset({0, 1})

    def test_sample_horizon(self) -> None:
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
        for _ in range(100):
            buf.insert(
                share_obs=np.random.randn(4, OBS_DIM * N_AGENTS).astype(np.float32),
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
                next_share_obs=np.random.randn(4, OBS_DIM * N_AGENTS).astype(np.float32),
                next_obs=[
                    np.random.randn(4, OBS_DIM).astype(np.float32)
                    for _ in range(N_AGENTS)
                ],
            )

        horizon = 5
        batch = buf.sample_horizon(horizon)
        assert batch["obs"][0].shape == (horizon, 32, OBS_DIM)
        assert batch["actions"][0].shape == (horizon, 32, ACT_DIM)
        assert batch["rewards"].shape == (horizon, 32, 1)
        assert batch["task_idx"].shape == (32,)


class TestFullPipeline:
    """完整管线形状传播测试（含任务嵌入）。"""

    def test_pipeline_with_task_emb(self) -> None:
        """obs → encoder(task_emb) → dynamics(task_emb) → reward(task_emb)
        → actor(task_emb) → critic(task_emb)。"""
        from src.algorithms.actor import GaussianPolicy
        from src.algorithms.critic import WorldModelCritic
        from src.models.dynamics import DenseDynamics
        from src.models.encoder import MLPEncoder
        from src.models.reward import DenseReward
        from src.models.task_embedding import TaskEmbeddingTable

        n_tasks = 2
        latent_dim = LATENT_DIM
        act_dim = ACT_DIM
        n_agents = N_AGENTS
        batch = BATCH

        task_table = TaskEmbeddingTable(n_tasks=n_tasks, task_dim=TASK_DIM)
        task_ids = torch.randint(0, n_tasks, (batch,))
        task_emb = task_table(task_ids)
        assert task_emb.shape == (batch, TASK_DIM)

        enc = MLPEncoder(
            obs_dim=OBS_DIM, latent_dim=latent_dim,
            task_dim=TASK_DIM,
            hidden_dims=[256, 256], simnorm_dim=8, device="cpu",
        )
        dyn = DenseDynamics(
            latent_dim=latent_dim, action_dim=act_dim,
            n_agents=n_agents, task_dim=TASK_DIM,
            hidden_dims=[256, 256], simnorm_dim=8, device="cpu",
        )
        rew = DenseReward(
            latent_dim=latent_dim, action_dim=act_dim,
            n_agents=n_agents, task_dim=TASK_DIM,
            num_bins=NUM_BINS, hidden_dims=[256, 256], device="cpu",
        )
        actor = GaussianPolicy(
            latent_dim=latent_dim, action_dim=act_dim,
            task_dim=TASK_DIM,
            hidden_sizes=[128, 128], device="cpu",
        )
        critic = WorldModelCritic(
            joint_latent_dim=latent_dim * n_agents,
            joint_action_dim=act_dim * n_agents,
            task_dim=TASK_DIM,
            num_bins=NUM_BINS, reward_min=-10, reward_max=10,
            hidden_sizes=[256, 256], device="cpu",
        )

        obs = torch.randn(batch, OBS_DIM)
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

    def test_pipeline_no_task_emb(self) -> None:
        """单任务模式（task_dim=0）下管线正常工作。"""
        from src.algorithms.actor import GaussianPolicy
        from src.algorithms.critic import WorldModelCritic
        from src.models.dynamics import DenseDynamics
        from src.models.encoder import MLPEncoder
        from src.models.reward import DenseReward

        latent_dim = LATENT_DIM
        act_dim = ACT_DIM
        n_agents = N_AGENTS
        batch = BATCH

        enc = MLPEncoder(
            obs_dim=OBS_DIM, latent_dim=latent_dim,
            hidden_dims=[256, 256], simnorm_dim=8, device="cpu",
        )
        dyn = DenseDynamics(
            latent_dim=latent_dim, action_dim=act_dim,
            n_agents=n_agents, hidden_dims=[256, 256], device="cpu",
        )
        rew = DenseReward(
            latent_dim=latent_dim, action_dim=act_dim,
            n_agents=n_agents, num_bins=NUM_BINS,
            hidden_dims=[256, 256], device="cpu",
        )
        actor = GaussianPolicy(
            latent_dim=latent_dim, action_dim=act_dim,
            hidden_sizes=[128, 128], device="cpu",
        )
        critic = WorldModelCritic(
            joint_latent_dim=latent_dim * n_agents,
            joint_action_dim=act_dim * n_agents,
            num_bins=NUM_BINS, reward_min=-10, reward_max=10,
            hidden_sizes=[256, 256], device="cpu",
        )

        obs = torch.randn(batch, OBS_DIM)
        z = enc.encode(obs)
        assert z.shape == (batch, latent_dim)

        z_joint = z.unsqueeze(1).expand(-1, n_agents, -1)
        a = torch.randn(batch, n_agents, act_dim)
        z_next = dyn.predict(z_joint, a)
        assert z_next.shape == (batch, n_agents, latent_dim)

        r_logits = rew.predict(z_joint, a)
        assert r_logits.shape[0] == batch

        act, _ = actor(z, stochastic=True)
        assert act.shape == (batch, act_dim)

        joint_z = z.unsqueeze(1).expand(
            -1, n_agents, -1,
        ).reshape(batch, -1)
        joint_a = a.reshape(batch, -1)
        q = critic.get_values(joint_z, joint_a)
        assert q.shape == (batch, 1)


class TestPlanner:
    """MPPI 规划器形状测试。"""

    def test_plan_output_shape(self) -> None:
        from src.models.dynamics import DenseDynamics
        from src.models.reward import DenseReward
        from src.models.utils import TwoHotProcessor
        from src.algorithms.actor import WorldModelActor
        from src.algorithms.critic import WorldModelCritic
        from src.algorithms.planner import MPPIPlanner

        n_threads = 2

        dynamics = DenseDynamics(
            LATENT_DIM, ACT_DIM, N_AGENTS,
            hidden_dims=[64, 64], device=DEVICE,
        )
        reward_model = DenseReward(
            LATENT_DIM, ACT_DIM, N_AGENTS,
            num_bins=NUM_BINS, hidden_dims=[64, 64], device=DEVICE,
        )
        reward_proc = TwoHotProcessor(NUM_BINS, -10, 10, DEVICE)
        actors = [
            WorldModelActor(LATENT_DIM, ACT_DIM, device=DEVICE)
            for _ in range(N_AGENTS)
        ]
        critic = WorldModelCritic(
            N_AGENTS * LATENT_DIM, N_AGENTS * ACT_DIM,
            num_bins=NUM_BINS, device=DEVICE,
        )

        planner = MPPIPlanner(
            n_agents=N_AGENTS,
            act_dims=[ACT_DIM] * N_AGENTS,
            latent_dim=LATENT_DIM,
            horizon=3,
            iterations=2,
            num_samples=16,
            num_pi_trajs=4,
            num_elites=4,
            device=DEVICE,
        )

        zs = [torch.randn(n_threads, LATENT_DIM) for _ in range(N_AGENTS)]
        t0 = [True] * n_threads

        out = planner.plan(
            zs, t0, dynamics, reward_model, reward_proc, actors, critic,
        )

        assert len(out) == N_AGENTS
        for i in range(N_AGENTS):
            assert out[i].shape == (n_threads, ACT_DIM)

    def test_plan_with_task_emb(self) -> None:
        from src.models.dynamics import DenseDynamics
        from src.models.reward import DenseReward
        from src.models.utils import TwoHotProcessor
        from src.algorithms.actor import WorldModelActor
        from src.algorithms.critic import WorldModelCritic
        from src.algorithms.planner import MPPIPlanner

        n_threads = 2

        dynamics = DenseDynamics(
            LATENT_DIM, ACT_DIM, N_AGENTS,
            task_dim=TASK_DIM,
            hidden_dims=[64, 64], device=DEVICE,
        )
        reward_model = DenseReward(
            LATENT_DIM, ACT_DIM, N_AGENTS,
            task_dim=TASK_DIM,
            num_bins=NUM_BINS, hidden_dims=[64, 64], device=DEVICE,
        )
        reward_proc = TwoHotProcessor(NUM_BINS, -10, 10, DEVICE)
        actors = [
            WorldModelActor(
                LATENT_DIM, ACT_DIM, task_dim=TASK_DIM, device=DEVICE,
            )
            for _ in range(N_AGENTS)
        ]
        critic = WorldModelCritic(
            joint_latent_dim=N_AGENTS * LATENT_DIM,
            joint_action_dim=N_AGENTS * ACT_DIM,
            task_dim=TASK_DIM,
            num_bins=NUM_BINS, device=DEVICE,
        )

        planner = MPPIPlanner(
            n_agents=N_AGENTS,
            act_dims=[ACT_DIM] * N_AGENTS,
            latent_dim=LATENT_DIM,
            horizon=3,
            iterations=2,
            num_samples=16,
            num_pi_trajs=4,
            num_elites=4,
            device=DEVICE,
        )

        zs = [torch.randn(n_threads, LATENT_DIM) for _ in range(N_AGENTS)]
        t0 = [True] * n_threads
        task_emb = torch.randn(n_threads, TASK_DIM)

        out = planner.plan(
            zs, t0, dynamics, reward_model, reward_proc, actors, critic,
            task_emb=task_emb,
        )

        assert len(out) == N_AGENTS
        for i in range(N_AGENTS):
            assert out[i].shape == (n_threads, ACT_DIM)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
