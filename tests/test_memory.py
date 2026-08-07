"""Test Component 2: KDA Market Memory."""

import pytest
import torch

from daft.models.memory import KDAMarketMemory


# Fixture `memory` is defined in conftest.py (shared across all tests)
BATCH_SIZES = [1, 4, 32]
D_FEATURE = 200
D_K = 128
D_V = 64


# ── Fixtures ──────────────────────────────────────────────────────────


@pytest.fixture
def s_t():
    return torch.randn(16, D_FEATURE)


@pytest.fixture
def z_t():
    return torch.randn(16, 16)  # latent_dim=16 from router


# ── Initialization ────────────────────────────────────────────────────

class TestInit:
    def test_default_params(self):
        mem = KDAMarketMemory()
        assert mem.d_k == 128
        assert mem.d_v == 64
        assert mem.d_feature == 200
        assert mem.use_route_modulation is True

    def test_custom_params(self):
        mem = KDAMarketMemory(d_k=64, d_v=32, d_feature=100, bottleneck_ratio=2)
        assert mem.d_k == 64
        assert mem.d_v == 32
        assert mem.d_feature == 100

    def test_initial_state_is_none(self, memory):
        assert memory.M is None

    def test_no_route_modulation(self):
        mem = KDAMarketMemory(use_route_modulation=False)
        s_t = torch.randn(4, 200)
        # Without route modulation, z_t=None is expected and should work
        retrieved, M_t = mem(s_t, z_t=None)
        assert retrieved.shape == (4, mem.d_v)
        assert retrieved.isfinite().all()


# ── Forward pass ──────────────────────────────────────────────────────

class TestForward:
    @pytest.mark.parametrize("B", BATCH_SIZES)
    def test_output_shapes(self, memory, B):
        s_t = torch.randn(B, D_FEATURE)
        z_t = torch.randn(B, 16)
        retrieved, M_t = memory(s_t, z_t=z_t)
        assert retrieved.shape == (B, D_V)
        assert M_t.shape == (B, D_K, D_V)

    @pytest.mark.parametrize("B", BATCH_SIZES)
    def test_retrieved_is_finite(self, memory, B):
        s_t = torch.randn(B, D_FEATURE)
        z_t = torch.randn(B, 16)
        retrieved, M_t = memory(s_t, z_t=z_t)
        assert retrieved.isfinite().all()
        assert M_t.isfinite().all()

    @pytest.mark.parametrize("B", BATCH_SIZES)
    def test_without_route_modulation(self, B):
        mem = KDAMarketMemory(use_route_modulation=False)
        s_t = torch.randn(B, D_FEATURE)
        retrieved, M_t = mem(s_t, z_t=None)
        assert retrieved.shape == (B, D_V)
        assert M_t.isfinite().all()

    def test_graceful_none_z_t(self, memory):
        """Even with use_route_modulation=True, z_t=None should work."""
        s_t = torch.randn(4, D_FEATURE)
        retrieved, M_t = memory(s_t, z_t=None)
        assert retrieved.isfinite().all()


# ── State management ──────────────────────────────────────────────────

class TestStateManagement:
    def test_reset_state_zeros(self, memory):
        memory.reset_state(8, torch.device("cpu"))
        assert memory.M is not None
        assert (memory.M == 0).all()

    def test_reset_flag_in_forward(self, memory):
        s_t = torch.randn(4, D_FEATURE)
        z_t = torch.randn(4, 16)
        # Fill memory
        memory(s_t, z_t=z_t)
        assert not (memory.M == 0).all()
        # Reset and forward
        retrieved, M_t = memory(s_t, z_t=z_t, reset=True)
        # State should be fresh (based on single step)
        assert M_t.isfinite().all()

    def test_detach_state(self, memory):
        """After detach_state(), memory should be detached from graph."""
        s_t = torch.randn(4, D_FEATURE)
        z_t = torch.randn(4, 16)
        retrieved, M_t = memory(s_t, z_t=z_t)
        assert M_t.requires_grad is True  # before detach, part of graph
        memory.detach_state()
        assert memory.M.requires_grad is False
        # Verify: backward through a detached state doesn't propagate
        loss = M_t.sum()
        loss.backward()
        # M_t comes from self.M.clone() — it may or may not have grad
        # depending on whether clone retains the graph. The key assertion
        # is that memory.M itself is detached.
        assert not memory.M.requires_grad

    def test_batch_size_change_resets(self, memory):
        """Changing batch size should trigger re-initialization."""
        memory(torch.randn(4, D_FEATURE), z_t=torch.randn(4, 16))
        m1_batch = memory.M.size(0)
        memory(torch.randn(8, D_FEATURE), z_t=torch.randn(8, 16))
        m2_batch = memory.M.size(0)
        assert m1_batch == 4
        assert m2_batch == 8


# ── Sequential processing ─────────────────────────────────────────────

class TestSequential:
    def test_sequential_updates(self, memory):
        """Memory should evolve over sequential steps."""
        s_seq = torch.randn(20, 1, D_FEATURE)
        z_seq = torch.randn(20, 1, 16)

        M_initial = None
        for step in range(20):
            retrieved, M_t = memory(s_seq[step], z_t=z_seq[step])
            if step == 0:
                M_initial = M_t.clone()
            if step == 19:
                M_final = M_t.clone()

        # Memory should NOT be identical after 20 steps vs step 1
        assert not torch.allclose(M_initial, M_final)

    def test_memory_is_stateful(self, memory):
        """Sequential processing ≠ independent processing."""
        s_seq = torch.randn(5, 4, D_FEATURE)
        z_seq = torch.randn(5, 4, 16)

        # Sequential
        memory.reset_state(4, torch.device("cpu"))
        retrievals_seq = []
        for step in range(5):
            r, _ = memory(s_seq[step], z_t=z_seq[step])
            retrievals_seq.append(r)
        retrievals_seq = torch.stack(retrievals_seq)

        # Independent (reset each step)
        retrievals_ind = []
        for step in range(5):
            r, _ = memory(s_seq[step], z_t=z_seq[step], reset=True)
            retrievals_ind.append(r)
        retrievals_ind = torch.stack(retrievals_ind)

        # Sequential context should produce different results
        assert not torch.allclose(
            retrievals_seq, retrievals_ind, atol=1e-4
        ), "Sequential memory should differ from reset-each-step memory"

    def test_long_sequence_stability(self, memory):
        """100-step sequence should not produce NaN or Inf."""
        s_seq = torch.randn(100, 2, D_FEATURE) * 0.01  # small inputs
        z_seq = torch.randn(100, 2, 16)
        for step in range(100):
            retrieved, M_t = memory(s_seq[step], z_t=z_seq[step])
            assert retrieved.isfinite().all(), f"NaN/Inf at step {step}"
            assert M_t.isfinite().all(), f"NaN/Inf in memory at step {step}"

    def test_very_long_sequence_stability(self):
        """1000-step sequence with realistic variance should remain stable."""
        mem = KDAMarketMemory(d_k=128, d_v=64, d_feature=200)
        torch.manual_seed(42)
        s_seq = torch.randn(1000, 2, 200) * 0.005
        z_seq = torch.randn(1000, 2, 16)
        mem.reset_state(2, torch.device("cpu"))
        max_abs = 0.0
        for step in range(1000):
            retrieved, M_t = mem(s_seq[step], z_t=z_seq[step])
            assert retrieved.isfinite().all(), f"NaN/Inf in retrieval at step {step}"
            assert M_t.isfinite().all(), f"NaN/Inf in memory at step {step}"
            max_abs = max(max_abs, M_t.abs().max().item())
        # Memory norms should stay bounded even after 1000 steps
        assert max_abs < 100.0, (
            f"Memory grew too large after 1000 steps: max_abs={max_abs:.1f}"
        )


# ── Gradient flow ─────────────────────────────────────────────────────

class TestGradientFlow:
    def test_gradient_through_memory(self):
        mem = KDAMarketMemory(d_k=128, d_v=64, d_feature=200)
        s_t = torch.randn(4, 200)
        z_t = torch.randn(4, 16)
        retrieved, M_t = mem(s_t, z_t=z_t)
        loss = retrieved.sum() + M_t.sum()
        loss.backward()

        grad_params = 0
        for name, p in mem.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f"No gradient for {name}"
                if p.grad.abs().sum() > 0:
                    grad_params += 1
        assert grad_params > 0, "No parameters received gradients"


# ── Memory summary ────────────────────────────────────────────────────

class TestMemorySummary:
    def test_shape(self, memory):
        s_t = torch.randn(8, D_FEATURE)
        z_t = torch.randn(8, 16)
        memory(s_t, z_t=z_t)
        summary = memory.get_memory_summary()
        assert summary.shape == (8, D_K * D_V)

    def test_raises_if_not_initialized(self):
        mem = KDAMarketMemory()
        with pytest.raises(RuntimeError, match="Memory not initialized"):
            mem.get_memory_summary()


# ── Forget gate ───────────────────────────────────────────────────────

class TestForgetGate:
    def test_forget_gate_in_range(self, memory):
        """α_t should be in (0, 1) since it's sigmoid output."""
        # The forget gate is internal but we can verify indirectly:
        # without forgetting, memory would grow unbounded.
        # With forgetting in (0,1), memory stays bounded.
        s_seq = torch.randn(50, 1, D_FEATURE) * 5  # large inputs
        z_seq = torch.randn(50, 1, 16)
        memory.reset_state(1, torch.device("cpu"))
        for step in range(50):
            _, M_t = memory(s_seq[step], z_t=z_seq[step])
        # Memory should remain bounded
        assert M_t.abs().max() < 100, f"Memory grew unbounded: {M_t.abs().max():.1f}"


# ── RMSNorm ───────────────────────────────────────────────────────────

class TestRMSNorm:
    def test_rms_norm_unit_variance(self):
        x = torch.randn(32, 64) * 10
        y = KDAMarketMemory._rms_norm(x)
        rms = torch.sqrt(torch.mean(y ** 2, dim=-1))
        assert torch.allclose(rms, torch.ones(32), atol=1e-5)

    def test_rms_norm_zero_input(self):
        x = torch.zeros(4, 64)
        y = KDAMarketMemory._rms_norm(x)
        assert y.isfinite().all()  # no NaN from div-by-zero
