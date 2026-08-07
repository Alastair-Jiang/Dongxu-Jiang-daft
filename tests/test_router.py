"""Test Component 1: Regime Router (Stable LatentMoE)."""

import pytest
import torch

from daft.models.router import RegimeRouter


# Fixture `router` is defined in conftest.py (shared across all tests)
BATCH_SIZES = [1, 4, 32, 64]


# ── Initialization ────────────────────────────────────────────────────

class TestInit:
    def test_default_params(self):
        r = RegimeRouter()
        assert r.input_dim == 200
        assert r.latent_dim == 16
        assert r.n_experts == 8
        assert r.top_k == 3
        assert r.temperature == 1.0

    def test_custom_params(self):
        r = RegimeRouter(input_dim=100, latent_dim=8, n_experts=6, top_k=2)
        assert r.input_dim == 100
        assert r.latent_dim == 8
        assert r.n_experts == 6
        assert r.top_k == 2

    def test_bias_initialized_zero(self, router):
        assert (router.expert_bias == 0).all()

    def test_activation_counts_initialized_zero(self, router):
        assert (router.activation_counts == 0).all()


# ── Forward pass ──────────────────────────────────────────────────────

class TestForward:
    @pytest.mark.parametrize("B", BATCH_SIZES)
    def test_output_shapes(self, router, B):
        s_t = torch.randn(B, router.input_dim)
        topk_probs, topk_indices, z_t, full_probs = router(s_t, mode="train")
        assert topk_probs.shape == (B, router.top_k)
        assert topk_indices.shape == (B, router.top_k)
        assert z_t.shape == (B, router.latent_dim)
        assert full_probs.shape == (B, router.n_experts)

    @pytest.mark.parametrize("B", BATCH_SIZES)
    def test_topk_probs_sum_to_one(self, router, B):
        s_t = torch.randn(B, router.input_dim)
        topk_probs, _, _, _ = router(s_t, mode="train")
        assert torch.allclose(topk_probs.sum(dim=-1), torch.ones(B), atol=1e-5)

    @pytest.mark.parametrize("B", BATCH_SIZES)
    def test_full_probs_sum_to_one(self, router, B):
        s_t = torch.randn(B, router.input_dim)
        _, _, _, full_probs = router(s_t, mode="train")
        assert torch.allclose(full_probs.sum(dim=-1), torch.ones(B), atol=1e-5)

    @pytest.mark.parametrize("B", BATCH_SIZES)
    def test_all_topk_positive(self, router, B):
        s_t = torch.randn(B, router.input_dim)
        topk_probs, _, _, _ = router(s_t, mode="train")
        assert (topk_probs > 0).all()

    @pytest.mark.parametrize("B", BATCH_SIZES)
    def test_topk_indices_in_range(self, router, B):
        s_t = torch.randn(B, router.input_dim)
        _, topk_indices, _, _ = router(s_t, mode="train")
        assert (topk_indices >= 0).all()
        assert (topk_indices < router.n_experts).all()


# ── Modes ─────────────────────────────────────────────────────────────

class TestModes:
    def test_train_mode_adds_noise(self, router):
        """Train mode adds exploration noise — outputs should differ across calls."""
        torch.manual_seed(42)
        s_t = torch.randn(8, router.input_dim)
        t1 = router(s_t, mode="train")
        t2 = router(s_t, mode="train")
        # With noisy_gating_std=0.1, two forward passes should produce
        # different routing distributions (we check all 4 return tensors)
        all_same = all(torch.equal(t1[i], t2[i]) for i in range(4))
        assert not all_same, "Train mode should produce non-identical outputs across calls"

    def test_val_mode_deterministic(self, router):
        torch.manual_seed(42)
        s_t = torch.randn(8, router.input_dim)
        with torch.no_grad():
            p1, _, _, _ = router(s_t, mode="val")
            p2, _, _, _ = router(s_t, mode="val")
        assert torch.allclose(p1, p2, atol=1e-6)

    def test_inference_mode_lower_temperature(self, router):
        """Inference (temp=0.1) should produce sharper routing than val (temp=1.0)."""
        torch.manual_seed(42)
        s_t = torch.randn(8, router.input_dim)
        with torch.no_grad():
            val_probs, _, _, _ = router(s_t, mode="val")
            inf_probs, _, _, _ = router(s_t, mode="inference")
        # Both should be valid distributions
        assert torch.allclose(val_probs.sum(dim=-1), torch.ones(8), atol=1e-5)
        assert torch.allclose(inf_probs.sum(dim=-1), torch.ones(8), atol=1e-5)
        # Inference entropy should be ≤ val entropy (sharper from lower temp)
        val_entropy = -(val_probs * (val_probs + 1e-8).log()).sum(dim=-1).mean()
        inf_entropy = -(inf_probs * (inf_probs + 1e-8).log()).sum(dim=-1).mean()
        assert inf_entropy <= val_entropy + 0.01, (
            f"Inference entropy should ≤ val entropy: "
            f"inf={inf_entropy:.4f}, val={val_entropy:.4f}"
        )

    def test_noisy_gating_zero_disables_noise(self):
        r = RegimeRouter(noisy_gating_std=0.0)
        torch.manual_seed(42)
        s_t = torch.randn(8, r.input_dim)
        p1, _, _, _ = r(s_t, mode="train")
        p2, _, _, _ = r(s_t, mode="train")
        assert torch.allclose(p1, p2, atol=1e-6)


# ── Quantile Balancing ────────────────────────────────────────────────

class TestQuantileBalancing:
    def test_no_op_when_no_activations(self, router):
        """Quantile balance is a no-op when activation_counts are all zero."""
        old_bias = router.expert_bias.clone()
        router.quantile_balance(lr=0.01)
        assert torch.equal(router.expert_bias, old_bias)

    def test_balances_after_forward(self, router):
        """After routing a batch, quantile_balance should adjust biases."""
        s_t = torch.randn(64, router.input_dim)
        router(s_t, mode="train")
        old_bias = router.expert_bias.clone()
        router.quantile_balance(lr=1.0)  # aggressive lr for visibility
        # Bias should change after routing + balancing
        assert not torch.allclose(router.expert_bias, old_bias)

    def test_no_nan_after_balance(self, router):
        s_t = torch.randn(64, router.input_dim)
        router(s_t, mode="train")
        router.quantile_balance(lr=1.0)
        assert not router.expert_bias.isnan().any()


# ── Regime Identification ─────────────────────────────────────────────

class TestRegimeId:
    def test_get_regime_id_shape(self, router):
        z_t = torch.randn(8, router.latent_dim)
        regime_id = router.get_regime_id(z_t)
        assert regime_id.shape == (8,)
        assert regime_id.dtype == torch.int64

    def test_get_regime_id_in_range(self, router):
        z_t = torch.randn(32, router.latent_dim)
        regime_id = router.get_regime_id(z_t)
        assert (regime_id >= 0).all()
        assert (regime_id < router.n_experts).all()


# ── Gradient flow ─────────────────────────────────────────────────────

class TestGradientFlow:
    def test_all_params_receive_gradients(self):
        r = RegimeRouter(input_dim=200, latent_dim=16, n_experts=8)
        s_t = torch.randn(16, 200)
        topk_probs, _, _, full_probs = r(s_t, mode="train")
        loss = topk_probs.sum() + full_probs.sum()
        loss.backward()

        for name, p in r.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f"No gradient for {name}"
                assert p.grad.abs().sum() > 0, f"Zero gradient for {name}"
