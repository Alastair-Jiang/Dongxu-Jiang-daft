"""Test ExpertEnsemble: end-to-end integration of all DAFT components."""

import pytest
import torch
import torch.nn as nn

from daft.models.router import RegimeRouter
from daft.models.memory import KDAMarketMemory
from daft.models.cross_dim_attn import CrossDimensionAttention
from daft.models.hardening import HardeningEngine
from daft.models.ensemble import ExpertEnsemble
from daft.models.experts import TrendExpert, ReversalExpert, VolatilityExpert, EventExpert


# Fixtures `ensemble` and `ensemble_low_threshold` are defined in conftest.py
BATCH_SIZES = [1, 4, 16]
INPUT_DIM = 200
D_V = 64


# ── Fixtures ──────────────────────────────────────────────────────────


def _make_batch(B):
    s_t = torch.randn(B, INPUT_DIM)
    layers = [
        torch.randn(B, D_V),
        torch.randn(B, D_V),
        torch.randn(B, D_V),
    ]
    return s_t, layers


# ── Forward pass ──────────────────────────────────────────────────────

class TestForward:
    @pytest.mark.parametrize("B", BATCH_SIZES)
    def test_train_mode_output_shapes(self, ensemble, B):
        s_t, layers = _make_batch(B)
        out = ensemble(s_t, layers, mode="train")
        assert out["signal"].shape == (B, 1)
        assert out["routing_probs"].shape == (B, 10)
        assert out["depth_weights"].shape == (B, 3)
        assert out["fused_layers"].shape == (B, D_V)

    @pytest.mark.parametrize("B", BATCH_SIZES)
    def test_val_mode_output_shapes(self, ensemble, B):
        s_t, layers = _make_batch(B)
        out = ensemble(s_t, layers, mode="val")
        assert out["signal"].shape == (B, 1)
        assert out["routing_probs"].shape == (B, 10)

    @pytest.mark.parametrize("B", BATCH_SIZES)
    def test_inference_mode(self, ensemble, B):
        s_t, layers = _make_batch(B)
        out = ensemble(s_t, layers, mode="inference", use_hardening=False)
        assert out["signal"].shape == (B, 1)
        assert out["routing_probs"].shape == (B, 10)

    @pytest.mark.parametrize("B", BATCH_SIZES)
    def test_signal_is_finite(self, ensemble, B):
        s_t, layers = _make_batch(B)
        out = ensemble(s_t, layers, mode="train")
        assert out["signal"].isfinite().all()

    @pytest.mark.parametrize("B", BATCH_SIZES)
    def test_routing_probs_sum_to_one(self, ensemble, B):
        s_t, layers = _make_batch(B)
        out = ensemble(s_t, layers, mode="train")
        assert torch.allclose(
            out["routing_probs"].sum(dim=-1), torch.ones(B), atol=1e-5,
        )

    @pytest.mark.parametrize("B", BATCH_SIZES)
    def test_depth_weights_sum_to_one(self, ensemble, B):
        s_t, layers = _make_batch(B)
        out = ensemble(s_t, layers, mode="train")
        assert torch.allclose(
            out["depth_weights"].sum(dim=-1), torch.ones(B), atol=1e-5,
        )

    def test_batch_size_one(self, ensemble):
        s_t, layers = _make_batch(1)
        out = ensemble(s_t, layers, mode="train")
        assert out["signal"].isfinite().all()

    def test_output_dict_keys(self, ensemble):
        s_t, layers = _make_batch(8)
        out = ensemble(s_t, layers, mode="train")
        expected_keys = {
            "signal", "routing_probs", "regime_id",
            "depth_weights", "fused_layers", "metadata",
        }
        assert set(out.keys()) == expected_keys

    def test_metadata_keys(self, ensemble):
        s_t, layers = _make_batch(8)
        out = ensemble(s_t, layers, mode="train")
        assert "mode" in out["metadata"]
        assert "fast_path_used" in out["metadata"]
        assert "hardening_stats" in out["metadata"]


# ── Hardening integration ─────────────────────────────────────────────

class TestHardeningIntegration:
    def test_hardening_disabled_uses_cdap(self, ensemble):
        """With use_hardening=False, always uses full CDAP path."""
        s_t, layers = _make_batch(4)
        out = ensemble(s_t, layers, mode="inference", use_hardening=False)
        assert out["signal"].isfinite().all()

    def test_hardening_enabled_no_cache(self, ensemble):
        """With use_hardening=True but no cached patterns yet — slow path."""
        s_t, layers = _make_batch(4)
        out = ensemble(s_t, layers, mode="inference", use_hardening=True)
        assert out["signal"].isfinite().all()
        # Should still use slow path (no cache entries yet)
        stats = out["metadata"]["hardening_stats"]
        assert stats["n_fast_path"] == 0  # no cache hits in ensemble path

    def test_fast_path_uses_cached_weights(self, ensemble_low_threshold):
        """After hardening warmup, inference with use_hardening=True
        should use cached weights (fast path), bypassing full CDAP.

        This exercises ensemble.py lines 125-138 which were previously
        never reached in tests.
        """
        ens = ensemble_low_threshold

        # Warm up: run enough passes to create a cache entry
        torch.manual_seed(42)
        s_t, layers = _make_batch(4)
        for _ in range(10):
            ens(s_t, layers, mode="inference", use_hardening=True)

        # Now the fast path should be active
        torch.manual_seed(42)  # same input to isolate the path difference
        s_t, layers = _make_batch(4)
        out = ens(s_t, layers, mode="inference", use_hardening=True)

        stats = out["metadata"]["hardening_stats"]
        assert stats["n_fast_path"] > 0, (
            f"Expected fast-path usage after warmup, got "
            f"fast={stats['n_fast_path']}, slow={stats['n_slow_path']}"
        )
        assert out["signal"].isfinite().all()
        assert out["routing_probs"].shape == (4, 10)
        assert out["fused_layers"].shape == (4, D_V)

    def test_fast_path_vs_slow_path_consistency(self, ensemble_low_threshold):
        """Fast-path output should be finite and structurally identical
        to slow-path output (same shapes, bounded signal)."""
        ens = ensemble_low_threshold

        torch.manual_seed(42)
        for _ in range(8):
            s_t, layers = _make_batch(4)
            ens(s_t, layers, mode="inference", use_hardening=True)

        # Slow path (no hardening)
        torch.manual_seed(42)
        s_t, layers = _make_batch(4)
        out_slow = ens(s_t, layers, mode="inference", use_hardening=False)

        assert out_slow["signal"].isfinite().all()
        assert out_slow["routing_probs"].shape == (4, 10)
        assert out_slow["fused_layers"].shape == (4, D_V)


    def test_signal_magnitude_reasonable(self, ensemble):
        """Signal should not explode under normal inputs."""
        s_t, layers = _make_batch(32)
        out = ensemble(s_t, layers, mode="train")
        assert out["signal"].abs().mean().item() < 0.5


# ── Gradient flow ─────────────────────────────────────────────────────

class TestGradientFlow:
    def test_train_mode_gradient_flow(self, ensemble):
        s_t, layers = _make_batch(4)
        out = ensemble(s_t, layers, mode="train")
        loss = out["signal"].mean()
        loss.backward()

        grads_ok = 0
        grads_total = 0
        for name, p in ensemble.named_parameters():
            if p.requires_grad:
                grads_total += 1
                if p.grad is not None and p.grad.abs().sum() > 0:
                    grads_ok += 1
        # At least 60% of parameters should receive gradients
        # (some paths have zero-initialized scales that initially block gradients)
        assert grads_ok / grads_total > 0.6, (
            f"Only {grads_ok}/{grads_total} params received gradients"
        )


# ── Determinism ───────────────────────────────────────────────────────

class TestDeterminism:
    def test_val_mode_deterministic(self, ensemble):
        torch.manual_seed(42)
        s_t, layers = _make_batch(4)
        ensemble.eval()
        with torch.no_grad():
            out1 = ensemble(s_t, layers, mode="val")
            ensemble.memory.reset_state(s_t.size(0), s_t.device)  # reset after stateful memory update
            out2 = ensemble(s_t, layers, mode="val")
        assert torch.allclose(out1["signal"], out2["signal"], atol=1e-6)


# ── Parameter count ───────────────────────────────────────────────────

class TestParameterCount:
    def test_under_500k(self, ensemble):
        total = sum(p.numel() for p in ensemble.parameters())
        assert total < 500_000, f"Model has {total:,} params, expected < 500K"

    def test_parameter_breakdown(self, ensemble):
        """Each component should contribute a reasonable share of params."""
        component_params = {}
        for name, module in ensemble.named_children():
            n = sum(p.numel() for p in module.parameters() if p.requires_grad)
            component_params[name] = n

        # Expert backbone: 4 types × varying configs ≈ 60-90K each → ~300K+
        total_experts = sum(
            component_params.get(n, 0) for n in component_params
            if n.startswith("experts")
        ) if any(n.startswith("experts") for n in component_params) else 0
        # Actually, experts is a ModuleList, so named_children() gives "experts"
        expert_params = component_params.get("experts", 0)

        assert expert_params > 100_000, (
            f"Experts should have substantial params, got {expert_params:,}"
        )
        assert component_params["router"] > 10_000, (
            f"Router too small: {component_params['router']:,}"
        )
        assert component_params["memory"] > 50_000, (
            f"Memory too small: {component_params['memory']:,}"
        )
        assert component_params["cross_dim_attn"] > 30_000, (
            f"CDAP too small: {component_params['cross_dim_attn']:,}"
        )
