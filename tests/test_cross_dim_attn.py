"""Test Component 3: Cross-Dimension Attention Protocol (CDAP). ★ ORIGINAL ★"""

import pytest
import torch

from daft.models.cross_dim_attn import CrossDimensionAttention


# Fixture `cdap` is defined in conftest.py (shared across all tests)
BATCH_SIZES = [1, 4, 32]
N_EXPERTS = 10
D_K = 128
D_V = 64
N_LAYERS = 3
JOINT_DIM = 64


# ── Fixtures ──────────────────────────────────────────────────────────


def _make_inputs(B, device=None):
    routing = torch.randn(B, N_EXPERTS).softmax(dim=-1)
    memory = torch.randn(B, D_K, D_V)
    layers = [
        torch.randn(B, D_V),
        torch.randn(B, D_V),
        torch.randn(B, D_V),
    ]
    if device:
        routing = routing.to(device)
        memory = memory.to(device)
        layers = [l.to(device) for l in layers]
    return routing, memory, layers


# ── Initialization ────────────────────────────────────────────────────

class TestInit:
    def test_default_params(self):
        c = CrossDimensionAttention()
        assert c.n_experts == 10
        assert c.d_k == 128
        assert c.d_v == 64
        assert c.n_layers == 3
        assert c.joint_dim == 64
        assert c.modulation_strength == 1.0

    def test_custom_params(self):
        c = CrossDimensionAttention(
            n_experts=6, d_k=64, d_v=32,
            n_layers=2, joint_dim=48, modulation_strength=0.5,
        )
        assert c.n_experts == 6
        assert c.joint_dim == 48
        assert c.modulation_strength == 0.5

    def test_modulation_scales_initialized_zero(self, cdap):
        """Learned scales should start near zero for stable training."""
        assert cdap.expert_bias_scale.item() == pytest.approx(0.0, abs=1e-4)
        assert cdap.memory_gate_scale.item() == pytest.approx(0.0, abs=1e-4)
        assert cdap.depth_weight_scale.item() == pytest.approx(0.0, abs=1e-4)


# ── Forward pass ──────────────────────────────────────────────────────

class TestForward:
    @pytest.mark.parametrize("B", BATCH_SIZES)
    def test_output_shapes(self, cdap, B):
        routing, memory, layers = _make_inputs(B)
        routing_mod, mem_gate, depth_w, fused = cdap(routing, memory, layers)
        assert routing_mod.shape == (B, N_EXPERTS)
        assert mem_gate.shape == (B, D_K)
        assert depth_w.shape == (B, N_LAYERS)
        assert fused.shape == (B, D_V)

    @pytest.mark.parametrize("B", BATCH_SIZES)
    def test_routing_mod_sums_to_one(self, cdap, B):
        routing, memory, layers = _make_inputs(B)
        routing_mod, _, _, _ = cdap(routing, memory, layers)
        assert torch.allclose(routing_mod.sum(dim=-1), torch.ones(B), atol=1e-5)

    @pytest.mark.parametrize("B", BATCH_SIZES)
    def test_depth_weights_sum_to_one(self, cdap, B):
        routing, memory, layers = _make_inputs(B)
        _, _, depth_w, _ = cdap(routing, memory, layers)
        assert torch.allclose(depth_w.sum(dim=-1), torch.ones(B), atol=1e-5)

    @pytest.mark.parametrize("B", BATCH_SIZES)
    def test_memory_gate_in_range(self, cdap, B):
        routing, memory, layers = _make_inputs(B)
        _, mem_gate, _, _ = cdap(routing, memory, layers)
        assert (mem_gate >= 0).all()
        assert (mem_gate <= 1).all()

    @pytest.mark.parametrize("B", BATCH_SIZES)
    def test_all_finite(self, cdap, B):
        routing, memory, layers = _make_inputs(B)
        routing_mod, mem_gate, depth_w, fused = cdap(routing, memory, layers)
        assert routing_mod.isfinite().all()
        assert mem_gate.isfinite().all()
        assert depth_w.isfinite().all()
        assert fused.isfinite().all()

    def test_batch_size_one(self, cdap):
        routing, memory, layers = _make_inputs(1)
        routing_mod, _, _, _ = cdap(routing, memory, layers)
        assert routing_mod.isfinite().all()


# ── Modulation strength ───────────────────────────────────────────────

class TestModulationStrength:
    def test_modulation_path_actually_works(self):
        """Verify that CDAP modulation actually changes outputs when scales
        are non-zero (breaking the 'zero-init immunity zone').

        With zero-initialized scales (expert_bias_scale etc.), modulation
        has NO effect — see test_partial_modulation docstring. This test
        manually activates the scales to verify the reverse-projection
        mechanism works correctly.
        """
        c = CrossDimensionAttention(modulation_strength=1.0)
        # Manually activate modulation scales
        c.expert_bias_scale.data = torch.tensor([0.5])
        c.memory_gate_scale.data = torch.tensor([0.5])
        c.depth_weight_scale.data = torch.tensor([0.5])

        routing, memory, layers = _make_inputs(8)
        routing_mod, mem_gate, depth_w, fused = c(routing, memory, layers)

        # With active scales, all outputs should differ from a zero-scale
        # version (same weights, scales zeroed)
        c.expert_bias_scale.data = torch.tensor([0.0])
        c.memory_gate_scale.data = torch.tensor([0.0])
        c.depth_weight_scale.data = torch.tensor([0.0])
        routing_mod0, mem_gate0, depth_w0, fused0 = c(routing, memory, layers)

        # At least one output dimension should differ
        assert not torch.allclose(routing_mod, routing_mod0, atol=1e-4), (
            "Routing modulation had no effect even with non-zero scales"
        )
        assert not torch.allclose(mem_gate, mem_gate0, atol=1e-4), (
            "Memory gate modulation had no effect even with non-zero scales"
        )
        assert not torch.allclose(depth_w, depth_w0, atol=1e-4), (
            "Depth weight modulation had no effect even with non-zero scales"
        )

    def test_modulation_strength_matters_with_active_scales(self):
        """δ controls modulation magnitude when scales are non-zero."""
        c_half = CrossDimensionAttention(modulation_strength=0.5)
        c_full = CrossDimensionAttention(modulation_strength=1.0)
        c_full.load_state_dict(c_half.state_dict())

        # Activate scales identically in both
        for c in [c_half, c_full]:
            c.expert_bias_scale.data = torch.tensor([0.5])
            c.memory_gate_scale.data = torch.tensor([0.5])
            c.depth_weight_scale.data = torch.tensor([0.5])

        routing, memory, layers = _make_inputs(8)
        mod_half, _, _, _ = c_half(routing, memory, layers)
        mod_full, _, _, _ = c_full(routing, memory, layers)

        # δ=1.0 should deviate MORE from original than δ=0.5
        half_dev = (mod_half - routing).abs().sum(dim=-1).mean()
        full_dev = (mod_full - routing).abs().sum(dim=-1).mean()
        assert half_dev < full_dev, (
            f"δ=0.5 deviation ({half_dev:.4f}) should be < "
            f"δ=1.0 deviation ({full_dev:.4f})"
        )

    def test_zero_modulation_preserves_routing(self):
        """With modulation_strength=0, routing should still be a valid
        probability distribution, though re-softmax may sharpen it."""
        c = CrossDimensionAttention(modulation_strength=0.0)
        routing, memory, layers = _make_inputs(8)
        routing_mod, _, _, _ = c(routing, memory, layers)
        # Output should still be a valid distribution (sums to 1)
        assert torch.allclose(routing_mod.sum(dim=-1), torch.ones(8), atol=1e-5)
        # All entries non-negative
        assert (routing_mod >= 0).all()

    def test_nonzero_modulation_changes_routing(self):
        """With zero-init scales, this actually tests re-softmax sharpening,
        not modulation. See test_modulation_path_actually_works for the
        true modulation test."""
        c = CrossDimensionAttention(modulation_strength=1.0)
        routing, memory, layers = _make_inputs(8)
        routing_mod, _, _, _ = c(routing, memory, layers)
        # Output differs from input due to re-softmax sharpening
        # (softmax is not idempotent)
        assert not torch.allclose(routing_mod, routing, atol=1e-3)

    def test_partial_modulation(self):
        """δ=0.5 should change routing but less than δ=1.0.

        NOTE: At initialization, expert_bias_scale ≈ 0, so modulation
        has near-zero effect regardless of δ. The deviations seen are
        from re-softmax sharpening, not from the modulation itself.
        Once training has learned non-zero scales, δ controls modulation
        magnitude.
        """
        c_half = CrossDimensionAttention(modulation_strength=0.5)
        c_full = CrossDimensionAttention(modulation_strength=1.0)
        c_full.load_state_dict(c_half.state_dict())

        routing, memory, layers = _make_inputs(8)
        mod_half, _, _, _ = c_half(routing, memory, layers)
        mod_full, _, _, _ = c_full(routing, memory, layers)

        # With zero-initialized scales, both produce valid distributions
        assert torch.allclose(mod_half.sum(dim=-1), torch.ones(8), atol=1e-5)
        assert torch.allclose(mod_full.sum(dim=-1), torch.ones(8), atol=1e-5)
        # They are identical (since modulation ≈ 0 in both)
        # but both are valid softmax outputs
        assert (mod_half >= 0).all()
        assert (mod_full >= 0).all()


# ── Gradient flow ─────────────────────────────────────────────────────

class TestGradientFlow:
    def test_all_params_receive_gradients(self):
        """CDAP: backward pass executes without error and most params get grads."""
        c = CrossDimensionAttention()
        routing, memory, layers = _make_inputs(8)
        routing_mod, mem_gate, depth_w, fused = c(routing, memory, layers)
        # Use a combined loss that exercises all output paths
        loss = (
            routing_mod.sum() + mem_gate.sum() +
            depth_w.sum() + fused.sum()
        )
        loss.backward()

        no_grad_params = []
        for name, p in c.named_parameters():
            if p.requires_grad and p.grad is None:
                # memory_gate_decay is only used when residual_gate=True;
                # it is expected to have no gradient in standard mode.
                if name == "memory_gate_decay" and not c.residual_gate:
                    continue
                no_grad_params.append(name)

        assert len(no_grad_params) == 0, (
            f"Params with no gradient: {no_grad_params}"
        )


# ── Joint space properties ────────────────────────────────────────────

class TestJointSpace:
    def test_elementwise_product_inductive_bias(self, cdap):
        """If any dimension input is zeros, joint is near-zero
        (elementwise product = 0), so gate ≈ sigmoid(0) = 0.5."""
        routing, memory, layers = _make_inputs(8)

        # Case 1: Zero routing → joint element e=0 → product=0
        zero_routing = torch.zeros_like(routing)
        _, gate1, _, _ = cdap(zero_routing, memory, layers)
        assert torch.allclose(gate1, torch.full_like(gate1, 0.5), atol=0.01)

        # Case 2: Zero memory → joint element m=0 → product=0
        zero_memory = torch.zeros_like(memory)
        _, gate2, _, _ = cdap(routing, zero_memory, layers)
        assert torch.allclose(gate2, torch.full_like(gate2, 0.5), atol=0.01)

        # Case 3: Zero depth layers → joint element d=0 → product=0
        zero_layers = [torch.zeros_like(l) for l in layers]
        _, gate3, _, _ = cdap(routing, memory, zero_layers)
        assert torch.allclose(gate3, torch.full_like(gate3, 0.5), atol=0.01)

    def test_same_input_deterministic(self, cdap):
        """CDAP is a deterministic module (no noise/dropout) →
        same input always produces identical output."""
        torch.manual_seed(42)
        routing, memory, layers = _make_inputs(8)
        cdap.eval()
        with torch.no_grad():
            out1 = cdap(routing, memory, layers)
            out2 = cdap(routing, memory, layers)
        # All 4 outputs should be bit-identical
        for i, (o1, o2) in enumerate(zip(out1, out2)):
            assert torch.equal(o1, o2), (
                f"Output {i} differs across calls (max diff: "
                f"{(o1 - o2).abs().max():.2e})"
            )


# ── Fused layers ──────────────────────────────────────────────────────

class TestFusedLayers:
    def test_fused_is_convex_combination(self, cdap):
        """Fused output is a convex combination of layer outputs."""
        routing, memory, layers = _make_inputs(8)
        _, _, depth_w, fused = cdap(routing, memory, layers)

        # Manual convex combination
        manual = sum(
            depth_w[:, k:k+1] * layers[k] for k in range(3)
        )
        assert torch.allclose(fused, manual, atol=1e-5)

    def test_single_layer_dominance(self, cdap):
        """If one layer has massively larger magnitude, the fused output
        should be dominated by that layer in the convex combination."""
        routing, memory, layers = _make_inputs(8)
        # Make L0 100× larger — it should dominate the weighted sum
        layers[0] = layers[0] * 100
        _, _, depth_w, fused = cdap(routing, memory, layers)
        assert fused.isfinite().all()

        # Cosine similarity: fused should be much more aligned with
        # the dominant layer (L0) than with any other layer
        def cos_sim(a, b):
            return torch.nn.functional.cosine_similarity(a, b, dim=-1).mean()

        sim_L0 = cos_sim(fused, layers[0])
        sim_L1 = cos_sim(fused, layers[1])
        sim_L2 = cos_sim(fused, layers[2])
        assert sim_L0 > sim_L1, (
            f"Fused should be more aligned with dominant L0 "
            f"(sim_L0={sim_L0:.3f}, sim_L1={sim_L1:.3f})"
        )
        assert sim_L0 > sim_L2, (
            f"Fused should be more aligned with dominant L0 "
            f"(sim_L0={sim_L0:.3f}, sim_L2={sim_L2:.3f})"
        )


# ── Residual Gate ──────────────────────────────────────────────────────

class TestResidualGate:
    """Tests for opt-in residual memory gate (3-layer defense)."""

    def test_default_is_backward_compatible(self):
        """Default residual_gate=False preserves original behavior."""
        c = CrossDimensionAttention()
        assert c.residual_gate is False
        # memory_gate_decay exists but doesn't affect standard mode
        assert c.memory_gate_decay is not None

    def test_residual_gate_defaults_to_one(self):
        """With residual_gate=True, gate defaults to ~1.0 (identity)
        instead of ~0.5."""
        c = CrossDimensionAttention(residual_gate=True)
        routing, memory, layers = _make_inputs(8)
        _, gate, _, _ = c(routing, memory, layers)
        # At init, raw ≈ 0 → 2·σ(0) = 1.0
        assert torch.allclose(gate, torch.ones_like(gate), atol=0.05), (
            f"Residual gate should default to ~1.0, got mean={gate.mean():.4f}"
        )

    def test_standard_gate_defaults_to_half(self):
        """Standard (residual_gate=False) gate defaults to ~0.5."""
        c = CrossDimensionAttention(residual_gate=False)
        routing, memory, layers = _make_inputs(8)
        _, gate, _, _ = c(routing, memory, layers)
        # At init, raw ≈ 0 → σ(0) = 0.5
        assert torch.allclose(gate, 0.5 * torch.ones_like(gate), atol=0.05), (
            f"Standard gate should default to ~0.5, got mean={gate.mean():.4f}"
        )

    def test_residual_gate_range(self):
        """Residual gate ∈ (0, 2) due to 2×sigmoid."""
        c = CrossDimensionAttention(residual_gate=True)
        routing, memory, layers = _make_inputs(32)
        _, gate, _, _ = c(routing, memory, layers)
        assert (gate > 0).all(), "Residual gate should be > 0"
        assert (gate < 2).all(), "Residual gate should be < 2"

    def test_standard_gate_range(self):
        """Standard gate ∈ (0, 1)."""
        c = CrossDimensionAttention(residual_gate=False)
        routing, memory, layers = _make_inputs(32)
        _, gate, _, _ = c(routing, memory, layers)
        assert (gate > 0).all(), "Standard gate should be > 0"
        assert (gate < 1).all(), "Standard gate should be < 1"

    def test_runtime_toggle(self):
        """residual_gate can be toggled at runtime."""
        c = CrossDimensionAttention(residual_gate=False)
        routing, memory, layers = _make_inputs(8)

        # Standard mode
        _, gate_std, _, _ = c(routing, memory, layers)
        assert gate_std.max() <= 1.0

        # Toggle to residual mode
        c.residual_gate = True
        _, gate_res, _, _ = c(routing, memory, layers)
        assert gate_res.max() > 1.0 or torch.allclose(gate_res, torch.ones_like(gate_res), atol=0.1)

        # Toggle back
        c.residual_gate = False
        _, gate_std2, _, _ = c(routing, memory, layers)
        assert gate_std2.max() <= 1.0

    def test_output_shapes_unchanged(self):
        """Residual gate doesn't change output shapes."""
        c_std = CrossDimensionAttention(residual_gate=False)
        c_res = CrossDimensionAttention(residual_gate=True)
        c_res.load_state_dict(c_std.state_dict())

        routing, memory, layers = _make_inputs(8)
        out_std = c_std(routing, memory, layers)
        out_res = c_res(routing, memory, layers)

        for i in range(4):
            assert out_std[i].shape == out_res[i].shape, (
                f"Output {i} shape mismatch: {out_std[i].shape} vs {out_res[i].shape}"
            )

    def test_gate_differs_when_residual_enabled(self):
        """With identical weights, residual vs standard produce different gates."""
        c_std = CrossDimensionAttention(residual_gate=False)
        c_res = CrossDimensionAttention(residual_gate=True)
        c_res.load_state_dict(c_std.state_dict())

        routing, memory, layers = _make_inputs(32)
        _, gate_std, _, _ = c_std(routing, memory, layers)
        _, gate_res, _, _ = c_res(routing, memory, layers)

        # Gates should differ (2× shifts the center)
        assert not torch.allclose(gate_std, gate_res, atol=1e-3), (
            "Residual and standard gates should differ"
        )

    def test_all_params_receive_gradients_with_residual(self):
        """All params (incl. memory_gate_decay) get gradients with residual_gate."""
        c = CrossDimensionAttention(residual_gate=True)
        routing, memory, layers = _make_inputs(8)
        routing_mod, mem_gate, depth_w, fused = c(routing, memory, layers)
        loss = (
            routing_mod.sum() + mem_gate.sum() +
            depth_w.sum() + fused.sum()
        )
        loss.backward()

        no_grad_params = []
        for name, p in c.named_parameters():
            if p.requires_grad and p.grad is None:
                no_grad_params.append(name)

        assert len(no_grad_params) == 0, (
            f"Params with no gradient: {no_grad_params}"
        )

    def test_memory_gate_decay_trainable(self):
        """memory_gate_decay is a trainable parameter."""
        c = CrossDimensionAttention(residual_gate=True)
        assert c.memory_gate_decay.requires_grad
        assert c.memory_gate_decay.item() == pytest.approx(0.0, abs=1e-4)

    def test_decay_effect_on_gate(self):
        """Higher decay compresses enhancement direction more aggressively."""
        c = CrossDimensionAttention(residual_gate=True)

        # Manually create a strong positive raw signal
        routing, memory, layers = _make_inputs(8)
        with torch.no_grad():
            # Artificially boost the scale so raw signal is large
            c.memory_gate_scale.data = torch.tensor([2.0])

        # With decay=0 (no compression)
        c.memory_gate_decay.data = torch.tensor([0.0])
        _, gate_no_decay, _, _ = c(routing, memory, layers)

        # With decay≈1 (max compression on enhancement)
        c.memory_gate_decay.data = torch.tensor([3.0])  # tanh(3) ≈ 0.995
        _, gate_decay, _, _ = c(routing, memory, layers)

        # Decay should reduce gate values for the enhancement direction
        # (positive raw) but allow suppression (negative raw) freely.
        # Since memory_gate_scale > 0, more pos raw → more gate, but decay shrinks it.
        # We verify decay changed gate values.
        assert not torch.allclose(gate_no_decay, gate_decay, atol=1e-3), (
            "memory_gate_decay should affect gate values"
        )

    def test_gate_deviation_loss_none_when_standard(self):
        """get_gate_deviation_loss() returns None for standard gate."""
        c = CrossDimensionAttention(residual_gate=False)
        routing, memory, layers = _make_inputs(8)
        c(routing, memory, layers)
        assert c.get_gate_deviation_loss() is None

    def test_gate_deviation_loss_not_none_when_residual(self):
        """get_gate_deviation_loss() returns valid loss for residual gate."""
        c = CrossDimensionAttention(residual_gate=True)
        routing, memory, layers = _make_inputs(8)
        c(routing, memory, layers)
        dev_loss = c.get_gate_deviation_loss()
        assert dev_loss is not None
        # At init, gate ≈ 1.0 → deviation ≈ 0
        assert dev_loss.item() == pytest.approx(0.0, abs=0.01)

    def test_gate_deviation_loss_grows_with_deviation(self):
        """Deviation loss increases as gate moves away from 1.0."""
        c = CrossDimensionAttention(residual_gate=True)
        routing, memory, layers = _make_inputs(8)

        # Force gate away from 1.0 by activating scale
        c.memory_gate_scale.data = torch.tensor([2.0])
        c(routing, memory, layers)
        dev_loss_high = c.get_gate_deviation_loss().item()

        # Reset and use zero scale (gate ≈ 1.0)
        c.memory_gate_scale.data = torch.tensor([0.0])
        c(routing, memory, layers)
        dev_loss_low = c.get_gate_deviation_loss().item()

        assert dev_loss_low < dev_loss_high, (
            f"Deviation loss with scale=0 ({dev_loss_low:.6f}) should be "
            f"less than with scale=2 ({dev_loss_high:.6f})"
        )
