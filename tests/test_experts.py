"""Test strategy experts: base, trend, reversal, volatility, event."""

import pytest
import torch

from daft.models.experts.base_expert import BaseExpert, SiTU
from daft.models.experts.trend_expert import TrendExpert
from daft.models.experts.reversal_expert import ReversalExpert
from daft.models.experts.volatility_expert import VolatilityExpert
from daft.models.experts.event_expert import EventExpert


BATCH_SIZES = [1, 4, 32]
INPUT_DIM = 200


# ── Fixtures ──────────────────────────────────────────────────────────

def _make_batch(batch_size, input_dim=INPUT_DIM):
    return torch.randn(batch_size, input_dim)


# ── SiTU activation ───────────────────────────────────────────────────

class TestSiTU:
    @pytest.mark.parametrize("B", BATCH_SIZES)
    def test_output_bounded(self, B):
        """SiTU output must be bounded in [-1, 1]."""
        act = SiTU()
        x = torch.randn(B, 64) * 10  # large values
        y = act(x)
        assert (y >= -1.01).all()
        assert (y <= 1.01).all()

    def test_monotonic(self):
        """SiTU is monotonically increasing over positive inputs."""
        act = SiTU()
        # Over positive inputs: sigmoid increases, |tanh| approaches 1
        x = torch.linspace(0.1, 5, 200).unsqueeze(-1)
        y = act(x).squeeze()
        assert (y[1:] >= y[:-1]).all(), "SiTU should be monotonic on x > 0"

    def test_zero_input(self):
        """SiTU(0) = σ(0)·tanh(0) = 0.5 × 0 = 0 exactly.
        Tolerance of 0.02 handles floating-point rounding."""
        act = SiTU()
        y = act(torch.zeros(10, 1))
        assert y.abs().max().item() < 0.02

    def test_extreme_values_converge(self):
        """At extreme values, output approaches bounded limits."""
        act = SiTU()
        # SiTU(x) = σ(x)·tanh(x): at extreme positive, σ→1, tanh→1 → ~1
        y_pos = act(torch.tensor([[100.0]]))
        assert y_pos.item() > 0.99
        # At extreme negative, σ→0 dominates tanh→-1 → ~0 (not -1!)
        y_neg = act(torch.tensor([[-100.0]]))
        assert abs(y_neg.item()) < 0.01
        # At moderate positive, value should be positive and growing
        y_mid = act(torch.tensor([[2.0]]))
        assert y_mid.item() > 0.8


# ── TrendExpert ───────────────────────────────────────────────────────

class TestTrendExpert:
    @pytest.mark.parametrize("B", BATCH_SIZES)
    def test_forward_shape(self, B):
        expert = TrendExpert(input_dim=INPUT_DIM, hidden_dim=64)
        s_t = _make_batch(B)
        signal = expert(s_t)
        assert signal.shape == (B, 1)

    @pytest.mark.parametrize("B", BATCH_SIZES)
    def test_signal_bounded(self, B):
        """Trend expert signals must be in [-1, 1] via SiTU."""
        expert = TrendExpert(input_dim=INPUT_DIM)
        s_t = _make_batch(B)
        signal = expert(s_t)
        assert (signal >= -1.01).all()
        assert (signal <= 1.01).all()

    @pytest.mark.parametrize("B", BATCH_SIZES)
    def test_return_hidden(self, B):
        expert = TrendExpert(input_dim=INPUT_DIM, hidden_dim=64)
        s_t = _make_batch(B)
        signal, hidden = expert(s_t, return_hidden=True)
        assert signal.shape == (B, 1)
        assert hidden.shape == (B, 64)

    def test_name(self):
        expert = TrendExpert()
        assert expert.name == "trend"

    def test_gradient_flow(self):
        expert = TrendExpert(input_dim=INPUT_DIM)
        s_t = _make_batch(8)
        signal = expert(s_t)
        loss = signal.mean()
        loss.backward()
        grads_ok = sum(
            1 for p in expert.parameters()
            if p.requires_grad and p.grad is not None and p.grad.abs().sum() > 0
        )
        total = sum(1 for p in expert.parameters() if p.requires_grad)
        assert grads_ok == total, f"Expected {total} grads, got {grads_ok}"

    def test_compute_loss_shape(self):
        expert = TrendExpert(input_dim=INPUT_DIM)
        pred = torch.randn(16, 1)
        target = torch.randn(16, 1)
        mask = torch.ones(16, 1)
        loss = expert.compute_loss(pred, target, mask)
        assert loss.ndim == 0  # scalar

    def test_compute_loss_sign_mismatch_penalty(self):
        """Trend loss penalizes sign flips more heavily."""
        expert = TrendExpert(input_dim=INPUT_DIM)
        pred = torch.tensor([[1.0]] * 4)
        target_correct = torch.tensor([[1.0]] * 4)  # same sign
        target_wrong = torch.tensor([[-1.0]] * 4)   # opposite sign
        mask = torch.ones(4, 1)
        loss_correct = expert.compute_loss(pred, target_correct, mask)
        loss_wrong = expert.compute_loss(pred, target_wrong, mask)
        assert loss_wrong > loss_correct

    def test_compute_loss_all_masked(self):
        """When mask is all zeros, loss should be finite (not NaN)
        thanks to .clamp(min=1) safety in the denominator."""
        expert = TrendExpert(input_dim=INPUT_DIM)
        pred = torch.randn(16, 1)
        target = torch.randn(16, 1)
        mask = torch.zeros(16, 1)
        loss = expert.compute_loss(pred, target, mask)
        assert loss.isfinite()
        assert loss.ndim == 0

    def test_regime_filter_not_implemented(self):
        expert = TrendExpert()
        with pytest.raises(NotImplementedError):
            expert._regime_filter(None)


# ── ReversalExpert ────────────────────────────────────────────────────

class TestReversalExpert:
    @pytest.mark.parametrize("B", BATCH_SIZES)
    def test_forward_shape(self, B):
        expert = ReversalExpert(input_dim=INPUT_DIM, hidden_dim=64)
        s_t = _make_batch(B)
        signal = expert(s_t)
        assert signal.shape == (B, 1)

    @pytest.mark.parametrize("B", BATCH_SIZES)
    def test_signal_bounded(self, B):
        expert = ReversalExpert(input_dim=INPUT_DIM)
        s_t = _make_batch(B)
        signal = expert(s_t)
        assert (signal >= -1.01).all() and (signal <= 1.01).all()

    def test_name(self):
        expert = ReversalExpert()
        assert expert.name == "reversal"

    def test_gradient_flow(self):
        expert = ReversalExpert(input_dim=INPUT_DIM)
        s_t = _make_batch(8)
        signal = expert(s_t)
        loss = signal.mean()
        loss.backward()
        grads_ok = sum(
            1 for p in expert.parameters()
            if p.requires_grad and p.grad is not None and p.grad.abs().sum() > 0
        )
        total = sum(1 for p in expert.parameters() if p.requires_grad)
        assert grads_ok == total

    def test_compute_loss_negative_ic(self):
        """Reversal loss = -IC, so perfect positive correlation → loss ≈ -1."""
        expert = ReversalExpert(input_dim=INPUT_DIM)
        pred = torch.tensor([[1.0], [2.0], [3.0]])
        target = torch.tensor([[1.0], [2.0], [3.0]])
        mask = torch.ones(3, 1)
        loss = expert.compute_loss(pred, target, mask)
        # Perfect positive corr → IC ≈ 1.0 → loss ≈ -1.0
        assert loss.item() < -0.9

    def test_compute_loss_anticorrelated(self):
        """Anti-correlated predictions → positive loss."""
        expert = ReversalExpert(input_dim=INPUT_DIM)
        pred = torch.tensor([[1.0], [2.0], [3.0]])
        target = torch.tensor([[-1.0], [-2.0], [-3.0]])
        mask = torch.ones(3, 1)
        loss = expert.compute_loss(pred, target, mask)
        # Perfect negative corr → IC ≈ -1.0 → loss ≈ 1.0
        assert loss.item() > 0.9

    def test_regime_filter_not_implemented(self):
        expert = ReversalExpert()
        with pytest.raises(NotImplementedError):
            expert._regime_filter(None)


# ── VolatilityExpert ──────────────────────────────────────────────────

class TestVolatilityExpert:
    @pytest.mark.parametrize("B", BATCH_SIZES)
    def test_forward_shape(self, B):
        expert = VolatilityExpert(input_dim=INPUT_DIM, hidden_dim=48)
        s_t = _make_batch(B)
        signal = expert(s_t)
        assert signal.shape == (B, 1)

    @pytest.mark.parametrize("B", BATCH_SIZES)
    def test_signal_bounded(self, B):
        expert = VolatilityExpert(input_dim=INPUT_DIM)
        s_t = _make_batch(B)
        signal = expert(s_t)
        assert (signal >= -1.01).all() and (signal <= 1.01).all()

    def test_name(self):
        expert = VolatilityExpert()
        assert expert.name == "volatility"

    def test_gradient_flow(self):
        expert = VolatilityExpert(input_dim=INPUT_DIM)
        s_t = _make_batch(8)
        signal = expert(s_t)
        loss = signal.mean()
        loss.backward()
        grads_ok = sum(
            1 for p in expert.parameters()
            if p.requires_grad and p.grad is not None and p.grad.abs().sum() > 0
        )
        total = sum(1 for p in expert.parameters() if p.requires_grad)
        assert grads_ok == total

    def test_compute_loss_shape(self):
        expert = VolatilityExpert(input_dim=INPUT_DIM)
        pred = torch.randn(16, 1)
        target = torch.randn(16, 1)
        mask = torch.ones(16, 1)
        loss = expert.compute_loss(pred, target, mask)
        assert loss.ndim == 0

    def test_compute_loss_positive(self):
        """MSE + variance penalty is always ≥ 0."""
        expert = VolatilityExpert(input_dim=INPUT_DIM)
        pred = torch.randn(16, 1) * 2
        target = torch.randn(16, 1)
        mask = torch.ones(16, 1)
        loss = expert.compute_loss(pred, target, mask)
        assert loss.item() >= 0

    def test_regime_filter_not_implemented(self):
        expert = VolatilityExpert()
        with pytest.raises(NotImplementedError):
            expert._regime_filter(None)


# ── EventExpert ───────────────────────────────────────────────────────

class TestEventExpert:
    @pytest.mark.parametrize("B", BATCH_SIZES)
    def test_forward_shape(self, B):
        expert = EventExpert(input_dim=INPUT_DIM, hidden_dim=48)
        s_t = _make_batch(B)
        signal = expert(s_t)
        assert signal.shape == (B, 1)

    @pytest.mark.parametrize("B", BATCH_SIZES)
    def test_signal_bounded(self, B):
        expert = EventExpert(input_dim=INPUT_DIM)
        s_t = _make_batch(B)
        signal = expert(s_t)
        assert (signal >= -1.01).all() and (signal <= 1.01).all()

    def test_name(self):
        expert = EventExpert()
        assert expert.name == "event"

    def test_gradient_flow(self):
        expert = EventExpert(input_dim=INPUT_DIM)
        s_t = _make_batch(8)
        signal = expert(s_t)
        loss = signal.mean()
        loss.backward()
        grads_ok = sum(
            1 for p in expert.parameters()
            if p.requires_grad and p.grad is not None and p.grad.abs().sum() > 0
        )
        total = sum(1 for p in expert.parameters() if p.requires_grad)
        assert grads_ok == total

    def test_compute_loss_shape(self):
        expert = EventExpert(input_dim=INPUT_DIM)
        pred = torch.randn(16, 1)
        target = torch.randn(16, 1)
        mask = torch.ones(16, 1)
        loss = expert.compute_loss(pred, target, mask)
        assert loss.ndim == 0

    def test_compute_loss_positive(self):
        """BCE loss is always ≥ 0."""
        expert = EventExpert(input_dim=INPUT_DIM)
        pred = torch.randn(16, 1) * 5
        target = torch.randn(16, 1)
        mask = torch.ones(16, 1)
        loss = expert.compute_loss(pred, target, mask)
        assert loss.item() >= 0

    def test_regime_filter_not_implemented(self):
        expert = EventExpert()
        with pytest.raises(NotImplementedError):
            expert._regime_filter(None)


# ── Expert consistency ────────────────────────────────────────────────

class TestExpertConsistency:
    """Cross-expert consistency checks."""

    @pytest.mark.parametrize("expert_cls", [
        TrendExpert, ReversalExpert, VolatilityExpert, EventExpert,
    ])
    def test_train_mode_produces_variation(self, expert_cls):
        """In train mode, Dropout(0.1) should cause variation across calls."""
        torch.manual_seed(42)
        expert = expert_cls(input_dim=INPUT_DIM).train()
        s_t = torch.randn(4, INPUT_DIM)
        # Collect outputs from multiple forward passes
        outputs = [expert(s_t) for _ in range(20)]
        stacked = torch.stack(outputs)  # (20, 4, 1)
        # With dropout active, not all outputs should be identical
        # (there's a very small chance all are the same due to dropout luck,
        #  but with 20 calls and p=0.1 it's astronomically unlikely)
        assert not torch.allclose(
            stacked[0], stacked[-1], atol=1e-6
        ), f"{expert_cls.__name__}: train mode should vary across calls"

    @pytest.mark.parametrize("expert_cls", [
        TrendExpert, ReversalExpert, VolatilityExpert, EventExpert,
    ])
    def test_deterministic_eval(self, expert_cls):
        """Same input → same output in eval mode."""
        torch.manual_seed(42)
        expert = expert_cls(input_dim=INPUT_DIM).eval()
        s_t = torch.randn(4, INPUT_DIM)
        with torch.no_grad():
            y1 = expert(s_t)
            y2 = expert(s_t)
        assert torch.allclose(y1, y2, atol=1e-6)

    @pytest.mark.parametrize("expert_cls", [
        TrendExpert, ReversalExpert, VolatilityExpert, EventExpert,
    ])
    def test_custom_hidden_dim(self, expert_cls):
        expert = expert_cls(input_dim=INPUT_DIM, hidden_dim=32, n_layers=3)
        s_t = torch.randn(4, INPUT_DIM)
        signal, hidden = expert(s_t, return_hidden=True)
        assert signal.shape == (4, 1)
        assert hidden.shape == (4, 32)

    @pytest.mark.parametrize("expert_cls", [
        TrendExpert, ReversalExpert, VolatilityExpert, EventExpert,
    ])
    def test_no_nan_output(self, expert_cls):
        expert = expert_cls(input_dim=INPUT_DIM).eval()
        s_t = torch.randn(8, INPUT_DIM) * 10  # large inputs
        with torch.no_grad():
            signal = expert(s_t)
        assert not signal.isnan().any()
