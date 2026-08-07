"""Test the momentum expert's regime filter and loss."""

import pytest
import torch

from daft.models.experts.momentum_expert import MomentumExpert, _compute_momentum_mask
from daft.models.experts.base_expert import SiTU
from daft.data.panel import Panel


def _make_panel(T=80, N=5, seed=0):
    """Build a Panel with synthetic OHLCV and an injected momentum episode."""
    torch.manual_seed(seed)
    values = torch.randn(T, N, 5)
    # Inject a persistent upward drift in the middle → momentum episode
    close = torch.linspace(1.0, 1.0, T).unsqueeze(1).expand(T, N)
    drift = torch.zeros(T, N)
    drift[30:55] = 0.02  # persistent move
    close = close + drift + 0.01 * torch.randn(T, N)
    values[..., 3] = close
    mask = torch.ones(T, N, dtype=torch.bool)
    return Panel(
        values=values,
        mask=mask,
        feature_names=["open", "high", "low", "close", "volume"],
    )


def test_momentum_mask_shape():
    p = _make_panel(T=80, N=5)
    mask = _compute_momentum_mask(p)
    assert mask.shape == (80,)
    assert mask.dtype == torch.bool


def test_momentum_mask_detects_drift():
    """A persistent upward drift should be flagged active."""
    p = _make_panel(T=80, N=5)
    mask = _compute_momentum_mask(p)
    # The drift window [30,55] should be largely active
    active_in_window = mask[35:55].float().mean()
    assert active_in_window > 0.5, f"expected momentum window active, got {active_in_window}"


def test_momentum_expert_forward_and_loss():
    exp = MomentumExpert(input_dim=200, hidden_dim=32, n_layers=2)
    x = torch.randn(16, 200)
    sig = exp(x)
    assert sig.shape == (16, 1)
    assert sig.abs().max().item() <= 1.01  # SiTU-bounded

    pred = torch.randn(16, 1)
    target = torch.randn(16, 1)
    mask = torch.ones(16, 1)
    loss = exp.compute_loss(pred, target, mask)
    assert torch.isfinite(loss)
    assert loss.item() >= 0


def test_momentum_loss_penalizes_wrong_sign():
    exp = MomentumExpert(input_dim=200, hidden_dim=32, n_layers=2)
    mask = torch.ones(4, 1)
    # Perfect sign: loss should be small
    pred_ok = torch.tensor([[0.5], [0.5], [-0.5], [-0.5]])
    tgt = torch.tensor([[0.5], [0.5], [-0.5], [-0.5]])
    l_ok = exp.compute_loss(pred_ok, tgt, mask)
    # Wrong sign: loss should be larger
    pred_wrong = torch.tensor([[0.5], [0.5], [0.5], [0.5]])
    l_wrong = exp.compute_loss(pred_wrong, tgt, mask)
    assert l_wrong.item() > l_ok.item()