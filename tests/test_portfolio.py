"""Test portfolio optimization module."""

import pytest
import torch

from daft.portfolio.markowitz import MarkowitzOptimizer


# ── Initialization ────────────────────────────────────────────────────

class TestInit:
    def test_default_params(self):
        opt = MarkowitzOptimizer()
        assert opt.risk_aversion == 1.0
        assert opt.max_weight == 0.05

    def test_custom_params(self):
        opt = MarkowitzOptimizer(
            risk_aversion=2.0, max_weight=0.1,
        )
        assert opt.risk_aversion == 2.0
        assert opt.max_weight == 0.1


# ── Optimize ──────────────────────────────────────────────────────────

class TestOptimize:
    def test_optimize_runs(self):
        """optimize() is now fully implemented — it should return weights."""
        opt = MarkowitzOptimizer()
        N = 10
        returns = torch.randn(N)
        cov = torch.eye(N)
        mask = torch.ones(N, dtype=torch.bool)
        weights = opt.optimize(returns, cov, mask)
        assert weights.shape == (N,)
        assert weights.isfinite().all()
        # Masked-out assets get zero weight
        assert weights[~mask].sum() == 0.0

    def test_optimize_masked_assets_zero(self):
        """Mask=False assets should receive zero allocation."""
        opt = MarkowitzOptimizer(max_weight=0.5)
        N = 5
        returns = torch.randn(N)
        cov = torch.eye(N)
        mask = torch.tensor([True, True, False, True, True], dtype=torch.bool)
        weights = opt.optimize(returns, cov, mask)
        assert weights[2] == 0.0
