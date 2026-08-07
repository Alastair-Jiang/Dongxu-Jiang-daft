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
        assert opt.use_mosek is False

    def test_custom_params(self):
        opt = MarkowitzOptimizer(
            risk_aversion=2.0, max_weight=0.1, use_mosek=True,
        )
        assert opt.risk_aversion == 2.0
        assert opt.max_weight == 0.1
        assert opt.use_mosek is True


# ── Optimize (placeholder) ────────────────────────────────────────────

class TestOptimize:
    def test_optimize_raises_not_implemented(self):
        opt = MarkowitzOptimizer()
        N = 10
        returns = torch.randn(N)
        cov = torch.eye(N)
        mask = torch.ones(N, dtype=torch.bool)
        with pytest.raises(NotImplementedError):
            opt.optimize(returns, cov, mask)
