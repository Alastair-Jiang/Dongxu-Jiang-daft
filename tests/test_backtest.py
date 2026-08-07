"""Test backtesting engine and performance metrics."""

import pytest
import torch

from daft.backtest.engine import BacktestEngine


# ── Initialization ────────────────────────────────────────────────────

class TestInit:
    def test_creates_with_config(self):
        engine = BacktestEngine({"rebalance_freq": "daily"})
        assert engine.config["rebalance_freq"] == "daily"

    def test_default_empty_config(self):
        engine = BacktestEngine({})
        assert engine.config == {}


# ── run (placeholder) ─────────────────────────────────────────────────

class TestRun:
    def test_raises_not_implemented(self):
        engine = BacktestEngine({})
        with pytest.raises(NotImplementedError):
            engine.run(
                signals=torch.randn(100, 10),
                prices=torch.randn(100, 10),
                mask=torch.ones(100, 10, dtype=torch.bool),
            )


# ── Sharpe ratio (static method) ──────────────────────────────────────

class TestSharpeRatio:
    def test_positive_returns(self):
        """Positive mean with small variance → positive Sharpe."""
        returns = torch.tensor([0.01] * 99 + [0.011])  # tiny variance
        sr = BacktestEngine.sharpe_ratio(returns, annualization=252)
        assert sr > 0

    def test_zero_mean_returns(self):
        """Zero-mean returns → Sharpe ≈ 0."""
        returns = torch.tensor([0.01, -0.01] * 50)
        sr = BacktestEngine.sharpe_ratio(returns, annualization=252)
        assert abs(sr) < 0.5

    def test_negative_returns(self):
        """Consistent negative returns with small variance → negative Sharpe."""
        returns = torch.tensor([-0.01] * 99 + [-0.011])
        sr = BacktestEngine.sharpe_ratio(returns, annualization=252)
        assert sr < 0

    def test_zero_std_returns(self):
        """Zero-variance returns → Sharpe = 0."""
        returns = torch.zeros(100)
        sr = BacktestEngine.sharpe_ratio(returns, annualization=252)
        assert sr == 0.0

    def test_annualization(self):
        """Sharpe annualized=252 should be sqrt(252)× non-annualized Sharpe."""
        # Use returns with non-zero mean AND non-zero std
        returns = torch.tensor([0.001, -0.002, 0.003, -0.001, 0.002] * 20)
        sr_1 = BacktestEngine.sharpe_ratio(returns, annualization=1)
        sr_252 = BacktestEngine.sharpe_ratio(returns, annualization=252)
        # sqrt(252) ≈ 15.87
        assert sr_252 == pytest.approx(sr_1 * (252 ** 0.5), rel=1e-4)

    def test_daily_annualization_factor(self):
        """Sharpe with annualization=252 should be ~sqrt(252) × daily Sharpe."""
        torch.manual_seed(42)
        returns = torch.randn(1000) * 0.01 + 0.001
        sr_daily = BacktestEngine.sharpe_ratio(returns, annualization=1)
        sr_annual = BacktestEngine.sharpe_ratio(returns, annualization=252)
        if sr_daily != 0:
            ratio = sr_annual / sr_daily
            expected = 252 ** 0.5
            assert pytest.approx(ratio, rel=0.01) == expected

    def test_known_sharpe(self):
        """Verify the Sharpe formula (mean/std * sqrt(annualization)) matches
        the same computation done externally. This is a formula-consistency
        check, not an independent validation — it guards against typos in
        the implementation."""
        returns = torch.tensor([0.02, -0.01, 0.03, -0.02, 0.01])
        mean = returns.mean().item()
        std = returns.std().item()
        expected = (mean / std) * (252 ** 0.5) if std != 0 else 0.0
        sr = BacktestEngine.sharpe_ratio(returns)
        assert sr == pytest.approx(expected, rel=1e-5)


# ── Max drawdown (static method) ──────────────────────────────────────

class TestMaxDrawdown:
    def test_no_drawdown(self):
        """Constantly rising portfolio → max drawdown = 0."""
        cumret = torch.tensor([1.0, 1.1, 1.2, 1.3, 1.4, 1.5])
        mdd = BacktestEngine.max_drawdown(cumret)
        assert mdd == pytest.approx(0.0, abs=1e-5)

    def test_full_recovery(self):
        """Drop then recover: drawdown should capture the trough."""
        cumret = torch.tensor([1.0, 0.9, 0.8, 0.9, 1.0, 1.1])
        mdd = BacktestEngine.max_drawdown(cumret)
        # Max drawdown = 0.8 / 1.0 - 1 = -0.2
        assert mdd == pytest.approx(-0.2, abs=1e-5)

    def test_monotonic_decline(self):
        """Constantly falling portfolio."""
        cumret = torch.tensor([1.0, 0.9, 0.8, 0.7, 0.6])
        mdd = BacktestEngine.max_drawdown(cumret)
        # Each step is a new low → drawdown deepens
        assert mdd == pytest.approx(-0.4, abs=1e-5)

    def test_peak_then_drop(self):
        """New peak → drawdown resets."""
        cumret = torch.tensor([1.0, 1.5, 1.2, 1.8, 1.3])
        mdd = BacktestEngine.max_drawdown(cumret)
        # Worst: 1.3 / 1.8 - 1 = -0.2778
        expected = 1.3 / 1.8 - 1.0
        assert mdd == pytest.approx(expected, abs=1e-5)

    def test_single_value(self):
        cumret = torch.tensor([1.0])
        mdd = BacktestEngine.max_drawdown(cumret)
        assert mdd == pytest.approx(0.0, abs=1e-5)

    def test_zero_start(self):
        """Should handle cumulative returns starting from a small value."""
        cumret = torch.tensor([0.01, 0.5, 1.0, 0.5])
        mdd = BacktestEngine.max_drawdown(cumret)
        assert mdd == pytest.approx(0.5 / 1.0 - 1.0, abs=1e-5)  # -0.5


# ── Info Coefficient (placeholder) ────────────────────────────────────

class TestInfoCoefficient:
    def test_raises_not_implemented(self):
        pred = torch.randn(100)
        target = torch.randn(100)
        mask = torch.ones(100, dtype=torch.bool)
        with pytest.raises(NotImplementedError):
            BacktestEngine.info_coefficient(pred, target, mask)


# ── Combined metrics sanity ───────────────────────────────────────────

class TestCombinedMetrics:
    def test_sharpe_and_mdd_consistent(self):
        """A strategy with large drawdown vs. none should differ."""
        # Smooth strategy
        smooth = torch.linspace(1.0, 1.5, 252)
        # Volatile strategy
        torch.manual_seed(42)
        volatile = torch.ones(252)
        for i in range(1, 252):
            volatile[i] = volatile[i-1] * (1 + torch.randn(1).item() * 0.02)

        sr_smooth = BacktestEngine.sharpe_ratio(smooth[1:] / smooth[:-1] - 1)
        sr_volatile = BacktestEngine.sharpe_ratio(volatile[1:] / volatile[:-1] - 1)
        mdd_smooth = BacktestEngine.max_drawdown(smooth)
        mdd_volatile = BacktestEngine.max_drawdown(volatile)

        # Smooth should have higher Sharpe and lower |MDD|
        assert sr_smooth > sr_volatile
        assert abs(mdd_smooth) < abs(mdd_volatile)
