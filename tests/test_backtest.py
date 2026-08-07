"""Test backtesting engine and performance metrics."""

import pytest
import torch

from daft.backtest.engine import BacktestEngine


# ── Initialization ────────────────────────────────────────────────────

class TestInit:
    def test_creates_with_config(self):
        engine = BacktestEngine({"rebalance_freq": 1})
        assert engine.rebalance_freq == 1

    def test_default_empty_config(self):
        engine = BacktestEngine({})
        assert engine.tc_bps == 2.0
        assert engine.annualization == 252


# ── run ────────────────────────────────────────────────────────────────

class TestRun:
    def test_run_executes(self):
        """run() is now fully implemented — should return metrics dict."""
        engine = BacktestEngine({})
        signals = torch.randn(100, 10)
        prices = torch.randn(100, 10).abs() + 10.0
        mask = torch.ones(100, 10, dtype=torch.bool)
        result = engine.run(signals, prices, mask)
        assert isinstance(result, dict)
        assert "sharpe_ratio" in result
        assert "max_drawdown" in result
        assert "annual_return" in result
        assert "turnover" in result


# ── Sharpe ratio (static method) ──────────────────────────────────────

class TestSharpeRatio:
    def test_positive_returns(self):
        returns = torch.tensor([0.01] * 99 + [0.011])
        sr = BacktestEngine.sharpe_ratio(returns, annualization=252)
        assert sr > 0

    def test_zero_mean_returns(self):
        returns = torch.tensor([0.01, -0.01] * 50)
        sr = BacktestEngine.sharpe_ratio(returns, annualization=252)
        assert abs(sr) < 0.5

    def test_negative_returns(self):
        returns = torch.tensor([-0.01] * 99 + [-0.011])
        sr = BacktestEngine.sharpe_ratio(returns, annualization=252)
        assert sr < 0

    def test_zero_std_returns(self):
        returns = torch.zeros(100)
        sr = BacktestEngine.sharpe_ratio(returns, annualization=252)
        assert sr == 0.0

    def test_annualization(self):
        returns = torch.tensor([0.001, -0.002, 0.003, -0.001, 0.002] * 20)
        sr_1 = BacktestEngine.sharpe_ratio(returns, annualization=1)
        sr_252 = BacktestEngine.sharpe_ratio(returns, annualization=252)
        assert sr_252 == pytest.approx(sr_1 * (252 ** 0.5), rel=1e-4)

    def test_daily_annualization_factor(self):
        torch.manual_seed(42)
        returns = torch.randn(1000) * 0.01 + 0.001
        sr_daily = BacktestEngine.sharpe_ratio(returns, annualization=1)
        sr_annual = BacktestEngine.sharpe_ratio(returns, annualization=252)
        if sr_daily != 0:
            ratio = sr_annual / sr_daily
            expected = 252 ** 0.5
            assert pytest.approx(ratio, rel=0.01) == expected

    def test_known_sharpe(self):
        returns = torch.tensor([0.02, -0.01, 0.03, -0.02, 0.01])
        mean = returns.mean().item()
        std = returns.std().item()
        expected = (mean / std) * (252 ** 0.5) if std != 0 else 0.0
        sr = BacktestEngine.sharpe_ratio(returns)
        assert sr == pytest.approx(expected, rel=1e-5)


# ── Max drawdown (static method) ──────────────────────────────────────

class TestMaxDrawdown:
    def test_no_drawdown(self):
        cumret = torch.tensor([1.0, 1.1, 1.2, 1.3, 1.4, 1.5])
        mdd = BacktestEngine.max_drawdown(cumret)
        assert mdd == pytest.approx(0.0, abs=1e-5)

    def test_full_recovery(self):
        cumret = torch.tensor([1.0, 0.9, 0.8, 0.9, 1.0, 1.1])
        mdd = BacktestEngine.max_drawdown(cumret)
        # MaxDD = 0.8 - 1.0 = -0.2 (raw diff, not percentage)
        assert mdd == pytest.approx(-0.2, abs=1e-5)

    def test_monotonic_decline(self):
        cumret = torch.tensor([1.0, 0.9, 0.8, 0.7, 0.6])
        mdd = BacktestEngine.max_drawdown(cumret)
        # Each step is a new low, worst: 0.6 - 1.0 = -0.4
        assert mdd == pytest.approx(-0.4, abs=1e-5)

    def test_peak_then_drop(self):
        cumret = torch.tensor([1.0, 1.5, 1.2, 1.8, 1.3])
        mdd = BacktestEngine.max_drawdown(cumret)
        # max_drawdown uses raw diff from peak: 1.3 - 1.8 = -0.5
        assert mdd == pytest.approx(-0.5, abs=1e-5)

    def test_single_value(self):
        cumret = torch.tensor([1.0])
        mdd = BacktestEngine.max_drawdown(cumret)
        assert mdd == pytest.approx(0.0, abs=1e-5)

    def test_zero_start(self):
        cumret = torch.tensor([0.01, 0.5, 1.0, 0.5])
        mdd = BacktestEngine.max_drawdown(cumret)
        # Worst: 0.5 - 1.0 = -0.5
        assert mdd == pytest.approx(-0.5, abs=1e-5)


# ── Info Coefficient ───────────────────────────────────────────────────

class TestInfoCoefficient:
    def test_ic_computes(self):
        """info_coefficient is now fully implemented — should return float."""
        pred = torch.randn(100)
        target = torch.randn(100)
        mask = torch.ones(100, dtype=torch.bool)
        ic = BacktestEngine.info_coefficient(pred, target, mask)
        assert isinstance(ic, float)
        assert -1.0 <= ic <= 1.0


# ── Combined metrics sanity ───────────────────────────────────────────

class TestCombinedMetrics:
    def test_sharpe_and_mdd_consistent(self):
        smooth = torch.linspace(1.0, 1.5, 252)
        torch.manual_seed(42)
        volatile = torch.ones(252)
        for i in range(1, 252):
            volatile[i] = volatile[i-1] * (1 + torch.randn(1).item() * 0.02)

        sr_smooth = BacktestEngine.sharpe_ratio(smooth[1:] / smooth[:-1] - 1)
        sr_volatile = BacktestEngine.sharpe_ratio(volatile[1:] / volatile[:-1] - 1)
        mdd_smooth = BacktestEngine.max_drawdown(smooth)
        mdd_volatile = BacktestEngine.max_drawdown(volatile)

        assert sr_smooth > sr_volatile
        assert abs(mdd_smooth) < abs(mdd_volatile)
