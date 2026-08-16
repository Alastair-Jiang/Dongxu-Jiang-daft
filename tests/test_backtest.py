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
        equity = torch.tensor([1.0, 1.1, 1.2, 1.3, 1.4, 1.5])
        mdd = BacktestEngine.max_drawdown(torch.log(equity))
        assert mdd == pytest.approx(0.0, abs=1e-5)

    def test_full_recovery(self):
        equity = torch.tensor([1.0, 0.9, 0.8, 0.9, 1.0, 1.1])
        mdd = BacktestEngine.max_drawdown(torch.log(equity))
        # 百分比口径: dd = 0.8/1.0 - 1 = -0.2
        assert mdd == pytest.approx(-0.2, abs=1e-5)

    def test_monotonic_decline(self):
        equity = torch.tensor([1.0, 0.9, 0.8, 0.7, 0.6])
        mdd = BacktestEngine.max_drawdown(torch.log(equity))
        # 百分比口径: 0.6/1.0 - 1 = -0.4
        assert mdd == pytest.approx(-0.4, abs=1e-5)

    def test_peak_then_drop(self):
        equity = torch.tensor([1.0, 1.5, 1.2, 1.8, 1.3])
        mdd = BacktestEngine.max_drawdown(torch.log(equity))
        # 百分比口径: 1.3/1.8 - 1 = -0.2778 (旧口径为 -0.5 原始差值)
        assert mdd == pytest.approx(-0.277777, abs=1e-4)

    def test_log_space_input_is_percentage(self):
        """回测调用点传入对数累计收益: dd 应为净值百分比回撤。"""
        log_cum = torch.tensor([0.0, 0.1, 0.05, 0.18, -0.1])  # 净值: 1, 1.105, 1.051, 1.197, 0.905
        mdd = BacktestEngine.max_drawdown(log_cum)
        # equity = exp(log_cum) → dd = 0.905/1.197 - 1 = -0.244
        assert mdd == pytest.approx(-0.244, abs=1e-3)

    def test_single_value(self):
        cumret = torch.tensor([1.0])
        mdd = BacktestEngine.max_drawdown(cumret)
        assert mdd == pytest.approx(0.0, abs=1e-5)

    def test_zero_start(self):
        equity = torch.tensor([0.01, 0.5, 1.0, 0.5])
        mdd = BacktestEngine.max_drawdown(torch.log(equity))
        # 百分比口径: 0.5/1.0 - 1 = -0.5
        assert mdd == pytest.approx(-0.5, abs=1e-5)


# ── 2026-08-16 回归: masked 资产不得被做空 / 换手率为真实仓位换手 ─────

class TestMaskedShortRegression:
    def test_masked_asset_never_shorted(self):
        """停牌资产(mask=False)不得进入空头(历史 bug: -(-inf)=+inf 优先做空)。"""
        engine = BacktestEngine({
            "transaction_cost_bps": 0, "slippage_bps": 0,
            "top_quantile": 0.2, "long_only": False,
        })
        sig = torch.tensor([[1.0, 0.5, 0.0, -0.5, -0.8]])
        prices = torch.full((2, 5), 10.0)
        mask = torch.tensor([[True, True, True, True, False]])
        pos = engine._signals_to_positions(sig, mask, torch.device("cpu"))
        assert pos[0, 4] == 0.0, "masked 资产拿到了仓位"
        # 真正的空头应为信号最差的有效资产(idx 3)
        assert pos[0, 3] < 0.0

    def test_turnover_is_actual_position_turnover(self):
        """报告的换手率必须等于真实仓位 L1 变化, 而不是信号代理。"""
        engine = BacktestEngine({
            "transaction_cost_bps": 0, "slippage_bps": 0,
            "top_quantile": 0.5, "long_only": True,
        })
        # 4 资产: t0 建仓 {A,B} (换手 1.0), t1 不变, t2 换到 {C,D} (换手 2.0)
        signals = torch.tensor([
            [3.0, 2.0, 1.0, 0.0],
            [3.0, 2.0, 1.0, 0.0],
            [1.0, 0.0, 3.0, 2.0],
            [1.0, 0.0, 3.0, 2.0],
        ])
        prices = torch.full((4, 4), 10.0)
        mask = torch.ones(4, 4, dtype=torch.bool)
        result = engine.run(signals, prices, mask)
        # T_ret=3, 换手序列 = [1.0, 0.0, 2.0] → 均值 1.0
        assert result["turnover"] == pytest.approx(1.0, abs=1e-5)

    def test_rebalance_freq_holds_positions(self):
        """rebalance_freq > 1 时, 持仓保持不变且仅换仓日收费。"""
        engine = BacktestEngine({
            "transaction_cost_bps": 5.0, "slippage_bps": 1.0,
            "top_quantile": 0.5, "long_only": True, "rebalance_freq": 3,
        })
        # 6 天: t0 建仓 {A,B}; t1/t2 持有; t3 换仓 {C,D}; t4 持有
        signals = torch.tensor([
            [3.0, 2.0, 1.0, 0.0],
            [1.0, 0.0, 3.0, 2.0],
            [1.0, 0.0, 3.0, 2.0],
            [1.0, 0.0, 3.0, 2.0],
            [1.0, 0.0, 3.0, 2.0],
            [1.0, 0.0, 3.0, 2.0],
        ])
        prices = torch.full((6, 4), 10.0)
        mask = torch.ones(6, 4, dtype=torch.bool)
        result = engine.run(signals, prices, mask)
        # T_ret=5, 换手序列 = [1.0, 0, 0, 2.0, 0] → 均值 0.6
        assert result["turnover"] == pytest.approx(0.6, abs=1e-5)


# ── 2026-08-16 交易制度约束: 停牌/涨跌停禁止开平仓 + 分数仓位 ─────────

class TestNoTradeOnMaskedDays:
    def test_masked_asset_keeps_position_no_cost(self):
        """停牌日(mask=False)不能平仓: 已持仓的停牌资产维持昨收、不计换手成本。

        场景: t0 建仓 {B}(换手 1.0); t1 A 停牌, 目标换成 {C} —
        B 可卖、C 可买, 换手 = 卖 B 1.0 + 买 C 1.0 = 2.0。
        若 B 停牌(见下个用例), 则 B 卖不掉。
        """
        engine = BacktestEngine({
            "transaction_cost_bps": 100, "slippage_bps": 0,
            "top_quantile": 0.5, "long_only": True,
            "respect_mask_no_trade": True,
        })
        # t0 持仓 {B}; t1 目标 {C}(B 正常可卖)
        signals = torch.tensor([
            [0.1, 3.0, 2.0],
            [3.0, 0.5, 5.0],
            [3.0, 0.5, 5.0],
        ])
        prices = torch.full((3, 3), 10.0)
        mask = torch.tensor([
            [True, True, True],
            [True, True, True],
            [True, True, True],
        ])
        result = engine.run(signals, prices, mask)
        # T_ret=2: t0 建仓 {B} 换手 1.0; t1 换仓 {C} 换手 2.0 → 均值 1.5
        assert result["turnover"] == pytest.approx(1.5, abs=1e-5)

    def test_masked_held_asset_cannot_be_sold(self):
        """已持仓资产在其停牌日卖不掉: 持仓维持昨收, 该部分不计换手。

        注: 每日调仓(freq=1)下 ret_mask 会提前一天把即将停牌的资产排除
        出目标, 该约束结构性不触发; 本用例用 rebalance_freq=2 让持仓
        跨过停牌日, 从而真正触发"停牌卖不掉"。
        """
        engine = BacktestEngine({
            "transaction_cost_bps": 100, "slippage_bps": 0,
            "top_quantile": 0.5, "long_only": True,
            "respect_mask_no_trade": True,
            "rebalance_freq": 2,
        })
        # t0 调仓建仓 {B}(换手 1.0); t1 持有; t2 调仓目标 {A}, 但 B 当日
        # 停牌卖不掉 → w=[1,1,0]: 只买 A, 换手 1.0。总换手 2.0 → 均值 2/3。
        # (无约束时 t2 换手 = 买 A + 卖 B = 2.0 → 总 3.0 → 均值 1.0。)
        signals = torch.tensor([
            [0.1, 3.0, 2.0],
            [3.0, 0.5, 5.0],
            [3.0, 0.5, 5.0],
            [3.0, 0.5, 5.0],
        ])
        prices = torch.full((4, 3), 10.0)
        mask = torch.tensor([
            [True, True, True],
            [True, True, True],
            [True, False, True],   # day2: B 停牌(跌停卖不掉)
            [True, True, True],
        ])
        result = engine.run(signals, prices, mask)
        assert result["turnover"] == pytest.approx(2.0 / 3.0, abs=1e-5)

    def test_masked_asset_never_shorted_with_no_trade(self):
        """respect_mask_no_trade 与 masked 做空修复互不冲突。"""
        engine = BacktestEngine({
            "transaction_cost_bps": 0, "slippage_bps": 0,
            "top_quantile": 0.2, "long_only": False,
            "respect_mask_no_trade": True,
        })
        sig = torch.tensor([[1.0, 0.5, 0.0, -0.5, -0.8]])
        prices = torch.full((2, 5), 10.0)
        mask = torch.tensor([[True, True, True, True, False]])
        pos = engine._signals_to_positions(sig, mask, torch.device("cpu"))
        assert pos[0, 4] == 0.0


class TestSignalZscoreWeights:
    def test_zscore_weights_positive_and_sum_one(self):
        engine = BacktestEngine({
            "top_quantile": 0.5, "long_only": True,
            "weight_mode": "signal_zscore",
        })
        signals = torch.tensor([[4.0, 3.0, 1.0, 0.0]])
        mask = torch.ones(1, 4, dtype=torch.bool)
        pos = engine._signals_to_positions(signals, mask, torch.device("cpu"))
        w = pos[0]
        selected = w[w > 0]
        assert torch.allclose(selected.sum(), torch.tensor(1.0), atol=1e-5)
        assert (w >= 0).all()
        # 信号更强的资产权重更大
        assert w[0] > w[1] > 0
        assert w[2] == 0.0 and w[3] == 0.0

    def test_zscore_long_short_sums_to_zero(self):
        engine = BacktestEngine({
            "top_quantile": 0.5, "long_only": False,
            "weight_mode": "signal_zscore",
        })
        signals = torch.tensor([[4.0, 3.0, 1.0, 0.0]])
        mask = torch.ones(1, 4, dtype=torch.bool)
        pos = engine._signals_to_positions(signals, mask, torch.device("cpu"))
        assert torch.allclose(pos[0].sum(), torch.tensor(0.0), atol=1e-5)
        assert pos[0][:2].sum() > 0.99  # 多头合计 ≈ +1
        assert pos[0][2:].sum() < -0.99  # 空头合计 ≈ −1


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
