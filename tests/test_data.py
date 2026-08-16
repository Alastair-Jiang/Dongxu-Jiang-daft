"""Test data pipeline: Panel dataclass and DataLoader."""

import numpy as np
import pytest
import torch

from daft.data.panel import Panel
from daft.data.loaders import DataLoader


# ── Fixtures ──────────────────────────────────────────────────────────

@pytest.fixture
def sample_panel():
    """Minimal 3×2×2 panel for unit tests."""
    T, N, F = 3, 2, 2
    values = torch.arange(T * N * F, dtype=torch.float32).reshape(T, N, F)
    mask = torch.ones(T, N, dtype=torch.bool)  # (T, N) — not (T, N, F)
    return Panel(
        values=values,
        mask=mask,
        dates=["2021-01-04", "2021-01-05", "2021-01-06"],
        asset_ids=["A", "B"],
        feature_names=["f0", "f1"],
    )


# ── Panel tests ───────────────────────────────────────────────────────

class TestPanel:
    def test_creation(self, sample_panel):
        assert sample_panel.values.shape == (3, 2, 2)
        assert sample_panel.mask.shape == (3, 2)           # (T, N)
        assert len(sample_panel.dates) == 3
        assert len(sample_panel.asset_ids) == 2
        assert len(sample_panel.feature_names) == 2

    def test_shape_property(self, sample_panel):
        assert sample_panel.shape == (3, 2, 2)

    def test_device_property(self, sample_panel):
        assert sample_panel.device == torch.device("cpu")

    def test_repr(self, sample_panel):
        r = repr(sample_panel)
        assert "Panel" in r
        assert "T=3" in r
        assert "N=2" in r
        assert "F=2" in r

    def test_panel_dataclass_allows_optional_fields(self):
        """Panel is a dataclass — optional fields default to None."""
        values = torch.zeros(3, 2, 2)
        mask = torch.ones(3, 2, dtype=torch.bool)
        p = Panel(values=values, mask=mask)
        assert p.values.shape == (3, 2, 2)
        assert p.dates is None
        assert p.asset_ids is None
        assert p.feature_names is None
        assert p.metadata is None

    def test_to_device(self, sample_panel):
        moved = sample_panel.to("cpu")
        assert moved.device == torch.device("cpu")
        assert moved.dates == sample_panel.dates
        assert moved.asset_ids == sample_panel.asset_ids
        assert moved.feature_names == sample_panel.feature_names

    def test_mask_defaults_true_for_synthetic(self):
        """Synthetic data should have all-True mask."""
        loader = DataLoader({"source": "synthetic", "n_stocks": 10, "n_days": 5})
        panel = loader.load()
        assert panel.mask.all()
        assert panel.mask.shape == (5, 10)  # (T, N)

    def test_slice_time(self, sample_panel):
        sliced = sample_panel.slice_time(0, 2)
        assert sliced.T == 2
        assert sliced.N == 2
        assert sliced.F == 2

    def test_train_val_test_split(self):
        values = torch.randn(100, 5, 2)
        mask = torch.ones(100, 5, dtype=torch.bool)
        panel = Panel(values=values, mask=mask)
        train, val, test = panel.train_val_test_split(0.7, 0.15)
        assert train.T == 70
        assert val.T == 15
        assert test.T == 15


# ── DataLoader tests ──────────────────────────────────────────────────

class TestDataLoader:
    def test_default_config(self):
        loader = DataLoader({})
        panel = loader.load()
        assert panel.shape[1] == 200   # default n_stocks
        assert panel.shape[2] == 5     # 5 OHLCV features

    def test_custom_n_stocks(self):
        loader = DataLoader({"source": "synthetic", "n_stocks": 50, "n_days": 10})
        panel = loader.load()
        assert panel.shape[1] == 50

    def test_feature_names(self):
        loader = DataLoader({"source": "synthetic", "n_stocks": 10, "n_days": 5})
        panel = loader.load()
        assert panel.feature_names == [
            "open", "high", "low", "close", "volume",
        ]

    def test_values_are_finite(self):
        loader = DataLoader({"source": "synthetic", "n_stocks": 50, "n_days": 20})
        panel = loader.load()
        assert panel.values.isfinite().all()

    def test_no_nans(self):
        loader = DataLoader({"source": "synthetic", "n_stocks": 50, "n_days": 20})
        panel = loader.load()
        assert not panel.values.isnan().any()

    def test_reproducibility(self):
        """Same seed should yield identical data."""
        cfg = {"source": "synthetic", "n_stocks": 10, "n_days": 5}
        torch.manual_seed(42)
        p1 = DataLoader(cfg).load()
        torch.manual_seed(42)
        p2 = DataLoader(cfg).load()
        assert torch.allclose(p1.values, p2.values)

    def test_unknown_source_raises(self):
        loader = DataLoader({"source": "bloomberg_terminal"})
        with pytest.raises(NotImplementedError, match="not supported"):
            loader.load()

    def test_baostock_source_available(self):
        """Baostock source is now wired (via adapter), not NotImplementedError."""
        loader = DataLoader({"source": "baostock"})
        # May fail at adapter level (no real data), but the dispatcher works
        try:
            loader.load()
        except ImportError:
            pytest.skip("baostock adapter not importable")

    def test_yfinance_source_available(self):
        """YFinance source dispatcher works — data may fail due to rate limits."""
        loader = DataLoader({"source": "yfinance"})
        try:
            loader.load()
        except (ImportError, RuntimeError):
            pytest.skip("yfinance adapter unavailable or rate-limited")

    def test_metadata_contains_regime_ids(self):
        loader = DataLoader({"source": "synthetic", "n_stocks": 20, "n_days": 10})
        panel = loader.load()
        assert panel.metadata is not None
        assert "regime_ids" in panel.metadata
        assert panel.metadata["regime_ids"].shape == (10,)


# ── 涨跌停 mask (2026-08-16 新增) ───────────────────────────────────────

class TestLimitMoveMask:
    """A 股涨跌停日应被标记为不可成交。"""

    def _mask(self, closes, ticker):
        from daft.data.adapters.baostock_adapter import _limit_move_mask
        return _limit_move_mask(np.array(closes, dtype=np.float32), ticker)

    def test_main_board_10pct(self):
        # 主板: 第 2 天 +10% → 涨停不可成交; 第 3 天 +5% 正常
        m = self._mask([10.0, 10.0, 11.0, 11.55, 11.55], "sh.600519")
        assert m.tolist() == [True, True, False, True, True]

    def test_gem_board_20pct(self):
        # 创业板: +10% 不触板, +20% 触板
        m = self._mask([10.0, 11.0, 13.2, 13.2], "sz.300750")
        assert m.tolist() == [True, True, False, True]

    def test_star_board_20pct(self):
        m = self._mask([10.0, 11.0, 13.2, 13.2], "sh.688001")
        assert m.tolist() == [True, True, False, True]

    def test_first_day_always_true(self):
        m = self._mask([10.0, 11.0], "sh.600000")
        assert m[0] == True  # 首日无前收盘

    def test_nan_prev_close_not_limit(self):
        m = self._mask([np.nan, 10.0, 11.0], "sh.600000")
        # prev NaN 的日子由 NaN mask 处理, 涨跌停判定保持 True
        assert m.tolist() == [True, True, False]

    def test_limit_down_also_masked(self):
        m = self._mask([10.0, 10.0, 9.0, 9.0], "sh.600000")
        assert m.tolist() == [True, True, False, True]
