"""Test data pipeline: Panel dataclass and DataLoader."""

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
    mask = torch.ones(T, N, F, dtype=torch.bool)
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
        assert sample_panel.mask.shape == (3, 2, 2)
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

    def test_shape_mismatch_raises(self):
        values = torch.zeros(3, 2, 2)
        mask = torch.ones(3, 4, 2)  # N doesn't match
        with pytest.raises(AssertionError):
            Panel(values, mask, [], [], [])

    def test_dates_asset_count_mismatch_raises(self):
        values = torch.zeros(3, 2, 2)
        mask = torch.ones(3, 2, 2)
        with pytest.raises(AssertionError):
            Panel(values, mask, ["d1"], ["a1", "a2"], ["f0", "f1"])

    def test_feature_count_mismatch_raises(self):
        values = torch.zeros(3, 2, 3)
        mask = torch.ones(3, 2, 3)
        with pytest.raises(AssertionError):
            Panel(
                values, mask,
                ["d1", "d2", "d3"], ["a1", "a2"],
                ["f0"],  # only 1, expected 3
            )

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


# ── DataLoader tests ──────────────────────────────────────────────────

class TestDataLoader:
    @pytest.mark.parametrize("freq,expected_bars", [
        ("1min", 240),
        ("5min", 48),
        ("15min", 16),
        ("30min", 8),
        ("1h", 4),
        ("1d", 1),
    ])
    def test_frequencies(self, freq, expected_bars):
        n_days = 10
        loader = DataLoader({
            "source": "synthetic",
            "n_stocks": 5,
            "n_days": n_days,
            "frequency": freq,
        })
        panel = loader.load()
        expected_T = n_days * expected_bars
        assert panel.shape[0] == expected_T

    def test_default_config(self):
        loader = DataLoader({})
        panel = loader.load()
        assert panel.shape[1] == 200   # default n_stocks
        assert panel.shape[2] == 5     # 5 synthetic features

    def test_custom_n_stocks(self):
        loader = DataLoader({"source": "synthetic", "n_stocks": 50, "n_days": 10})
        panel = loader.load()
        assert panel.shape[1] == 50

    def test_feature_names(self):
        loader = DataLoader({"source": "synthetic", "n_stocks": 10, "n_days": 5})
        panel = loader.load()
        assert panel.feature_names == [
            "close", "log_return", "volume", "volume_ratio", "volatility_20",
        ]

    def test_asset_id_format(self):
        loader = DataLoader({"source": "synthetic", "n_stocks": 8, "n_days": 5})
        panel = loader.load()
        assert all(aid.startswith("SYNTH_") for aid in panel.asset_ids)

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
        with pytest.raises(ValueError, match="Unknown data source"):
            loader.load()

    def test_baostock_not_implemented(self):
        loader = DataLoader({"source": "baostock"})
        with pytest.raises(NotImplementedError):
            loader.load()

    def test_yfinance_not_implemented(self):
        loader = DataLoader({"source": "yfinance"})
        with pytest.raises(NotImplementedError):
            loader.load()

    def test_custom_not_implemented(self):
        loader = DataLoader({"source": "custom"})
        with pytest.raises(NotImplementedError):
            loader.load()
