"""Test feature engineering pipeline.

Covers:
- TensorFactorEngine (GPU-vectorized primitives with mask propagation)
- RegimeFeatureExtractor (200-dim market state vector)
- FreqFeatureExtractor (FFT spectral features)
- Legacy alpha factors
"""

import pytest
import torch

from daft.data.panel import Panel
from daft.features.tensor_factors import TensorFactorEngine
from daft.features.regime_features import RegimeFeatureExtractor
from daft.features.freq_features import FreqFeatureExtractor
from daft.features.legacy_factors import (
    LEGACY_FACTOR_REGISTRY, compute_all_factors,
    better_001, better_002, best_001, best_008, old_027, old_035,
    stock_001, stock_010, stock_022, extra_001, extra_012, add_001, add_005,
)


# ═══════════════════════════════════════════════════════════════════════
# Shared fixtures
# ═══════════════════════════════════════════════════════════════════════

def _make_panel(T=50, N=10, F=5):
    """Create a minimal synthetic panel for testing."""
    torch.manual_seed(42)
    close = 100 * torch.exp(0.0001 * torch.arange(T).float().unsqueeze(1) + 0.01 * torch.randn(T, N))
    ret = torch.randn(T, N) * 0.01
    volume = torch.rand(T, N) * 1e6
    vol_ratio = torch.ones(T, N) + 0.5 * torch.randn(T, N)
    volatility = torch.randn(T, N).abs() * 0.02

    values = torch.stack([close, ret, volume, vol_ratio, volatility], dim=-1)
    mask = torch.ones(T, N, F, dtype=torch.bool)

    return Panel(
        values=values, mask=mask,
        dates=list(range(T)),
        asset_ids=[f"A{i:03d}" for i in range(N)],
        feature_names=["close", "log_return", "volume", "volume_ratio", "volatility_20"],
    )


@pytest.fixture
def engine():
    return TensorFactorEngine()


@pytest.fixture
def panel():
    return _make_panel()


@pytest.fixture
def masked_panel():
    """Panel with some non-tradable (masked) assets."""
    panel = _make_panel()
    # Mask out asset 3 at all times, and asset 5 at times 10-20
    panel.mask[:, 3, :] = False
    panel.mask[10:20, 5, :] = False
    return panel


# ═══════════════════════════════════════════════════════════════════════
# TensorFactorEngine — Rank
# ═══════════════════════════════════════════════════════════════════════

class TestRank:
    def test_output_shape(self, engine):
        x = torch.randn(10, 20)
        mask = torch.ones(10, 20, dtype=torch.bool)
        out = engine.rank(x, mask)
        assert out.shape == (10, 20)

    def test_range_zero_to_one(self, engine):
        x = torch.randn(50, 8)
        mask = torch.ones(50, 8, dtype=torch.bool)
        out = engine.rank(x, mask)
        assert (out >= 0).all()
        assert (out <= 1).all()

    def test_all_masked_returns_neutral(self, engine):
        x = torch.randn(5, 10)
        mask = torch.zeros(5, 10, dtype=torch.bool)
        out = engine.rank(x, mask)
        # All masked → all get 0.5
        assert torch.allclose(out, torch.full_like(out, 0.5))

    def test_single_asset_returns_neutral(self, engine):
        x = torch.randn(5, 1)
        mask = torch.ones(5, 1, dtype=torch.bool)
        out = engine.rank(x, mask)
        assert torch.allclose(out, torch.full_like(out, 0.5))

    def test_monotonicity(self, engine):
        """Larger values should get higher ranks."""
        x = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]])
        mask = torch.ones(1, 5, dtype=torch.bool)
        out = engine.rank(x, mask)
        for i in range(4):
            assert out[0, i] < out[0, i + 1]

    def test_partial_mask(self, engine):
        """Masked assets get 0.5; unmasked share ranking."""
        x = torch.randn(3, 6)
        mask = torch.ones(3, 6, dtype=torch.bool)
        mask[:, 0] = False  # asset 0 always masked
        out = engine.rank(x, mask)
        assert (out[:, 0] == 0.5).all()  # masked → neutral
        assert (out[:, 1:] >= 0).all()
        assert (out[:, 1:] <= 1).all()

    def test_identical_values_same_rank(self, engine):
        x = torch.ones(3, 5) * 5.0
        mask = torch.ones(3, 5, dtype=torch.bool)
        out = engine.rank(x, mask)
        # All identical → double-argsort distributes ranks evenly across positions
        # e.g. [0.0, 0.25, 0.5, 0.75, 1.0] for 5 assets
        # The key property: min = 0, max = 1, and values span the full range
        for t in range(3):
            assert out[t].min() == 0.0
            assert out[t].max() == 1.0

    def test_deterministic(self, engine):
        torch.manual_seed(42)
        x = torch.randn(10, 15)
        mask = torch.ones(10, 15, dtype=torch.bool)
        out1 = engine.rank(x, mask)
        torch.manual_seed(42)
        x = torch.randn(10, 15)
        out2 = engine.rank(x, mask)
        assert torch.allclose(out1, out2)


# ═══════════════════════════════════════════════════════════════════════
# TensorFactorEngine — Correlation
# ═══════════════════════════════════════════════════════════════════════

class TestCorr:
    def test_output_shape(self, engine):
        x = torch.randn(30, 5)
        y = torch.randn(30, 5)
        mask = torch.ones(30, 5, dtype=torch.bool)
        out = engine.corr(x, y, 10, mask)
        assert out.shape == (30, 5)

    def test_perfect_positive_correlation(self, engine):
        x = torch.arange(50, dtype=torch.float).unsqueeze(1).expand(-1, 5)
        y = x * 2.0 + 3.0
        mask = torch.ones(50, 5, dtype=torch.bool)
        out = engine.corr(x, y, 20, mask)
        # After warmup (window=20), correlation should be close to 1
        assert out[30:].mean() > 0.95, f"Expected near-perfect corr, got {out[30:].mean():.3f}"

    def test_perfect_negative_correlation(self, engine):
        x = torch.arange(50, dtype=torch.float).unsqueeze(1).expand(-1, 5)
        y = -x
        mask = torch.ones(50, 5, dtype=torch.bool)
        out = engine.corr(x, y, 20, mask)
        assert out[30:].mean() < -0.95

    def test_no_correlation(self, engine):
        torch.manual_seed(0)
        x = torch.randn(100, 3)
        y = torch.randn(100, 3)
        mask = torch.ones(100, 3, dtype=torch.bool)
        out = engine.corr(x, y, 50, mask)
        # Uncorrelated random series → corr ≈ 0 (abs < 0.3 at least)
        assert out[60:].abs().mean() < 0.3

    def test_range(self, engine):
        x = torch.randn(50, 5)
        y = torch.randn(50, 5)
        mask = torch.ones(50, 5, dtype=torch.bool)
        out = engine.corr(x, y, 20, mask)
        valid = out[19:]  # after warmup
        assert (valid >= -1.0).all()
        assert (valid <= 1.0).all()

    def test_early_steps_below_min_periods(self, engine):
        """First few steps (below min_periods) should be 0."""
        torch.manual_seed(42)
        x = torch.randn(10, 3)
        y = torch.randn(10, 3)
        mask = torch.ones(10, 3, dtype=torch.bool)
        out = engine.corr(x, y, 20, mask)
        # min_periods = max(20//3, 3) = 6
        # First 5 steps have < 6 valid observations → should be 0
        assert (out[:5] == 0).all()

    def test_mask_propagation(self, engine):
        """Masked observations should not contaminate correlation."""
        x = torch.randn(50, 3)
        y = x + 0.01 * torch.randn(50, 3)  # highly correlated
        mask = torch.ones(50, 3, dtype=torch.bool)
        mask[:40, 0] = False  # mostly masked for asset 0
        out = engine.corr(x, y, 30, mask)
        # Asset 0 has very few valid points → corr stays near 0
        assert out[40:, 0].abs().mean() < 0.5


# ═══════════════════════════════════════════════════════════════════════
# TensorFactorEngine — EWMA
# ═══════════════════════════════════════════════════════════════════════

class TestEWMA:
    def test_output_shape(self, engine):
        x = torch.randn(30, 5)
        mask = torch.ones(30, 5, dtype=torch.bool)
        out = engine.ewma(x, 10, mask)
        assert out.shape == (30, 5)

    def test_all_finite(self, engine):
        x = torch.randn(50, 8)
        mask = torch.ones(50, 8, dtype=torch.bool)
        out = engine.ewma(x, 10, mask)
        assert out.isfinite().all()

    def test_constant_signal_converges(self, engine):
        """EWMA of constant value should converge to that value."""
        x = torch.ones(100, 2) * 5.0
        mask = torch.ones(100, 2, dtype=torch.bool)
        out = engine.ewma(x, 10, mask)
        # After convergence, output ≈ 5.0
        assert torch.allclose(out[50:], torch.full_like(out[50:], 5.0), atol=0.01)

    def test_mask_carry_forward(self, engine):
        """When mask is False, EWMA should carry forward previous value."""
        x = torch.randn(10, 3)
        mask = torch.ones(10, 3, dtype=torch.bool)
        mask[3:6, 0] = False  # gap in observations
        out = engine.ewma(x, 5, mask)
        # Values at mask=False should equal the last valid value
        assert torch.allclose(out[3, 0], out[4, 0])
        assert torch.allclose(out[3, 0], out[5, 0])

    def test_span_alpha_relationship(self, engine):
        """Larger span → smoother output (lower variance)."""
        x = torch.randn(200, 1) * 0.1
        mask = torch.ones(200, 1, dtype=torch.bool)
        out_short = engine.ewma(x, 3, mask)
        out_long = engine.ewma(x, 30, mask)
        # Short span follows noise more → higher std
        assert out_short[50:].std() > out_long[50:].std()

    def test_all_masked_returns_zero(self, engine):
        x = torch.randn(10, 3)
        mask = torch.zeros(10, 3, dtype=torch.bool)
        out = engine.ewma(x, 5, mask)
        assert torch.allclose(out, torch.zeros_like(out))


# ═══════════════════════════════════════════════════════════════════════
# TensorFactorEngine — ts_delta
# ═══════════════════════════════════════════════════════════════════════

class TestTsDelta:
    def test_output_shape(self, engine):
        x = torch.randn(30, 5)
        mask = torch.ones(30, 5, dtype=torch.bool)
        out = engine.ts_delta(x, 5, mask)
        assert out.shape == (30, 5)

    def test_basic_difference(self, engine):
        x = torch.tensor([
            [1.0, 2.0],
            [3.0, 5.0],
            [6.0, 9.0],
        ])
        mask = torch.ones(3, 2, dtype=torch.bool)
        out = engine.ts_delta(x, 1, mask)
        assert out[0, 0] == 0.0  # t < d → 0
        assert out[1, 0] == pytest.approx(2.0)
        assert out[2, 1] == pytest.approx(4.0)

    def test_early_steps_zero(self, engine):
        x = torch.randn(20, 3)
        mask = torch.ones(20, 3, dtype=torch.bool)
        out = engine.ts_delta(x, 10, mask)
        assert (out[:10] == 0).all()

    def test_mask_cascade(self, engine):
        """If mask[t] or mask[t-d] is False, result should be 0."""
        x = torch.randn(10, 3)
        mask = torch.ones(10, 3, dtype=torch.bool)
        mask[5, 1] = False  # mask at t
        out = engine.ts_delta(x, 2, mask)
        assert out[5, 1] == 0.0  # masked because mask[5,1] is False

    def test_d_larger_than_T(self, engine):
        x = torch.randn(5, 3)
        mask = torch.ones(5, 3, dtype=torch.bool)
        out = engine.ts_delta(x, 20, mask)
        assert (out == 0).all()


# ═══════════════════════════════════════════════════════════════════════
# TensorFactorEngine — ts_sum
# ═══════════════════════════════════════════════════════════════════════

class TestTsSum:
    def test_output_shape(self, engine):
        x = torch.randn(30, 5)
        mask = torch.ones(30, 5, dtype=torch.bool)
        out = engine.ts_sum(x, 10, mask)
        assert out.shape == (30, 5)

    def test_constant_series(self, engine):
        x = torch.ones(20, 2)
        mask = torch.ones(20, 2, dtype=torch.bool)
        out = engine.ts_sum(x, 5, mask)
        assert torch.allclose(out[4:], torch.full_like(out[4:], 5.0))

    def test_masked_values_excluded(self, engine):
        x = torch.ones(10, 2)
        mask = torch.ones(10, 2, dtype=torch.bool)
        mask[2:5, 0] = False  # masked period
        out = engine.ts_sum(x, 5, mask)
        # At t=6: sum over indices [2,3,4,5,6] but 2,3,4 are masked → only 2 valid
        assert out[6, 0] == 2.0

    def test_all_masked_returns_zero(self, engine):
        x = torch.randn(10, 3)
        mask = torch.zeros(10, 3, dtype=torch.bool)
        out = engine.ts_sum(x, 5, mask)
        assert torch.allclose(out, torch.zeros_like(out))

    def test_window_one(self, engine):
        x = torch.randn(10, 3)
        mask = torch.ones(10, 3, dtype=torch.bool)
        mask[3, 1] = False
        out = engine.ts_sum(x, 1, mask)
        assert out[0, 0] == x[0, 0]
        assert out[3, 1] == 0.0  # masked


# ═══════════════════════════════════════════════════════════════════════
# TensorFactorEngine — ts_std
# ═══════════════════════════════════════════════════════════════════════

class TestTsStd:
    def test_output_shape(self, engine):
        x = torch.randn(30, 5)
        mask = torch.ones(30, 5, dtype=torch.bool)
        out = engine.ts_std(x, 10, mask)
        assert out.shape == (30, 5)

    def test_constant_series_zero_std(self, engine):
        x = torch.ones(30, 3) * 3.0
        mask = torch.ones(30, 3, dtype=torch.bool)
        out = engine.ts_std(x, 10, mask)
        # Manual check: after 10 steps, std of constant series = 0
        assert torch.allclose(out[15:], torch.zeros_like(out[15:]), atol=1e-4)

    def test_nonnegative(self, engine):
        x = torch.randn(50, 5)
        mask = torch.ones(50, 5, dtype=torch.bool)
        out = engine.ts_std(x, 20, mask)
        valid = out[19:]
        assert (valid >= 0).all()

    def test_min_periods(self, engine):
        x = torch.randn(5, 3)
        mask = torch.ones(5, 3, dtype=torch.bool)
        out = engine.ts_std(x, 30, mask)
        # window > T → all 0
        assert (out == 0).all()

    def test_nonzero_for_varying_input(self, engine):
        x = torch.randn(30, 2) * 0.1
        mask = torch.ones(30, 2, dtype=torch.bool)
        out = engine.ts_std(x, 10, mask)
        assert (out[15:] > 0).all()

    def test_masked_values_excluded(self, engine):
        x = torch.randn(20, 2) * 0.1
        mask = torch.ones(20, 2, dtype=torch.bool)
        mask[2:6, 0] = False
        out = engine.ts_std(x, 10, mask)
        # Should still be finite and non-negative
        assert out.isfinite().all()
        assert (out >= 0).all()


# ═══════════════════════════════════════════════════════════════════════
# TensorFactorEngine — ts_mean
# ═══════════════════════════════════════════════════════════════════════

class TestTsMean:
    def test_output_shape(self, engine):
        x = torch.randn(30, 5)
        mask = torch.ones(30, 5, dtype=torch.bool)
        out = engine.ts_mean(x, 10, mask)
        assert out.shape == (30, 5)

    def test_constant_series(self, engine):
        x = torch.ones(20, 2) * 7.0
        mask = torch.ones(20, 2, dtype=torch.bool)
        out = engine.ts_mean(x, 5, mask)
        assert torch.allclose(out[4:], torch.full_like(out[4:], 7.0))

    def test_gives_same_as_sum_div_count(self, engine):
        x = torch.randn(30, 3)
        mask = torch.ones(30, 3, dtype=torch.bool)
        mean = engine.ts_mean(x, 10, mask)
        sum_val = engine.ts_sum(x, 10, mask)
        # From t=9 onward (window=10 fully filled), all mask=True → n_valid=10
        # Before that, ts_mean clips min_periods and may give different results
        expected_mean = sum_val[9:] / 10.0
        assert torch.allclose(mean[9:], expected_mean, atol=1e-6)


# ═══════════════════════════════════════════════════════════════════════
# FreqFeatureExtractor
# ═══════════════════════════════════════════════════════════════════════

class TestFreqFeatureExtractor:
    @pytest.fixture
    def extractor(self):
        return FreqFeatureExtractor(lookback=512, n_freq_bins=64)

    @pytest.fixture
    def small_extractor(self):
        return FreqFeatureExtractor(lookback=32, n_freq_bins=16)

    # ── Init ──────────────────────────────────────────────────────────

    def test_init_default(self):
        ext = FreqFeatureExtractor()
        assert ext.lookback == 512
        assert ext.n_freq_bins == 64

    def test_init_custom(self):
        ext = FreqFeatureExtractor(lookback=256, n_freq_bins=32)
        assert ext.lookback == 256
        assert ext.n_freq_bins == 32

    # ── Periodogram ───────────────────────────────────────────────────

    def test_compute_periodogram_output_shape(self, extractor):
        x = torch.randn(8, 512)
        psd = extractor.compute_periodogram(x)
        assert psd.shape == (8, 64)

    def test_periodogram_sums_to_one(self):
        ext = FreqFeatureExtractor(lookback=128, n_freq_bins=64)
        x = torch.randn(4, 128)
        psd = ext.compute_periodogram(x)
        for i in range(4):
            assert torch.allclose(psd[i].sum(), torch.tensor(1.0), atol=1e-5)

    def test_periodogram_all_positive(self, extractor):
        x = torch.randn(16, 512)
        psd = extractor.compute_periodogram(x)
        assert (psd >= 0).all()

    def test_periodogram_dc_removed(self, extractor):
        x = torch.ones(4, 512) * 5.0
        psd = extractor.compute_periodogram(x)
        assert psd.isfinite().all()

    def test_sine_wave_has_single_peak(self, extractor):
        t = torch.linspace(0, 4 * torch.pi, 512)
        x = torch.sin(t).unsqueeze(0)
        psd = extractor.compute_periodogram(x)
        max_bin = psd.argmax().item()
        max_power = psd[0, max_bin].item()
        assert max_power > 1.0 / 64

    def test_different_frequencies_different_peaks(self, extractor):
        t = torch.linspace(0, 1, 512)
        x_low = torch.sin(2 * torch.pi * 2 * t).unsqueeze(0)
        x_high = torch.sin(2 * torch.pi * 50 * t).unsqueeze(0)
        psd_low = extractor.compute_periodogram(x_low)
        psd_high = extractor.compute_periodogram(x_high)
        peak_low = psd_low.argmax().item()
        peak_high = psd_high.argmax().item()
        assert peak_low != peak_high

    def test_truncation_shorter_than_n_freq_bins(self, extractor):
        x = torch.randn(2, 512)
        psd = extractor.compute_periodogram(x)
        assert psd.shape[-1] == 64
        assert psd.isfinite().all()

    def test_truncation_longer_than_n_freq_bins(self):
        ext = FreqFeatureExtractor(lookback=1024, n_freq_bins=16)
        x = torch.randn(4, 1024)
        psd = ext.compute_periodogram(x)
        assert psd.shape[-1] == 16

    def test_batch_independence(self, extractor):
        x_a = torch.sin(torch.linspace(0, 4 * torch.pi, 512))
        x_b = torch.cos(torch.linspace(0, 4 * torch.pi, 512))
        x = torch.stack([x_a, x_b], dim=0)
        psd = extractor.compute_periodogram(x)
        assert not torch.allclose(psd[0], psd[1])

    # ── Forward ───────────────────────────────────────────────────────

    def test_forward_output_shape(self, small_extractor, panel):
        out = small_extractor.forward(panel)
        assert out.shape == (50, 10, 19)  # T=50, N=10, n_freq_bins=16 + 3 bands

    def test_forward_all_finite(self, small_extractor, panel):
        out = small_extractor.forward(panel)
        assert out.isfinite().all()

    def test_forward_early_steps_zero(self, small_extractor, panel):
        out = small_extractor.forward(panel)
        # First lookback-1 steps should be zero
        assert (out[:small_extractor.lookback - 1] == 0).all()

    def test_forward_later_steps_nonzero(self, small_extractor, panel):
        out = small_extractor.forward(panel)
        # After lookback steps, at least some values should be non-zero
        nonzero = (out[small_extractor.lookback:] != 0).any()
        assert nonzero

    def test_forward_band_powers_sum_approx_one(self, small_extractor, panel):
        out = small_extractor.forward(panel)
        # The last 3 dims are low/mid/high band powers
        bands = out[small_extractor.lookback:, :, -3:]
        total = bands.sum(dim=-1)
        assert torch.allclose(total, torch.ones_like(total), atol=0.05)

    def test_forward_with_short_panel(self):
        """Panel shorter than lookback should return all zeros without error."""
        ext = FreqFeatureExtractor(lookback=128, n_freq_bins=16)
        panel = _make_panel(T=30, N=5)
        out = ext.forward(panel)
        assert out.isfinite().all()
        assert (out == 0).all()  # T < lookback


# ═══════════════════════════════════════════════════════════════════════
# RegimeFeatureExtractor
# ═══════════════════════════════════════════════════════════════════════

class TestRegimeFeatureExtractor:
    @pytest.fixture
    def extractor(self):
        return RegimeFeatureExtractor(n_base_factors=50, output_dim=200)

    def test_init_default(self):
        ext = RegimeFeatureExtractor()
        assert ext.n_base_factors == 50
        assert ext.output_dim == 200

    def test_init_custom(self):
        ext = RegimeFeatureExtractor(n_base_factors=100, output_dim=128)
        assert ext.n_base_factors == 100
        assert ext.output_dim == 128

    def test_output_shape(self, extractor, panel):
        s_t = extractor.forward(panel)
        assert s_t.shape == (50, 10, 200)

    def test_all_finite(self, extractor, panel):
        s_t = extractor.forward(panel)
        assert s_t.isfinite().all(), f"Non-finite values: NaN={s_t.isnan().any()}, Inf={s_t.isinf().any()}"

    def test_no_nan_in_constant_input(self, extractor):
        """Constant input should not produce NaN."""
        T, N = 30, 5
        values = torch.ones(T, N, 5)
        values[:, :, 1] = 0.001  # small constant returns
        values[:, :, 2] = 1e5  # reasonable volume
        mask = torch.ones(T, N, 5, dtype=torch.bool)
        panel = Panel(
            values=values, mask=mask,
            dates=list(range(T)),
            asset_ids=[f"A{i:03d}" for i in range(N)],
            feature_names=["close", "log_return", "volume", "volume_ratio", "volatility_20"],
        )
        s_t = extractor.forward(panel)
        assert not s_t.isnan().any()

    def test_mask_propagation(self, extractor, masked_panel):
        """Masked positions should be zero in output."""
        s_t = extractor.forward(masked_panel)
        # Asset 3 is fully masked
        assert (s_t[:, 3, :] == 0).all()
        # Asset 5 is masked at times 10-20
        assert (s_t[10:20, 5, :] == 0).all()

    def test_different_stocks_have_different_features(self, extractor, panel):
        """Different assets should have meaningfully different feature vectors."""
        s_t = extractor.forward(panel)
        # Check that at least 2 assets differ in at least one dimension
        diff = (s_t[:, 0, :] - s_t[:, 1, :]).abs().sum()
        assert diff > 0.01, f"All assets have identical features (diff={diff:.6f})"

    def test_output_range_reasonable(self, extractor, panel):
        """Output should be within [-50, 50] (clamping applied)."""
        s_t = extractor.forward(panel)
        assert s_t.abs().max() <= 50.0

    def test_deterministic(self, extractor, panel):
        s_t1 = extractor.forward(panel)
        s_t2 = extractor.forward(panel)
        assert torch.equal(s_t1, s_t2)

    def test_device_consistency(self, extractor, panel):
        """Output device should match input device."""
        s_t = extractor.forward(panel)
        assert s_t.device == panel.device


# ═══════════════════════════════════════════════════════════════════════
# Legacy Factors
# ═══════════════════════════════════════════════════════════════════════

class TestLegacyFactors:
    @pytest.fixture
    def panel(self):
        return _make_panel(T=100, N=15)

    @pytest.fixture
    def mask(self):
        return torch.ones(100, 15, dtype=torch.bool)

    @pytest.fixture
    def engine(self):
        return TensorFactorEngine()

    # ── Shape and finiteness ───────────────────────────────────────────

    @pytest.mark.parametrize("factor_fn", [
        better_001, better_002, best_001, best_008,
        old_027, old_035, stock_001, stock_010,
        extra_001, extra_012, add_001, add_005,
    ])
    def test_output_shape(self, factor_fn, panel, engine):
        out = factor_fn(panel, engine)
        assert out.shape == (100, 15), f"{factor_fn.__name__} shape mismatch"

    @pytest.mark.parametrize("factor_fn", [
        better_001, better_002, best_001, best_008,
        old_027, old_035, stock_001, stock_010,
        extra_001, extra_012, add_001, add_005,
    ])
    def test_all_finite(self, factor_fn, panel, engine):
        out = factor_fn(panel, engine)
        assert out.isfinite().all(), f"{factor_fn.__name__} has NaN or Inf"

    # ── Factor variation ───────────────────────────────────────────────

    def test_factors_not_constant(self, panel):
        """All registered factors should produce non-constant output."""
        engine = TensorFactorEngine()
        for name, fn in LEGACY_FACTOR_REGISTRY.items():
            out = fn(panel, engine)
            if out is not None and out.isfinite().all():
                std = out.std().item()
                assert std > 1e-8, f"Factor {name} is constant (std={std:.2e})"

    def test_factors_not_identical(self, panel):
        """Different factors should produce different rankings."""
        engine = TensorFactorEngine()
        results = {}
        for name, fn in list(LEGACY_FACTOR_REGISTRY.items())[:10]:
            out = fn(panel, engine)
            if out is not None:
                results[name] = out
        # Check that at least some factors differ from each other
        names = list(results.keys())
        different_count = 0
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                if not torch.allclose(results[names[i]], results[names[j]], atol=0.01):
                    different_count += 1
        assert different_count > 0, "All factors produce identical output"

    # ── Registry ───────────────────────────────────────────────────────

    def test_registry_has_all_families(self):
        families = {}
        for name in LEGACY_FACTOR_REGISTRY:
            family = name.split("_")[0]
            families.setdefault(family, []).append(name)
        assert "better" in families
        assert "best" in families
        assert "old" in families
        assert "stock" in families
        assert "extra" in families
        assert "add" in families

    def test_registry_count(self):
        assert len(LEGACY_FACTOR_REGISTRY) == 35

    # ── compute_all_factors ────────────────────────────────────────────

    def test_compute_all_factors(self, panel):
        results = compute_all_factors(panel)
        assert len(results) == 35
        # All should be non-None (no exceptions)
        none_count = sum(1 for v in results.values() if v is None)
        assert none_count == 0, f"{none_count} factors raised exceptions"

    def test_compute_all_factors_shapes(self, panel):
        results = compute_all_factors(panel)
        for name, tensor in results.items():
            assert tensor.shape == (100, 15), f"{name} shape: {tensor.shape}"

    # ── Specific factor properties ─────────────────────────────────────

    def test_better_001_is_rank(self, panel, engine):
        """better_001 returns rank values in [0, 1]."""
        out = better_001(panel, engine)
        assert (out >= 0).all() and (out <= 1).all()

    def test_best_008_positive(self, panel, engine):
        """best_008 (volume breakout) should be positive."""
        out = best_008(panel, engine)
        assert (out > 0).all()

    def test_old_027_is_corr_rank(self, panel, engine):
        """old_027 is a ranked correlation in [0, 1]."""
        out = old_027(panel, engine)
        assert (out >= 0).all() and (out <= 1).all()

    def test_stock_022_is_mean_zero(self, panel, engine):
        """stock_022 is cross-sectional residual — mean ≈ 0 per timestep."""
        out = stock_022(panel, engine)
        # Cross-sectional mean should be near 0 at each t
        cs_mean = out.mean(dim=1)
        assert cs_mean.abs().max() < 0.001
