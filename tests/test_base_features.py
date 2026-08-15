"""通道契约测试 (2026-08-16): 数据源 OHLCV → 特征引擎基础布局。

历史上数据源的 [open, high, low, close, volume] 被特征引擎直接当作
[close, log_return, volume, volume_ratio, volatility_20] 读取, 所有实验的
s_t 都建立在错列上且测试从未覆盖真实布局。这些测试保证该事故不再发生。
"""

import math

import pytest
import torch

from daft.data.panel import Panel
from daft.data.loaders import SyntheticDataGenerator
from daft.features.base_features import (
    BASE_FEATURE_NAMES,
    OHLCV_FEATURE_NAMES,
    ensure_base_panel,
    ohlcv_to_base_panel,
)
from daft.features.regime_features import RegimeFeatureExtractor
from daft.features.freq_features import FreqFeatureExtractor
from daft.features.legacy_factors import compute_all_factors


def _make_ohlcv_panel(T=30, N=4):
    """确定性 OHLCV panel: close 每天 ×2 (log_return = log 2), 其余列可控。"""
    close = torch.full((T, N), 1.0)
    close *= 2.0 ** torch.arange(T).float().unsqueeze(1)  # 1, 2, 4, 8, ...
    open_ = close.clone()
    high = close * 1.01
    low = close * 0.99
    volume = torch.full((T, N), 1e6)
    values = torch.stack([open_, high, low, close, volume], dim=-1)
    mask = torch.ones(T, N, dtype=torch.bool)
    return Panel(values=values, mask=mask, dates=list(range(T)),
                 asset_ids=[f"A{i}" for i in range(N)],
                 feature_names=list(OHLCV_FEATURE_NAMES))


class TestOhlcvToBase:
    def test_close_passthrough(self):
        panel = _make_ohlcv_panel()
        base = ohlcv_to_base_panel(panel)
        assert base.feature_names == BASE_FEATURE_NAMES
        assert torch.allclose(base.values[:, :, 0], panel.values[:, :, 3])

    def test_log_return_is_close_diff(self):
        """log_return 必须等于 close 的对数差分, 而不是 high 之类。"""
        panel = _make_ohlcv_panel()
        base = ohlcv_to_base_panel(panel)
        lr = base.values[:, :, 1]
        assert lr[0].abs().sum() == 0.0  # 首日无前值
        assert torch.allclose(lr[1:], torch.full_like(lr[1:], math.log(2.0)),
                              atol=1e-6)

    def test_volume_ratio_neutral_when_flat_volume(self):
        panel = _make_ohlcv_panel()
        base = ohlcv_to_base_panel(panel)
        # 成交量恒定 → 20 日窗口后 ratio 应为 1
        vr = base.values[:, :, 3]
        assert torch.allclose(vr[20:], torch.ones_like(vr[20:]), atol=1e-4)

    def test_volatility_zero_when_deterministic(self):
        panel = _make_ohlcv_panel()
        base = ohlcv_to_base_panel(panel)
        vol = base.values[:, :, 4]
        assert vol[20:].abs().max() < 1e-5  # 确定性收益 → 波动率≈0

    def test_mask_respected_for_returns(self):
        panel = _make_ohlcv_panel()
        panel.mask[5, 0] = False  # 第 5 日停牌 → 5 与 6 的收益均不可用
        base = ohlcv_to_base_panel(panel)
        lr = base.values[:, :, 1]
        assert lr[5, 0] == 0.0 and lr[6, 0] == 0.0
        assert lr[5, 1] != 0.0  # 其他股票不受影响


class TestEnsureBasePanel:
    def test_base_layout_passthrough(self):
        panel = _make_ohlcv_panel()
        base = ohlcv_to_base_panel(panel)
        out = ensure_base_panel(base)
        assert out is base  # 已符合契约则原样返回

    def test_unknown_layout_raises(self):
        panel = _make_ohlcv_panel()
        panel.feature_names = ["f0", "f1", "f2", "f3", "f4"]
        with pytest.raises(ValueError, match="无法识别"):
            ensure_base_panel(panel)

    def test_missing_names_raises(self):
        panel = _make_ohlcv_panel()
        panel.feature_names = None
        with pytest.raises(ValueError, match="未设置"):
            ensure_base_panel(panel)


class TestEndToEndChannel:
    def test_extractor_on_real_ohlcv_layout(self):
        """真实数据路径: SyntheticDataGenerator 产出 OHLCV → extractor。"""
        panel = SyntheticDataGenerator(n_stocks=8, n_days=60, seed=7).generate()
        assert panel.feature_names == OHLCV_FEATURE_NAMES
        ext = RegimeFeatureExtractor(n_base_factors=50, output_dim=200)
        s_t = ext(panel)
        assert s_t.shape == (60, 8, 200)
        assert torch.isfinite(s_t).all()

    def test_extractor_return_dims_reflect_close_not_high(self):
        """close 翻倍时, 收益族特征应为常数; 修复前它们随 high/low 漂移。"""
        panel = _make_ohlcv_panel(T=40, N=2)
        ext = RegimeFeatureExtractor(n_base_factors=50, output_dim=200)
        s_t = ext(panel)
        # Group 1 的 1 日收益特征 (dim 0) 在 warm-up 后应为 log 2 常数
        d0 = s_t[:, :, 0]
        assert torch.allclose(d0[5:], torch.full_like(d0[5:], math.log(2.0)),
                              atol=1e-3)

    def test_freq_extractor_accepts_ohlcv(self):
        panel = _make_ohlcv_panel(T=40, N=2)
        fe = FreqFeatureExtractor(lookback=16, n_freq_bins=8)
        out = fe(panel)
        assert out.shape == (40, 2, 11)
        assert torch.isfinite(out).all()

    def test_legacy_factors_accept_real_2d_mask_ohlcv(self):
        """compute_all_factors 在真实 2D mask 的 OHLCV panel 上不应崩溃。"""
        panel = SyntheticDataGenerator(n_stocks=6, n_days=40, seed=3).generate()
        panel.mask[10, 0] = False  # 2D mask
        factors = compute_all_factors(panel)
        errors = factors.get("_errors", [])
        n_ok = sum(1 for v in factors.values() if v is not None) - (1 if errors else 0)
        assert n_ok >= 30  # 绝大多数因子必须成功
        assert not errors, f"因子失败: {errors[:5]}"
