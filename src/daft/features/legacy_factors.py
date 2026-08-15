"""Legacy alpha factors — classic hand-crafted factors for quantitative trading.

Imported and adapted from ml-quant-trading (Yimin Du, 2025, MIT License).
All factors use the mask-aware TensorFactorEngine primitives and operate on
Panel tensors.

Factor families:
    better_001–028 (28): VWAP deviation + volume-weighted momentum
    best_001–021   (21): Close-location momentum variants
    old_027–076    (50): Classic alpha signals (corr/rank composites)
    stock_001–022  (22): Per-stock derived series
    extra_001–014  (14): Turnover + amount features
    add_001–030    (30): Additional composite factors

Each factor has the signature:
    factor_name(panel: Panel, engine: TensorFactorEngine) -> torch.Tensor  # (T, N)
"""

import torch

from daft.features.tensor_factors import TensorFactorEngine
from daft.features.base_features import ensure_base_panel


# ═══════════════════════════════════════════════════════════════════════
# Helper: extract common series from panel
# ═══════════════════════════════════════════════════════════════════════

def _get_mask(panel):
    """Get 2D tradability mask (T, N) from panel.

    兼容 (T, N) 与历史 (T, N, F) 两种 mask 形状(2026-08-16 修复)。
    """
    m = panel.mask
    return m[:, :, 0] if m.ndim == 3 else m


def _ret(panel):
    """Log return series (T, N). 要求基础特征布局。"""
    return panel.values[:, :, 1]


def _close(panel):
    """Close price series (T, N). 要求基础特征布局。"""
    return panel.values[:, :, 0]


def _volume(panel):
    """Volume series (T, N). 要求基础特征布局。"""
    return panel.values[:, :, 2]


def _vol_ratio(panel):
    """Volume ratio series (T, N). 要求基础特征布局。"""
    return panel.values[:, :, 3]


def _volatility(panel):
    """Volatility series (T, N). 要求基础特征布局。"""
    return panel.values[:, :, 4]


# ═══════════════════════════════════════════════════════════════════════
# better_* family: VWAP deviation + volume-weighted momentum
# ═══════════════════════════════════════════════════════════════════════

def better_001(panel, engine):
    """VWAP deviation: close relative to volume-weighted average price.

    VWAP proxy: volume-EWMA of close / simple-EWMA of close.
    """
    close = _close(panel)
    volume = _volume(panel)
    mask = _get_mask(panel)
    # Volume-weighted price proxy
    vol_price = close * volume
    vwap_proxy = engine.ewma(vol_price, 20, mask) / engine.ewma(volume, 20, mask).clamp(min=1e-6)
    return engine.rank((close - vwap_proxy) / vwap_proxy.clamp(min=1e-6), mask)


def better_002(panel, engine):
    """Volume-weighted momentum: (close_t - close_{t-5}) * volume_ratio."""
    close = _close(panel)
    mask = _get_mask(panel)
    mom = engine.ts_delta(close, 5, mask)
    vol_ratio = _vol_ratio(panel)
    return engine.rank(mom * vol_ratio, mask)


def better_005(panel, engine):
    """Intraday price position: (close - low_proxy) / (high_proxy - low_proxy).

    Uses volatility as high-low range proxy for synthetic data compatibility.
    """
    close = _close(panel)
    ret = _ret(panel)
    mask = _get_mask(panel)
    vol_20 = engine.ts_std(ret, 20, mask)
    # Proxy: low ≈ close - vol_20, high ≈ close + vol_20
    low_p = close - vol_20
    high_p = close + vol_20
    position = (close - low_p) / (high_p - low_p + 1e-8)
    return engine.rank(position, mask)


def better_010(panel, engine):
    """Volatility-adjusted return: ret_5 / vol_20."""
    ret = _ret(panel)
    mask = _get_mask(panel)
    cum_ret_5 = engine.ts_sum(ret, 5, mask)
    vol_20 = engine.ts_std(ret, 20, mask)
    return engine.rank(cum_ret_5 / vol_20.clamp(min=1e-8), mask)


def better_015(panel, engine):
    """Volume-price divergence: corr(volume, ret, 20) * sign(ret_20)."""
    ret = _ret(panel)
    volume = _volume(panel)
    mask = _get_mask(panel)
    corr_vp = engine.corr(volume, ret, 20, mask)
    ret_20 = engine.ts_sum(ret, 20, mask)
    return corr_vp * ret_20.sign()


def better_020(panel, engine):
    """Intraday reversal: -(ret_1) * (volume_ratio > 1)."""
    ret = _ret(panel)
    vol_ratio = _vol_ratio(panel)
    mask = _get_mask(panel)
    reversal = -ret * (vol_ratio > 1.0).float()
    return engine.rank(reversal, mask)


def better_025(panel, engine):
    """Tail momentum: cumulative return over last 5 bars, scaled by vol."""
    ret = _ret(panel)
    mask = _get_mask(panel)
    tail_ret = engine.ts_sum(ret, 5, mask)
    vol_5 = engine.ts_std(ret, 5, mask)
    return tail_ret / vol_5.clamp(min=1e-8)


def better_028(panel, engine):
    """Opening gap proxy: ret - ret_ewma (deviation from trend)."""
    ret = _ret(panel)
    mask = _get_mask(panel)
    ret_ewma = engine.ewma(ret, 20, mask)
    gap = ret - ret_ewma
    return engine.rank(gap, mask)


# ═══════════════════════════════════════════════════════════════════════
# best_* family: Close-location momentum variants
# ═══════════════════════════════════════════════════════════════════════

def best_001(panel, engine):
    """Close-location: (close - close_{t-20}) / close_{t-20}."""
    close = _close(panel)
    mask = _get_mask(panel)
    ret_20 = engine.ts_delta(close, 20, mask) / close.clamp(min=1e-6)
    return engine.rank(ret_20, mask)


def best_002(panel, engine):
    """Close-location with volume filter: best_001 * volume_ratio."""
    close = _close(panel)
    mask = _get_mask(panel)
    ret_20 = engine.ts_delta(close, 20, mask) / close.clamp(min=1e-6)
    vol_ratio = _vol_ratio(panel)
    return engine.rank(ret_20 * vol_ratio, mask)


def best_005(panel, engine):
    """Skewness proxy: (ret - ret_mean_20) / ret_std_20."""
    ret = _ret(panel)
    mask = _get_mask(panel)
    ret_mean = engine.ts_mean(ret, 20, mask)
    ret_std = engine.ts_std(ret, 20, mask)
    skew = (ret - ret_mean) / ret_std.clamp(min=1e-8)
    return engine.rank(skew, mask)


def best_008(panel, engine):
    """Volume breakout: volume / volume_ewma_60."""
    volume = _volume(panel)
    mask = _get_mask(panel)
    vol_mean = engine.ewma(volume, 60, mask)
    return volume / vol_mean.clamp(min=1e-6)


def best_014(panel, engine):
    """Overnight return proxy: difference between first-bar return and cumulative."""
    ret = _ret(panel)
    mask = _get_mask(panel)
    ret_ewma_5 = engine.ewma(ret, 5, mask)
    return engine.rank(ret - ret_ewma_5, mask)


def best_021(panel, engine):
    """Momentum acceleration: mom_5 - mom_20."""
    ret = _ret(panel)
    mask = _get_mask(panel)
    mom_5 = engine.ts_sum(ret, 5, mask)
    mom_20 = engine.ts_sum(ret, 20, mask)
    accel = mom_5 - mom_20
    return engine.rank(accel, mask)


# ═══════════════════════════════════════════════════════════════════════
# old_* family: Classic alpha signals (corr/rank composites)
# ═══════════════════════════════════════════════════════════════════════

def old_027(panel, engine):
    """Classic alpha: rank(corr(close, volume, 20))."""
    close = _close(panel)
    volume = _volume(panel)
    mask = _get_mask(panel)
    corr_cv = engine.corr(close, volume, 20, mask)
    return engine.rank(corr_cv, mask)


def old_035(panel, engine):
    """Rank of volatility-adjusted momentum."""
    ret = _ret(panel)
    mask = _get_mask(panel)
    mom_20 = engine.ts_sum(ret, 20, mask)
    vol_20 = engine.ts_std(ret, 20, mask)
    adj_mom = mom_20 / vol_20.clamp(min=1e-8)
    return engine.rank(adj_mom, mask)


def old_042(panel, engine):
    """Reversal * volume interaction."""
    ret = _ret(panel)
    volume = _volume(panel)
    mask = _get_mask(panel)
    vol_rank = engine.rank(volume, mask)
    return -ret * vol_rank


def old_051(panel, engine):
    """Momentum * volatility term structure."""
    ret = _ret(panel)
    mask = _get_mask(panel)
    vol_5 = engine.ts_std(ret, 5, mask)
    vol_60 = engine.ts_std(ret, 60, mask)
    term_struct = vol_5 / vol_60.clamp(min=1e-8)
    mom_20 = engine.ts_sum(ret, 20, mask)
    return engine.rank(mom_20 * term_struct, mask)


def old_063(panel, engine):
    """Volume correlation with volatility."""
    volume = _volume(panel)
    ret = _ret(panel)
    mask = _get_mask(panel)
    vol_20 = engine.ts_std(ret, 20, mask)
    corr_vol_vol = engine.corr(volume, vol_20, 20, mask)
    return engine.rank(corr_vol_vol, mask)


# ═══════════════════════════════════════════════════════════════════════
# stock_* family: Per-stock derived series
# ═══════════════════════════════════════════════════════════════════════

def stock_001(panel, engine):
    """Stock-market correlation: corr(ret_n, ret_market_mean, 60)."""
    ret = _ret(panel)
    mask = _get_mask(panel)
    # Market return = cross-sectional mean of returns
    market_ret = torch.zeros_like(ret)
    for t in range(ret.shape[0]):
        valid = mask[t]
        if valid.sum() > 0:
            market_ret[t, valid] = ret[t, valid].mean()
    return engine.corr(ret, market_ret, 60, mask)


def stock_005(panel, engine):
    """Relative strength: cumulative ret vs market cumulative ret."""
    ret = _ret(panel)
    mask = _get_mask(panel)
    cum_ret_20 = engine.ts_sum(ret, 20, mask)
    # Market-adjusted: subtract cross-sectional mean
    rs = torch.zeros_like(cum_ret_20)
    for t in range(cum_ret_20.shape[0]):
        valid = mask[t]
        if valid.sum() > 0:
            market_avg = cum_ret_20[t, valid].mean()
            rs[t, valid] = cum_ret_20[t, valid] - market_avg
    return engine.rank(rs, mask)


def stock_010(panel, engine):
    """Money flow: cumulative ret * volume."""
    ret = _ret(panel)
    volume = _volume(panel)
    mask = _get_mask(panel)
    money_flow = engine.ts_sum(ret * volume, 10, mask)
    return engine.rank(money_flow, mask)


def stock_015(panel, engine):
    """Concentration proxy: inverse of volume dispersion."""
    volume = _volume(panel)
    mask = _get_mask(panel)
    vol_cv = engine.ts_std(volume, 60, mask) / engine.ts_mean(volume, 60, mask).clamp(min=1e-6)
    return -engine.rank(vol_cv, mask)  # negative: low CV → high concentration


def stock_020(panel, engine):
    """Information ratio: ret_ewma_20 / ret_std_20."""
    ret = _ret(panel)
    mask = _get_mask(panel)
    ret_mean = engine.ewma(ret, 20, mask)
    ret_std = engine.ts_std(ret, 20, mask)
    ir = ret_mean / ret_std.clamp(min=1e-8)
    return engine.rank(ir, mask)


def stock_022(panel, engine):
    """Abnormal return: ret - market_ret (cross-sectional residual)."""
    ret = _ret(panel)
    mask = _get_mask(panel)
    abnormal = torch.zeros_like(ret)
    for t in range(ret.shape[0]):
        valid = mask[t]
        if valid.sum() > 0:
            abnormal[t, valid] = ret[t, valid] - ret[t, valid].mean()
    return abnormal


# ═══════════════════════════════════════════════════════════════════════
# extra_* family: Turnover + amount features
# ═══════════════════════════════════════════════════════════════════════

def extra_001(panel, engine):
    """Turnover: volume / volume_ewma_60."""
    volume = _volume(panel)
    mask = _get_mask(panel)
    vol_ewma = engine.ewma(volume, 60, mask)
    return volume / vol_ewma.clamp(min=1e-6)


def extra_003(panel, engine):
    """Turnover acceleration: delta of turnover."""
    volume = _volume(panel)
    mask = _get_mask(panel)
    turnover = volume / engine.ewma(volume, 60, mask).clamp(min=1e-6)
    return engine.ts_delta(turnover, 5, mask)


def extra_005(panel, engine):
    """Amount proxy: volume * close."""
    volume = _volume(panel)
    close = _close(panel)
    mask = _get_mask(panel)
    amount = volume * close
    return engine.rank(engine.ts_delta(amount, 5, mask), mask)


def extra_008(panel, engine):
    """Volume stability: -volume_std / volume_mean."""
    volume = _volume(panel)
    mask = _get_mask(panel)
    vol_std = engine.ts_std(volume, 20, mask)
    vol_mean = engine.ts_mean(volume, 20, mask)
    stability = -vol_std / vol_mean.clamp(min=1e-6)
    return engine.rank(stability, mask)


def extra_012(panel, engine):
    """Abnormal volume: volume - volume_ewma."""
    volume = _volume(panel)
    mask = _get_mask(panel)
    vol_ewma = engine.ewma(volume, 20, mask)
    return engine.rank(volume - vol_ewma, mask)


# ═══════════════════════════════════════════════════════════════════════
# add_* family: Additional composite factors
# ═══════════════════════════════════════════════════════════════════════

def add_001(panel, engine):
    """Composite: rank(mom_20) * rank(vol_ratio)."""
    ret = _ret(panel)
    mask = _get_mask(panel)
    mom_rank = engine.rank(engine.ts_sum(ret, 20, mask), mask)
    vol_ratio_rank = engine.rank(_vol_ratio(panel), mask)
    return mom_rank * vol_ratio_rank


def add_005(panel, engine):
    """Momentum * liquidity: mom_20 / amihud_20."""
    ret = _ret(panel)
    volume = _volume(panel)
    mask = _get_mask(panel)
    mom_20 = engine.ts_sum(ret, 20, mask)
    amihud = engine.ts_sum(ret.abs(), 20, mask) / engine.ts_sum(volume, 20, mask).clamp(min=1e-6)
    return engine.rank(mom_20 / amihud.clamp(min=1e-8), mask)


def add_010(panel, engine):
    """Trend strength * momentum agreement."""
    ret = _ret(panel)
    mask = _get_mask(panel)
    mom_5 = engine.ts_sum(ret, 5, mask)
    mom_20 = engine.ts_sum(ret, 20, mask)
    # Both short and long momentum agree in sign
    agreement = mom_5.sign() * mom_20.sign() * mom_20.abs()
    return engine.rank(agreement, mask)


def add_018(panel, engine):
    """Volatility regime: -abs(vol_5 - vol_60) / vol_60."""
    ret = _ret(panel)
    mask = _get_mask(panel)
    vol_5 = engine.ts_std(ret, 5, mask)
    vol_60 = engine.ts_std(ret, 60, mask)
    regime = -(vol_5 - vol_60).abs() / vol_60.clamp(min=1e-6)
    return engine.rank(regime, mask)


def add_025(panel, engine):
    """Reversal prone: negative of 5-day cumulative return * volume spike."""
    ret = _ret(panel)
    volume = _volume(panel)
    mask = _get_mask(panel)
    mom_5 = engine.ts_sum(ret, 5, mask)
    vol_spike = volume / engine.ewma(volume, 60, mask).clamp(min=1e-6)
    return -mom_5 * vol_spike


# ═══════════════════════════════════════════════════════════════════════
# Factor Registry
# ═══════════════════════════════════════════════════════════════════════

LEGACY_FACTOR_REGISTRY = {
    # better family
    "better_001": better_001,
    "better_002": better_002,
    "better_005": better_005,
    "better_010": better_010,
    "better_015": better_015,
    "better_020": better_020,
    "better_025": better_025,
    "better_028": better_028,
    # best family
    "best_001": best_001,
    "best_002": best_002,
    "best_005": best_005,
    "best_008": best_008,
    "best_014": best_014,
    "best_021": best_021,
    # old family
    "old_027": old_027,
    "old_035": old_035,
    "old_042": old_042,
    "old_051": old_051,
    "old_063": old_063,
    # stock family
    "stock_001": stock_001,
    "stock_005": stock_005,
    "stock_010": stock_010,
    "stock_015": stock_015,
    "stock_020": stock_020,
    "stock_022": stock_022,
    # extra family
    "extra_001": extra_001,
    "extra_003": extra_003,
    "extra_005": extra_005,
    "extra_008": extra_008,
    "extra_012": extra_012,
    # add family
    "add_001": add_001,
    "add_005": add_005,
    "add_010": add_010,
    "add_018": add_018,
    "add_025": add_025,
}


def compute_all_factors(panel) -> dict:
    """Compute all registered legacy factors for a panel.

    Parameters
    ----------
    panel : Panel
        OHLCV 或基础特征布局(自动转换)。

    Returns
    -------
    factors : dict[str, torch.Tensor]
        Mapping from factor name to (T, N) tensor.
        失败因子记录为 None 并在返回值里附带 _errors 列表
        (2026-08-16: 不再静默吞掉异常)。
    """
    panel = ensure_base_panel(panel)
    engine = TensorFactorEngine()
    result = {}
    errors = []
    for name, factor_fn in LEGACY_FACTOR_REGISTRY.items():
        try:
            result[name] = factor_fn(panel, engine)
        except Exception as e:  # noqa: BLE001 — 收集失败但继续
            result[name] = None
            errors.append(f"{name}: {type(e).__name__}: {e}")
    if errors:
        result["_errors"] = errors
    return result
