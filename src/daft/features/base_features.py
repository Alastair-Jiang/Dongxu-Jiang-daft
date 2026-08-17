"""Base feature contract: the single conversion point between data-source
layout and feature-engine layout.

面板通道契约(2026-08-16 修复 — 修复前所有实验的 s_t 建立在错列上):
    数据源(baostock / yfinance / synthetic)统一产出 OHLCV 布局:
        ["open", "high", "low", "close", "volume"]
    而 RegimeFeatureExtractor / legacy factors / FreqFeatureExtractor 需要
    基础特征布局:
        ["close", "log_return", "volume", "volume_ratio", "volatility_20"]
    历史上两者被直接混用: 提取器把 open/high/low 当成 close/return/volume,
    所谓"动量/波动率"特征实际是价格水平的滚动函数。

本模块提供唯一转换点。所有特征引擎入口必须首先调用 ``ensure_base_panel``:
    - OHLCV 布局  → 自动计算基础特征
    - 基础布局    → 原样返回(mask 归一化为 2D)
    - 其他/未命名 → 抛出明确错误(失败要响亮,不再静默错列)

所有计算严格因果且 mask 感知:
    - log_return[t]      = log(close[t]) - log(close[t-1]), 需要 t 与 t-1 均可交易
    - volume_ratio[t]    = volume[t] / 滚动 20 日均量(mask 感知)
    - volatility_20[t]   = log_return 的滚动 20 日标准差(mask 感知)
"""

from __future__ import annotations

import torch

from daft.data.panel import Panel
from daft.features.tensor_factors import TensorFactorEngine

OHLCV_FEATURE_NAMES = ["open", "high", "low", "close", "volume"]
BASE_FEATURE_NAMES = ["close", "log_return", "volume", "volume_ratio", "volatility_20"]


def normalize_mask_2d(mask: torch.Tensor) -> torch.Tensor:
    """Normalize a panel mask to (T, N) bool.

    Some legacy callers construct (T, N, F) masks; the Panel contract is
    (T, N). Handle both.
    """
    if mask.ndim == 3:
        return mask[:, :, 0]
    if mask.ndim == 2:
        return mask
    raise ValueError(f"mask 必须是 (T, N) 或 (T, N, F), 得到 {mask.shape}")


def ohlcv_to_base_panel(panel: Panel, vol_window: int = 20) -> Panel:
    """Convert an OHLCV panel to the base feature layout.

    Parameters
    ----------
    panel : Panel
        values (T, N, 5) with feature_names == OHLCV_FEATURE_NAMES.
    vol_window : int
        Rolling window for volume_ratio and volatility_20. Default 20.

    Returns
    -------
    Panel with values (T, N, 5) in BASE_FEATURE_NAMES layout, mask (T, N).
    """
    values = panel.values
    if values.size(2) < 5:
        raise ValueError(f"OHLCV panel 至少需要 5 列, 得到 F={values.size(2)}")

    mask = normalize_mask_2d(panel.mask)
    T, N = values.shape[0], values.shape[1]
    device = values.device
    dtype = values.dtype

    close = values[:, :, 3].clone()
    volume = values[:, :, 4].clone()

    # --- log_return: log(close_t) - log(close_{t-1}), 需 t 与 t-1 均可交易 ---
    log_c = torch.log(close.clamp(min=1e-8))
    log_return = torch.zeros(T, N, device=device, dtype=dtype)
    log_return[1:] = log_c[1:] - log_c[:-1]
    prev_valid = torch.cat([torch.zeros(1, N, device=device, dtype=torch.bool), mask[:-1]], dim=0)
    valid_ret = mask & prev_valid
    log_return[~valid_ret] = 0.0

    # --- volume_ratio: volume / 滚动均量 (mask 感知), 无效处取中性值 1.0 ---
    engine = TensorFactorEngine()
    vol_mean = engine.ts_mean(volume, vol_window, mask)
    volume_ratio = volume / vol_mean.clamp(min=1e-6)
    volume_ratio[~mask | (vol_mean <= 1e-6)] = 1.0

    # --- volatility_20: log_return 的滚动标准差 (mask 感知) ---
    volatility_20 = engine.ts_std(log_return, vol_window, mask)

    base_values = torch.stack(
        [close, log_return, volume, volume_ratio, volatility_20], dim=-1
    )  # (T, N, 5)

    metadata = dict(panel.metadata or {})
    metadata["derived_from"] = "ohlcv"
    return Panel(
        values=base_values,
        mask=mask,
        dates=panel.dates,
        asset_ids=panel.asset_ids,
        feature_names=list(BASE_FEATURE_NAMES),
        metadata=metadata,
    )


def ensure_base_panel(panel: Panel, vol_window: int = 20) -> Panel:
    """Route a panel into the base feature layout (fail loudly on unknowns).

    OHLCV layout → converted; base layout → returned with 2D mask.
    Anything else raises a descriptive error.

    vol_window 透传给 ohlcv_to_base_panel, 周线验证(K3 2026-08-17)下
    lookback_scale=0.2 → 20 日→4 周。
    """
    names = panel.feature_names
    if names is None:
        raise ValueError(
            "panel.feature_names 未设置 — 无法确定通道语义。\n"
            f"支持的布局: OHLCV={OHLCV_FEATURE_NAMES} 或 BASE={BASE_FEATURE_NAMES}。"
        )
    names = list(names)

    if names == BASE_FEATURE_NAMES:
        mask2d = normalize_mask_2d(panel.mask)
        if mask2d is panel.mask:
            return panel
        return Panel(
            values=panel.values, mask=mask2d, dates=panel.dates,
            asset_ids=panel.asset_ids, feature_names=names,
            metadata=panel.metadata,
        )

    if names == OHLCV_FEATURE_NAMES:
        return ohlcv_to_base_panel(panel, vol_window=vol_window)

    raise ValueError(
        f"无法识别的 feature_names: {names}\n"
        f"数据源应产出 {OHLCV_FEATURE_NAMES}, "
        f"特征引擎需要 {BASE_FEATURE_NAMES}。"
    )
