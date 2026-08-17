"""日线 → 周线重采样（W-FRI，取每周最后一个交易日）。

K3 周线验证（2026-08-17）：不引新数据源、不重新下载，从现有日线 Panel
本地生成周线。聚合规则：
  open   = 周第一个交易日开盘
  high   = 周内可交易日最高（mask 感知）
  low    = 周内可交易日最低（mask 感知）
  close  = 周最后一个交易日收盘
  volume = 周内合计
  mask   = 周最后一个交易日 mask
"""

from __future__ import annotations
import datetime as dt
import torch

from daft.data.panel import Panel


def _week_key(d) -> tuple:
    """Return (ISO year, ISO week) for a date (datetime or str)."""
    if isinstance(d, str):
        d = dt.date.fromisoformat(d[:10])
    iso = d.isocalendar()
    return (iso[0], iso[1])


def resample_weekly(panel: Panel) -> Panel:
    """Resample a daily OHLCV Panel to weekly (W-FRI) Panel."""
    dates = panel.dates
    if dates is None:
        raise ValueError("Panel.dates 未设置，无法按周分组")
    values = panel.values  # (T, N, F)
    mask = panel.mask      # (T, N)
    T, N, F = values.shape

    # 按 (ISO year, ISO week) 分组 → [(start_idx, end_idx), ...]
    groups: list = []
    prev_key = None
    for t in range(T):
        key = _week_key(dates[t])
        if key != prev_key:
            groups.append([t, t + 1])
            prev_key = key
        else:
            groups[-1][1] = t + 1

    n_weeks = len(groups)
    values_w = torch.zeros(n_weeks, N, F, dtype=values.dtype, device=values.device)
    mask_w = torch.zeros(n_weeks, N, dtype=torch.bool, device=mask.device)
    dates_w = []

    for gi, (s, e) in enumerate(groups):
        seg = values[s:e]      # (len, N, F)
        seg_mask = mask[s:e]   # (len, N)

        # open = 第一个交易日开盘
        values_w[gi, :, 0] = seg[0, :, 0]
        # high = 周内可交易日最高
        high = seg[:, :, 1].clone()
        high[~seg_mask] = float("-inf")
        values_w[gi, :, 1] = high.max(dim=0).values
        # low = 周内可交易日最低
        low = seg[:, :, 2].clone()
        low[~seg_mask] = float("inf")
        values_w[gi, :, 2] = low.min(dim=0).values
        # close = 最后一个交易日收盘
        values_w[gi, :, 3] = seg[-1, :, 3]
        # volume = 周内合计
        values_w[gi, :, 4] = seg[:, :, 4].sum(dim=0)
        # mask = 最后一个交易日 mask
        mask_w[gi] = seg_mask[-1]
        dates_w.append(dates[e - 1])

    meta = dict(panel.metadata or {})
    meta["resampled"] = "weekly_wfri"
    return Panel(
        values=values_w,
        mask=mask_w,
        dates=dates_w,
        asset_ids=panel.asset_ids,
        feature_names=panel.feature_names,
        metadata=meta,
    )
