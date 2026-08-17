"""Baostock A-share market data → DAFT Panel adapter.

Baostock is a free, registration-required Python SDK for Chinese A-share
market data (daily and minute bars). This adapter downloads OHLCV data
for a configurable set of stocks and converts it to the Panel format.

Usage:
    adapter = BaostockAdapter({"start_date": "2021-01-01", "n_stocks": 50})
    panel = adapter.load()
"""

from __future__ import annotations
from typing import Dict, List, Optional
import logging
from pathlib import Path

import torch

from daft.data.panel import Panel

_logger = logging.getLogger(__name__)

# Major CSI 300 constituent tickers (Shanghai + Shenzhen)
CSI300_SAMPLE = [
    "sh.600519", "sh.600036", "sh.601318", "sh.600276", "sh.600900",
    "sh.601166", "sh.600030", "sh.600887", "sh.601398", "sh.600585",
    "sh.601288", "sh.600809", "sh.601668", "sh.600028", "sh.600050",
    "sh.601012", "sh.600031", "sh.601899", "sh.600690", "sh.601088",
    "sz.000858", "sz.000333", "sz.002415", "sz.000651", "sz.000002",
    "sz.000568", "sz.300750", "sz.002475", "sz.000725", "sz.002714",
    "sz.000001", "sz.002304", "sz.300059", "sz.000063", "sz.002142",
    "sz.000776", "sz.002352", "sz.300015", "sz.000538", "sz.002230",
    "sz.000625", "sz.002236", "sz.300124", "sz.000895", "sz.002271",
    "sz.000963", "sz.002410", "sz.300498", "sz.000661", "sz.002050",
]


class BaostockAdapter:
    """Download A-share OHLCV via baostock and convert to Panel (T, N, 5).

    Parameters
    ----------
    config : dict
        Keys: start_date, end_date, frequency, n_stocks, tickers.
    """

    def __init__(self, config: Dict):
        self.start_date = config.get("start_date", "2021-01-01")
        self.end_date = config.get("end_date", "2025-12-31")
        self.frequency = config.get("frequency", "d")   # d, w, m, 5, 15, 30, 60
        self.n_stocks = config.get("n_stocks", 50)
        self.tickers = config.get("tickers", None)        # None → use universe
        # 股票池(2026-08-16 新增): "sample" = 内置 CSI300_SAMPLE 静态清单;
        # "hs300" = 用 baostock query_hs300_stocks 按 start_date 拉取真实
        # 沪深300 成分(缓解幸存者偏差 + 解除 50 只上限)
        self.universe = config.get("universe", "sample")
        self.adjust = config.get("adjust", "2")            # 2 = forward-adjusted
        # 涨跌停处理(2026-08-16 新增): True → 涨跌停日 mask=False(不可成交)
        self.handle_limit_up_down = config.get("handle_limit_up_down", True)
        # 磁盘缓存(2026-08-16 新增): True → 首次下载后缓存 Panel, 同样参数
        # 再跑时直接加载, 免重复联网下载(100 股×2400 天约省 20-40 分钟)
        self.use_cache = config.get("use_cache", True)
        self.cache_dir = Path(config.get("cache_dir", "data/cache"))
        self.feature_names = ["open", "high", "low", "close", "volume"]

    # ------------------------------------------------------------------
    def _cache_path(self) -> Path:
        """缓存文件路径, 由影响数据的全部参数决定。"""
        key = (
            f"baostock_{self.universe}_{self.n_stocks}_{self.start_date}"
            f"_{self.end_date}_{self.adjust}_{int(self.handle_limit_up_down)}"
            f"_{self.tickers or ''}"
        )
        import hashlib
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:12]
        return self.cache_dir / f"{digest}.pt"

    def load(self) -> Panel:
        """Download data (with disk cache) and return Panel.

        Requires ``pip install baostock``. Registration is free at
        http://baostock.com but not required for basic daily data.
        """
        cache_path = self._cache_path()
        if self.use_cache and cache_path.exists():
            _logger.info("Loading cached panel: %s", cache_path)
            return torch.load(cache_path, weights_only=False)

        try:
            import baostock as bs
            import pandas as pd
            import numpy as np
        except ImportError:
            raise ImportError(
                "baostock is not installed. Install it with: "
                "pip install baostock"
            )

        # Login
        lg = bs.login()
        if lg.error_code != "0":
            _logger.warning("baostock login: %s — %s", lg.error_code, lg.error_msg)

        tickers = self.tickers or self._resolve_universe(bs)
        _logger.info("Downloading %d stocks from baostock (%s → %s, universe=%s)",
                      len(tickers), self.start_date, self.end_date, self.universe)

        all_data: Dict[str, pd.DataFrame] = {}
        try:
            for ticker in tickers:
                rows = self._query_ticker(bs, ticker)
                if not rows:
                    continue

                df = pd.DataFrame(rows, columns=["date", "open", "high", "low", "close", "volume"])
                df["date"] = pd.to_datetime(df["date"])
                for col in ["open", "high", "low", "close", "volume"]:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
                df = df.set_index("date")
                all_data[ticker] = df

        finally:
            bs.logout()

        if not all_data:
            raise RuntimeError("No data downloaded — check baostock connection and dates.")

        # --- Align to common date index ---
        panel = self._to_panel(all_data)

        # --- 写磁盘缓存(2026-08-16): 下次同参数直接加载, 免重复下载 ---
        if self.use_cache:
            try:
                self.cache_dir.mkdir(parents=True, exist_ok=True)
                torch.save(panel, cache_path)
                _logger.info("Panel cached → %s", cache_path)
            except Exception as e:  # noqa: BLE001 — 缓存失败不影响返回数据
                _logger.warning("Cache write failed: %r", e)

        return panel

    # ------------------------------------------------------------------
    def _resolve_universe(self, bs) -> list:
        """按 universe 解析股票池(2026-08-16 新增)。

        - "hs300": 用 baostock query_hs300_stocks 按 start_date 拉取
          沪深300 成分(取前 n_stocks 只); 失败回退到内置静态清单。
        - "sample": 内置 CSI300_SAMPLE 静态清单(最多 50 只)。
        """
        if self.universe == "hs300" and hasattr(bs, "query_hs300_stocks"):
            try:
                rs = bs.query_hs300_stocks(date=self.start_date)
                if rs.error_code == "0":
                    rows = []
                    while rs.next():
                        rows.append(rs.get_row_data())
                    # 字段: updateDate, code, code_name
                    codes = [r[1] for r in rows]
                    if codes:
                        _logger.info("hs300 universe: %d constituents as of %s",
                                     len(codes), self.start_date)
                        return codes[: self.n_stocks]
            except Exception as e:  # noqa: BLE001 — 回退静态清单
                _logger.warning("query_hs300_stocks failed: %r; 回退 sample", e)
            _logger.warning("hs300 universe 不可用, 回退 sample")
        return CSI300_SAMPLE[: self.n_stocks]

    # ------------------------------------------------------------------
    def _query_ticker(self, bs, ticker: str, max_attempts: int = 3):
        """Query one ticker with retries (baostock 批量拉取偶发瞬时失败,
        2026-08-16 起增加重试, 避免 30 只股票池静默缩水到 23 只)。"""
        import time as _time

        last_err = ""
        for attempt in range(max_attempts):
            try:
                rs = bs.query_history_k_data_plus(
                    ticker,
                    "date,open,high,low,close,volume",
                    start_date=self.start_date,
                    end_date=self.end_date,
                    frequency=self.frequency,
                    adjustflag=self.adjust,
                )
                if rs.error_code != "0":
                    last_err = rs.error_msg
                else:
                    rows = []
                    while rs.next():
                        rows.append(rs.get_row_data())
                    if rows:
                        return rows
                    last_err = "empty result"
            except Exception as e:  # noqa: BLE001 — 重试后仍失败再跳过
                last_err = repr(e)

            if attempt < max_attempts - 1:
                _time.sleep(1.0 * (attempt + 1))

        _logger.warning("Skip %s after %d attempts: %s", ticker, max_attempts, last_err)
        return None

    # ------------------------------------------------------------------
    def _to_panel(self, data_dict: Dict) -> Panel:
        """Align multiple stock DataFrames to a common Panel tensor."""
        import pandas as pd
        import numpy as np

        # Union of all dates
        all_dates = sorted(set().union(*(df.index for df in data_dict.values())))
        date_index = pd.DatetimeIndex(all_dates).sort_values()

        T = len(date_index)
        N = len(data_dict)
        F = len(self.feature_names)

        values = np.zeros((T, N, F), dtype=np.float32)
        mask = np.zeros((T, N), dtype=bool)
        ticker_list = list(data_dict.keys())

        for j, ticker in enumerate(ticker_list):
            df = data_dict[ticker]
            # Reindex to common date grid
            df_aligned = df.reindex(date_index)
            for k, feat in enumerate(self.feature_names):
                col = df_aligned[feat].values
                # Forward-fill up to 5 consecutive missing bars
                col = _forward_fill_limit(col, limit=5)
                values[:, j, k] = np.nan_to_num(col, nan=0.0)

            # Mask: True where close is valid (traded)
            close_col = df_aligned["close"].values
            mask[:, j] = ~np.isnan(close_col)

            # 涨跌停 mask (2026-08-16 新增): 当日 |涨跌幅| 达到涨停板阈值的
            # 日期不可成交(A 股 T+1 下, 涨停买不进/跌停卖不出)。
            # 阈值按交易所板块: 创业板(300/301)、科创板(688) ±20%, 其余 ±10%。
            if self.handle_limit_up_down:
                limit_mask = _limit_move_mask(close_col, ticker)
                mask[:, j] &= limit_mask

        return Panel(
            values=torch.from_numpy(values),
            mask=torch.from_numpy(mask),
            dates=list(date_index),
            asset_ids=ticker_list,
            feature_names=self.feature_names,
            metadata={
                "source": "baostock",
                "start_date": self.start_date,
                "end_date": self.end_date,
                "frequency": self.frequency,
                "handle_limit_up_down": self.handle_limit_up_down,
            },
        )


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

def _limit_move_mask(close_col: "np.ndarray", ticker: str) -> "np.ndarray":
    """(T,) bool — False 表示当日触及涨跌停, 视为不可成交。

    A 股规则(主板 ±10%, 创业板 300/301 与科创板 688 ±20%, 留 0.5% 余量
    防止复权误差): 当日 close 相对前一交易日 close 的 |涨跌幅| ≥ 阈值,
    或当日一字板(开盘即封死, open==close 且涨幅 ≥ 阈值)时置 False。

    首个交易日无前收盘, 不判涨跌停(True); 缺失收盘(NaN)的日子保持
    True — 它们已由 NaN mask 置 False, 不再重复处理。
    """
    import numpy as np

    limit = 0.195 if _is_gem_or_star(ticker) else 0.095
    T = len(close_col)
    out = np.ones(T, dtype=bool)

    prev = close_col[:-1]
    curr = close_col[1:]
    with np.errstate(divide="ignore", invalid="ignore"):
        pct = (curr - prev) / np.abs(prev)
    valid = np.isfinite(pct) & (prev > 0)
    hit = valid & (np.abs(pct) >= limit)
    out[1:][hit] = False
    return out


def _is_gem_or_star(ticker: str) -> bool:
    """创业板(300/301) 或科创板(688) → ±20% 涨跌幅限制。"""
    code = ticker.split(".")[-1]
    return code.startswith("300") or code.startswith("301") or code.startswith("688")


def _forward_fill_limit(arr: "np.ndarray", limit: int = 5) -> "np.ndarray":
    """Forward-fill NaN values up to ``limit`` consecutive bars."""
    import numpy as np
    out = arr.copy()
    nan_count = 0
    last_valid = np.nan
    for i in range(len(out)):
        if np.isnan(out[i]):
            nan_count += 1
            if nan_count <= limit and not np.isnan(last_valid):
                out[i] = last_valid
        else:
            last_valid = out[i]
            nan_count = 0
    return out
