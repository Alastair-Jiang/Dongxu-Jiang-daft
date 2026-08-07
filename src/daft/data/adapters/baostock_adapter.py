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
        self.tickers = config.get("tickers", None)        # None → use sample
        self.adjust = config.get("adjust", "2")            # 2 = forward-adjusted
        self.feature_names = ["open", "high", "low", "close", "volume"]

    # ------------------------------------------------------------------
    def load(self) -> Panel:
        """Download data and return Panel.

        Requires ``pip install baostock``. Registration is free at
        http://baostock.com but not required for basic daily data.
        """
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

        tickers = self.tickers or CSI300_SAMPLE[:self.n_stocks]
        _logger.info("Downloading %d stocks from baostock (%s → %s)",
                      len(tickers), self.start_date, self.end_date)

        all_data: Dict[str, pd.DataFrame] = {}
        try:
            for ticker in tickers:
                rs = bs.query_history_k_data_plus(
                    ticker,
                    "date,open,high,low,close,volume",
                    start_date=self.start_date,
                    end_date=self.end_date,
                    frequency=self.frequency,
                    adjustflag=self.adjust,
                )
                if rs.error_code != "0":
                    _logger.debug("Skip %s: %s", ticker, rs.error_msg)
                    continue

                rows = []
                while rs.next():
                    rows.append(rs.get_row_data())
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
        return self._to_panel(all_data)

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
            },
        )


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

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
