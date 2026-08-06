"""Yahoo Finance → DAFT Panel adapter.

Downloads OHLCV data for US/international equities via yfinance and
converts to the Panel (T, N, 5) format.

Usage:
    adapter = YFinanceAdapter({"tickers": ["AAPL", "MSFT", "GOOGL"]})
    panel = adapter.load()
"""

from __future__ import annotations
from typing import Dict, List, Optional
import logging

import torch

from daft.data.panel import Panel

_logger = logging.getLogger(__name__)

# Default tickers: liquid US large-caps across sectors
DEFAULT_TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "BRK-B",
    "JPM", "V", "JNJ", "WMT", "PG", "MA", "UNH", "HD", "BAC", "XOM",
    "DIS", "NFLX", "ADBE", "CRM", "CSCO", "INTC", "AMD", "QCOM", "TXN",
    "PYPL", "NKE", "ABT", "LLY", "MRK", "PFE", "KO", "PEP", "T", "VZ",
    "COST", "AVGO", "ORCL", "ACN", "TMO", "DHR", "ABBV", "BMY", "AMGN",
    "SBUX", "CMCSA", "MDT", "LOW", "UPS",
]


class YFinanceAdapter:
    """Download US equity OHLCV via yfinance and convert to Panel (T, N, 5).

    Parameters
    ----------
    config : dict
        Keys: tickers, start_date, end_date, interval, auto_adjust.
    """

    def __init__(self, config: Dict):
        self.tickers = config.get("tickers", DEFAULT_TICKERS[:30])
        self.start_date = config.get("start_date", "2021-01-01")
        self.end_date = config.get("end_date", "2025-12-31")
        self.interval = config.get("frequency", "1d")   # 1d, 1h, 30m, etc.
        self.auto_adjust = config.get("auto_adjust", True)
        self.feature_names = ["open", "high", "low", "close", "volume"]

    # ------------------------------------------------------------------
    def load(self) -> Panel:
        """Download data and return Panel.

        Requires ``pip install yfinance``.
        """
        try:
            import yfinance as yf
            import pandas as pd
            import numpy as np
        except ImportError:
            raise ImportError(
                "yfinance is not installed. Install it with: "
                "pip install yfinance"
            )

        _logger.info("Downloading %d tickers from yfinance (%s → %s)",
                      len(self.tickers), self.start_date, self.end_date)

        # Download all tickers at once (efficient)
        data = yf.download(
            " ".join(self.tickers),
            start=self.start_date,
            end=self.end_date,
            interval=self.interval,
            auto_adjust=self.auto_adjust,
            group_by="ticker",
            progress=False,
            threads=True,
        )

        if data is None or data.empty:
            raise RuntimeError("No data downloaded — check tickers and dates.")

        return self._to_panel(data)

    # ------------------------------------------------------------------
    def _to_panel(self, data: "pd.DataFrame") -> Panel:
        """Convert yfinance multi-level DataFrame to Panel tensor."""
        import numpy as np

        # Handle single-ticker case (no multi-level columns)
        if not isinstance(data.columns, type(data.columns)):
            data = data.copy()
            data.columns = pd.MultiIndex.from_product([[self.tickers[0]], data.columns])

        tickers_available = list(data.columns.get_level_values(0).unique())
        T = len(data)
        N = len(tickers_available)
        F = 5

        values = np.zeros((T, N, F), dtype=np.float32)
        mask = np.zeros((T, N), dtype=bool)
        dates = list(data.index)

        # Map yfinance column names to our [open, high, low, close, volume]
        col_map = {"Open": 0, "High": 1, "Low": 2, "Close": 3, "Volume": 4}

        for j, ticker in enumerate(tickers_available):
            try:
                ticker_data = data[ticker]
            except KeyError:
                continue

            for yf_col, idx in col_map.items():
                if yf_col in ticker_data.columns:
                    col_vals = ticker_data[yf_col].values
                    values[:, j, idx] = np.nan_to_num(col_vals, nan=0.0)

            # Mask: valid where close price exists
            if "Close" in ticker_data.columns:
                mask[:, j] = ~np.isnan(ticker_data["Close"].values)

        return Panel(
            values=torch.from_numpy(values),
            mask=torch.from_numpy(mask),
            dates=dates,
            asset_ids=tickers_available,
            feature_names=self.feature_names,
            metadata={
                "source": "yfinance",
                "start_date": self.start_date,
                "end_date": self.end_date,
                "interval": self.interval,
            },
        )
