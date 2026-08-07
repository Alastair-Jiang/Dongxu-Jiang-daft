"""Real-data adapters: baostock (A-shares), yfinance (US/international)."""

from daft.data.adapters.baostock_adapter import BaostockAdapter
from daft.data.adapters.yfinance_adapter import YFinanceAdapter

__all__ = ["BaostockAdapter", "YFinanceAdapter"]
