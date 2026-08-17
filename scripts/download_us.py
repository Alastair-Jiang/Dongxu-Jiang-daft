"""下载美股大盘数据并缓存(yfinance), 供跨市场泛化实验。

用法: python scripts/download_us.py [--stocks 100]
缓存: data/cache/us_<n>.pt (torch.save Panel)
"""
from __future__ import annotations
import argparse, sys, time
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from daft.data.adapters.yfinance_adapter import YFinanceAdapter

# S&P 500 大盘股清单(按行业分散, 2021 年之前均已上市)
US_LARGE = [
    "AAPL","MSFT","NVDA","AMZN","GOOGL","META","TSLA","AVGO","BRK-B","JPM",
    "V","UNH","XOM","LLY","MA","HD","PG","COST","MRK","ABBV",
    "CVX","PEP","KO","ADBE","CRM","AMD","NFLX","BAC","TMO","WMT",
    "ACN","ABT","MCD","CSCO","LIN","INTC","ORCL","WFC","QCOM","TXN",
    "AMGN","IBM","INTU","VZ","CAT","PM","GE","GS","CMCSA","DIS",
    "NEE","RTX","SPGI","DHR","UBER","PFE","HON","LOW","T","MS","AXP",
    "AMAT","BKNG","BLK","C","COP","ELV","ETN","ISRG","LMT","MDT",
    "MO","MU","NKE","PANW","PLD","SBUX","SCHW","SO","SYK","TJX",
    "TMUS","UNP","UPS","DE","GILD","REGN","ADI","CB","CI","CL",
    "DUK","EOG","EQIX","FIS","GD","ITW","KLAC","LRCX","MCO","MMC",
    "MPC","NOW","PGR","PSX","PYPL","SNPS","USB","WM","ZTS","APH",
]
print(f"内置清单 {len(US_LARGE)} 只")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stocks", type=int, default=100)
    parser.add_argument("--start", default="2021-01-01")
    parser.add_argument("--end", default="2025-12-31")
    args = parser.parse_args()

    cache = PROJECT_ROOT / "data" / "cache" / f"us_{args.stocks}.pt"
    if cache.exists():
        print(f"缓存已存在: {cache} — 跳过下载")
        return

    tickers = US_LARGE[: args.stocks]
    t0 = time.time()
    print(f"下载 {len(tickers)} 只美股 ({args.start} → {args.end}) ...")
    panel = YFinanceAdapter({
        "tickers": tickers, "start_date": args.start,
        "end_date": args.end, "frequency": "1d",
    }).load()
    print(f"完成: Panel {panel.shape}, {time.time()-t0:.0f}s")

    cache.parent.mkdir(parents=True, exist_ok=True)
    torch.save(panel, cache)
    print(f"缓存 → {cache}")


if __name__ == "__main__":
    main()
