"""独立数据下载脚本 — 预下载 baostock 数据到本地缓存。

训练/实验脚本的 BaostockAdapter 默认启用磁盘缓存(data/cache/):
参数一致时直接秒加载, 不再联网。本脚本用于**提前单独下载**, 让后续
训练立即开始(尤其 100 股 × 长历史需要 20-40 分钟下载)。

用法:
  python scripts/download_data.py --stocks 100 --universe hs300 --start 2016-01-01 --end 2025-12-31
  (参数必须与实验脚本一致才会命中缓存)

强制重新下载: 加 --refresh
"""
from __future__ import annotations
import argparse, sys, time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from daft.data.adapters.baostock_adapter import BaostockAdapter


def main():
    parser = argparse.ArgumentParser(description="预下载 baostock 数据到本地缓存")
    parser.add_argument("--stocks", type=int, default=100)
    parser.add_argument("--universe", default="hs300", choices=["hs300", "sample"])
    parser.add_argument("--start", default="2021-01-01")
    parser.add_argument("--end", default="2025-12-31")
    parser.add_argument("--frequency", default="d")
    parser.add_argument("--refresh", action="store_true",
                        help="忽略已有缓存, 强制重新下载")
    args = parser.parse_args()

    cfg = {
        "start_date": args.start, "end_date": args.end,
        "frequency": args.frequency, "n_stocks": args.stocks,
        "universe": args.universe, "adjust": "2",
        "use_cache": not args.refresh,
    }
    adapter = BaostockAdapter(cfg)
    cache_path = adapter._cache_path()
    if cache_path.exists() and not args.refresh:
        print(f"缓存已存在: {cache_path} — 无需下载(要强制重下用 --refresh)")
        return

    t0 = time.time()
    print(f"开始下载: {args.stocks} 股 {args.universe} {args.start} → {args.end} ...")
    panel = adapter.load()
    dt = time.time() - t0
    print(f"完成: Panel {panel.shape}, 可交易覆盖率 {panel.mask.float().mean().item():.1%}, "
          f"耗时 {dt:.0f}s")
    print(f"缓存位置: {adapter._cache_path()}")
    print("后续训练/实验用相同参数即可秒加载。")


if __name__ == "__main__":
    main()
