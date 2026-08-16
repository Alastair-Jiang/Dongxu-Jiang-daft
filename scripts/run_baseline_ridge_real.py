"""DAFT Baseline on Real Data — Ridge Regression on baostock A-shares.

业界阶段 1 的关键一步:把 baseline 从合成数据搬到真实 A 股。
- 数据:baostock CSI300 样本(默认 30 只,可配),日线,前复权
- 评估:严格样本外(时间切分 80/20),标准化仅训练段拟合
- 输出:outputs/baseline_ridge_real_report.json

用法:
  python scripts/run_baseline_ridge_real.py [--stocks 30] [--start 2021-01-01] [--end 2025-12-31] [--lambda 1.0]
"""

from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from daft.data.adapters.baostock_adapter import BaostockAdapter
from daft.features.regime_features import RegimeFeatureExtractor
from daft.backtest.engine import BacktestEngine
from daft.utils.metrics import rank_info_coefficient, ic_summary, hit_rate
from daft.utils.experiment import config_hash, next_exp_path

OUTPUT_DIR = PROJECT_ROOT / "outputs"


def ridge_fit(X, y, lam):
    n, F = X.shape
    XtX = X.T @ X
    Xty = X.T @ y
    reg = lam * torch.eye(F, device=X.device) * max(n, 1)
    return torch.linalg.solve(XtX + reg, Xty)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stocks", type=int, default=30)
    parser.add_argument("--start", default="2021-01-01")
    parser.add_argument("--end", default="2025-12-31")
    parser.add_argument("--lam", type=float, default=1.0)
    parser.add_argument("--train-frac", type=float, default=0.8)
    parser.add_argument("--universe", default="hs300",
                        choices=["hs300", "sample"],
                        help="股票池: hs300=按 start 日拉取真实沪深300 成分(默认); sample=内置静态清单")
    args = parser.parse_args()

    t0 = time.time()
    print(f"=== DAFT Baseline: Ridge on REAL A-share data ===")
    print(f"    {args.stocks} 只 CSI300 成分股, {args.start} → {args.end}, 日线, 前复权")

    # ---------- 1. 拉真实数据(baostock) ----------
    adapter = BaostockAdapter({
        "start_date": args.start,
        "end_date": args.end,
        "frequency": "d",
        "n_stocks": args.stocks,
        "universe": args.universe,
        "adjust": "2",           # 2 = 前复权
    })
    panel = adapter.load()
    T, N, F = panel.shape
    print(f"    Panel: (T={T}, N={N}, F={F})")
    print(f"    区间: {panel.dates[0]} → {panel.dates[-1]}")
    n_tradable = panel.mask.float().mean().item()
    print(f"    可交易覆盖率: {n_tradable:.1%}")

    # ---------- 2. 特征 + 目标 ----------
    extractor = RegimeFeatureExtractor(n_base_factors=50, output_dim=200)
    with torch.no_grad():
        s_t_raw = extractor(panel)
    s_t_raw = torch.nan_to_num(s_t_raw, nan=0.0, posinf=1e6, neginf=-1e6).clamp(-1e6, 1e6)

    close = panel.values[..., 3]
    log_c = torch.log(close.clamp(min=1e-8))
    targets = (log_c[1:] - log_c[:-1]).clamp(-0.5, 0.5)
    s_aligned = s_t_raw[:-1]
    mask_aligned = panel.mask[1:]

    # ---------- 3. 严格样本外切分 ----------
    T_m1 = targets.size(0)
    n_train = int(T_m1 * args.train_frac)
    print(f"    时间切分: train {n_train} 步 / test {T_m1 - n_train} 步 (样本外)")

    # 逐段压缩(按时间边界, 不做跨段混排)
    X_tr_all = s_aligned[:n_train].reshape(-1, 200).float()
    y_tr_all = targets[:n_train].reshape(-1).float()
    m_tr_all = mask_aligned[:n_train].reshape(-1).bool()
    X_te_all = s_aligned[n_train:].reshape(-1, 200).float()
    y_te_all = targets[n_train:].reshape(-1).float()
    m_te_all = mask_aligned[n_train:].reshape(-1).bool()

    X_train, y_train = X_tr_all[m_tr_all], y_tr_all[m_tr_all]
    X_test, y_test = X_te_all[m_te_all], y_te_all[m_te_all]
    print(f"    训练样本: {X_train.size(0):,} | 测试样本: {X_test.size(0):,}")

    # ---------- 4. 标准化(仅训练段) ----------
    mu = X_train.mean(dim=0, keepdim=True)
    sd = X_train.std(dim=0, keepdim=True).clamp(min=1e-4)
    X_train_z = (X_train - mu) / sd
    X_test_z = (X_test - mu) / sd

    # ---------- 5. 岭回归 ----------
    t_fit = time.time()
    beta = ridge_fit(X_train_z, y_train, args.lam)
    print(f"    岭回归拟合: 耗时 {time.time()-t_fit:.2f}s, |β|max={beta.abs().max().item():.4f}")

    # ---------- 6. 样本外指标 ----------
    with torch.no_grad():
        pred_test = X_test_z @ beta

    # 还原到 (T_test, N) 用于截面 IC —— 需要按时间组织
    T_test = T_m1 - n_train
    pred_2d = torch.full((T_test, N), float("nan"))
    y_2d = torch.full((T_test, N), float("nan"))
    m_2d = torch.zeros(T_test, N, dtype=torch.bool)
    valid_pos = 0
    for t in range(T_test):
        m_row = mask_aligned[n_train + t]
        k = m_row.sum().item()
        if k > 0:
            pred_2d[t][m_row] = pred_test[valid_pos:valid_pos + k]
            y_2d[t][m_row] = y_test[valid_pos:valid_pos + k]
            m_2d[t] = m_row
            valid_pos += k

    ic_series = rank_info_coefficient(pred_2d, y_2d, m_2d, per_timestep=True)
    ic_stats = ic_summary(ic_series)
    hit = hit_rate(pred_2d, y_2d, m_2d)

    print(f"\n  ── 真实数据 · 样本外结果 ──")
    print(f"    Rank IC     : {ic_stats['ic_mean']:+.4f}")
    print(f"    ICIR        : {ic_stats['icir']:+.3f}")
    print(f"    IC t-stat   : {ic_stats['ic_t_stat']:+.2f}")
    print(f"    IC>0 比例   : {ic_stats['ic_positive_ratio']:.1%}")
    print(f"    Hit rate    : {hit:.3f}")

    # ---------- 7. 回测 ----------
    # 对齐 k→k+1 (2026-08-16 修复): pred_2d[t] 预测 p[n_train+t+1]-p[n_train+t],
    # 因此 positions[t] 必须取 pred_2d[t](旧实现 prepend 一行 0 导致信号晚用一天)。
    signals_padded = torch.cat([pred_2d.nan_to_num(0.0), torch.zeros(1, N)], dim=0)
    prices_test = panel.values[n_train:, :, 3]
    mask_test = panel.mask[n_train:]
    engine = BacktestEngine({
        "transaction_cost_bps": 5.0, "slippage_bps": 1.0,
        "top_quantile": 0.2, "long_only": False,
    })
    bt = engine.run(signals_padded, prices_test, mask=mask_test)

    print(f"\n  ── 真实数据 · 样本外回测(扣费后) ──")
    for k, v in bt.items():
        if isinstance(v, float):
            print(f"    {k:<22s}: {v:+.4f}")

    # ---------- 8. 保存 ----------
    report = {
        "baseline": "ridge_regression",
        "alignment": "k→k+1 (2026-08-16 统一; 与 DAFT-OOS 同口径)",
        "data": {
            "source": "baostock", "stocks": N, "tickers": panel.asset_ids,
            "start": args.start, "end": args.end,
            "frequency": "daily", "adjust": "forward",
            "T": T, "tradable_coverage": round(n_tradable, 4),
        },
        "config": {"lambda": args.lam, "train_frac": args.train_frac, "features": 200, "params": 200},
        "config_hash": config_hash({"lambda": args.lam, "train_frac": args.train_frac}),
        "note": ("与 DAFT-OOS 同口径对比时使用 --train-frac 0.6 "
                 "(DAFT 训练段=60%+val 20%; 本脚本默认 80% 无 val)"),
        "out_of_sample": {
            "train_samples": int(X_train.size(0)), "test_samples": int(X_test.size(0)),
            "ic_mean": ic_stats["ic_mean"], "ic_std": ic_stats["ic_std"],
            "icir": ic_stats["icir"], "ic_t_stat": ic_stats["ic_t_stat"],
            "ic_positive_ratio": ic_stats["ic_positive_ratio"], "hit_rate": hit,
        },
        "backtest": bt,
        "time_seconds": round(time.time() - t0, 1),
    }
    # 唯一产物名 (2026-08-16): 不再覆盖历史报告
    out_path = next_exp_path(OUTPUT_DIR, "ridge-real")
    report["experiment_id"] = out_path.stem
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n    报告 → {out_path}")


if __name__ == "__main__":
    main()
