"""周线 Ridge 基线（K3 周线验证 · 强制对照，2026-08-17）。

方向 1 的唯一对照组。关键对齐（见 decision-prespec-2026-08-17.md 冻结表）：
  - 数据：与日线同一 baostock 源，本地 resample_weekly(W-FRI)，不重新下载
  - 特征：同一特征集，lookback_scale=0.2（窗口 ÷5）
  - 切分：保持日频相同的日历切分日期（60%/80%），按周重索引 → OOS 完全一致
  - 目标/评估：周线 log 收益，截面 Rank IC + 回测（5bp+1bp, top20%, 多空）

输出逐周 test 段 IC 序列 + 对应周结束日期，供 #28 做配对 ΔRankIC(DAFT−Ridge)。

用法：
  D:\\env\\python.exe scripts\\run_baseline_ridge_weekly.py
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
from daft.data.resample import resample_weekly
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
    parser.add_argument("--stocks", type=int, default=100)
    parser.add_argument("--start", default="2021-01-01")
    parser.add_argument("--end", default="2025-12-31")
    parser.add_argument("--lam", type=float, default=1.0)
    parser.add_argument("--train-frac", type=float, default=0.6,
                        help="训练段占日线比例(对齐 DAFT 60/20/20)")
    parser.add_argument("--val-frac", type=float, default=0.8,
                        help="val 结束占日线比例(test 起点, 对齐 DAFT)")
    args = parser.parse_args()

    t0 = time.time()
    print("=== 周线 Ridge 基线（K3 周线验证 · 强制对照）===")

    # ---------- 1. 日线数据(同一源) ----------
    adapter = BaostockAdapter({
        "start_date": args.start,
        "end_date": args.end,
        "frequency": "d",
        "n_stocks": args.stocks,
        "universe": "hs300",
        "adjust": "2",
    })
    daily = adapter.load()
    T_d, N, F = daily.shape
    print(f"    日线 Panel: (T={T_d}, N={N}, F={F})  {daily.dates[0]} → {daily.dates[-1]}")

    # ---------- 2. 周线重采样 ----------
    weekly = resample_weekly(daily)
    T_w = weekly.values.shape[0]
    print(f"    周线 Panel: T={T_w} 周  {weekly.dates[0]} → {weekly.dates[-1]}")

    # ---------- 3. 周线切分(日历日期对齐日频) ----------
    cut_train_date = daily.dates[int(T_d * args.train_frac)]
    cut_val_date = daily.dates[int(T_d * args.val_frac)]
    T_wm1 = T_w - 1  # targets 长度
    weekly_dates = weekly.dates

    # s_aligned[t] 预测 weekly_dates[t+1] 的收益
    n_train_w = sum(1 for t in range(T_wm1) if weekly_dates[t + 1] <= cut_train_date)
    n_val_w = sum(1 for t in range(T_wm1) if weekly_dates[t + 1] <= cut_val_date)
    print(f"    切分日期: train≤{cut_train_date}  val≤{cut_val_date}")
    print(f"    周线索引: n_train={n_train_w}  n_val={n_val_w}  (test={T_wm1 - n_val_w} 周)")

    # ---------- 4. 周线特征(lookback ÷5) ----------
    extractor = RegimeFeatureExtractor(n_base_factors=50, output_dim=200, lookback_scale=0.2)
    with torch.no_grad():
        s_t_raw = extractor(weekly)
    s_t_raw = torch.nan_to_num(s_t_raw, nan=0.0, posinf=1e6, neginf=-1e6).clamp(-1e6, 1e6)

    close_w = weekly.values[..., 3]
    log_c = torch.log(close_w.clamp(min=1e-8))
    targets = (log_c[1:] - log_c[:-1]).clamp(-0.5, 0.5)  # 周线收益 (T_w-1, N)
    s_aligned = s_t_raw[:-1]
    mask_aligned = weekly.mask[1:]

    # ---------- 5. 逐段压缩(按时间边界, 不做跨段混排) ----------
    def compress(idx_slice):
        X = s_aligned[idx_slice].reshape(-1, 200).float()
        y = targets[idx_slice].reshape(-1).float()
        m = mask_aligned[idx_slice].reshape(-1).bool()
        return X[m], y[m]

    X_train, y_train = compress(slice(0, n_train_w))
    X_test, y_test = compress(slice(n_val_w, T_wm1))
    print(f"    训练样本: {X_train.size(0):,} | 测试样本: {X_test.size(0):,}")

    # ---------- 6. 标准化(仅训练段) ----------
    mu = X_train.mean(dim=0, keepdim=True)
    sd = X_train.std(dim=0, keepdim=True).clamp(min=1e-4)
    X_train_z = (X_train - mu) / sd
    X_test_z = (X_test - mu) / sd

    # ---------- 7. 岭回归 ----------
    t_fit = time.time()
    beta = ridge_fit(X_train_z, y_train, args.lam)
    print(f"    岭回归拟合: {time.time() - t_fit:.2f}s, |β|max={beta.abs().max().item():.4f}")

    # ---------- 8. 样本外指标(逐周 IC) ----------
    with torch.no_grad():
        pred_test = X_test_z @ beta

    T_test = T_wm1 - n_val_w
    pred_2d = torch.full((T_test, N), float("nan"))
    y_2d = torch.full((T_test, N), float("nan"))
    m_2d = torch.zeros(T_test, N, dtype=torch.bool)
    valid_pos = 0
    for t in range(T_test):
        m_row = mask_aligned[n_val_w + t]
        k = m_row.sum().item()
        if k > 0:
            pred_2d[t][m_row] = pred_test[valid_pos:valid_pos + k]
            y_2d[t][m_row] = y_test[valid_pos:valid_pos + k]
            m_2d[t] = m_row
            valid_pos += k

    ic_series = rank_info_coefficient(pred_2d, y_2d, m_2d, per_timestep=True)
    ic_stats = ic_summary(ic_series)
    hit = hit_rate(pred_2d, y_2d, m_2d)

    print(f"\n  ── 周线 · 样本外结果 ──")
    print(f"    Rank IC     : {ic_stats['ic_mean']:+.4f}")
    print(f"    ICIR        : {ic_stats['icir']:+.3f}")
    print(f"    IC t-stat   : {ic_stats['ic_t_stat']:+.2f}")
    print(f"    IC>0 比例   : {ic_stats['ic_positive_ratio']:.1%}")
    print(f"    Hit rate    : {hit:.3f}")

    # ---------- 9. 回测 ----------
    signals_padded = torch.cat([pred_2d.nan_to_num(0.0), torch.zeros(1, N)], dim=0)
    prices_test = close_w[n_val_w:]
    mask_test = weekly.mask[n_val_w:]
    engine = BacktestEngine({
        "transaction_cost_bps": 5.0, "slippage_bps": 1.0,
        "top_quantile": 0.2, "long_only": False,
    })
    bt = engine.run(signals_padded, prices_test, mask=mask_test)

    print(f"\n  ── 周线 · 样本外回测(扣费后) ──")
    for k, v in bt.items():
        if isinstance(v, float):
            print(f"    {k:<22s}: {v:+.4f}")

    # ---------- 10. 逐周 IC 序列(供 #28 配对 ΔRankIC) ----------
    ic_dates = [str(weekly_dates[n_val_w + j + 1]) for j in range(T_test)]
    ic_list = [None if (x != x) else float(x) for x in ic_series]

    report = {
        "baseline": "ridge_weekly",
        "experiment": "weekly_validation_ridge",
        "data": {
            "source": "baostock", "stocks": N, "frequency": "weekly_wfri",
            "start": args.start, "end": args.end,
            "T_daily": T_d, "T_weekly": T_w,
        },
        "split": {
            "train_frac": args.train_frac, "val_frac": args.val_frac,
            "cut_train_date": str(cut_train_date), "cut_val_date": str(cut_val_date),
            "n_train_weekly": n_train_w, "n_val_weekly": n_val_w,
            "n_test_weekly": T_test,
        },
        "config": {"lambda": args.lam, "features": 200, "lookback_scale": 0.2},
        "out_of_sample": {
            "ic_mean": ic_stats["ic_mean"], "ic_std": ic_stats["ic_std"],
            "icir": ic_stats["icir"], "ic_t_stat": ic_stats["ic_t_stat"],
            "ic_positive_ratio": ic_stats["ic_positive_ratio"], "hit_rate": hit,
            # 供 #28 配对 ΔRankIC(DAFT−Ridge)
            "ic_series": ic_list,
            "ic_dates": ic_dates,
        },
        "backtest": bt,
        "config_hash": config_hash({"lambda": args.lam, "freq": "weekly"}),
        "time_seconds": round(time.time() - t0, 1),
    }
    out_path = next_exp_path(OUTPUT_DIR, "ridge-weekly")
    report["experiment_id"] = out_path.stem
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n    报告 → {out_path}")


if __name__ == "__main__":
    main()
