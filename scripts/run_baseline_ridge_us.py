"""美股 Ridge 基线 — 与 A 股 run_baseline_ridge_real.py 同口径。

数据: data/cache/us_<n>.pt
用法: python scripts/run_baseline_ridge_us.py --stocks 100 [--train-frac 0.6]
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

from daft.features.regime_features import RegimeFeatureExtractor
from daft.backtest.engine import BacktestEngine
from daft.utils.metrics import rank_info_coefficient, ic_summary, hit_rate, eligible_mask
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
    parser.add_argument("--lam", type=float, default=1.0)
    parser.add_argument("--train-frac", type=float, default=0.6)
    args = parser.parse_args()

    t0 = time.time()
    cache = PROJECT_ROOT / "data" / "cache" / f"us_{args.stocks}.pt"
    if not cache.exists():
        raise FileNotFoundError(f"{cache} 不存在 — 先跑 scripts/download_us.py")
    panel = torch.load(cache, weights_only=False)
    T, N, F = panel.shape
    print(f"Panel: (T={T}, N={N})  {panel.dates[0]} → {panel.dates[-1]}")

    extractor = RegimeFeatureExtractor(n_base_factors=50, output_dim=200)
    with torch.no_grad():
        s_t_raw = extractor(panel)
    s_t_raw = torch.nan_to_num(s_t_raw, nan=0.0, posinf=1e6, neginf=-1e6).clamp(-1e6, 1e6)

    close = panel.values[..., 3]
    log_c = torch.log(close.clamp(min=1e-8))
    targets = (log_c[1:] - log_c[:-1]).clamp(-0.5, 0.5)
    s_aligned = s_t_raw[:-1]
    # A4 (2026-08-18): 双条件入样, 与 DAFT-OOS(US) 同口径
    mask_aligned = eligible_mask(panel.mask)

    T_m1 = targets.size(0)
    n_train = int(T_m1 * args.train_frac)
    X_tr = s_aligned[:n_train].reshape(-1, 200).float()
    y_tr = targets[:n_train].reshape(-1).float()
    m_tr = mask_aligned[:n_train].reshape(-1).bool()
    X_te = s_aligned[n_train:].reshape(-1, 200).float()
    y_te = targets[n_train:].reshape(-1).float()
    m_te = mask_aligned[n_train:].reshape(-1).bool()
    X_train, y_train = X_tr[m_tr], y_tr[m_tr]
    X_test, y_test = X_te[m_te], y_te[m_te]

    mu = X_train.mean(dim=0, keepdim=True)
    sd = X_train.std(dim=0, keepdim=True).clamp(min=1e-4)
    beta = ridge_fit((X_train - mu) / sd, y_train, args.lam)
    with torch.no_grad():
        pred_test = (X_test - mu) / sd @ beta

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

    ic_stats = ic_summary(rank_info_coefficient(pred_2d, y_2d, m_2d, per_timestep=True))
    hit = hit_rate(pred_2d, y_2d, m_2d)

    signals_padded = torch.cat([pred_2d.nan_to_num(0.0), torch.zeros(1, N)], dim=0)
    prices_test = panel.values[n_train:, :, 3]
    mask_test = panel.mask[n_train:]
    engine = BacktestEngine({
        "transaction_cost_bps": 5.0, "slippage_bps": 1.0,
        "top_quantile": 0.2, "long_only": False,
    })
    bt = engine.run(signals_padded, prices_test, mask=mask_test)

    print(f"\n  ── Ridge 美股 · 样本外 ──")
    print(f"    IC: {ic_stats['ic_mean']:+.4f}  t: {ic_stats['ic_t_stat']:+.2f}  "
          f"Sharpe: {bt['sharpe_ratio']:+.4f}  换手: {bt['turnover']:.3f}")

    report = {
        "baseline": "ridge_us",
        "market": "us",
        "data": {"stocks": N, "T": T, "start": str(panel.dates[0]), "end": str(panel.dates[-1])},
        "config": {"lambda": args.lam, "train_frac": args.train_frac},
        "config_hash": config_hash({"market": "us", "lam": args.lam}),
        "out_of_sample": {
            "ic_mean": ic_stats["ic_mean"], "ic_std": ic_stats["ic_std"],
            "icir": ic_stats["icir"], "ic_t_stat": ic_stats["ic_t_stat"],
            "ic_positive_ratio": ic_stats["ic_positive_ratio"], "hit_rate": hit,
        },
        "backtest": bt,
        "time_seconds": round(time.time() - t0, 1),
    }
    out_path = next_exp_path(OUTPUT_DIR, "ridge-us")
    report["experiment_id"] = out_path.stem
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"    报告 → {out_path}")


if __name__ == "__main__":
    main()
