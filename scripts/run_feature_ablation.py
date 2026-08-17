"""特征边际: Ridge 特征组消融 — 200 维 s_t 的 6 组特征逐组移除, 看哪组贡献 IC。

特征组(regime_features.py): 
  g1 价格动态 0:45, g2 量能 45:80, g3 波动结构 80:120, 
  g4 微观结构 120:140, g5 截面 140:170, g6 动量因子 170:200

用法: python scripts/run_feature_ablation.py [--stocks 100] [--market cn|us]
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from daft.features.regime_features import RegimeFeatureExtractor
from daft.utils.metrics import rank_info_coefficient, ic_summary
from daft.utils.experiment import config_hash, next_exp_path

OUTPUT_DIR = PROJECT_ROOT / "outputs"

GROUPS = [
    ("g1_price", slice(0, 45)),
    ("g2_volume", slice(45, 80)),
    ("g3_volatility", slice(80, 120)),
    ("g4_microstructure", slice(120, 140)),
    ("g5_cross_sectional", slice(140, 170)),
    ("g6_momentum", slice(170, 200)),
]


def ridge_ic(X_train, y_train, X_test, y_test, m_test, N, T_test, mask_aligned, n_train):
    n = X_train.size(0)
    lam = 1.0
    mu = X_train.mean(dim=0, keepdim=True)
    sd = X_train.std(dim=0, keepdim=True).clamp(min=1e-4)
    Xz = (X_train - mu) / sd
    beta = torch.linalg.solve(Xz.T @ Xz + lam * torch.eye(Xz.size(1)) * n, Xz.T @ y_train)
    pred = (X_test - mu) / sd @ beta
    pred_2d = torch.full((T_test, N), float("nan"))
    y_2d = torch.full((T_test, N), float("nan"))
    m_2d = torch.zeros(T_test, N, dtype=torch.bool)
    vp = 0
    for t in range(T_test):
        m_row = mask_aligned[n_train + t]
        k = m_row.sum().item()
        if k > 0:
            pred_2d[t][m_row] = pred[vp:vp + k]
            y_2d[t][m_row] = y_test[vp:vp + k]
            m_2d[t] = m_row
            vp += k
    ic = ic_summary(rank_info_coefficient(pred_2d, y_2d, m_2d, per_timestep=True))
    return ic["ic_mean"], ic["ic_t_stat"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stocks", type=int, default=100)
    parser.add_argument("--market", default="cn", choices=["cn", "us"])
    args = parser.parse_args()

    t0 = time.time()
    if args.market == "cn":
        from daft.data.adapters.baostock_adapter import BaostockAdapter
        panel = BaostockAdapter({"start_date": "2021-01-01", "end_date": "2025-12-31",
                                 "frequency": "d", "n_stocks": args.stocks,
                                 "universe": "hs300", "adjust": "2"}).load()
    else:
        cache = PROJECT_ROOT / "data" / "cache" / f"us_{args.stocks}.pt"
        if not cache.exists():
            raise FileNotFoundError(f"{cache} 不存在 — 先跑 download_us.py")
        panel = torch.load(cache, weights_only=False)

    T, N, _ = panel.shape
    extractor = RegimeFeatureExtractor(n_base_factors=50, output_dim=200)
    with torch.no_grad():
        s_t = extractor(panel)
    s_t = torch.nan_to_num(s_t, nan=0.0, posinf=1e6, neginf=-1e6).clamp(-1e6, 1e6)
    close = panel.values[..., 3]
    log_c = torch.log(close.clamp(min=1e-8))
    targets = (log_c[1:] - log_c[:-1]).clamp(-0.5, 0.5)
    s_aligned = s_t[:-1]
    mask_aligned = panel.mask[1:]
    T_m1 = targets.size(0)
    n_train = int(T_m1 * 0.6)
    T_test = T_m1 - n_train

    X_tr = s_aligned[:n_train].reshape(-1, 200).float()
    y_tr = targets[:n_train].reshape(-1).float()
    m_tr = mask_aligned[:n_train].reshape(-1).bool()
    X_te = s_aligned[n_train:].reshape(-1, 200).float()
    y_te = targets[n_train:].reshape(-1).float()
    m_te = mask_aligned[n_train:].reshape(-1).bool()
    X_train, y_train = X_tr[m_tr], y_tr[m_tr]
    X_test, y_test = X_te[m_te], y_te[m_te]

    # 基线: 全 200 维
    base_ic, base_t = ridge_ic(X_train, y_train, X_test, y_test, m_te, N, T_test,
                               mask_aligned, n_train)
    print(f"基线(full 200 维): IC={base_ic:+.4f} t={base_t:+.2f}")

    results = {"full": {"ic": base_ic, "t": base_t, "drop": None}}
    for name, sl in GROUPS:
        mask_cols = torch.ones(200, dtype=torch.bool)
        mask_cols[sl] = False
        ic, t = ridge_ic(X_train[:, mask_cols], y_train, X_test[:, mask_cols],
                         y_test, m_te, N, T_test, mask_aligned, n_train)
        results[name] = {"ic": ic, "t": t, "drop": name}
        print(f"去掉 {name:18s}: IC={ic:+.4f} t={t:+.2f}  (ΔIC={ic-base_ic:+.4f})")

    report = {
        "experiment": "feature_marginal",
        "market": args.market,
        "stocks": N,
        "groups": {k: v for k, v in results.items()},
        "config_hash": config_hash({"market": args.market, "stocks": N}),
        "time_seconds": round(time.time() - t0, 1),
    }
    out_path = next_exp_path(OUTPUT_DIR, f"feature-ablation-{args.market}")
    report["experiment_id"] = out_path.stem
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"报告 → {out_path}")


if __name__ == "__main__":
    main()
