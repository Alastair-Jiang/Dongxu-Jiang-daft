"""DAFT Baseline Experiment — Ridge Regression (业界阶段 1: 建立评估基线).

Purpose
-------
业界研发铁律:复杂模型必须打败简单 baseline 才配被讨论。
本脚本建立第一个可信 baseline:岭回归预测下一期收益。

与 DAFT 主管道的区别(这是关键):
  1. 严格样本外:按时间切分 train(前 80%)/ test(后 20%),只在 test 上评估
  2. 标准化只用 train 段拟合(fit on train, apply frozen to test)——无 look-ahead
  3. 无任何神经网络/路由/记忆组件——纯线性,参数量 = 200 特征

输出
----
outputs/baseline_ridge_report.json — 与 full_pipeline_report.json 同口径指标

用法
----
python scripts/run_baseline_ridge.py [--stocks 50] [--days 300] [--lambda 1.0] [--seed 42]
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

from daft.data.loaders import DataLoader
from daft.features.regime_features import RegimeFeatureExtractor
from daft.backtest.engine import BacktestEngine
from daft.utils.metrics import rank_info_coefficient, ic_summary, hit_rate

OUTPUT_DIR = PROJECT_ROOT / "outputs"


# ---------------------------------------------------------------------------
# 岭回归:闭式解  β = (XᵀX + λI)⁻¹ Xᵀ y   (纯 torch,无 sklearn 依赖)
# ---------------------------------------------------------------------------
def ridge_fit(
    X: torch.Tensor,   # (n_train, F)
    y: torch.Tensor,   # (n_train,)
    lam: float,
) -> torch.Tensor:
    """Fit ridge regression, return weight vector β (F,)."""
    n, F = X.shape
    XtX = X.T @ X
    Xty = X.T @ y
    reg = lam * torch.eye(F, device=X.device) * max(n, 1)  # λ·n 与 MSE loss 对齐
    beta = torch.linalg.solve(XtX + reg, Xty)
    return beta


def main():
    parser = argparse.ArgumentParser(description="DAFT Baseline: Ridge Regression")
    parser.add_argument("--stocks", type=int, default=50)
    parser.add_argument("--days", type=int, default=300)
    parser.add_argument("--lambda", dest="lam", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-frac", type=float, default=0.8,
                        help="时间切分:前 train_frac 做训练,后 1-train_frac 做样本外测试")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    t0 = time.time()

    # ------------------------------------------------------------------
    # 1. 数据:与 DAFT QUICK 配置完全一致(50 股 × 300 天,合成)
    # ------------------------------------------------------------------
    print(f"=== DAFT Baseline: Ridge Regression ===")
    print(f"    数据: {args.stocks} 股 × {args.days} 天 (合成, seed={args.seed})")
    loader = DataLoader({
        "source": "synthetic",
        "n_stocks": args.stocks,
        "n_days": args.days,
        "seed": args.seed,
    })
    panel = loader.load()
    T, N, F = panel.shape
    print(f"    Panel: (T={T}, N={N}, F={F})")

    # ------------------------------------------------------------------
    # 2. 特征:同一 200 维特征引擎(与 DAFT 公平对比)
    # ------------------------------------------------------------------
    extractor = RegimeFeatureExtractor(n_output=200)
    with torch.no_grad():
        s_t_raw = extractor(panel)                 # (T, N, 200)
    s_t_raw = torch.nan_to_num(s_t_raw, nan=0.0, posinf=1e6, neginf=-1e6)
    s_t_raw = s_t_raw.clamp(-1e6, 1e6)

    # 目标:下一期对数收益
    close = panel.values[..., 3]
    log_c = torch.log(close.clamp(min=1e-8))
    targets = (log_c[1:] - log_c[:-1]).clamp(-0.5, 0.5)     # (T-1, N)
    s_aligned = s_t_raw[:-1]                                 # (T-1, N, 200)

    # ------------------------------------------------------------------
    # 3. 时间切分(业界铁律:严格样本外)
    # ------------------------------------------------------------------
    T_m1 = targets.size(0)
    n_train = int(T_m1 * args.train_frac)
    print(f"    时间切分: train {n_train} 步 / test {T_m1 - n_train} 步 (样本外)")

    # 展平为 (样本, 特征)
    X_all = s_aligned.reshape(T_m1 * N, 200).float()
    y_all = targets.reshape(T_m1 * N).float()

    train_idx = torch.arange(n_train * N)
    test_idx = torch.arange(n_train * N, T_m1 * N)

    X_train, X_test = X_all[train_idx], X_all[test_idx]
    y_train, y_test = y_all[train_idx], y_all[test_idx]

    # ------------------------------------------------------------------
    # 4. 标准化:只在 train 段拟合(无 look-ahead)
    # ------------------------------------------------------------------
    mu = X_train.mean(dim=0, keepdim=True)
    sd = X_train.std(dim=0, keepdim=True).clamp(min=1e-4)
    X_train_z = (X_train - mu) / sd
    X_test_z = (X_test - mu) / sd              # 冻结的 train 统计量,应用到 test

    # ------------------------------------------------------------------
    # 5. 训练:岭回归闭式解
    # ------------------------------------------------------------------
    t_fit = time.time()
    beta = ridge_fit(X_train_z, y_train, args.lam)
    print(f"    岭回归拟合: β 维度 {beta.shape[0]}, λ={args.lam}, "
          f"耗时 {time.time()-t_fit:.2f}s")
    print(f"    |β| max={beta.abs().max().item():.4f}, "
          f"非零权重数={(beta.abs()>1e-8).sum().item()}/200")

    # ------------------------------------------------------------------
    # 6. 样本外预测 + 指标
    # ------------------------------------------------------------------
    with torch.no_grad():
        pred_test = X_test_z @ beta                      # (n_test,)
    y_test_np = y_test

    # 时间-截面还原:test 段 (T_test, N)
    # 注意对齐:targets[t] = log_c[t+1] - log_c[t],故 mask 取 mask[1:]
    T_test = T_m1 - n_train
    pred_2d = pred_test.reshape(T_test, N)
    y_2d = y_test_np.reshape(T_test, N)
    mask_2d = panel.mask[1:][n_train:].reshape(T_test, N)

    # --- 样本外 IC(截面秩相关,逐时步)---
    ic_series = rank_info_coefficient(pred_2d, y_2d, mask_2d, per_timestep=True)
    ic_stats = ic_summary(ic_series)
    hit = hit_rate(pred_2d, y_2d, mask_2d)

    print(f"\n  ── 样本外结果 (test, 时间切分) ──")
    print(f"    Rank IC     : {ic_stats['ic_mean']:+.4f}")
    print(f"    IC std      : {ic_stats['ic_std']:.4f}")
    print(f"    ICIR        : {ic_stats['icir']:+.3f}")
    print(f"    IC t-stat   : {ic_stats['ic_t_stat']:+.2f}")
    print(f"    IC>0 比例   : {ic_stats['ic_positive_ratio']:.1%}")
    print(f"    Hit rate    : {hit:.3f}")

    # ------------------------------------------------------------------
    # 7. 回测(与 DAFT 同口径:成本 5bp + 滑点 1bp, top20%, 多空)
    # ------------------------------------------------------------------
    # 信号序列需要与价格对齐:pred_2d 是 t→t+1 的预测,用 t 时刻信号交易 t+1 收益
    # BacktestEngine 的 run(signals, prices):signals[t] 对应 prices[t+1] 收益
    signals_padded = torch.cat([torch.zeros(1, N), pred_2d], dim=0)   # (T_test+1, N)
    prices_test = panel.values[n_train:, :, 3]                         # (T_test+1, N)
    mask_test = panel.mask[n_train:]                                   # (T_test+1, N)

    engine = BacktestEngine({
        "transaction_cost_bps": 5.0,
        "slippage_bps": 1.0,
        "top_quantile": 0.2,
        "long_only": False,
    })
    bt_metrics = engine.run(signals_padded, prices_test, mask=mask_test)

    print(f"\n  ── 样本外回测 (扣费后) ──")
    for k, v in bt_metrics.items():
        if isinstance(v, float):
            print(f"    {k:<22s}: {v:+.4f}")
        else:
            print(f"    {k:<22s}: {v}")

    # ------------------------------------------------------------------
    # 8. 保存报告
    # ------------------------------------------------------------------
    report = {
        "baseline": "ridge_regression",
        "config": {
            "stocks": args.stocks, "days": args.days,
            "lambda": args.lam, "seed": args.seed,
            "train_frac": args.train_frac,
            "features": 200, "params": 200,
        },
        "data": {"T": T, "N": N, "F": F},
        "out_of_sample": {
            "test_steps": T_test,
            "ic_mean": ic_stats["ic_mean"],
            "ic_std": ic_stats["ic_std"],
            "icir": ic_stats["icir"],
            "ic_t_stat": ic_stats["ic_t_stat"],
            "ic_positive_ratio": ic_stats["ic_positive_ratio"],
            "hit_rate": hit,
        },
        "backtest": bt_metrics,
        "note": "严格样本外:前80%时间训练,后20%评估;标准化仅训练段拟合。"
                "与 full_pipeline_report.json 对比时注意:DAFT 主管道为样本内评估,口径不同。",
        "time_seconds": round(time.time() - t0, 1),
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "baseline_ridge_report.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n    报告 → {out_path}")


if __name__ == "__main__":
    main()
