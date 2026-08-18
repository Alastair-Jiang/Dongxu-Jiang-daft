"""EXP-20260816-08: 调仓频率 × 平滑 消融 — 换手控制(有条件 GO 第一动作)。

背景(2026-08-16 扩池后): DAFT 100 股 IC=0.037 有信号, 但换手 2.34/步
在 5bp+1bp 成本下把净 Sharpe 压到 −1.72; 平滑 λ*=0.7 把换手减半后
净 Sharpe −0.60, 仍低于 Ridge 基线的 +0.53。

本实验在平滑基础上叠加调仓频率(rebalance_freq): 回测引擎自 2026-08-16
起支持每 N 天调仓、仅调仓日收费。网格 rebalance_freq ∈ {1,2,5} ×
λ ∈ {0, 0.7}, 参数组合由 **val 段净 Sharpe** 选择, test 段仅报告。

用法: python scripts/run_rebalance_ablation.py [--stocks 100] [--freqs 1,2,5] [--lambdas 0,0.7]
前置: checkpoints/oos/ 必须存在(100 股 hs300 模型产物)。
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from daft.data.adapters.baostock_adapter import BaostockAdapter
from daft.features.regime_features import RegimeFeatureExtractor
from daft.models.factory import build_model
from daft.training.joint_trainer import JointTrainer
from daft.backtest.engine import BacktestEngine
from daft.utils.metrics import rank_info_coefficient, ic_summary, eligible_mask
from daft.utils.experiment import config_hash, next_exp_path
from daft.utils.device import get_device

OUTPUT_DIR = PROJECT_ROOT / "outputs"


def build_ablation_model(ablate="none", hidden=64, n_layers=2):
    return build_model(
        cdap_strength=1.0, router_temperature=0.1, noisy_gating_std=0.0,
        ablate=ablate, hidden=hidden, n_layers=n_layers,
    )


def ema_smooth(signals: torch.Tensor, lam: float) -> torch.Tensor:
    """因果 EMA 平滑: s'_t = (1-λ)·s_t + λ·s'_{t-1}。"""
    if lam <= 0:
        return signals
    smoothed = torch.zeros_like(signals)
    smoothed[0] = signals[0]
    for t in range(1, signals.shape[0]):
        smoothed[t] = (1 - lam) * signals[t] + lam * smoothed[t - 1]
    return smoothed


def evaluate(engine_cfg: dict, smoothed: torch.Tensor, prices_seg: torch.Tensor,
             mask_seg: torch.Tensor):
    """回测 + IC(信号已在脚本内平滑)。"""
    engine = BacktestEngine(engine_cfg)
    bt = engine.run(smoothed, prices_seg, mask=mask_seg)
    log_pt = torch.log(prices_seg.clamp(min=1e-8))
    returns = log_pt[1:] - log_pt[:-1]
    # A4 (2026-08-18): 双条件入样, 与对决主口径统一
    ic = ic_summary(rank_info_coefficient(
        smoothed[:-1], returns, eligible_mask(mask_seg), per_timestep=True))
    return {
        "sharpe": bt["sharpe_ratio"], "annual_return": bt["annual_return"],
        "max_drawdown": bt["max_drawdown"], "turnover": bt["turnover"],
        "ic": ic["ic_mean"], "ic_t": ic["ic_t_stat"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stocks", type=int, default=100)
    parser.add_argument("--freqs", default="1,2,5")
    parser.add_argument("--lambdas", default="0,0.7")
    parser.add_argument("--weight-modes", default="equal,signal_zscore")
    parser.add_argument("--start", default="2021-01-01")
    parser.add_argument("--end", default="2025-12-31")
    parser.add_argument("--universe", default="hs300", choices=["hs300", "sample"])
    parser.add_argument("--ckpt-dir", default="checkpoints/oos",
                        help="checkpoint 目录")
    parser.add_argument("--ablate", default="none",
                        choices=["none", "cdap", "memory", "router"])
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--n-layers", type=int, default=2)
    args = parser.parse_args()
    freqs = [int(x) for x in args.freqs.split(",")]
    lambdas = [float(x) for x in args.lambdas.split(",")]
    weight_modes = [w for w in args.weight_modes.split(",")]
    device = get_device()

    t0 = time.time()
    print(f"=== 调仓频率×平滑消融 (device={device}) ===")

    panel = BaostockAdapter({"start_date": args.start, "end_date": args.end,
                             "frequency": "d", "n_stocks": args.stocks,
                             "universe": args.universe, "adjust": "2"}).load()
    T, N, _ = panel.shape
    t_train_end, t_val_end = int(T * 0.6), int(T * 0.8)
    print(f"Panel: (T={T}, N={N})  train:[0,{t_train_end}) val:[{t_train_end},{t_val_end}) test:[{t_val_end},{T})")

    model, layer_proj = build_ablation_model(ablate=args.ablate,
                                             hidden=args.hidden, n_layers=args.n_layers)
    ckpt_dir = Path(args.ckpt_dir)
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"{ckpt_dir} 不存在 — 先跑 run_full_pipeline_oos.py --ckpt-dir ...")
    JointTrainer.load_checkpoints(model, layer_proj, str(ckpt_dir))
    model = model.to(device)
    layer_proj = layer_proj.to(device)
    model.eval(); layer_proj.eval()
    print(f"Checkpoints 已加载 ({ckpt_dir}, ablate={args.ablate}, device={device})")

    # 标准化(仅 train)
    ext = RegimeFeatureExtractor(n_base_factors=50, output_dim=200)
    with torch.no_grad():
        s_tr = ext(panel.slice_time(0, t_train_end))
    s_tr = torch.nan_to_num(s_tr, nan=0.0, posinf=1e6, neginf=-1e6).clamp(-1e6, 1e6)
    mu = s_tr.reshape(-1, 200).mean(0, keepdim=True)
    sd = s_tr.reshape(-1, 200).std(0, keepdim=True).clamp(min=1e-4)

    # 全量特征 + 因果信号生成
    print("生成信号(逐时步, 记忆因果推进)...")
    with torch.no_grad():
        s_all = ext(panel)
    s_all = torch.nan_to_num(s_all, nan=0.0, posinf=1e6, neginf=-1e6).clamp(-1e6, 1e6)
    s_all = ((s_all - mu) / sd).clamp(-10.0, 10.0)

    model.memory.reset_state(1, device)
    signals = torch.zeros(T - 1, N)
    with torch.no_grad():
        for t in range(T - 1):
            s_b = s_all[t].to(device)
            if model.memory.M is None or model.memory.M.size(0) != N:
                model.memory.reset_state(N, device)
            l0 = layer_proj["l0"](s_b)
            l1 = layer_proj["l1"](s_b)
            l2 = layer_proj["l2"](s_b)
            out = model(s_b, [l0, l1, l2], mode="inference",
                        mask=panel.mask[t].to(device))  # A3: 记忆行语义对齐
            signals[t] = out["signal"].squeeze(-1).cpu()
            model.memory.detach_state()
    print(f"信号: {signals.shape}")

    # 段切片(对齐 k→k+1)
    signals_val = signals[t_train_end:t_val_end]
    prices_val = panel.values[t_train_end:t_val_end, :, 3]
    mask_val = panel.mask[t_train_end:t_val_end]

    signals_test = torch.cat([signals[t_val_end:], torch.zeros(1, N)], dim=0)
    prices_test = panel.values[t_val_end:, :, 3]
    mask_test = panel.mask[t_val_end:]

    base_cfg = {"transaction_cost_bps": 5.0, "slippage_bps": 1.0,
                "top_quantile": 0.2, "long_only": False}

    grid = []
    for freq in freqs:
        for lam in lambdas:
            for wmode in weight_modes:
                cfg = {**base_cfg, "rebalance_freq": freq, "weight_mode": wmode}
                val_r = evaluate(cfg, ema_smooth(signals_val, lam), prices_val, mask_val)
                test_r = evaluate(cfg, ema_smooth(signals_test, lam), prices_test, mask_test)
                grid.append({"freq": freq, "lambda": lam, "weight_mode": wmode,
                             "val": val_r, "test": test_r})
                print(f"\n  freq={freq} λ={lam} w={wmode}: VAL  Sharpe={val_r['sharpe']:+.4f} Turnover={val_r['turnover']:.3f} IC={val_r['ic']:+.4f}")
                print(f"                          TEST Sharpe={test_r['sharpe']:+.4f} Turnover={test_r['turnover']:.3f} IC={test_r['ic']:+.4f} t={test_r['ic_t']:+.2f}")

    best = max(grid, key=lambda g: g["val"]["sharpe"])
    print(f"\n  最优(freq={best['freq']}, λ={best['lambda']}, w={best['weight_mode']}) 由 val 净 Sharpe 选择")

    report = {
        "experiment": "EXP-20260816-08",
        "alignment": "k→k+1",
        "selection": "val 段净 Sharpe 最优(test 仅报告)",
        "ckpt_dir": str(ckpt_dir),
        "ablate": args.ablate,
        "stocks": N, "freqs": freqs, "lambdas": lambdas,
        "weight_modes": weight_modes,
        "best": {"freq": best["freq"], "lambda": best["lambda"],
                 "weight_mode": best["weight_mode"]},
        "config_hash": config_hash({"stocks": N, "freqs": freqs, "lambdas": lambdas,
                                    "weight_modes": weight_modes, **base_cfg}),
        "grid": grid,
        "time_seconds": round(time.time() - t0, 1),
    }
    out_path = next_exp_path(OUTPUT_DIR, "daft-rebalance-ablation")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n    报告 → {out_path}")


if __name__ == "__main__":
    main()
