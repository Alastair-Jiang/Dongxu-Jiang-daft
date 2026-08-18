"""美股跨市场实验 — 与 A 股 run_full_pipeline_oos.py 完全同口径。

数据: data/cache/us_<n>.pt (先跑 scripts/download_us.py)
口径: 60/20/20 严格样本外, 成本 5bp+1bp, top20% 多空, k→k+1 对齐。
美股无涨跌停 mask(全 True)。

用法:
  python scripts/run_full_pipeline_us.py --stocks 100 --seed 42 [--ablate memory] [--hidden 128]
  python scripts/run_baseline_ridge_us.py --stocks 100   (Ridge 对照)
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
from daft.models.factory import build_experts, build_ensemble, build_layer_proj
from daft.training.expert_trainer import Stage1ExpertTrainer
from daft.training.router_trainer import RouterTrainer
from daft.training.joint_trainer import JointTrainer
from daft.backtest.engine import BacktestEngine
from daft.utils.metrics import rank_info_coefficient, ic_summary, hit_rate
from daft.utils.experiment import config_hash, next_exp_path
from daft.utils.device import get_device

OUTPUT_DIR = PROJECT_ROOT / "outputs"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stocks", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ablate", default="none",
                        choices=["none", "cdap", "memory", "router"])
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--ckpt-dir", default="checkpoints/oos-us")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = get_device()
    t_total = time.time()
    print(f"=== DAFT 美股实验 (stocks={args.stocks}, seed={args.seed}, "
          f"ablate={args.ablate}, hidden={args.hidden}, device={device}) ===")

    cache = PROJECT_ROOT / "data" / "cache" / f"us_{args.stocks}.pt"
    if not cache.exists():
        raise FileNotFoundError(f"{cache} 不存在 — 先跑 scripts/download_us.py --stocks {args.stocks}")
    panel = torch.load(cache, weights_only=False)
    T, N, F = panel.shape
    print(f"Panel: (T={T}, N={N}, F={F})  {panel.dates[0]} → {panel.dates[-1]}")

    t_train_end, t_val_end = int(T * 0.6), int(T * 0.8)
    train_panel = panel.slice_time(0, t_train_end)
    val_panel = panel.slice_time(t_train_end, t_val_end)

    cfg = {
        "stage1": {"epochs": 15, "batch_size": 1024, "lr": 1e-3},
        "stage2": {"epochs": 10, "batch_size": 512, "lr": 1e-3},
        "stage3": {"epochs": 8, "batch_size": 512, "lr": 1e-5},
    }

    print("[Stage 1] 专家训练...")
    experts = build_experts(hidden=args.hidden)
    s1 = Stage1ExpertTrainer(experts=experts, panel=train_panel, device=device)
    s1.train_all(epochs=cfg["stage1"]["epochs"], batch_size=cfg["stage1"]["batch_size"],
                 lr=cfg["stage1"]["lr"], verbose=False)

    print("[Stage 2+3] 路由/记忆/CDAP + 联合微调...")
    model = build_ensemble(experts, cdap_strength=0.1, ablate=args.ablate)
    layer_proj = build_layer_proj()
    s2 = RouterTrainer(model=model, config=cfg["stage2"], device=device)
    s2.train(train_panel, val_panel)
    model.cross_dim_attn.modulation_strength = 1.0
    model.router.temperature = 0.1
    s3 = JointTrainer(model=model, layer_proj=layer_proj, config=cfg["stage3"], device=device)
    s3.train(train_panel, val_panel)
    Path(args.ckpt_dir).mkdir(parents=True, exist_ok=True)
    s3.save_checkpoints(args.ckpt_dir)

    print("[信号生成] 因果推进...")
    extractor = RegimeFeatureExtractor(n_base_factors=50, output_dim=200)
    with torch.no_grad():
        s_tr = extractor(train_panel)
    s_tr = torch.nan_to_num(s_tr, nan=0.0, posinf=1e6, neginf=-1e6).clamp(-1e6, 1e6)
    flat = s_tr.reshape(-1, 200)
    mu = flat.mean(dim=0, keepdim=True)
    sd = flat.std(dim=0, keepdim=True).clamp(min=1e-4)

    with torch.no_grad():
        s_all = extractor(panel)
    s_all = torch.nan_to_num(s_all, nan=0.0, posinf=1e6, neginf=-1e6).clamp(-1e6, 1e6)
    s_all = ((s_all - mu) / sd).clamp(-10.0, 10.0)

    model.eval(); layer_proj.eval()
    model.memory.reset_state(1, device)
    signals = torch.zeros(T - 1, N)
    with torch.no_grad():
        for t in range(T - 1):
            s_b = s_all[t].to(device)
            if model.memory.M is None or model.memory.M.size(0) != N:
                model.memory.reset_state(N, device)
            l0 = layer_proj["l0"](s_b); l1 = layer_proj["l1"](s_b); l2 = layer_proj["l2"](s_b)
            out = model(s_b, [l0, l1, l2], mode="inference",
                       mask=panel.mask[t].to(device))  # A3: 记忆行语义对齐
            signals[t] = out["signal"].squeeze(-1).cpu()
            model.memory.detach_state()

    signals_test = torch.cat([signals[t_val_end:], torch.zeros(1, N)], dim=0)
    prices_test = panel.values[t_val_end:, :, 3]
    mask_test = panel.mask[t_val_end:]

    engine = BacktestEngine({
        "transaction_cost_bps": 5.0, "slippage_bps": 1.0,
        "top_quantile": 0.2, "long_only": False,
    })
    bt = engine.run(signals_test, prices_test, mask=mask_test)
    log_pt = torch.log(prices_test.clamp(min=1e-8))
    returns_test = log_pt[1:] - log_pt[:-1]
    ic_series = rank_info_coefficient(signals_test[:-1], returns_test, mask_test[1:], per_timestep=True)
    ic_stats = ic_summary(ic_series)
    hit = hit_rate(signals_test[:-1], returns_test, mask_test[1:])

    print(f"\n  ── DAFT 美股 · 样本外 ──")
    print(f"    IC: {ic_stats['ic_mean']:+.4f}  t: {ic_stats['ic_t_stat']:+.2f}  "
          f"Sharpe: {bt['sharpe_ratio']:+.4f}  换手: {bt['turnover']:.3f}")

    report = {
        "model": "DAFT_us",
        "market": "us",
        "ablate": args.ablate,
        "seed": args.seed,
        "hidden": args.hidden,
        "data": {"stocks": N, "T": T, "start": str(panel.dates[0]), "end": str(panel.dates[-1])},
        "split": {"train": t_train_end, "val": t_val_end, "test": T},
        "out_of_sample": {
            "ic_mean": ic_stats["ic_mean"], "ic_std": ic_stats["ic_std"],
            "icir": ic_stats["icir"], "ic_t_stat": ic_stats["ic_t_stat"],
            "ic_positive_ratio": ic_stats["ic_positive_ratio"], "hit_rate": hit,
        },
        "backtest": bt,
        "config_hash": config_hash({"market": "us", "ablate": args.ablate,
                                    "hidden": args.hidden, **cfg}),
        "time_seconds": round(time.time() - t_total, 1),
    }
    out_path = next_exp_path(OUTPUT_DIR, "daft-us")
    report["experiment_id"] = out_path.stem
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"    报告 → {out_path}")


if __name__ == "__main__":
    main()
