"""EXP-20260816-09: 多窗口 walk-forward — 单窗口结论复核。

2 折滚动(扩展窗):
  折 1: train [0, t1), test [t1, t2)
  折 2: train [0, t2), test [t2, T)   (t2 = 默认 80% 处)
每折独立训练(quick 配置) + 因果信号生成 + 测试段回测/IC。
输出每折 IC/Sharpe 的均值±std, 用唯一产物名登记。

用法: python scripts/run_walk_forward.py [--stocks 100] [--universe hs300]
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import importlib.util

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from daft.data.adapters.baostock_adapter import BaostockAdapter
from daft.features.regime_features import RegimeFeatureExtractor
from daft.models.factory import build_experts, build_ensemble, build_layer_proj
from daft.training.expert_trainer import Stage1ExpertTrainer
from daft.training.router_trainer import RouterTrainer
from daft.training.joint_trainer import JointTrainer
from daft.backtest.engine import BacktestEngine
from daft.utils.metrics import rank_info_coefficient, ic_summary
from daft.utils.experiment import config_hash, next_exp_path
from daft.utils.device import get_device

OUTPUT_DIR = PROJECT_ROOT / "outputs"


def run_fold(panel, t_train_end, t_test_start, t_test_end, cfg, seed=42):
    """一折: train [0,t_train_end) 训练, test [t_test_start,t_test_end) 评估。"""
    torch.manual_seed(seed)
    device = get_device()
    T, N, _ = panel.shape
    train_panel = panel.slice_time(0, t_train_end)
    # 训练段内部再切 85/15 供 stage2/3 早停
    t_val_inner = int(t_train_end * 0.85)
    train_inner = panel.slice_time(0, t_val_inner)
    val_inner = panel.slice_time(t_val_inner, t_train_end)

    # Stage 1
    experts = build_experts()
    s1 = Stage1ExpertTrainer(experts=experts, panel=train_inner, device=device)
    s1.train_all(epochs=cfg["stage1"]["epochs"], batch_size=cfg["stage1"]["batch_size"],
                 lr=cfg["stage1"]["lr"], verbose=False)

    # Stage 2
    model = build_ensemble(experts, cdap_strength=0.1)
    layer_proj = build_layer_proj()
    s2 = RouterTrainer(model=model, config=cfg["stage2"], device=device)
    s2.train(train_inner, val_inner)

    # Stage 3
    model.cross_dim_attn.modulation_strength = 1.0
    model.router.temperature = 0.1
    s3 = JointTrainer(model=model, layer_proj=layer_proj, config=cfg["stage3"], device=device)
    s3.train(train_inner, val_inner)

    # 标准化(仅该折 train) + 因果信号生成
    extractor = RegimeFeatureExtractor(n_base_factors=50, output_dim=200)
    with torch.no_grad():
        s_tr = extractor(train_panel)
    s_tr = torch.nan_to_num(s_tr, nan=0.0, posinf=1e6, neginf=-1e6).clamp(-1e6, 1e6)
    mu = s_tr.reshape(-1, 200).mean(0, keepdim=True)
    sd = s_tr.reshape(-1, 200).std(0, keepdim=True).clamp(min=1e-4)

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
            l0 = layer_proj["l0"](s_b)
            l1 = layer_proj["l1"](s_b)
            l2 = layer_proj["l2"](s_b)
            out = model(s_b, [l0, l1, l2], mode="inference",
                       mask=panel.mask[t].to(device))  # A3: 记忆行语义对齐
            signals[t] = out["signal"].squeeze(-1).cpu()
            model.memory.detach_state()

    # 测试段评估(对齐 k→k+1): signals[t] 预测 p[t+1]-p[t]。
    # signals 只有 T-1 行, 测试段终点为 T 时需补一行哑元与 prices 对齐。
    n_test = t_test_end - t_test_start
    n_sig = min(t_test_end, T - 1) - t_test_start
    seg = signals[t_test_start:min(t_test_end, T - 1)]
    if n_sig < n_test:
        seg = torch.cat([seg, torch.zeros(n_test - n_sig, N)], dim=0)
    signals_test = seg
    prices_test = panel.values[t_test_start:t_test_end, :, 3]
    mask_test = panel.mask[t_test_start:t_test_end]

    engine = BacktestEngine({"transaction_cost_bps": 5.0, "slippage_bps": 1.0,
                             "top_quantile": 0.2, "long_only": False})
    bt = engine.run(signals_test, prices_test, mask=mask_test)
    log_pt = torch.log(prices_test.clamp(min=1e-8))
    returns = log_pt[1:] - log_pt[:-1]
    ic = ic_summary(rank_info_coefficient(
        signals_test[:-1], returns, mask_test[1:], per_timestep=True))
    return {"train_days": t_train_end, "test_days": n_test,
            "ic": ic["ic_mean"], "ic_t": ic["ic_t_stat"],
            "sharpe": bt["sharpe_ratio"], "turnover": bt["turnover"],
            "max_drawdown": bt["max_drawdown"]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stocks", type=int, default=100)
    parser.add_argument("--universe", default="hs300", choices=["hs300", "sample"])
    parser.add_argument("--start", default="2021-01-01")
    parser.add_argument("--end", default="2025-12-31")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    t0 = time.time()
    print("=== EXP-20260816-09: 多窗口 walk-forward (2 折扩展窗) ===")
    panel = BaostockAdapter({"start_date": args.start, "end_date": args.end,
                             "frequency": "d", "n_stocks": args.stocks,
                             "universe": args.universe, "adjust": "2"}).load()
    T, N, _ = panel.shape
    t1, t2 = int(T * 0.6), int(T * 0.8)
    print(f"Panel: (T={T}, N={N})  折1: train[0,{t1}) test[{t1},{t2}) | 折2: train[0,{t2}) test[{t2},{T})")

    cfg = {
        "stage1": {"epochs": 15, "batch_size": 1024, "lr": 1e-3},
        "stage2": {"epochs": 10, "batch_size": 512, "lr": 1e-3},
        "stage3": {"epochs": 8,  "batch_size": 512, "lr": 1e-5},
    }

    folds = []
    for i, (tr_end, te_start, te_end) in enumerate([(t1, t1, t2), (t2, t2, T)], 1):
        print(f"\n──── 折 {i} ────")
        r = run_fold(panel, tr_end, te_start, te_end, cfg, seed=args.seed)
        folds.append({"fold": i, **r})
        print(f"  折{i}: test {r['test_days']} 天 → IC={r['ic']:+.4f} t={r['ic_t']:+.2f} "
              f"Sharpe={r['sharpe']:+.4f} 换手={r['turnover']:.3f}")

    ics = [f["ic"] for f in folds]
    sharpes = [f["sharpe"] for f in folds]
    report = {
        "experiment": "EXP-20260816-09",
        "seed": args.seed,
        "stocks": N, "folds": folds,
        "summary": {
            "ic_mean": sum(ics) / len(ics),
            "ic_std": (sum((x - sum(ics) / len(ics)) ** 2 for x in ics) / max(len(ics) - 1, 1)) ** 0.5,
            "sharpe_mean": sum(sharpes) / len(sharpes),
            "sharpe_std": (sum((x - sum(sharpes) / len(sharpes)) ** 2 for x in sharpes) / max(len(sharpes) - 1, 1)) ** 0.5,
        },
        "config_hash": config_hash({"stocks": N, "universe": args.universe, **cfg}),
        "time_seconds": round(time.time() - t0, 1),
    }
    out_path = next_exp_path(OUTPUT_DIR, "daft-walk-forward")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n    报告 → {out_path}")


if __name__ == "__main__":
    main()
