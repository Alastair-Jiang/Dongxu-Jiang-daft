"""EXP-20260809-02: 信号平滑对比 — DAFT OOS checkpoint 复用 + EMA 平滑扫描。

目的: 验证信号 EMA 平滑能否压低换手、改善净 Sharpe(Kimi K3 评审建议)。

用法: python scripts/run_smoothing_ablation.py [--stocks 30] [--lambdas 0,0.3,0.5,0.7]

前置: checkpoints/oos/ 必须存在(EXP-20260809-01 产物)
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import torch, torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from daft.data.adapters.baostock_adapter import BaostockAdapter
from daft.features.regime_features import RegimeFeatureExtractor
from daft.models.experts import TrendExpert, ReversalExpert, VolatilityExpert, EventExpert, MomentumExpert
from daft.models.router import RegimeRouter
from daft.models.memory import KDAMarketMemory
from daft.models.cross_dim_attn import CrossDimensionAttention
from daft.models.hardening import HardeningEngine
from daft.models.ensemble import ExpertEnsemble
from daft.training.joint_trainer import JointTrainer
from daft.backtest.engine import BacktestEngine
from daft.utils.metrics import rank_info_coefficient, ic_summary, hit_rate

OUTPUT_DIR = PROJECT_ROOT / "outputs"


def build_model():
    experts = nn.ModuleList([
        TrendExpert(200,64), TrendExpert(200,64),
        ReversalExpert(200,64), ReversalExpert(200,64),
        VolatilityExpert(200,48), VolatilityExpert(200,48),
        EventExpert(200,48), EventExpert(200,48),
        MomentumExpert(200,64), MomentumExpert(200,64),
    ])
    model = ExpertEnsemble(experts,
        RegimeRouter(200,16,10,3,temperature=0.1,noisy_gating_std=0.0),
        KDAMarketMemory(128,64,200,bottleneck_ratio=4,use_route_modulation=True),
        CrossDimensionAttention(10,128,64,3,64,modulation_strength=1.0),
        HardeningEngine(10,10))
    layer_proj = nn.ModuleDict({
        "l0": nn.Sequential(nn.Linear(200,128),nn.SiLU(),nn.Linear(128,64),nn.LayerNorm(64)),
        "l1": nn.Sequential(nn.Linear(200,128),nn.SiLU(),nn.Linear(128,64),nn.LayerNorm(64)),
        "l2": nn.Sequential(nn.Linear(200,128),nn.SiLU(),nn.Linear(128,64),nn.LayerNorm(64)),
    })
    return model, layer_proj


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stocks", type=int, default=30)
    parser.add_argument("--lambdas", default="0,0.3,0.5,0.7")
    parser.add_argument("--start", default="2021-01-01")
    parser.add_argument("--end", default="2025-12-31")
    args = parser.parse_args()
    lambdas = [float(x) for x in args.lambdas.split(",")]

    t0 = time.time()
    print("=== EXP-20260809-02: 信号平滑对比 ===")

    panel = BaostockAdapter({"start_date":args.start,"end_date":args.end,
                             "frequency":"d","n_stocks":args.stocks,"adjust":"2"}).load()
    T, N, F = panel.shape
    t_train_end, t_val_end = int(T*0.6), int(T*0.8)
    print(f"Panel: (T={T}, N={N})")

    model, layer_proj = build_model()
    JointTrainer.load_checkpoints(model, layer_proj, str(PROJECT_ROOT/"checkpoints"/"oos"))
    model.eval(); layer_proj.eval()
    print("Checkpoints 已加载 (EXP-20260809-01 产物)")

    # 标准化(仅 train)
    ext = RegimeFeatureExtractor(n_base_factors=50, output_dim=200)
    with torch.no_grad():
        s_tr = ext(panel.slice_time(0, t_train_end))
    s_tr = torch.nan_to_num(s_tr, nan=0.0, posinf=1e6, neginf=-1e6).clamp(-1e6,1e6)
    mu = s_tr.reshape(-1,200).mean(0,keepdim=True)
    sd = s_tr.reshape(-1,200).std(0,keepdim=True).clamp(min=1e-4)

    # 全量特征 + 因果信号生成
    print("生成信号(逐时步,记忆因果推进)...")
    with torch.no_grad():
        s_all = ext(panel)
    s_all = torch.nan_to_num(s_all, nan=0.0, posinf=1e6, neginf=-1e6).clamp(-1e6,1e6)
    s_all = ((s_all - mu) / sd).clamp(-10.0, 10.0)

    device = torch.device("cpu")
    model.memory.reset_state(1, device)
    signals = torch.zeros(T-1, N)
    with torch.no_grad():
        for t in range(T-1):
            s_b = s_all[t].to(device)
            if model.memory.M is None or model.memory.M.size(0) != N:
                model.memory.reset_state(N, device)
            l0 = layer_proj["l0"](s_b); l1 = layer_proj["l1"](s_b); l2 = layer_proj["l2"](s_b)
            out = model(s_b, [l0,l1,l2], mode="inference")
            signals[t] = out["signal"].squeeze(-1).cpu()
            model.memory.detach_state()
    print(f"信号: {signals.shape}")

    # test 段切片
    signals_test = signals[t_val_end-1:]
    prices_test = panel.values[t_val_end:, :, 3]
    mask_test = panel.mask[t_val_end:]

    # 每个 λ 跑一次回测
    results = {}
    for lam in lambdas:
        engine = BacktestEngine({
            "transaction_cost_bps":5.0, "slippage_bps":1.0,
            "top_quantile":0.2, "long_only":False,
            "signal_smoothing": lam,
        })
        bt = engine.run(signals_test, prices_test, mask=mask_test)

        log_pt = torch.log(prices_test.clamp(min=1e-8))
        returns_test = log_pt[1:] - log_pt[:-1]
        ic_series = rank_info_coefficient(signals_test[:-1], returns_test, mask_test[1:], per_timestep=True)
        ic = ic_summary(ic_series)

        results[f"lambda_{lam}"] = {
            "sharpe": bt["sharpe_ratio"],
            "annual_return": bt["annual_return"],
            "max_drawdown": bt["max_drawdown"],
            "turnover": bt["turnover"],
            "ic": ic["ic_mean"], "ic_t": ic["ic_t_stat"],
        }
        print(f"\n  λ={lam}: Sharpe={bt['sharpe_ratio']:+.4f}  "
              f"Turnover={bt['turnover']:.4f}  IC={ic['ic_mean']:+.4f}  "
              f"t={ic['ic_t_stat']:+.2f}")

    report = {
        "experiment": "EXP-20260809-02",
        "stocks": N, "lambdas": lambdas,
        "results": results,
        "time_seconds": round(time.time()-t0, 1),
    }
    out_path = OUTPUT_DIR / "smoothing_ablation.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n    报告 → {out_path}")


if __name__ == "__main__":
    main()
