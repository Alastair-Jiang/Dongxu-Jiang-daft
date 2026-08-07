"""DAFT OOS — 仅信号生成 + 回测(复用已有 checkpoint,不重训)。

用法: python scripts/run_oos_backtest_only.py [--stocks 20]
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import torch, torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from daft.data.adapters.baostock_adapter import BaostockAdapter
from daft.features.regime_features import RegimeFeatureExtractor
from daft.models.experts import TrendExpert, ReversalExpert, VolatilityExpert, EventExpert
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
    ])
    model = ExpertEnsemble(experts,
        RegimeRouter(200,16,8,3,temperature=0.1,noisy_gating_std=0.0),
        KDAMarketMemory(128,64,200,bottleneck_ratio=4,use_route_modulation=True),
        CrossDimensionAttention(8,128,64,3,64,modulation_strength=1.0),
        HardeningEngine(8,8))
    layer_proj = nn.ModuleDict({
        "l0": nn.Sequential(nn.Linear(200,128),nn.SiLU(),nn.Linear(128,64),nn.LayerNorm(64)),
        "l1": nn.Sequential(nn.Linear(200,128),nn.SiLU(),nn.Linear(128,64),nn.LayerNorm(64)),
        "l2": nn.Sequential(nn.Linear(200,128),nn.SiLU(),nn.Linear(128,64),nn.LayerNorm(64)),
    })
    return model, layer_proj


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stocks", type=int, default=20)
    parser.add_argument("--start", default="2021-01-01")
    parser.add_argument("--end", default="2025-12-31")
    args = parser.parse_args()

    t0 = time.time()
    print("=== DAFT OOS backtest-only (复用 checkpoint) ===")

    # 数据
    panel = BaostockAdapter({"start_date":args.start,"end_date":args.end,
                             "frequency":"d","n_stocks":args.stocks,"adjust":"2"}).load()
    T, N, F = panel.shape
    t_train_end, t_val_end = int(T*0.6), int(T*0.8)
    print(f"Panel: (T={T}, N={N})  test 段: [{t_val_end}, {T}) = {T-t_val_end} 步")

    # 模型 + checkpoint
    model, layer_proj = build_model()
    JointTrainer.load_checkpoints(model, layer_proj, str(PROJECT_ROOT/"checkpoints"/"oos"))
    model.eval(); layer_proj.eval()
    print("Checkpoints 已加载")

    # 标准化(仅 train 段)
    ext = RegimeFeatureExtractor(n_output=200)
    with torch.no_grad():
        s_tr = ext(panel.slice_time(0, t_train_end))
    s_tr = torch.nan_to_num(s_tr, nan=0.0, posinf=1e6, neginf=-1e6).clamp(-1e6,1e6)
    mu = s_tr.reshape(-1,200).mean(0,keepdim=True)
    sd = s_tr.reshape(-1,200).std(0,keepdim=True).clamp(min=1e-4)

    # 全量特征 + 因果信号生成(记忆预热覆盖全 panel)
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

    # 对齐切片
    signals_test = signals[t_val_end-1:]           # (T-t_val_end, N)
    prices_test = panel.values[t_val_end:, :, 3]
    mask_test = panel.mask[t_val_end:]
    print(f"test: signals={signals_test.shape}, prices={prices_test.shape}")

    # 回测
    engine = BacktestEngine({"transaction_cost_bps":5.0,"slippage_bps":1.0,
                             "top_quantile":0.2,"long_only":False})
    bt = engine.run(signals_test, prices_test, mask=mask_test)

    # IC
    log_pt = torch.log(prices_test.clamp(min=1e-8))
    returns_test = log_pt[1:] - log_pt[:-1]
    ic_series = rank_info_coefficient(signals_test[:-1], returns_test, mask_test[1:], per_timestep=True)
    ic = ic_summary(ic_series)
    hit = hit_rate(signals_test[:-1], returns_test, mask_test[1:])

    print(f"\n  ── DAFT · 真实数据 · 样本外(最终) ──")
    print(f"    Rank IC   : {ic['ic_mean']:+.4f}")
    print(f"    ICIR      : {ic['icir']:+.3f}")
    print(f"    IC t-stat : {ic['ic_t_stat']:+.2f}")
    print(f"    IC>0 比例 : {ic['ic_positive_ratio']:.1%}")
    print(f"    Hit rate  : {hit:.3f}")
    print(f"\n  ── 样本外回测(扣费后) ──")
    for k, v in bt.items():
        if isinstance(v, float): print(f"    {k:<22s}: {v:+.4f}")

    report = {
        "model": "DAFT_full_pipeline_OOS",
        "data": {"source":"baostock","stocks":N,"start":args.start,"end":args.end,
                 "frequency":"daily","adjust":"forward","T":T,
                 "split":{"train":t_train_end,"val":t_val_end,"test":T}},
        "out_of_sample": {
            "ic_mean": ic["ic_mean"], "ic_std": ic["ic_std"], "icir": ic["icir"],
            "ic_t_stat": ic["ic_t_stat"], "ic_positive_ratio": ic["ic_positive_ratio"],
            "hit_rate": hit,
        },
        "backtest": bt,
        "time_seconds": round(time.time()-t0, 1),
    }
    out_path = OUTPUT_DIR / "full_pipeline_oos_report.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n    报告 → {out_path}")


if __name__ == "__main__":
    main()
