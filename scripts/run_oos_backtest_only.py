"""DAFT OOS — 仅信号生成 + 回测(复用已有 checkpoint,不重训)。

用法: python scripts/run_oos_backtest_only.py [--stocks 20]
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from daft.data.adapters.baostock_adapter import BaostockAdapter
from daft.features.regime_features import RegimeFeatureExtractor
from daft.models.factory import build_model as _build_model
from daft.training.joint_trainer import JointTrainer
from daft.backtest.engine import BacktestEngine
from daft.utils.metrics import rank_info_coefficient, ic_summary, hit_rate
from daft.utils.experiment import next_exp_path

OUTPUT_DIR = PROJECT_ROOT / "outputs"


def build_model():
    """推理用: 10 专家 + 温度 0.1(与 OOS checkpoint 对齐)。"""
    return _build_model(
        cdap_strength=1.0, router_temperature=0.1, noisy_gating_std=0.0,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stocks", type=int, default=20)
    parser.add_argument("--start", default="2021-01-01")
    parser.add_argument("--end", default="2025-12-31")
    parser.add_argument("--universe", default="hs300", choices=["hs300", "sample"])
    args = parser.parse_args()

    t0 = time.time()
    print("=== DAFT OOS backtest-only (复用 checkpoint) ===")

    # 数据
    panel = BaostockAdapter({"start_date":args.start,"end_date":args.end,
                             "frequency":"d","n_stocks":args.stocks,
                             "universe":args.universe,"adjust":"2"}).load()
    T, N, F = panel.shape
    t_train_end, t_val_end = int(T*0.6), int(T*0.8)
    print(f"Panel: (T={T}, N={N})  test 段: [{t_val_end}, {T}) = {T-t_val_end} 步")

    # 模型 + checkpoint
    model, layer_proj = build_model()
    JointTrainer.load_checkpoints(model, layer_proj, str(PROJECT_ROOT/"checkpoints"/"oos"))
    model.eval(); layer_proj.eval()
    print("Checkpoints 已加载")

    # 标准化(仅 train 段)
    ext = RegimeFeatureExtractor(n_base_factors=50, output_dim=200)
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
            out = model(s_b, [l0,l1,l2], mode="inference",
                        mask=panel.mask[t].to(device))  # A3: 记忆行语义对齐
            signals[t] = out["signal"].squeeze(-1).cpu()
            model.memory.detach_state()
    print(f"信号: {signals.shape}")

    # 对齐切片 (2026-08-16 统一 k→k+1, 末尾补哑元行)
    signals_test = torch.cat([signals[t_val_end:], torch.zeros(1, N)], dim=0)  # (T-t_val_end, N)
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
        "alignment": "k→k+1 (2026-08-16 统一)",
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
    # 唯一产物名 (2026-08-16): 不再覆盖 run_full_pipeline_oos 的报告
    out_path = next_exp_path(OUTPUT_DIR, "daft-oos-backtest-only")
    report["experiment_id"] = out_path.stem
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n    报告 → {out_path}")


if __name__ == "__main__":
    main()
