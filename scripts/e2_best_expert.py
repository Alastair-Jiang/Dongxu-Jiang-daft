"""E2 — 最佳单专家基线（K3 计划第二步，不重训）。

对每个专家 + ensemble，用 test 段信号跑完整回测（5bp+1bp, top20%, 多空），
回答 K3 的问题：
  若单专家（尤其 e8_momentum IC 0.0355）Sharpe 转正，而路由版 ensemble -2.44，
  → 路由是净负贡献，MoE 路由层需要改或删。

对照：
  Ridge 基线：Sharpe +0.555
  ensemble（方案 A full）：Sharpe -2.4443

用法：D:\\env\\python.exe scripts\\e2_best_expert.py
"""

from __future__ import annotations
import json
import sys
import time
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from daft.data.adapters.baostock_adapter import BaostockAdapter
from daft.features.regime_features import RegimeFeatureExtractor
from daft.models.factory import build_experts, build_ensemble, build_layer_proj
from daft.training.joint_trainer import JointTrainer
from daft.backtest.engine import BacktestEngine
from daft.utils.device import get_device

CKPT_DIR = PROJECT_ROOT / "checkpoints" / "oos"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

EXPERT_LABELS = [
    "e0_trend", "e1_trend",
    "e2_reversal", "e3_reversal",
    "e4_volatility", "e5_volatility",
    "e6_event", "e7_event",
    "e8_momentum", "e9_momentum",
]


def compute_normalization(panel, extractor):
    with torch.no_grad():
        s_raw = extractor(panel)
    s_raw = torch.nan_to_num(s_raw, nan=0.0, posinf=1e6, neginf=-1e6).clamp(-1e6, 1e6)
    flat = s_raw.reshape(-1, 200)
    mu = flat.mean(dim=0, keepdim=True)
    sd = flat.std(dim=0, keepdim=True).clamp(min=1e-4)
    return mu, sd


def main():
    t_total = time.time()
    device = get_device()
    print(f"Device: {device}")

    print("[1/3] 数据 + 模型...")
    adapter = BaostockAdapter({
        "start_date": "2021-01-01", "end_date": "2025-12-31",
        "frequency": "d", "n_stocks": 100,
        "universe": "hs300", "adjust": "2",
    })
    panel = adapter.load()
    T, N, F = panel.shape
    t_train_end = int(T * 0.6)
    t_val_end = int(T * 0.8)
    train_panel = panel.slice_time(0, t_train_end)

    extractor = RegimeFeatureExtractor(n_base_factors=50, output_dim=200)
    mu, sd = compute_normalization(train_panel, extractor)
    with torch.no_grad():
        s_raw = extractor(panel)
    s_raw = torch.nan_to_num(s_raw, nan=0.0, posinf=1e6, neginf=-1e6).clamp(-1e6, 1e6)
    s_t = ((s_raw - mu) / sd).clamp(-10.0, 10.0)

    experts = build_experts()
    model = build_ensemble(experts, cdap_strength=0.1)
    layer_proj = build_layer_proj()
    JointTrainer.load_checkpoints(model, layer_proj, str(CKPT_DIR))
    model.cross_dim_attn.modulation_strength = 1.0
    model.router.temperature = 0.1
    model.to(device)
    layer_proj.to(device)
    model.eval()
    layer_proj.eval()
    n_experts = model.n_experts

    print("[2/3] causal 遍历收集逐专家 test 段信号...")
    expert_signals = torch.zeros(T - 1, N, n_experts)
    final_signal = torch.zeros(T - 1, N)
    model.memory.reset_state(1, device)
    with torch.no_grad():
        for t in range(T - 1):
            s_b = s_t[t].to(device)
            if model.memory.M is None or model.memory.M.size(0) != N:
                model.memory.reset_state(N, device)
            for i, expert in enumerate(model.experts):
                expert_signals[t, :, i] = expert(s_b).squeeze(-1).cpu()
            l0 = layer_proj["l0"](s_b)
            l1 = layer_proj["l1"](s_b)
            l2 = layer_proj["l2"](s_b)
            outputs = model(s_b, [l0, l1, l2], mode="inference",
                            mask=panel.mask[t].to(device))  # A3: 记忆行语义对齐
            final_signal[t] = outputs["signal"].squeeze(-1).cpu()
            model.memory.detach_state()

    print("[3/3] test 段回测（5bp+1bp, top20%, 多空）...")
    engine = BacktestEngine({
        "transaction_cost_bps": 5.0, "slippage_bps": 1.0,
        "top_quantile": 0.2, "long_only": False,
    })
    prices_test = panel.values[t_val_end:, :, 3]          # (T-t_val_end, N)
    mask_test = panel.mask[t_val_end:]

    def backtest(sig_series):
        # sig_series: (T-1-t_val_end, N) → 补哑元对齐 prices
        sig_padded = torch.cat([sig_series, torch.zeros(1, N)], dim=0)
        return engine.run(sig_padded, prices_test, mask=mask_test)

    print("\n" + "=" * 78)
    print("E2 · 单专家 vs ensemble 样本外回测（test 段）")
    print("=" * 78)
    print(f"{'model':<18}{'Sharpe':>9}{'ann_ret':>11}{'max_dd':>9}{'turnover':>10}")
    results = {}
    for i in range(n_experts):
        bt = backtest(expert_signals[t_val_end:, :, i])
        results[EXPERT_LABELS[i]] = {
            k: (v if isinstance(v, float) else None) for k, v in bt.items()
        }
        print(f"{EXPERT_LABELS[i]:<18}{bt.get('sharpe_ratio', float('nan')):>+9.3f}"
              f"{bt.get('annual_return', float('nan')):>+11.4f}"
              f"{bt.get('max_drawdown', float('nan')):>+9.4f}"
              f"{bt.get('turnover', float('nan')):>10.4f}")

    bt_ens = backtest(final_signal[t_val_end:])
    results["ensemble"] = {
        k: (v if isinstance(v, float) else None) for k, v in bt_ens.items()
    }
    print(f"{'ensemble(路由版)':<18}{bt_ens.get('sharpe_ratio', float('nan')):>+9.3f}"
          f"{bt_ens.get('annual_return', float('nan')):>+11.4f}"
          f"{bt_ens.get('max_drawdown', float('nan')):>+9.4f}"
          f"{bt_ens.get('turnover', float('nan')):>10.4f}")
    print(f"{'Ridge(基线)':<18}{'+0.555':>9}{'(已知)':>11}")

    report = {
        "experiment": "E2_best_expert_baseline",
        "checkpoint": str(CKPT_DIR),
        "device": str(device),
        "panel": {"T": T, "N": N, "split": {"train": t_train_end, "val": t_val_end, "test": T}},
        "per_expert_backtest": results,
        "ensemble_backtest": results["ensemble"],
        "ridge_baseline": {"sharpe_ratio": 0.555},
        "time_seconds": round(time.time() - t_total, 1),
    }
    OUTPUT_DIR.mkdir(exist_ok=True)
    out = OUTPUT_DIR / "e2_best_expert.json"
    with open(out, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n报告 → {out}")


if __name__ == "__main__":
    main()
