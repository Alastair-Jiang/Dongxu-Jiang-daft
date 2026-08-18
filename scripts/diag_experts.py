"""E1 — 专家个体诊断（K3 计划第一步，不重训）。

加载方案 A full 训练的 checkpoint（checkpoints/oos），在 val + test 段：
  1. 逐专家 Rank IC（mean / ICIR / t-stat / 正比例）
  2. 路由权重分布：router 原始 full_probs vs CDAP 调制后的 final_routing
  3. 专家间相关矩阵（10×10 Pearson，样本外 test 段）
  4. final ensemble 信号 IC（对照，应复现 ≈ 方案 A full 的 +0.0066）

回答 K3 的核心三问：
  (a) 信号集中在哪几个专家？
  (b) balance KL 是否把权重摊到了零信号专家上？
  (c) 10 个专家是否同质（|ρ| 均值 > 0.7 即同质）？

用法：D:\\env\\python.exe scripts\\diag_experts.py
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
from daft.utils.metrics import rank_info_coefficient, ic_summary, eligible_mask
from daft.utils.device import get_device

CKPT_DIR = PROJECT_ROOT / "checkpoints" / "oos"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

# 与 factory.build_experts() 的顺序一致（5 类 × 2 实例）
EXPERT_LABELS = [
    "e0_trend", "e1_trend",
    "e2_reversal", "e3_reversal",
    "e4_volatility", "e5_volatility",
    "e6_event", "e7_event",
    "e8_momentum", "e9_momentum",
]


def compute_normalization(panel, extractor):
    """标准化统计量仅来自 train 段（与 pipeline 完全一致）。"""
    with torch.no_grad():
        s_raw = extractor(panel)
    s_raw = torch.nan_to_num(s_raw, nan=0.0, posinf=1e6, neginf=-1e6).clamp(-1e6, 1e6)
    flat = s_raw.reshape(-1, 200)
    mu = flat.mean(dim=0, keepdim=True)
    sd = flat.std(dim=0, keepdim=True).clamp(min=1e-4)
    return mu, sd


def corr_matrix(X: torch.Tensor) -> torch.Tensor:
    """X: (K, n_experts) → (n_experts, n_experts) Pearson 相关矩阵。"""
    Xc = X - X.mean(dim=0, keepdim=True)
    Xs = Xc / Xc.std(dim=0, keepdim=True).clamp(min=1e-8)
    return (Xs.T @ Xs) / Xs.shape[0]


def main():
    t_total = time.time()
    device = get_device()
    print(f"Device: {device}")

    # ---------- 1. 数据（与 full 训练同口径） ----------
    print("\n[1/4] 拉取真实数据 (baostock, hs300 100 股)...")
    adapter = BaostockAdapter({
        "start_date": "2021-01-01", "end_date": "2025-12-31",
        "frequency": "d", "n_stocks": 100,
        "universe": "hs300", "adjust": "2",
    })
    panel = adapter.load()
    T, N, F = panel.shape
    print(f"      Panel: (T={T}, N={N}, F={F})  {panel.dates[0]} → {panel.dates[-1]}")

    # ---------- 2. 严格切分 ----------
    t_train_end = int(T * 0.6)
    t_val_end = int(T * 0.8)
    train_panel = panel.slice_time(0, t_train_end)
    print(f"      train 0→{t_train_end}  val {t_train_end}→{t_val_end}  test {t_val_end}→{T}")

    # ---------- 3. 标准化（只 train）+ 模型加载 ----------
    extractor = RegimeFeatureExtractor(n_base_factors=50, output_dim=200)
    mu, sd = compute_normalization(train_panel, extractor)
    with torch.no_grad():
        s_raw = extractor(panel)
    s_raw = torch.nan_to_num(s_raw, nan=0.0, posinf=1e6, neginf=-1e6).clamp(-1e6, 1e6)
    s_t = ((s_raw - mu) / sd).clamp(-10.0, 10.0)  # (T, N, 200)

    print("[2/4] 构建模型 + 加载 checkpoint...")
    experts = build_experts()
    model = build_ensemble(experts, cdap_strength=0.1)
    layer_proj = build_layer_proj()
    JointTrainer.load_checkpoints(model, layer_proj, str(CKPT_DIR))
    # 复现 Stage 3 后的推理态（这两个是普通属性，不进 state_dict）
    model.cross_dim_attn.modulation_strength = 1.0
    model.router.temperature = 0.1
    model.to(device)
    layer_proj.to(device)
    model.eval()
    layer_proj.eval()

    n_experts = model.n_experts

    # ---------- 4. causal 逐时间步遍历，收集 ----------
    print("[3/4] causal 遍历 full panel（含 memory warmup）...")
    log_p = torch.log(panel.values[..., 3].clamp(min=1e-8))
    returns = (log_p[1:] - log_p[:-1])            # (T-1, N), returns[t]=p[t+1]-p[t]
    # A4 (2026-08-18): 双条件入样 mask[t]&mask[t+1], 与对决主口径统一
    # (原"未来端 mask"单条件会把信号日涨停样本的次日收益计入 IC)
    mask_ic = eligible_mask(panel.mask)           # (T-1, N)

    expert_signals = torch.zeros(T - 1, N, n_experts)
    full_probs = torch.zeros(T - 1, N, n_experts)
    final_routing = torch.zeros(T - 1, N, n_experts)
    final_signal = torch.zeros(T - 1, N)

    model.memory.reset_state(1, device)
    with torch.no_grad():
        for t in range(T - 1):
            s_b = s_t[t].to(device)  # (N, 200)
            if model.memory.M is None or model.memory.M.size(0) != N:
                model.memory.reset_state(N, device)

            # 逐专家独立前向（只依赖 s_t）
            for i, expert in enumerate(model.experts):
                expert_signals[t, :, i] = expert(s_b).squeeze(-1).cpu()

            # router 原始路由分布（未过 CDAP）
            _, _, _, fp = model.router(s_b, mode="inference")
            full_probs[t] = fp.cpu()

            # ensemble 前向（拿 CDAP 调制后的 final_routing + final signal）
            l0 = layer_proj["l0"](s_b)
            l1 = layer_proj["l1"](s_b)
            l2 = layer_proj["l2"](s_b)
            outputs = model(s_b, [l0, l1, l2], mode="inference",
                            mask=panel.mask[t].to(device))  # A3: 记忆行语义对齐
            final_routing[t] = outputs["routing_probs"].cpu()
            final_signal[t] = outputs["signal"].squeeze(-1).cpu()

            model.memory.detach_state()

    val_slice = slice(t_train_end, t_val_end)
    test_slice = slice(t_val_end, T - 1)

    # ---------- 报告 ----------
    print("\n[4/4] 汇总诊断结果\n")
    print("=" * 78)
    print("E1 · 逐专家样本外诊断")
    print("=" * 78)

    # (1) 逐专家 IC
    print("\n── 逐专家 Rank IC ───────────────────────────────────────")
    print(f"{'expert':<16}{'val IC':>9}{'val ICIR':>10}{'test IC':>9}{'test ICIR':>11}{'test t':>8}{'IC>0%':>8}")
    expert_ic = {}
    for i in range(n_experts):
        ic_v = rank_info_coefficient(
            expert_signals[val_slice, :, i], returns[val_slice], mask_ic[val_slice],
            per_timestep=True)
        ic_t = rank_info_coefficient(
            expert_signals[test_slice, :, i], returns[test_slice], mask_ic[test_slice],
            per_timestep=True)
        sv = ic_summary(ic_v)
        st = ic_summary(ic_t)
        expert_ic[EXPERT_LABELS[i]] = {"val": sv, "test": st}
        print(f"{EXPERT_LABELS[i]:<16}{sv['ic_mean']:>+9.4f}{sv['icir']:>+10.3f}"
              f"{st['ic_mean']:>+9.4f}{st['icir']:>+11.3f}{st['ic_t_stat']:>+8.2f}"
              f"{st['ic_positive_ratio']:>8.1%}")

    # final ensemble 对照
    ic_fin = rank_info_coefficient(
        final_signal[test_slice], returns[test_slice], mask_ic[test_slice],
        per_timestep=True)
    sf = ic_summary(ic_fin)
    print(f"{'[ensemble]':<16}{'':>9}{'':>10}{sf['ic_mean']:>+9.4f}{sf['icir']:>+11.3f}"
          f"{sf['ic_t_stat']:>+8.2f}{sf['ic_positive_ratio']:>8.1%}   ← 应 ≈ +0.0066")

    # (2) 路由权重分布
    print("\n── 路由权重分布（test 段均值，10 专家）──────────────────")
    fp_mean = full_probs[test_slice].reshape(-1, n_experts).mean(0)
    fr_mean = final_routing[test_slice].reshape(-1, n_experts).mean(0)
    print(f"{'expert':<16}{'full_probs':>12}{'final_routing':>15}")
    for i in range(n_experts):
        print(f"{EXPERT_LABELS[i]:<16}{fp_mean[i].item():>12.4f}{fr_mean[i].item():>15.4f}")

    # (3) 相关矩阵
    X = expert_signals[test_slice].reshape(-1, n_experts)
    m = mask_ic[test_slice].reshape(-1)
    Xv = X[m]
    C = corr_matrix(Xv)
    offdiag = C - torch.eye(n_experts)
    print("\n── 专家间相关矩阵（test 段，Pearson）───────────────────")
    header = " " * 16 + "".join(f"{i:>7d}" for i in range(n_experts))
    print(header)
    for i in range(n_experts):
        row = "".join(f"{C[i, j].item():>+7.2f}" for j in range(n_experts))
        print(f"{EXPERT_LABELS[i]:<16}{row}")
    print(f"\n  非对角 |ρ| 均值 = {offdiag.abs().mean().item():.3f}  "
          f"（>0.7 判同质）")

    # ---------- 保存 ----------
    report = {
        "experiment": "E1_expert_diagnosis",
        "checkpoint": str(CKPT_DIR),
        "device": str(device),
        "panel": {"T": T, "N": N, "split": {"train": t_train_end, "val": t_val_end, "test": T}},
        "per_expert_ic": expert_ic,
        "ensemble_test_ic": sf,
        "routing": {
            "full_probs_mean": fp_mean.tolist(),
            "final_routing_mean": fr_mean.tolist(),
        },
        "correlation": C.tolist(),
        "corr_offdiag_abs_mean": offdiag.abs().mean().item(),
        "expert_labels": EXPERT_LABELS,
        "time_seconds": round(time.time() - t_total, 1),
    }
    OUTPUT_DIR.mkdir(exist_ok=True)
    out = OUTPUT_DIR / "diag_experts_e1.json"
    with open(out, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n报告 → {out}")


if __name__ == "__main__":
    main()
