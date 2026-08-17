"""DAFT Full Pipeline — OUT-OF-SAMPLE version on REAL A-share data.

业界阶段 3 的核心改造:把 DAFT 主管道塞进与 Ridge baseline 完全相同的
样本外框架,实现公平对决。

与 run_full_pipeline.py 的区别(关键):
  1. 数据:baostock 真实 A 股(不再是合成)
  2. 切分:train 60% / val 20% / test 20%(时间顺序,严格样本外)
  3. 训练:Stage 1/2/3 只接触 train(+val 早停),绝不接触 test
  4. 回测:只在 test 段,同样的成本模型(5bp + 1bp, top20%, 多空)
  5. 记忆预热:冻结模型先在 train+val 段推进记忆状态,test 段从热状态开始

输出:
  outputs/full_pipeline_oos_report.json  — DAFT 真实数据样本外报告
  (与 outputs/baseline_ridge_real_report.json 同口径,可直接对比)

用法:
  python scripts/run_full_pipeline_oos.py [--stocks 30] [--start 2021-01-01] [--end 2025-12-31] [--quick|--full]
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

from daft.data.adapters.baostock_adapter import BaostockAdapter
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
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints" / "oos"


def compute_normalization(panel, extractor):
    """Compute s_t stats from a panel (used ONLY on the train segment)."""
    with torch.no_grad():
        s_raw = extractor(panel)
    s_raw = torch.nan_to_num(s_raw, nan=0.0, posinf=1e6, neginf=-1e6).clamp(-1e6, 1e6)
    flat = s_raw.reshape(-1, 200)
    mu = flat.mean(dim=0, keepdim=True)
    sd = flat.std(dim=0, keepdim=True).clamp(min=1e-4)
    return mu, sd


def generate_oos_signals(model, layer_proj, panel, mu, sd, device):
    """Generate signals for the whole panel using frozen (train-fit) stats.

    Memory is warmed up on the panel sequentially (causal: state at t only
    depends on t and earlier), so the test segment starts from a warm state.
    """
    extractor = RegimeFeatureExtractor(n_base_factors=50, output_dim=200)
    with torch.no_grad():
        s_t_raw = extractor(panel)
    s_t_raw = torch.nan_to_num(s_t_raw, nan=0.0, posinf=1e6, neginf=-1e6).clamp(-1e6, 1e6)
    s_t = ((s_t_raw - mu) / sd).clamp(-10.0, 10.0)

    T, N, _ = s_t.shape
    model.eval()
    layer_proj.eval()
    model.memory.reset_state(1, device)

    signals = torch.zeros(T - 1, N)
    with torch.no_grad():
        for t in range(T - 1):
            s_b = s_t[t].to(device)
            if model.memory.M is None or model.memory.M.size(0) != N:
                model.memory.reset_state(N, device)
            l0 = layer_proj["l0"](s_b)
            l1 = layer_proj["l1"](s_b)
            l2 = layer_proj["l2"](s_b)
            outputs = model(s_b, [l0, l1, l2], mode="inference")
            signals[t] = outputs["signal"].squeeze(-1).cpu()
            model.memory.detach_state()
    return signals


def main():
    parser = argparse.ArgumentParser(description="DAFT OOS pipeline on real A-shares")
    parser.add_argument("--stocks", type=int, default=30)
    parser.add_argument("--start", default="2021-01-01")
    parser.add_argument("--end", default="2025-12-31")
    parser.add_argument("--full", action="store_true", help="full training config")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--universe", default="hs300", choices=["hs300", "sample"],
                        help="股票池(默认 hs300 真实成分)")
    parser.add_argument("--ablate", default="none",
                        choices=["none", "cdap", "memory", "router"],
                        help="消融开关(研究项目): 关闭 CDAP/记忆/路由")
    parser.add_argument("--ckpt-dir", default="checkpoints/oos",
                        help="checkpoint 目录(默认 checkpoints/oos)")
    parser.add_argument("--hidden", type=int, default=64,
                        help="专家 hidden 维度(容量扫描, 默认 64)")
    parser.add_argument("--n-layers", type=int, default=2,
                        help="专家 MLP/Transformer 层数(容量扫描, 默认 2)")
    parser.add_argument("--arch", default="mlp", choices=["mlp", "transformer"],
                        help="专家主干架构(2026-08-17): mlp=5类regime专家, "
                             "transformer=特征自注意力专家")
    parser.add_argument("--n-heads", type=int, default=4,
                        help="Transformer 专家注意力头数(默认 4)")
    parser.add_argument("--no-regime", action="store_true",
                        help="Stage1 专家全量训练(README regime 专业化对照)")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    # 设备自动检测(2026-08-16 接线): CUDA → XPU(Intel Arc) → DirectML → MPS → CPU
    device = get_device()
    print(f"      Device: {device}")
    t_total = time.time()

    cfg = {
        "stage1": {"epochs": 15, "batch_size": 1024, "lr": 1e-3},
        "stage2": {"epochs": 10, "batch_size": 512, "lr": 1e-3},
        "stage3": {"epochs": 8,  "batch_size": 512, "lr": 1e-5},
    }
    if args.full:
        cfg = {
            "stage1": {"epochs": 50, "batch_size": 2048, "lr": 1e-3},
            "stage2": {"epochs": 30, "batch_size": 1024, "lr": 1e-3},
            "stage3": {"epochs": 20, "batch_size": 1024, "lr": 1e-5},
        }

    print("=" * 66)
    print("DAFT FULL PIPELINE — OUT-OF-SAMPLE · REAL A-SHARE DATA")
    print("=" * 66)

    # ---------- 1. Real data ----------
    print("\n[1/6] 拉取真实数据 (baostock)...")
    adapter = BaostockAdapter({
        "start_date": args.start, "end_date": args.end,
        "frequency": "d", "n_stocks": args.stocks,
        "universe": args.universe, "adjust": "2",
    })
    panel = adapter.load()
    T, N, F = panel.shape
    print(f"      Panel: (T={T}, N={N}, F={F})  {panel.dates[0]} → {panel.dates[-1]}")

    # ---------- 2. Strict time split: train 60% / val 20% / test 20% ----------
    t_train_end = int(T * 0.6)
    t_val_end = int(T * 0.8)
    train_panel = panel.slice_time(0, t_train_end)
    val_panel = panel.slice_time(t_train_end, t_val_end)
    test_panel = panel.slice_time(t_val_end, T)
    print(f"\n[2/6] 严格样本外切分:")
    print(f"      train: 0 → {t_train_end} ({train_panel.T} 步)")
    print(f"      val  : {t_train_end} → {t_val_end} ({val_panel.T} 步)")
    print(f"      test : {t_val_end} → {T} ({test_panel.T} 步)  ← 模型训练全程不可见")

    # ---------- 3. Stage 1: experts (train only) ----------
    print("\n[3/6] Stage 1: 独立专家训练 (仅 train 段)...")
    experts = build_experts(hidden=args.hidden, n_layers=args.n_layers,
                            arch=args.arch, n_heads=args.n_heads)
    s1 = Stage1ExpertTrainer(experts=experts, panel=train_panel, device=device)
    t0 = time.time()
    s1_hist = s1.train_all(epochs=cfg["stage1"]["epochs"],
                           batch_size=cfg["stage1"]["batch_size"],
                           lr=cfg["stage1"]["lr"], verbose=False,
                           use_regime=not args.no_regime)
    stage1_seconds = time.time() - t0
    print(f"      Stage 1 耗时: {stage1_seconds:.1f}s")

    # ---------- 4. Stage 2 + 3 (train + val only) ----------
    print("\n[4/6] Stage 2 + 3: 路由/记忆/CDAP + 联合微调 (仅 train+val 段)...")
    model = build_ensemble(experts, cdap_strength=0.1, ablate=args.ablate)
    layer_proj = build_layer_proj()

    s2 = RouterTrainer(model=model, config=cfg["stage2"], device=device)
    t0 = time.time()
    s2_hist = s2.train(train_panel, val_panel)
    print(f"      Stage 2 耗时: {time.time()-t0:.1f}s")

    model.cross_dim_attn.modulation_strength = 1.0
    model.router.temperature = 0.1
    s3 = JointTrainer(model=model, layer_proj=layer_proj, config=cfg["stage3"], device=device)
    t0 = time.time()
    s3_hist = s3.train(train_panel, val_panel)
    print(f"      Stage 3 耗时: {time.time()-t0:.1f}s")

    # Save checkpoints
    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    s3.save_checkpoints(str(ckpt_dir))
    print(f"      Checkpoints → {ckpt_dir}")

    # ---------- 5. OOS signals: normalization from TRAIN only ----------
    print("\n[5/6] 样本外信号生成 (标准化统计量仅来自 train 段)...")
    extractor = RegimeFeatureExtractor(n_base_factors=50, output_dim=200)
    mu, sd = compute_normalization(train_panel, extractor)
    signals_full = generate_oos_signals(model, layer_proj, panel, mu, sd, device)
    print(f"      信号: {signals_full.shape}")

    # Test-segment signals & prices
    # 对齐约定(2026-08-16 统一为 k→k+1): signals_full[t] 预测 p[t+1]-p[t]。
    # 配对 (signals_full[t], p[t+1]-p[t]) for t ∈ [t_val_end, T-2]。
    # engine.run 要求 len(signals)==len(prices), 因此末尾补一行哑元
    # (会被 signals[:-1] 丢弃)。
    signals_test = torch.cat([
        signals_full[t_val_end:],                         # (T-1-t_val_end, N)
        torch.zeros(1, N),                                # 哑元行(被丢弃)
    ], dim=0)                                             # (T-t_val_end, N)
    prices_test = panel.values[t_val_end:, :, 3]          # (T-t_val_end, N)
    mask_test = panel.mask[t_val_end:]                    # (T-t_val_end, N)
    print(f"      test 段信号: {signals_test.shape}, prices: {prices_test.shape}")

    # ---------- 6. Backtest on TEST ONLY (same costs as Ridge) ----------
    print("\n[6/6] 样本外回测 (仅 test 段, 成本 5bp+1bp, top20%, 多空)...")
    engine = BacktestEngine({
        "transaction_cost_bps": 5.0, "slippage_bps": 1.0,
        "top_quantile": 0.2, "long_only": False,
    })
    bt = engine.run(signals_test, prices_test, mask=mask_test)

    # IC on test segment (cross-sectional, per-timestep)
    # 对齐(2026-08-16 统一 k→k+1): returns_test[t] = p[t_val_end+t+1]-p[t_val_end+t],
    # signals_test[:-1][t] = signals_full[t_val_end+t] 恰好预测该收益。
    log_pt = torch.log(prices_test.clamp(min=1e-8))
    returns_test = (log_pt[1:] - log_pt[:-1])             # (T_test-1, N)
    ic_series = rank_info_coefficient(signals_test[:-1], returns_test, mask_test[1:], per_timestep=True)
    ic_stats = ic_summary(ic_series)
    hit = hit_rate(signals_test[:-1], returns_test, mask_test[1:])

    print(f"\n  ── DAFT · 真实数据 · 样本外结果 ──")
    print(f"    Rank IC     : {ic_stats['ic_mean']:+.4f}")
    print(f"    ICIR        : {ic_stats['icir']:+.3f}")
    print(f"    IC t-stat   : {ic_stats['ic_t_stat']:+.2f}")
    print(f"    IC>0 比例   : {ic_stats['ic_positive_ratio']:.1%}")
    print(f"    Hit rate    : {hit:.3f}")
    print(f"\n  ── 样本外回测 (扣费后) ──")
    for k, v in bt.items():
        if isinstance(v, float):
            print(f"    {k:<22s}: {v:+.4f}")

    # ---------- Save ----------
    report = {
        "experiment_id": None,  # 下方写入
        "model": "DAFT_full_pipeline",
        "ablate": args.ablate,
        "seed": args.seed,
        "hidden": args.hidden,
        "n_layers": args.n_layers,
        "arch": args.arch,
        "n_heads": args.n_heads if args.arch == "transformer" else None,
        "stage1_regime": not args.no_regime,
        "alignment": "k→k+1 (signal[t] 预测 p[t+1]-p[t], 2026-08-16 统一)",
        "data": {
            "source": "baostock", "stocks": N, "tickers": panel.asset_ids,
            "start": args.start, "end": args.end, "frequency": "daily",
            "adjust": "forward", "T": T,
        },
        "split": {"train": t_train_end, "val": t_val_end, "test": T,
                  "test_steps": test_panel.T},
        "config": cfg,
        "config_hash": config_hash(cfg),
        "training": {
            "stage1_seconds": round(stage1_seconds, 1),
            "experts_trained": len(s1_hist),
        },
        "out_of_sample": {
            "ic_mean": ic_stats["ic_mean"], "ic_std": ic_stats["ic_std"],
            "icir": ic_stats["icir"], "ic_t_stat": ic_stats["ic_t_stat"],
            "ic_positive_ratio": ic_stats["ic_positive_ratio"], "hit_rate": hit,
        },
        "backtest": bt,
        "normalization": "train-segment stats only",
        "memory_warmup": "causal sequential warmup over full panel before test",
        "time_seconds": round(time.time() - t_total, 1),
    }
    # 唯一产物名 (2026-08-16): 不再用固定文件名覆盖历史报告
    out_path = next_exp_path(OUTPUT_DIR, "daft-oos")
    report["experiment_id"] = out_path.stem
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n    报告 → {out_path}")
    print("    对比基准 → outputs/EXP-*-ridge-real.json (最新一条 Ridge 同口径报告)")


if __name__ == "__main__":
    main()
