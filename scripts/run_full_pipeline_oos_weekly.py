"""周线 DAFT 训练(方向 1 · 周线验证, 2026-08-17)。

K3 周线验证的执行脚本。与日线 run_full_pipeline_oos.py 同框架, 差异:
  - 数据: 日线 → resample_weekly(W-FRI), 不重新下载
  - 切分: 日历日期对齐日频(train≤60%日期 / val≤80%日期), OOS 一致
  - 特征: lookback_scale=0.2(窗口 ÷5)
  - Stage 2: balance_weight 可配(DAFT-A=0.01 / DAFT-noKL=0)
  - 输出: 逐周 test 段 IC 序列 + 周结束日期(供 #28 配对 ΔRankIC)

用法:
  D:\\env\\python.exe scripts\\run_full_pipeline_oos_weekly.py --balance-weight 0.01 --seed 42
  (DAFT-noKL: --balance-weight 0)

checkpoint 存 checkpoints/weekly/, 不覆盖日线 checkpoints/oos/。
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
from daft.data.resample import resample_weekly
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
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints" / "weekly"
LOOKBACK_SCALE = 0.2  # 窗口 ÷5


def main():
    parser = argparse.ArgumentParser(description="DAFT weekly OOS pipeline")
    parser.add_argument("--stocks", type=int, default=100)
    parser.add_argument("--start", default="2021-01-01")
    parser.add_argument("--end", default="2025-12-31")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--balance-weight", type=float, default=0.01,
                        help="Stage2 负载均衡 KL 权重: 0.01=DAFT-A, 0=DAFT-noKL")
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = get_device()
    t_total = time.time()
    tag = "A" if args.balance_weight > 0 else "noKL"

    cfg = {
        "stage1": {"epochs": 15, "batch_size": 1024, "lr": 1e-3},
        "stage2": {"epochs": 10, "batch_size": 512, "lr": 1e-3,
                   "balance_weight": args.balance_weight},
        "stage3": {"epochs": 8, "batch_size": 512, "lr": 1e-5},
    }
    if args.full:
        cfg = {
            "stage1": {"epochs": 50, "batch_size": 2048, "lr": 1e-3},
            "stage2": {"epochs": 30, "batch_size": 1024, "lr": 1e-3,
                       "balance_weight": args.balance_weight},
            "stage3": {"epochs": 20, "batch_size": 1024, "lr": 1e-5},
        }

    print(f"=== DAFT 周线训练 (variant={tag}, seed={args.seed}, device={device}) ===")

    # ---------- 1. 日线数据(同一源, 命中缓存) ----------
    adapter = BaostockAdapter({
        "start_date": args.start, "end_date": args.end,
        "frequency": "d", "n_stocks": args.stocks,
        "universe": "hs300", "adjust": "2",
    })
    daily = adapter.load()
    T_d, N, _ = daily.shape

    # ---------- 2. 周线重采样 + 切分(日历对齐, 复用 ridge weekly 逻辑) ----------
    weekly = resample_weekly(daily)
    T_w = weekly.values.shape[0]
    weekly_dates = weekly.dates
    cut_train_date = daily.dates[int(T_d * 0.6)]
    cut_val_date = daily.dates[int(T_d * 0.8)]
    T_wm1 = T_w - 1
    n_train_w = sum(1 for t in range(T_wm1) if weekly_dates[t + 1] <= cut_train_date)
    n_val_w = sum(1 for t in range(T_wm1) if weekly_dates[t + 1] <= cut_val_date)
    print(f"    日线 T={T_d} → 周线 T={T_w}; train≤{n_train_w}周, val≤{n_val_w}周, test={T_wm1 - n_val_w}周")

    train_panel = weekly.slice_time(0, n_train_w)
    val_panel = weekly.slice_time(n_train_w, n_val_w)

    # ---------- 3. Stage 1 (仅 train 段) ----------
    print("[Stage 1] 独立专家训练...")
    experts = build_experts()
    s1 = Stage1ExpertTrainer(experts=experts, panel=train_panel, device=device,
                             lookback_scale=LOOKBACK_SCALE)
    t0 = time.time()
    s1_hist = s1.train_all(epochs=cfg["stage1"]["epochs"],
                           batch_size=cfg["stage1"]["batch_size"],
                           lr=cfg["stage1"]["lr"], verbose=False)
    stage1_seconds = time.time() - t0
    print(f"    Stage 1: {stage1_seconds:.1f}s")

    # ---------- 4. Stage 2 + 3 ----------
    print("[Stage 2+3] 路由/记忆/CDAP + 联合微调...")
    model = build_ensemble(experts, cdap_strength=0.1)
    layer_proj = build_layer_proj()
    model = model.to(device)
    layer_proj = layer_proj.to(device)

    s2 = RouterTrainer(model=model, config=cfg["stage2"], device=device,
                       lookback_scale=LOOKBACK_SCALE)
    t0 = time.time()
    s2.train(train_panel, val_panel)
    stage2_seconds = time.time() - t0

    model.cross_dim_attn.modulation_strength = 1.0
    model.router.temperature = 0.1
    s3 = JointTrainer(model=model, layer_proj=layer_proj, config=cfg["stage3"],
                      device=device, lookback_scale=LOOKBACK_SCALE)
    t0 = time.time()
    s3.train(train_panel, val_panel)
    stage3_seconds = time.time() - t0

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    s3.save_checkpoints(str(CHECKPOINT_DIR))

    # ---------- 5. 信号生成(全周序列, 因果记忆推进) ----------
    print("[信号生成] 逐周因果推进...")
    extractor = RegimeFeatureExtractor(n_base_factors=50, output_dim=200,
                                       lookback_scale=LOOKBACK_SCALE)
    with torch.no_grad():
        s_tr = extractor(train_panel)
    s_tr = torch.nan_to_num(s_tr, nan=0.0, posinf=1e6, neginf=-1e6).clamp(-1e6, 1e6)
    flat = s_tr.reshape(-1, 200)
    mu = flat.mean(dim=0, keepdim=True)
    sd = flat.std(dim=0, keepdim=True).clamp(min=1e-4)

    with torch.no_grad():
        s_all = extractor(weekly)
    s_all = torch.nan_to_num(s_all, nan=0.0, posinf=1e6, neginf=-1e6).clamp(-1e6, 1e6)
    s_all = ((s_all - mu) / sd).clamp(-10.0, 10.0)

    model.eval()
    layer_proj.eval()
    model.memory.reset_state(1, device)
    signals = torch.zeros(T_w - 1, N)
    with torch.no_grad():
        for t in range(T_w - 1):
            s_b = s_all[t].to(device)
            if model.memory.M is None or model.memory.M.size(0) != N:
                model.memory.reset_state(N, device)
            l0 = layer_proj["l0"](s_b)
            l1 = layer_proj["l1"](s_b)
            l2 = layer_proj["l2"](s_b)
            out = model(s_b, [l0, l1, l2], mode="inference")
            signals[t] = out["signal"].squeeze(-1).cpu()
            model.memory.detach_state()

    # ---------- 6. test 段评估(对齐 k→k+1, 逐周 IC) ----------
    n_test = T_wm1 - n_val_w
    signals_test = torch.cat([signals[n_val_w:], torch.zeros(1, N)], dim=0)
    prices_test = weekly.values[n_val_w:, :, 3]
    mask_test = weekly.mask[n_val_w:]

    engine = BacktestEngine({
        "transaction_cost_bps": 5.0, "slippage_bps": 1.0,
        "top_quantile": 0.2, "long_only": False,
    })
    bt = engine.run(signals_test, prices_test, mask=mask_test)

    log_pt = torch.log(prices_test.clamp(min=1e-8))
    returns_test = log_pt[1:] - log_pt[:-1]
    ic_series = rank_info_coefficient(signals_test[:-1], returns_test,
                                      mask_test[1:], per_timestep=True)
    ic_stats = ic_summary(ic_series)
    hit = hit_rate(signals_test[:-1], returns_test, mask_test[1:])

    print(f"\n  ── DAFT-{tag} · 周线 · 样本外 ──")
    print(f"    Rank IC: {ic_stats['ic_mean']:+.4f}  t: {ic_stats['ic_t_stat']:+.2f}  "
          f"Sharpe: {bt['sharpe_ratio']:+.4f}  换手: {bt['turnover']:.3f}")

    # 逐周 IC + 周日期(供 #28 配对)
    ic_dates = [str(weekly_dates[n_val_w + j + 1]) for j in range(n_test)]
    ic_list = [None if (x != x) else float(x) for x in ic_series]

    report = {
        "model": "DAFT_weekly",
        "variant": tag,
        "seed": args.seed,
        "balance_weight": args.balance_weight,
        "lookback_scale": LOOKBACK_SCALE,
        "frequency": "weekly_wfri",
        "data": {"stocks": N, "T_daily": T_d, "T_weekly": T_w,
                 "start": args.start, "end": args.end},
        "split": {"cut_train_date": str(cut_train_date), "cut_val_date": str(cut_val_date),
                  "n_train_w": n_train_w, "n_val_w": n_val_w, "n_test_w": n_test},
        "out_of_sample": {
            "ic_mean": ic_stats["ic_mean"], "ic_std": ic_stats["ic_std"],
            "icir": ic_stats["icir"], "ic_t_stat": ic_stats["ic_t_stat"],
            "ic_positive_ratio": ic_stats["ic_positive_ratio"], "hit_rate": hit,
            "ic_series": ic_list, "ic_dates": ic_dates,
        },
        "backtest": bt,
        "config_hash": config_hash({"variant": tag, "balance_weight": args.balance_weight,
                                    "lookback_scale": LOOKBACK_SCALE, **cfg}),
        "time_seconds": round(time.time() - t_total, 1),
    }
    out_path = next_exp_path(OUTPUT_DIR, f"daft-weekly-{tag}")
    report["experiment_id"] = out_path.stem
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"    报告 → {out_path}")


if __name__ == "__main__":
    main()
