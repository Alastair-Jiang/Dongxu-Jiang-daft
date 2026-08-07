"""DAFT Full Pipeline — End-to-end: data → features → Stage 1 → Stage 2 → Stage 3 → backtest.

The complete DAFT workflow from raw data to backtest report. Each stage loads
the previous stage's checkpoints and produces its own. The final report
includes Sharpe, MaxDD, Calmar, IC, ICIR, and hit rate from walk-forward
backtesting.

Usage:
  python scripts/run_full_pipeline.py                     # quick run
  python scripts/run_full_pipeline.py --full              # full training
  python scripts/run_full_pipeline.py --source yfinance   # real US data
  python scripts/run_full_pipeline.py --cpu               # force CPU
"""

from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from daft.data.loaders import DataLoader
from daft.data.panel import Panel
from daft.features.regime_features import RegimeFeatureExtractor
from daft.models.experts import (
    TrendExpert, ReversalExpert, VolatilityExpert, EventExpert, MomentumExpert,
)
from daft.models.router import RegimeRouter
from daft.models.memory import KDAMarketMemory
from daft.models.cross_dim_attn import CrossDimensionAttention
from daft.models.hardening import HardeningEngine
from daft.models.ensemble import ExpertEnsemble
from daft.training.expert_trainer import Stage1ExpertTrainer
from daft.training.router_trainer import RouterTrainer
from daft.training.joint_trainer import JointTrainer
from daft.backtest.engine import BacktestEngine
from daft.utils.metrics import rank_info_coefficient, ic_summary

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
OUTPUT_DIR = PROJECT_ROOT / "outputs"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"

QUICK_CONFIG = {
    "data": {"source": "synthetic", "n_stocks": 50, "n_days": 300, "seed": 42},
    "stage1": {"epochs": 15, "batch_size": 1024, "lr": 1e-3},
    "stage2": {"epochs": 10, "batch_size": 512, "lr": 1e-3},
    "stage3": {"epochs": 8,  "batch_size": 512, "lr": 1e-5},
}

FULL_CONFIG = {
    "data": {"source": "synthetic", "n_stocks": 100, "n_days": 500, "seed": 42},
    "stage1": {"epochs": 50, "batch_size": 2048, "lr": 1e-3},
    "stage2": {"epochs": 30, "batch_size": 1024, "lr": 1e-3},
    "stage3": {"epochs": 20, "batch_size": 1024, "lr": 1e-5},
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def banner(msg: str):
    w = 72
    print(f"\n{'=' * w}")
    print(f"  {msg}")
    print(f"{'=' * w}")


def build_experts() -> nn.ModuleList:
    return nn.ModuleList([
        TrendExpert(input_dim=200, hidden_dim=64),
        TrendExpert(input_dim=200, hidden_dim=64),
        ReversalExpert(input_dim=200, hidden_dim=64),
        ReversalExpert(input_dim=200, hidden_dim=64),
        VolatilityExpert(input_dim=200, hidden_dim=48),
        VolatilityExpert(input_dim=200, hidden_dim=48),
        EventExpert(input_dim=200, hidden_dim=48),
        EventExpert(input_dim=200, hidden_dim=48),
        MomentumExpert(input_dim=200, hidden_dim=64),
        MomentumExpert(input_dim=200, hidden_dim=64),
    ])


def build_ensemble(experts: nn.ModuleList, cdap_strength: float = 0.1) -> ExpertEnsemble:
    router = RegimeRouter(
        input_dim=200, latent_dim=16, n_experts=8, top_k=3,
        temperature=1.0, noisy_gating_std=0.1,
    )
    memory = KDAMarketMemory(
        d_k=128, d_v=64, d_feature=200,
        bottleneck_ratio=4, use_route_modulation=True,
    )
    cdap = CrossDimensionAttention(
        n_experts=8, d_k=128, d_v=64, n_layers=3,
        joint_dim=64, modulation_strength=cdap_strength,
    )
    hardening = HardeningEngine(n_regimes=8, n_experts=8)
    return ExpertEnsemble(experts, router, memory, cdap, hardening)


def build_layer_proj(d_v: int = 64, input_dim: int = 200) -> nn.ModuleDict:
    return nn.ModuleDict({
        "l0": nn.Sequential(
            nn.Linear(input_dim, 128), nn.SiLU(),
            nn.Linear(128, d_v), nn.LayerNorm(d_v),
        ),
        "l1": nn.Sequential(
            nn.Linear(input_dim, 128), nn.SiLU(),
            nn.Linear(128, d_v), nn.LayerNorm(d_v),
        ),
        "l2": nn.Sequential(
            nn.Linear(input_dim, 128), nn.SiLU(),
            nn.Linear(128, d_v), nn.LayerNorm(d_v),
        ),
    })


def generate_backtest_signals(
    model: ExpertEnsemble,
    layer_proj: nn.ModuleDict,
    panel: Panel,
    device: torch.device,
) -> torch.Tensor:
    """Generate trading signals for all timesteps using the trained model.

    Returns
    -------
    signals : (T-1, N) float tensor of predicted returns.
    """
    extractor = RegimeFeatureExtractor(n_base_factors=50, output_dim=200)
    with torch.no_grad():
        s_t_raw = extractor(panel)
    s_t_raw = torch.nan_to_num(s_t_raw, nan=0.0, posinf=1e6, neginf=-1e6)
    s_t_raw = s_t_raw.clamp(-1e6, 1e6)
    s_flat = s_t_raw.reshape(-1, 200)
    s_mean = s_flat.mean(dim=0, keepdim=True)
    s_std = s_flat.std(dim=0, keepdim=True).clamp(min=1e-4)
    s_t = ((s_t_raw - s_mean) / s_std).clamp(-10.0, 10.0)

    T, N, _ = s_t.shape
    model.eval()
    layer_proj.eval()
    model.memory.reset_state(1, device)

    signals = torch.zeros(T - 1, N)

    for t in range(T - 1):
        s_b = s_t[t].to(device)  # (N, 200)

        if model.memory.M is None or model.memory.M.size(0) != N:
            model.memory.reset_state(N, device)

        l0 = layer_proj["l0"](s_b)
        l1 = layer_proj["l1"](s_b)
        l2 = layer_proj["l2"](s_b)

        with torch.no_grad():
            outputs = model(s_b, [l0, l1, l2], mode="inference")
            signals[t] = outputs["signal"].squeeze(-1).cpu()

        model.memory.detach_state()

    return signals


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="DAFT Full Pipeline")
    parser.add_argument("--full", action="store_true",
                        help="Full training (more epochs, more stocks)")
    parser.add_argument("--source", type=str, default="synthetic",
                        choices=["synthetic", "baostock", "yfinance"],
                        help="Data source")
    parser.add_argument("--cpu", action="store_true",
                        help="Force CPU")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--skip-backtest", action="store_true",
                        help="Skip backtest (train only)")
    args = parser.parse_args()

    cfg = FULL_CONFIG if args.full else QUICK_CONFIG
    cfg["data"]["source"] = args.source

    device = torch.device("cpu") if args.cpu or not torch.cuda.is_available() \
        else torch.device("cuda")
    torch.manual_seed(args.seed)

    total_t0 = time.time()

    # ===================================================================
    # STAGE 1: Independent Expert Training
    # ===================================================================
    banner("STAGE 1: Independent Expert Training")

    print(f"  Data source : {cfg['data']['source']}")
    print(f"  Device      : {device}")
    print(f"  Config      : {'FULL' if args.full else 'QUICK'}")

    # Load data
    t0 = time.time()
    loader = DataLoader(cfg["data"])
    panel = loader.load()
    print(f"  Panel shape : {panel.shape}  ({time.time() - t0:.1f}s)")

    # Train experts
    experts = build_experts()
    s1_trainer = Stage1ExpertTrainer(experts=experts, panel=panel, device=device)

    t0 = time.time()
    s1_hist = s1_trainer.train_all(
        epochs=cfg["stage1"]["epochs"],
        batch_size=cfg["stage1"]["batch_size"],
        lr=cfg["stage1"]["lr"],
        verbose=True,
    )
    dt_s1 = time.time() - t0
    print(f"  Stage 1 time: {dt_s1:.1f}s")

    # Save Stage 1
    s1_ckpt = CHECKPOINT_DIR / "stage1"
    s1_trainer.save_checkpoints(str(s1_ckpt))

    # ===================================================================
    # STAGE 2: Router + Memory + CDAP Training
    # ===================================================================
    banner("STAGE 2: Router + Memory + CDAP Training")

    T = panel.T
    split_t = int(T * 0.8)
    train_panel = panel.slice_time(0, split_t)
    val_panel = panel.slice_time(split_t, T)

    model = build_ensemble(experts, cdap_strength=0.1)
    layer_proj = build_layer_proj()

    s2_trainer = RouterTrainer(model=model, config=cfg["stage2"], device=device)

    t0 = time.time()
    s2_hist = s2_trainer.train(train_panel, val_panel)
    dt_s2 = time.time() - t0
    print(f"  Stage 2 time: {dt_s2:.1f}s")

    s2_ckpt = CHECKPOINT_DIR / "stage2"
    s2_trainer.save_checkpoints(str(s2_ckpt))

    # ===================================================================
    # STAGE 3: Joint Fine-Tuning
    # ===================================================================
    banner("STAGE 3: Joint Fine-Tuning")

    model.cross_dim_attn.modulation_strength = 1.0
    model.router.temperature = 0.1

    s3_trainer = JointTrainer(
        model=model, layer_proj=layer_proj,
        config=cfg["stage3"], device=device,
    )

    t0 = time.time()
    s3_hist = s3_trainer.train(train_panel, val_panel)
    dt_s3 = time.time() - t0
    print(f"  Stage 3 time: {dt_s3:.1f}s")

    s3_ckpt = CHECKPOINT_DIR / "stage3"
    s3_trainer.save_checkpoints(str(s3_ckpt))

    # ===================================================================
    # BACKTEST
    # ===================================================================
    if not args.skip_backtest:
        banner("BACKTEST: Walk-Forward Evaluation")

        print("  Generating signals...")
        t0 = time.time()
        signals = generate_backtest_signals(model, layer_proj, panel, device)
        print(f"  Signals shape : {signals.shape}  ({time.time() - t0:.1f}s)")

        # Pad to match price length (T): signals[t] is the prediction at t
        # BacktestEngine expects signals.shape[0] == prices.shape[0]
        signals = torch.cat([torch.zeros(1, signals.size(1)), signals], dim=0)
        print(f"  Padded shape  : {signals.shape}")

        # Prices: use close prices for returns
        prices = panel.values[..., 3]  # (T, N)

        engine = BacktestEngine({
            "transaction_cost_bps": 5.0,
            "slippage_bps": 1.0,
            "top_quantile": 0.2,
            "long_only": False,
        })

        t0 = time.time()
        metrics = engine.run(signals, prices, mask=panel.mask)
        dt_bt = time.time() - t0
        print(f"  Backtest time: {dt_bt:.1f}s")

        # --- Print metrics ---
        print(f"\n  {'─' * 50}")
        print(f"  {'Metric':<30s} {'Value':>18s}")
        print(f"  {'─' * 50}")
        for k, v in metrics.items():
            if isinstance(v, float):
                print(f"  {k:<30s} {v: 18.4f}")
            else:
                print(f"  {k:<30s} {str(v):>18s}")
        print(f"  {'─' * 50}")
    else:
        metrics = {}
        dt_bt = 0.0

    # ===================================================================
    # FINAL REPORT
    # ===================================================================
    banner("FINAL REPORT")

    dt_total = time.time() - total_t0
    print(f"\n  Total time     : {dt_total:.1f}s  ({dt_total/60:.1f} min)")
    print(f"  Stage 1        : {dt_s1:.1f}s")
    print(f"  Stage 2        : {dt_s2:.1f}s")
    print(f"  Stage 3        : {dt_s3:.1f}s")
    print(f"  Backtest       : {dt_bt:.1f}s")

    # Collect final metrics
    s1_improved = sum(
        1 for k, h in s1_hist.items()
        if h and len(h) > 1 and h[-1] < h[0]
    )
    s2_final_ic = s2_hist.get("val_ic_mean", [0])[-1] if s2_hist.get("val_ic_mean") else 0
    s3_final_ic = s3_hist.get("val_ic_mean", [0])[-1] if s3_hist.get("val_ic_mean") else 0

    print(f"\n  Stage 1        : {s1_improved}/8 experts improved")
    print(f"  Stage 2 final IC : {s2_final_ic:+.4f}")
    print(f"  Stage 3 final IC : {s3_final_ic:+.4f}")
    if metrics:
        print(f"  Backtest Sharpe  : {metrics.get('sharpe_ratio', 0):.4f}")
        print(f"  Backtest MaxDD   : {metrics.get('max_drawdown', 0):.4f}")
        print(f"  Backtest IC      : {metrics.get('ic_rank', 0):.4f}")

    # Write full report
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    full_report = {
        "config": {k: v for k, v in cfg.items()},
        "timing": {
            "total_seconds": round(dt_total, 1),
            "stage1_seconds": round(dt_s1, 1),
            "stage2_seconds": round(dt_s2, 1),
            "stage3_seconds": round(dt_s3, 1),
            "backtest_seconds": round(dt_bt, 1),
        },
        "stage1": {
            "experts_trained": len(s1_hist),
            "experts_improved": s1_improved,
        },
        "stage2": {
            "final_val_loss": s2_hist.get("val_loss", [None])[-1],
            "final_val_ic": s2_final_ic,
        },
        "stage3": {
            "final_val_loss": s3_hist.get("val_loss", [None])[-1],
            "final_val_ic": s3_final_ic,
        },
        "backtest": metrics,
        "device": str(device),
        "source": cfg["data"]["source"],
    }

    report_path = OUTPUT_DIR / "full_pipeline_report.json"
    with open(report_path, "w") as f:
        json.dump(full_report, f, indent=2, default=str)
    print(f"\n  Full report → {report_path}")

    # Save all histories
    histories = {
        "stage1": {k: v for k, v in s1_hist.items()},
        "stage2": {k: v for k, v in s2_hist.items()},
        "stage3": {k: v for k, v in s3_hist.items()},
    }
    hist_path = OUTPUT_DIR / "full_pipeline_histories.json"
    with open(hist_path, "w") as f:
        json.dump(histories, f, indent=2)
    print(f"  Histories   → {hist_path}")

    banner("PIPELINE COMPLETE")
    print(f"\n  Checkpoints : {CHECKPOINT_DIR}")
    print(f"  Reports     : {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
