"""DAFT Stage 1 — Independent Expert Training (end-to-end).

Generates synthetic data → extracts 200-dim market state s_t → trains
8 experts on their regime-specific subsets → outputs loss curves and
checkpoints.

This is the first concrete milestone: data → features → trained experts.

Usage:
  python scripts/run_stage1.py                # quick run (20 epochs)
  python scripts/run_stage1.py --full         # full training (100 epochs)
  python scripts/run_stage1.py --epochs 50    # custom epoch count
"""

from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

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
from daft.training.expert_trainer import Stage1ExpertTrainer

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
OUTPUT_DIR = PROJECT_ROOT / "outputs"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints" / "stage1"

DEFAULT_CONFIG = {
    "data": {
        "source": "synthetic",
        "n_stocks": 100,
        "n_days": 500,
        "seed": 42,
    },
    "training": {
        "epochs": 20,
        "batch_size": 2048,
        "lr": 1e-3,
        "weight_decay": 1e-5,
        "early_stop_patience": 10,
        "val_frac": 0.1,
    },
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
    """Create 8 experts: 2 per strategy type."""
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


def count_params(experts: nn.ModuleList) -> int:
    return sum(p.numel() for expert in experts for p in expert.parameters())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="DAFT Stage 1 Training")
    parser.add_argument("--full", action="store_true",
                        help="Full training: 100 epochs")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override epoch count")
    parser.add_argument("--lr", type=float, default=None,
                        help="Override learning rate")
    parser.add_argument("--cpu", action="store_true",
                        help="Force CPU even if CUDA available")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    args = parser.parse_args()

    # Device
    device = torch.device("cpu") if args.cpu or not torch.cuda.is_available() \
        else torch.device("cuda")
    torch.manual_seed(args.seed)

    # --------------- Step 1: Generate data ---------------
    banner("Step 1: Generate Synthetic Data")
    t0 = time.time()
    loader = DataLoader(DEFAULT_CONFIG["data"])
    panel = loader.load()
    T, N, F = panel.shape
    dt = time.time() - t0
    print(f"  Panel shape : (T={T}, N={N}, F={F})")
    print(f"  Regime distribution (ground truth):")
    regime_ids = panel.metadata.get("regime_ids")
    if regime_ids is not None:
        for rid, label in enumerate(["bull", "bear", "choppy"]):
            pct = (regime_ids == rid).float().mean().item() * 100
            print(f"    {label:8s}: {pct:5.1f}%")
    print(f"  Time        : {dt:.1f}s")

    # --------------- Step 2: Extract features ---------------
    banner("Step 2: Extract Market State Vectors s_t")
    t0 = time.time()
    extractor = RegimeFeatureExtractor(n_base_factors=50, output_dim=200)
    with torch.no_grad():
        s_t = extractor(panel)
    dt = time.time() - t0
    print(f"  s_t shape   : {s_t.shape}  (T={s_t.shape[0]}, N={s_t.shape[1]}, F=200)")
    print(f"  s_t stats   : mean={s_t.mean():+.4f}, std={s_t.std():.4f}, "
          f"min={s_t.min():.4f}, max={s_t.max():.4f}")
    has_nan = s_t.isnan().any().item()
    has_inf = s_t.isinf().any().item()
    print(f"  NaNs        : {has_nan}")
    print(f"  Infs        : {has_inf}")
    if has_nan:
        print("  [WARN] s_t contains NaN — replacing with zeros for stability")
        s_t = torch.nan_to_num(s_t)
    print(f"  Time        : {dt:.1f}s")

    # --------------- Step 3: Create experts ---------------
    banner("Step 3: Create Expert Pool")
    experts = build_experts()
    n_params = count_params(experts)
    print(f"  Experts     : 8 (2×trend, 2×reversal, 2×volatility, 2×event)")
    for i, e in enumerate(experts):
        ep = sum(p.numel() for p in e.parameters())
        print(f"    [{i}] {e.name:12s}  {ep:>7,} params")
    print(f"  Total params: {n_params:,}")

    # --------------- Step 4: Stage 1 training ---------------
    banner("Step 4: Stage 1 — Independent Expert Training")

    cfg = DEFAULT_CONFIG["training"].copy()
    if args.full:
        cfg["epochs"] = 100
    if args.epochs is not None:
        cfg["epochs"] = args.epochs
    if args.lr is not None:
        cfg["lr"] = args.lr

    print(f"  Epochs      : {cfg['epochs']}")
    print(f"  Batch size  : {cfg['batch_size']}")
    print(f"  LR          : {cfg['lr']}")
    print(f"  Device      : {device}")

    trainer = Stage1ExpertTrainer(
        experts=experts,
        panel=panel,
        config=cfg,
        device=device,
    )

    t0 = time.time()
    histories = trainer.train_all(
        epochs=cfg["epochs"],
        batch_size=cfg["batch_size"],
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"],
        early_stop_patience=cfg["early_stop_patience"],
        verbose=True,
    )
    dt_train = time.time() - t0

    # --------------- Step 5: Summary ---------------
    banner("Step 5: Training Summary")

    print(f"\n  {'Expert':<24s} {'Initial':>10s} {'Final':>10s} {'Δ%':>8s} {'Epochs':>8s}")
    print(f"  {'-'*24} {'-'*10} {'-'*10} {'-'*8} {'-'*8}")

    summary_records = []
    for key, hist in histories.items():
        if hist:
            initial = hist[0]
            final = hist[-1]
            delta = (initial - final) / (abs(initial) + 1e-8) * 100
            print(f"  {key:<24s} {initial:10.6f} {final:10.6f} {delta:+7.1f}% {len(hist):8d}")
            summary_records.append({
                "expert": key,
                "initial_loss": initial,
                "final_loss": final,
                "improvement_pct": round(delta, 2),
                "epochs_trained": len(hist),
            })
        else:
            print(f"  {key:<24s} {'(skipped)':>10s}")

    print(f"\n  Total training time: {dt_train:.1f}s")

    # --------------- Step 6: Persist ---------------
    banner("Step 6: Save Outputs")

    # Checkpoints
    trainer.save_checkpoints(str(CHECKPOINT_DIR))
    print(f"  Checkpoints → {CHECKPOINT_DIR}")

    # Loss histories as JSON
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    loss_path = OUTPUT_DIR / "stage1_losses.json"
    with open(loss_path, "w") as f:
        json.dump(histories, f, indent=2)
    print(f"  Loss curves → {loss_path}")

    # Summary report
    report = {
        "config": cfg,
        "data": {"T": T, "N": N, "F": F},
        "total_expert_params": n_params,
        "training_time_seconds": round(dt_train, 1),
        "device": str(device),
        "experts": summary_records,
    }
    report_path = OUTPUT_DIR / "stage1_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"  Report      → {report_path}")

    # --------------- Done ---------------
    banner("STAGE 1 COMPLETE")

    trained = len([r for r in summary_records if r["epochs_trained"] > 0])
    improved = len([r for r in summary_records if r["improvement_pct"] > 0])
    print(f"\n  {trained}/8 experts trained, {improved}/8 improved")
    print(f"  Outputs     : {OUTPUT_DIR}")
    print(f"  Checkpoints : {CHECKPOINT_DIR}")
    print(f"\n  Next step: python scripts/run_stage2.py  (router + memory training)")


if __name__ == "__main__":
    main()
