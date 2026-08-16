"""DAFT Stage 2 — Router + Memory + CDAP Training (experts frozen).

Loads Stage 1 expert checkpoints, freezes them, then trains the Regime Router,
KDA Market Memory, and Cross-Dimension Attention Protocol (CDAP) on synthetic
data.

Usage:
  python scripts/run_stage2.py                  # quick run (20 epochs)
  python scripts/run_stage2.py --epochs 50      # custom epochs
  python scripts/run_stage2.py --cpu            # force CPU
"""

from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

import torch

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from daft.data.loaders import DataLoader
from daft.models.factory import build_experts, build_ensemble
from daft.training.router_trainer import RouterTrainer
from daft.training.expert_trainer import Stage1ExpertTrainer

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
OUTPUT_DIR = PROJECT_ROOT / "outputs"
STAGE1_CKPT = PROJECT_ROOT / "checkpoints" / "stage1"
STAGE2_CKPT = PROJECT_ROOT / "checkpoints" / "stage2"

DEFAULT_CONFIG = {
    "data": {
        "source": "synthetic",
        "n_stocks": 100,
        "n_days": 500,
        "seed": 42,
    },
    "training": {
        "epochs": 20,
        "batch_size": 1024,
        "lr": 1e-3,
        "weight_decay": 1e-5,
        "early_stop_patience": 8,
        "balance_every": 50,
        "entropy_weight": 0.01,
        "grad_clip_norm": 1.0,
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="DAFT Stage 2 Training")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override epoch count")
    parser.add_argument("--lr", type=float, default=None,
                        help="Override learning rate")
    parser.add_argument("--cpu", action="store_true",
                        help="Force CPU even if CUDA available")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--stage1-dir", type=str, default=str(STAGE1_CKPT),
                        help="Stage 1 checkpoint directory")
    args = parser.parse_args()

    device = torch.device("cpu") if args.cpu or not torch.cuda.is_available() \
        else torch.device("cuda")
    torch.manual_seed(args.seed)

    # --------------- Step 1: Load Stage 1 experts ---------------
    banner("Step 1: Load Stage 1 Expert Checkpoints")
    experts = build_experts()
    stage1_dir = Path(args.stage1_dir)
    if stage1_dir.exists():
        experts = Stage1ExpertTrainer.load_checkpoints(experts, str(stage1_dir))
        print(f"  Loaded expert weights from {stage1_dir}")
    else:
        print(f"  [WARN] {stage1_dir} not found — using untrained experts")
        print("  Run 'python scripts/run_stage1.py' first.")

    for i, e in enumerate(experts):
        ep = sum(p.numel() for p in e.parameters())
        print(f"    [{i}] {e.name:12s}  {ep:>7,} params")

    # --------------- Step 2: Build ensemble ---------------
    banner("Step 2: Build DAFT Ensemble")
    model = build_ensemble(experts)
    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.router.parameters()) + \
                sum(p.numel() for p in model.memory.parameters()) + \
                sum(p.numel() for p in model.cross_dim_attn.parameters())
    print(f"  Router params      : {sum(p.numel() for p in model.router.parameters()):,}")
    print(f"  Memory params      : {sum(p.numel() for p in model.memory.parameters()):,}")
    print(f"  CDAP params        : {sum(p.numel() for p in model.cross_dim_attn.parameters()):,}")
    print(f"  Trainable (Stage 2): {trainable:,}")
    print(f"  Total params       : {total_params:,}")

    # --------------- Step 3: Generate data ---------------
    banner("Step 3: Generate Synthetic Data")
    t0 = time.time()
    loader = DataLoader(DEFAULT_CONFIG["data"])
    panel = loader.load()
    dt = time.time() - t0
    print(f"  Panel shape : {panel.shape}  (T={panel.T}, N={panel.N}, F={panel.F})")
    print(f"  Time        : {dt:.1f}s")

    # Split: 80% train, 20% val (temporal split)
    T = panel.T
    split_t = int(T * 0.8)
    train_panel = panel.slice_time(0, split_t)
    val_panel = panel.slice_time(split_t, T)
    print(f"  Train       : T=0..{split_t}  ({split_t} steps)")
    print(f"  Val         : T={split_t}..{T}  ({T - split_t} steps)")

    # --------------- Step 4: Stage 2 training ---------------
    banner("Step 4: Stage 2 — Router + Memory + CDAP Training")

    cfg = DEFAULT_CONFIG["training"].copy()
    if args.epochs is not None:
        cfg["epochs"] = args.epochs
    if args.lr is not None:
        cfg["lr"] = args.lr

    print(f"  Epochs      : {cfg['epochs']}")
    print(f"  Batch size  : {cfg['batch_size']}")
    print(f"  LR          : {cfg['lr']}")
    print(f"  CDAP δ      : 0.1 (low modulation)")
    print(f"  Device      : {device}")

    trainer = RouterTrainer(model=model, config=cfg, device=device)

    t0 = time.time()
    history = trainer.train(train_panel, val_panel)
    dt_train = time.time() - t0

    # --------------- Step 5: Summary ---------------
    banner("Step 5: Training Summary")
    if history["val_loss"]:
        print(f"  Initial val_loss : {history['val_loss'][0]:.6f}")
        print(f"  Final val_loss   : {history['val_loss'][-1]:.6f}")
    if history["val_ic_mean"]:
        print(f"  Final val_IC     : {history['val_ic_mean'][-1]:+.4f}")
        print(f"  Final ICIR       : {history['val_icir'][-1]:+.3f}")
    print(f"  Training time    : {dt_train:.1f}s")

    # --------------- Step 6: Persist ---------------
    banner("Step 6: Save Outputs")

    trainer.save_checkpoints(str(STAGE2_CKPT))
    print(f"  Checkpoints → {STAGE2_CKPT}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    loss_path = OUTPUT_DIR / "stage2_losses.json"
    with open(loss_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"  Loss curves → {loss_path}")

    report = {
        "config": cfg,
        "data": {"T": panel.T, "N": panel.N},
        "trainable_params_stage2": trainable,
        "training_time_seconds": round(dt_train, 1),
        "device": str(device),
        "final_val_loss": history["val_loss"][-1] if history["val_loss"] else None,
        "final_val_ic": history["val_ic_mean"][-1] if history["val_ic_mean"] else None,
        "final_icir": history["val_icir"][-1] if history["val_icir"] else None,
    }
    report_path = OUTPUT_DIR / "stage2_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"  Report      → {report_path}")

    banner("STAGE 2 COMPLETE")
    print(f"\n  Next step: python scripts/run_stage3.py  (joint fine-tuning)")


if __name__ == "__main__":
    main()
