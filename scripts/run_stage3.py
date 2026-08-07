"""DAFT Stage 3 — Joint Fine-Tuning (all parameters unfrozen).

Loads Stage 2 checkpoints (router + memory + CDAP + frozen experts),
unfreezes all parameters, and jointly fine-tunes with full CDAP modulation
(δ = 1.0) at very low learning rate (η = 1e-5).

Usage:
  python scripts/run_stage3.py                  # quick run (15 epochs)
  python scripts/run_stage3.py --epochs 30      # custom epochs
  python scripts/run_stage3.py --cpu            # force CPU
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
from daft.models.experts import (
    TrendExpert, ReversalExpert, VolatilityExpert, EventExpert, MomentumExpert,
)
from daft.models.router import RegimeRouter
from daft.models.memory import KDAMarketMemory
from daft.models.cross_dim_attn import CrossDimensionAttention
from daft.models.hardening import HardeningEngine
from daft.models.ensemble import ExpertEnsemble
from daft.training.joint_trainer import JointTrainer
from daft.training.expert_trainer import Stage1ExpertTrainer
from daft.training.router_trainer import RouterTrainer

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
OUTPUT_DIR = PROJECT_ROOT / "outputs"
STAGE1_CKPT = PROJECT_ROOT / "checkpoints" / "stage1"
STAGE2_CKPT = PROJECT_ROOT / "checkpoints" / "stage2"
STAGE3_CKPT = PROJECT_ROOT / "checkpoints" / "stage3"

DEFAULT_CONFIG = {
    "data": {
        "source": "synthetic",
        "n_stocks": 100,
        "n_days": 500,
        "seed": 42,
    },
    "training": {
        "epochs": 15,
        "batch_size": 1024,
        "lr": 1e-5,
        "weight_decay": 1e-6,
        "early_stop_patience": 5,
        "grad_clip_norm": 0.5,
        "expert_lr_ratio": 0.1,
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


def build_ensemble(experts: nn.ModuleList) -> ExpertEnsemble:
    router = RegimeRouter(
        input_dim=200, latent_dim=16, n_experts=8, top_k=3,
        temperature=0.1, noisy_gating_std=0.0,
    )
    memory = KDAMarketMemory(
        d_k=128, d_v=64, d_feature=200,
        bottleneck_ratio=4, use_route_modulation=True,
    )
    cdap = CrossDimensionAttention(
        n_experts=8, d_k=128, d_v=64, n_layers=3,
        joint_dim=64, modulation_strength=1.0,
    )
    hardening = HardeningEngine(
        n_regimes=8, n_experts=8, threshold=100,
    )
    return ExpertEnsemble(experts, router, memory, cdap, hardening)


def build_layer_proj(d_v: int = 64, input_dim: int = 200) -> nn.ModuleDict:
    """Reconstruct the same layer_proj architecture used in RouterTrainer."""
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="DAFT Stage 3 Training")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override epoch count")
    parser.add_argument("--lr", type=float, default=None,
                        help="Override learning rate")
    parser.add_argument("--cpu", action="store_true",
                        help="Force CPU")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--stage1-dir", type=str, default=str(STAGE1_CKPT),
                        help="Stage 1 checkpoint directory")
    parser.add_argument("--stage2-dir", type=str, default=str(STAGE2_CKPT),
                        help="Stage 2 checkpoint directory")
    parser.add_argument("--skip-stage2", action="store_true",
                        help="Skip loading Stage 2 weights (start from Stage 1 only)")
    args = parser.parse_args()

    device = torch.device("cpu") if args.cpu or not torch.cuda.is_available() \
        else torch.device("cuda")
    torch.manual_seed(args.seed)

    # --------------- Step 1: Load experts (Stage 1) ---------------
    banner("Step 1: Load Stage 1 Expert Checkpoints")
    experts = build_experts()
    stage1_dir = Path(args.stage1_dir)
    if stage1_dir.exists():
        experts = Stage1ExpertTrainer.load_checkpoints(experts, str(stage1_dir))
        print(f"  Loaded expert weights from {stage1_dir}")
    else:
        print(f"  [WARN] {stage1_dir} not found — using untrained experts")

    # --------------- Step 2: Build ensemble ---------------
    banner("Step 2: Build DAFT Ensemble")
    model = build_ensemble(experts)
    layer_proj = build_layer_proj()

    # --------------- Step 3: Load Stage 2 weights ---------------
    if not args.skip_stage2:
        banner("Step 3: Load Stage 2 (Router + Memory + CDAP) Checkpoints")
        stage2_dir = Path(args.stage2_dir)
        if stage2_dir.exists():
            RouterTrainer.load_checkpoints(model, layer_proj, str(stage2_dir))
            print(f"  Loaded Stage 2 weights from {stage2_dir}")
        else:
            print(f"  [WARN] {stage2_dir} not found — router/memory/CDAP untrained")
            print("  Run 'python scripts/run_stage2.py' first, or use --skip-stage2")
    else:
        banner("Step 3: Skipping Stage 2 (--skip-stage2)")
        print("  Router, memory, CDAP, and layer projections are randomly initialized.")

    total_params = sum(p.numel() for p in model.parameters())
    trainable = total_params  # all unfrozen in Stage 3
    print(f"  Total params : {total_params:,}  (all trainable in Stage 3)")

    # --------------- Step 4: Generate data ---------------
    banner("Step 4: Generate Synthetic Data")
    t0 = time.time()
    loader = DataLoader(DEFAULT_CONFIG["data"])
    panel = loader.load()
    dt = time.time() - t0
    print(f"  Panel shape : {panel.shape}  (T={panel.T}, N={panel.N}, F={panel.F})")

    T = panel.T
    split_t = int(T * 0.8)
    train_panel = panel.slice_time(0, split_t)
    val_panel = panel.slice_time(split_t, T)
    print(f"  Time        : {dt:.1f}s")

    # --------------- Step 5: Stage 3 training ---------------
    banner("Step 5: Stage 3 — Joint Fine-Tuning")

    cfg = DEFAULT_CONFIG["training"].copy()
    if args.epochs is not None:
        cfg["epochs"] = args.epochs
    if args.lr is not None:
        cfg["lr"] = args.lr

    print(f"  Epochs      : {cfg['epochs']}")
    print(f"  Batch size  : {cfg['batch_size']}")
    print(f"  LR          : {cfg['lr']}  (experts: {cfg['lr'] * cfg['expert_lr_ratio']:.1e})")
    print(f"  CDAP δ      : 1.0 (full modulation)")
    print(f"  Grad clip   : {cfg['grad_clip_norm']}")
    print(f"  Device      : {device}")

    trainer = JointTrainer(model=model, layer_proj=layer_proj, config=cfg, device=device)

    t0 = time.time()
    history = trainer.train(train_panel, val_panel)
    dt_train = time.time() - t0

    # --------------- Step 6: Summary ---------------
    banner("Step 6: Training Summary")
    if history["val_loss"]:
        print(f"  Initial val_loss : {history['val_loss'][0]:.6f}")
        print(f"  Final val_loss   : {history['val_loss'][-1]:.6f}")
    if history["val_ic_mean"]:
        print(f"  Final val_IC     : {history['val_ic_mean'][-1]:+.4f}")
        print(f"  Final ICIR       : {history['val_icir'][-1]:+.3f}")
    print(f"  Training time    : {dt_train:.1f}s")

    # --------------- Step 7: Persist ---------------
    banner("Step 7: Save Outputs")

    trainer.save_checkpoints(str(STAGE3_CKPT))
    print(f"  Checkpoints → {STAGE3_CKPT}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    loss_path = OUTPUT_DIR / "stage3_losses.json"
    with open(loss_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"  Loss curves → {loss_path}")

    report = {
        "config": cfg,
        "data": {"T": panel.T, "N": panel.N},
        "total_params": total_params,
        "training_time_seconds": round(dt_train, 1),
        "device": str(device),
        "final_val_loss": history["val_loss"][-1] if history["val_loss"] else None,
        "final_val_ic": history["val_ic_mean"][-1] if history["val_ic_mean"] else None,
        "final_icir": history["val_icir"][-1] if history["val_icir"] else None,
    }
    report_path = OUTPUT_DIR / "stage3_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"  Report      → {report_path}")

    banner("STAGE 3 COMPLETE")
    print(f"\n  Final model saved to {STAGE3_CKPT}")
    print(f"  Next step: python scripts/run_full_pipeline.py  (backtest + report)")


if __name__ == "__main__":
    main()
