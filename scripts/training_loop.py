"""DAFT Training Loop — Synthetic Data Demo.

Runs a mini 3-stage training cycle on synthetic data to demonstrate:
  Stage 1: Independent expert training
  Stage 2: Router + Memory training (experts frozen)
  Stage 3: Joint fine-tuning
  Stage 4: Hardening statistics collection

Outputs:
  - Training metrics per stage
  - Hardening stats
  - Model checkpoint (for future evaluation)

Usage:
  python scripts/training_loop.py
"""

import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from daft.data.loaders import DataLoader
from daft.models.experts import TrendExpert, ReversalExpert, VolatilityExpert, EventExpert
from daft.models.router import RegimeRouter
from daft.models.memory import KDAMarketMemory
from daft.models.cross_dim_attn import CrossDimensionAttention
from daft.models.hardening import HardeningEngine
from daft.models.ensemble import ExpertEnsemble


# ======================================================================
# Config
# ======================================================================
DEVICE = torch.device("cpu")
BATCH_SIZE = 128
N_EPOCHS_STAGE1 = 5
N_EPOCHS_STAGE2 = 10
N_EPOCHS_STAGE3 = 5
LR = 0.001

RESULTS = {}


def banner(msg):
    print(f"\n{'='*60}")
    print(f"  {msg}")
    print(f"{'='*60}")


def make_batch(panel, batch_size, start_idx):
    """Create mock training batches from panel data."""
    T, N, F = panel.shape
    returns = panel.values[..., 1]
    vol = panel.values[..., 4]

    end = min(start_idx + batch_size, T - 5)
    actual_batch = end - start_idx
    if actual_batch < 2:
        return None, None, None, start_idx

    idx = torch.arange(start_idx, end)
    sr = returns[idx]
    sv = vol[idx]
    s_batch = torch.cat([sr, sv, sr.roll(1, 0), sv.roll(1, 0), sr.roll(5, 0)], dim=-1)
    s_t = s_batch[:, :200]
    if s_t.size(1) < 200:
        pad = torch.zeros(actual_batch, 200 - s_t.size(1))
        s_t = torch.cat([s_t, pad], dim=-1)

    # Target: next-bar return
    target = returns[idx + 1][:, :1] if torch.isnan(returns[idx + 1]).sum() == 0 else returns[idx][:, :1]
    # Match batch size
    min_len = min(s_t.size(0), target.size(0))
    s_t = s_t[:min_len]
    target = target[:min_len]

    # Mock 3-layer features
    mock_layers = [
        torch.randn(min_len, 64),
        torch.randn(min_len, 64),
        torch.randn(min_len, 64),
    ]

    return s_t, target, mock_layers, end


def run_epoch(model, panel, optimizer=None, mode="train", use_hardening=False):
    """Run one epoch over the synthetic data."""
    total_loss = 0.0
    n_batches = 0
    idx = 0
    T = panel.shape[0]

    while idx < T - BATCH_SIZE:
        result = make_batch(panel, BATCH_SIZE, idx)
        if result[0] is None:
            break
        s_t, target, mock_layers, idx = result

        if mode == "train":
            optimizer.zero_grad()

        outputs = model(s_t, mock_layers, mode="train", use_hardening=use_hardening)
        signal = outputs["signal"]

        # Simple MSE loss
        loss = F.mse_loss(signal, target.unsqueeze(-1) if target.dim() == 1 else target)

        if mode == "train":
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            model.memory.detach_state()  # Break computation graph between batches

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1), n_batches


# ======================================================================
# Setup
# ======================================================================
banner("DAFT TRAINING LOOP — Synthetic Data Demo")
print(f"  Device: {DEVICE}  |  Batch size: {BATCH_SIZE}")
print(f"  Stage-1 epochs: {N_EPOCHS_STAGE1}")
print(f"  Stage-2 epochs: {N_EPOCHS_STAGE2}")
print(f"  Stage-3 epochs: {N_EPOCHS_STAGE3}")

# Load data
loader = DataLoader({"source": "synthetic", "n_stocks": 50, "n_days": 300, "frequency": "5min"})
panel = loader.load()
print(f"  Data: {panel.shape}")

# Build model
experts_full = nn.ModuleList([
    TrendExpert(input_dim=200, hidden_dim=64),
    ReversalExpert(input_dim=200, hidden_dim=64),
    VolatilityExpert(input_dim=200, hidden_dim=48),
    EventExpert(input_dim=200, hidden_dim=48),
    TrendExpert(input_dim=200, hidden_dim=64),
    ReversalExpert(input_dim=200, hidden_dim=64),
    VolatilityExpert(input_dim=200, hidden_dim=48),
    EventExpert(input_dim=200, hidden_dim=48),
])

router_full = RegimeRouter(input_dim=200, latent_dim=16, n_experts=8, top_k=3)
memory_full = KDAMarketMemory(d_k=128, d_v=64, d_feature=200, use_route_modulation=True)
cdap_full = CrossDimensionAttention(n_experts=8, d_k=128, d_v=64, n_layers=3, joint_dim=64)
hardening_full = HardeningEngine(n_regimes=8, n_experts=8, threshold=30)

model = ExpertEnsemble(
    experts=experts_full, router=router_full,
    memory=memory_full, cross_dim_attn=cdap_full,
    hardening=hardening_full,
)

total_params = sum(p.numel() for p in model.parameters())
print(f"  Model parameters: {total_params:,}")

# ======================================================================
# Stage 1: Independent Expert Training (all params trainable)
# ======================================================================
banner("STAGE 1: Expert Pre-training")
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

stage1_losses = []
t0 = time.time()
for epoch in range(N_EPOCHS_STAGE1):
    loss, n = run_epoch(model, panel, optimizer, mode="train")
    stage1_losses.append(loss)
    print(f"  Epoch {epoch+1:3d}/{N_EPOCHS_STAGE1}  loss={loss:.6f}  batches={n}")

dt1 = time.time() - t0
RESULTS["stage1"] = {
    "epochs": N_EPOCHS_STAGE1,
    "final_loss": stage1_losses[-1],
    "loss_trajectory": stage1_losses,
    "time_seconds": round(dt1, 1),
}

# ======================================================================
# Stage 2: Router + Memory (experts at low LR, router+memory at normal LR)
# ======================================================================
banner("STAGE 2: Router + Memory Training")

# Freeze experts, train router + memory
for expert in model.experts:
    for p in expert.parameters():
        p.requires_grad = False
# Unfreeze router, memory, cdap
for name, param in model.named_parameters():
    if any(k in name for k in ["router", "memory", "cross_dim_attn"]):
        param.requires_grad = True

# Reduce CDAP modulation for Stage 2
model.cross_dim_attn.modulation_strength = 0.1

optimizer2 = torch.optim.Adam(
    [p for p in model.parameters() if p.requires_grad], lr=LR * 0.5
)

stage2_losses = []
t0 = time.time()
for epoch in range(N_EPOCHS_STAGE2):
    loss, n = run_epoch(model, panel, optimizer2, mode="train")
    stage2_losses.append(loss)
    if epoch % 3 == 0 or epoch == N_EPOCHS_STAGE2 - 1:
        print(f"  Epoch {epoch+1:3d}/{N_EPOCHS_STAGE2}  loss={loss:.6f}  batches={n}")

dt2 = time.time() - t0
RESULTS["stage2"] = {
    "epochs": N_EPOCHS_STAGE2,
    "final_loss": stage2_losses[-1],
    "loss_trajectory": stage2_losses,
    "time_seconds": round(dt2, 1),
}

# ======================================================================
# Stage 3: Joint Fine-tuning
# ======================================================================
banner("STAGE 3: Joint Fine-tuning")

# Unfreeze everything, full CDAP
for p in model.parameters():
    p.requires_grad = True
model.cross_dim_attn.modulation_strength = 1.0

optimizer3 = torch.optim.Adam(model.parameters(), lr=LR * 0.01)

stage3_losses = []
t0 = time.time()
for epoch in range(N_EPOCHS_STAGE3):
    loss, n = run_epoch(model, panel, optimizer3, mode="train")
    stage3_losses.append(loss)
    print(f"  Epoch {epoch+1:3d}/{N_EPOCHS_STAGE3}  loss={loss:.6f}  batches={n}")

dt3 = time.time() - t0
RESULTS["stage3"] = {
    "epochs": N_EPOCHS_STAGE3,
    "final_loss": stage3_losses[-1],
    "loss_trajectory": stage3_losses,
    "time_seconds": round(dt3, 1),
}

# ======================================================================
# Stage 4: Hardening Statistics Collection
# ======================================================================
banner("STAGE 4: Hardening Statistics")

# Forward pass over data with inference mode to collect hardening stats
model.eval()
hardening_stats = []
t0 = time.time()
with torch.no_grad():
    # Run hardening collection (non-hardened first, then check stats)
    idx = 0
    T = panel.shape[0]
    n_steps = 0
    while idx < T - BATCH_SIZE:
        result = make_batch(panel, BATCH_SIZE, idx)
        if result[0] is None:
            break
        s_t, target, mock_layers, idx = result

        outputs = model(s_t, mock_layers, mode="inference", use_hardening=True)
        n_steps += 1
        if n_steps >= 50:
            break

    stats = model.hardening.get_stats()
    hardening_stats.append(stats)

dt4 = time.time() - t0

print(f"  Hardening stats after collection:")
print(f"    Decisions: {stats['total_decisions']}")
print(f"    Cached patterns: {stats['n_cached_patterns']}")
print(f"    Baseline entropy: {stats['baseline_entropy']:.4f}")
print(f"    Fast/Slow/Regime-shifts: {stats['n_fast_path']}/{stats['n_slow_path']}/{stats['n_degradations']}")

RESULTS["stage4"] = {
    "hardening_stats": {k: v for k, v in stats.items() if isinstance(v, (int, float))},
    "collection_steps": 50,
    "time_seconds": round(dt4, 1),
}

# ======================================================================
# Summary
# ======================================================================
banner("TRAINING SUMMARY")

total_time = dt1 + dt2 + dt3 + dt4
RESULTS["summary"] = {
    "total_params": total_params,
    "total_time_seconds": round(total_time, 1),
    "device": str(DEVICE),
    "stage1_final_loss": stage1_losses[-1],
    "stage2_final_loss": stage2_losses[-1],
    "stage3_final_loss": stage3_losses[-1],
    "loss_reduction_pct": round((stage1_losses[-1] - stage3_losses[-1]) / stage1_losses[-1] * 100, 1),
}

print(f"  Loss trajectory:")
print(f"    Stage 1 (expert pre-train):   {stage1_losses[-1]:.6f}")
print(f"    Stage 2 (router+memory):       {stage2_losses[-1]:.6f}")
print(f"    Stage 3 (joint fine-tune):     {stage3_losses[-1]:.6f}")
print(f"    Reduction:                     {RESULTS['summary']['loss_reduction_pct']:.1f}%")
print(f"  Total time: {total_time:.1f}s")
print(f"  Parameters: {total_params:,}")

# Save results
results_path = Path(__file__).parent.parent / "outputs" / "training_results.json"
results_path.parent.mkdir(exist_ok=True)
with open(results_path, "w") as f:
    json.dump(RESULTS, f, indent=2, default=str)
print(f"\n  Results saved to: {results_path}")

# Save model checkpoint
ckpt_path = Path(__file__).parent.parent / "checkpoints" / "daft_demo.pt"
torch.save({
    "model_state_dict": model.state_dict(),
    "results": RESULTS,
    "config": {
        "n_experts": 8, "d_k": 128, "d_v": 64,
        "n_layers": 3, "joint_dim": 64, "top_k": 3,
    }
}, ckpt_path)
print(f"  Checkpoint saved to: {ckpt_path}")

print(f"\n  *** TRAINING COMPLETE ***")
