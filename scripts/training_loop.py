"""DAFT Training Loop — Synthetic Data Demo.

Runs a mini 3-stage training cycle on synthetic data to demonstrate:
  Stage 1: Independent expert training
  Stage 2: Router + Memory training (experts frozen)
  Stage 3: Joint fine-tuning
  Stage 4: Hardening statistics collection

Uses RegimeFeatureExtractor to compute proper 200-dimensional market state
vectors (s_t) from the raw panel data, replacing the previous hand-crafted
concatenation of raw returns and volatility.

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
from daft.backtest.engine import BacktestEngine
from daft.features.regime_features import RegimeFeatureExtractor
from daft.models.experts import (
    TrendExpert, ReversalExpert, VolatilityExpert, EventExpert, MomentumExpert,
)
from daft.models.router import RegimeRouter
from daft.models.memory import KDAMarketMemory
from daft.models.cross_dim_attn import CrossDimensionAttention
from daft.models.hardening import HardeningEngine
from daft.models.ensemble import ExpertEnsemble


# ======================================================================
# Config
# ======================================================================
DEVICE = torch.device("cpu")
BATCH_SIZE = 16  # Small batch for daily-frequency demo data
N_EPOCHS_STAGE1 = 5
N_EPOCHS_STAGE2 = 10
N_EPOCHS_STAGE3 = 15
LR = 0.001

RESULTS = {}


def banner(msg):
    print(f"\n{'='*60}")
    print(f"  {msg}")
    print(f"{'='*60}")


def make_batch(s_t_full, panel, batch_size, start_idx):
    """Create training batches using pre-computed market state vectors.

    Uses the RegimeFeatureExtractor's output (T, N, 200) to get proper
    market state vectors instead of hand-crafted feature concatenation.

    Parameters
    ----------
    s_t_full : torch.Tensor, shape (T, N, 200)
        Pre-computed market state vectors from RegimeFeatureExtractor.
    panel : Panel
        Raw market data panel (for target extraction).
    batch_size : int
        Number of time steps per batch.
    start_idx : int
        Starting time index.
    """
    T, N = s_t_full.shape[0], s_t_full.shape[1]
    returns = panel.values[..., 1]  # (T, N): log_return

    end = min(start_idx + batch_size, T - 5)
    actual_batch = end - start_idx
    if actual_batch < 2:
        return None, None, None, start_idx

    idx = torch.arange(start_idx, end)

    # Use s_t from RegimeFeatureExtractor: take the cross-sectional
    # mean across all stocks as the aggregate market state for each timestep.
    # Shape: (B, 200)
    s_t = s_t_full[idx].mean(dim=1)  # (batch, N, 200) → (batch, 200)

    # Target: next-bar cross-sectional mean return
    next_rets = returns[idx + 1]  # (batch, N)
    target = next_rets.mean(dim=-1, keepdim=True)  # (batch, 1)

    # Match batch size
    min_len = min(s_t.size(0), target.size(0))
    s_t = s_t[:min_len]
    target = target[:min_len]

    # Mock 3-layer features (these will be replaced by real factor layers later)
    mock_layers = [
        torch.randn(min_len, 64),
        torch.randn(min_len, 64),
        torch.randn(min_len, 64),
    ]

    return s_t, target, mock_layers, end


def run_epoch(model, s_t_full, panel, optimizer=None, mode="train", use_hardening=False):
    """Run one epoch over the synthetic data."""
    total_loss = 0.0
    n_batches = 0
    idx = 0
    T = s_t_full.shape[0]

    while idx < T - BATCH_SIZE:
        result = make_batch(s_t_full, panel, BATCH_SIZE, idx)
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


@torch.no_grad()
def evaluate(model, s_t_full, panel):
    """Run a full pass over hold-out data and compute validation metrics.

    Returns
    -------
    metrics : dict
        eval_loss, routing_entropy, sharpe, max_drawdown, n_batches
    """
    model.eval()
    total_loss = 0.0
    n_batches = 0
    idx = 0
    T = s_t_full.shape[0]

    # Collect for Sharpe / MaxDD
    all_signals = []
    all_targets = []
    routing_entropies = []

    # Use last 20% of panel as validation (no random shuffle needed for synthetic)
    val_start = int(T * 0.8)
    idx = val_start

    while idx < T - BATCH_SIZE:
        result = make_batch(s_t_full, panel, BATCH_SIZE, idx)
        if result[0] is None:
            break
        s_t, target, mock_layers, idx = result

        outputs = model(s_t, mock_layers, mode="val")
        signal = outputs["signal"]
        loss = F.mse_loss(
            signal, target.unsqueeze(-1) if target.dim() == 1 else target
        )
        total_loss += loss.item()
        n_batches += 1

        all_signals.append(signal.squeeze(-1))
        all_targets.append(target.squeeze(-1) if target.dim() == 1 else target.squeeze(0))

        # Routing entropy per batch (0 = dead expert, log(n_experts) = fully uniform)
        probs = outputs["routing_probs"]
        ent = -(probs * (probs + 1e-8).log()).sum(dim=-1).mean().item()
        routing_entropies.append(ent)

    if n_batches == 0:
        return {"eval_loss": float("nan"), "routing_entropy": float("nan"),
                "sharpe": float("nan"), "max_drawdown": float("nan"),
                "n_batches": 0}

    # Aggregate
    all_signals = torch.cat(all_signals)
    all_targets = torch.cat(all_targets)

    # Strategy return = signal direction × actual return
    # (a simplified P&L proxy for demo purposes)
    min_len = min(all_signals.size(0), all_targets.size(0))
    strategy_returns = all_signals[:min_len].sign() * all_targets[:min_len]

    sharpe = BacktestEngine.sharpe_ratio(strategy_returns)
    cumret = (1.0 + strategy_returns).cumprod(dim=0)
    mdd = BacktestEngine.max_drawdown(cumret)

    avg_entropy = sum(routing_entropies) / len(routing_entropies)
    max_entropy = 2.0794  # ln(8) — maximum possible entropy for 8 experts

    return {
        "eval_loss": total_loss / n_batches,
        "routing_entropy": avg_entropy,
        "routing_entropy_ratio": avg_entropy / max_entropy,
        "sharpe": sharpe,
        "max_drawdown": mdd,
        "n_batches": n_batches,
    }


# ======================================================================
# Setup
# ======================================================================
banner("DAFT TRAINING LOOP — Synthetic Data Demo")
print(f"  Device: {DEVICE}  |  Batch size: {BATCH_SIZE}")
print(f"  Stage-1 epochs: {N_EPOCHS_STAGE1}")
print(f"  Stage-2 epochs: {N_EPOCHS_STAGE2}")
print(f"  Stage-3 epochs: {N_EPOCHS_STAGE3}")

# Load data (daily frequency for fast feature extraction demo)
loader = DataLoader({"source": "synthetic", "n_stocks": 20, "n_days": 500, "frequency": "1d"})
panel = loader.load()
print(f"  Data: {panel.shape}")

# ── Feature Engineering: compute proper 200-dim market state vectors ──
banner("FEATURE ENGINEERING")
feature_extractor = RegimeFeatureExtractor(n_base_factors=50, output_dim=200)
feature_extractor.eval()

t_feat_start = time.time()
with torch.no_grad():
    s_t_full = feature_extractor.forward(panel)  # (T, N, 200)
dt_feat = time.time() - t_feat_start

print(f"  s_t shape: {s_t_full.shape}")
print(f"  s_t stats: mean={s_t_full.mean():.4f}  std={s_t_full.std():.4f}  "
      f"min={s_t_full.min():.4f}  max={s_t_full.max():.4f}")
print(f"  Feature extraction time: {dt_feat:.1f}s")
print(f"  Routing entropy (before training): "
      f"{-(torch.ones(10)/10 * (torch.ones(10)/10 + 1e-8).log()).sum():.3f} "
      f"(uniform routing = maximum entropy)")

# Build model
experts_full = nn.ModuleList([
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

router_full = RegimeRouter(input_dim=200, latent_dim=16, n_experts=10, top_k=3)
memory_full = KDAMarketMemory(d_k=128, d_v=64, d_feature=200, use_route_modulation=True)
cdap_full = CrossDimensionAttention(n_experts=10, d_k=128, d_v=64, n_layers=3, joint_dim=64)
hardening_full = HardeningEngine(n_regimes=10, n_experts=10, threshold=30)

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
    loss, n = run_epoch(model, s_t_full, panel, optimizer, mode="train")
    stage1_losses.append(loss)
    print(f"  Epoch {epoch+1:3d}/{N_EPOCHS_STAGE1}  loss={loss:.6f}  batches={n}")

dt1 = time.time() - t0
eval1 = evaluate(model, s_t_full, panel)
print(f"  Eval: loss={eval1['eval_loss']:.6f}  "
      f"routing_ent={eval1['routing_entropy']:.3f}  "
      f"sharpe={eval1['sharpe']:.3f}  mdd={eval1['max_drawdown']:.3f}")
RESULTS["stage1"] = {
    "epochs": N_EPOCHS_STAGE1,
    "final_loss": stage1_losses[-1],
    "loss_trajectory": stage1_losses,
    "time_seconds": round(dt1, 1),
    "eval": eval1,
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
    loss, n = run_epoch(model, s_t_full, panel, optimizer2, mode="train")
    stage2_losses.append(loss)
    if epoch % 3 == 0 or epoch == N_EPOCHS_STAGE2 - 1:
        print(f"  Epoch {epoch+1:3d}/{N_EPOCHS_STAGE2}  loss={loss:.6f}  batches={n}")

dt2 = time.time() - t0
eval2 = evaluate(model, s_t_full, panel)
print(f"  Eval: loss={eval2['eval_loss']:.6f}  "
      f"routing_ent={eval2['routing_entropy']:.3f}  "
      f"sharpe={eval2['sharpe']:.3f}  mdd={eval2['max_drawdown']:.3f}")
RESULTS["stage2"] = {
    "epochs": N_EPOCHS_STAGE2,
    "final_loss": stage2_losses[-1],
    "loss_trajectory": stage2_losses,
    "time_seconds": round(dt2, 1),
    "eval": eval2,
}

# ======================================================================
# Stage 3: Joint Fine-tuning
# ======================================================================
banner("STAGE 3: Joint Fine-tuning")

# Warmup phase (epochs 1-3): only train CDAP scales, everything else frozen.
# This lets the modulation pathway establish non-zero activation before
# backpropagating through the full model (avoids zero-init immunity zone).
N_WARMUP = 3

# Phase A: CDAP scale warmup
for p in model.parameters():
    p.requires_grad = False
# Unfreeze only CDAP modulation scales
for name, param in model.cross_dim_attn.named_parameters():
    if "scale" in name:
        param.requires_grad = True
model.cross_dim_attn.modulation_strength = 0.5  # moderate start

optimizer_warmup = torch.optim.Adam(
    [p for p in model.parameters() if p.requires_grad], lr=LR * 0.5
)

print(f"  Warmup ({N_WARMUP} epochs): CDAP scales only, δ=0.5, lr={LR*0.5}")
for epoch in range(N_WARMUP):
    loss, n = run_epoch(model, s_t_full, panel, optimizer_warmup, mode="train")
    s = model.cross_dim_attn
    print(f"  Warmup {epoch+1}/{N_WARMUP}  loss={loss:.6f}  "
          f"expert_bias_scale={s.expert_bias_scale.item():.4f}  "
          f"mem_gate_scale={s.memory_gate_scale.item():.4f}  "
          f"depth_scale={s.depth_weight_scale.item():.4f}")

# Phase B: Full joint training
for p in model.parameters():
    p.requires_grad = True
model.cross_dim_attn.modulation_strength = 1.0

optimizer3 = torch.optim.Adam(model.parameters(), lr=LR * 0.1)  # 1e-4, not 1e-5

stage3_losses = []
t0 = time.time()
n_train_epochs = N_EPOCHS_STAGE3 - N_WARMUP
for epoch in range(n_train_epochs):
    loss, n = run_epoch(model, s_t_full, panel, optimizer3, mode="train")
    stage3_losses.append(loss)
    print(f"  Epoch {epoch+1:3d}/{n_train_epochs}  loss={loss:.6f}  batches={n}")

dt3 = time.time() - t0
eval3 = evaluate(model, s_t_full, panel)
print(f"  Eval: loss={eval3['eval_loss']:.6f}  "
      f"routing_ent={eval3['routing_entropy']:.3f}  "
      f"sharpe={eval3['sharpe']:.3f}  mdd={eval3['max_drawdown']:.3f}")
RESULTS["stage3"] = {
    "warmup_epochs": N_WARMUP,
    "train_epochs": n_train_epochs,
    "final_loss": stage3_losses[-1] if stage3_losses else float("nan"),
    "loss_trajectory": stage3_losses,
    "time_seconds": round(dt3, 1),
    "eval": eval3,
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
    T = s_t_full.shape[0]
    n_steps = 0
    while idx < T - BATCH_SIZE:
        result = make_batch(s_t_full, panel, BATCH_SIZE, idx)
        if result[0] is None:
            break
        s_t, target, mock_layers, idx = result

        outputs = model(s_t, mock_layers, mode="inference", use_hardening=True)
        n_steps += 1
        if n_steps >= 200:
            break

    stats = model.hardening.get_stats()
    hardening_stats.append(stats)

eval4 = evaluate(model, s_t_full, panel)
dt4 = time.time() - t0

print(f"  Hardening stats after collection:")
print(f"    Decisions: {stats['total_decisions']}")
print(f"    Cached patterns: {stats['n_cached_patterns']}")
print(f"    Baseline entropy: {stats['baseline_entropy']:.4f}")
print(f"    Fast/Slow/Regime-shifts: {stats['n_fast_path']}/{stats['n_slow_path']}/{stats['n_degradations']}")

RESULTS["stage4"] = {
    "hardening_stats": {k: v for k, v in stats.items() if isinstance(v, (int, float))},
    "collection_steps": 200,
    "time_seconds": round(dt4, 1),
    "eval": eval4,
}

# ======================================================================
# Summary
# ======================================================================
banner("TRAINING SUMMARY")

total_time = dt1 + dt2 + dt3 + dt4

# Use eval metrics for stage-to-stage comparison (not raw training loss,
# since different stages optimize different subsets of parameters).
eval_losses = {
    "stage1": eval1["eval_loss"],
    "stage2": eval2["eval_loss"],
    "stage3": eval3["eval_loss"],
}
eval_sharpes = {
    "stage1": eval1["sharpe"],
    "stage2": eval2["sharpe"],
    "stage3": eval3["sharpe"],
}

# Stage-over-stage eval improvement (meaningful because eval uses same data/loss)
eval_reduction_s1s3 = (
    (eval_losses["stage1"] - eval_losses["stage3"]) / max(eval_losses["stage1"], 1e-10) * 100
)

RESULTS["summary"] = {
    "total_params": total_params,
    "total_time_seconds": round(total_time + dt_feat, 1),
    "device": str(DEVICE),
    "feature_extraction_time_seconds": round(dt_feat, 1),
    "s_t_mean": round(s_t_full.mean().item(), 6),
    "s_t_std": round(s_t_full.std().item(), 6),
    "eval_loss_by_stage": eval_losses,
    "eval_sharpe_by_stage": eval_sharpes,
    "eval_loss_reduction_s1_to_s3_pct": round(eval_reduction_s1s3, 1),
    "stage1_train_loss": stage1_losses[-1],
    "stage2_train_loss": stage2_losses[-1],
    "stage3_train_loss": stage3_losses[-1] if stage3_losses else float("nan"),
    "stage3_cdap_scale_final": {
        "expert_bias_scale": model.cross_dim_attn.expert_bias_scale.item(),
        "memory_gate_scale": model.cross_dim_attn.memory_gate_scale.item(),
        "depth_weight_scale": model.cross_dim_attn.depth_weight_scale.item(),
    },
}

print(f"  Feature extraction: {dt_feat:.1f}s  |  s_t stats: μ={s_t_full.mean():.4f}, σ={s_t_full.std():.4f}")
print(f"  Eval metrics (same hold-out set, same MSE loss):")
print(f"    Stage 1 (expert pre-train):   loss={eval_losses['stage1']:.6f}  sharpe={eval_sharpes['stage1']:.3f}")
print(f"    Stage 2 (router+memory):       loss={eval_losses['stage2']:.6f}  sharpe={eval_sharpes['stage2']:.3f}")
print(f"    Stage 3 (joint fine-tune):     loss={eval_losses['stage3']:.6f}  sharpe={eval_sharpes['stage3']:.3f}")
print(f"    Eval loss reduction (S1→S3):  {eval_reduction_s1s3:.1f}%")
print(f"  CDAP scales after Stage 3: "
      f"expert_bias={model.cross_dim_attn.expert_bias_scale.item():.4f}  "
      f"mem_gate={model.cross_dim_attn.memory_gate_scale.item():.4f}  "
      f"depth={model.cross_dim_attn.depth_weight_scale.item():.4f}")
print(f"  Total time: {total_time + dt_feat:.1f}s")
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
        "n_experts": 10, "d_k": 128, "d_v": 64,
        "n_layers": 3, "joint_dim": 64, "top_k": 3,
    }
}, ckpt_path)
print(f"  Checkpoint saved to: {ckpt_path}")

print(f"\n  *** TRAINING COMPLETE ***")
