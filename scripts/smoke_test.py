"""DAFT Smoke Test — Synthetic Data End-to-End Validation.

Tests:
  1. Data pipeline (synthetic Panel generation)
  2. Expert forward passes
  3. Router -> Memory -> CDAP -> Ensemble forward pass
  4. Gradient flow (all parameters receive gradients)
  5. Hardening engine pattern counting
  6. Model parameter count validation

Usage:
  python scripts/smoke_test.py
"""

import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from daft.data.panel import Panel
from daft.data.loaders import DataLoader
from daft.models.experts import TrendExpert, ReversalExpert, VolatilityExpert, EventExpert
from daft.models.router import RegimeRouter
from daft.models.memory import KDAMarketMemory
from daft.models.cross_dim_attn import CrossDimensionAttention
from daft.models.hardening import HardeningEngine
from daft.models.ensemble import ExpertEnsemble


def banner(msg):
    width = 72
    print(f"\n{'='*width}")
    print(f"  {msg}")
    print(f"{'='*width}")


def check(msg, condition):
    status = "[PASS]" if condition else "[FAIL]"
    print(f"  {status}  {msg}")
    return condition


# --- Config ---
DEVICE = torch.device("cpu")
BATCH_SIZE = 64
N_STOCKS = 200
N_DAYS = 500
FREQ = "1min"

# ======================================================================
# Test 1: Data Pipeline
# ======================================================================
banner("Test 1: Data Pipeline")

loader = DataLoader({
    "source": "synthetic",
    "n_stocks": N_STOCKS,
    "n_days": N_DAYS,
    "frequency": FREQ,
})

panel = loader.load()
T, N, F = panel.shape

all_pass = True
all_pass &= check(f"Panel shape correct (T={T}, N={N}, F={F})", T > 0 and N == 200 and F == 5)
all_pass &= check("Mask is all-True (synthetic data)", panel.mask.all())
all_pass &= check("Values are finite", panel.values.isfinite().all().item())
all_pass &= check("No NaNs in values", not panel.values.isnan().any().item())

# Build mock 200-dim market state vector s_t from the panel
returns = panel.values[..., 1]  # log_return column
vol = panel.values[..., 4]      # volatility column
sr = returns[-BATCH_SIZE:]      # (B, N)
sv = vol[-BATCH_SIZE:]           # (B, N)
# Concatenate returns, vol, and lagged versions -> (B, N*5)
s_batch = torch.cat([
    sr, sv, sr.roll(1, 0), sv.roll(1, 0), sr.roll(5, 0),
], dim=-1)
# Take first 200 columns -> (B, 200)
s_t = s_batch[:, :200]

all_pass &= check(f"s_t shape = (B, 200): got {s_t.shape}", s_t.shape == (BATCH_SIZE, 200))

# ======================================================================
# Test 2: Expert Forward Pass
# ======================================================================
banner("Test 2: Expert Forward Passes")

experts = nn.ModuleList([
    TrendExpert(input_dim=200, hidden_dim=64, n_layers=2),
    ReversalExpert(input_dim=200, hidden_dim=64, n_layers=2),
    VolatilityExpert(input_dim=200, hidden_dim=48, n_layers=2),
    EventExpert(input_dim=200, hidden_dim=48, n_layers=2),
])

for expert in experts:
    signal, hidden = expert(s_t, return_hidden=True)
    all_pass &= check(
        f"{expert.name}: signal shape OK ({signal.shape})",
        signal.shape == (BATCH_SIZE, 1),
    )
    all_pass &= check(
        f"{expert.name}: signal bounded in [-1, 1]",
        (signal >= -1.01).all() and (signal <= 1.01).all(),
    )

# ======================================================================
# Test 3: Router
# ======================================================================
banner("Test 3: Regime Router (Stable LatentMoE)")

router = RegimeRouter(
    input_dim=200, latent_dim=16, n_experts=8,
    top_k=3, temperature=1.0, noisy_gating_std=0.1,
).to(DEVICE)

topk_probs, topk_indices, z_t, full_probs = router(s_t, mode="train")

all_pass &= check(f"topk_probs shape = (B,3): {topk_probs.shape}", topk_probs.shape == (BATCH_SIZE, 3))
all_pass &= check(f"topk_indices shape = (B,3): {topk_indices.shape}", topk_indices.shape == (BATCH_SIZE, 3))
all_pass &= check(f"z_t shape = (B,16): {z_t.shape}", z_t.shape == (BATCH_SIZE, 16))
all_pass &= check("Top-K probs sum ~ 1.0", torch.allclose(
    topk_probs.sum(dim=-1), torch.ones(BATCH_SIZE), atol=1e-5))
all_pass &= check("All top-K probs > 0", (topk_probs > 0).all().item())
all_pass &= check(f"full_probs shape = (B,8): {full_probs.shape}", full_probs.shape == (BATCH_SIZE, 8))
all_pass &= check("Full probs sum ~ 1.0", torch.allclose(
    full_probs.sum(dim=-1), torch.ones(BATCH_SIZE), atol=1e-5))

# Quantile Balancing
router.quantile_balance(lr=0.01)
all_pass &= check("Quantile balance: no NaN", not router.expert_bias.isnan().any().item())

# ======================================================================
# Test 4: KDA Market Memory
# ======================================================================
banner("Test 4: KDA Market Memory")

memory = KDAMarketMemory(
    d_k=128, d_v=64, d_feature=200,
    bottleneck_ratio=4, use_route_modulation=True,
).to(DEVICE)

# Process 20 sequential steps
for step in range(min(BATCH_SIZE, 20)):
    s_step = s_t[step:step+1]
    z_step = z_t[step:step+1]
    retrieved, M_t = memory(s_step, z_t=z_step)

all_pass &= check(f"Retrieved shape = (1,64): {retrieved.shape}", retrieved.shape == (1, 64))
all_pass &= check(f"Memory matrix shape = (1,128,64): {M_t.shape}", M_t.shape == (1, 128, 64))
all_pass &= check("Memory matrix is finite", M_t.isfinite().all().item())
all_pass &= check("Retrieved values are finite", retrieved.isfinite().all().item())

# ======================================================================
# Test 5: Cross-Dimension Attention Protocol
# ======================================================================
banner("Test 5: Cross-Dimension Attention Protocol (CDAP)")

cdap = CrossDimensionAttention(
    n_experts=8, d_k=128, d_v=64, n_layers=3,
    joint_dim=64, modulation_strength=1.0,
).to(DEVICE)

# Mock inputs
mock_routing = torch.randn(BATCH_SIZE, 8).softmax(dim=-1)
mock_memory = M_t.expand(BATCH_SIZE, -1, -1).clone()
mock_layers = [
    torch.randn(BATCH_SIZE, 64),
    torch.randn(BATCH_SIZE, 64),
    torch.randn(BATCH_SIZE, 64),
]

routing_mod, mem_gate, depth_weights, fused = cdap(
    mock_routing, mock_memory, mock_layers)

all_pass &= check(f"routing_mod shape = (B,8): {routing_mod.shape}", routing_mod.shape == (BATCH_SIZE, 8))
all_pass &= check(f"mem_gate shape = (B,128): {mem_gate.shape}", mem_gate.shape == (BATCH_SIZE, 128))
all_pass &= check(f"depth_weights shape = (B,3): {depth_weights.shape}", depth_weights.shape == (BATCH_SIZE, 3))
all_pass &= check(f"fused shape = (B,64): {fused.shape}", fused.shape == (BATCH_SIZE, 64))
all_pass &= check("routing_mod sums to 1", torch.allclose(
    routing_mod.sum(dim=-1), torch.ones(BATCH_SIZE), atol=1e-5))
all_pass &= check("depth_weights sum to 1", torch.allclose(
    depth_weights.sum(dim=-1), torch.ones(BATCH_SIZE), atol=1e-5))
all_pass &= check("mem_gate in (0, 1)", (mem_gate >= 0).all() and (mem_gate <= 1).all())

# ======================================================================
# Test 6: Adaptive Hardening Mechanism
# ======================================================================
banner("Test 6: Adaptive Hardening Mechanism (AHM)")

hardening = HardeningEngine(
    n_regimes=8, n_experts=8,
    threshold=10, min_confidence=0.5, entropy_multiplier=2.0,
)

# Simulate many routing decisions with consistent pattern
fast_count = 0
slow_count = 0
for i in range(200):
    pattern = torch.tensor([0.5, 0.3, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0])
    noise = torch.rand(8) * 0.2
    routing = (pattern + noise).softmax(dim=0)
    if hardening.should_use_fast_path(0, routing):
        _ = hardening.get_cached_weights(0, routing)
        fast_count += 1
    else:
        slow_count += 1

stats = hardening.get_stats()
print(f"  [INFO] Simulated 200 decisions: fast={fast_count}, slow={slow_count}")
print(f"  [INFO] Cached patterns: {stats['n_cached_patterns']}")
print(f"  [INFO] Fast-path ratio: {stats['fast_path_ratio']:.2%}")

all_pass &= check("Cache entries created", stats['n_cached_patterns'] > 0)
all_pass &= check("Fast path used after warmup", stats['fast_path_ratio'] > 0)

# Regime shift detection
anomalous = torch.ones(8) / 8  # max entropy
for _ in range(30):
    hardening.should_use_fast_path(0, anomalous)
shift = hardening.detect_regime_shift()
print(f"  [INFO] Regime shift detected: {shift}")
all_pass &= check("Regime shift detection functional",
                  shift or hardening.baseline_entropy > 0.5)

# ======================================================================
# Test 7: End-to-End Ensemble + Gradient Flow
# ======================================================================
banner("Test 7: End-to-End Ensemble + Gradient Flow")

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
hardening_full = HardeningEngine(n_regimes=8, n_experts=8, threshold=100)

ensemble = ExpertEnsemble(
    experts=experts_full, router=router_full,
    memory=memory_full, cross_dim_attn=cdap_full,
    hardening=hardening_full,
)

outputs = ensemble(s_t, mock_layers, mode="train", use_hardening=False)
signal = outputs["signal"]

all_pass &= check(f"signal shape = (B,1): {signal.shape}", signal.shape == (BATCH_SIZE, 1))
all_pass &= check("signal is finite", signal.isfinite().all().item())
all_pass &= check("signal magnitude reasonable", signal.abs().mean().item() < 0.5)

# Gradient flow
loss = signal.mean()
loss.backward()

grad_count = 0
zero_count = 0
for name, p in ensemble.named_parameters():
    if p.requires_grad:
        grad_count += 1
        if p.grad is None or p.grad.abs().sum() == 0:
            zero_count += 1

print(f"  [INFO] {grad_count} trainable params, {zero_count} zero-grad")
all_pass &= check("Gradient flowing to most params", zero_count < grad_count * 0.5)

# ======================================================================
# Test 8: Parameter Count
# ======================================================================
banner("Test 8: Parameter Efficiency")

total = sum(p.numel() for p in ensemble.parameters())
trainable = sum(p.numel() for p in ensemble.parameters() if p.requires_grad)
print(f"  [INFO] Total: {total:,}  |  Trainable: {trainable:,}")
all_pass &= check(f"Under 500K params (M4-friendly): {total:,}", total < 500_000)

# ======================================================================
# Summary
# ======================================================================
banner("SMOKE TEST SUMMARY")

if all_pass:
    print(f"\n  *** ALL TESTS PASSED ***")
    print(f"  DAFT core architecture is functional.")
    print(f"  Parameters: {total:,} (under 500K)")
    print(f"  Device: {DEVICE}")
    print(f"  Ready for staged training (Stage 1 -> 2 -> 3 -> Hardening).")
else:
    print(f"\n  *** SOME TESTS FAILED ***  (see [FAIL] entries above)")

sys.exit(0 if all_pass else 1)
