"""Run backtest using saved Stage 3 checkpoints (no re-training)."""
import sys, json, time
sys.path.insert(0, "src")
from pathlib import Path
import torch

from daft.data.loaders import DataLoader
from daft.features.regime_features import RegimeFeatureExtractor
from daft.models.factory import build_model
from daft.training.joint_trainer import JointTrainer
from daft.backtest.engine import BacktestEngine

CKPT = Path("checkpoints/stage3")
OUTPUT = Path("outputs")
device = torch.device("cpu")

# Build model (must match run_full_pipeline architecture)
model, layer_proj = build_model(
    cdap_strength=1.0, router_temperature=0.1, noisy_gating_std=0.0,
)

# Load checkpoints
JointTrainer.load_checkpoints(model, layer_proj, str(CKPT))
print("Checkpoints loaded.")

# Generate data
loader = DataLoader({"source": "synthetic", "n_stocks": 50, "n_days": 300, "seed": 42})
panel = loader.load()
print(f"Panel: {panel.shape}")

# Generate signals
t0 = time.time()
extractor = RegimeFeatureExtractor(n_base_factors=50, output_dim=200)
with torch.no_grad():
    s_t_raw = extractor(panel)
s_t_raw = torch.nan_to_num(s_t_raw, nan=0.0, posinf=1e6, neginf=-1e6).clamp(-1e6, 1e6)
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
    s_b = s_t[t].to(device)
    if model.memory.M is None or model.memory.M.size(0) != N:
        model.memory.reset_state(N, device)
    l0 = layer_proj["l0"](s_b)
    l1 = layer_proj["l1"](s_b)
    l2 = layer_proj["l2"](s_b)
    with torch.no_grad():
        out = model(s_b, [l0, l1, l2], mode="inference")
        signals[t] = out["signal"].squeeze(-1).cpu()
    model.memory.detach_state()

print(f"Signals generated: {signals.shape}  ({time.time() - t0:.1f}s)")

# Pad and backtest
signals_padded = torch.cat([torch.zeros(1, N), signals], dim=0)
prices = panel.values[..., 3]

engine = BacktestEngine({
    "transaction_cost_bps": 5.0,
    "slippage_bps": 1.0,
    "top_quantile": 0.2,
    "long_only": False,
})

t0 = time.time()
metrics = engine.run(signals_padded, prices, mask=panel.mask)
print(f"Backtest: {time.time() - t0:.1f}s")

print()
print(f"{'─' * 50}")
print(f"{'Metric':<30s} {'Value':>18s}")
print(f"{'─' * 50}")
for k, v in metrics.items():
    print(f"{k:<30s} {v: 18.4f}")
print(f"{'─' * 50}")

# Save
OUTPUT.mkdir(parents=True, exist_ok=True)
with open(OUTPUT / "backtest_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)
print(f"\nSaved to outputs/backtest_metrics.json")
