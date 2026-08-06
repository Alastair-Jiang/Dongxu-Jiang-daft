"""Quick smoke test for all newly-implemented DAFT modules."""
import sys
sys.path.insert(0, "src")

import torch

# Test 1: data adapters
print("1. Data adapters...")
from daft.data.adapters.baostock_adapter import BaostockAdapter
from daft.data.adapters.yfinance_adapter import YFinanceAdapter
print("   OK")

# Test 2: loaders dispatch
print("2. DataLoader dispatch...")
from daft.data.loaders import DataLoader
loader = DataLoader({"source": "synthetic", "n_stocks": 10, "n_days": 50})
panel = loader.load()
print(f"   synthetic OK: {panel.shape}")

# Test 3: Panel.slice_time
print("3. Panel.slice_time...")
sliced = panel.slice_time(0, 25)
assert sliced.T == 25, f"Expected T=25, got {sliced.T}"
print("   OK")

# Test 4: RouterTrainer import
print("4. RouterTrainer...")
from daft.training.router_trainer import RouterTrainer
print("   OK")

# Test 5: JointTrainer import
print("5. JointTrainer...")
from daft.training.joint_trainer import JointTrainer
print("   OK")

# Test 6: BacktestEngine + MarkowitzOptimizer
print("6. Backtest + Portfolio...")
from daft.backtest.engine import BacktestEngine
from daft.portfolio.markowitz import MarkowitzOptimizer
print("   OK")

# Test 7: Device detection
print("7. Device detection...")
from daft.utils.device import get_device, get_backend_name, device_info
dev = get_device()
print(f"   Device: {dev}  Backend: {get_backend_name()}")

# Test 8: Rank IC
print("8. Rank IC...")
from daft.utils.metrics import rank_info_coefficient, ic_summary, hit_rate
pred = torch.randn(200, 30)
tgt = torch.randn(200, 30)
ic = rank_info_coefficient(pred, tgt, per_timestep=True)
s = ic_summary(ic)
print(f"   IC mean={s['ic_mean']:.4f}, ICIR={s['icir']:.3f}")
print("   OK")

# Test 9: BacktestEngine.run() quick test
print("9. BacktestEngine.run()...")
engine = BacktestEngine({
    "transaction_cost_bps": 5.0,
    "slippage_bps": 1.0,
    "top_quantile": 0.2,
    "long_only": False,
})
signals = torch.randn(100, 10)
close = torch.randn(100, 10).abs() + 5.0
metrics = engine.run(signals, close)
print(f"   Sharpe={metrics['sharpe_ratio']:.4f}, IC={metrics['ic_rank']:.4f}")
print("   OK")

# Test 10: MarkowitzOptimizer.optimize()
print("10. MarkowitzOptimizer.optimize()...")
opt = MarkowitzOptimizer(risk_aversion=3.0, max_weight=0.3)
mu = torch.randn(20) * 0.001
S = torch.randn(20, 100).mm(torch.randn(100, 20)) / 100  # random PSD-ish
S = S @ S.T + torch.eye(20) * 0.01
mask = torch.ones(20, dtype=torch.bool)
w = opt.optimize(mu, S, mask)
assert w.shape == (20,), f"Expected (20,), got {w.shape}"
assert abs(w.sum().item() - 1.0) < 0.01, f"Weights sum to {w.sum().item():.4f}"
print(f"   Sum={w.sum().item():.4f}, Max={w.max().item():.4f}")
print("   OK")

print()
print("=== ALL 10 SMOKE TESTS PASSED ===")
