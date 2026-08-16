"""DirectML 冒烟: DAFT 全模型一步前向+反向+验证 (2026-08-16)。"""
import sys, time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import torch
import torch_directml

from daft.models.factory import build_model
from daft.data.loaders import DataLoader
from daft.features.regime_features import RegimeFeatureExtractor
from daft.training.router_trainer import RouterTrainer

dml = torch_directml.device()
print(f"DML device: {dml} ({torch_directml.device_name(0)})")

# 1) 小数据
loader = DataLoader({"source": "synthetic", "n_stocks": 20, "n_days": 60, "seed": 42})
panel = loader.load()
print(f"Panel: {panel.shape}")

ext = RegimeFeatureExtractor(n_base_factors=50, output_dim=200)
with torch.no_grad():
    s_t = ext(panel)
s_t = torch.nan_to_num(s_t, nan=0.0, posinf=1e6, neginf=-1e6).clamp(-1e6, 1e6)
flat = s_t.reshape(-1, 200)
s_t = ((s_t - flat.mean(0, keepdim=True)) / flat.std(0, keepdim=True).clamp(min=1e-4)).clamp(-10, 10)
print(f"s_t: {tuple(s_t.shape)} on CPU")

# 2) 模型搬到 DML
model, layer_proj = build_model(cdap_strength=0.1)
model = model.to(dml)
layer_proj = layer_proj.to(dml)
print(f"model on: {next(model.parameters()).device}")

# 3) 一步训练 forward+backward
trainer = RouterTrainer(model=model, config={"epochs": 1, "batch_size": 40}, device=dml)
train_s, train_t, train_m, train_tidx = trainer._build_dataset(panel)
print(f"dataset rows: {train_s.shape[0]}")

# 手动跑 3 个 batch 的训练步骤(绕过完整 epoch 以快速暴露 op 问题)
loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(train_s, train_t, train_m, train_tidx),
    batch_size=40, shuffle=False,
)
opt = torch.optim.Adam(
    list(model.router.parameters()) + list(model.memory.parameters())
    + list(model.cross_dim_attn.parameters()) + list(layer_proj.parameters()),
    lr=1e-3,
)
t0 = time.time()
model.train()
layer_proj.train()
model.memory.reset_state(1, dml)
n_steps = 0
for s_b, t_b, m_b, ti_b in loader:
    s_b, t_b, m_b = s_b.to(dml), t_b.to(dml), m_b.to(dml)
    B = s_b.size(0)
    if model.memory.M is None or model.memory.M.size(0) != B:
        model.memory.reset_state(B, dml)
    l0 = layer_proj["l0"](s_b); l1 = layer_proj["l1"](s_b); l2 = layer_proj["l2"](s_b)
    out = model(s_b, [l0, l1, l2], mode="train")
    routing = out["routing_probs"]
    with torch.no_grad():
        losses = [e.compute_loss(e(s_b), t_b, m_b) for e in model.experts]
    loss = sum(routing.mean(0)[i] * losses[i] for i in range(model.n_experts))
    ent = -(routing * (routing + 1e-8).log()).sum(-1).mean()
    loss = loss - 0.01 * ent
    opt.zero_grad(); loss.backward(); opt.step()
    model.memory.detach_state()
    n_steps += 1
    if n_steps >= 3:
        break
dt = time.time() - t0
print(f"DML 训练 3 步 OK: {dt:.1f}s, loss={float(loss.cpu()):.4f}")

# 4) 验证推理
model.eval()
with torch.no_grad():
    s_b = train_s[:40].to(dml)
    l0 = layer_proj["l0"](s_b); l1 = layer_proj["l1"](s_b); l2 = layer_proj["l2"](s_b)
    out = model(s_b, [l0, l1, l2], mode="inference")
print(f"推理 OK, signal: {tuple(out['signal'].shape)}")
print("SMOKE PASSED")
