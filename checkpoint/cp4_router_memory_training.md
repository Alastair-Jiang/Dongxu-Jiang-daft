# CP4: 路由 + 记忆训练 (Stage 2)

> **状态**: pending
> **依赖**: [CP3 专家训练](cp3_expert_training.md)
> **预计工作量**: 5–7 天
> **后续**: [CP5 联合微调](cp5_joint_finetune.md)

---

## 目标

在专家权重冻结的前提下，训练 Regime Router 和 KDA Market Memory。
让路由器学会"为每个市场状态分配合适的专家组合"，让记忆模块学会保留和检索
有用的历史 pattern。CDAP 以低调制强度 (δ=0.1) 运行，防止初期不稳定。

## 前置依赖

- CP3：四个专家已独立训练并冻结权重
- CP2：`s_t` 和三层深度表示可用

---

## 任务清单

### 4.1 RouterTrainer 实现

当前 `RouterTrainer.train()` 全空：

- [ ] 训练循环：逐时间步 forward（记忆需序列推进）
- [ ] Loss: `Σ_i w_i · expert_i_loss` — 路由加权的专家预测质量
- [ ] Load balancing: 每 N 步调用 `router.quantile_balance(lr=0.01)`
- [ ] 监控指标：专家利用率分布、路由熵、Top-K 专家平均命中率
- [ ] 验证：用 argmax 路由 (非 soft) 计算 val loss，看路由器是否选对了专家
- [ ] 记忆模块的 `detach_state()` 在 backward 后调用

**文件**: `src/daft/training/router_trainer.py`

### 4.2 序列化训练支持

- [ ] KDA Memory 是有状态的（M_t 跨时间步递推），训练需要顺序遍历时间步
- [ ] 每个 epoch 开始时 `memory.reset_state(B, device)`
- [ ] batch 跨时间步不 shuffle（保持时序依赖）
- [ ] 支持 truncated BPTT（截断反向传播），可配 `bptt_steps`

### 4.3 低强度 CDAP 验证

- [ ] 设置 `cross_dim_attn.modulation_strength = 0.1`
- [ ] 验证 CDAP 信号在低强度时确实不会剧烈改变路由分布
- [ ] 记录路由分布修正前后的 KL 散度，不应 > 0.1

### 4.4 Quantile Balancing 监控

- [ ] 每 100 步 log 一次专家利用率直方图
- [ ] 理想状态：8 个专家的利用率在 0.08–0.17 范围（围绕 1/8=0.125）
- [ ] 利用率 < 0.03 的专家标记为 "可能坍塌"

---

## 验收标准

| # | 标准 | 验证 |
|---|------|------|
| 1 | Stage 2 训练 loss 收敛，val loss 低于随机路由 baseline | Test log |
| 2 | Quantile Balancing 生效：专家利用率方差 < 0.01 | 训练日志直方图 |
| 3 | 记忆模块在 1000 步序列上不 OOM，状态矩阵数值稳定 | Stress test |
| 4 | CDAP modulation_strength=0.1 时路由修正 KL < 0.1 | 训练日志 |
| 5 | 路由器 argmax 选出的专家组合在测试集上整体 Sharpe > 0 | 回测初步验证 |

---

## 快速验证

### 4.1 RouterTrainer 冒烟

```python
from daft.training.router_trainer import RouterTrainer
from daft.data.sources import SyntheticSource
from daft.models import RegimeRouter, KDAMarketMemory, ExpertEnsemble, CrossDimensionAttention, HardeningEngine
from daft.models.experts import TrendExpert, ReversalExpert, VolatilityExpert, EventExpert
import torch, yaml

with open("configs/small.yaml") as f:
    config = yaml.safe_load(f)

# 构建模型（专家冻结）
experts = torch.nn.ModuleList([
    TrendExpert(), ReversalExpert(), VolatilityExpert(), EventExpert()
])
for e in experts:
    for p in e.parameters():
        p.requires_grad = False

router = RegimeRouter(input_dim=200, latent_dim=16, n_experts=8, top_k=3)
memory = KDAMarketMemory(d_k=128, d_v=64, d_feature=200)
cross_dim = CrossDimensionAttention(n_experts=8, d_k=128, d_v=64, modulation_strength=0.1)
hardening = HardeningEngine()
model = ExpertEnsemble(experts, router, memory, cross_dim, hardening)

trainer = RouterTrainer(model, config["training"]["stage2"], device=torch.device("cpu"))

panel = SyntheticSource(n_assets=10, n_days=200, seed=42).load()
metrics = trainer.train(panel, panel)  # 少量数据冒烟
assert metrics["val_loss"] < 10, f"loss 过高: {metrics['val_loss']}"
print(f"✅ RouterTrainer 冒烟: loss={metrics['val_loss']:.4f}")
```

### 4.2 记忆稳定性

```python
# 长序列不退火测试
memory.reset_state(batch_size=2, device=torch.device("cpu"))
zero_input = torch.zeros(2, 200)
Ms = []

for t in range(1000):
    _, M_t = memory(zero_input)
    Ms.append(M_t.norm().item())
    memory.detach_state()

# 零输入下记忆应衰减（遗忘门起作用）
assert Ms[0] > Ms[-1], f"记忆未衰减: M[0]={Ms[0]:.4f}, M[-1]={Ms[-1]:.4f}"
# 平均状态范数应稳定，不爆炸
assert Ms[-1] < 100, f"记忆爆炸: {Ms[-1]:.1f}"
# 不应出现 NaN
assert not any(np.isnan(m) for m in Ms)
print(f"✅ 记忆稳定性: M0={Ms[0]:.2f} → M999={Ms[-1]:.4f}, min={min(Ms):.4f}, max={max(Ms):.2f}")
```

### 4.3 Quantile Balancing 效果

```python
# 模拟极端偏好专家 0
router = RegimeRouter(n_experts=8)
router.activation_counts = torch.tensor([1000., 10., 10., 10., 10., 10., 10., 10.])

before = router.expert_bias.clone()
router.quantile_balance(lr=0.1)
after = router.expert_bias

assert after[0] < before[0], f"过度使用的专家 0 bias 应降低: {before[0]:.3f} → {after[0]:.3f}"
assert after[1] > before[1], f"使用不足的专家 1 bias 应升高: {before[1]:.3f} → {after[1]:.3f}"
print(f"✅ Quantile Balancing: bias delta[0]={after[0]-before[0]:.3f}, delta[1]={after[1]-before[1]:.3f}")
```

### 4.4 CDAP 低强度验证

```python
cross_dim = CrossDimensionAttention(modulation_strength=0.1)
p = torch.softmax(torch.randn(4, 8), dim=-1)
M = torch.randn(4, 128, 64)
h = [torch.randn(4, 64) for _ in range(3)]

p_mod, _, _, _ = cross_dim(p, M, h)
kl = (p * (p / p_mod).log()).sum(-1).mean().item()
assert kl < 0.2, f"低强度时路由修正 KL 应很小, 实际: {kl:.4f}"
print(f"✅ CDAP 低强度: KL(p||p_mod)={kl:.4f} < 0.2")
```

### 一键验证

```bash
python -m pytest tests/test_router.py tests/test_memory.py tests/test_cross_dim_attn.py -v --tb=short
```

---

## 风险点

| 风险 | 影响 | 缓解 |
|------|------|------|
| 专家坍塌 (collapse)：路由器永远只选 1-2 个专家 | Routing 失效 | 调大 noisy_gating_std (0.1→0.2)，加大 quantile_balance lr |
| 记忆数值不稳定 (NaN 或发散) | 训练崩溃 | L2 normalize key vector (已做)，加 grad clip，Monitor M_t 的范数 |
| BPTT 截断导致长程依赖丢失 | 记忆无法学到远距离 pattern | 调大 bptt_steps (默认 128→256) |
