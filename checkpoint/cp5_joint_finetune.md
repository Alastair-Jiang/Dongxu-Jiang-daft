# CP5: 联合微调 (Stage 3)

> **状态**: pending
> **依赖**: [CP4 路由+记忆训练](cp4_router_memory_training.md)
> **预计工作量**: 5–6 天
> **后续**: [CP6 硬化统计收集](cp6_hardening.md)

---

## 目标

解冻所有参数（专家 + 路由 + 记忆 + CDAP），以极低学习率 (η=1e-5) 联合微调。
CDAP 调制强度从 0.1 提升到 1.0（完整双向调制）。
核心目标：让 CDAP 三条调制链路在联合训练中自动学习有意义的交互模式，
防止灾难性遗忘。

## 前置依赖

- CP4：路由器 + 记忆已收敛，专家仍冻结
- CP3：各专家的 Stage 1 权重快照保存完好 (用于对比微调前后变化)

---

## 任务清单

### 5.1 JointTrainer 实现

当前 `JointTrainer.train()` 全空：

- [ ] 全部参数 `requires_grad=True`
- [ ] 学习率 `lr=1e-5` (微调专用，防止覆盖 Stage 1+2 学到的知识)
- [ ] CDAP `modulation_strength = 1.0`
- [ ] Gradient clipping: max_norm=0.5 (联合训练梯度噪声更大)
- [ ] Loss: 路由加权专家 loss + 额外正则项（见 5.2）
- [ ] Early stopping: validation IC 15 轮不升即停，恢复最佳 checkpoint
- [ ] 对比微调前后：各专家 loss、路由分布位移 (KL 散度)

**文件**: `src/daft/training/joint_trainer.py`

### 5.2 CDAP 正则化

- [ ] **Sparsity loss**: 鼓励 joint space 中大部分维度接近 0 或 1（稀疏激活），`λ_sparse * ||j · (1-j)||₁`
- [ ] **Consistency loss**: 路由修正后的分布不应与原始分布差太大，`λ_cons * KL(p'_t || p_t)`，防止 CDAP 劫持路由器
- [ ] **Symmetry check**: 三个反向投影都不应坍缩到恒等映射（监控 weight norm）

### 5.3 CDAP 链路有效性验证

- [ ] Router → Memory 链路：对比有/无 route_modulate 时的遗忘门差异
- [ ] Memory → Depth 链路：对比不同记忆状态下层权重的变化
- [ ] Depth → Router 链路：对比深度层信号翻转时路由是否响应
- [ ] **关键实验**：构造一个人工假突破场景序列，验证 CDAP 能否纠正路由分配

### 5.4 灾难性遗忘防护

- [ ] 每 10 epoch 在 Stage-1 的 val 集上 eval 各专家的原始 loss
- [ ] 如果某专家 loss 恶化超过 20%，降低其学习率 10x
- [ ] 记录微调前后专家权重余弦相似度

---

## 验收标准

| # | 标准 | 验证 |
|---|------|------|
| 1 | Joint training val loss 收敛，CDAP 三条链路都有非平凡的调制信号 | Train log |
| 2 | 专家权重微调前后余弦相似度 > 0.85（未遗忘） | 权重分析 |
| 3 | CDAP 正则项有效：joint 激活稀疏度 > 0.3 | Train log |
| 4 | 人工假突破序列上，CDAP 修正后的路由不同于原始路由 | 单元/集成测试 |
| 5 | 最终 model checkpoint 可成功保存和加载 | Roundtrip 测试 |

---

## 快速验证

### 5.1 JointTrainer 冒烟

```python
from daft.training.joint_trainer import JointTrainer
from daft.data.sources import SyntheticSource
import torch, yaml

with open("configs/small.yaml") as f:
    config = yaml.safe_load(f)

# 加载 Stage 2 完成的模型 checkpoint
model = torch.load("checkpoints/stage2_best.pt")  # 从 CP4 产出
# 也可以直接构造（冒烟测试用）
# model = build_full_model(config)

# 解冻全部参数
for p in model.parameters():
    p.requires_grad = True

# CDAP 开到 1.0
model.cross_dim_attn.modulation_strength = 1.0

trainer = JointTrainer(model, config["training"]["stage3"], device=torch.device("cpu"))
panel = SyntheticSource(n_assets=10, n_days=200, seed=42).load()
metrics = trainer.train(panel, panel)
assert metrics["val_loss"] < 10
print(f"✅ JointTrainer 冒烟: loss={metrics['val_loss']:.4f}")
```

### 5.2 灾难性遗忘检查

```python
# 加载微调前的权重对比
stage1_state = torch.load("checkpoints/stage1_best.pt")
cos = torch.nn.CosineSimilarity(dim=0)

for name, param in model.named_parameters():
    if name in stage1_state and "expert" in name:
        sim = cos(param.flatten(), stage1_state[name].flatten())
        assert sim > 0.8, f"{name} 遗忘严重: cos_sim={sim:.3f}"
print("✅ 灾难性遗忘检查: 所有专家权重 cos_sim > 0.8")
```

### 5.3 CDAP 三条链路非平凡

```python
# 构造两个不同的市场状态
s_bull = torch.randn(4, 200) * 0.5 + 0.3   # 模拟牛市
s_bear = torch.randn(4, 200) * 0.5 - 0.3   # 模拟熊市

model.eval()
with torch.no_grad():
    out_bull = model(s_bull, [torch.randn(4,64) for _ in range(3)], mode="val")
    out_bear = model(s_bear, [torch.randn(4,64) for _ in range(3)], mode="val")

# 不同输入应有不同路由分布（CDAP 不能输出恒定值）
route_diff = (out_bull["routing_probs"] - out_bear["routing_probs"]).abs().mean()
assert route_diff > 0.01, f"路由分布在 bull/bear 间无差异: {route_diff:.4f}"

# depth weights 不应是均匀的 [1/3, 1/3, 1/3]
dw = out_bull["depth_weights"]
dw_entropy = -(dw * dw.log()).sum(-1).mean()
assert dw_entropy < 1.09, f"depth weights 太均匀, entropy={dw_entropy:.4f} (max=1.099)"
print(f"✅ CDAP 非平凡: route_diff={route_diff:.4f}, dw_entropy={dw_entropy:.4f}")
```

### 5.4 人工假突破场景

```python
# 构造: 趋势 → 突破（新高）→ 快速回落到突破前
# 期望: CDAP 应在回落时纠正路由
prices = torch.tensor([100, 102, 105, 108, 112, 115, 118, 120, 119, 117, 114, 112])
# 前 7 步趋势向上, 后 5 步回落
s_seq = build_state_sequence(prices)  # 转成 s_t 序列

model.eval()
routing_history = []
for t in range(len(s_seq)):
    out = model(s_seq[t:t+1], [...], mode="val")
    routing_history.append(out["routing_probs"][0])

# 趋势末期的路由 vs 回落后的路由应有明显差异
assert (routing_history[6] - routing_history[-1]).abs().sum() > 0.1
print("✅ 假突破场景: CDAP 在趋势逆转时修正了路由")
```

### 一键验证

```bash
python -m pytest tests/test_joint_trainer.py tests/test_cdap_integration.py -v --tb=short
```

---

## 风险点

| 风险 | 影响 | 缓解 |
|------|------|------|
| CDAP 学到 trivial 解（恒等映射或全零映射） | 整个机制白费 | 多样性正则 + 监控激活模式 |
| 学习率太高导致覆盖 Stage 1/2 成果 | 灾难性遗忘 | 坚持 1e-5，加 early stop + 回退机制 |
| CDAP 调制太激进导致路由不稳定 | 交易信号频繁切换 | Consistency loss 约束 |
