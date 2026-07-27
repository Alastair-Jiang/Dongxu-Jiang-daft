# CP11: Memory Gate 残差化

> **状态**: done
> **依赖**: CP5（联合微调 — 共享 CDAP 引擎）
> **完成日期**: 2026-07-28
> **改动文件**: `src/daft/models/cross_dim_attn.py` (+32 行)

---

## 问题

CDAP 三条反向投影不对称：

| 路径 | 公式 | 默认行为 |
|------|------|----------|
| Router | `softmax(log(p) + δ·bias)` | 恒等（不干预路由） |
| Memory | `σ(W·j)` | 强制砍半（gate=0.5） |
| Depth | `softmax(W·j)` | uniform（不干预深度） |

Memory 路径是唯一一个 (a) 原始信号完全消失 (b) 默认不是恒等映射的路径。这导致 Stage 3 联合微调时 CDAP 一接入就破坏已训练好的 KDA forget gate。

---

## 方案：三层防御

```
Layer 1: 残差          gate = 2·σ(raw)         默认 1.0，不干预
Layer 2: 收缩 prior    raw *= (1 - enh·decay)  增强方向被压缩，抑制自由
Layer 3: L2 正则       loss += λ·(gate-1.0)²   偏离 1.0 要交税（训练时施加）
```

### 开关设计

```python
cdap = CrossDimensionAttention(residual_gate=True)   # 新行为（默认）
cdap = CrossDimensionAttention(residual_gate=False)  # 原始行为
```

运行时也可切换：

```python
model.cross_dim_attn.residual_gate = False  # 关闭，回退到原始
```

---

## 改动详情

### `cross_dim_attn.py`

#### `__init__` 新增

| 位置 | 内容 |
|------|------|
| 参数 | `residual_gate: bool = True` |
| 属性 | `self.residual_gate = residual_gate` |
| 参数 | `self.memory_gate_decay = nn.Parameter(torch.zeros(1))` |

初始化时 `tanh(0) = 0`，decay 不生效，增强方向不受限。

#### `forward` 新增（第 204-216 行）

```python
if self.residual_gate:
    # Layer 2: Shrink prior
    enhance_mask = (memory_gate_raw > 0).float()
    decay = self.memory_gate_decay.tanh().abs()
    memory_gate_raw *= (1.0 - enhance_mask * decay)

    # Layer 1: Residual
    memory_gate = 2.0 * torch.sigmoid(memory_gate_raw)   # ∈ (0, 2)
else:
    memory_gate = torch.sigmoid(memory_gate_raw)          # ∈ (0, 1)
```

---

## 消融实验设计

对比三组实验，验证每层的独立贡献：

| 实验 | `residual_gate` | decay 启用 | L2 正则 | gate 值域 |
|------|:---:|:---:|:---:|:---:|
| A: 基线（原始） | ❌ | — | — | (0, 1) |
| B: 仅残差 | ✅ | 冻结为 0 | ❌ | (0, 2) |
| C: 残差+收缩 | ✅ | 可学 | ❌ | (0, ~1.6) |
| D: 全三层 | ✅ | 可学 | λ=0.001 | (0, ~1.6) |

### 评估指标

1. **Stage 3 收敛速度**: val loss 下降到稳定值的 epoch 数
2. **遗忘门分布**: `alpha`（施加 gate 后）的均值/方差，对比 Stage 2 末
3. **灾难性遗忘**: Stage-1 专家 loss 在 Stage 3 后的恶化幅度
4. **CDAP 链路有效性**: 与 CP5 5.3 节相同的验证指标
5. **Gate 值分布**: gate 的直方图 — 是否集中在 1.0 附近，尾部多长

---

## 验收

| # | 标准 | 状态 |
|---|------|:---:|
| 1 | `residual_gate=True` 时 smoke_test.py 全绿 | ✅ |
| 2 | `residual_gate=False` 时行为与原始完全一致 | ✅ |
| 3 | 新增 `memory_gate_decay` 参数可正常保存/加载 | ✅ |
| 4 | CP5 文档已引用 `residual_gate` 参数 | ✅ |
