# DAFT 引导式深度导览

> 面向项目作者 Alastair 的完整代码走读。
> 目标：读完这份文档后，你对 DAFT 的每一个模块都能讲清楚"它做什么、怎么做的、为什么这样做"。

---

## 目录

1. [宏观架构：数据如何流过 DAFT](#1-宏观架构)
2. [数据层：Panel 和 DataLoader](#2-数据层)
3. [特征工程：从 OHLCV 到 200 维 s_t](#3-特征工程)
4. [模型组件 1：RegimeRouter](#4-regimerouter)
5. [模型组件 2：KDAMarketMemory](#5-kdamarketmemory)
6. [模型组件 3：CrossDimensionAttention (CDAP)](#6-cdap)
7. [模型组件 4：Adaptive Hardening Mechanism](#7-ahm)
8. [专家池：5 种策略专家](#8-专家池)
9. [集成层：ExpertEnsemble](#9-expertensemble)
10. [训练循环：4 阶段训练](#10-训练循环)
11. [测试套件：371 个测试的结构](#11-测试套件)
12. [已知问题和下一步](#12-已知问题)

---

## 1. 宏观架构

### 1.1 一句话总结

DAFT 是一个 **4 组件 + 5 专家的 PyTorch 量化交易模型**。核心创新是让路由（哪个专家被激活）、记忆（过去市场状态的压缩表示）、深度（多层特征的权重）三个维度在联合空间中互相调制，而不是各自独立决策。

### 1.2 数据流

```
原始 OHLCV 数据
    │
    ▼
DataLoader ──→ Panel (T, N, F)     T=时间步, N=股票数, F=特征数
    │
    ▼
RegimeFeatureExtractor ──→ s_t (T, N, 200)   市场状态向量
    │
    ▼  (每个时间步取一批 B 个样本)
    │
    ├──→ RegimeRouter ──→ routing_probs (B,8), z_t (B,16)
    │
    ├──→ KDAMarketMemory ──→ retrieved (B,64), M_t (B,128,64)
    │
    ├──→ 8 个策略专家 ──→ expert_outputs (B,8)
    │
    └──→ CDAP ──→ 调制后的 routing/ memory_gate/ depth_weights
              │
              ▼
         ExpertEnsemble ──→ signal (B,1)   最终交易信号
              │
              ▼
         (可选) Hardening ──→ 缓存快路径 或 完整 CDAP
```

### 1.3 关键维度常数

| 符号 | 值 | 含义 |
|---|---|---|
| input_dim | 200 | s_t 的维度，进入 Router 和 Memory 的输入 |
| latent_dim | 16 | 路由潜在空间 z_t 的维度 |
| n_experts | 8 | 策略专家总数 |
| top_k | 3 | 每个前向传播激活的专家数 |
| d_k | 128 | Memory 的 key 维度（= 记忆槽位数）|
| d_v | 64 | Memory 的 value 维度 |
| n_layers | 3 | 深度层的数量 |
| joint_dim | 64 | CDAP 联合空间的维度 |

---

## 2. 数据层

### 2.1 Panel (`src/daft/data/panel.py`)

Panel 是 DAFT 最基本的数据结构——一个 **mask-aware** 的 T×N×F 张量。

```python
@dataclass
class Panel:
    values: torch.Tensor   # (T, N, F) — 特征值
    mask: torch.Tensor     # (T, N, F) — True=可交易, False=停牌/涨跌停
    dates: list            # 长度为 T 的时间索引
    asset_ids: list        # 长度为 N 的股票代码
    feature_names: list    # 长度为 F 的特征名
```

**为什么需要 mask？** A 股有涨跌停和停牌。如果某只股票涨停了，你无法买入，它当天的数据不应该参与计算。Mask 贯穿整个特征工程管线，确保被 mask 的数据不会"污染"因子计算。

### 2.2 DataLoader (`src/daft/data/loaders.py`)

目前只有合成数据源可用：

```python
loader = DataLoader({"source": "synthetic", "n_stocks": 200, "n_days": 500, "frequency": "5min"})
panel = loader.load()  # → Panel(T, N, 5)
```

合成数据生成 5 个基础特征：
- `f0`: close（几何布朗运动模拟的价格）
- `f1`: log_return
- `f2`: volume
- `f3`: volume_ratio（相对均量的比例）
- `f4`: volatility_20（20 日滚动波动率）

真实数据源（Baostock A 股、YFinance）标记为 `NotImplementedError`。

---

## 3. 特征工程

特征工程管线由 4 个模块组成，按依赖关系自底向上排列。

### 3.1 TensorFactorEngine (`src/daft/features/tensor_factors.py`)

**定位**：最底层的计算原语库。所有操作都接受 `(T, N)` 的张量和 `(T, N)` 的 bool mask。

**7 个原语**及其金融含义：

| 原语 | 金融含义 | 实现技巧 |
|---|---|---|
| `rank(x, mask)` | 截面排名——这只股票在全市场排第几 | 双 argsort：`argsort(argsort(x))`，mask=False 的资产给 0.5 |
| `corr(x, y, window, mask)` | 滚动相关性——量价相关性等 | `unfold` 向量化 + mask 加权均值 |
| `ewma(x, span, mask)` | 指数加权移动平均 | 递归循环 T 步，`torch.where` 向量化 N 维 |
| `ts_delta(x, d, mask)` | 滞后差分——动量、加速度 | `x[t] - x[t-d]`，mask 级联 |
| `ts_sum(x, d, mask)` | 滚动求和——累计收益 | `unfold` + `sum(dim=-1)` |
| `ts_std(x, d, mask)` | 滚动标准差——波动率 | 两步法：先算 mask 均值，再算 mask 方差 |
| `ts_mean(x, d, mask)` | 滚动均值——移动平均线 | ts_sum / n_valid |

**关键设计原则**：mask=False 的值替换为 0（不参与求和/求均值），但计数 `n_valid` 只统计 mask=True 的步数。

### 3.2 FreqFeatureExtractor (`src/daft/features/freq_features.py`)

**定位**：从价格序列中提取频域特征。灵感来自 Super-Linear 的 FFT-gated MoE router。

**原理**：
1. 取 `lookback` 长度的 returns 序列
2. 去均值（移除 DC 分量）
3. FFT → 取正频率部分 → 功率谱密度 (PSD = |DFT|²)
4. 归一化到 sum=1
5. 聚合为低/中/高频段功率

**为什么频域有用？** 不同市场状态有不同的频谱特征：
- 趋势市：低频功率占优（价格缓慢漂移）
- 震荡市：中频功率占优（均值回归）
- 高波动：全频段功率抬升
- 事件驱动：高频瞬态尖峰

输出：`(T, N, n_freq_bins + 3)` — PSD 各频段 + low/mid/high 三频段聚合。

### 3.3 RegimeFeatureExtractor (`src/daft/features/regime_features.py`)

**定位**：最重要的特征工程模块。从 Panel 的 5 个基础特征构建 200 维市场状态向量 s_t。

**内部结构**：6 个特征组，每组一个私有方法：

```
200 dims = 45 (价格动力学) + 35 (量动力学) + 40 (波动率结构)
         + 20 (微结构)    + 30 (截面特征)  + 30 (动量/因子)
```

**各组内容举例**：

| 组 | 典型特征 | 用的原语 |
|---|---|---|
| 价格动力学 (45) | 多周期收益(1,2,3,5,10,20,60日)、价格相对均线偏离、收益加速度 | ts_sum, ts_mean, ts_delta, ewma |
| 量动力学 (35) | 量比(5/20/60日)、量趋势、量价相关性、换手率代理 | ewma, corr, ts_std, rank |
| 波动率结构 (40) | 多窗口波动率、vol-of-vol、收益/风险比、波动率偏度 | ts_std, ewma, ts_delta, rank |
| 微结构 (20) | Amihud 非流动性、价格冲击代理、价差代理、Roll 模型 | ts_sum, ewma, corr |
| 截面特征 (30) | 收益排名、量排名、截面离散度、z-score、市场相对强度 | rank, 逐时间步的 std/mean |
| 动量/因子 (30) | 动量(5/20/60)、短期反转、RSI、布林带位置、MACD | ts_sum, ewma, ts_std, ts_delta |

**forward() 流程**：
1. 从 Panel 提取 2D mask: `panel.mask[:,:,0]`
2. 依次调用 6 个组方法，每个返回 `(T, N, group_dims)`
3. `torch.cat(groups, dim=-1)` → `(T, N, 200)`
4. `nan_to_num` + `clamp(-50, 50)` 安全处理
5. mask=False 的位置填 0

### 3.4 Legacy Factors (`src/daft/features/legacy_factors.py`)

**定位**：35 个经典手写 alpha 因子，来自 ml-quant-trading。

每个因子的签名：`factor_name(panel, engine) → (T, N)`

**因子家族**：
- `better_*` (8个): VWAP 偏离、量加权动量、日内价格位置
- `best_*` (6个): Close-location 动量、日内偏度、隔夜收益分离
- `old_*` (5个): corr/rank 复合因子
- `stock_*` (6个): 个股-市场相关性、相对强度、资金流
- `extra_*` (5个): 换手率、成交额因子
- `add_*` (5个): 动量×流动性、趋势强度×动量一致性

注册表 `LEGACY_FACTOR_REGISTRY` 提供批量调用接口：
```python
from daft.features.legacy_factors import compute_all_factors
factors = compute_all_factors(panel)  # → dict[str, Tensor]
```

---

## 4. RegimeRouter

文件：`src/daft/models/router.py` (~190 行)

### 4.1 设计思路

Kimi K3 的 Stable LatentMoE：不在原始特征空间做路由，而是先把输入投影到一个低维潜在空间（latent space），在潜在空间里做路由决策。低维空间更稳定，噪声更少。

### 4.2 前向传播（4 步）

```
s_t (B, 200)
    │
    ▼ Linear(200→100) → SiLU → Linear(100→16) → LayerNorm
    │
    ▼ z_t (B, 16)     ←── 潜在 regime 向量
    │
    ▼ Linear(16→8)    ←── 从潜在空间路由到 8 个专家
    │
    ▼ logits (B, 8)
    │  + expert_bias       ←── Quantile Balancing 偏置
    │  + noise (if train)  ←── 探索噪声 (std=0.1)
    │
    ▼ softmax(logits / temperature)
    │
    ▼ topk_probs (B, 3), topk_indices (B, 3)   ←── Top-3 稀疏化 + 重归一化
```

**关键细节**：
- **温度调度**: train=1.0 (软路由), val=0.5, inference=0.1 (接近离散)
- **探索噪声**: 训练时加高斯噪声 (std=0.1)，防止路由过早收敛到次优解
- **重归一化**: Top-K 选中的概率除以选中概率之和，保证 sum=1

### 4.3 Quantile Balancing

替代传统 MoE 的 auxiliary load-balancing loss（如 Switch Transformer 的负载均衡损失）。

```python
def quantile_balance(self, lr=0.01):
    current_frac = activation_counts / activation_counts.sum()
    target_frac = 1.0 / n_experts       # 均匀分布
    delta = lr * (target_frac - current_frac)
    expert_bias += delta                 # 少用的专家 → bias↑，多用的 → bias↓
```

**优势**：不需要额外的 loss term（auxiliary loss 会与主任务 loss 竞争），通过直接调整 bias 来实现负载均衡。

---

## 5. KDAMarketMemory

文件：`src/daft/models/memory.py` (~150 行)

### 5.1 设计思路

Kimi K3 的 KDA (Kernelized Delta Attention) 在 NLP 中用于线性注意力。DAFT 将其改造为金融时序的记忆模块。核心是一个可更新的 key-value 记忆矩阵 M。

### 5.2 记忆更新公式（Delta-Rule）

```
M_t = α_t · M_{t-1}  +  (1 - α_t) · ΔM_t

其中：
  α_t = sigmoid(ForgetGate(s_t, z_t))   ←── 遗忘门，在 (0,1) 之间
  ΔM_t = outer(k_t, v_t)                ←── 新记忆 = key ⊗ value
  k_t = Linear(s_t)                     ←── 从市场状态提取 key
  v_t = Linear(s_t)                     ←── 从市场状态提取 value
```

**直觉**：
- 当市场状态稳定（趋势延续），α_t → 1，保留旧记忆
- 当市场状态变化（regime shift），α_t → 0，写入新记忆
- 这类似于 LSTM 的遗忘门，但作用在 key-value 记忆矩阵上

### 5.3 记忆检索

```
retrieved = M · query(s_t)    ←── 用当前状态查询记忆，得到 (B, d_v)
```

检索结果是一个 64 维向量，代表"基于历史记忆，当前市场状态应该产生什么信号"。

### 5.4 Router Modulation

当 `use_route_modulation=True` 时，Router 的潜在向量 z_t 会输入到 Memory 的遗忘门中：

```
α_t = sigmoid(W_forget · [s_t, z_t])
```

这是 CDAP 架构中 Router → Memory 的连接——路由决策影响记忆的更新速率。

### 5.5 关键实现细节

- **RMSNorm**: 自定义的 Root Mean Square Normalization（比 LayerNorm 更高效），用于稳定 key/query
- **detach_state()**: 每个 batch 后切断 M 的计算图。原因：如果不 detach，梯度会跨 batch 反向传播，导致显存爆炸
- **Bottleneck 遗忘门**: 遗忘门的线性层使用低秩分解（ratio=4），减少参数量
- **reset_state(B, device)**: 初始化 M 为全零矩阵，支持 batch size 变化时自动重新初始化

---

## 6. CrossDimensionAttention (CDAP)

文件：`src/daft/models/cross_dim_attn.py` (~200 行)

### 6.1 核心创新（你的原创贡献）

传统架构中，Router、Memory、Layer Fusion 三个模块独立决策。CDAP 让它们在联合空间中互相调制：

```
e = expert_projection(routing_probs)    ←── 路由分布 → expert 空间
m = memory_projection(memory_matrix)    ←── 记忆矩阵 → memory 空间
d = depth_projection(layer_outputs)     ←── 深度层输出 → depth 空间

joint = e ⊙ m ⊙ d                      ←── element-wise product (⊙)

expert_bias  ← reverse_proj_e(joint)   ←── 反向投影：联合空间 → 路由调制
memory_gate  ← reverse_proj_m(joint)   ←── 反向投影：联合空间 → 记忆门控
depth_weight ← reverse_proj_d(joint)   ←── 反向投影：联合空间 → 深度权重
```

**直觉**：如果 Memory 说"市场波动很大"，Depth 说"L0 层（原始数据层）信号很强"，那么 Expert 路由应该偏向 VolatilityExpert——三个维度的信息在联合空间中融合后再反向调制每个维度。

### 6.2 element-wise product 的物理意义

选择乘法而非加法（concatenation + Linear）的原因：
- **乘法是 AND 门**: e ⊙ m ⊙ d = 0 当且仅当任意一个维度为 0。如果 Memory 对当前状态没有信息（m≈0），整个调制信号为 0——不会做无意义的调制
- **乘法是稀疏门控**: 联合空间的非零区域对应三个维度都有信息的"高置信度"区域
- **测试可验证**: `test_elementwise_product_inductive_bias` 验证了"任一输入为零 → gate=0.5（中性）"

### 6.3 Modulation Scales 和 Zero-Init

```python
self.expert_bias_scale = nn.Parameter(torch.zeros(1))   # 初始化为 0
self.memory_gate_scale = nn.Parameter(torch.zeros(1))
self.depth_weight_scale = nn.Parameter(torch.zeros(1))
```

**为什么初始化为 0？** 训练开始时，CDAP 的调制应该是"不做任何事"——让专家先各自学好，再逐步引入跨维度调制。这类似于 Transformer 中残差连接的思路。

**Zero-Init Immunity Zone**（10 轮自检中的关键发现）：因为 scales=0，初始时 `modulation_strength` 参数无效——δ=0.5 和 δ=1.0 产生完全相同的输出。这不是 bug，但意味着测试时不能直接用随机初始化的模型来验证 modulation_strength 的效果。

### 6.4 前向传播细节

```python
def forward(routing_probs, memory_matrix, layer_outputs):
    # 1. 投影到联合空间
    e = expert_proj(routing_probs)   # (B, joint_dim)
    m = memory_proj(memory_matrix)   # (B, joint_dim)
    d = depth_proj(layer_outputs)    # (B, joint_dim)
    joint = e * m * d                # element-wise product

    # 2. 反向投影 + 学习到的 scale
    expert_bias = expert_bias_scale * δ * reverse_proj_e(joint)  # δ = modulation_strength
    memory_gate = sigmoid(memory_gate_scale * δ * reverse_proj_m(joint))
    depth_w_raw = depth_weight_scale * δ * reverse_proj_d(joint)

    # 3. 应用调制
    routing_mod = softmax(log(routing_probs) + expert_bias)  # log-空间加偏置 → 保持概率
    depth_w = softmax(depth_w_raw)                           # 权重归一化
    fused = sum(depth_w[k] * layer_outputs[k] for k in range(n_layers))

    return routing_mod, memory_gate, depth_w, fused
```

**关键设计**：routing 调制在 log-空间做（加偏置后 softmax），保证输出仍是合法概率分布。

---

## 7. Adaptive Hardening Mechanism (AHM)

文件：`src/daft/models/hardening.py` (256 行)

### 7.1 设计动机

Kimi K3 使用静态的 3:1 KDA-to-full-attention 比例。DAFT 将其泛化为**数据驱动、regime-自适应**的路由策略。

**核心类比**：
- 熟练的司机在熟悉的路上用"自动驾驶"（快路径 = 缓存的 hardened weights）
- 遇到陌生路况时切换回"全神贯注"（慢路径 = 完整 CDAP）
- 路况剧烈变化时强制降级（entropy guard = regime shift 检测）

### 7.2 四个核心机制

#### (1) Pattern Counter
```python
pattern = discretize_pattern(routing_probs)  # Top-K indices → 排序 → tuple
key = (regime_id, pattern)
pattern_counter[key] += 1
```
将软的 routing distribution 离散化为一个模式 ID，追踪每个 (regime, expert_pattern) 组合的出现频率。

#### (2) Cache Builder
```python
if pattern_counter[key] >= threshold and confidence >= min_confidence:
    cache[key] = routing_probs.clone().detach()  # 冻结权重
```
当某个模式出现足够多次（≥θ=30）且占比较高（≥ρ=0.95），说明这是"熟悉的常规模式"→ 缓存其路由权重。

#### (3) Entropy Guard
```python
if recent_entropy_avg > entropy_multiplier * baseline_entropy:
    return True  # 检测到 regime shift → 降级到完整 CDAP
```
路由熵是市场状态不确定性的代理指标。熵突然飙升 → 市场进入了不熟悉的 regime → 禁用缓存，回到完整 CDAP。

基线熵计算：滚动窗口（30-1000 步）的平均值。这个参数经过一轮重要修复（从 100 降到 30 步）。

#### (4) Staleness Eviction
```python
if age > max_age and hit_count < 10:
    stale_keys.append(key)  # 长时间不用的缓存条目 → 删除
```
市场结构会变化（如注册制改革），旧的缓存模式可能不再适用。

### 7.3 在 Ensemble 中的集成

```python
# 推理模式下的分支逻辑：
if use_hardening and hardening.should_use_fast_path(regime_id, routing_avg):
    # 快路径：使用缓存权重，跳过完整 CDAP
    final_routing = hardening.get_cached_weights(...)
    depth_weights = uniform_weights  # 均匀混合各层
else:
    # 慢路径：完整 CDAP 计算
    routing_mod, memory_gate, depth_weights, fused = cdap(...)
```

### 7.4 关键修复历史
- **`baseline_entropy` 阈值为 100 步** → 在 50 步的收集阶段永远达不到 → 改为 30 步
- **硬化收集步数 50** → 不足以创建缓存 → 改为 200 步

---

## 8. 专家池

文件：`src/daft/models/experts/` 目录下 5 个文件

### 8.1 BaseExpert (`base_expert.py`)

所有专家的父类。提供：
- 共享的 MLP 骨架（Linear → SiTU → Linear）
- `compute_loss(signal, target, mask)`：mask-aware MSE loss
- `return_hidden=True` 时返回中间层表示（供 CDAP 的深度层使用）

### 8.2 SiTU 激活函数

```python
def situ(x):
    return torch.sigmoid(x) * torch.tanh(x)
```

**为什么不用 GELU/ReLU？**
- σ(x) 是门控：0~1，控制信息流
- tanh(x) 是激活：-1~1，对称
- 组合效果：正值 → 接近 tanh（门开），负值 → 接近 0（门关）
- 在金融数据上的直觉：正收益和负收益的"意义"不对称（跌 10% vs 涨 10% 对投资者影响不同），SiTU 天然支持这种非对称性

### 8.3 四种专家

| 专家 | 隐藏维 | 层数 | 策略定位 |
|---|---|---|---|
| TrendExpert | 64 | 2 | 趋势跟踪——顺势而为 |
| ReversalExpert | 64 | 2 | 均值回归——抄底逃顶 |
| VolatilityExpert | 48 | 2 | 波动率交易——做多/做空波动率 |
| EventExpert | 48 | 2 | 事件驱动——财报/政策/突发事件 |

每个专家有 2 个实例（共 8 个），给 Router 更多选择空间。隐藏维和层数的差异使得不同专家有不同的模型容量。

### 8.4 Dropout 的作用

每个专家有 `Dropout(0.1)`。在训练时：
- 提供正则化，防止过拟合
- 产生一定的随机性——同一输入多次前向传播产生略有不同的输出
- 测试 `test_train_mode_produces_variation` 验证了这点

---

## 9. ExpertEnsemble

文件：`src/daft/models/ensemble.py` (177 行)

### 9.1 定位

整个 DAFT 的顶层集成模块。它不包含自己的可学习参数——所有参数在子模块中。

### 9.2 forward() 流程

```python
def forward(s_t, layer_outputs, mode="train", use_hardening=False):
    # Step 1: 路由
    topk_probs, topk_indices, z_t, full_probs = router(s_t, mode=mode)

    # Step 2: 记忆检索
    retrieved, M_t = memory(s_t, z_t=z_t)

    # Step 3: 专家前向传播（所有 8 个专家都计算）
    for expert in experts:
        signal_i, hidden_i = expert(s_t, return_hidden=True)
    expert_outputs = stack(signals)  # (B, 8)

    # Step 4: CDAP 或硬化快路径
    if not use_hardening or not hardening.should_use_fast_path(...):
        routing_mod, memory_gate, depth_weights, fused = cdap(...)
        final_routing = routing_mod
    else:
        final_routing = hardening.get_cached_weights(...)
        depth_weights = uniform
        fused = uniform_blend(layer_outputs)

    # Step 5: 加权融合
    signal = (final_routing * expert_outputs).sum(dim=-1)  # Σw_i · expert_i
    signal += 0.1 * fused.mean(dim=-1)  # 深度层修正

    return {"signal": signal, "routing_probs": final_routing, ...}
```

### 9.3 三种模式

| 模式 | 路由温度 | 噪声 | 硬化 | 用途 |
|---|---|---|---|---|
| train | 1.0 | 0.1 | 无 | 训练 |
| val | 0.5 | 无 | 无 | 验证 |
| inference | 0.1 | 无 | 可选 | 推理/回测 |

### 9.4 输出字典

```python
{
    "signal": (B, 1),          # 最终交易信号（预期收益）
    "routing_probs": (B, 8),   # 最终路由权重（可能被 CDAP 调制过）
    "regime_id": (B,),         # 离散 regime ID（仅推理模式）
    "depth_weights": (B, 3),   # 深度层权重
    "fused_layers": (B, 64),   # 融合后的深度层输出
    "metadata": {
        "mode": str,
        "fast_path_used": bool,
        "hardening_stats": dict,
    }
}
```

---

## 10. 训练循环

文件：`scripts/training_loop.py` (540 行)

### 10.1 四阶段训练

```
Stage 1: 独立专家训练
  ├─ 所有参数可训练
  ├─ 5 epochs, lr=0.001
  └─ 目标：每个专家学会基础预测能力

Stage 2: Router + Memory 训练
  ├─ 专家参数冻结
  ├─ Router, Memory, CDAP 可训练
  ├─ CDAP δ = 0.1（低强度调制）
  ├─ 10 epochs, lr=0.0005
  └─ 目标：路由学会选择专家，记忆学会压缩历史

Stage 3: 联合微调
  ├─ Warmup (3 epochs): 只训练 CDAP scales
  │   └─ 目标：突破 zero-init immunity zone
  ├─ Full training (12 epochs): 所有参数可训练
  ├─ CDAP δ = 1.0（完全调制）
  ├─ lr=0.0001
  └─ 目标：端到端优化

Stage 4: 硬化统计收集
  ├─ 200 steps inference + use_hardening=True
  ├─ 收集 (regime, pattern) 频率
  └─ 目标：为推断时的快路径建立缓存
```

### 10.2 损失函数

当前使用简单的 MSE Loss：`MSE(signal, next_bar_return)`。这是最基础的设定——未来应该用排序损失（pairwise）、夏普比率相关的损失等。

### 10.3 evaluate() 函数

在 hold-out 集（最后 20% 数据）上计算：
- eval_loss：验证损失
- routing_entropy：路由熵（0 = 只用一个专家，ln(8)=2.079 = 均匀使用）
- sharpe：年化夏普比率
- max_drawdown：最大回撤

---

## 11. 测试套件

### 11.1 总览

```
371 tests, 100% pass, ~9 秒 (CPU)

tests/
├── conftest.py             共享 fixtures (router, memory, cdap, hardening, ensemble)
├── test_router.py          ~30 tests — 路由形状/温度/负载均衡/梯度
├── test_memory.py          ~25 tests — 记忆读写/状态管理/1000步稳定性
├── test_cross_dim_attn.py  ~30 tests — CDAP 联合空间/调制/梯度/zero-init
├── test_hardening.py       ~30 tests — 模式离散化/缓存/熵守卫/驱逐
├── test_experts.py         ~40 tests — 4种专家前向/SiTU/Dropout/masked loss
├── test_ensemble.py        ~35 tests — 端到端/硬化快慢路径/参数分解
├── test_backtest.py        ~18 tests — Sharpe/MaxDD/IC
├── test_data.py            ~25 tests — 数据加载/Panel
├── test_features.py        ~101 tests — 特征工程完整测试 ★ NEW
├── test_portfolio.py       ~15 tests — 组合优化接口
└── test_training.py        ~10 tests — 训练器接口
```

### 11.2 conftest.py 的共享 Fixtures

```python
INPUT_DIM=200, D_K=128, D_V=64, N_EXPERTS=8, TOP_K=3, N_LAYERS=3, JOINT_DIM=64

@pytest.fixture
def router(): return RegimeRouter(...)

@pytest.fixture
def memory(): return KDAMarketMemory(...)

@pytest.fixture
def cdap(): return CrossDimensionAttention(...)

@pytest.fixture
def hardening(): return HardeningEngine(threshold=30)

@pytest.fixture
def ensemble(): return ExpertEnsemble(...)  # 完整集成模型

@pytest.fixture
def ensemble_low_threshold(): ...  # threshold=5 用于硬化快路径测试
```

### 11.3 关键测试模式

| 测试类别 | 测试内容 | 为什么重要 |
|---|---|---|
| 形状测试 | `assert output.shape == (B, expected)` | 确保张量维度正确 |
| 有限性测试 | `assert output.isfinite().all()` | 捕获 NaN/Inf 爆炸 |
| 概率测试 | `assert routing.sum(-1) ≈ 1` | 路由输出是合法概率 |
| 梯度测试 | `loss.backward(); assert p.grad is not None` | 确保梯度流不中断 |
| 确定性测试 | `torch.manual_seed(42); assert out1 == out2` | eval 模式应该确定 |
| 边界测试 | 零输入、全 mask、单资产 | 极端情况不崩溃 |
| 调制测试 | 手动设 scales=0.5，验证输出变化 | CDAP 确实在工作 |
| 稳定性测试 | 1000 步序列 boundedness | 防止数值发散 |

---

## 12. 已知问题

### 12.1 memory_gate_scale 死通路 🔴

**现象**：训练后 `memory_gate_scale = 0.0000`。

**根因假说**：
1. 梯度路径太长：MSE → signal → routing → joint → memory_gate
2. 3 epoch warmup 不够
3. memory.detach_state() 切断了跨 batch 的梯度

**建议**：per-pathway 独立 LR；直接监督 memory gate（正则化鼓励偏离 0.5）

### 12.2 路由最大熵 🔴

**现象**：`routing_entropy_ratio ≈ 0.998`（99.8% 最大熵）。

**可能原因**：
- 合成数据缺乏 regime 切换（都是几何布朗运动）
- Top-K=3 过于宽松
- Quantile Balancing 强制均衡

**建议**：真实数据上测试；降低 Top-K；curriculum learning 温度退火

### 12.3 合成数据的局限性 🟡

几何布朗运动生成的价格有恒定的波动率和零自相关，与真实市场完全不同。在合成数据上训练出的模型无法直接用于真实交易。

### 12.4 未实现的模块 🟡

- 3 个 Trainer 类（ExpertTrainer, RouterTrainer, JointTrainer）
- BacktestEngine.run()（完整回测）
- 真实数据源（Baostock, YFinance）
- 组合优化（Markowitz）
- 实验/Benchmark（README 中的表格全是占位符）

---

## A. 附录：文件快速索引

| 想了解... | 看这个文件 |
|---|---|
| 数据怎么加载 | `src/daft/data/loaders.py` |
| 特征怎么算 | `src/daft/features/tensor_factors.py` |
| s_t 怎么构建 | `src/daft/features/regime_features.py` |
| FFT 频域特征 | `src/daft/features/freq_features.py` |
| alpha 因子列表 | `src/daft/features/legacy_factors.py` |
| 路由机制 | `src/daft/models/router.py` |
| 记忆机制 | `src/daft/models/memory.py` |
| CDAP 调制 | `src/daft/models/cross_dim_attn.py` |
| 硬化机制 | `src/daft/models/hardening.py` |
| 专家实现 | `src/daft/models/experts/trend_expert.py` 等 |
| 整体集成 | `src/daft/models/ensemble.py` |
| 训练流程 | `scripts/training_loop.py` |
| 测试怎么写 | `tests/conftest.py` + `tests/test_*.py` |
| 项目配置 | `configs/small.yaml`, `pyproject.toml` |
| 架构文档 | `docs/architecture.md` |
| 项目 README | `README.md` |
