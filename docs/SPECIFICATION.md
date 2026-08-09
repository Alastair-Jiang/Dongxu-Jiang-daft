# DAFT 技术说明书

> **Dimension-Aware Financial Trading**  
> 面向中频量化交易的跨维度注意力架构  
> Version 1.0 · 2026-08-09

---

## 目录

1. [项目概览](#1-项目概览)
2. [灵感来源：Kimi K3 → 金融时序](#2-灵感来源kimi-k3--金融时序)
3. [系统架构](#3-系统架构)
4. [数据层](#4-数据层)
5. [特征工程](#5-特征工程)
6. [核心模型](#6-核心模型)
   - 6.1 [RegimeRouter](#61-regimerouter)
   - 6.2 [KDA Market Memory](#62-kda-market-memory)
   - 6.3 [Cross-Dimension Attention Protocol](#63-cross-dimension-attention-protocol)
   - 6.4 [Adaptive Hardening Mechanism](#64-adaptive-hardening-mechanism)
7. [专家池](#7-专家池)
8. [集成层](#8-集成层)
9. [训练管线](#9-训练管线)
10. [组合优化](#10-组合优化)
11. [回测引擎](#11-回测引擎)
12. [配置系统](#12-配置系统)
13. [扩展指南](#13-扩展指南)

**附录**
- [A. 完整参数列表](#附录-a-完整参数列表)
- [B. 文件结构索引](#附录-b-文件结构索引)
- [C. 已知问题和 TODO](#附录-c-已知问题和-todo)

---

## 1. 项目概览

### 1.1 一句话定义

DAFT 是一套受 **Kimi K3**（Moonshot AI, 2026年7月, 2.8T参数开源大模型）启发的 PyTorch 量化交易模型。核心创新是让 **路由（Router）、记忆（Memory）、特征深度（Depth）** 三个信息维度在共享的联合潜空间中**互相调制**，形成闭环决策引擎，而不是像传统系统那样各自独立运行。

### 1.2 设计目标

| 维度 | 指标 |
|------|------|
| 交易频率 | 中频（分钟级到日频） |
| 资产类别 | A股（可扩展到任意股票池） |
| 模型规模 | < 300K 参数（轻量，可研究迭代） |
| 推理效率 | 硬化后常见行情 O(1) 快路径 |
| 硬件需求 | Mac Mini M4 / ThinkBook 14+ 可训练 |
| 可扩展性 | 专家、数据源、路由策略均可插拔替换 |

### 1.3 与现有方案的区别

| 方案 | DAFT 超越了什么 |
|------|----------------|
| **Time-MoE** (ICLR 2025) | MoE 路由把记忆当被动状态；DAFT 让路由主动调制记忆的遗忘策略 |
| **Dynamic TMoE** (ICML 2026) | 专家池适应分布漂移，但专家选择从不反过来影响记忆保留 |
| **KDA** (Kimi, 2025) | Per-channel forget gate 只依赖当前输入；DAFT 加了路由信号和深度信号的调制 |
| **AttnRes** (K3, 2026) | 跨层注意力权重不考虑路由分布和记忆状态；DAFT 让三者互相耦合 |
| **PatchTST / TimesNet / iTransformer** | 单模型单行情，无专家专业化 |

**核心差距：没有现有系统允许路由决策改变记忆策略，或让记忆状态改变对特征层的信任权重。DAFT 填补了这个空白。**

### 1.4 核心指标

| 指标 | 数值 |
|------|------|
| 总参数 | 275,099 |
| 测试套件 | 364 passed, 1 skipped |
| 特征提取速度 | 2.0s（500天 × 20股, 日频） |
| 内存状态大小 | 128 × 64 = 8,192 float32 ≈ 32 KB |
| 每步推理复杂度 | O(d_k · d_v) = O(8192)，与序列长度无关 |

---

## 2. 灵感来源：Kimi K3 → 金融时序

DAFT 不是凭空设计的。三个核心组件各有一个 Kimi K3 的对应物，每次映射都做了领域适配。

### 2.1 K3 → DAFT 映射表

| K3 组件 | K3 实现 | DAFT 映射 | 关键适配 |
|---------|---------|-----------|----------|
| **Stable LatentMoE** | 896 专家, 16 激活, 潜空间路由 + Quantile Balancing | 8 策略专家, Top-3 激活, regime 潜空间 (R^16) | 给专家赋予金融语义：趋势/反转/波动率/事件 |
| **KDA** | Per-channel forget gate (α_t), delta-rule 状态更新 (S_t), 3:1 KDA-to-MLA 层比例 | Per-slot forget gate, 路由调制遗忘 (α'_t), 固定大小市场记忆 (M_t ∈ R^{128×64}) | 路由信号调制遗忘门；无位置编码（市场时间非均匀） |
| **AttnRes** | 跨层注意力 over [h_0, …, h_{l-1}] | 跨层检索 over 因子层次 (L0 raw → L1 base → L2 composite) | 深度权重对记忆状态敏感（CDAP 连接） |
| **SiTU 激活** | σ(x) ⊙ tanh(x), 自然输出界 [-1, 1] | 专家输出激活，确保多空信号量级可比 | 防止 MoE 门控融合前输出量级漂移 |
| **3:1 混合比例** | 静态：3 KDA 层配 1 MLA 全注意力层 | **动态：AHM 学习的快/慢路径比例** | **DAFT 原创扩展**：比例随行情自适应 |

### 2.2 DAFT 的原创扩展

DAFT 做了 K3 没有的一件事：**让三个组件互相通信**。

- **K3 的局限**：MoE routing、KDA forget gate、AttnRes retrieval 是三个孤岛——各自做决策，互不知晓
- **DAFT 的方案**：CDAP（Cross-Dimension Attention Protocol）建立一个 64 维联合潜空间，三个维度的信号投影进去，逐元素乘法融合，再反向投影回各自组件
- **AHM（Adaptive Hardening）** 进一步把 K3 的静态 3:1 比例替换成数据驱动的动态路由——常见行情走硬化快路径，罕见行情降级回完整 CDAP 计算

---

## 3. 系统架构

### 3.1 完整数据流

```
                              ┌──────────────────┐
                              │   市场数据 (OHLCV) │
                              │   分钟级/日级      │
                              └────────┬─────────┘
                                       │
                              ┌────────▼─────────┐
                              │   特征引擎        │
                              │  · 213 个经典因子  │
                              │  · FFT 频域特征   │
                              │  · Regime 特征    │
                              │  → s_t ∈ R^200    │
                              └────────┬─────────┘
                                       │
                ┌──────────────────────┼──────────────────────┐
                │                      │                      │
                ▼                      ▼                      ▼
      ┌─────────────────┐   ┌─────────────────┐   ┌─────────────────┐
      │  L0: 原始数据    │   │  L1: 基础因子    │   │  L2: 复合特征   │
      │  (价格/成交量)   │   │  (MA/Vol/RSI)   │   │  (Regime/Risk)  │
      └────────┬────────┘   └────────┬────────┘   └────────┬────────┘
               │                     │                     │
               └─────────────────────┼─────────────────────┘
                                     │
                      ┌──────────────┼──────────────┐
                      │              │              │
                      ▼              ▼              ▼
      ┌───────────────────────────────────────────────────────┐
      │           CROSS-DIMENSION ATTENTION (CDAP)             │
      │                                                       │
      │    ┌─────────┐       ┌──────────┐      ┌───────────┐ │
      │    │ Router  │◄─────►│  Memory  │◄────►│   Depth   │ │
      │    │(Regime) │       │  (KDA)   │      │ (AttnRes) │ │
      │    └────┬────┘       └────┬─────┘      └─────┬─────┘ │
      │         │                 │                   │       │
      │         └─────────────────┼───────────────────┘       │
      │                           │                           │
      │               Joint Latent Space (R^64)               │
      │               j = e ⊙ m ⊙ d                           │
      └───────────────────────────┬───────────────────────────┘
                                  │
                        ┌─────────▼─────────┐
                        │  Hardening Engine  │
                        │  快路径 / 慢路径    │
                        │  自适应路由        │
                        └─────────┬─────────┘
                                  │
                        ┌─────────▼─────────┐
                        │  Expert Ensemble   │
                        │  8 专家加权融合     │
                        │  → 交易信号        │
                        └─────────┬─────────┘
                                  │
                        ┌─────────▼─────────┐
                        │  Portfolio Optim   │
                        │  (Markowitz 均值-方差)│
                        └─────────┬─────────┘
                                  │
                        ┌─────────▼─────────┐
                        │  Backtest Engine   │
                        │  (矢量化回测)       │
                        └───────────────────┘
```

### 3.2 组件关系

系统由 **4 个核心模型组件** + **5 类策略专家** + **4 阶段训练管线** 组成：

```
DAFT/
├── 数据层        Panel(T×N×F) + DataLoader + 适配器(Baostock/YFinance)
├── 特征层        5 个特征提取器 → s_t ∈ R^200 + 3 层特征 [h0, h1, h2]
├── 模型层        [C1] RegimeRouter → [C2] KDA Market Memory → [C3] CDAP
│                 → [C4] AHM → ExpertEnsemble
├── 训练层        4 阶段：独立专家 → Router+Memory → 联合微调 → 硬化收集
├── 组合层        Markowitz 均值-方差 + Ledoit-Wolf 收缩
└── 回测层        矢量化回测 + 成本模型 + 绩效指标
```

---

## 4. 数据层

### 4.1 Panel 数据结构

DAFT 使用三维 Panel 表示市场数据，统一接口解耦数据源和模型。

```python
@dataclass
class Panel:
    values: torch.Tensor     # (T, N, F) — 时间 × 资产 × 特征
    mask: torch.Tensor       # (T, N), bool — 可交易性掩码
    dates: Optional[list]    # T 个日期标签
    asset_ids: Optional[list] # N 个资产标识
    feature_names: Optional[list] # F 个特征名称
    metadata: Optional[dict] # 数据源、频率、生成参数
```

**OHLCV 数据**：F=5，第 `[open, high, low, close, volume]`。

**Mask** 的作用：标记停牌、涨跌停板等不可交易状态，在训练和回测中被自动跳过。

### 4.2 DataLoader 接口

```python
# 合成数据（测试/快速实验用）
from daft.data.loaders import SyntheticDataLoader
loader = SyntheticDataLoader(n_stocks=20, n_days=500, freq="daily")
panel = loader.load()

# 真实 A 股数据
from daft.data.loaders import BaostockDataLoader
loader = BaostockDataLoader(symbols=["sh.600000", "sz.000001"], 
                             start="2022-01-01", end="2024-12-31")
panel = loader.load()
```

### 4.3 数据源适配器

| 适配器 | 文件 | 数据来源 | 状态 |
|--------|------|---------|------|
| Baostock | `data/adapters/baostock_adapter.py` | A 股免费数据 | 可用 |
| YFinance | `data/adapters/yfinance_adapter.py` | 美股/全球 | 可用 |
| Synthetic | `data/loaders.py` | 几何布朗运动生成 | 可用 |

### 4.4 合成数据生成

用于快速测试和实验。基于几何布朗运动（Geometric Brownian Motion）生成价格序列：

```
S_t = S_{t-1} · exp(μΔt + σ√Δt · ε_t),  ε_t ~ N(0,1)
```

支持配置漂移率 μ、波动率 σ、股票数量、时间步数。

---

## 5. 特征工程

### 5.1 特征管线总览

原始 OHLCV → 4 条特征管线并行提取 → 拼接为 s_t ∈ R^200：

```
Panel(T×N×F)
    │
    ├─→ TensorFactorEngine    (7 GPU 原语 → 因子矩阵)
    ├─→ LegacyFactorRegistry   (213 个经典 alpha 因子)
    ├─→ FreqFeatureExtractor   (FFT 频域 → 低/中/高频段)
    └─→ RegimeFeatureExtractor (6 组市场状态 × 压缩 → 200 维)
         │
         ▼
    s_t ∈ R^200  市场状态向量
```

### 5.2 TensorFactorEngine — GPU 加速原语

7 个 mask-aware 原语，全部在 GPU 上矢量化计算：

| 原语 | 功能 | 公式 |
|------|------|------|
| `rank` | 截面排名归一化 | rank(x_i) / N → [0, 1] |
| `corr` | 滚动 Pearson 相关 | cov(x, y) / (σ_x · σ_y) |
| `ewma` | 指数加权移动平均 | α·x_t + (1-α)·EMA_{t-1} |
| `ts_delta` | 时间序列差分 | x_t - x_{t-lag} |
| `ts_sum` | 滚动求和 | Σ_{i=t-lag}^{t} x_i |
| `ts_std` | 滚动标准差 | σ(x_{t-lag:t}) |
| `ts_mean` | 滚动均值 | mean(x_{t-lag:t}) |

### 5.3 FreqFeatureExtractor — 频域特征

FFT 变换后按频率分段聚合，捕捉周期性模式。

```
FFT(signal) → 功率谱
    ├── 低频段 (长期趋势)   → mean/power
    ├── 中频段 (中级周期)   → mean/power
    └── 高频段 (短期波动)   → mean/power + noise ratio
```

### 5.4 RegimeFeatureExtractor — 市场状态向量

从 6 组原始特征中提取，每组内部压缩后拼接：

| 组别 | 内容 | 维度 |
|------|------|------|
| 自适应窗口价格趋势 | 多时间尺度价格变化率 | ~33 |
| 波动率结构 | 历史波动率、GARCH 信号 | ~33 |
| 流动性/成交量 | 成交量变化、换手率代理 | ~33 |
| 技术指标 | MA 交叉、RSI、布林带 | ~33 |
| 截面统计 | 横截面均值、离散度、排名相关性 | ~33 |
| FFT 频域 | 频段功率比、主频检测 | ~35 |

**总计**：200 维。

### 5.5 经典因子注册表

213 个 hand-crafted alpha 因子，参考 WorldQuant 101 公式，分为：

- 价量因子（趋势、反转、突破）
- 波动率因子（低波/高波异常）
- 流动性因子（成交量变化模式）
- 时间序列算子组合（rank, ts_mean, ts_std, corr 等组合）

所有因子通过 `LegacyFactorRegistry` 统一管理和调用。

---

## 6. 核心模型

### 6.1 RegimeRouter

> **Inspiration**: Kimi K3 Stable LatentMoE — 896 专家, 16 激活, 潜空间路由

#### 6.1.1 功能

将 200 维市场状态向量 s_t 映射到 16 维 regime 潜空间 z_t，在潜空间中决定激活哪些专家。

#### 6.1.2 数学公式

**潜空间投影**（低秩瓶颈，过滤噪声）：

```
z_t = LayerNorm( W_down · SiLU(W_up · s_t) )  ∈  R^16
```

其中 W_up 将 200 维升到 64 维，W_down 将 64 维降到 16 维。

**温度缩放路由**：

```
                                exp((W_i^route · z_t + b_i) / τ)
p(expert_i | z_t) = ──────────────────────────────────────────────
                      Σ_{j=1}^{8} exp((W_j^route · z_t + b_j) / τ)

activated = TopK(p, k=3)
```

**Quantile Balancing**（零辅助损失负载均衡）：

```
b_i ← b_i + η · (1/8 - count_i / Σ_j count_j)
```

如果某个专家被选得太少，自动加 bias；被选得太多，自动减 bias。

**Noisy Gating**（仅训练模式）：

```
p_noisy = softmax( (W_route · z_t + b + ε · softplus(W_noise · z_t)) / τ )
```

训练时加可学习的噪声 ε，鼓励探索不同的专家组合。

#### 6.1.3 参数表

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `input_dim` | 200 | 市场状态向量维度 |
| `latent_dim` | 16 | Regime 潜空间维度 |
| `n_experts` | 8 | 策略专家总数 |
| `top_k` | 3 | 每次激活的专家数 |
| `τ_train` | 1.0 | 训练时温度（软路由） |
| `τ_val` | 1.0 | 验证时温度（确定性） |
| `τ_inference` | 0.1 | 推理时温度（近离散） |
| `noisy_gating` | True | 训练时是否使用噪声门控 |

#### 6.1.4 三种运行模式

```python
# 训练模式：temperature=1.0 + noisy gating
router(s_t, mode="train")  

# 验证模式：temperature=1.0, 无噪声, 确定性
router(s_t, mode="val")

# 推理模式：temperature=0.1, 近离散, 配合硬化快路径
router(s_t, mode="inference")
```

---

### 6.2 KDA Market Memory

> **Inspiration**: Kimi Delta Attention (KDA) — per-channel forget gate + delta-rule state update

#### 6.2.1 功能

维护一个固定大小的记忆矩阵 M_t ∈ R^{128×64}，每个时间步通过 delta-rule 在线学习机制更新，不需要存储整个历史序列。

#### 6.2.2 数学公式

**Per-Channel Forget Gate**（低秩瓶颈，KDA FineGrainedGating）：

```
α_t = σ(exp(A_log) · (SiTU(W_up · SiTU(W_down · s_t)) + dt_bias))  ∈  (0.001, 1)^{128}
```

128 个 memory slot 各有自己的遗忘率。A_log 和 dt_bias 是可学习参数。

**Route-Modulated Forgetting**（CDAP 连接：Router → Memory）：

```
α'_t = α_t ⊙ σ(W_route · z_t)   （路由信号调制遗忘）
α''_t = α'_t ⊙ cdap_gate        （CDAP 联合空间反馈）
```

当 Router 识别出趋势行情时，σ(W_route · z_t) 会压低反转相关 memory channel 的保留率。

**Delta-Rule 状态更新**（KDA 在线学习公式）：

```
k_t = L2Norm(W_k · s_t)         # L2 归一化关键向量（防数值爆炸）
v_t = W_v · s_t                  # 值向量
β_t = σ(W_β · s_t)               # 可学习更新步长 ∈ (0, 1)

M_t = M_{t-1} - β_t · k_t ⊗ (M_{t-1} · k_t) + β_t · k_t ⊗ v_t
      ─────────   ─────────────────────────────   ─────────────────
      遗忘旧记忆      Delta correction (先擦除)       KV write (再写入)
```

**记忆检索**：

```
o_t = M_t^T · q_t,   q_t = W_q · s_t
output = RMSNorm(σ(W_out_up · SiTU(W_out_down · s_t)) ⊙ o_t)   # 带输出门
```

#### 6.2.3 复杂度分析

```
每步计算：O(d_k · d_v) = O(128 × 64) = O(8192)
状态大小：128 × 64 × 4 bytes = 32 KB
序列长度：无关 — 无 KV-cache，无历史窗口限制
```

#### 6.2.4 参数表

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `d_k` | 128 | Key 维度 / memory slot 数量 |
| `d_v` | 64 | Value 维度 / 每 slot 存储信息 |
| `d_feature` | 200 | 输入 s_t 维度 |
| `bottleneck_ratio` | 4 | Forget gate 低秩压缩比 |
| `use_route_modulation` | True | 是否启用 Router → Memory CDAP |
| `safe_gate_lower_bound` | 0.001 | K3 安全门下限，防止完全遗忘 |

---

### 6.3 Cross-Dimension Attention Protocol

> ★ DAFT 的**核心方法论创新**

#### 6.3.1 设计动机

三个信息流——
- **Router 输出** p_t（专家选择分布）
- **Memory 状态** M_t（历史模式压缩）
- **Depth 特征** {h_0, h_1, h_2}（L0/L1/L2 层次特征）

——在传统架构中是**单向独立**的。CDAP 将它们投影到一个共享联合空间，实现**双向调制**。

#### 6.3.2 数学公式

**Step 1 — 三维投影到联合空间**：

```
e = f_{expert→joint}(p_t)         ∈ R^64   （路由→联合空间）
m = f_{memory→joint}(flatten(M_t)) ∈ R^64   （记忆→联合空间）
d = f_{depth→joint}(concat(h))     ∈ R^64   （深度→联合空间）
```

**Step 2 — 逐元素乘法融合** ★ 关键设计决策：

```
j = e ⊙ m ⊙ d   ∈ R^64
```

**为什么是乘法不是加法？**
- 加法假设三个维度独立贡献 → 如果 memory 不确定（接近零），它仍然可以加噪声到路由信号中
- 逐元素乘法 → 任何维度接近零激活，整条调制通路被静音
- 这是一个强归纳偏置，适合稀疏的、行情依赖的计算
- 在金融市场中，路由、记忆、深度天然耦合（趋势行情中三者应协同，震荡行情中 memory 应保持沉默）

**Step 3 — 反向投影（Joint → 各组件）**：

| 方向 | 公式 | 效果 |
|------|------|------|
| → Router | p'_t = softmax(log p_t + δ · W^router_out · j) | 记忆状态 + 深度信号修正路由偏好 |
| → Memory | g_t = σ(W^memory_out · j) ∈ (0,1)^{128} | 额外遗忘门调制 |
| → Depth | w_t = softmax(W^depth_out · j) ∈ Δ² | 跨层特征检索权重 |

**Step 4 — 融合输出**：

```
h_fused = Σ_{k=0}^{2} w_t^(k) · h_k
```

三个特征层按 CDAP 学到的权重加权求和。

**CDAP Scales**（可学习的三维调制强度）：

```
expert_bias_scale   ∈ R  — 记忆+深度对路由的调制强度
memory_gate_scale   ∈ R  — 路由+深度对记忆的调制强度
depth_weight_scale  ∈ R  — 路由+记忆对深度的调制强度
```

#### 6.3.3 代码接口

```python
# CDAP forward
outputs = cross_dim_attn(
    routing_probs=p_t,      # (B, 8) router 输出
    memory_state=M_t,       # (B, 128, 64) memory 矩阵
    layer_outputs=[h0, h1, h2],  # 3 层特征, each (B, 64)
    mode="train"
)
# returns: routing_mod, memory_gate, depth_weights, h_fused
```

---

### 6.4 Adaptive Hardening Mechanism

> ★ DAFT 的**第二大原创贡献**

#### 6.4.1 设计动机

Kimi K3 使用静态的 3:1 比例——每 3 个 KDA 层配 1 个 MLA 全注意力层。这对 NLP 有效，但金融市场的信息密度是**行情依赖的**：

```
低波动趋势市 → 大部分时间很常规 → 应该 ~90% 走快路径
高波动事件市 → 大部分时间很反常 → 应该 ~10% 走快路径
```

DAFT 的 AHM 从数据中学习这个比例，而不是硬编码。

#### 6.4.2 四大机制

**1. Pattern Counter（模式计数）**

```python
pattern = discretize_pattern(routing_probs)  # Top-3 专家索引排序
key = (regime_id, pattern)
pattern_counter[key] += 1
total_decisions += 1
```

**2. Cache Builder（缓存构建）**

```
硬化条件：count(key) ≥ θ_harden(=100)  AND  confidence > ρ_min(=0.95)
```

满足条件 → 将该 (regime, pattern) 的三维调制向量缓存为快路径：

```
C_hardened(regime, pattern) = {
    expert_weights*,
    memory_gate*,
    depth_weights*
}
```

**3. Entropy Guard（熵守卫）**

连续监控路由熵。如果最近 20 步的平均熵 > λ (=2.0) × 基线熵 → 认为行情切换，自动降级到完整的 CDAP 计算：

```python
if H(p_recent) > λ_entropy × H_baseline:
    degrade_to_full_exploration()
```

**4. Staleness Eviction（过期驱逐）**

缓存项如果长时间未命中 + 命中次数 < 10 → 清除，防止过时的缓存污染推理。

#### 6.4.3 统计指标

```python
stats = hardening.get_stats()
# → {
#     'total_decisions': int,      # 总决策次数
#     'n_cached_patterns': int,    # 已缓存模式数
#     'n_fast_path': int,          # 快路径命中次数
#     'n_slow_path': int,          # 慢路径次数
#     'n_degradations': int,       # 熵守卫触发次数
#     'fast_path_ratio': float,    # 快路径占比
#     'baseline_entropy': float,   # 基线路由熵
#     'cache_hit_rate': float,     # 缓存命中率
# }
```

---

## 7. 专家池

### 7.1 BaseExpert 接口

所有专家继承 `BaseExpert`，共享统一的输入/输出接口：

```python
class BaseExpert(nn.Module):
    def __init__(self, input_dim=200, hidden_dim=64, n_layers=2, name="base"):
        # MLP backbone: Linear → LayerNorm → SiLU → Dropout (× n_layers)
        # Prediction head: hidden_dim → 1
        # Output activation: SiTU (σ(x) ⊙ tanh(x))
    
    def forward(s_t, return_hidden=False) → signal ∈ [-1, 1]
    
    @abstractmethod
    def _regime_filter(panel) → mask: Tensor[bool]  # (T,) 哪些时刻这个专家应该激活
    
    @abstractmethod
    def compute_loss(pred, target, mask) → loss: Tensor  # 专家特定的损失函数
```

### 7.2 四类专家

| 类型 | 数量 | Regime Filter | 损失函数 | 训练逻辑 |
|------|------|--------------|----------|----------|
| **TrendExpert** | 2 | ADX > 25（趋势市） | Direction-weighted MSE（方向错误 ×11 惩罚） | 趋势市样本上回归 |
| **ReversalExpert** | 2 | ADX < 20（震荡市） | Negative Rank IC（最大化 Spearman 相关性） | 震荡市样本上排序学习 |
| **VolatilityExpert** | 2 | Vol > P80（高波动） | MSE + 0.01·Var(pred)（抑制过度波动） | 高波动市样本 + 方差正则 |
| **EventExpert** | 2 | 全部数据（兜底） | Binary Cross-Entropy on return direction | 全样本方向分类 |

### 7.3 SiTU 激活函数

```
SiTU(x) = σ(x) ⊙ tanh(x),  输出 ∈ [-1, 1]
```

来自 Kimi K3。相比于 SiLU/ReLU，SiTU 输出自然有界，防止专家间输出量级漂移导致 MoE 门控梯度失真。

### 7.4 专家内部结构

```
s_t (200) → Linear(200→64) → LayerNorm → SiLU → Dropout(0.1)
          → Linear(64→64) → LayerNorm → SiLU → Dropout(0.1)
          → Linear(64→1) → SiTU → signal ∈ [-1, 1]
```

每个专家约 17K 参数，8 个专家共约 136K 参数。

---

## 8. 集成层

### 8.1 ExpertEnsemble

`ExpertEnsemble` 是顶层模型，串联所有组件：

```python
class ExpertEnsemble(nn.Module):
    def __init__(self, experts, router, memory, cross_dim_attn, hardening)
    
    def forward(s_t, layer_outputs, mode="train", use_hardening=False) → dict:
        # mode: "train" | "val" | "inference"
        # 返回: signal, routing_probs, regime_id, depth_weights, metadata
```

### 8.2 Forward 的三种模式

| 模式 | Router 行为 | CDAP | 硬化 | 梯度 |
|------|------------|------|------|------|
| `train` | τ=1.0, noisy gating ON | 完整计算 | 不使用 | 跟踪 |
| `val` | τ=1.0, noisy gating OFF | 完整计算 | 不使用 | 不跟踪 |
| `inference` | τ=0.1, noisy gating OFF | 先查缓存 | 按需使用 | 不跟踪 |

### 8.3 推理时的快/慢路径

```python
# 推理时自动路径选择
if use_hardening:
    if hardening.should_use_fast_path(regime_id, routing_probs):
        # 快路径: O(1) 查表
        cached = hardening.get_cached_weights(regime_id, routing_probs)
        signal = expert_fusion(cached, h_fused)
    else:
        # 慢路径: 完整 CDAP 计算
        signal = full_cdap_forward(s_t, layer_outputs, mode="inference")
    
    if hardening.detect_regime_shift():
        # 熵飙升 → 临时降级，不信任缓存
        signal = full_cdap_forward(s_t, layer_outputs, mode="inference")
```

---

## 9. 训练管线

### 9.1 四阶段总览

```
Stage 1: 独立专家训练 ──→ Stage 2: Router + Memory ──→ Stage 3: CDAP 联合微调 ──→ Stage 4: 硬化收集
(experts only)             (experts frozen)              (all unfrozen, low LR)        (no gradient)
```

### 9.2 Stage 1 — 独立专家训练

**目标**：让每个专家在自己的行情子集上学到有效的预测策略。

```
训练参数: 专家 MLP 权重 + head
冻结参数: Router, Memory, CDAP, AHM

每个专家:
  ├─ regime_filter(panel) → 选择适配的样本子集
  ├─ 专家特定的 loss function
  ├─ 15 epochs, CosineAnnealingLR (1e-3 → 0)
  ├─ Early stop: patience=5, 按验证集 loss
  └─ 保存 best checkpoint → checkpoints/stage1/expert_{i}_{name}.pt
```

**数据划分**（以 Baostock A 股为例）：

| 专家类型 | 样本数（~1212 天 × 50 股） | ADX 阈值 | 额外过滤 |
|---------|---------------------------|----------|----------|
| Trend [0,1] | ~42,466 | ADX > 25 | — |
| Reversal [2,3] | ~8,082 | ADX < 20 | — |
| Volatility [4,5] | ~12,110 | Vol > P80 | — |
| Event [6,7] | ~60,550 | 全部 | — |

### 9.3 Stage 2 — Router + Memory 训练

**目标**：在专家冻结的前提下，让 Router 学会为给定行情选择合适的专家，让 Memory 学会保留有用的历史模式。

```
训练参数: Router 权重 + Memory 权重 + CDAP scale 参数（温和）
冻结参数: 所有专家

损失: L = L_MSE(signal, target) + λ_entropy · H(routing)     (λ=0.01 轻量熵正则)
LR: 1e-4, CosineAnnealing
Epochs: 20
```

### 9.4 Stage 3 — CDAP 联合微调

**目标**：解冻全参数，以极低学习率让三维调制 scale 参数和所有组件协调优化。

```
训练参数: 全部
LR: 1e-5 (极低, 避免破坏学到的专家专长)
Epochs: 10

关键操作:
  ├─ CDAP scale warmup: 前 3 epochs 只训练 expert_bias_scale,
  │   memory_gate_scale, depth_weight_scale
  └─ 后 7 epochs: 全参数微调
```

### 9.5 Stage 4 — 硬化收集

**目标**：无梯度前向传播，收集和缓存常见 (regime, pattern) 的三维调制向量。

```
梯度: 关闭
步数: 200+ 步前向传播
温度: τ=0.1（推理模式）

输出:
  ├─ 缓存表 (regime_id, pattern) → {expert_weights, memory_gate, depth_weights}
  ├─ 基线路由熵
  └─ 硬化统计（快/慢路径比例）
```

---

## 10. 组合优化

### 10.1 Markowitz 均值-方差优化

使用 Ledoit-Wolf 收缩估计协方差矩阵，然后用封闭式二次规划求解：

```
max_w    w^T · μ - γ · w^T · Σ_shrunk · w

s.t.     w_i ≥ 0        (禁止做空, long-only)
         Σ w_i = 1       (全额投资)
         w_i ≤ w_max     (单仓位上限, 默认 5%)
```

其中：
- **μ** = DAFT 模型输出的预期收益信号（截面方向，非量级）
- **Σ_shrunk** = Ledoit-Wolf 收缩协方差矩阵（比样本协方差更稳健）
- **γ** = 风险厌恶系数（越大越保守）
- **w_max** = 单票最大权重（默认 5%，即 ≥20 只持仓）

### 10.2 参数表

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `risk_aversion` | 1.0 | γ：风险厌恶系数 |
| `max_weight` | 0.05 | 单票最大权重 |
| `use_mosek` | False | 是否使用 MOSEK QP 求解器 |

---

## 11. 回测引擎

### 11.1 矢量化回测

```python
from daft.backtest import BacktestEngine

engine = BacktestEngine(config={
    "transaction_cost_bps": 2.0,   # 手续费：万分之二
    "slippage_bps": 1.0,           # 滑点：万分之一
    "top_quantile": 0.2,           # 信号 → 持仓：top 20% 做多
    "long_only": True,             # 仅做多
    "annualization": 252,          # 年化天数
    "rebalance_freq": 1,           # 每 1 个 bar 调仓
})

results = engine.run(signals, returns, panel)
```

### 11.2 信号 → 持仓转换

```
信号 s ∈ R^N（N 只股票）
  → 截面排名 → top_quantile(=20%) 的股票等权做多
  → 如果 long_only=False: bottom_quantile 等权做空
```

### 11.3 交易成本模型

```
总成本 = 手续费 + 滑点
       = tc_bps × |Δw| + slippage_bps × |Δw|
       = (tc_bps + slippage_bps) × turnover

其中 Δw 是相邻两期持仓的权重变化向量
```

### 11.4 绩效指标

| 指标 | 公式 | 说明 |
|------|------|------|
| **Sharpe Ratio** | (μ - r_f) / σ × √252 | 年化风险调整收益 |
| **Max Drawdown** | max(peak - trough) / peak | 最大回撤 |
| **Calmar Ratio** | 年化收益 / MaxDD | 收益/回撤比 |
| **Rank IC** | Spearman(signal, forward_return) | 信号截面预测能力 |
| **ICIR** | mean(IC) / std(IC) | IC 的信息比率 |
| **Hit Rate** | P(sign(signal) = sign(return)) | 方向准确率 |
| **Turnover** | mean(\|Δw\|) | 平均换手率 |

---

## 12. 配置系统

### 12.1 YAML 配置结构

```yaml
# configs/paper.yaml
data:
  source: baostock           # baostock | yfinance | synthetic
  start_date: "2022-01-01"
  end_date: "2024-12-31"
  symbols: [...]

model:
  n_experts: 8
  top_k: 3
  latent_dim: 16
  d_k: 128
  d_v: 64
  
training:
  stage1:
    epochs: 15
    lr: 0.001
    patience: 5
  stage2:
    epochs: 20
    lr: 0.0001
  stage3:
    epochs: 10
    lr: 0.00001
  stage4:
    n_steps: 200

evaluation:
  transaction_cost_bps: 2.0
  slippage_bps: 1.0
  top_quantile: 0.2
  long_only: true
```

### 12.2 三个预设配置

| 文件 | 用途 | 运行时间 |
|------|------|----------|
| `small.yaml` | 烟雾测试（合成数据, 200 股 × 500 天） | ~30 秒 |
| `paper.yaml` | 完整实验（真实数据） | ~5-10 分钟 |
| `hardening.yaml` | 硬化分析专用 | ~2 分钟 |

---

## 13. 扩展指南

### 13.1 如何添加新专家

```python
# 1. 创建新文件 src/daft/models/experts/my_expert.py
from daft.models.experts.base_expert import BaseExpert

class MyExpert(BaseExpert):
    def __init__(self, **kwargs):
        super().__init__(name="my_expert", **kwargs)
    
    def _regime_filter(self, panel):
        # 定义你的行情筛选逻辑
        # 返回 (T,) bool tensor
        return my_regime_mask
    
    def compute_loss(self, pred, target, mask):
        # 定义你的专家损失函数
        return my_loss(pred, target, mask)

# 2. 在 __init__.py 中注册
# 3. 训练时加入 experts 列表
# 4. 更新 config 中的 n_experts
```

### 13.2 如何接入新数据源

```python
# 1. 参考 src/daft/data/adapters/baostock_adapter.py
class MyDataAdapter:
    def fetch(self, symbols, start, end) -> Panel:
        # 从你的数据源拉取 → 构造 Panel
        return Panel(values=..., mask=..., dates=..., ...)

# 2. 在 src/daft/data/loaders.py 中注册
# 3. 在 config 中设置 data.source = "my_source"
```

### 13.3 如何自定义路由策略

```python
# 继承 RegimeRouter, 覆盖 forward() 中的路由逻辑
class MyRouter(RegimeRouter):
    def forward(self, s_t, mode="train"):
        # 自定义 latent projection
        # 自定义 Top-K 策略
        # 自定义 temperature schedule
        return routing_probs, z_t

# 在 ExpertEnsemble 中用 MyRouter 替换 RegimeRouter
```

---

## 附录 A. 完整参数列表

### A.1 模型参数

| 模块 | 参数 | 默认值 | 可训练参数数 |
|------|------|--------|-------------|
| RegimeRouter | input_dim=200, latent_dim=16, n_experts=8, top_k=3 | ~18K |
| KDAMarketMemory | d_k=128, d_v=64, d_feature=200, bottleneck_ratio=4 | ~85K |
| CDAP | n_experts=8, d_k=128, d_v=64, joint_dim=64 | ~28K |
| HardeningEngine | n_regimes=8, n_experts=8, threshold=100, min_confidence=0.95 | 0 (非参数化) |
| Experts (×8) | input_dim=200, hidden_dim=64, n_layers=2 | ~136K (8×17K) |
| **总计** | | | **~275K** |

### A.2 训练参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| Stage1 epochs | 15 | 独立专家训练轮数 |
| Stage1 LR | 1e-3 | 余弦退火到 0 |
| Stage1 patience | 5 | Early stop 耐心 |
| Stage2 epochs | 20 | Router+Memory 训练轮数 |
| Stage2 LR | 1e-4 | |
| Stage3 epochs | 10 | 联合微调轮数 |
| Stage3 LR | 1e-5 | 极低学习率 |
| Stage4 n_steps | 200 | 硬化收集步数 |
| CDAP warmup epochs | 3 | CDAP scale 预热轮数 |

### A.3 硬化参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `threshold` (θ) | 100 | 模式最小出现次数才能硬化 |
| `min_confidence` (ρ) | 0.95 | 模式最小置信度 |
| `entropy_multiplier` (λ) | 2.0 | 熵阈值倍数（触发降级） |
| `n_regimes_tracked` | 8 | 离散 regime 聚类数 |

### A.4 回测参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `transaction_cost_bps` | 2.0 | 手续费（万分之二） |
| `slippage_bps` | 1.0 | 滑点（万分之一） |
| `top_quantile` | 0.2 | 做多仓位选择的信号分位数 |
| `long_only` | True | 仅做多 |
| `annualization` | 252 | 年化天数 |
| `rebalance_freq` | 1 | 调仓频率（bar 数） |

---

## 附录 B. 文件结构索引

```
daft/
├── README.md                         # 高层概述（论文风格）
├── LICENSE                           # MIT
├── pyproject.toml                    # 项目配置 + 依赖
│
├── docs/
│   ├── SPECIFICATION.md              # ★ 本文档 — 技术说明书
│   ├── guided-tour.md                # 代码走读导览
│   ├── PROJECT_REPORT.md             # 项目进度报告
│   ├── architecture.md               # 架构规范（骨架）
│   └── experiments.md                # 实验记录
│
├── Learn-new/                         # 工作日志
│   ├── DAFT-工作日志-2026-08-07-全流程.md
│   ├── DAFT-项目日志-v0.2.0.md
│   └── DAFT-项目日志-v0.3.0-样本外公平对决.md
│
├── configs/                           # YAML 实验配置
│   ├── small.yaml                     #   烟雾测试
│   ├── paper.yaml                     #   完整实验
│   └── hardening.yaml                 #   硬化分析
│
├── src/daft/                          # 主包
│   ├── __init__.py                    #   v0.1.0, 公开 API
│   │
│   ├── data/                          # 数据管道
│   │   ├── panel.py                   #   Panel 数据结构 (T×N×F)
│   │   ├── loaders.py                 #   DataLoader 接口 + 合成数据
│   │   └── adapters/                  #   真实数据源适配器
│   │       ├── baostock_adapter.py    #     A 股
│   │       └── yfinance_adapter.py    #     美股/全球
│   │
│   ├── features/                      # 特征工程
│   │   ├── tensor_factors.py          #   7 GPU 原语 (rank, corr, ewma, ts_*)
│   │   ├── legacy_factors.py          #   213 经典 alpha 因子 + 注册表
│   │   ├── regime_features.py         #   s_t 市场状态向量构造
│   │   └── freq_features.py           #   FFT 频域特征
│   │
│   ├── models/                        # ★ 核心架构
│   │   ├── router.py                  #   [C1] RegimeRouter (Stable LatentMoE)
│   │   ├── memory.py                  #   [C2] KDA Market Memory
│   │   ├── cross_dim_attn.py          #   [C3] CDAP ★ 核心创新
│   │   ├── hardening.py               #   [C4] AHM ★ 原创贡献
│   │   ├── ensemble.py                #   ExpertEnsemble 集成层
│   │   └── experts/                   #   策略专家池
│   │       ├── base_expert.py         #     BaseExpert 接口 + SiTU 激活
│   │       ├── trend_expert.py        #     趋势跟踪
│   │       ├── reversal_expert.py     #     均值回归
│   │       ├── volatility_expert.py   #     波动率 regime
│   │       └── event_expert.py        #     事件驱动
│   │
│   ├── training/                      # 训练管线
│   │   ├── expert_trainer.py          #   Stage 1: 独立专家训练
│   │   ├── router_trainer.py          #   Stage 2: Router + Memory
│   │   └── joint_trainer.py           #   Stage 3: 联合微调
│   │
│   ├── portfolio/                     # 组合优化
│   │   └── markowitz.py               #   Markowitz Ledoit-Wolf 收缩
│   │
│   ├── backtest/                      # 回测
│   │   └── engine.py                  #   矢量化回测引擎
│   │
│   └── utils/                         # 工具
│       ├── metrics.py                 #   IC, ICIR, Hit Rate
│       └── device.py                  #   设备检测 (CPU/CUDA/MPS)
│
├── scripts/                           # 运行脚本
│   ├── training_loop.py               #   完整 4 阶段训练
│   ├── run_stage1.py                  #   仅 Stage 1
│   ├── run_stage2.py                  #   仅 Stage 2
│   ├── run_stage3.py                  #   仅 Stage 3
│   ├── run_full_pipeline.py           #   全管道
│   ├── run_backtest_only.py           #   仅回测
│   ├── run_baseline_ridge.py          #   Ridge 基线对比
│   ├── smoke_test.py                  #   烟雾测试
│   └── generate_flowcharts.py         #   流程图生成
│
├── tests/                             # 测试套件 (364 passed)
│   ├── test_router.py
│   ├── test_memory.py
│   ├── test_cdap.py
│   ├── test_experts.py
│   ├── test_hardening.py
│   ├── test_ensemble.py
│   ├── test_features.py
│   └── test_training.py
│
└── checkpoints/                       # 训练好的模型权重
    └── stage1/                        #   Stage 1 专家 checkpoint
```

---

## 附录 C. 已知问题和 TODO

### C.1 已知结构性问题

1. **路由最大熵** — routing entropy ratio 始终在 0.997–0.999，接近理论最大值 ln(8) ≈ 2.079。这意味着 8 个专家几乎均匀激活，MoE 退化为普通 ensemble。可能的解决方向：降低 Top-K 到 2、添加熵正则、在真实行情数据上测试、curriculum learning 降低温度。

2. **CDAP memory_gate 死通路** — memory_gate_scale 在早期训练中始终为 0。已在 2026-08-07 通过修复两处串联断路（safe_gate and route_modulate 之间的 gradient path 阻断 + cdap_gate 未被正确传入 memory.forward）解决。

### C.2 待实现功能

| 功能 | 说明 | 优先级 |
|------|------|--------|
| BacktestEngine.run() | 完整回测（含成本/滑点/持仓管理） | 高 |
| 真实事件过滤器 | EventExpert 目前训练全数据，需要财报/宏观日历 | 中 |
| 组合优化回退 | 当 QP 不可行时的回退策略 | 低 |
| 实验/Benchmark 表格 | README 中大部分是占位符 | 高 |
| Stage 2 和 Stage 3 的训练验证 | 目前只有 Stage 1 有完整的训练日志 | 高 |

### C.3 未来方向

- 在 ≥500 只 A 股上做完整回测
- 添加 turnover constraint 到组合优化
- 实现 multi-frequency 特征（15min / 30min / 60min）
- 探索 Top-K=2 以减少路由熵
- 与 Time-MoE (ICLR 2025) 做系统性的样本外对比

---

> **DAFT: Dimension-Aware Financial Trading**  
> Author: Alastair (Dongxu Jiang)  
> GitHub: [github.com/Alastair-Jiang/Dongxu-Jiang-daft](https://github.com/Alastair-Jiang/Dongxu-Jiang-daft)  
> Version: 0.1.0 · License: MIT  
> Inspired by Kimi K3 (Moonshot AI, July 2026)
