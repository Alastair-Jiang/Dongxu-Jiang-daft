# DAFT 技术说明书

> **Dimension-Aware Financial Trading**
> 面向中频量化交易的跨维度注意力架构
> Version **v0.1.0（旧规则:v0.3.0 代码义）** · 2026-08-16

---

## 版本演进

| 版本 | 日期 | 里程碑 |
|------|------|--------|
| v0.1.0（旧规则最早版本） | 2026-07 | 核心架构（MoE 专家池 + Regime Router + KDA Memory + CDAP + AHM） |
| v0.0.1（旧规则:v0.2.0） | 2026-08-06/07 | 全管道打通，消除所有 `NotImplementedError` |
| **v0.1.0（旧规则:v0.3.0 代码义）** | **2026-08-16** | **工程修复(PR #9) + 全面升级**：通道契约/口径统一/10 专家、工厂去重、涨跌停 mask、hs300 成分股池、交易制度约束 |

> v0.1.0（旧规则:v0.3.0 代码义） 是一次**工程修复 + A 股口径升级**，不含架构变更。修复批次（PR #9）
> 修正了早期实现与文档的多处出入——最严重的是**数据通道错列**（数据源的
> OHLCV 被特征引擎当作基础特征读取），此前所有实验的 s_t 都建立在错列之上。
> 全面升级进一步补齐 A 股交易制度（涨跌停 mask、T+1 成交约束、调仓频率与
> 分数仓位）、把股票池从 50 只内置样本升级到 hs300 真实成分（`--universe hs300`，
> 按 start 日拉取、解除 50 只上限并缓解幸存者偏差），并完成扩池重测与稳健性
> 验证。本说明书以 v0.1.0（旧规则:v0.3.0 代码义） 的真实实现为准。

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
8. [集成层与模型工厂](#8-集成层与模型工厂)
9. [训练管线](#9-训练管线)
10. [组合优化](#10-组合优化)
11. [回测引擎](#11-回测引擎)
12. [配置系统](#12-配置系统)
13. [实验结果](#13-实验结果)
14. [扩展指南](#14-扩展指南)

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
| 模型规模 | 核心 ≈31.5 万参数（轻量，可研究迭代） |
| 推理效率 | 硬化后常见行情 O(1) 快路径（研究性） |
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
| 专家数 | 10（5 类 × 2 实例） |
| 总参数 | 核心 ≈31.5 万；含 layer_proj ≈41.7 万 |
| 测试套件 | 396 项（pytest collected） |
| 特征提取速度 | 2.0s（500天 × 20股, 日频） |
| 内存状态大小 | 128 × 64 = 8,192 float32 ≈ 32 KB |
| 每步推理复杂度 | O(d_k · d_v) = O(8192)，与序列长度无关 |

---

## 2. 灵感来源：Kimi K3 → 金融时序

DAFT 不是凭空设计的。三个核心组件各有一个 Kimi K3 的对应物，每次映射都做了领域适配。

### 2.1 K3 → DAFT 映射表

| K3 组件 | K3 实现 | DAFT 映射 | 关键适配 |
|---------|---------|-----------|----------|
| **Stable LatentMoE** | 896 专家, 16 激活, 潜空间路由 + Quantile Balancing | 10 策略专家, Top-3 激活, regime 潜空间 (R^16) | 给专家赋予金融语义：趋势/反转/波动率/事件/动量 |
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
                              │  通道契约          │
                              │  ensure_base_panel │  ← 唯一转换点
                              │  OHLCV → 基础特征   │
                              └────────┬─────────┘
                                       │
                              ┌────────▼─────────┐
                              │   特征引擎        │
                              │  · 35 个经典因子  │
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
                        │  (研究性, 默认禁用) │
                        └─────────┬─────────┘
                                  │
                        ┌─────────▼─────────┐
                        │  Expert Ensemble   │
                        │  10 专家加权融合    │
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

系统由 **4 个核心模型组件** + **5 类策略专家（10 实例）** + **4 阶段训练管线** 组成：

```
DAFT/
├── 数据层        Panel(T×N×F) + 通道契约(ensure_base_panel) + 适配器(Baostock/YFinance)
├── 特征层        5 个特征提取器 → s_t ∈ R^200 + 3 层特征 [h0, h1, h2]
├── 模型层        [C1] RegimeRouter → [C2] KDA Market Memory → [C3] CDAP
│                 → [C4] AHM → ExpertEnsemble
│                 └── factory.py 单一权威构建入口
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

**Mask** 的作用：标记停牌、涨跌停板等不可交易状态，在训练和回测中被自动跳过。
A 股口径下，mask 由两部分合成：`close` 有效性（NaN → 停牌/缺失，置 False）与
涨跌停检测（`_limit_move_mask`，见 4.6）。

### 4.2 通道契约（v0.1.0（旧规则:v0.3.0 代码义） 修复 ★ 关键）

数据源与特征引擎之间存在**两种不同的通道布局**，历史上被直接混用，导致特征错列：

```
数据源产出 (baostock / yfinance / synthetic):
    ["open", "high", "low", "close", "volume"]          ← OHLCV 布局

特征引擎需要 (RegimeFeatureExtractor / legacy / Freq):
    ["close", "log_return", "volume", "volume_ratio", "volatility_20"]  ← 基础布局
```

**v0.1.0（旧规则:v0.3.0 代码义） 修复**：新增 `src/daft/features/base_features.py` 作为**唯一转换点**，
所有特征引擎入口必须首先调用 `ensure_base_panel`：

```python
from daft.features.base_features import ensure_base_panel
panel = ensure_base_panel(panel)   # OHLCV → 基础特征布局
```

转换规则（严格因果 + mask 感知）：
- `log_return[t] = log(close[t]) - log(close[t-1])`（需 t 与 t-1 均可交易）
- `volume_ratio[t] = volume[t] / 滚动20日均量`
- `volatility_20[t] = log_return 的滚动20日标准差`

若布局既不是 OHLCV 也不是基础布局，则**抛出明确错误**（失败要响亮，不再静默错列）。

### 4.3 DataLoader 接口

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

### 4.4 数据源适配器

| 适配器 | 文件 | 数据来源 | 状态 |
|--------|------|---------|------|
| Baostock | `data/adapters/baostock_adapter.py` | A 股免费数据 | 可用（批量重试 + 涨跌停 mask + hs300 成分） |
| YFinance | `data/adapters/yfinance_adapter.py` | 美股/全球 | 可用 |
| Synthetic | `data/loaders.py` | 几何布朗运动生成 | 可用 |

### 4.5 合成数据生成

基于几何布朗运动（Geometric Brownian Motion）生成价格序列：

```
S_t = S_{t-1} · exp(μΔt + σ√Δt · ε_t),  ε_t ~ N(0,1)
```

### 4.6 A 股口径升级（v0.1.0（旧规则:v0.3.0 代码义） 新增）

#### 4.6.1 涨跌停 mask

A 股有涨跌停制度（主板 ±10%，创业板 300/301 与科创板 688 ±20%）。涨停买不进、
跌停卖不出，若不处理会把无法成交的"假信号"计入回测收益。`baostock_adapter.py`
的 `_limit_move_mask` 在加载数据时把触及涨跌停的交易日 mask 置 `False`：

- 阈值：`_is_gem_or_star(ticker)` 判断板块，主板 0.095、创业板/科创板 0.195
  （留 0.5% 余量防前复权误差）。
- 判定：当日 close 相对前一交易日 close 的 |涨跌幅| ≥ 阈值 → 该日不可成交。
- 边界：首个交易日无前收盘不判（True）；缺失收盘（NaN）的日子保持 True，
  由 NaN mask 另行置 False，不重复处理。

开关由 `config["handle_limit_up_down"]`（默认 `True`）控制。

#### 4.6.2 hs300 真实成分股池

早期实验用内置 50 只静态样本，既有人工上限（截面太小、权重股同质），也有
幸存者偏差。v0.1.0（旧规则:v0.3.0 代码义） 各运行脚本新增 `--universe hs300`（默认），按 `start` 日
拉取真实沪深 300 成分股，解除 50 只上限并缓解幸存者偏差：

```bash
python scripts/run_full_pipeline_oos.py --stocks 100 --universe hs300
```

`--universe sample` 退回内置静态清单。100 股 hs300 口径下的重测结果见第 13 章
（Ridge IC 0.048/Sharpe +0.53 达 GO 线，30 股无信号被证伪为股票池效应）。

---

## 5. 特征工程

### 5.1 特征管线总览

原始 OHLCV →（通道契约）→ 基础特征 → 4 条特征管线：

```
Panel(T×N×F)
    │
    │  ensure_base_panel (通道契约)
    ▼
基础特征 [close, log_return, volume, volume_ratio, volatility_20]
    │
    ├─→ TensorFactorEngine    (7 GPU 原语 → 因子矩阵)
    ├─→ LegacyFactorRegistry   (35 个经典 alpha 因子)
    ├─→ FreqFeatureExtractor   (FFT 频域 → 低/中/高频段)
    └─→ RegimeFeatureExtractor (6 组市场状态 → 200 维)
         │
         ▼
    s_t ∈ R^200  市场状态向量
```

> **重要澄清（v0.1.0（旧规则:v0.3.0 代码义））**：手工因子注册表为 **35 个**（非早期文档的 213）；
> 且 legacy 因子与 FFT 特征**当前未接入**任何训练/评估管线。200 维 s_t
> 全部由 `RegimeFeatureExtractor` 从基础特征派生。

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

### 5.3 FreqFeatureExtractor — 频域特征（未接入管线）

FFT 变换后按频率分段聚合，捕捉周期性模式：

```
FFT(signal) → 功率谱
    ├── 低频段 (长期趋势)   → mean/power
    ├── 中频段 (中级周期)   → mean/power
    └── 高频段 (短期波动)   → mean/power + noise ratio
```

> 当前为独立模块，**未接入**训练/评估管线（v0.1.0（旧规则:v0.3.0 代码义） 澄清）。

### 5.4 RegimeFeatureExtractor — 市场状态向量

从基础特征提取 6 组市场状态，每组内部压缩后拼接为 200 维 s_t：

| 组别 | 内容 | 维度 |
|------|------|------|
| 自适应窗口价格趋势 | 多时间尺度价格变化率 | ~33 |
| 波动率结构 | 历史波动率、GARCH 信号 | ~33 |
| 流动性/成交量 | 成交量变化、换手率代理 | ~33 |
| 技术指标 | MA 交叉、RSI、布林带 | ~33 |
| 截面统计 | 横截面均值、离散度、排名相关性 | ~33 |
| FFT 频域 | 频段功率比、主频检测 | ~35 |

**总计**：200 维。**这是 s_t 的唯一来源**（legacy 与 FFT 特征不参与）。

### 5.5 经典因子注册表（未接入管线）

35 个 hand-crafted alpha 因子，参考 ml-quant-trading，分为：

- 价量因子（趋势、反转、突破）
- 波动率因子（低波/高波异常）
- 流动性因子（成交量变化模式）
- 时间序列算子组合（rank, ts_mean, ts_std, corr 等组合）

所有因子通过 `LegacyFactorRegistry` 统一管理。> 当前为独立模块，**未接入**训练/评估管线（v0.1.0（旧规则:v0.3.0 代码义） 澄清）。

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
                      Σ_{j=1}^{10} exp((W_j^route · z_t + b_j) / τ)

activated = TopK(p, k=3)
```

**Quantile Balancing**（零辅助损失负载均衡）：

```
b_i ← b_i + η · (1/10 - count_i / Σ_j count_j)
```

如果某个专家被选得太少，自动加 bias；被选得太多，自动减 bias。

**Noisy Gating**（仅训练模式）：

```
p_noisy = softmax( (W_route · z_t + b + ε · softplus(W_noise · z_t)) / τ )
```

#### 6.1.3 参数表

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `input_dim` | 200 | 市场状态向量维度 |
| `latent_dim` | 16 | Regime 潜空间维度 |
| `n_experts` | **10** | 策略专家总数 |
| `top_k` | 3 | 每次激活的专家数（稀疏目前为声明，见附录 C） |
| `τ_train` | 1.0 | 训练时温度（软路由） |
| `τ_val` | 1.0 | 验证时温度（确定性） |
| `τ_inference` | 0.1 | 推理时温度（近离散） |
| `noisy_gating` | True | 训练时是否使用噪声门控 |

#### 6.1.4 三种运行模式

```python
router(s_t, mode="train")      # τ=1.0 + noisy gating
router(s_t, mode="val")        # τ=1.0, 无噪声, 确定性
router(s_t, mode="inference")  # τ=0.1, 近离散
```

---

### 6.2 KDA Market Memory

> **Inspiration**: Kimi Delta Attention (KDA) — per-channel forget gate + delta-rule state update

#### 6.2.1 功能

维护一个固定大小的记忆矩阵 M_t ∈ R^{128×64}，每个时间步通过 delta-rule 在线学习机制更新，不需要存储整个历史序列。

#### 6.2.2 数学公式

**Per-Channel Forget Gate**（低秩瓶颈）：

```
α_t = σ(exp(A_log) · (SiTU(W_up · SiTU(W_down · s_t)) + dt_bias))  ∈  (0.001, 1)^{128}
```

**Route-Modulated Forgetting**（CDAP 连接：Router → Memory）：

```
α'_t = α_t ⊙ σ(W_route · z_t)   （路由信号调制遗忘）
α''_t = α'_t ⊙ cdap_gate        （CDAP 联合空间反馈）
```

**Delta-Rule 状态更新**：

```
k_t = L2Norm(W_k · s_t)         # L2 归一化关键向量（防数值爆炸）
v_t = W_v · s_t                  # 值向量
β_t = σ(W_β · s_t)               # 可学习更新步长 ∈ (0, 1)

M_t = M_{t-1} - β_t · k_t ⊗ (M_{t-1} · k_t) + β_t · k_t ⊗ v_t
```

**记忆检索**：

```
o_t = M_t^T · q_t,   q_t = W_q · s_t
output = RMSNorm(σ(W_out_up · SiTU(W_out_down · s_t)) ⊙ o_t)
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

三个信息流——Router 输出 p_t、Memory 状态 M_t、Depth 特征 {h_0, h_1, h_2}——在传统架构中是**单向独立**的。CDAP 将它们投影到一个共享联合空间，实现**双向调制**。

#### 6.3.2 数学公式

**Step 1 — 三维投影到联合空间**：

```
e = f_{expert→joint}(p_t)          ∈ R^64
m = f_{memory→joint}(flatten(M_t)) ∈ R^64
d = f_{depth→joint}(concat(h))      ∈ R^64
```

**Step 2 — 逐元素乘法融合** ★ 关键设计决策：

```
j = e ⊙ m ⊙ d   ∈ R^64
```

**为什么是乘法不是加法？**
- 加法假设三个维度独立贡献 → memory 不确定时仍会加噪声到路由信号
- 逐元素乘法 → 任何维度接近零，整条调制通路被静音
- 强归纳偏置，适合稀疏的、行情依赖的计算

**Step 3 — 反向投影（Joint → 各组件）**：

| 方向 | 公式 | 效果 |
|------|------|------|
| → Router | p'_t = softmax(**log p_t** + δ · W^router_out · j) | 记忆+深度修正路由偏好（**logit 空间**） |
| → Memory | g_t = σ(W^memory_out · j) ∈ (0,1)^{128} | 额外遗忘门调制 |
| → Depth | w_t = softmax(W^depth_out · j) ∈ Δ² | 跨层特征检索权重 |

> **v0.1.0（旧规则:v0.3.0 代码义） 修正**：路由调制在 **logit 空间**进行（`softmax(log p + δ·bias)`），
> 保证零调制（δ=0 或 j=0）时严格无扰动——旧实现直接在概率空间加偏置，
> 初始化即扭曲路由。

**Step 4 — 融合输出**：

```
h_fused = Σ_{k=0}^{2} w_t^(k) · h_k
```

**CDAP Scales**（可学习的三维调制强度）：

```
expert_bias_scale   — 记忆+深度对路由的调制强度
memory_gate_scale   — 路由+深度对记忆的调制强度
depth_weight_scale  — 路由+记忆对深度的调制强度
```

三个 scale 初始化为 0（zero-init），训练中逐步放开。

#### 6.3.3 代码接口

```python
outputs = cross_dim_attn(
    routing_probs=p_t,          # (B, 10) router 输出
    memory_state=M_t,           # (B, 128, 64) memory 矩阵
    layer_outputs=[h0, h1, h2], # 3 层特征, each (B, 64)
    mode="train"
)
# returns: routing_mod, memory_gate, depth_weights, h_fused
```

---

### 6.4 Adaptive Hardening Mechanism

> ★ DAFT 的**第二大原创贡献**（**研究性实现，默认禁用**）

#### 6.4.1 设计动机

Kimi K3 使用静态的 3:1 比例。DAFT 的 AHM 从数据中学习快/慢路径比例，而非硬编码：

```
低波动趋势市 → 大部分时间很常规 → 应该 ~90% 走快路径
高波动事件市 → 大部分时间很反常 → 应该 ~10% 走快路径
```

#### 6.4.2 四大机制

**1. Pattern Counter（模式计数）**

```python
pattern = discretize_pattern(routing_probs)  # Top-3 专家索引排序
key = (regime_id, pattern)
pattern_counter[key] += 1
```

**2. Cache Builder（缓存构建）**

```
硬化条件：count(key) ≥ θ_harden  AND  confidence > ρ_min
→ 缓存该 (regime, pattern) 的三维调制向量
```

**3. Entropy Guard（熵守卫）**

```python
if H(p_recent) > λ_entropy × H_baseline:
    degrade_to_full_exploration()
```

**4. Staleness Eviction（过期驱逐）**

长时间未命中 + 命中次数 < 阈值的缓存项 → 清除。

#### 6.4.3 当前状态

> **v0.1.0（旧规则:v0.3.0 代码义） 澄清**：AHM 为**研究性实现，默认禁用**。`ExpertEnsemble` 的
> `use_hardening` 参数默认 `False`。硬化快路径的完整接线（`detect_regime_shift`、
> `evict` 在生产管线中的调用）尚未完成，见附录 C 待办。

---

## 7. 专家池

### 7.1 BaseExpert 接口

```python
class BaseExpert(nn.Module):
    def __init__(self, input_dim=200, hidden_dim=64, n_layers=2, name="base"):
        # MLP backbone: Linear → LayerNorm → SiLU → Dropout (× n_layers)
        # Prediction head: hidden_dim → 1
        # Output activation: SiTU (σ(x) ⊙ tanh(x))

    def forward(s_t, return_hidden=False) → signal ∈ [-1, 1]

    @abstractmethod
    def _regime_filter(panel) → mask: Tensor[bool]   # 哪些时刻该专家激活

    @abstractmethod
    def compute_loss(pred, target, mask) → loss      # 专家特定损失
```

### 7.2 五类专家（10 实例）

| 类型 | 实例数 | hidden_dim | Regime Filter | 损失函数 |
|------|--------|-----------|---------------|----------|
| **TrendExpert** | 2 | 64 | ADX > 25（趋势市） | Direction-weighted MSE（方向错 ×11） |
| **ReversalExpert** | 2 | 64 | ADX < 20（震荡市） | Negative Rank IC |
| **VolatilityExpert** | 2 | 48 | Vol > P80（高波动） | MSE + 0.01·Var(pred) |
| **EventExpert** | 2 | 48 | 全部数据（兜底） | BCE on return direction |
| **MomentumExpert** | 2 | 64 | 20日截面动量显著非零 | Direction-weighted MSE（方向错 ×8） |

**MomentumExpert（v0.1.0（旧规则:v0.3.0 代码义） 新增）**：专注**截面动量** regime，与纯趋势跟踪区分——
趋势专家捕捉方向性 ADX 过滤运动，动量专家捕捉"赢家恒赢"的截面持续性
（20 日形成期横截面动量显著非零，且 top/bottom decile 价差拉大）。
研究依据：Chichernea et al. (JFE 2021) 论证动量与反转是**独立的、regime 依赖**
现象；DHMoE (AAAI 2025) 支持细粒度专家专业化。方向错惩罚 ×8，介于趋势
（×11）与事件（×4）之间。

### 7.3 SiTU 激活函数

```
SiTU(x) = σ(x) ⊙ tanh(x),  输出 ∈ [-1, 1]
```

来自 Kimi K3。自然有界，防止专家间输出量级漂移导致 MoE 门控梯度失真。

### 7.4 专家内部结构

```
s_t (200) → Linear(200→64) → LayerNorm → SiLU → Dropout(0.1)
          → Linear(64→64) → LayerNorm → SiLU → Dropout(0.1)
          → Linear(64→1) → SiTU → signal ∈ [-1, 1]
```

10 专家核心约 31.5 万参数。

---

## 8. 集成层与模型工厂

### 8.1 ExpertEnsemble

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

### 8.3 模型工厂 factory.py（v0.1.0（旧规则:v0.3.0 代码义） 新增）

历史上 `build_experts` / `build_ensemble` / `build_layer_proj` 在 7 个脚本里
各复制一份，**n_experts 8 vs 10 的 forward 崩溃正是这样漂出来的**。v0.1.0（旧规则:v0.3.0 代码义）
新增 `src/daft/models/factory.py` 提供单一权威入口：

```python
from daft.models.factory import build_model, build_experts, build_ensemble

experts = build_experts()       # 10 专家池：5 类 × 2 实例
model, layer_proj = build_model(cdap_strength=0.1)  # 一键组装 + 一致性守卫
```

标准架构常量（与 `tests/conftest.py` 一致）：

```python
INPUT_DIM=200, LATENT_DIM=16, D_K=128, D_V=64, N_LAYERS=3,
JOINT_DIM=64, N_EXPERTS=10, TOP_K=3
```

`build_ensemble` 内置 n_experts 一致性 assert，专家数 / router / CDAP 不一致时
立即报错。

---

## 9. 训练管线

### 9.1 四阶段总览

```
Stage 1: 独立专家训练 ──→ Stage 2: Router + Memory ──→ Stage 3: CDAP 联合微调 ──→ Stage 4: 硬化收集
(experts only)             (experts frozen)              (all unfrozen, low LR)        (no gradient)
```

### 9.2 Stage 1 — 独立专家训练

每个专家在自己的行情子集上学到有效的预测策略。

```
训练参数: 专家 MLP 权重 + head
冻结参数: Router, Memory, CDAP, AHM
流程: regime_filter 选样本 → 专家特定 loss → 15 epochs → 保存 checkpoint
```

### 9.3 Stage 2 — Router + Memory 训练

```
训练参数: Router + Memory + CDAP scale（温和）
冻结参数: 所有专家
损失: L = L_MSE + λ_entropy · H(routing)
LR: 1e-4, 20 epochs
```

### 9.4 Stage 3 — CDAP 联合微调

```
训练参数: 全部
LR: 1e-5（极低，防灾难性遗忘）
前 3 epochs 只训 CDAP scale（突破 zero-init），后 7 epochs 全参数
```

### 9.5 Stage 4 — 硬化收集

```
梯度关闭, 200+ 步前向, τ=0.1
输出: 缓存表 + 基线路由熵 + 硬化统计
```

---

## 10. 组合优化

### 10.1 Markowitz 均值-方差优化

```
max_w    w^T · μ - γ · w^T · Σ_shrunk · w

s.t.     w_i ≥ 0        (long-only)
         Σ w_i = 1       (全额投资)
         w_i ≤ w_max     (单仓位上限, 默认 5%)
```

- **μ** = DAFT 输出的预期收益信号（截面方向，非量级）
- **Σ_shrunk** = Ledoit-Wolf 收缩协方差矩阵
- **γ** = 风险厌恶系数

### 10.2 参数表

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `risk_aversion` | 1.0 | γ：风险厌恶系数 |
| `max_weight` | 0.05 | 单票最大权重 |

---

## 11. 回测引擎

### 11.1 矢量化回测

```python
from daft.backtest import BacktestEngine

engine = BacktestEngine(config={
    "transaction_cost_bps": 2.0,
    "slippage_bps": 1.0,
    "top_quantile": 0.2,
    "long_only": True,
    "annualization": 252,
    "rebalance_freq": 1,            # 调仓频率（v0.1.0（旧规则:v0.3.0 代码义））
    "weight_mode": "equal",         # equal | signal_zscore（分数仓位）
    "signal_smoothing": 0.0,        # EMA 平滑系数 λ（0=不启用）
    "respect_mask_no_trade": True,  # 成交约束（停牌/涨跌停日禁开平仓）
})
results = engine.run(signals, returns, panel, mask=panel.mask)
```

### 11.2 信号 → 持仓转换

```
信号 s ∈ R^N → 截面排名 → top_quantile(=20%) 的股票做多
long_only=False 时: bottom_quantile 等权做空
```

**仓位模式（v0.1.0（旧规则:v0.3.0 代码义） 新增）**：`weight_mode` 决定候选股内部的权重分配：

| 模式 | 说明 |
|------|------|
| `equal` | top-k 候选等权（原行为） |
| `signal_zscore` | 按信号 z 分数加权（候选内），信号越强仓位越重 |

### 11.3 交易制度约束（v0.1.0（旧规则:v0.3.0 代码义） 新增）

A 股的两条硬约束在回测层落实：

1. **T+1**：日线收盘-收盘持仓结构下，t 日收盘建仓的头寸最早于 t+1 日收盘
   平仓（持有满 1 日），T+1 约束结构性满足。
2. **停牌/涨跌停成交约束**：`respect_mask_no_trade=True`（默认）时，mask=False
   的交易日（停牌或触及涨跌停）既不能开仓也不能平仓——该资产持仓维持昨收、
   不计换手与成本。信号选择阶段已把 masked 资产排除在建仓之外；此处补齐
   "已持仓资产在跌停日卖不掉"这一半。

**调仓频率（v0.1.0（旧规则:v0.3.0 代码义） 新增）**：`rebalance_freq` 控制每 N 根 bar 才换一次仓，
其间持仓不变、成本只在换仓日计提。实验 EXP-20260816-08 显示降频（freq=5）
比信号平滑更有效地压换手且不衰减 IC（见第 13 章）。

**信号平滑**：`signal_smoothing=λ` 用因果 EMA 平滑信号
（s'_t = (1-λ)·s_t + λ·s'_{t-1}），降换手不引入 look-ahead；λ=0 等价原信号。

### 11.4 交易成本模型

```
总成本 = (tc_bps + slippage_bps) × turnover
```

成本只在换仓日计提；停牌/涨跌停日持仓维持昨收不产生换手成本。

### 11.5 绩效指标

| 指标 | 公式 | 说明 |
|------|------|------|
| **Sharpe Ratio** | (μ - r_f) / σ × √252 | 年化风险调整收益 |
| **Max Drawdown** | max(peak - trough) / peak | 净值百分比回撤 |
| **Rank IC** | Spearman(signal[t], forward_return[t+1]) | 逐时步截面 rank IC |
| **ICIR** | mean(IC) / std(IC) | IC 的信息比率 |
| **Hit Rate** | P(sign(signal) = sign(return)) | 方向准确率 |
| **Turnover** | mean(\|Δw\|) | 真实仓位换手 |

> **v0.1.0（旧规则:v0.3.0 代码义） 修正**：IC 对齐统一为 **k→k+1**（signal[t] 预测 p[t+1]−p[t]）；
> val-IC 为**逐时步截面 rank IC**（旧为 pooled Pearson，ICIR/t-stat 退化）；
> 回测换手率为**真实仓位换手**；MaxDD 为**净值百分比**回撤。

---

## 12. 配置系统

> **v0.1.0（旧规则:v0.3.0 代码义） 澄清**：`configs/*.yaml` 为**参考死配置**，未接入运行脚本。
> 实际配置在各脚本的 `DEFAULT_CONFIG` 字典中。修改参数请直接改脚本内的
> `DEFAULT_CONFIG`，或后续将 yaml 接线（见附录 C 待办）。

---

## 13. 实验结果

### 13.1 新口径基线（2021-2025 日线，前复权，严格样本外）

> 修复前全部实验数字作废（错列特征产物）。以下为通道契约与口径修复后的
> 唯一有效数字，产物为 `outputs/EXP-20260816-*.json`，完整登记与 config hash
> 见 `docs/EXPERIMENT_REGISTRY.md`。

| 实验 | 变体 | IC | t | 净 Sharpe | 真实换手 | 结论 |
|------|------|-----|---|-----------|----------|------|
| EXP-20260816-02 | Ridge / 30股 | +0.0001 | +0.01 | −1.10 | 1.74 | 30 股无信号（股票池效应） |
| EXP-20260816-03 | DAFT / 30股 quick | +0.0077 | +0.51 | −1.66 | 2.37 | 30 股同数量级 ≈0 |
| EXP-20260816-04 | DAFT / 30股 平滑 λ*=0.7 | +0.0128 | +0.81 | −0.14 | 0.96 | 换手减半，Sharpe −1.66→−0.14 |
| EXP-20260816-05 | **Ridge / 100股 hs300** | **+0.0482** | **+5.19** | **+0.53** | 1.85 | **基线即达 GO 线** |
| EXP-20260816-06 | DAFT / 100股 quick | +0.0368 | +3.65 | −1.72 | 2.34 | 有信号，弱于 Ridge，换手吃收益 |
| EXP-20260816-07 | DAFT / 100股 平滑 λ*=0.7 | +0.0274 | +2.36 | −0.60 | 0.98 | 换手减半，Sharpe 回升，仍输 Ridge |
| EXP-20260816-08 | **DAFT / 100股 换手控制网格** | +0.0353 | +3.50 | **+0.25** | 0.63 | **freq=5+λ=0+分数仓位 → test 净 Sharpe 转正**；降频优于平滑 |
| EXP-20260816-10 | DAFT / 100股 更强平滑 λ∈{0.7,0.8,0.9} | +0.0229 | +1.94 | −0.47 | 0.75 | 平滑越强 Sharpe 越高但 IC 衰减，平滑路线近极限 |
| EXP-20260816-11 | DAFT / 100股 --full 训练 | +0.0251 | +2.51 | −1.33 | 2.15 | 更多训练无益（OOS IC 低于 quick 0.037），加训练量路线证伪 |
| EXP-20260816-12 | Ridge / 100股 train 60% | +0.0482 | +5.19 | +0.56 | 1.83 | Ridge 对训练量不敏感（727 天即够） |
| EXP-20260816-13 | DAFT / 100股 walk-forward 2折 | 0.030±0.025 | — | −1.09±0.55 | 2.20 | 折间极不稳：折1 IC 0.013 弱，折2 IC 0.048 ≈ Ridge |

> EXP-20260816-01（新口径基线、baostock 瞬时失败致股票池不全）与
> EXP-20260816-09（多窗口首跑崩于对齐 bug，已并入 EXP-13）为过程记录，
> 上表从 02 起列有效结果。

### 13.2 关键发现

1. 修复通道契约与对齐后，30 股 Ridge IC 从 0.029 塌到 ≈0——旧"弱信号"是
   错列特征产物；且 30 股无信号本身是**股票池效应**（权重股同质 + 截面太小）。
2. **100 股 hs300 真实成分 + 涨跌停 mask 下，Ridge IC=0.048 / t=5.19 /
   净 Sharpe=+0.53 — 基线即达预注册 GO 线。**
3. DAFT 有信号（IC 0.035~0.037）但弱于 Ridge 且换手更高；换手控制
   （freq=5 + 分数仓位）把净 Sharpe 拉到 **+0.25 转正**，仍低于线性基线。
4. **加训练量证伪**：DAFT --full 训练 OOS IC 0.025 反低于 quick 0.037；
   而 Ridge 对训练量不敏感（EXP-12）。DAFT 的训练量敏感在 walk-forward 折间
   暴露（折1 727d IC 0.013 vs 折2 969d IC 0.048）。

### 13.3 判定预判

对照预注册判据（IC ≥ 0.04 且 t ≥ 2.0 = GO）：DAFT 典型 IC 0.035~0.037 落入
"有条件 GO"区间，但**最优变体仍打不过 Ridge**（0.035 < 0.048），且换手控制
后净 Sharpe +0.25 仍低于 Ridge +0.53 → **判定书草稿预判 NO-GO（架构）方向**。
正式签字前保留三条推翻路径：长历史训练（EXP-13 折2 显示训练量足够时 DAFT
≈ Ridge）、no-cdap/no-memory 消融归因、成本真实性重测。判定书见
`docs/DECISION_20260930.md`。

---

## 14. 扩展指南

### 14.1 如何添加新专家

```python
# 1. 创建 src/daft/models/experts/my_expert.py
from daft.models.experts.base_expert import BaseExpert

class MyExpert(BaseExpert):
    def _regime_filter(self, panel):
        return my_regime_mask
    def compute_loss(self, pred, target, mask):
        return my_loss(pred, target, mask)

# 2. 在 experts/__init__.py 注册
# 3. 在 factory.py 的 build_experts() 中按 5 类 × 2 的约定加入
# 4. 同步 N_EXPERTS 常量（并注意 conftest.py 的一致性）
```

### 14.2 如何接入新数据源

```python
class MyDataAdapter:
    def fetch(self, symbols, start, end) -> Panel:
        # 产出 OHLCV 布局 Panel，其余交给 ensure_base_panel
        return Panel(values=..., mask=..., dates=..., ...)
```

### 14.3 如何自定义路由策略

```python
class MyRouter(RegimeRouter):
    def forward(self, s_t, mode="train"):
        return routing_probs, z_t
# 在 factory.py 中替换 RegimeRouter
```

---

## 附录 A. 完整参数列表

### A.1 模型参数（10 专家）

| 模块 | 参数 | 说明 |
|------|------|------|
| RegimeRouter | input_dim=200, latent_dim=16, n_experts=10, top_k=3 | 潜空间路由 |
| KDAMarketMemory | d_k=128, d_v=64, d_feature=200, bottleneck_ratio=4 | 在线记忆 |
| CDAP | n_experts=10, d_k=128, d_v=64, joint_dim=64 | 三维互调 |
| HardeningEngine | n_regimes=10, n_experts=10, threshold=100 | 非参数化 |
| Experts (×10) | hidden_dim 64/48, n_layers=2 | 核心 ≈31.5 万 |
| layer_proj | 3 层 L0/L1/L2 投影 | 含此 ≈41.7 万 |

### A.2 训练参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| Stage1 epochs / LR | 15 / 1e-3 | 独立专家训练 |
| Stage2 epochs / LR | 20 / 1e-4 | Router+Memory |
| Stage3 epochs / LR | 10 / 1e-5 | 联合微调 |
| Stage4 n_steps | 200 | 硬化收集 |
| CDAP warmup | 3 epochs | scale 预热 |
| sparsity_weight | 0.01 | 熵正则权重（v0.1.0（旧规则:v0.3.0 代码义） 从 0.05 调低） |

### A.3 回测参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `transaction_cost_bps` | 2.0 | 手续费 |
| `slippage_bps` | 1.0 | 滑点 |
| `top_quantile` | 0.2 | 做多分位 |
| `long_only` | True | 仅做多 |
| `rebalance_freq` | 1 | 调仓频率 |

---

## 附录 B. 文件结构索引

```
daft/
├── README.md
├── LICENSE
├── pyproject.toml                    # version 字段需与 __init__.py 同步
│
├── docs/
│   ├── SPECIFICATION.md              # ★ 本文档 — 技术说明书
│   ├── guided-tour.md                # 代码走读导览
│   ├── FIX_REPORT_20260816.md        # v0.1.0（旧规则:v0.3.0 代码义） 修复与重验报告
│   ├── EXPERIMENT_REGISTRY.md        # 实验登记 & 预注册判据
│   ├── ROADMAP.md                    # 路线图与挂起项
│   ├── PROJECT_EVALUATION.md         # 项目评判评分卡
│   ├── DECISION_20260930.md          # Go/No-Go 判定书（草稿）
│   └── ...
│
├── configs/                          # 参考死配置（未接线）
│
├── src/daft/
│   ├── __init__.py                   # __version__
│   │
│   ├── data/
│   │   ├── panel.py                  # Panel 数据结构 (T×N×F)
│   │   ├── loaders.py                # DataLoader + 合成数据
│   │   └── adapters/                 # baostock / yfinance
│   │
│   ├── features/
│   │   ├── base_features.py          # ★ 通道契约 ensure_base_panel
│   │   ├── tensor_factors.py         # 7 GPU 原语
│   │   ├── legacy_factors.py         # 35 经典因子（未接入）
│   │   ├── regime_features.py        # s_t 200 维构造
│   │   └── freq_features.py          # FFT 频域（未接入）
│   │
│   ├── models/
│   │   ├── router.py                 # [C1] RegimeRouter
│   │   ├── memory.py                 # [C2] KDA Market Memory
│   │   ├── cross_dim_attn.py         # [C3] CDAP
│   │   ├── hardening.py              # [C4] AHM
│   │   ├── ensemble.py               # ExpertEnsemble
│   │   ├── factory.py                # ★ 模型工厂（v0.1.0（旧规则:v0.3.0 代码义） 新增）
│   │   └── experts/                  # 5 类专家
│   │       ├── base_expert.py
│   │       ├── trend_expert.py
│   │       ├── reversal_expert.py
│   │       ├── volatility_expert.py
│   │       ├── event_expert.py
│   │       └── momentum_expert.py    # ★ v0.1.0（旧规则:v0.3.0 代码义） 新增
│   │
│   ├── training/
│   │   ├── expert_trainer.py         # Stage 1
│   │   ├── router_trainer.py         # Stage 2
│   │   └── joint_trainer.py          # Stage 3
│   │
│   ├── portfolio/markowitz.py
│   ├── backtest/engine.py
│   └── utils/                        # metrics.py, device.py
│
├── scripts/                          # 运行脚本（DEFAULT_CONFIG 为准）
└── tests/                            # 396 项测试
```

---

## 附录 C. 已知问题和 TODO

### C.1 已知结构性问题

1. **路由最大熵** — routing entropy ratio 接近理论最大值 ln(10) ≈ 2.303，
   10 个专家几乎均匀激活，MoE 退化为普通 ensemble。可能的解决方向：降低
   Top-K、添加熵正则、在真实行情数据上测试、curriculum learning 降温度。

2. **top-3 稀疏未真正落实（挂起）** — 专家池为**稠密软门控**（所有专家都计算，
   软概率加权融合），"Top-K 稀疏"目前只是声明。**决策：挂起**，稠密软门控可跑通
   实验，稀疏是推理优化，判定后再决定（见 ROADMAP 挂起项）。

3. **AHM 未接线（挂起，默认禁用）** — `detect_regime_shift` / `evict` 在生产
   管线中的调用未完成。**决策：挂起**，推理加速不应先于信号验证。

4. **legacy 因子 / FFT 未接入** — 35 个 legacy 因子和 FFT 频域特征是独立模块，
   不参与训练/评估。需接入或明确其定位。

5. **residual-gate-port 分支挂起** — `feat/residual-gate-port`（记忆门收缩先验
   CP1-CP11）的数学推理部分与 v0.1.0（旧规则:v0.3.0 代码义） CDAP logit 空间修复冲突。**决策：挂起，
   不移植**，数学实现保留不动；移植条件是 M5 判定为 GO 或有条件 GO 后按新架构
   重做。

### C.2 已修复（历史记录）

- **CDAP memory_gate 死通路** — memory_gate_scale 早期始终为 0，2026-08-07
  通过修复 safe_gate 与 route_modulate 的梯度断路解决。
- **通道契约错列** — 2026-08-16 通过 base_features.py::ensure_base_panel 解决。
- **n_experts 不一致崩溃** — 2026-08-16 通过 factory.py 单一入口 + 一致性 assert 解决。
- **涨跌停 mask + hs300 成分股池** — 2026-08-16（v0.1.0（旧规则:v0.3.0 代码义）），见 4.6。
- **交易制度约束（T+1 / 成交约束 / 调仓频率 / 分数仓位）** — 2026-08-16（v0.1.0（旧规则:v0.3.0 代码义）），见 11.3。

### C.3 待实现功能

| 功能 | 说明 | 优先级 |
|------|------|--------|
| 消融归因 | no-cdap / no-memory 组件贡献归因 | 高（NO-GO 签字前） |
| 长历史训练验证 | 训练量足够时 DAFT ≈ Ridge 的路径复核（EXP-13 折2） | 高（NO-GO 签字前） |
| 成本真实性重测 | 冲击模型（T+1/涨跌停已做，剩按 ADV 的冲击） | 高（NO-GO 签字前） |
| 配置系统接线 | configs/*.yaml 接入脚本 | 中 |
| 换手率约束 | 组合优化加 turnover constraint | 中 |
| top-3 稀疏 | 真正实现专家稀疏计算（判定后） | 低 |

### C.4 未来方向

- 若 **GO**：扩池到 300 股、Markowitz 组合优化接入、分钟级中频（见 ROADMAP v0.4/v0.5）
- 若 **NO-GO**：转型"LLM 架构金融化改造"研究项目，保留 CDAP/KDA 架构作研究载体，
  面向论文/比赛/教学，停止交易系统投入
- 与 Time-MoE (ICLR 2025) 做系统性的样本外对比
- `feat/residual-gate-port` 在判定为 GO 后按新架构重做移植（见 C.1-5）

---

> **DAFT: Dimension-Aware Financial Trading**
> Author: Alastair (Dongxu Jiang)
> GitHub: [github.com/Alastair-Jiang/Dongxu-Jiang-daft](https://github.com/Alastair-Jiang/Dongxu-Jiang-daft)
> Version: v0.1.0（旧规则:v0.3.0 代码义） · License: MIT
> Inspired by Kimi K3 (Moonshot AI, July 2026)
