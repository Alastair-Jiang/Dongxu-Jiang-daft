# DAFT 项目进度报告

**项目**:DAFT (Dimension-Aware Financial Trading) — 面向中频量化的跨维度注意力架构
**作者**:Alastair (Dongxu Jiang)
**报告日期**:2026-08-07
**代码规模**:约 5,800 行 Python(src/ 7 模块 + scripts/ 8 脚本)
**GitHub**:github.com/Dongxu-Jiang/daft

---

## 1. 项目是什么

DAFT 是一套受 Kimi K3(2026 年 7 月,Moonshot AI,2.8T 参数开源大模型)启发的量化交易模型。核心思想:**把 K3 中三个原本互相独立的架构组件——MoE 专家路由 (Stable LatentMoE)、Delta 注意力记忆 (KDA)、跨层注意力残差 (AttnRes)——改造成金融时间序列版本,并让三者双向调制**,形成一个统一的决策引擎。

用大白话说:传统量化 ML 里,"现在是什么市场状态"(路由)、"历史上类似时刻怎么走的"(记忆)、"该信哪个层面的信号"(特征深度)是三个各管各的模块;DAFT 让它们互相"打招呼"——路由结果影响记忆存什么,记忆状态影响该信哪层特征,特征信号反过来纠正路由判断。

**设计目标**:中频(分钟级)A 股交易。总参数量控制在 35 万以下(实际全模型 275,099),可在 Mac mini M4 / CPU 上训练。

---

## 2. 架构总览

```
[行情数据] → Panel(T×N×F) → 特征引擎 → s_t(200维市场状态向量)
                                        │
        ┌───────────────────────────────┤
        ▼                ▼              ▼
   RegimeRouter      KDA Memory     FactorLayers
   (潜空间路由)      (δ规则记忆)    (L0/L1/L2 特征层级)
        │                │              │
        └───────► CDAP 交叉维度注意力 ◄──┘
                    (e ⊙ m ⊙ d 乘法融合)
                        │
                    ExpertEnsemble → signal(B,1)
                        │
              [PortfolioOptimizer] → [BacktestEngine]  ← 未实现
```

四个核心组件(全部已实现,代码完整):

| 组件 | 设计来源 | 状态 |
|---|---|---|
| **RegimeRouter** 潜空间市场状态路由 | K3 Stable LatentMoE(896 专家→16 激活,缩为 8 专家→3 激活) | ✅ 完成 |
| **KDAMarketMemory** δ规则市场记忆 | KDA (Kimi Delta Attention, arXiv:2510.26692) | ✅ 完成 |
| **CDAP** 交叉维度注意力协议(原创) | K3 AttnRes + 原创三向调制 | ✅ 完成 |
| **AHM** 自适应硬化机制(原创) | K3 静态 3:1 快慢层比 → 数据驱动 | ✅ 完成 |

---

## 3. 已完成部分(带证据)

### 3.1 数据层 ✅
- **`SyntheticDataGenerator`**:HMM 三状态马尔可夫 regime(牛/熊/震荡,95% 状态持续性)+ 三因子 CAPM 收益生成 + OHLCV 合成。带 ground-truth regime 标签——这是评估 router 是否学到真实市场状态的"免费答案卷"。
- **`Panel` 数据结构**:T×N×F 张量 + 可交易 mask + 时间序列切分(不打乱时间顺序)。

### 3.2 特征层 ✅
- **`RegimeFeatureExtractor`**:输出 200 维市场状态向量 s_t,6 组特征:
  - 价格/收益动态(55 维):多周期收益、Sharpe 代理、偏度/峰度、回撤、自相关
  - 波动率结构(40 维):多周期已实现波动率、Parkinson 极差波动率、波动率的波动率、ATR
  - 量能/流动性(30 维):量比、OBV、量价相关、换手代理
  - 技术/动量(35 维):RSI、随机指标、MACD、布林带 %B、动量、ADX
  - 横截面(30 维):排名、离散度、市场宽度、beta 代理
  - 频谱(10 维):FFT 频带能量、谱质心、谱熵
- **`TensorFactorEngine`**:mask 感知的 GPU 向量化因子原语(rank/corr/ewma/delta/sum/std)——质量高,与 ml-quant-trading 同源。
- **`FreqFeatureExtractor`**(FFT 周期图):已实现计算逻辑,`forward()` 仍为占位符。

### 3.3 模型层 ✅(全部可跑通)
- **RegimeRouter**:200→100→16 潜空间瓶颈 + SiTU 激活 + Top-3 稀疏路由 + 分位数平衡(无需辅助损失)+ 训练噪声门控 + 温度调度(推理时 0.1 近似离散)。
- **KDAMarketMemory**:128×64 固定状态矩阵,δ 规则在线更新(M -= β·k⊗(M·k) + β·k⊗v),逐通道遗忘门 + K3 Safe Gate 下界 + 输出门 + RMSNorm + 路由调制遗忘(CDAP R→M 连接)。
- **CDAP**(原创):三维(路由分布/记忆状态/层级特征)投影到 64 维联合潜空间,元素级乘法融合 j=e⊙m⊙d,再反向投影修正三维信号。用可学习缩放(`*.scale.tanh()`)做残差式保守调制——这个设计比裸乘法聪明,避免了低激活静音问题。
- **AHM**(原创):模式计数 → 阈值硬化缓存 → 熵守卫(regime 突变时降级全探索)→ 陈旧驱逐。快路径 O(1) 查询。
- **ExpertEnsemble**:全部组件串起来的前向,含快/慢路径分支,输出最终交易信号。
- **8 个专家**(4 类 × 2):趋势/反转/波动/事件,每个有专属损失函数(趋势:方向加权 MSE 符号错罚 11 倍;反转:负秩 IC;波动:MSE+方差正则;事件:方向加权 MSE 罚 5 倍)。

### 3.4 训练系统 ✅(Stage 1 已真实跑通)
- **`Stage1ExpertTrainer`**:ADX/波动率启发式 regime 掩码(替代专家占位的 `_regime_filter`)、余弦退火、早停、梯度裁剪、best-state 恢复、checkpoint 持久化——工程上很正规。
- **Smoke test**(2026-07-25):8/8 通过,梯度流验证通过,参数量 275,099。

### 3.5 Stage 1 训练结果(2026-08-07 01:28,50 epochs,CPU)

数据:100 只合成股票 × 500 天。训练时长 527.6 秒。**8/8 专家全部收敛**:

| 专家 | 初始 loss | 最终 loss | 改善 | 训练轮数 |
|---|---|---|---|---|
| trend #0 | 0.01025 | 0.00287 | **72.0%** | 50 |
| trend #1 | 0.01107 | 0.00286 | **74.1%** | 40(早停) |
| reversal #2 | -0.1879 | -0.2081 | 10.7% | 16(早停) |
| reversal #3 | 0.0180 | -0.2195 | **1320%** | 50 |
| volatility #4 | 0.00340 | 0.00099 | **71.0%** | 33(早停) |
| volatility #5 | 0.00471 | 0.00089 | **81.0%** | 41(早停) |
| event #6 | 0.00473 | 0.00157 | **66.8%** | 41(早停) |
| event #7 | 0.01021 | 0.00150 | **85.3%** | 41(早停) |

> 注:反转专家的 loss 是负秩 IC,所以是负数(越小越好);expert_3 从正转负说明学会了排序相关性。checkpoint 已保存至 `checkpoints/stage1/`。

### 3.6 工程配套 ✅
- pyproject.toml(black/ruff/mypy 配置齐全)、configs(small/paper/hardening 三套实验配置)、GitHub PR 报告自动生成工作流、流程图生成脚本、MIT 许可证、README(32KB,含完整架构文档)。

---

## 4. 未完成部分(重要)

### 🔴 核心缺口(决定项目能否"产出数字")

| 模块 | 状态 | 影响 |
|---|---|---|
| **`BacktestEngine.run()`** | `NotImplementedError` | **无法回答"策略赚不赚钱"**——全项目最关键的缺口 |
| **`MarkowitzOptimizer.optimize()`** | `NotImplementedError` | 组合优化是空的,信号无法转成仓位 |
| **Rank IC** | `NotImplementedError` | 回测核心指标缺失 |
| **`RouterTrainer.train()`(Stage 2)** | `NotImplementedError` | router+memory 联合训练未实现(只有 demo 版) |
| **`JointTrainer.train()`(Stage 3)** | `NotImplementedError` | 全参数微调未实现 |
| **真实数据加载(baostock/yfinance)** | `NotImplementedError` | 目前全部跑在合成数据上,未碰过真实 A 股 |
| **`FreqFeatureExtractor.forward()`** | `NotImplementedError` | FFT 特征未接入主流程 |
| **专家 `_regime_filter()`** | 占位 | 已被 Stage1 的 ADX 掩码替代,不算阻塞 |

### 🟡 实验文档空缺
- `docs/experiments.md` 的消融实验表格(去 CDAP / 去 AHM / 各方向调制 / 线性 baseline / Transformer baseline)全是空行——**这是论文级项目必须补的,也是证明架构价值的唯一方式**。
- `docs/architecture.md` 部分章节是注释骨架。

---

## 5. 这个项目现在能用来干什么

**诚实回答:现阶段(Stage 1 完成)它还不是一个交易系统,但已经是:**
1. **一个能跑通的端到端研究原型**——数据→特征→训练→checkpoint 闭环已验证,证明架构可行性
2. **一份高质量的研究骨架**——想发论文/做课程设计,架子已经立好了,缺的是实验数据
3. **一个学习机器**——整套 K3 机制的金融化改写,代码本身就是最好的教材

**还差 BacktestEngine + 真实数据,才能变成:**
- 策略回测平台(合成数据上验证多专家路由的有效性)
- A 股中频信号生成器(接 baostock 后)
- 论文的实验主体(消融表 + baseline 对比)

**它现在不适合:**
- ❌ 实盘/模拟盘交易(没有回测、没有成本建模、没有真实数据验证)
- ❌ 作为"能赚钱的策略"看待(合成数据上的 loss 下降 ≠ 真实市场 alpha)

---

## 6. 项目健康度评估

**优点:**
- 架构还原度高,K3 机制的金融化改造有思考(不是简单搬运)
- 工程规范:mask 传播、防 NaN、早停、checkpoint、配置分离都做了
- 损失函数设计有讲究(方向加权、秩 IC)
- 参数量控制得当(27.5 万,CPU/M4 可训)

**风险:**
1. **复杂度 vs 收益未验证**:还没有任何证据证明 CDAP/AHM 比简单 MLP 强——消融实验是生死线
2. **合成数据陷阱**:HMM 生成的"可预测"数据会让所有模型都显得很聪明,真实 A 股是另一回事
3. **Stage 2/3 未实现**:router+memory 联合训练才是 DAFT 的灵魂,现在只完成了"专家预训练"
4. **回测/成本缺失**:没有交易成本模型,任何 Sharpe 都是幻觉
5. **数据泄露风险**:KDA 记忆的时序状态在训练中必须严格因果,设计时要特别小心

---

## 7. 相关前沿知识背景

### DAFT 的灵感来源(2026 年最前沿)
- **Kimi K3**(Moonshot AI, 2026-07):世界最大开源权重模型,2.8T 参数、896 专家、16 激活。三大设计:**Stable LatentMoE**(潜空间路由 + 分位数平衡,无需负载均衡辅助损失)、**KDA Kimi Delta Attention**(线性注意力 δ 规则记忆,无 KV cache)、**AttnRes**(跨层注意力残差)。
- **Time-MoE**(ICLR 2025):首个通用时序 MoE 基础模型,证明稀疏路由在时序任务上有效且高效。
- **xLSTM**(Beck et al., 2024):将 LSTM 升级为可扩展架构,在金融时序 benchmark 上表现强劲(详见论文推荐)。

### 对你的项目最重要的三个认知
1. **线性模型悖论**:时序预测上复杂架构经常打不过一个线性层(LTSF-Linear 论文),所以你的消融实验必须包含简单 baseline——这是证明 DAFT 价值的唯一途径。
2. **结论不跨市场迁移**:日线期货上的成功(如 xLSTM Sharpe 2.4)不代表分钟级 A 股有效,必须在自己目标数据上验证。
3. **成本是分钟级策略的生死线**:A 股 T+1、印花税、滑点——回测必须建模,否则一切数字无意义。

---

## 8. 推荐阅读论文

### 核心灵感(必读)
1. **Kimi K3 Technical Report** — Moonshot AI, 2026-07
   → 项目架构的源头。发布时搜索 "Kimi K3 technical report"
2. **KDA: Kimi Delta Attention** — arXiv:2510.26692
   → https://arxiv.org/abs/2510.26692 (记忆组件的直接来源)

### 时序 MoE(直接相关)
3. **Time-MoE: Billion-Scale Time Series Foundation Models** — ICLR 2025
   → https://arxiv.org/abs/2405.16073
4. **xLSTM: Extended Long Short-Term Memory** — NeurIPS 2024
   → https://arxiv.org/abs/2405.04517

### 金融时序深度学习(实验对比对象)
5. **Deep Learning for Financial Time Series: A Large-Scale Benchmark** — arXiv:2603.01820 (2026, Oxford/Zohren 组)
   → https://arxiv.org/abs/2603.01820 ← **重点关注**:日线多资产 benchmark,xLSTM Sharpe 1.80、VSN+xLSTM 组合 2.40。你的 Q3 对比实验应参考此框架
6. **FinStressTS: Parametric Synthetic Benchmark for Time-Series in Finance** — arXiv:2606.03184
   → https://arxiv.org/abs/2606.03184 (合成数据基准,验证你的合成数据方法论)
7. **PatchTST: A Time Series is Worth 64 Words** — ICLR 2023
   → https://arxiv.org/abs/2211.14730
8. **iTransformer: Inverted Transformers Are Effective for Time Series Forecasting** — ICLR 2024
   → https://arxiv.org/abs/2310.06625

### 必读"防坑"文献
9. **Are Transformers Effective for Time Series Forecasting?**(LTSF-Linear)— AAAI 2023
   → https://arxiv.org/abs/2205.13504 ← **最重要的一篇**:证明线性层打败复杂 Transformer,是你消融实验的设计依据
10. **Advances in Financial Machine Learning**(López de Prado, 2018 书)
    → https://www.wiley.com/en-us/Advances+in+Financial+Machine+Learning-p-9781119482086 (防数据泄露、样本加权、回测方法论的圣经)

### 轻量化调制(如果要简化 CDAP)
11. **FiLM: Visual Reasoning with a General Conditioning Layer** — AAAI 2018
    → https://arxiv.org/abs/1709.07871 (特征级线性调制,CDAP 的轻量替代方案)
12. **Language Modeling with Gated Convolutional Networks**(GLU)— EMNLP 2016
    → https://arxiv.org/abs/1612.08083 (门控线性单元,乘法融合问题的经典修复)

---

## 9. 下一步建议(按优先级)

1. **实现 BacktestEngine**(1-2 天):signal → 仓位 → 扣费净值 → Sharpe/回撤/IC。这是全项目最值钱的一块
2. **接真实数据**(2-3 天):baostock/akshare 拉 5-10 只流动性好的 A 股分钟数据,替换合成数据
3. **Stage 2 训练**:实现 RouterTrainer(router+memory 联合训练,专家冻结)
4. **消融实验**:先跑 baseline(线性模型)+ 完整 DAFT,证明架构价值;再逐个关 CDAP 连接测 ΔSharpe
5. 全部跑通后,`docs/experiments.md` 的表格填满——论文/比赛报告的素材就有了

---

*报告生成:cuda (OpenClaw),基于 2026-08-07 01:30 代码库快照。所有"已完成"判断均基于实际代码阅读与训练产物验证,非 README 转述。*
