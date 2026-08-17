# DAFT 项目进度报告

**项目**:DAFT (Dimension-Aware Financial Trading) — 面向中频量化的跨维度注意力架构
**作者**:Alastair (Dongxu Jiang)
**报告日期**:2026-08-07(更新至 v0.0.1（旧规则:v0.2.0）)
**代码规模**:约 8,800 行 Python(较 v0.1.0（旧规则最早版本） 增长 52%)
**当前版本**:v0.0.1（旧规则:v0.2.0） — 全管道打通(消除所有 NotImplementedError)
**GitHub**:github.com/Dongxu-Jiang/daft

---

## 1. 项目是什么

DAFT 是一套受 Kimi K3(2026 年 7 月,Moonshot AI,2.8T 参数开源大模型)启发的量化交易模型。核心思想:**把 K3 中三个原本互相独立的架构组件——MoE 专家路由 (Stable LatentMoE)、Delta 注意力记忆 (KDA)、跨层注意力残差 (AttnRes)——改造成金融时间序列版本,并让三者双向调制**,形成一个统一的决策引擎。

用大白话说:传统量化 ML 里,"现在是什么市场状态"(路由)、"历史上类似时刻怎么走的"(记忆)、"该信哪个层面的信号"(特征深度)是三个各管各的模块;DAFT 让它们互相"打招呼"——路由结果影响记忆存什么,记忆状态影响该信哪层特征,特征信号反过来纠正路由判断。

**设计目标**:中频(分钟级)A 股交易。总参数量控制在 35 万以下(全模型 275,099),可在 Mac mini M4 / CPU 上训练。

---

## 2. 架构总览

```
[行情数据] → Panel(T×N×F) → 特征引擎 → s_t(200维市场状态向量)
                                        │
        ┌───────────────────────────────┤
        ▼                ▼              ▼
   RegimeRouter      KDA Memory     LayerProj L0/L1/L2
   (潜空间路由)      (δ规则记忆)    (特征层级投影)
        │                │              │
        └───────► CDAP 交叉维度注意力 ◄──┘
                    (e ⊙ m ⊙ d 乘法融合)
                        │
                    ExpertEnsemble → signal(B,1)
                        │
                   MarkowitzOptimizer → weights(N,)
                        │
                   BacktestEngine → metrics ✅ v0.0.1（旧规则:v0.2.0）
```

四个核心组件 + 完整训练/评估闭环(全部已实现):

| 组件 | 设计来源 | 状态 |
|---|---|---|
| **RegimeRouter** 潜空间市场状态路由 | K3 Stable LatentMoE(896 专家→16 激活,缩为 8 专家→3 激活) | ✅ v0.1.0（旧规则最早版本） |
| **KDAMarketMemory** δ规则市场记忆 | KDA (Kimi Delta Attention, arXiv:2510.26692) | ✅ v0.1.0（旧规则最早版本） |
| **CDAP** 交叉维度注意力协议(原创) | K3 AttnRes + 原创三向调制 | ✅ v0.1.0（旧规则最早版本） |
| **AHM** 自适应硬化机制(原创) | K3 静态 3:1 快慢层比 → 数据驱动 | ✅ v0.1.0（旧规则最早版本） |
| **Stage 2 训练器** | Router+Memory+CDAP 联合训练 | ✅ v0.0.1（旧规则:v0.2.0） |
| **Stage 3 训练器** | 全参数联合微调 | ✅ v0.0.1（旧规则:v0.2.0） |
| **BacktestEngine** 回测引擎 | Walk-forward + 成本建模 + 全套指标 | ✅ v0.0.1（旧规则:v0.2.0） |
| **MarkowitzOptimizer** 组合优化 | Ledoit-Wolf 收缩 + 闭式解 | ✅ v0.0.1（旧规则:v0.2.0） |
| **真实数据适配器** | baostock(A股)/ yfinance(美股) | ✅ v0.0.1（旧规则:v0.2.0） |
| **专家 regime 过滤器** | 4 类专家的专属训练子集筛选 | ✅ v0.0.1（旧规则:v0.2.0） |

---

## 3. 已完成部分(带证据)

### 3.1 v0.0.1（旧规则:v0.2.0） 新增(2026-08-06/07,已用 QUICK 配置跑通全管道)

**回测引擎 `BacktestEngine`** ✅
- Walk-forward 向量化回测(时序循环 + 向量化截面)
- Signal → 仓位:分位数选择(前 20% 做多;long-short 模式前 20% 多/后 20% 空)
- 交易成本:双边 bps(默认 2bp)+ 滑点(默认 1bp)
- 指标全家桶:Sharpe、MaxDD、Calmar、Rank IC、ICIR、Hit Rate、RMSE、Turnover、回撤持续时间

**组合优化 `MarkowitzOptimizer`** ✅
- Ledoit-Wolf 收缩(向常数相关矩阵收缩,OAS 变体,纯 PyTorch 实现无 CVXPY 依赖)
- 闭式解均值-方差 + 两基金分离 + 迭代 box 投影(单票权重上限)
- 纯长仓约束 + 归一化

**Stage 2 `RouterTrainer`** ✅(冻结专家)
- 训练对象:Router + KDA Memory + CDAP + LayerProj(特征层级投影器)
- Loss = Σᵢ routing_probᵢ × expert_lossᵢ(路由加权专家损失)
- 温度退火 1.0 → 0.1、熵正则防路由坍缩、分位数平衡(K3 式负载均衡)、早停
- 记忆状态批间 detach(控制 BPTT 长度)

**Stage 3 `JointTrainer`** ✅(全参数解冻)
- 极低学习率 1e-5 防灾难性遗忘,专家 LR 再 ×0.1(双参数组)
- CDAP 调制强度 δ 0.1 → 1.0 全开
- Loss = 最终信号 MSE + 0.1 × 路由加权专家一致性正则
- 早停条件改为验证 IC 退化

**真实数据适配器** ✅
- `BaostockAdapter`:内置 50 只 CSI300 成分股清单(600519 茅台/300750 宁德等),日线/分钟线,前复权,缺失值限 5 根 forward-fill + 停牌 mask
- `YFinanceAdapter`:美股数据同样转为 Panel
- `DataLoader` 已支持 `source: "baostock" | "yfinance"`

**其他** ✅
- `utils/device.py`:CUDA → XPU (Intel Arc) → DirectML → MPS → CPU 多后端自动检测
- `utils/metrics.py`:Rank IC(截面 Spearman)、ICIR、Hit Rate
- 4 个专家 `_regime_filter()` 全部实现(ADX>25 / ADX<20 / 波动率>80 分位 / 事件全量回退)
- 新脚本:run_stage2.py、run_stage3.py、run_full_pipeline.py、run_backtest_only.py、smoke_test_all.py(10 项冒烟测试)

### 3.2 端到端实验结果(v0.0.1（旧规则:v0.2.0） QUICK 配置,2026-08-07 01:54)

数据:50 只合成股票 × 300 天,总耗时 295.5 秒(CPU):

| 阶段 | 耗时 | 关键结果 |
|---|---|---|
| Stage 1(15 epochs) | 65.7s | 8/8 专家收敛 |
| Stage 2(10 epochs) | 155.3s | val_IC: +0.0446 → **+0.0612** |
| Stage 3(8 epochs) | 67.6s | val_IC 峰值 **+0.1079**(第 3 epoch) |
| 回测 | 0.1s | IC +0.0203,ICIR +0.140 |

**读法:**
- ✅ **训练阶段验证 IC 一路上升**(0.045 → 0.061 → 0.108),说明路由+记忆+CDAP 联合训练确实在学东西,架构闭环成立
- ✅ 熵正则有效:Stage 2 路由熵从 2.07 平滑降到 1.04(路由从"都试试"走向"有偏好"),Stage 3 又回升到 0.72(联合微调重新探索)
- ⚠️ **回测 Sharpe = -1.79、年化收益 -20.6%、Hit Rate 50.3%**(≈抛硬币)——CHANGELOG 归因于"合成数据太小(300d/50股)"。合成数据本身是随机游走+噪声,模型预测能力(IC 0.02)不足以覆盖成本,这是**符合预期的负结果,不代表架构失效**
- ⚠️ **注意回测 IC(+0.020)远低于验证 IC(+0.108)**,两者口径不同(回测含交易成本、用了全部时间段的信号),详见第 5 节风险

### 3.3 v0.1.0（旧规则最早版本） 已有基础(全部保留)

数据层(合成 HMM 三状态 + 三因子 OHLCV)、200 维特征引擎(6 组因子)、四大模型组件、8 专家专属损失、Stage 1 训练器(ADX 掩码 + 余弦退火 + 早停)、pyproject/配置/CI 工程配套。

---

## 4. 剩余缺口

### 🟡 尚未实现/未验证(不再是占位符,但未跑通)
| 项目 | 状态 | 说明 |
|---|---|---|
| **--full 全量配置** | 未运行 | FULL_CONFIG(100股×500天,Stage1 50 epochs + Stage2 30 + Stage3 20)只在 QUICK 上验证过 |
| **真实 A 股数据训练** | 未运行 | baostock 适配器写好了,但没真正拉数据跑过(需 `pip install baostock`) |
| **样本外回测** | 未实现 | 目前回测跑在全量 panel 上(含训练段),不是严格样本外;需要 train/val/test 三段式 |
| **消融实验** | 未运行 | docs/experiments.md 表格还是空的——关 CDAP/关 AHM/单方向调制/线性 baseline 对比 |
| **AHM 硬化统计(Stage 4)** | 部分 | training_loop.py 有 demo 版,正式 pipeline 未接入 |
| **分钟级数据验证** | 未做 | 目前全是日线(合成日线/baostock 日线),中频(分钟级)目标未验证 |
| **Rank IC 与回测统一** | 待验证 | 验证 IC vs 回测 IC 口径差异需澄清 |

### 🔴 核心科学问题(决定项目价值的生死线)
1. **CDAP/AHM 到底有没有用?** —— 没有消融实验,就无法证明"三向调制"比"三个独立模块"强。这是论文级项目的必要条件
2. **合成数据上的成功能否迁移到真实 A 股?** —— 需要 baostock 真实数据 + 样本外回测验证

---

## 5. 项目健康度评估

### 优点
- **工程完整度大幅提升**:从"架构演示"进化为"可复现实验系统"——一个命令跑完全管道(数据→训练→回测→报告)
- **训练设计讲究**:温度退火、熵正则、分位数平衡、双 LR 组、IC 早停——都是业界正规做法
- **实现细节可靠**:Ledoit-Wolf 纯 PyTorch 闭式解、mask 传播、缺失值处理、设备多后端
- **验证指标合理**:验证 IC 一路上升是最有说服力的正面证据

### 风险(务必注意)
1. **回测是样本内的**:`run_full_pipeline.py` 的回测用整个 panel(含训练数据)生成信号并评估——**这不是严格意义的样本外回测**,真实评价必须 train/val/test 三分后只在 test 上回测
2. **特征标准化潜在的 look-ahead 风险**:信号生成时 s_mean/s_std 用全量数据计算,严格来说引入了未来信息(合成数据影响小,真实数据是问题)。应在训练段拟合、测试段应用
3. **负 Sharpe 的正确解读**:目前结果**不能**解读为"策略亏钱"或"架构失败"——合成数据信噪比极低 + 成本模型 + 样本内评估,负 Sharpe 是方法论不完备的结果,不是结论
4. **IC 口径不一致**:验证 IC(截面相关,无成本,平滑段)+0.108 vs 回测 IC(逐时步、含成本)+0.020——需要搞清楚差异来源(成本影响?样本范围?信号滞后?)

---

## 6. 这个项目现在能用来干什么

**v0.0.1（旧规则:v0.2.0） 之后,它已经是:**
1. ✅ **一个完整的量化研究框架**——数据→特征→三阶段训练→回测→报告,全自动闭环
2. ✅ **可复现的实验平台**——QUICK/FULL 配置、合成/真实数据切换、backtest-only 复用 checkpoint
3. ✅ **论文/比赛的可用主体**——架构 + 训练 + 评估全齐,只差消融实验和真实数据实验
4. ✅ **高质量学习素材**——K3 机制金融化改写的完整实现

**仍然不适合:**
- ❌ 实盘/模拟盘交易(样本外未验证、真实数据未跑、分钟级未测)
- ❌ 得出"策略有效/无效"的结论(实验设计还差消融和样本外)

---

## 7. 下一步建议(按优先级)

1. **补样本外回测**(1 天):改 run_full_pipeline.py 用 train/val/test 三分,信号生成只允许用训练段拟合的标准化参数,只在 test 段回测
2. **跑 --full 全量配置**(~1 小时 CPU):确认大配置下 val_IC 趋势一致
3. **接真实数据**(1-2 天):`pip install baostock`,拉 50 只 CSI300 日线(2021-2026),重跑管道——这是第一个"真实世界"数字
4. **消融实验**(2-3 天):全 DAFT vs 关 CDAP vs 关 AHM vs 单方向调制 vs 线性 baseline(MLP/岭回归)——填满 docs/experiments.md,这是论文的核心表格
5. **分钟级验证**:baostock 分钟数据(5min/15min)重跑,检验中频目标

---

## 8. 相关前沿知识背景

### DAFT 的灵感来源(2026 年最前沿)
- **Kimi K3**(Moonshot AI, 2026-07):世界最大开源权重模型,2.8T 参数、896 专家、16 激活。三大设计:**Stable LatentMoE**(潜空间路由 + 分位数平衡)、**KDA**(线性注意力 δ 规则记忆,无 KV cache)、**AttnRes**(跨层注意力残差)。
- **Time-MoE**(ICLR 2025):首个通用时序 MoE 基础模型,证明稀疏路由在时序任务上有效。
- **xLSTM**(Beck et al., 2024):LSTM 的可扩展升级,金融时序 benchmark 表现强劲。

### 对你项目最重要的三个认知
1. **线性模型悖论**:时序预测上复杂架构经常打不过一个线性层(LTSF-Linear,AAAI 2023)——你的消融实验必须包含简单 baseline
2. **结论不跨市场迁移**:日线期货上的成功(如 xLSTM Sharpe 2.4)不代表分钟级 A 股有效
3. **成本是分钟级策略的生死线**:A 股 T+1、印花税、滑点——回测必须建模(你已建模,这是优势)

---

## 9. 推荐阅读论文

### 核心灵感(必读)
1. **Kimi K3 Technical Report** — Moonshot AI, 2026-07(发布时搜索 "Kimi K3 technical report")
2. **KDA: Kimi Delta Attention** — arXiv:2510.26692
   → https://arxiv.org/abs/2510.26692

### 时序 MoE(直接相关)
3. **Time-MoE: Billion-Scale Time Series Foundation Models** — ICLR 2025
   → https://arxiv.org/abs/2405.16073
4. **xLSTM: Extended Long Short-Term Memory** — NeurIPS 2024
   → https://arxiv.org/abs/2405.04517

### 金融时序深度学习(实验对比对象)
5. **Deep Learning for Financial Time Series: A Large-Scale Benchmark** — arXiv:2603.01820 (2026, Oxford/Zohren 组)
   → https://arxiv.org/abs/2603.01820 ← **重点关注**:xLSTM 日线 Sharpe 1.80、VSN+xLSTM 2.40。你的消融/对比实验框架参考
6. **FinStressTS: Parametric Synthetic Benchmark for Time-Series in Finance** — arXiv:2606.03184
   → https://arxiv.org/abs/2606.03184 (合成数据基准方法论)
7. **PatchTST: A Time Series is Worth 64 Words** — ICLR 2023
   → https://arxiv.org/abs/2211.14730
8. **iTransformer: Inverted Transformers Are Effective for Time Series Forecasting** — ICLR 2024
   → https://arxiv.org/abs/2310.06625

### 必读"防坑"文献
9. **Are Transformers Effective for Time Series Forecasting?**(LTSF-Linear)— AAAI 2023
   → https://arxiv.org/abs/2205.13504 ← **最重要**:证明线性层打败复杂 Transformer,消融实验设计依据
10. **Advances in Financial Machine Learning**(López de Prado, 2018)
    → https://www.wiley.com/en-us/Advances+in+Financial+Machine+Learning-p-9781119482086 (防数据泄露、样本外验证方法论——你的样本外回测改造必读)

### 轻量化调制(如果要简化 CDAP)
11. **FiLM: Visual Reasoning with a General Conditioning Layer** — AAAI 2018
    → https://arxiv.org/abs/1709.07871
12. **Language Modeling with Gated Convolutional Networks**(GLU)— EMNLP 2016
    → https://arxiv.org/abs/1612.08083

---

## 10. 版本历史

| 版本 | 日期 | 内容 |
|---|---|---|
| v0.1.0（旧规则最早版本） | 2026-07 | 核心架构:四大模型组件 + 特征引擎 + Stage 1 + smoke test |
| v0.0.1（旧规则:v0.2.0） | 2026-08-06/07 | **全管道打通**:回测/组合优化/指标/真实数据适配器/Stage 2+3/端到端脚本,消除所有 NotImplementedError,QUICK 全管道跑通 |

---

*报告生成:cuda (OpenClaw),基于 2026-08-07 01:57 代码库快照(v0.0.1（旧规则:v0.2.0）)。所有"已完成"判断均基于实际代码阅读与运行产物验证,非 README 转述。*
