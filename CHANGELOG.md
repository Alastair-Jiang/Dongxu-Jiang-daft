# DAFT 更新日志

## v0.1.0 — 工程修复 + 全面升级 (2026-08-16)（旧规则:v0.3.0 代码义）

### 修复批次(PR #9, 10 提交 squashed)

- **通道契约修复**: 数据源 OHLCV 曾被特征引擎按 `[close, log_return, ...]`
  错列读取, 此前所有实验 s_t 建立在错列上 → `base_features.py` 唯一转换点
  + 12 个语义测试
- **口径统一**: IC 对齐 k→k+1; 换手率改真实仓位; MaxDD 百分比; val-IC 改
  逐时步截面 rank IC; CDAP 改 logit 空间(零调制严格无扰动)
- **脚本健康**: n_experts 统一 10(修 6 个脚本 forward 崩溃); baostock 重试;
  实验产物唯一文件名 + config hash; trend_expert 缺导入
- **环境守卫**: pytest `pythonpath=["src"]` + conftest 源码守卫;
  CI(PR 自动 pytest)上线
- 测试: **384 passed / 1 skipped**

### 全面升级(本轮)

- **共享模型工厂** `src/daft/models/factory.py`: 7 个脚本的重复构建代码
  统一(-189 行), 自带 n_experts 守卫
- **涨跌停 mask**: A 股 ±10%(创业板/科创板 ±20%)涨跌停日不可成交
- **hs300 真实成分股池**: `--universe hs300` 用 baostock 按 start_date 拉取
  沪深300 成分(解除 50 只上限 + 缓解幸存者偏差)
- **扩池重测**(hs300 100 股 + 涨跌停 mask, train 60%):
  - Ridge 基线: **IC +0.048 / t +5.19 / 净 Sharpe +0.53** ← 30 股无信号
    是股票池效应, 100 股下基线即达 GO 线
  - DAFT 100 股: 待登记
- 文档: ROADMAP.md(路线图) / PROJECT_EVALUATION.md(项目评判) /
  FIX_REPORT_20260816.md(修复报告) / EXPERIMENT_REGISTRY 扩池登记

### 决策记录

- `feat/residual-gate-port`(记忆门收缩先验): **挂起不移植**, 数学实现
  保留原样, 待信号验证后按新架构重做移植
- AHM / top-3 稀疏 / Markowitz: 挂起(信号验证优先)

## v0.0.1 — 全管道打通 (2026-08-06/07)（旧规则:v0.2.0）

### 新增模块（消除所有 NotImplementedError）

#### GPU 设备检测 (`src/daft/utils/device.py`)
- 多后端自动检测: CUDA → XPU (Intel Arc) → DirectML → MPS → CPU
- 模块级缓存，进程内仅检测一次
- Intel Arc 140T 核显支持就绪（需 Python 3.11/3.12 + torch-directml）

#### 回测引擎 (`src/daft/backtest/engine.py`)
- Walk-forward 向量化回测
- Signal → 头寸：分位数选择（支持多空）
- 交易成本：双边 bps + 滑点
- 指标：Sharpe, MaxDD, Calmar, IC Rank, ICIR, Hit Rate, RMSE, Turnover, MDD Duration

#### 组合优化 (`src/daft/portfolio/markowitz.py`)
- Ledoit-Wolf 收缩估计（向常数相关矩阵收缩）
- 纯 PyTorch 闭式解（无 CVXPY 依赖）
- Box 约束：迭代 clipping + 重归一化

#### 独立指标模块 (`src/daft/utils/metrics.py`)
- Rank IC：截面 Spearman 秩相关
- ICIR = IC_mean / IC_std
- Hit Rate：方向正确率
- 支持 1D/2D 输入

#### 真实数据适配器 (`src/daft/data/adapters/`)
- `BaostockAdapter`：A 股日线 → Panel (T,N,5)
- `YFinanceAdapter`：美股数据 → Panel (T,N,5)
- `DataLoader.load()` 新增 `source: "baostock"` / `"yfinance"`

#### 专家制度过滤器（4 个专家）
- `TrendExpert._regime_filter()`：ADX > 25
- `ReversalExpert._regime_filter()`：ADX < 20
- `VolatilityExpert._regime_filter()`：滚动 20d 波动率 > 80 分位
- `EventExpert._regime_filter()`：事件窗口 / 全量回退

#### Stage 2 — 路由+记忆+CDAP 训练 (`src/daft/training/router_trainer.py`)
- 冻结专家，训练 Router + KDAMemory + CDAP
- 加权专家损失 + 熵正则化
- 温度退火：1.0 → 0.1
- CDAP 调制强度 δ = 0.1
- Quantile Balancing（K3 式负载均衡）

#### Stage 3 — 联合微调 (`src/daft/training/joint_trainer.py`)
- 全部参数解冻
- 极低学习率 1e-5（防灾难性遗忘）
- 全 CDAP 调制 δ = 1.0
- 专家学习率 ×0.1（更保守）

### 新增脚本
- `scripts/run_stage1.py`：独立专家训练
- `scripts/run_stage2.py`：路由+记忆+CDAP 训练
- `scripts/run_stage3.py`：联合微调
- `scripts/run_full_pipeline.py`：端到端管道（数据 → 训练 → 回测 → 报告）
- `scripts/run_backtest_only.py`：仅回测（加载已有 checkpoints）
- `scripts/smoke_test_all.py`：全部 10 项冒烟测试

### 修改的文件
- `src/daft/data/panel.py`：新增 `slice_time()` 方法
- `src/daft/data/loaders.py`：接入 baostock/yfinance 数据源分派
- `src/daft/models/experts/base_expert.py`：提取共享 ADX/vol 筛选器
- `src/daft/training/expert_trainer.py`：Stage 1 完整实现
- `src/daft/training/__init__.py`：更新导出

### 实验结果（v0.0.1 QUICK 配置）（旧规则:v0.2.0）

| 阶段 | 耗时 | 关键结果 |
|------|------|----------|
| Stage 1 | 66s | 8/8 专家收敛 |
| Stage 2 | 155s | val_IC: +0.045 → +0.061 |
| Stage 3 | 68s | val_IC 峰值 **+0.1079** |
| Backtest IC | — | +0.0203, ICIR=+0.14 |

### 已知限制
- Python 3.13 无 torch-directml wheel → CPU 训练
- 合成数据太小（300d/50股）→ Sharpe 为负（预期内）
- --full 配置（500d/100股/50epochs）待运行

---

## v0.1.0 — 核心架构 (2026-07)（旧规则最早版本，新规则下未重编号）

- MoE 专家池：Trend/Reversal/Volatility/Event × 2
- Regime Router（Stable LatentMoE, K3 设计）
- KDA Market Memory（delta-rule + CDAP 调制）
- Cross-Dimension Attention Protocol（★ 原创贡献）
- Adaptive Hardening Mechanism（★ 原创贡献）
- RegimeFeatureExtractor：200 维市场状态向量 s_t
