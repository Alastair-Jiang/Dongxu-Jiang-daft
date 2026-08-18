# DAFT 更新日志

## 未发布 — K3 纲领正确性修复批次（进行中）

### A1 — Stage 1 专家训练 train/val 切分改时间后段验证（2026-08-18）

- **问题**: `Stage1ExpertTrainer._train_one_expert` 用 `torch.randperm` 随机打乱后切分
  train/val，验证样本与训练样本在时间上随机交错（K3 纲领 `docs/K3_GUIDANCE_2026-08-18.md` A1）。
  时序自相关下随机切分显著低估泛化误差，早停依据的 val loss 被时间泄漏污染，
  所有 Stage 1 专家的"最优 epoch"选择偏乐观。
- **修复**: 新增 `_temporal_split(n_total)` 方法——利用布尔掩码展平的行优先（时间优先）
  顺序，按连续时间切分：val = 最后 `val_frac` 比例的样本（时间上最晚），train = 其余。
  小数据退化保护（n≤1 不切，val_frac≤0 禁用验证，val 至少 1 个但保留 ≥1 训练样本）。
  与主管线严格时序切分保持同一哲学。
- **守卫**: `tests/test_training.py::TestStage1TemporalSplit`（6 项：
  默认/自定义比例、小数据保底、退化、零比例禁用、展平时间序前提验证）
  + `self_check.py` 公式抽查新增 2 项断言（`_temporal_split` 存在；`randperm` 不存在）。
- **影响面**: 仅训练期 val 口径与早停路径，不动模型结构与 OOS 推理逻辑；
  既有实验结论（NO-GO 判定）不受影响，但 Stage 1 专家的最优 epoch 选择口径与修复后不可直接比较。

### A2 — Stage 2/3 标准化统计量 train-only 贯穿（2026-08-18）

- **问题**: `RouterTrainer` / `JointTrainer` 的 `_build_dataset` 对
  train / val 段各自拟合 mean/std，早停与选型依据的 val-IC 分布 ≠
  OOS 推理（train-only 统计）的分布（K3 纲领 `docs/K3_GUIDANCE_2026-08-18.md` A2）。
- **修复**: 统计量只在训练段拟合一次并记录 `trainer.norm_stats`；
  val 段经 `norm_stats=` 参数强制复用；返回签名不变（4 元组，
  `dml_smoke.py` 等外部调用兼容）。
- **守卫**: `tests/test_training.py::TestNormStatsConsistency`（含注入统计量
  精确复算测试）+ `self_check.py` 公式抽查新增 2 项断言。
- **影响面**: 仅训练期 val 口径，不动模型结构与 OOS 推理逻辑；
  既有实验结论（NO-GO 判定）不受影响，但其 Stage2/3 早停路径的
  val-IC 数值口径与修复后不可直接比较。

### A3 — KDA 记忆行语义：按资产对齐（不删行，mask 进门控）（2026-08-18）

- **问题**: KDA 记忆的行语义在训练与推理间不一致。训练时展平流按 mask
  过滤行，记忆第 b 行对应的 (t, 资产) 随每日停牌/涨跌停数漂移，记忆学到的是
  "跨资产混合序列"状态；推理时记忆行=固定资产列。"记忆无增益"消融结论的
  潜在混淆因子（K3 纲领 `docs/K3_GUIDANCE_2026-08-18.md` A3）。
- **修复（方向① 按资产对齐记忆行）**:
  - `models/memory.py`: `forward` 新增 `mask` 参数——mask=0 行
    α→1（不衰减）、β→0（不写入）⇒ M 逐位不变（IEEE 精确），检索输出置零；
    mask=None 走旧路径（外部调用兼容）。
  - `training/router_trainer.py` / `joint_trainer.py`: 展平流不再删行
    （保持 (T-1)·N 网格）；`eff_batch = N`（每批恰一个时间步 ⇒ 记忆行=资产列）；
    mask 透传给模型；路由统计（routing_mean / 负载均衡 current_frac / 熵监控）
    改为只对有效行计算——与修复前"先删行再统计"口径严格等价；
    新增全线停牌日守卫（跳过该批，防负载均衡 KL 发散）。
  - 推理侧 11 个入口脚本（`run_full_pipeline*.py` / `run_*backtest*.py` /
    `run_walk_forward.py` / 两个消融 / `diag_experts.py` / `e2_best_expert.py`）
    逐日循环同口径传 mask——训练/推理记忆语义严格一致。
- **守卫**: `tests/test_memory.py::TestA3MaskSemantics`（5 项：跳过更新逐位不变/
  检索置零/形状等价/None 兼容/混合 mask 抽查）+ `tests/test_training.py::`
  `TestA3MemoryRowSemantics`（6 项：两 trainer 不删行/按日对齐/Stage2+Stage3
  的"训练 val vs OOS 推理"信号与记忆终态逐位一致）+ `self_check.py` 5.5 节
  9 项断言（模型门控/两 trainer/不删行回归守卫/推理脚本全传 mask）。
- **影响面**: 不动模型结构与推理数学（mask=None 时行为不变）；但 Stage2/3 的
  优化轨迹改变（批大小 N vs 旧多日批）——修复后训练出的 val-IC 数值与历史
  实验不可直接比较。既有 NO-GO 判定基于 DAFT vs Ridge 整体差距（0.02→0.05），
  结论不变；"KDA 记忆无增益"归因已加**条件性脚注**
  （`RESEARCH_FINDINGS_2026-08-17.md` 一节），是否值得预注册内复核待 K3
  定夺（纲领 §4.3 待决问题 1，成本 ≈2 个训练 run）。

### A4 — 对决入样口径：双条件 mask[t] AND mask[t+1]（2026-08-18）

- **问题**: DAFT 与 Ridge 的样本入样口径不一致。回测引擎内部虽已是
  双条件（`ret_mask = mask[:-1] & mask[1:]`），但评估层全部单条件：
  Ridge 侧训练入样 + test IC/hit 用 `mask[1:]`（收益实现日单条件），
  DAFT 侧 test IC/hit 同样单条件——涨停日（信号日不可交易）但次日
  复牌的样本收益被计入 IC，高估可交易信号预测力；停牌恢复日两侧
  处理不对称（K3 纲领 `docs/K3_GUIDANCE_2026-08-18.md` A4），
  属"公平对决"声明的瑕疵。
- **修复**: `utils/metrics.py` 新增 `eligible_mask(mask)`——唯一口径点，
  输出第 t 行 = `mask[t] & mask[t+1]`（信号日可交易且收益实现日可交易），
  与回测引擎 `ret_mask` 同一公式。13 个对决/评估脚本统一引用：
  - Ridge 5 个（`run_baseline_ridge{,_real,_us,_weekly}.py` +
    `run_feature_ablation.py`）: 训练入样、test 入样、IC、hit 全走双条件
    （`mask_aligned` 形状不变，纯语义收紧，无破坏）；
  - DAFT 8 个（`run_full_pipeline_oos{,_weekly,_us}.py` +
    `run_oos_backtest_only.py` + `run_walk_forward.py` + 两消融 +
    `diag_experts.py`）: test IC/hit 改双条件（回测层本就双条件）。
- **守卫**: `tests/test_metrics.py::TestA4EligibleMask`（7 项：形状与
  逐行语义/全 1/全 0/涨停日剔除/停牌恢复日/与引擎公式逐位一致/float
  兼容）+ `self_check.py` 5.6 节 3 项断言（工具存在/13 脚本引用/
  单条件模式回归守卫）。
- **影响面**: 纯评估口径收紧，不动模型/训练/回测数学。IC/hit 数值
  预期轻微下移（剔除不可交易样本）；既有 NO-GO 判定基于整体差距
  （IC 0.02→0.05 量级 + Sharpe 差距），双条件两侧同收紧不改变结论
  ——但按纲领要求，两对决脚本需同口径重跑一次登记（EXP 产物待补）。

### A5 — 特征注册表：真实特征清单 + 显式零填充（2026-08-18）

- **问题**: 200 维 s_t 的 6 个特征组均用 wh&#8203;il&#8203;e 循环反复追加同一表达式
  补齐到固定宽度（K3 纲领 `do&#8203;cs/K3_GU&#8203;ID&#8203;AN&#8203;CE_2026-08-18.md` A5）。
  200 维中 70 维为纯复制列，另有 8 处跨组/组内同值重复（G1 累计收益与
  多尺度波动 6 列、G2 体量秩 w=1、G3 裸 vo&#8203;l_20、G4 裸 sp&#8203;re&#8203;ad、G5 两处
  秩重复、G6 动量三列）——有效维度仅约 128，虚假放大共线性；
  "6 组 45/35/40/20/30/30"的声明与实际不符，特征组消融（g2/g3 负贡献）
  的归因解释力受损。
- **修复**: `fe&#8203;at&#8203;ur&#8203;es/re&#8203;gi&#8203;me_fe&#8203;at&#8203;ur&#8203;es.py` 改为命名注册表——每个特征
  具名（`fe&#8203;at&#8203;ur&#8203;e_na&#8203;me&#8203;s` 200 项两两不同），新增 `_pa&#8203;d_an&#8203;d_st&#8203;ac&#8203;k`
  （组内同名去重 + 不足槽位零填充并命名 g{k}_pa&#8203;d_XX）与
  `re&#8203;al_fe&#8203;at&#8203;ur&#8203;e_ma&#8203;sk`（Tr&#8203;ue=真实特征）/ `n_re&#8203;al_fe&#8203;at&#8203;ur&#8203;es` /
  `n_pa&#8203;dd&#8203;in&#8203;g` / `gr&#8203;ou&#8203;p_re&#8203;al_co&#8203;un&#8203;ts` 诊断出口。日频真实特征 115 个
  （g1:27 / g2:20 / g3:20 / g4:14 / g5:15 / g6:19）+ 85 个显式零填充列；
  全部 wh&#8203;il&#8203;e 复制列与 8 处同值重复列删除；顺手修复 G4 脆弱索引
  `fe&#8203;at&#8203;s[-2]`。周线 `lo&#8203;ok&#8203;ba&#8203;ck_sc&#8203;al&#8203;e=0.2` 窗口塌缩产生的同名列按
  注册表自动去重（首现保留）。下游 200 维契约（输出形状与组槽位宽度）
  不变。
- **守卫**: `te&#8203;st&#8203;s/te&#8203;st_fe&#8203;at&#8203;ur&#8203;es.py::Te&#8203;st&#8203;A5Fe&#8203;at&#8203;ur&#8203;eR&#8203;e&#8203;gi&#8203;st&#8203;ry`（9 项：
  wh&#8203;il&#8203;e 填充回归守卫/注册表完整性/名称唯一性/日频计数精确断言/
  填充列恒零/填充显式命名/真实列信息量/注册表幂等/周线去重）+
  `se&#8203;lf_ch&#8203;ec&#8203;k.py` 5.7 节 4 项断言（wh&#8203;il&#8203;e 清零/注册表齐备/填充命名/
  历史重复表达式清除）。
- **影响面**: 纯特征层重构——删除的重复列不携带增量信息，全部真实特征
  表达式逐字保留（gi&#8203;t di&#8203;ff 全量核对）。填充列恒 0，不再放大共线性；
  ro&#8203;ut&#8203;er/专家的输入分布改变（原复制列消失），历史训练产物与修复后
  数值不可直接比较。按纲领要求：特征组消融（g2/g3 负贡献结论）需在
  去重后同口径重跑复核（EX&#8203;P 产物待补，可与 A4 的对决重跑合并执行）。


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
