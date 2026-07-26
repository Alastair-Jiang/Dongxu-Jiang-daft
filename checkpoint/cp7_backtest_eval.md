# CP7: 回测引擎与基准评估

> **状态**: pending
> **依赖**: [CP6 硬化统计收集](cp6_hardening.md)
> **预计工作量**: 6–8 天
> **后续**: [CP8 组合优化](cp8_portfolio.md)

---

## 目标

实现完整的向量化回测引擎，计算所有标准量化指标。
复现 README 中声称的基准对比（DAFT vs Time-MoE, PatchTST, iTransformer, Super-Linear），
用数据验证 DAFT 是否真的有优势。

## 前置依赖

- CP6：硬化缓存就绪
- CP5：联合微调模型可用

---

## 任务清单

### 7.1 BacktestEngine 实现

当前全部 `NotImplementedError`：

- [ ] `run(signals, prices, mask) → metrics` 主流程
- [ ] 仓位计算：信号 → 目标仓位 (long/flat/short)
- [ ] 交易成本模型：固定费率 (0.0003 A 股) + 滑点 (线性于交易量)
- [ ] 换手率约束：单次调仓不超过 20%
- [ ] 日度/分钟级 rebalance 频率支持
- [ ] 输出 metrics：Sharpe, Max Drawdown, Calmar, IC (Rank), ICIR, Hit Rate, Win/Loss Ratio, Annual Return, Annual Vol

**文件**: `src/daft/backtest/engine.py`

### 7.2 性能指标实现

- [ ] `sharpe_ratio(returns)` — 年化 (已有静态方法骨架)
- [ ] `max_drawdown(cumulative_returns)` — 峰值回撤 (已有静态方法骨架)
- [ ] `calmar_ratio` — 年化收益 / MaxDD
- [ ] `info_coefficient(predictions, targets, mask)` — Rank IC
- [ ] `icir` — IC 均值 / IC 标准差
- [ ] `hit_rate` — 方向正确率
- [ ] `turnover` — 平均换手率
- [ ] `win_loss_ratio` — 盈亏比

**文件**: `src/daft/backtest/metrics.py`

### 7.3 Walk-Forward 交叉验证

- [ ] Expanding window：train 窗口不断扩大，test 窗口向前滚动
- [ ] 5 折 walk-forward (configs/paper.yaml: `n_splits: 5`)
- [ ] 每折独立训练 + 测试，汇总指标均值/标准差

**文件**: `src/daft/backtest/walk_forward.py`

### 7.4 基准对比

- [ ] 实现或调用以下基准模型的推理接口：
  - **PatchTST** (ICLR 2023)：patch-based time series Transformer
  - **iTransformer** (ICLR 2024)：inverted Transformer 架构
  - **TimesNet** (ICLR 2023)：多周期分解
- [ ] **Time-MoE** 和 **Super-Linear** 如果没有开源权重，使用原作者报告的指标
- [ ] 统一评估协议：相同训练/测试切分、相同交易成本、相同指标集
- [ ] 生成对比表格（Markdown + LaTeX）

**文件**: `scripts/run_benchmarks.py`

### 7.5 结果可视化

- [ ] 累计收益曲线 (DAFT vs baselines)
- [ ] 滚动 Sharpe (12 个月窗口)
- [ ] Drawdown 水下图
- [ ] 月度收益热力图
- [ ] 各指标雷达图 (DAFT vs best baseline)

**文件**: `scripts/plot_results.py`

---

## 验收标准

| # | 标准 | 验证 |
|---|------|------|
| 1 | BacktestEngine 在 synthetic 信号上指标计算正确（对照手工计算） | 单元测试 |
| 2 | Walk-forward 5 折跑通，每折指标在合理范围 | 集成测试 |
| 3 | 交易成本模型正确：总扣费 = 固定费率 × 成交额 + 滑点 × trade_size² | 单元测试 |
| 4 | DAFT 在至少一个指标上优于至少一个 baseline | 基准对比表 |
| 5 | 所有图表可复现生成 | `make plots` |

---

## 快速验证

### 7.1 BacktestEngine——已知结果验证

```python
from daft.backtest.engine import BacktestEngine
from daft.backtest.metrics import sharpe_ratio, max_drawdown, calmar_ratio, hit_rate
import torch

engine = BacktestEngine(config={"annualization": 252, "tx_cost_bps": 3.0, "slippage_bps": 1.0})

# 构造已知结果的数据
torch.manual_seed(42)
T, N = 500, 50
# 完美预测: signal = 真实收益率
true_returns = torch.randn(T, N) * 0.02
signals = true_returns + torch.randn(T, N) * 0.001  # 加微量噪声
prices = (1 + true_returns).cumprod(dim=0)
mask = torch.ones(T, N, dtype=torch.bool)

metrics = engine.run(signals, prices, mask)
assert metrics["sharpe"] > 1.0, f"完美预测 Sharpe 应很高, 实际: {metrics['sharpe']:.2f}"
assert metrics["hit_rate"] > 0.7, f"方向正确率应很高, 实际: {metrics['hit_rate']:.2f}"
print(f"✅ 完美预测: Sharpe={metrics['sharpe']:.2f}, HitRate={metrics['hit_rate']:.1%}, MaxDD={metrics['max_drawdown']:.2%}")
```

### 7.2 交易成本验证

```python
# 信号频繁切换 → 成本应显著
noisy_signals = torch.randn(T, N)  # 纯随机信号
metrics_noisy = engine.run(noisy_signals, prices, mask)
# 纯随机信号扣除成本后 Sharpe 应为负
print(f"✅ 交易成本: 随机信号 Sharpe={metrics_noisy['sharpe']:.2f} (应 < 0)")
assert metrics_noisy["sharpe"] < 0
```

### 7.3 Static methods——对照 NumPy

```python
import numpy as np

returns = torch.randn(1000) * 0.02

# Sharpe: sqrt(T) * mean / std
sr = BacktestEngine.sharpe_ratio(returns, annualization=252)
np_returns = returns.numpy()
np_sr = np.sqrt(252) * np_returns.mean() / np_returns.std()
assert abs(sr - np_sr) < 1e-4, f"Sharpe: torch={sr:.4f} vs numpy={np_sr:.4f}"
print(f"✅ Sharpe: {sr:.4f} (vs numpy {np_sr:.4f})")

# Max Drawdown
# 构造有已知最大回撤的序列
cum = torch.tensor([1.0, 1.2, 1.1, 0.9, 0.8, 1.0, 1.3, 1.1, 0.7, 1.2])
mdd = BacktestEngine.max_drawdown(cum)
expected_mdd = (0.7 - 1.3) / 1.3  # peak=1.3, trough=0.7
assert abs(mdd - expected_mdd) < 0.01, f"MaxDD: {mdd:.4f} vs expected {expected_mdd:.4f}"
print(f"✅ Max Drawdown: {mdd:.4f} (expected {expected_mdd:.4f})")
```

### 7.4 Walk-Forward——2 折冒烟

```python
from daft.backtest.walk_forward import WalkForwardCV

wf = WalkForwardCV(n_splits=2, train_size=0.7)
# 2 折划分
for fold, (train_range, test_range) in enumerate(wf.split(total_days=200)):
    assert len(train_range) > 50
    assert len(test_range) > 20
    print(f"   Fold {fold}: train={train_range[0]}:{train_range[-1]}, test={test_range[0]}:{test_range[-1]}")
print("✅ Walk-Forward 划分正确")
```

### 7.5 基准对比——接口验证

```python
from scripts.run_benchmarks import benchmark_model

# 确保基准模型 wrapper 接口一致
dummy_model = lambda x: torch.randn(x.shape[0], x.shape[1]) * 0.01
result = benchmark_model(dummy_model, n_stocks=10, n_days=100)
assert "sharpe" in result and "ic" in result
print(f"✅ Benchmark wrapper: { {k: f'{v:.3f}' for k,v in result.items()} }")
```

### 一键验证

```bash
python -m pytest tests/test_backtest.py tests/test_metrics.py -v --tb=short
python scripts/run_benchmarks.py --config configs/small.yaml --synthetic
```

---

## 风险点

| 风险 | 影响 | 缓解 |
|------|------|------|
| DAFT 在实测中不优于 baseline | 论文声称失信 | 诚实报告结果，分析原因 |
| 基准模型 API 不兼容 | 对比无法进行 | 实现统一 wrapper 层 |
| 回测结果对超参数敏感 | 指标不可信 | Walk-forward + 多 seed 取均值 |
