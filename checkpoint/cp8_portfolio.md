# CP8: 组合优化

> **状态**: pending
> **依赖**: [CP7 回测引擎](cp7_backtest_eval.md)
> **预计工作量**: 4–5 天
> **后续**: [CP9 测试体系](cp9_testing.md)

---

## 目标

实现 Ledoit-Wolf 收缩 + 马科维兹均值-方差组合优化，
将模型预测的信号（期望收益）转换为实际持仓权重。
支持多约束：非负权重、最大单票权重、行业集中度上限。

## 前置依赖

- CP7：回测引擎可跑，能产出 cross-sectional 预测信号
- CP2：截面特征（协方差矩阵所需的历史收益率）

---

## 任务清单

### 8.1 Ledoit-Wolf 协方差收缩

- [ ] 实现 Ledoit-Wolf (2004) 收缩估计器
- [ ] 收缩目标：constant correlation model
- [ ] 收缩强度自动计算（闭式解，无需超参调优）
- [ ] 验证：在 N=500 >> T=200 的高维设定下，收缩后方差优于 sample covariance

**文件**: `src/daft/portfolio/covariance.py`

### 8.2 MarkowitzOptimizer 实现

当前全空：

- [ ] `optimize(expected_returns, covariance, mask) → weights`
- [ ] 目标函数: `max_w  wᵀμ - γ wᵀΣ_shrunk w`
- [ ] 约束: `w ≥ 0` (A 股不可卖空), `Σw = 1`, `w_i ≤ max_weight`
- [ ] 可选行业约束: `Σ_{i∈sector_k} w_i ≤ sector_max`
- [ ] 默认用 CVXPY (pip 安装轻量), 可选 MOSEK (更快但需 license)
- [ ] Fallback: 如 CVXPY 不可用，用简单等权或信号加权作为退化方案

**文件**: `src/daft/portfolio/markowitz.py`

### 8.3 交易执行模拟

- [ ] 过去权重 → 目标权重 → 实际执行权重（考虑交易成本后的净调仓）
- [ ] Turnover penalty: 权重变化 < 0.5% 的票不调仓（节省交易成本）
- [ ] 调仓延迟：T 时刻信号 → T+1 时刻开盘执行（避免 look-ahead bias）

**文件**: `src/daft/portfolio/executor.py`

### 8.4 组合回测集成

- [ ] 将组合优化插入 BacktestEngine pipeline
- [ ] 对比组合优化 vs 纯信号加权 (signal → weight 直接归一化)
- [ ] 监控组合层面的 risk metrics：CVaR, diversification ratio, effective N

---

## 验收标准

| # | 标准 | 验证 |
|---|------|------|
| 1 | 收缩后协方差矩阵条件数低于样本协方差 | 单元测试 |
| 2 | 优化器对 500 只股票在 5 秒内求解 | 性能测试 |
| 3 | 最优权重满足所有约束 | 单测 + 边界检查 |
| 4 | 组合优化后 Sharpe 不低于等权 baseline | 回测对比 |
| 5 | CVXPY 不可用时能 fallback 到信号加权 | 环境测试 |

---

## 快速验证

### 8.1 Ledoit-Wolf 收缩——数值验证

```python
from daft.portfolio.covariance import ledoit_wolf_shrinkage
import torch

# 高维设定: N=100, T=50 (N >> T), 样本协方差矩阵奇异
torch.manual_seed(42)
returns = torch.randn(50, 100) * 0.02  # T=50, N=100

sample_cov = torch.cov(returns.T)
shrunk_cov = ledoit_wolf_shrinkage(returns)

# 收缩后条件数应改善
sample_cond = torch.linalg.cond(sample_cov)
shrunk_cond = torch.linalg.cond(shrunk_cov)
assert shrunk_cond < sample_cond, f"收缩未改善条件数: {sample_cond:.1f} → {shrunk_cond:.1f}"
print(f"✅ Ledoit-Wolf: cond {sample_cond:.0f} → {shrunk_cond:.0f} (改善 {sample_cond/shrunk_cond:.1f}x)")

# 收缩后协方差仍为正定
eigvals = torch.linalg.eigvalsh(shrunk_cov)
assert (eigvals > 0).all(), f"收缩后协方差非正定, min(eigval)={eigvals.min():.6f}"
print(f"   正定性: ✓, min(eigval)={eigvals.min():.6f}")
```

### 8.2 MarkowitzOptimizer——小规模验证

```python
from daft.portfolio.markowitz import MarkowitzOptimizer
import torch

# 小规模: 5 只股票，方便手工验证
N = 5
mu = torch.tensor([0.10, 0.05, 0.03, 0.01, -0.02])  # 期望收益
# 构造简单的对角协方差
cov = torch.diag(torch.tensor([0.04, 0.03, 0.02, 0.01, 0.05]))
mask = torch.ones(N, dtype=torch.bool)

opt = MarkowitzOptimizer(risk_aversion=1.0, max_weight=0.4)
weights = opt.optimize(mu, cov, mask)

assert weights.shape == (N,)
assert abs(weights.sum() - 1.0) < 1e-4, f"weights 未归一化: sum={weights.sum():.4f}"
assert (weights >= 0).all(), f"有负权重: {weights}"
assert (weights <= 0.4 + 1e-4).all(), f"超过 max_weight: max={weights.max():.4f}"

# 高收益资产应有更高权重
assert weights[0] > weights[-1], f"高收益资产权重应更高: w0={weights[0]:.3f}, w4={weights[4]:.3f}"
print(f"✅ Markowitz: weights={[f'{w:.2%}' for w in weights.tolist()]}, sum={weights.sum():.4f}")
```

### 8.3 大尺度性能测试

```python
# 500 只股票，验证不超时
N = 500
mu = torch.randn(N) * 0.05
cov = ledoit_wolf_shrinkage(torch.randn(200, N) * 0.02)

import time
t0 = time.perf_counter()
weights = opt.optimize(mu, cov, torch.ones(N, dtype=torch.bool))
elapsed = time.perf_counter() - t0

assert elapsed < 10, f"500 只股票优化超时: {elapsed:.1f}s"
print(f"✅ 500 只股票: {elapsed:.1f}s (目标 <10s), max_weight={weights.max():.3f}")
```

### 8.4 Fallback 机制

```python
# 模拟 CVXPY 不可用 → 应 fallback 到信号加权
import unittest.mock
with unittest.mock.patch.dict('sys.modules', {'cvxpy': None}):
    opt_fallback = MarkowitzOptimizer(risk_aversion=1.0)
    w = opt_fallback.optimize(mu[:5], cov[:5,:5], mask)
    assert w.sum() > 0.99
    print(f"✅ CVXPY fallback: weights={[f'{v:.2%}' for v in w.tolist()]}")
```

### 一键验证

```bash
python -m pytest tests/test_covariance.py tests/test_markowitz.py -v --tb=short
```

---

## 风险点

| 风险 | 影响 | 缓解 |
|------|------|------|
| CVXPY 优化 500 维问题在 CPU 上太慢 | 无法实时调仓 | 用 MOSEK 或缩减 universe 到 top-50 |
| 预测协方差与真实协方差偏差大 | 组合风险失控 | 保守调参 (γ 偏大) + 滚动窗口估计 |
| 组合优化引入额外的 look-ahead bias | 指标虚高 | 严格使用 historical data only |
