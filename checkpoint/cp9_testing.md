# CP9: 测试体系

> **状态**: pending
> **依赖**: [CP3](cp3_expert_training.md)–[CP8](cp8_portfolio.md)（与各 CP 并行推进更佳）
> **预计工作量**: 5–7 天
> **后续**: [CP10 文档与 Notebook](cp10_docs_notebooks.md)

---

## 目标

建立完整的三层测试体系（单元/集成/E2E），整体覆盖率达到 80%+。
当前仓库 **零测试**，此 CP 是项目从原型到可靠软件的关键一步。

## 前置依赖

实际上本 CP 应与 CP1–CP8 **并行推进**（每个模块写完立即写测试）。
此处集中列出是为了 checkpoint 完整性。每个前序 CP 的任务清单中已包含基本测试，
本 CP 补充跨模块集成测试、边界条件、stress test。

---

## 任务清单

### 9.1 模型单元测试

- [ ] `test_router.py`:
  - 路由分布 shape: `(B, n_experts)`, topk 的确 sum=1
  - 不同 temperature 下分布熵变化 (temp=0.1 更离散，temp=10.0 更均匀)
  - Quantile Balancing: 模拟极端偏爱专家 0 的场景，验证 bias 修正方向
  - `train` / `val` / `inference` 三种 mode 行为差异

- [ ] `test_memory.py`:
  - 初始 state 全零
  - 单步 forward 后 M 非零
  - forget gate α ∈ (0, 1)
  - 退火测试：输入 constant=0，10 步后记忆矩阵是否趋近于零
  - `reset_state()` 后 M 重新为零
  - `detach_state()` 后 `M.requires_grad = False`
  - 新：`set_external_gate()` 后遗忘门额外调制生效

- [ ] `test_cross_dim_attn.py`:
  - joint space 维度正确 (64)
  - 路由修正后仍为合法概率分布
  - depth_weights sum to 1
  - memory_gate ∈ (0, 1)
  - 乘法融合 vs 加法融合：构造全零输入验证乘法时 joint→0

- [ ] `test_hardening.py`:
  - pattern 计数正确递增
  - 到达 threshold 后 cache entry 被创建
  - 相同 pattern 命中 fast path
  - 不同 pattern 不命中
  - 熵检测触发 regime_shift → degrade
  - eviction 逻辑

- [ ] `test_experts.py`:
  - 四个专家的 `forward()` 输出 ∈ [-1, 1]
  - 各专家 Loss 函数数值范围合理
  - 梯度向后传播不报错

- [ ] `test_ensemble.py`:
  - 端到端 forward 不报错
  - 输出 `signal` ∈ [-1, 1]
  - 训练/推理 mode 输出差异有预期行为
  - 新：逐样本 hardening 逻辑 — 构造不同 regime 的样本验证分组正确

**文件**: `tests/test_*.py` (共 ~10 个文件)

### 9.2 数据与特征测试

- [ ] `test_panel.py`: Panel 构造、索引、mask 正确
- [ ] `test_sources.py`: SyntheticSource 确定性输出 (固定 seed)
- [ ] `test_preprocessing.py`: Winsorizer 截尾、停牌 mask、归一化
- [ ] `test_tensor_factors.py`: 每个因子原语 vs NumPy 参考，tol=1e-5
- [ ] `test_regime_features.py`: s_t shape = (T, N, 200)，无 NaN/Inf

### 9.3 集成测试

- [ ] `test_training_pipeline.py`: Stage 1→2→3 在 synthetic 数据上跑通，不报错
- [ ] `test_backtest_integration.py`: 完整 pipeline data→train→backtest 一条龙
- [ ] `test_checkpoint_roundtrip.py`: save→load→输出一致

### 9.4 覆盖率

- [ ] 配置 `pytest-cov` 或 `c8`
- [ ] 目标：整体行覆盖率 ≥ 80%，核心模型 ≥ 90%
- [ ] CI 集成：`node tests/run-all.js` (或 Python 等效: `pytest`)

---

## 验收标准

| # | 标准 | 验证 |
|---|------|------|
| 1 | `pytest` 全部通过，零 failure | CI |
| 2 | 行覆盖率 ≥ 80% | coverage report |
| 3 | 每个公开 API 方法至少有一个 happy-path 测试 | 人工 review |
| 4 | 边界条件覆盖率：空 batch、单样本 batch、NaN 输入、极端值输入 | 测试报告 |

---

## 快速验证

### 9.1 冒烟——全模块导入

```python
# 验证所有模块可导入，无 import 错误
from daft.models import RegimeRouter, KDAMarketMemory, CrossDimensionAttention, HardeningEngine, ExpertEnsemble
from daft.models.experts import TrendExpert, ReversalExpert, VolatilityExpert, EventExpert, BaseExpert
from daft.data import Panel, load_data_config, make_dataloader
from daft.data.sources import SyntheticSource
from daft.data.preprocessing import Winsorizer, SuspensionHandler, Normalizer
from daft.features import TensorFactorEngine, RegimeFeatureExtractor, FreqFeatureExtractor
from daft.training import ExpertTrainer, RouterTrainer, JointTrainer
from daft.backtest import BacktestEngine
from daft.portfolio import MarkowitzOptimizer
print("✅ 所有模块 import 通过")
```

### 9.2 快速覆盖率检查

```bash
# 冒烟测试 + 覆盖率
pytest tests/ --cov=src/daft --cov-report=term-missing --tb=short -q

# 目标输出:
# Name                                          Stmts   Miss  Cover
# -----------------------------------------------------------------
# src/daft/models/router.py                        90      5    94%
# src/daft/models/memory.py                       110      8    93%
# src/daft/models/cross_dim_attn.py               105      6    94%
# src/daft/models/hardening.py                    130     15    88%
# src/daft/models/ensemble.py                     115     10    91%
# src/daft/models/experts/*.py                     94     10    89%
# src/daft/data/*.py                                -      -     -
# src/daft/features/*.py                            -      -     -
# src/daft/training/*.py                            -      -     -
# src/daft/backtest/*.py                            -      -     -
# src/daft/portfolio/*.py                           -      -     -
# -----------------------------------------------------------------
# TOTAL                                          1500+     80+   80%+
```

### 9.3 边界条件测试——必须全部通过

```python
# 在 pytest 中运行的边界测试案例
def test_empty_batch():
    """空 batch 输入应优雅报错而非崩溃"""
    router = RegimeRouter()
    with pytest.raises(ValueError, match="batch_size"):
        router(torch.empty(0, 200))

def test_single_sample_batch():
    """单样本 batch 不应崩溃"""
    router = RegimeRouter()
    probs, indices, z, full = router(torch.randn(1, 200))
    assert probs.shape == (1, 3)
    assert indices.shape == (1, 3)

def test_nan_input():
    """NaN 输入应被检测并报错"""
    memory = KDAMarketMemory()
    memory.reset_state(2, torch.device("cpu"))
    with pytest.raises(ValueError):
        memory(torch.tensor([[float('nan')]*200, [1.0]*200]))

def test_inf_input():
    """Inf 输入应被检测"""
    memory = KDAMarketMemory()
    memory.reset_state(2, torch.device("cpu"))
    with pytest.raises(ValueError):
        memory(torch.tensor([[float('inf')]*200, [1.0]*200]))

print("✅ 边界条件: empty/single/NaN/Inf 全部正确处理")
```

### 9.4 CI 集成检查

```yaml
# .github/workflows/test.yml 中应包含:
# - pytest --cov + coverage report
# - 全绿才算通过
```

本地模拟 CI：

```bash
# 完整测试套件
pytest tests/ -v --cov=src/daft --cov-report=html --cov-fail-under=80 --tb=long 2>&1 | tail -20

# 预期: FAILED 数量为 0, Coverage >= 80%
```

### 一键验证

```bash
# 完整测试 + 覆盖率
pytest tests/ -v --cov=src/daft --cov-report=term --cov-fail-under=80 --tb=short

# 所有测试必须全绿
# 输出末尾应显示: Required test coverage of 80% reached.
```

---

## 风险点

| 风险 | 影响 | 缓解 |
|------|------|------|
| 写测试时发现模块 bug 需返工 | 延迟 | 本 CP 与 CP1–CP8 并行推进 |
| 随机性测试 (softmax, noise) 不稳定 | flaky test | 固定 seed + 宽松 tolerance |
| 测试数据太大导致 CI 慢 | 反馈延迟 | 单元测试用 small config (synthetic, 短序列) |
