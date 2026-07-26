# CP2: 特征工程实现

> **状态**: pending
> **依赖**: [CP1 数据管道](cp1_data_pipeline.md)
> **预计工作量**: 8–10 天
> **后续**: [CP3 专家训练](cp3_expert_training.md)

---

## 目标

实现完整的 213 因子计算引擎、regime 特征提取器、FFT 频谱特征提取器，
将原始 OHLCV 数据转换为 200 维市场状态向量 `s_t`，并产出三层深度表示
`[h0 (raw), h1 (base), h2 (composite)]` 供 CDAP 消费。

## 前置依赖

- CP1：Panel 数据结构 + mask 机制 + SyntheticSource 可用于冒烟测试

---

## 任务清单

### 2.1 TensorFactorEngine — GPU 向量化因子计算

- [ ] `rank(x, mask)` — 截面排序，输出 [0, 1] 百分位
- [ ] `corr(x, y, window, mask)` — 滚动 Pearson 相关
- [ ] `ewma(x, span, mask)` — 指数加权移动平均
- [ ] `ts_delta(x, d, mask)` — lag-d 差分
- [ ] `ts_sum(x, d, mask)` — 滚动求和
- [ ] `ts_std(x, d, mask)` — 滚动标准差
- [ ] `ts_max(x, d, mask)` / `ts_min(x, d, mask)` — 滚动极值
- [ ] `ts_argmax(x, d, mask)` / `ts_argmin(x, d, mask)` — 极值位置
- [ ] 所有操作 mask-aware：被 mask 的值不出现在计算窗口内
- [ ] 单元测试：对照 NumPy 参考实现验证正确性，容忍度 1e-5

**文件**: `src/daft/features/tensor_factors.py`

### 2.2 Legacy Factors — 204 个手工因子

- [ ] 从 ml-quant-trading 导入 `better_001–028` (28 个)：VWAP 偏离 + 量加权动量
- [ ] 从 ml-quant-trading 导入 `best_001–021` (21 个)：收盘价动量变体
- [ ] 从 ml-quant-trading 导入 `old_027–076` (50 个)：经典 Alpha 信号
- [ ] 从 ml-quant-trading 导入 `stock_001–022` (22 个)：个股衍生序列
- [ ] 从 ml-quant-trading 导入 `extra_001–014` (14 个)：换手率 + 成交额
- [ ] 从 ml-quant-trading 导入 `add_001–030` (30 个)：复合因子
- [ ] 从 ml-quant-trading 导入 `change_001–005` (5 个)：短窗变化率
- [ ] 从 ml-quant-trading 导入 `original_001–028` (28 个)：量价直接统计
- [ ] 从 ml-quant-trading 导入 `cs_rank_*` (6 个)：市场宽度指标
- [ ] 所有因子统一为 `factor(panel) → (T, N)` 签名

**文件**: `src/daft/features/legacy_factors.py`

### 2.3 RegimeFeatureExtractor — 200 维 s_t 构建

- [ ] 价格动态特征 (~40 维)：多周期收益率、价格与 MA 偏离
- [ ] 成交量特征 (~20 维)：量比、换手率异常、OBV
- [ ] 波动率结构 (~30 维)：多周期滚动波动率、波动率的波动率
- [ ] 微观结构 (~20 维)：买卖价差代理、价格冲击代理、Amihud 非流动性
- [ ] 截面特征 (~30 维)：截面排名、离散度
- [ ] 动量/因子暴露 (~40 维)：短/中/长期动量、多因子暴露
- [ ] 剩余 20 维：FFT 频谱特征拼接
- [ ] 输出 `shape (T, N, 200)` 的 `s_t`
- [ ] 冒烟测试：synthetic 数据上跑通，验证 shape 和数值范围

**文件**: `src/daft/features/regime_features.py`

### 2.4 FreqFeatureExtractor — FFT 频谱特征

- [ ] `compute_periodogram(x)` — 去 DC → FFT → PSD → 归一化 (已实现骨架)
- [ ] `forward(panel)` — 对所有资产计算 FFT 特征
- [ ] 频段能量比例（低频/中频/高频），作为 regime 检测的辅助信号
- [ ] 低频功率占比 → regime 判别（趋势市低频占优，震荡市中频占优）

**文件**: `src/daft/features/freq_features.py`

### 2.5 三层深度表示 [h0, h1, h2]

- [ ] L0 (Raw): 原始价格/量信息投影 → 64 维
- [ ] L1 (Base): 技术指标/因子投影 → 64 维
- [ ] L2 (Composite): regime 标签/风险指标等高级特征投影 → 64 维
- [ ] 三层均映射到 64 维以匹配 `CrossDimensionAttention.d_v`

**文件**: `src/daft/features/depth_layers.py`

---

## 验收标准

| # | 标准 | 验证 |
|---|------|------|
| 1 | 所有 204 个因子在 synthetic 数据上输出 shape 正确 | 单元测试 |
| 2 | mask-aware 操作：被 mask 位置的因子值 = NaN/zero，不影响其他位置 | 单元测试 |
| 3 | `s_t` shape = `(T, N, 200)`，所有值有限（无 NaN/Inf） | 集成测试 |
| 4 | FFT periodogram 归一化到 sum=1 | 单元测试 |
| 5 | `[h0, h1, h2]` 每个 shape = `(T, N, 64)` | 单元测试 |

---

## 快速验证

### 2.1 TensorFactorEngine

```python
from daft.features.tensor_factors import TensorFactorEngine
import torch

engine = TensorFactorEngine()
# 构造简单数据: T=50, N=10
x = torch.randn(50, 10)
mask = torch.ones(50, 10, dtype=torch.bool)
mask[20:30, 3] = False  # 部分 mask

r = engine.rank(x, mask)
assert r.shape == (50, 10), f"rank shape: {r.shape}"
assert (r >= 0).all() and (r <= 1).all(), "rank 应在 [0,1]"
print("✅ rank 通过")

c = engine.corr(x, x.flip(-1), window=10, mask=mask)
assert c.shape == (50, 10)
assert (c >= -1).all() and (c <= 1).all(), f"corr 范围不对: min={c.min():.2f} max={c.max():.2f}"
print("✅ corr 通过")

e = engine.ewma(x, span=5, mask=mask)
assert e.shape == (50, 10)
print("✅ ewma 通过")
```

### 2.2 Legacy Factors（抽检 3 个）

```python
from daft.features.legacy_factors import better_001, best_001, cs_rank_pct
from daft.data.sources import SyntheticSource

panel = SyntheticSource(n_assets=30, n_days=200, seed=42).load()
f1 = better_001(panel)   # VWAP 偏离
f2 = best_001(panel)     # 收盘动量
f3 = cs_rank_pct(panel)  # 截面排名

assert f1.shape == (200, 30), f"better_001 shape: {f1.shape}"
assert f2.shape == (200, 30)
assert f3.shape == (200, 30)
assert not torch.isnan(f1).any(), "有 NaN"
assert not torch.isinf(f1).any(), "有 Inf"
print(f"✅ 因子抽检: better_001 range=[{f1.min():.3f}, {f1.max():.3f}], "
      f"best_001 range=[{f2.min():.3f}, {f2.max():.3f}]")
```

### 2.3 RegimeFeatureExtractor (s_t)

```python
from daft.features.regime_features import RegimeFeatureExtractor
from daft.data.sources import SyntheticSource

panel = SyntheticSource(n_assets=10, n_days=100, seed=42).load()
extractor = RegimeFeatureExtractor(n_base_factors=50)
s_t = extractor(panel)

assert s_t.shape == (100, 10, 200), f"s_t shape: {s_t.shape}"
assert s_t.std() > 0, "s_t 方差为 0，可能未计算"
assert not torch.isnan(s_t).any()
assert not torch.isinf(s_t).any()
print(f"✅ s_t: shape={s_t.shape}, mean={s_t.mean():.4f}, std={s_t.std():.4f}")
```

### 2.4 FreqFeatureExtractor

```python
from daft.features.freq_features import FreqFeatureExtractor
import torch

extractor = FreqFeatureExtractor(lookback=128, n_freq_bins=32)
# 模拟一段价格序列
x = torch.randn(10, 128)  # 10 assets × 128 bars
psd = extractor.compute_periodogram(x)
assert psd.shape == (10, 32), f"PSD shape: {psd.shape}"
assert abs(psd.sum(-1) - 1.0).max() < 1e-4, "PSD 未归一化"
print(f"✅ FFT PSD: shape={psd.shape}, sum≈1")
```

### 2.5 三层深度表示

```python
from daft.features.depth_layers import build_depth_layers
from daft.data.sources import SyntheticSource

panel = SyntheticSource(n_assets=10, n_days=100, seed=42).load()
h0, h1, h2 = build_depth_layers(panel, d_v=64)

assert h0.shape == (100, 10, 64)
assert h1.shape == (100, 10, 64)
assert h2.shape == (100, 10, 64)
print(f"✅ 三层深度: h0.std={h0.std():.3f}, h1.std={h1.std():.3f}, h2.std={h2.std():.3f}")
```

### 一键验证

```bash
python -m pytest tests/test_features.py -v --tb=short
```

---

## 风险点

| 风险 | 影响 | 缓解 |
|------|------|------|
| ml-quant-trading 因子定义依赖原始仓库内部 API | 导入失败 | 自实现核心因子，仅参考公式 |
| 200 维特征共线性严重 | 模型过拟合 | 后续在 CP3 加 PCA/特征选择 |
| FFT 在大 batch 上内存爆炸 | OOM | 分批计算 FFT，复用中间结果 |
