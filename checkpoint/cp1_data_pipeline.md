# CP1: 数据管道基础

> **状态**: pending
> **依赖**: 无
> **预计工作量**: 5–7 天
> **后续**: [CP2 特征工程](cp2_feature_engine.md)

---

## 目标

建立可扩展的金融数据管道，支持多数据源输入、预处理、可交易性 mask 生成，
输出标准化的 Panel 数据结构供所有下游模块消费。

## 前置依赖

无。本 CP 是整个项目的起点。

---

## 任务清单

### 1.1 Panel 核心数据结构

- [ ] 定义 `Panel` 类：`(T, N, F)` 三维张量 — 时间步 × 资产数 × 特征数
- [ ] 支持多频率数据 (`1min`, `5min`, `15min`, `30min`, `1h`, `1d`)
- [ ] `dates` 时间轴索引、`assets` 资产代码列表
- [ ] `mask` 可交易性布尔矩阵 (limit-up/down、停牌、涨跌停)
- [ ] 索引/切片操作：按时间、按资产、按特征维度
- [ ] `to_torch()` 方法导出 PyTorch tensor

**文件**: `src/daft/data/panel.py`

### 1.2 数据源适配器

- [ ] `DataSource` 抽象基类，定义 `load(start, end, assets) → Panel` 接口
- [ ] `BaostockSource` — 中国 A 股 (baostock)，支持 OHLCV + 复权
- [ ] `YFinanceSource` — 美股/港股 (yfinance)，支持 OHLCV + 股息调整
- [ ] `SyntheticSource` — 合成数据生成器（几何布朗运动 + regime 切换），用于冒烟测试
- [ ] `CSVSource` — 通用 CSV 加载器，支持自定义列映射

**文件**: `src/daft/data/sources/`

### 1.3 数据预处理

- [ ] `Winsorizer`: 极端值截尾（1st / 99th 分位），可配置分位数
- [ ] `SuspensionHandler`: 停牌前向填充，最大填充步数可配置 (默认 5)
- [ ] `LimitUpDownHandler`: 涨跌停 mask 生成 — 涨跌停期间不可交易
- [ ] `Normalizer`: 跨资产截面标准化 (z-score) 或时序标准化

**文件**: `src/daft/data/preprocessing.py`

### 1.4 数据加载配置

- [ ] YAML → config dict 解析，匹配 `configs/paper.yaml` 和 `configs/small.yaml` 的 `data` 段
- [ ] `DataConfig` dataclass：source, n_stocks, n_years, frequency, preprocessing 开关
- [ ] `make_dataloader(config) → DataLoader` 工厂函数

**文件**: `src/daft/data/config.py`, `src/daft/data/__init__.py`

---

## 验收标准

| # | 标准 | 验证方式 |
|---|------|----------|
| 1 | SyntheticSource 可生成 shape `(500, 200, 10)` 的 Panel | 单元测试 |
| 2 | 涨跌停 mask 正确：涨跌停当天 mask=False | 单元测试 + 人工抽查 |
| 3 | 停牌前向填充：连续停牌超过 5 根 K 线后 mask=False | 单元测试 |
| 4 | BaostockSource 可拉取 000300 成分股 5 年日线数据 | 集成测试 |
| 5 | `to_torch()` 输出 shape 正确的 `(T, N, F)` tensor | 单元测试 |

---

## 涉及文件

```
src/daft/data/
├── __init__.py          # 公开 API
├── panel.py             # Panel 数据结构
├── config.py            # 配置解析
├── preprocessing.py     # Winsorizer, SuspensionHandler, LimitUpDownHandler, Normalizer
└── sources/
    ├── __init__.py
    ├── base.py          # DataSource 抽象基类
    ├── baostock.py      # A 股 baostock
    ├── yfinance.py      # 美股/港股 yfinance
    ├── synthetic.py     # 合成数据生成器
    └── csv_source.py    # CSV 加载器
```

---

## 快速验证

每个子任务完成后，逐条跑通以下验证。全部通过才算 CP1 完成。

### 1.1 Panel 数据结构

```python
# 验证: Panel 构造 + 索引 + to_torch
from daft.data import Panel
import torch

# 合成数据
dates = pd.date_range("2024-01-01", periods=100, freq="1d")
assets = [f"stock_{i:04d}" for i in range(50)]

panel = Panel.from_synthetic(dates=dates, assets=assets, n_features=10, seed=42)
assert panel.shape == (100, 50, 10), f"shape mismatch: {panel.shape}"

# 插入一些 mask=False 的样本模拟停牌
panel.mask[10:15, 3] = False

tensor, mask = panel.to_torch()
assert tensor.shape == (100, 50, 10)
assert mask.shape == (100, 50)
assert mask[10:15, 3].sum() == 0, "停牌样本 mask 应为 False"
print("✅ Panel 构造+索引+mask+to_torch 通过")
```

### 1.2 数据源

```python
# 验证: 3 种数据源均可加载并输出 Panel
from daft.data.sources import SyntheticSource, BaostockSource, YFinanceSource, CSVSource

# Synthetic (最快，冒烟用)
src = SyntheticSource(n_assets=30, n_days=200, seed=42)
panel = src.load()
assert panel.shape == (200, 30, 7)  # OHLCV + volume + amount
assert not panel.dates.empty
assert len(panel.assets) == 30
print("✅ SyntheticSource 通过")

# Baostock — 只拉 1 只股票 30 天日线做冒烟
# src = BaostockSource()
# panel = src.load(start="2024-01-01", end="2024-02-01", assets=["000001"])
# assert panel.shape[0] > 15  # 至少 15 个交易日
print("✅ BaostockSource 冒烟通过 (需手动取消注释验证)")

# YFinance — 同理
# src = YFinanceSource()
# panel = src.load(start="2024-01-01", end="2024-02-01", assets=["AAPL"])
# assert panel.shape[0] > 15
print("✅ YFinanceSource 冒烟通过 (需手动取消注释验证)")
```

### 1.3 预处理

```python
# 验证: 预处理管线
from daft.data.preprocessing import Winsorizer, SuspensionHandler, Normalizer
from daft.data.sources import SyntheticSource

panel = SyntheticSource(n_assets=50, n_days=300, seed=42).load()
panel.values[0:10, 0:5, 3] = 1e10  # 插入极端值

w = Winsorizer(q_lower=0.01, q_upper=0.99)
panel = w.fit_transform(panel)
assert not (panel.values > 1e9).any(), "极端值未被截尾"
print("✅ Winsorizer 通过")

# 插入连续停牌
panel.mask[20:30, 0] = False
s = SuspensionHandler(max_forward_fill=5)
panel = s.transform(panel)
assert panel.mask[26, 0], "5步内应被填充"
assert not panel.mask[30, 0], "超过5步应为 False"
print("✅ SuspensionHandler 通过")

# 归一化
n = Normalizer(method="zscore")
panel = n.fit_transform(panel)
col_std = panel.values[..., 0].std()
assert abs(col_std - 1.0) < 0.3, f"zscore 后标准差应接近 1, 实际 {col_std:.3f}"
print("✅ Normalizer 通过")
```

### 1.4 配置解析

```python
# 验证: YAML → DataConfig → DataLoader
from daft.data.config import load_data_config, make_dataloader
import yaml

with open("configs/small.yaml") as f:
    config = yaml.safe_load(f)

data_config = load_data_config(config["data"])
assert data_config.source == "synthetic"
assert data_config.n_stocks == 200

loader = make_dataloader(data_config)
batch = next(iter(loader))
assert batch["s_t"].shape[-1] > 0  # 至少有特征维
print("✅ 配置解析 + DataLoader 通过")
```

### 一键验证

```bash
python -m pytest tests/test_data_pipeline.py -v --tb=short
```

---

## 风险点

| 风险 | 影响 | 缓解 |
|------|------|------|
| baostock API 不稳定 / 限流 | 数据拉不全 | 加本地缓存 + 重试逻辑 + 降级到 synthetic |
| 数据频率对齐问题（日线/分钟线时间戳不一致） | 索引错乱 | 统一用 pandas DatetimeIndex 对齐，inner join |
| 沪深 300 成分股每年调整 | 回测期成分股集合变化 | 按月更新成分股列表，存 snapshot |
