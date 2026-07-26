# CP3: 专家独立训练 (Stage 1)

> **状态**: pending
> **依赖**: [CP2 特征工程](cp2_feature_engine.md)
> **预计工作量**: 5–7 天
> **后续**: [CP4 路由+记忆训练](cp4_router_memory_training.md)

---

## 目标

完成四个策略专家的独立训练。每个专家在自己的 regime-filtered 数据子集上，
用专属 loss 函数训练到收敛。训练后冻结所有专家权重，为 CP4 Stage 2 做准备。

## 前置依赖

- CP2：200 维 `s_t` + 三层深度表示 `[h0, h1, h2]`
- CP2：regime 标签（ADX 代理、ATR 分位等）用于 `_regime_filter()`

---

## 任务清单

### 3.1 专家 _regime_filter 实现

当前四个专家全部 `raise NotImplementedError`，需要补完：

- [ ] `TrendExpert._regime_filter(panel)` — ADX > 25 的样本，用 ADX 代理 (DX 公式)
- [ ] `ReversalExpert._regime_filter(panel)` — ADX < 20 的样本
- [ ] `VolatilityExpert._regime_filter(panel)` — ATR 高于滚动 80 分位
- [ ] `EventExpert._regime_filter(panel)` — 事件日历 ±3 天窗口。初版可 fallback 到全量训练（事件专家最稀疏）

**文件**: `src/daft/models/experts/{trend,reversal,volatility,event}_expert.py`

### 3.2 ExpertTrainer 实现

当前 `ExpertTrainer.train()` 全空，需实现：

- [ ] 训练循环：epoch × batch loop
- [ ] 每个专家的专属 loss 调用（已实现：AdjMSE / -IC / MSE+Var / BCE）
- [ ] Optimizer: AdamW, lr=0.001, weight_decay=1e-5, cosine warm-restart scheduler
- [ ] Gradient clipping: max_norm=1.0
- [ ] Early stopping: validation loss 20 轮不降即停
- [ ] Checkpoint 保存：每个专家的 best model weights
- [ ] 验证：每 5 epoch 输出 train/val loss + 各专家的方向准确率

**文件**: `src/daft/training/expert_trainer.py`

### 3.3 训练配置

- [ ] 从 `configs/small.yaml` 和 `configs/paper.yaml` 的 `training.stage1` 段读取配置
- [ ] `small.yaml`: 50 epochs, batch 2048, lr 0.001 — 冒烟测试用
- [ ] `paper.yaml`: 200 epochs, batch 4096, lr 0.001 — 完整训练用

### 3.4 训练数据准备

- [ ] 按 `train_split=0.7 / val_split=0.15 / test_split=0.15` 切分
- [ ] 时间序列 walk-forward 切分（非随机打乱，防止未来信息泄露）
- [ ] DataLoader 批量加载 `(s_t_batch, target_batch, mask_batch)`

**文件**: `src/daft/data/split.py`

---

## 验收标准

| # | 标准 | 验证 |
|---|------|------|
| 1 | 四个专家各自在 regime-filtered 子集上完成训练，val loss 收敛 | Train log 输出 |
| 2 | Trend 专家方向准确率 > 基准（时序均值预测） | 测试集评估 |
| 3 | Reversal 专家 Rank IC > 0.02 | 测试集评估 |
| 4 | 所有专家权重成功保存/加载（roundtrip 通过） | 单元测试 |
| 5 | 冻结专家后权重确实不可训练（`requires_grad=False`） | 检查 `param.requires_grad` |
| 6 | 小数据集 (small.yaml) 上训练时间 < 5 分钟 (GPU) / < 15 分钟 (CPU) | 计时 |

---

## 快速验证

### 3.1 _regime_filter

```python
from daft.features.regime_features import RegimeFeatureExtractor
from daft.data.sources import SyntheticSource
from daft.models.experts import TrendExpert, ReversalExpert, VolatilityExpert, EventExpert
import torch

# 构造包含不同 regime 的合成数据
panel = SyntheticSource(n_assets=5, n_days=500, seed=42).load()
# 手动注入 regime 标签用于验证
extractor = RegimeFeatureExtractor()
s_t = extractor(panel)  # (500, 5, 200)

# 趋势专家
trend = TrendExpert()
mask_t = trend._regime_filter(panel)
# 趋势样本应在价格序列有持续方向性时出现
assert mask_t.shape == (500,)
assert mask_t.sum() > 50, f"趋势样本太少: {mask_t.sum()}"
print(f"✅ Trend filter: {mask_t.sum()}/{500} samples = {mask_t.sum()/500:.1%}")

# 反转专家
reversal = ReversalExpert()
mask_r = reversal._regime_filter(panel)
assert mask_r.sum() > 50, f"反转样本太少: {mask_r.sum()}"
# 同一时刻不应该既 trend 又 reversal (ADX 不可能同时 >25 和 <20)
overlap = (mask_t & mask_r).sum()
assert overlap == 0, f"有 {overlap} 个样本同时被 trend 和 reversal 选中"
print(f"✅ Reversal filter: {mask_r.sum()}/{500}, overlap with trend={overlap} ✅")

# 波动率专家
vol = VolatilityExpert()
mask_v = vol._regime_filter(panel)
assert mask_v.sum() > 50
print(f"✅ Volatility filter: {mask_v.sum()}/{500}")
```

### 3.2 ExpertTrainer 冒烟

```python
from daft.training.expert_trainer import ExpertTrainer
from daft.data.sources import SyntheticSource
import yaml

with open("configs/small.yaml") as f:
    config = yaml.safe_load(f)["training"]["stage1"]

experts = [TrendExpert(), ReversalExpert(), VolatilityExpert(), EventExpert()]
trainer = ExpertTrainer(experts, config, device=torch.device("cpu"))

# 用小数据集跑 2 个 epoch，验证不报错
panel = SyntheticSource(n_assets=20, n_days=300, seed=42).load()
metrics = trainer.train(panel, panel)  # train=val for smoke test

assert "trend_loss" in metrics
assert metrics["trend_loss"] > 0
assert metrics["trend_loss"] < 100  # 不应爆炸
print(f"✅ ExpertTrainer 冒烟: metrics={ {k: f'{v:.4f}' for k,v in metrics.items()} }")
```

### 3.3 权重冻结

```python
# 验证冻结逻辑
for expert in experts:
    expert.train(False)
    for p in expert.parameters():
        assert not p.requires_grad, f"{expert.name} 未冻结"
print("✅ 所有专家权重已冻结")
```

### 3.4 Checkpoint roundtrip

```python
import tempfile, os
tmp = tempfile.mkdtemp()
path = os.path.join(tmp, "trend_expert.pt")
torch.save(trend.state_dict(), path)

loaded = TrendExpert()
loaded.load_state_dict(torch.load(path))

x = torch.randn(4, 200)
assert torch.allclose(trend(x), loaded(x), atol=1e-5), "checkpoint 不一致"
print("✅ Checkpoint roundtrip 通过")
```

### 一键验证

```bash
python -m pytest tests/test_experts.py tests/test_expert_trainer.py -v --tb=short
```

---

## 风险点

| 风险 | 影响 | 缓解 |
|------|------|------|
| 某个 experts 的 regime 样本太少，训练不充分 | 专家能力弱 | 设最小样本数阈值，不足则用全量 |
| 不同专家的 loss 量级差异大，联合训练时权重失调 | Stage 3 不稳定 | 记录各 loss 量级，Stage 3 做 loss rescaling |
| Event expert 无事件日历则无法训练 | 少一个专家 | 初版用全量数据训练，loss 不变 |
