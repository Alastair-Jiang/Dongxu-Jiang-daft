# CP6: 硬化统计收集与消融实验 (Stage 4)

> **状态**: pending
> **依赖**: [CP5 联合微调](cp5_joint_finetune.md)
> **预计工作量**: 4–5 天
> **后续**: [CP7 回测与评估](cp7_backtest_eval.md)

---

## 目标

在训练完成的模型上运行硬化统计收集（forward pass 遍历全量数据，记录 pattern 频率），
构建硬化缓存。完成 `configs/hardening.yaml` 中定义的消融实验：
对比硬化 ON vs OFF 在速度、精度、regime 降级率上的差异。

## 前置依赖

- CP5：联合微调完成，模型权重已固定
- CP2：全量数据可遍历 (train + val + test)

---

## 任务清单

### 6.1 硬化统计收集

- [ ] 遍历训练集全部时间步，每个时间步调用 `hardening.should_use_fast_path()`
- [ ] 收集 stage 从 0 到 `threshold` 的 pattern 频率分布
- [ ] 输出：`n_cached_patterns`, `cache_hit_rate`, `baseline_entropy`
- [ ] 可视化：pattern 频率 Zipf 分布图（预期：少量高频 pattern + 大量长尾）
- [ ] 保存 `cache.pt` 供推理时加载

**文件**: `scripts/collect_hardening_stats.py`

### 6.2 消融实验：Hardening ON vs OFF

对照 `configs/hardening.yaml`：

- [ ] 在测试集上跑两遍 forward：
  - `hardening_enabled=False` — 所有样本走全量 CDAP (slow path)
  - `hardening_enabled=True` — fast/slow 自适应
- [ ] 对比指标：
  - `hardened_ratio`: slow path 占比
  - `speedup_ratio`: 推理耗时比
  - `accuracy_delta`: Sharp/IC 差异
  - `regime_degradation_rate`: fast path 被熵检测拉回到 slow 的比例
- [ ] 参数扫描：`threshold ∈ [50, 100, 200, 500]`, `min_confidence ∈ [0.90, 0.95, 0.99]`

**文件**: `scripts/run_hardening_ablation.py`

### 6.3 Regime Shift 检测验证

- [ ] 构造"平稳 → 闪崩"的人工序列
- [ ] 验证 `detect_regime_shift()` 在崩盘时刻触发
- [ ] 验证触发后 fast path 被临时禁用，恢复后重新启用

**文件**: `tests/test_hardening.py` (本 CP 只需硬化相关的测试)

### 6.4 Staleness Eviction

- [ ] 模拟长时间不使用的 cache entry → 验证 `evict_stale_entries()` 正确清除
- [ ] 验证 eviction 后对应 pattern 重新积累到 threshold 才会重新硬化

---

## 验收标准

| # | 标准 | 验证 |
|---|------|------|
| 1 | 训练集 pattern 统计完成，cache 条目数 ≥ 10 | Script 输出 |
| 2 | 硬化模式下推理速度提升 ≥ 40% | 计时对比 |
| 3 | 硬化后 Sharpe/IC 相对 full path 下降 < 5% | 消融表 |
| 4 | Regime shift 检测在闪崩场景准确触发 | 单元测试 |
| 5 | `threshold=200, min_confidence=0.95` 时 hardened_ratio > 60% | 数据 |

---

## 快速验证

### 6.1 硬化统计收集

```python
from scripts.collect_hardening_stats import collect_stats
from daft.models.hardening import HardeningEngine
import torch

# 用已训练的模型跑统计收集
hardening = HardeningEngine(threshold=50, min_confidence=0.90)

# 模拟 500 个时间步，同一 pattern 反复出现
for t in range(500):
    probs = torch.tensor([0.5, 0.2, 0.1, 0.1, 0.05, 0.03, 0.01, 0.01])
    regime_id = 3 if t < 400 else 6  # 前 400 步 regime=3, 后 100 步 regime=6
    hardening.should_use_fast_path(regime_id, probs)

stats = hardening.get_stats()
assert stats["n_cached_patterns"] >= 1, "没有 pattern 被缓存"
assert stats["total_decisions"] == 500
cache_hit = stats["cache_hit_rate"]
print(f"✅ 硬化统计: cached={stats['n_cached_patterns']}, "
      f"fast={stats['n_fast_path']}, slow={stats['n_slow_path']}, "
      f"hit_rate={cache_hit:.3f}")
```

### 6.2 消融实验：ON vs OFF

```python
import time

# 模拟一批样本，一半常见 pattern (走 fast), 一半罕见 (走 slow)
hardening = HardeningEngine(threshold=10, min_confidence=0.90)

# 先预热：让一个 pattern 达到 threshold
for _ in range(10):
    hardening.should_use_fast_path(3, torch.tensor([0.5, 0.2, 0.1, 0.1, 0.05, 0.03, 0.01, 0.01]))

# === Full path (hardening disabled) ===
t0 = time.perf_counter()
for _ in range(100):
    # 模拟一次 CDAP forward (相当于 slow path)
    _ = torch.softmax(torch.randn(4, 8), dim=-1)
t_full = time.perf_counter() - t0

# === Hardened path ===
t0 = time.perf_counter()
for _ in range(100):
    hardening.should_use_fast_path(3, torch.tensor([0.5, 0.2, 0.1, 0.1, 0.05, 0.03, 0.01, 0.01]))
t_hard = time.perf_counter() - t0

speedup = t_full / t_hard
assert speedup > 1.5, f"硬化未加速: speedup={speedup:.1f}x"
print(f"✅ 硬化加速: full={t_full*1000:.1f}ms, hard={t_hard*1000:.1f}ms, speedup={speedup:.1f}x")

stats = hardening.get_stats()
fast_ratio = stats["fast_path_ratio"]
print(f"   fast_path_ratio={fast_ratio:.1%}")
```

### 6.3 Regime Shift 检测

```python
hardening = HardeningEngine(entropy_multiplier=2.0)

# 建立低熵 baseline (稳定市场)
for _ in range(100):
    hardening.should_use_fast_path(3, torch.tensor([0.7, 0.1, 0.05, 0.05, 0.04, 0.03, 0.02, 0.01]))

# 注入高熵（不确定）样本模拟 regime shift
for _ in range(20):
    hardening.should_use_fast_path(3, torch.tensor([0.2, 0.2, 0.15, 0.15, 0.1, 0.1, 0.05, 0.05]))

shifted = hardening.detect_regime_shift()
assert shifted, "高熵样本应触发 regime shift 检测"
print(f"✅ Regime shift 检测: shifted={shifted}, degradation_count={hardening.n_degradations}")
```

### 6.4 Staleness Eviction

```python
hardening = HardeningEngine(threshold=10)
# 创建一个 cache entry 但从不使用
for _ in range(10):
    hardening.should_use_fast_path(1, torch.tensor([0.1]*8))

assert len(hardening.cache) >= 1

# 强制 eviction (max_age=1, 几乎没有 hit)
n = hardening.evict_stale_entries(max_age=1)
assert n >= 1, "过期 entry 应被清除"
print(f"✅ Eviction: 清除了 {n} 个过期 entry")
```

### 一键验证

```bash
python scripts/collect_hardening_stats.py --config configs/hardening.yaml
python scripts/run_hardening_ablation.py --config configs/hardening.yaml
```

---

## 风险点

| 风险 | 影响 | 缓解 |
|------|------|------|
| 缓存条目过多 (>500) 反而增加查表开销 | fast path 不加速 | 限制 cache 大小，LRU eviction |
| 过于激进的硬化导致 regime 转换时高误差 | 误判市场 | 调低 entropy_multiplier → 更容易触发降级 |
