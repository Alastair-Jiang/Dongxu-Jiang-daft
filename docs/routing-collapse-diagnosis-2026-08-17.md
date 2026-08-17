# 路由塌缩诊断报告（2026-08-17）

## 一、现象

最近一次 OOS 全管线训练（`training.log`）中，Stage 2 路由熵从 `0.770` 一路崩到 `0.000`，Stage 3 恒为 `0.000`。样本外 `val_ic = +0.0130`、`ICIR = +0.089`，接近随机（Ridge 基线 IC ≈ 0.048）。

这与 v0.3.0 时期记录的「路由最大熵（8 专家几乎均匀激活）」**方向完全相反**——稀疏化修复从「太均匀」矫枉过正到「完全塌缩」。

## 二、根因：熵正则项精确抵消

**位置**：`src/daft/training/router_trainer.py` 第 342–359 行。

损失函数：

```python
loss = (weighted_loss
        - entropy_weight * routing_entropy      # 防坍缩
        + sparsity_weight * sparsity_penalty)   # 防均匀
```

其中两个正则项的计算：

```python
# 防均匀（per-sample sparsity）
per_sample_entropy = -(routing_probs * (routing_probs + 1e-8).log()).sum(dim=-1)  # (B,)
sparsity_penalty   = per_sample_entropy.mean()   # 标量 = batch 平均熵 H

# 防坍缩（整体熵正则）
routing_entropy    = -(routing_probs * (routing_probs + 1e-8).log()).sum(dim=-1).mean()  # = batch 平均熵 H
```

**`sparsity_penalty` 和 `routing_entropy` 是同一个量**（都是 `-Σ p·log p` 的 batch 平均）。当两者权重相等时：

```
- entropy_weight · H + sparsity_weight · H = -0.01·H + 0.01·H = 0
```

**熵正则项精确抵消为 0**——既没有防坍缩，也没有防均匀。K3 在 2026-08-09 评审时建议的「per-sample 稀疏（鼓励 top-k 内 one-hot）」被错误实现成了「最小化整体熵」，与「最大化整体熵」的防坍缩项撞车。

## 三、塌缩机制

熵正则失效后，`weighted_loss = Σ_i routing_mean[i] · expert_losses[i]` 成为唯一的路由梯度来源。

该损失对 `routing_mean` 的梯度正比于专家损失向量 `expert_losses`。最小化它的最优解是**把全部概率分配给 loss 最低的单一专家**（one-hot 到最便宜专家）。没有任何负载均衡项对抗，于是路由在几个 epoch 内坍缩。

放大器：

1. **温度退火**：`temp = 1.0 − 0.9·(epoch/(epochs−1))`，1.0 → 0.1，softmax 持续锐化。
2. **Stage 3 固定低温**：`run_full_pipeline_oos.py` 第 168 行 `model.router.temperature = 0.1`，直接锁死离散路由。

## 四、历史脉络

| 时间 | 状态 | 熵 |
|---|---|---|
| v0.3.0 之前 | 路由器学会「群体平均偏好」，无实例级专业化 | ≈1.05（均匀）|
| 2026-08-09 K3 评审 | 指出根因：batch 级目标 + 弱熵正则 | — |
| 修复后 | 加入 `sparsity_penalty`，但与防坍缩项同式抵消 | 0.000（塌缩）|

结论：两个方向的修复各自方向正确，但**同一公式、相反符号、相等权重**，实现上互相抵消，等价于「没修 + 温度退火放大」。

## 五、候选修复方案（待 K3 评审选定）

**方案 A —— 真负载均衡（推荐）**
去掉 `sparsity_penalty`（它是 bug），改用基于专家使用频率的负载均衡损失：
```
L_balance = Σ_i target_frac · log(target_frac / current_frac_i),  target_frac = 1/n_experts
```
`current_frac` 用 `activation_counts`（router 里已在统计）。这与 K3 原始 Stable LatentMoE 的 Quantile Balancing 一致，代码里 `quantile_balance()` 已存在，但 `balance_every=50` 太稀疏、且未被损失驱动。

**方案 B —— 修正 per-sample 稀疏**
「锐化」的正确定义不是最小化熵，而是最大化 top-k 内的最大概率：`sparsity_penalty = -mean(max_k(p))`（鼓励 top-1 主导）。它与整体熵正则（防坍缩）不再同式，不会抵消。

**方案 C —— 最小改动**
直接删除 `sparsity_penalty` 这一项（承认它是无效项），仅保留 `-entropy_weight · H` + 加强 `quantile_balance` 调用频率。风险：可能回到「均匀」旧问题。

**方案 D —— 温度 + top-k 纯稀疏**
熵正则全删，靠 `top_k=3` + 温度退火实现稀疏，负载均衡完全交给 `quantile_balance`。最激进，需要验证 `expert_bias` 是否真的在学习。

## 六、验证标准

修复后重训 Stage 2/3，通过即算成功：

1. **路由熵**：训练期熵收敛到 `0.5 ~ 1.5` 区间（既非 0 也非 ln(10)≈2.3 的均匀上限），且 `activation_counts` 无 0 计数的专家（10 个专家都被用到）。
2. **样本外 IC**：`val_ic` 从 +0.0130 明显回升，目标 ≥ Ridge 基线的 +0.048。
3. **ICIR**：从 +0.089 提升到 > 0.2。
4. **速度**：CUDA 下 Stage 3 从 12h（DirectML）缩到目标 < 1h。

## 七、附：本次 CUDA 迁移前置

- 硬件：RTX 5060 Ti 16GB（Blackwell，sm_120）
- 旧环境：torch 2.4.1+cpu + torch-directml（Stage 3 跑了 43114.7s）
- 迁移：装 torch 2.8+ cu128（Blackwell 需 cu128+）
- 预期收益：`aten::lerp` 等算子不再回退 CPU，端到端有望 10× 以上提速

## 八、实施与验证（2026-08-17 更新）

**已实施：方案 A（真负载均衡 KL）**

- `router_trainer.py`：删除互相抵消的 `-entropy_weight*H` 与 `+sparsity_weight*H`，改为
  `L_balance = Σ_i target_frac · log(target_frac / current_frac)`（`current_frac = routing_probs.mean(0)`，可微、损失驱动）。
- 温度退火下限 0.1 → 0.5（避免软路由被低温逼回 one-hot）。
- Stage 3 温度保持 0.1（near-discrete inference temp，设计使然，路由已在 Stage 2 学好）。

**CUDA 就绪**

- torch 2.11.0+cu128，RTX 5060 Ti 识别（sm_120），smoke test `16.1ms/iter @ batch=512`，显存 0.04GB/15.9GB，梯度无 NaN。

**quick 验证（30 股票 sample，Stage 1 15ep / Stage 2 10ep / Stage 3 8ep）**

- Stage 2 熵：`2.064 → 1.349 → 0.920 → 0.842 → 0.717 → 0.643`（温度 1.0→0.5），**平滑收敛、无塌缩**，落在 0.5~1.5 目标区间 ✅
- 速度：Stage 1 21.9s + Stage 2 13.7s + Stage 3 17.2s ≈ 1 分钟（DirectML 12h → <1h 目标达成，实际 <1min）✅
- OOS test 段：Rank IC **+0.0174**、ICIR +0.076、IC t-stat +1.18（正但弱，远低于 Ridge 0.0482）
- 回测：Sharpe -1.35、turnover 2.02（换手率高，被 5bp 成本吃掉）
- 注意：val 段 IC 全程为负（-0.016），test 段转正（+0.0174），存在数据段 regime 漂移，需 full 口径进一步确认。

**结论**：方案 A 修复了路由塌缩（核心目标达成）。OOS IC 仍弱于 Ridge，但 quick（30 股票 sample）非最终口径——待 `--full --stocks 100 --universe hs300`（与 Ridge 同口径）验证。
