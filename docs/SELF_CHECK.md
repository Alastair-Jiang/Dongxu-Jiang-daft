# DAFT 自检流程（固定工作流, 2026-08-17 建立）

> 执行时机: **每次实验后 / 每次提交前**。命令: `D:\env\python.exe scripts\self_check.py`
> 退出码: 0 = 通过(可能有历史债告警), 1 = 存在阻塞性问题, 必须修复后才能提交。

## 一、自动化检查（scripts/self_check.py）

| # | 检查项 | 阻塞? |
|---|---|---|
| 1 | 全量 py 编译（src/scripts/tests） | ✅ 阻塞 |
| 2 | outputs/*.json 结构完整性（按产物类型: oos/rebalance/smoothing/walk-forward/weekly） | ✅ 阻塞 |
| 2b | 训练型产物 seed 字段（可追溯性） | ⚠️ 告警（历史债） |
| 3 | 登记表表格行内 EXP ID 唯一性 | ⚠️ 告警（跨表重复正常） |
| 4 | 关键公式实现抽查（10 项 grep 断言，见下表） | ✅ 阻塞 |
| 5 | 路由损失结构（balance KL 存在、无熵抵消回归） | ✅ 阻塞 |

### 公式抽查清单（第 4 项，与 SELF_CHECK 公式表对应）

| 公式/约束 | 文件 | 断言 |
|---|---|---|
| CDAP logit 空间 | cross_dim_attn.py | `log_p = (routing_probs + 1e-12).log()` 后 softmax |
| KDA delta rule | memory.py | einsum 实现 M−βk⊗(Mk)+βk⊗v |
| safe gate 修复 | memory.py | `lb + (1-lb)·σ(exp(A_log)·(x+dt_bias))` |
| 专家 mask float | trend/reversal/volatility(+momentum/event) | `mask_f = mask.float()` |
| bincount CPU 化 | router.py | `topk_indices.flatten().cpu()` |
| MaxDD 百分比 | engine.py | `equity = torch.exp(cumret)` |
| 涨跌停 mask | baostock_adapter.py | `_limit_move_mask` |
| 通道契约 | base_features.py | `ensure_base_panel` |
| 标准化 train-only(A2) | router/joint_trainer.py | val 段 `norm_stats=self.norm_stats` 复用训练段统计量 |

## 二、人工检查（脚本覆盖不到的部分）

1. **数据**：新实验产物写入后，核对登记表行（EXP ID / IC / t / Sharpe / 换手 / 产物文件名 / config hash）与实际 JSON 逐字段一致。数字必须来自 `outputs/*.json`，不许凭记忆。
2. **公式**：新增模型代码时，对照 `docs/SPECIFICATION.md` 公式逐项核对（历史教训: 通道错列、概率空间 CDAP、safe gate 上下界乘反——全是"公式与代码漂移"）。
3. **口径**：新实验必须声明并对齐 k→k+1 对齐、val 选参、test 仅报告、唯一产物名。
4. **测试**：改动 src/ 后跑 `D:\env\python.exe -m pytest tests`（当前 395 passed / 1 skipped）。

## 三、已知历史债（不阻塞，但禁止继续新增）

- **33 个历史产物缺 seed 字段**（EXP-20260816-01~03、EXP-20260817-01~30 的 daft-oos）。
  根因: 2026-08-17 之前 run_full_pipeline_oos.py 的 report 未存 seed; 已修复。
  处置: 历史产物仅用于归档, **不得**作为未来结论的统计依据（种子不可追溯）;
  新产物自动带 seed, 告警数不再增长。

## 四、自检结果记录（最近一次）

- 2026-08-17: 0 阻塞 / 34 告警（33 seed 历史债 + 1 登记表跨表重复）。全量编译通过; 10 项公式断言全过。
