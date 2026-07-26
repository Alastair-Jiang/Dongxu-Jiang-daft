# DAFT 开发检查点总览

> 最后更新: 2026-07-26
> 当前状态: 核心架构完成 (1451 行 Python)，管道/训练/测试/文档全部待补完

---

## 检查点列表

| CP | 名称 | 状态 | 依赖 | 工作量 | 核心交付 |
|----|------|------|------|--------|----------|
| [CP1](cp1_data_pipeline.md) | 数据管道基础 | `pending` | — | 5–7d | Panel 数据结构 + 3 种数据源 |
| [CP2](cp2_feature_engine.md) | 特征工程实现 | `pending` | CP1 | 8–10d | 213 因子 + s_t 构建 + 三层深度 |
| [CP3](cp3_expert_training.md) | 专家独立训练 | `pending` | CP2 | 5–7d | 4 个专家完成 Stage 1 训练 |
| [CP4](cp4_router_memory_training.md) | 路由+记忆训练 | `pending` | CP3 | 5–7d | Router + KDA Memory 收敛 |
| [CP5](cp5_joint_finetune.md) | 联合微调 | `pending` | CP4 | 5–6d | CDAP 三链路全通 |
| [CP6](cp6_hardening.md) | 硬化统计收集 | `pending` | CP5 | 4–5d | Cache 构建 + 消融实验 |
| [CP7](cp7_backtest_eval.md) | 回测与基准评估 | `pending` | CP6 | 6–8d | 完整回测 + 基准对比 |
| [CP8](cp8_portfolio.md) | 组合优化 | `pending` | CP7 | 4–5d | Ledoit-Wolf + 马科维兹 |
| [CP9](cp9_testing.md) | 测试体系 | `pending` | CP3–8 | 5–7d | 80% 覆盖率三层测试 |
| [CP10](cp10_docs_notebooks.md) | 文档与 Notebook | `pending` | CP7/8/9 | 3–4d | 8 教程 + API 文档 + 复现指南 |

**总预计工作量**: 50–64 天

---

## 依赖关系图

```
CP1 ──→ CP2 ──→ CP3 ──→ CP4 ──→ CP5 ──→ CP6 ──→ CP7 ──→ CP8 ──→ CP10
                                              │                │
                                              └──→ CP9 ←──────┘
                                              (CP9 可与 CP3–CP8 并行推进)
```

---

## 当前已完成的代码（CP 开始前的基础）

| 文件 | 行数 | 完成度 | 备注 |
|------|------|--------|------|
| `models/router.py` | 188 | 95% | Quantile Balancing 完整 |
| `models/memory.py` | 230 | 90% | 含 set_external_gate (PR 修复后) |
| `models/cross_dim_attn.py` | 198 | 90% | 正反向投影完整 |
| `models/hardening.py` | 255 | 85% | 统计+缓存完整 |
| `models/ensemble.py` | 214 | 80% | 逐样本硬化 (PR 修复后) |
| `models/experts/base_expert.py` | 141 | 85% | SiTU 激活 + MLP 骨架 |
| `models/experts/trend_expert.py` | 72 | 40% | forward OK, _regime_filter 空 |
| `models/experts/reversal_expert.py` | 71 | 40% | forward OK, _regime_filter 空 |
| `models/experts/volatility_expert.py` | 65 | 40% | forward OK, _regime_filter 空 |
| `models/experts/event_expert.py` | 72 | 40% | forward OK, _regime_filter 空 |
| `features/tensor_factors.py` | 95 | 5% | 全部 NotImplementedError |
| `features/regime_features.py` | 55 | 5% | 全部 NotImplementedError |
| `features/freq_features.py` | 89 | 40% | compute_periodogram OK, forward 空 |
| `features/legacy_factors.py` | 24 | 5% | 空壳 |
| `training/expert_trainer.py` | 57 | 5% | 全部 NotImplementedError |
| `training/router_trainer.py` | 37 | 5% | 全部 NotImplementedError |
| `training/joint_trainer.py` | 35 | 5% | 全部 NotImplementedError |
| `backtest/engine.py` | 82 | 10% | 仅静态方法骨架 |
| `portfolio/markowitz.py` | 66 | 5% | 全部 NotImplementedError |
| `tests/__init__.py` | 11 | 0% | 空文件 |
| `configs/*.yaml` | 3 文件 | 90% | paper/small/hardening 配置完整 |

---

## 快速启动指引

### 第一次参与开发

1. 阅读 `README.md` + 本文档
2. 阅读 `src/daft/models/` 下的核心模型代码（可运行的 1451 行）
3. 从 **CP1** 开始 — 这是所有下游模块的基础

### 已有 ML 背景、熟悉量化

可以从 **CP3** 开始（专家训练），CP1–CP2 用 synthetic 数据快速 mock。

### 可以做并行开发的任务

- CP1+CP2 先行 → CP3 启动后 → CP9 可开始写模型测试
- CP7+CP8 可在 CP5 完成后并行（回测与组合优化解耦）

---

## 状态标记

- `pending` — 尚未开始
- `in_progress` — 正在开发
- `review` — 等待审查
- `done` — 已完成并合并
- `blocked` — 被其他 CP 阻塞
