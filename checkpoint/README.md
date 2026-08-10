# DAFT Development Checkpoints

> 从 PR #2 (Washington5533) 移植的开发路线文档。
> 共 11 个检查点，覆盖从数据管道到硬化推理的完整开发流程。

## 检查点概览

| CP | 主题 | 估计工期 | 状态 |
|----|------|---------|------|
| CP1 | 数据管道 (Data Pipeline) | 5–7 天 | ✅ |
| CP2 | 特征工程 (Feature Engine) | 5–7 天 | ✅ |
| CP3 | 专家训练 (Expert Training) | 5–7 天 | ✅ |
| CP4 | 路由器+记忆训练 (Router+Memory) | 5–7 天 | ✅ |
| CP5 | 联合微调 (Joint Finetune) | 5–7 天 | ✅ |
| CP6 | 硬化引擎 (Hardening Engine) | 4–5 天 | ✅ |
| CP7 | 回测评估 (Backtest Eval) | 5–7 天 | ✅ |
| CP8 | 投资组合 (Portfolio) | 4–5 天 | ✅ |
| CP9 | 测试框架 (Testing) | 5–7 天 | ✅ |
| CP10 | 文档与 Notebook | 4–5 天 | ✅ |
| CP11 | Gate 残差化 (Gate Residual) | 3–5 天 | 🟡 代码已完成 |

## 使用方式

每个 CP 包含：
1. **目标**：该阶段要达成的具体指标
2. **输入/输出**：数据流和交付物
3. **可运行验证**：一键验证命令
4. **潜在风险**：可能出问题的地方和应对方案

## 建议阅读顺序

- **新贡献者**：CP1 → CP2 → CP3 → CP4 → CP5 → CP6 → CP7 → CP8 → CP9
- **只关心模型架构**：CP3 → CP4 → CP5 → CP6
- **只关心交易/回测**：CP1 → CP7 → CP8
