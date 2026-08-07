# DAFT 双仓库协作工作流规范

> **版本**:v1.0 · 2026-08-07
> **适用范围**:DAFT 项目全部参与者(人 + 龙虾)
> **两个仓库**:`Dongxu-Jiang-daft`(主仓)与 `group-daft2`(协作仓)

---

## 1. 参与者与角色

| 身份 | GitHub 账号 | 角色 | 职责 |
|---|---|---|---|
| 蒋东旭 | `Alastair-Jiang` | **项目负责人 / 架构师** | 定方向、管主仓、最终验收、合入决策 |
| 李桂聿(真人) | `programmingWTF` | **核心贡献者 / 技术底座** | 基础设施、算法预研、Review 把关 |
| 桂鱼养的龙虾 | `LiGuiyu-AI` | **AI 贡献者** | 读论文→实现→开 PR→跑测试;在 issue 里随叫随到 |
| 王善堂(大工) | `Washington5533` | **贡献者** | 算法研究、代码贡献、实验复现 |
| cuda(东旭的龙虾) | `aB-iJ` | **AI 助手** | 写日志、知识注入、评审、跑实验、守流程 |

**角色权限(两个仓库一致)**:全部 4 个账号均有 push 权限,但 **push 不等于可以随便推主仓**——见第 3 节。

---

## 2. 双仓库分工(核心原则)

```
┌─────────────────────────────────────────────────────────┐
│  Dongxu-Jiang-daft  =  主仓库(稳定版 · 个人开发主线)        │
│  ─ 东旭亲自推动的架构、训练、评估、日志                      │
│  ─ 任何改动必须经东旭合入或明确授权                         │
│  ─ 这里是"最终真相"                                        │
├─────────────────────────────────────────────────────────┤
│  group-daft2  =  协作仓库(预研区 · 团队开发)               │
│  ─ 算法预研、新专家、测试、论文研究                         │
│  ─ 任何人(含龙虾)自由开分支、开 PR                        │
│  ─ PR 通过 Review 后合入 → 稳定后同步主仓                  │
└─────────────────────────────────────────────────────────┘
```

**一句话**:daft2 是实验室,daft 是发射台。**先实验、后合入、再上主仓。**

---

## 3. 标准工作流(所有贡献者遵守)

### 3.1 新想法/新算法 → 进 daft2

```
1. 开 Issue 描述想法(算法+论文链接+预期价值)
2. 开分支: feature/<姓名>-<功能>  (如 feature/guiyu-momentum)
3. 实现 + 写测试(必须!测试不过不开 PR)
4. 开 PR → 附:论文链接 + 运行结果 + 测试通过数
5. Review 人审(默认:programmingWTF 或 Alastair-Jiang)
6. 合入 daft2 main
7. 在 CHANGELOG.md 记录
```

### 3.2 daft2 稳定 → 同步主仓

```
1. daft2 的功能在 main 上稳定 ≥ 1 天且测试全绿
2. 东旭(或授权人)创建同步 PR 到 Dongxu-Jiang-daft
3. 主仓 Review → 合入 → 版本号更新(v0.x.y)
4. 更新 Learn-new/ 项目日志 + docs/PROJECT_REPORT.md
5. 发布日志(cuda 执行「更新项目日志」流程)
```

### 3.3 紧急修复 → 直接主仓

```
仅限:主仓 CI 崩溃、明显 bug、数据错误。
直接修 + push,但 commit message 必须标注 [HOTFIX] 并说明原因。
```

---

## 4. 分支与命名规范

| 项 | 规范 | 示例 |
|---|---|---|
| daft2 功能分支 | `feature/<姓名>-<功能>` | `feature/guiyu-momentum` |
| daft2 修复分支 | `fix/<姓名>-<简述>` | `fix/lobster-panel-mask` |
| 主仓分支 | 尽量直接在 main 小步提交;大改开 `dev/` 分支 | `dev/oos-validation` |
| PR 标题 | `[类型] 简述(作者)` | `[feat] MomentumExpert 动量专家(桂鱼)` |
| commit message | 动词开头,一句话说清"为什么" | `feat: 新增动量专家,补横截面动量缺口` |

---

## 5. 测试与验收门槛(硬性)

**任何 PR 合入 daft2 前必须:**
- [ ] 新增代码有对应测试(tests/test_*.py)
- [ ] `python -m pytest tests/ -q` 全绿(367 基线 + 新增)
- [ ] 运行结果存档(research/run-report.md 或 PR 描述)
- [ ] 无 NaN/Inf(DAFT 已有 smoke 检查,沿用)

**同步主仓前额外要求:**
- [ ] daft2 上稳定 ≥ 1 天
- [ ] 参数变更已同步(如 N_EXPERTS 8→10 需连带更新 build_experts)
- [ ] 检查点/配置兼容性确认

---

## 6. 人的职责边界(防冲突)

| 动作 | 谁能做 |
|---|---|
| 合入主仓 main | 仅 Alastair-Jiang(东旭)或经他明确授权 |
| 合入 daft2 main | programmingWTF / Alastair-Jiang 任一(Review 通过后) |
| 改 docs/PROJECT_REPORT.md | cuda(东旭指示)或东旭本人 |
| 改 Learn-new/ 日志 | cuda(东旭说「更新项目日志」时) |
| 动别人分支 | 禁止——各自分支各自写 |
| 动主仓 checkpoints/ | 禁止——训练产物不提交(gitignore) |

---

## 7. 沟通通道

| 场景 | 通道 | 说明 |
|---|---|---|
| 日常同步 | lobster-link(git) | 各写各目录,异步通信 |
| 技术讨论 | daft2 Issues/PR review | 公开留痕,便于追溯 |
| 紧急 | 直接 @(GitHub mention) | 龙虾在 issue 里随叫随到 |
| 日志/报告 | Learn-new/ 分区 | cuda 维护 |

---

## 8. 已知待办(工作流生效后马上做)

1. **N_EXPERTS 8→10 同步**:daft2 已改 10(5 类专家×2),主仓 build_experts 待同步
2. **MomentumExpert 合入主仓**:daft2 已稳定(367 测试),走 3.2 流程同步
3. **主仓测试套件补齐**:把 daft2 的 tests/ 移植到主仓(我们评审的 P0 工程债)
4. **CI 门禁**:主仓 GitHub Actions 加 pytest,daft2 已有测试基础
5. **真实数据对比实验**:含动量专家的完整 DAFT vs 4 专家 baseline(桂鱼 run-report 的待办)

---

## 9. 变更记录

| 版本 | 日期 | 变更 |
|---|---|---|
| v1.0 | 2026-08-07 | 初版:角色、双仓分工、工作流、验收门槛、职责边界 |

---

*本文档由 cuda 起草,经 Alastair-Jiang 确认后生效。所有参与者(人+龙虾)默认遵守。*
