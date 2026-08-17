# 更新日志 2026-08-17(Transformer 专家架构批次)

> 范围: Transformer 专家层架构升级的第一轮实现与 GPU 训练尝试。
> 状态: **训练已按用户要求暂停**;代码、结果、模型已分组归档(本文档),
> 待完成内容见 `docs/PLAN_2026-08-17-transformer-scaleup.md`。

---

## 1. 摘要

用户指令: 当前三维专家模型的数据量科学性与规模不够 → 参考 Transformer
架构、使用独立显卡进行下一步训练。

本轮完成:

1. **实现 Transformer 专家层**(特征维自注意力, Set-Transformer 风格),
   与现有 Stage1/2/3 管线、路由、CDAP、KDA 记忆无缝兼容(`--arch transformer`)。
2. **同批重跑 MLP 锚点**(`EXP-20260817-51`):IC +0.0331 / Sharpe −0.886,
   与历史最优锚点(+0.0331 / −0.886)**逐位一致 → 环境与种子可复现性确认**。
3. **GPU 训练两轮事故定位与修复**(蓝屏 → OOM → 显存封顶机制),Transformer
   训练在第三轮配置下运行中, 按用户要求暂停。

## 2. 新增/修改代码清单

| 文件 | 内容 |
|---|---|
| `src/daft/models/experts/transformer_expert.py` | **新增** TransformerExpert: 200 维特征 → 40 token×5 维 → 可学习位置编码 → pre-LN TransformerEncoder(GELU, 4×FFN, n_layers 块) → 均值池化 → SiTU head。全量训练(regime 专业化已被证伪), 接口完全兼容 BaseExpert |
| `src/daft/models/experts/__init__.py` | 导出 TransformerExpert |
| `src/daft/models/factory.py` | `build_experts(..., arch="mlp"/"transformer", n_heads)`;10 个 TransformerExpert 独立初始化、路由器分工 |
| `scripts/run_full_pipeline_oos.py` | `--arch` / `--n-heads` 参数;报告新增 `arch` / `n_heads` 字段 |
| `src/daft/utils/device.py` | **新增显存封顶钩子**: 环境变量 `DAFT_CUDA_FRACTION` → `torch.cuda.set_per_process_memory_fraction`。超限时干净 CUDA OOM, 不再溢出共享内存 |
| `scripts/train_transformer.py` | **新增** GPU 实验启动器 v3: 分波执行 + 每进程显存封顶 + 256×8 失败自动降级 192×8 |

## 3. 实验状态

| 实验 | 配置 | 状态 | 结果 |
|---|---|---|---|
| EXP-20260817-51(锚点重跑) | MLP 100 股 128×4, mem-ablate, 全量训练, seed 42 | ✅ 完成 | **IC +0.0331, t +3.11, ICIR +0.200, Sharpe −0.886, MaxDD −10.0%, 换手 2.18** — 与历史锚点完全一致 |
| tf100_128x4 | Transformer 100 股 128×4块×4头 | ⏸ 暂停(Stage3 OOM 后重启, 未完成) | 无报告;Stage1 31min / Stage2 11min 完成 |
| tf300_128x4 | Transformer 300 股 128×4块×4头 | ⏸ 暂停(Stage1 中途) | 无报告 |
| tf100_256x8 | Transformer 100 股 256×4块×8头 | ⏳ 未开始 | — |

完整报告: `outputs/EXP-20260817-51-daft-oos.json`;训练日志: `outputs/tf_*.log/.err`。

## 4. GPU 事故时间线与修复(重要教训)

| 版本 | 事件 | 处置 |
|---|---|---|
| v1(4 任务并行) | 显存推到 15.5GB 并溢出 Windows 共享内存 → **系统蓝屏**, 全部任务丢失 | 教训: 物理可用线约 15GB, 并发需显存预算 |
| v2(每任务 0.42×16G≈6.9GB) | 封顶生效: tf100 在 Stage3 干净 OOM(需要 ~6.7GB > 6.69GB 上限), **无蓝屏** | 验证: 0.10 封顶下 3GB 分配正确报 OOM |
| v3(双任务 0.45 / 单任务 0.85) | 0.45×2 峰值 14.3GB < 15GB;Stage3 实测 ~6.7GB < 7.17GB 上限 | 当前配置, 训练中暂停 |

## 5. 产物分组清单(保存在项目文件夹;git 策略: 权重/输出不入库)

**训练出的模型(checkpoints/, 磁盘保存)**:

- `checkpoints/mlp_128x4_anchor/` — **完整可用模型**(14 个 .pt, 共 3.75MB):
  experts×10 + router + memory + cdap + layer_proj(EXP-20260817-51 对应权重)
- 历史完整 checkpoint 集: `checkpoints/fit-{reg,all}-{mem,none}/`(专家层拟合批次)、
  `checkpoints/cap*/`(容量扫描)、`checkpoints/oos*/`、`checkpoints/weekly/`、`checkpoints/stage1|2|3/`
- Transformer 任务的 checkpoint 目录未生成(Stage3 完成前不落盘;恢复后自动写入)

**训练结果(outputs/, 磁盘保存)**:

- `outputs/EXP-20260817-51-daft-oos.json` — 本轮锚点(见 §3)
- `outputs/EXP-20260817-46~50-*.json` — 专家层拟合批次(regime 专业化证伪,
  详见 commit f75d61f 与 `docs/RESEARCH_FINDINGS_2026-08-17.md`)
- 历史全部 EXP-*.json(自 2026-08-16 起唯一命名, 78 份, 自检已核验结构)

**数据(data/cache/, 磁盘保存)**: `6f63423117ef.pt`(100 股 hs300)、
`bb8238f88a82.pt`(300 股 hs300)、`cc7530cfaf2e.pt`、`e3eba1b7061a.pt`、
`d8fcfe900658.pt`(universe 元数据)、`us_100.pt`(美股 100)

## 6. 质量门

- `scripts/self_check.py`: **13 通过, 0 阻塞, 35 告警(历史债)** ✓
- CUDA 冒烟测试: TransformerExpert 前向 / 10 专家池 / ensemble 组装 / 损失均通过
- 显存封顶机制: 独立验证通过(§4)

## 7. 暂停点与恢复

恢复命令: `D:\env\python.exe scripts\train_transformer.py`
(波1: tf100_128x4 + tf300_128x4 @0.45;波2: tf100_256x8 @0.85, 失败自动降级 192×8)。
详细计划见 `docs/PLAN_2026-08-17-transformer-scaleup.md`。
