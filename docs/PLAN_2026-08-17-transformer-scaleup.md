# 待开发计划 — Transformer 架构升级批次(2026-08-17)

> 状态: 训练暂停(用户指令)。恢复入口与步骤见 §1。
> 配套: `docs/UPDATE_LOG_2026-08-17-transformer.md`(已完成内容与产物清单)。

---

## 1. 立即恢复步骤

1. 确认 GPU 空闲: `nvidia-smi`(可用线 ~15GB)。
2. 运行: `D:\env\python.exe scripts\train_transformer.py`
   - 波1: `tf100_128x4` + `tf300_128x4`, 每进程 `DAFT_CUDA_FRACTION=0.45`(≈7.2GB),
     并行峰值 14.3GB, 安全;
   - 波2: `tf100_256x8` 单独跑, fraction 0.85(≈13.5GB);若 Stage3 OOM
     (exit≠0), 启动器自动降级 `tf100_192x8`。
3. **显存纪律(硬规则)**: 并发 ≤2 进程;每进程必须带 `DAFT_CUDA_FRACTION`;
   峰值预算 = Σ(fraction×16GB) + 桌面开销 ≤ 14.5GB。任何超预算改动先算账。

预计墙钟: tf100 ≈ 70–80min(Stage1≈31m / Stage2≈12m / Stage3≈18m / 回测≈12m);
tf300 ≈ 3.5–4h(数据 3×, Stage1 已实测最慢);波2 ≈ 1.5–2h。合计约 4.5–6h。

## 2. 剩余实验矩阵与判定映射

| 任务 | 假设 | 判据(对照锚点) |
|---|---|---|
| tf100_128x4 | 特征自注意力优于 MLP | IC > **0.0331**(MLP 锚点 EXP-20260817-51);> **0.0482** 才超 Ridge |
| tf300_128x4 | Transformer 规模鲁棒 | MLP@300 股曾崩至 ~0;Transformer IC 不掉 = 规模鲁棒性证据 |
| tf100_256x8 | 容量增益 | IC > tf100_128x4 才有扩宽价值 |

每个完成实验必须: 登记 `docs/EXPERIMENT_REGISTRY.md`(唯一 EXP ID / seed /
arch / hidden / ckpt-dir / config hash), 报告自动含 `arch`、`n_heads` 字段。

## 3. 完成后判定路径

1. **架构升级有效**(tf100 IC > 0.0331 且 t ≥ 2): 扩大数据规模
   (300→800 股中证 800、更早历史 2016–2020), 重测 Ridge 同口径;
2. **300 股不崩**: 单独登记规模证据;结合 1 决定是否推翻 NO-GO(架构);
3. **无效**(IC ≤ 0.0331 或 300 股再崩): 维持 NO-GO(架构)结论,
   将 transformer 结果并入研究存档, 项目转向归档。

## 4. 后续架构候选(按优先级)

- **P0 · 特征语义分组的 token 化**: 现为固定 40×5 分块;改为 6 个语义组
  (g1 价格 45 / g2 量 35 / g3 波动 40 / g4 微观 20 / g5 截面 30 / g6 动量 30),
  各组独立投影到 d_model(不等长 token → 单 token/组, 组间自注意力)。
- **P1 · 时序 Transformer**: 样本从单步 s_t 改为 K 天窗口序列
  (股票 × K 天 × 200 维), 时间维自注意力;需新建数据管线(窗口化 +
  因果 mask), 与 Ridge 基线口径对齐成本较高。
- **P2 · 训练规模**: transformer 对 epochs 的敏感性扫描(quick 15/10/8
  → full 50/30/20, 注意 MLP 的 --full 已证伪, 先查早停曲线再扩)。
- **P3 · 混合专家**: transformer + MLP 专家各 5 个, 检验架构多样性对
  路由器的价值(需 factory 扩展, 低优先级)。

## 5. 工程守则(沿用并强化)

- 提交前必跑 `scripts/self_check.py`(0 阻塞);
- main 受保护: 分支 + PR, 不直接/强推;产物唯一命名 `next_exp_path`;
- 参数只准在 val 上选, test 只出报告;
- 新增大容量实验前先按 §1 显存纪律预算。
