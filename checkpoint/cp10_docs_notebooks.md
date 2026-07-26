# CP10: 文档与 Notebook 教程

> **状态**: pending
> **依赖**: [CP7](cp7_backtest_eval.md), [CP8](cp8_portfolio.md), [CP9](cp9_testing.md)
> **预计工作量**: 3–4 天
> **后续**: 项目收尾，进入维护阶段

---

## 目标

补全文档、撰写 Jupyter Notebook 教程、生成 API 文档，
使项目达到"外部用户能跑通 + 能看懂 + 能复现实验"的水平。

## 前置依赖

- CP7：回测能跑，有产出指标
- CP8：组合优化可用
- CP9：所有测试通过

---

## 任务清单

### 10.1 教程 Notebook

- [ ] `00_quickstart.ipynb` — 5 分钟快速上手：安装 → 加载数据 → 加载预训练模型 → 跑一次预测
- [ ] `01_data_pipeline.ipynb` — 数据源配置、预处理 pipeline、mask 可视化
- [ ] `02_feature_engineering.ipynb` — 因子计算演示、FFT 频谱分析、s_t 构成解析
- [ ] `03_training.ipynb` — Stage 1→3 完整训练流程，loss 曲线，路由分布演化
- [ ] `04_cdap_analysis.ipynb` — CDAP 三条调制链路可视化、消融实验、联合空间激活模式
- [ ] `05_hardening_analysis.ipynb` — 硬化统计、fast/slow path 分布、regime shift 案例
- [ ] `06_backtest.ipynb` — 回测结果分析、绩效归因、与基准对比图表
- [ ] `07_portfolio.ipynb` — 组合优化、有效前沿、权重热力图

**文件**: `notebooks/*.ipynb`

### 10.2 API 文档

- [ ] 所有公开类和方法的 docstring 补全（NumPy 风格）
- [ ] 生成 Sphinx / MkDocs 静态文档
- [ ] `docs/api/` 下每个模块一页
- [ ] README 中的 Architecture 章节引用 API 文档链接

**文件**: `docs/api/`, `mkdocs.yml` 或 `conf.py`

### 10.3 复现指南

- [ ] `docs/reproduce.md` — 从头复现 README 中所有实验结果的完整步骤
  - 环境配置 (conda/pip)
  - 数据获取 (baostock 注册, 数据下载脚本)
  - 训练命令 (+ 预期耗时)
  - 评估命令
  - 预期结果范围 (mean ± std)
- [ ] `scripts/download_data.sh` / `.ps1` — 一键下载所需数据
- [ ] `Makefile` — `make data`, `make train`, `make eval`, `make test`, `make docs`

### 10.4 README 完善

- [ ] 添加 Badge: test status, coverage, docs
- [ ] Quick Start section: 3 条命令跑通
- [ ] 将声称的 Benchmark 数字替换为实际实验输出
- [ ] 添加 Limitations 章节的实测讨论

---

## 验收标准

| # | 标准 | 验证 |
|---|------|------|
| 1 | 8 个 notebook 全部可从头到尾运行不出错 | CI notebook 自动化 |
| 2 | 新用户按 Quick Start 能在 10 分钟内跑通 | 找一个人试用 |
| 3 | API 文档覆盖率 = 所有公开函数 | doc coverage tool |
| 4 | `make test` + `make docs` 一次性通过 | CI |

---

## 快速验证

### 10.1 Notebook 自动化——全部运行不报错

```bash
# 用 jupyter nbconvert 跑所有 notebook，确保无异常
for nb in notebooks/0*.ipynb; do
    echo "--- Running $nb ---"
    jupyter nbconvert --to notebook --execute --inplace "$nb" 2>&1 || {
        echo "❌ $nb failed"
        exit 1
    }
    echo "✅ $nb passed"
done
```

### 10.2 快速上手——3 条命令

```bash
# 模拟新用户从零开始
pip install -e ".[dev]"
python -c "
from daft.data.sources import SyntheticSource
from daft.models import RegimeRouter
import torch

# 生成数据
panel = SyntheticSource(n_assets=10, n_days=100, seed=42).load()
s_t = torch.randn(4, 200)  # 模拟 market state

# 创建模型并运行
router = RegimeRouter()
probs, indices, z, full = router(s_t)
print(f'DAFT version: {__import__(\"daft\").__version__}')
print(f'Routing: top-3 experts = {indices[0].tolist()}, probs = {probs[0].tolist()}')
print('✅ Quick start 通过')
"
```

### 10.3 API 文档覆盖

```bash
# 检查所有公开 API 是否有 docstring
python scripts/check_doc_coverage.py

# 预期输出:
# modules: 10/10 documented
# classes:  15/15 documented
# methods:  45/45 documented
# ✅ Doc coverage: 100%
```

### 10.4 复现指南——冒烟测试

```bash
# 确保复现流程每一步可运行
make data-synthetic   # 生成合成数据
make train-small      # small.yaml 训练 (<30s CPU)
make eval-small       # 评估 + 输出指标
make test             # 跑测试套件

# 检查评估输出是否包含所有核心指标
python -c "
import json
with open('results/small_metrics.json') as f:
    m = json.load(f)
required = ['sharpe_ratio', 'max_drawdown', 'calmar_ratio', 'ic_rank', 'icir', 'hit_rate']
for r in required:
    assert r in m, f'Missing metric: {r}'
print(f'✅ All {len(required)} required metrics present')
"
```

### 10.5 Markdown lint

```bash
# 所有 .md 文件格式检查
npx markdownlint-cli '**/*.md' --ignore node_modules 2>&1 | grep -c "MD"
# 预期: 0 warnings
echo "✅ Markdown lint 通过 (0 warnings)"
```

### 一键验证

```bash
# 完整文档验证
make test          # 代码测试
make docs          # 生成文档
make smoke-docs    # notebook 自动化
python scripts/check_doc_coverage.py  # API 文档覆盖率
npx markdownlint-cli '**/*.md' --ignore node_modules  # Markdown lint
echo "✅ CP10 全部验证通过"
```

---

## 风险点

| 风险 | 影响 | 缓解 |
|------|------|------|
| 某些实验指标不如预期，复现指南只能写"不如 baseline" | 论文价值受损 | 诚实报告；分析差异原因也是贡献 |
| Notebook 依赖外部数据，CI 上跑不通 | 自动化失效 | 提供 synthetic fallback + 预跑好的 HTML 输出 |
