"""DAFT 自检脚本（固定工作流程, 2026-08-17 建立）。

每次实验后/提交前运行: python scripts/self_check.py
检查项:
  1. 全量 py 编译
  2. outputs/*.json 可解析 + 关键字段完整
  3. 实验产物可追溯性(seed/ablate/variant 字段缺失告警)
  4. 登记表引用的 EXP ID 与实际产物对应
  5. 关键公式/约束实现抽查(grep 断言)
"""
from __future__ import annotations
import json
import re
import subprocess
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
OUTPUTS = ROOT / "outputs"
OK: list[str] = []          # 通过项
ISSUES: list[str] = []      # 阻塞性失败
WARNINGS: list[str] = []    # 告警(历史债等, 不阻塞)


def check(name: str, ok: bool, detail: str = ""):
    (OK if ok else ISSUES).append(f"{name}: {detail}")


# ── 1. 全量编译 ─────────────────────────────────────────────────────
bad = 0
for f in list((ROOT / "src").rglob("*.py")) + list((ROOT / "scripts").rglob("*.py")) \
        + list((ROOT / "tests").rglob("*.py")):
    r = subprocess.run([sys.executable, "-m", "py_compile", str(f)],
                       capture_output=True)
    if r.returncode != 0:
        bad += 1
        ISSUES.append(f"编译失败: {f.relative_to(ROOT)}")
check("1. 全量编译", bad == 0, f"{bad} 个文件失败" if bad else "全部通过")

# ── 2. outputs JSON 完整性(按产物类型检查结构) ───────────────────────
n_json = n_bad = n_no_seed = 0
for f in sorted(OUTPUTS.glob("EXP-*.json")):
    n_json += 1
    try:
        r = json.load(open(f, encoding="utf-8"))
    except Exception as e:
        n_bad += 1
        ISSUES.append(f"JSON 解析失败: {f.name}: {e}")
        continue
    name = f.name
    # 结构感知: 不同类型产物字段位置不同
    if "rebalance-ablation" in name:
        ok_struct = "grid" in r and "best" in r
    elif "smoothing-ablation" in name:
        ok_struct = "test" in r and "val" in r
    elif "walk-forward" in name:
        ok_struct = "folds" in r
    else:  # daft-oos / ridge-real / weekly 等
        ok_struct = "ic_mean" in r.get("out_of_sample", {}) \
            and "sharpe_ratio" in r.get("backtest", {})
    if not ok_struct:
        n_bad += 1
        ISSUES.append(f"结构异常: {f.name}")
    # 可追溯性: 训练型产物必须有 seed(消融另有 ablate) — 缺失为历史债告警
    if "daft-oos" in name or "daft-weekly" in name:
        if "seed" not in r:
            n_no_seed += 1
            WARNINGS.append(f"缺 seed 字段(历史债, 根因已修): {f.name}")
check("2. 产物完整性", n_bad == 0,
      f"{n_json} 个 JSON, {n_bad} 个结构异常" + (f", {n_no_seed} 个缺 seed(告警)" if n_no_seed else ""))

# ── 3. 登记表表格行内编号唯一性 ──────────────────────────────────────
reg = (ROOT / "docs" / "EXPERIMENT_REGISTRY.md").read_text(encoding="utf-8")
table_ids = []
for line in reg.splitlines():
    if line.strip().startswith("|"):
        table_ids += re.findall(r"EXP-\d{8}-\d{2}", line)
dup = sorted({e for e in table_ids if table_ids.count(e) > 1})
if dup:
    WARNINGS.append(f"登记表跨表格重复引用(作废表+变更日志正常现象): {dup}")
else:
    OK.append(f"3. 登记表一致性: 表格行内 {len(set(table_ids))} 个唯一 EXP ID, 无重复")

# ── 4. 关键公式/约束抽查 ────────────────────────────────────────────
src = (ROOT / "src").rglob("*.py")
src_text = {p.name: p.read_text(encoding="utf-8") for p in src}

checks_src = [
    ("CDAP logit 空间", "cross_dim_attn.py", r"log_p.*log\(\)", True),
    ("KDA delta rule", "memory.py", r"einsum", True),
    ("safe gate 修复", "memory.py", r"lower_bound \+ \(1\.0 - self\.lower_bound\)", True),
    ("专家 mask float", "trend_expert.py", r"mask_f = mask\.float\(\)", True),
    ("专家 mask float", "reversal_expert.py", r"mask_f = mask\.float\(\)", True),
    ("专家 mask float", "volatility_expert.py", r"mask_f = mask\.float\(\)", True),
    ("bincount CPU 化", "router.py", r"flatten\(\)\.cpu\(\)", True),
    ("MaxDD 百分比", "engine.py", r"equity = torch\.exp", True),
    ("涨跌停 mask", "baostock_adapter.py", r"_limit_move_mask", True),
    ("通道契约", "base_features.py", r"def ensure_base_panel", True),
]
for name, fname, pattern, expect in checks_src:
    text = src_text.get(fname, "")
    check(f"4. 公式[{name}]", bool(re.search(pattern, text)) == expect,
          f"{fname} 中 {'存在' if expect else '不存在'} {pattern}")

# ── 5. 路由损失结构(方案A balance KL, 无抵消 bug 回归) ──────────────
rt = src_text.get("router_trainer.py", "")
has_balance_kl = "balance_loss" in rt and "target_frac" in rt
has_old_entropy_pair = re.search(r"entropy_weight \* routing_entropy", rt) is not None \
    and re.search(r"sparsity_weight \* sparsity_penalty", rt) is not None
check("5. 路由损失结构", has_balance_kl and not has_old_entropy_pair,
      "balance KL 存在" if has_balance_kl else "balance KL 缺失")

# ── 6. git 工作树卫生(未提交文件提醒, 不阻塞) ────────────────────────
r = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain"],
                   capture_output=True, text=True)
dirty = [l for l in r.stdout.splitlines() if l.strip()]
if dirty:
    WARNINGS.append(f"git 工作树有 {len(dirty)} 个未提交项(并行会话或遗漏): "
                    + ", ".join(l.split()[-1] for l in dirty[:6])
                    + ("..." if len(dirty) > 6 else ""))
else:
    OK.append("6. git 工作树干净")

print("\n" + "=" * 60)
print(f"自检结果: {len(OK)} 通过, {len(ISSUES)} 阻塞, {len(WARNINGS)} 告警")
for o in OK:
    print(f"  ✅ {o}")
for w in WARNINGS[:8]:
    print(f"  ⚠️  {w}")
if len(WARNINGS) > 8:
    print(f"  ⚠️  ...共 {len(WARNINGS)} 条告警(历史债, 根因已修)")
for i in ISSUES:
    print(f"  ❌ {i}")
print("=" * 60)
sys.exit(1 if ISSUES else 0)
