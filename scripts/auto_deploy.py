"""自动部署器 — 自检通过后自动启动下一批 GPU 计算任务(固定工作流)。

工作流:
  1. 跑 self_check.py(exit 0 才继续, 否则中止)
  2. 按内置任务队列并行启动 GPU 训练任务(并行度可配)
  3. 等待全部完成, 汇总关键指标(IC/t/Sharpe)写 outputs/deploy_summary.json

用法:
  python scripts/auto_deploy.py [--batch next] [--parallel 6] [--dry-run]

任务队列(batch=next, 基于 2026-08-17 研究归档的最新发现):
  128x4 层是 DAFT 历史最强配置(IC 0.032), 本批任务:
  - 补种子: 128x4 × none × {123, 555, 777}(已有 42/7)
  - 深度+消融组合: 128x4 × memory × {42, 7, 123}(消融最优变体 × 深度)
  - 更久训练: 128x4 × none seed42 --full(验证"早停前更多 epoch"是否再提升)
"""
from __future__ import annotations
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import torch

ROOT = Path(__file__).resolve().parent.parent
OUTPUTS = ROOT / "outputs"
PYTHON = sys.executable

# ── 任务队列 ─────────────────────────────────────────────────────────────
BATCHES = {
    "next": [
        # (tag, [args...])
        ("cap128x4_s123", ["scripts/run_full_pipeline_oos.py", "--stocks", "100",
                           "--seed", "123", "--hidden", "128", "--n-layers", "4",
                           "--ckpt-dir", "checkpoints/cap128l4-s123"]),
        ("cap128x4_s555", ["scripts/run_full_pipeline_oos.py", "--stocks", "100",
                           "--seed", "555", "--hidden", "128", "--n-layers", "4",
                           "--ckpt-dir", "checkpoints/cap128l4-s555"]),
        ("cap128x4_s777", ["scripts/run_full_pipeline_oos.py", "--stocks", "100",
                           "--seed", "777", "--hidden", "128", "--n-layers", "4",
                           "--ckpt-dir", "checkpoints/cap128l4-s777"]),
        ("cap128x4_mem42", ["scripts/run_full_pipeline_oos.py", "--stocks", "100",
                            "--seed", "42", "--hidden", "128", "--n-layers", "4",
                            "--ablate", "memory", "--ckpt-dir", "checkpoints/cap128l4-mem42"]),
        ("cap128x4_mem7", ["scripts/run_full_pipeline_oos.py", "--stocks", "100",
                           "--seed", "7", "--hidden", "128", "--n-layers", "4",
                           "--ablate", "memory", "--ckpt-dir", "checkpoints/cap128l4-mem7"]),
        ("cap128x4_mem123", ["scripts/run_full_pipeline_oos.py", "--stocks", "100",
                             "--seed", "123", "--hidden", "128", "--n-layers", "4",
                             "--ablate", "memory", "--ckpt-dir", "checkpoints/cap128l4-mem123"]),
        ("cap128x4_full42", ["scripts/run_full_pipeline_oos.py", "--stocks", "100",
                             "--seed", "42", "--hidden", "128", "--n-layers", "4",
                             "--full", "--ckpt-dir", "checkpoints/cap128l4-full42"]),
    ],
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", default="next", choices=list(BATCHES))
    parser.add_argument("--parallel", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true",
                        help="只列任务不执行")
    args = parser.parse_args()

    # ── 1. 自检 ──────────────────────────────────────────────────────
    print("=== 第 1 步: 自检 ===")
    r = subprocess.run([PYTHON, str(ROOT / "scripts" / "self_check.py")],
                       capture_output=True, text=True, encoding="utf-8",
                       errors="replace")
    if r.returncode != 0:
        print("❌ 自检未通过, 中止部署。修复阻塞项后重跑:")
        for line in r.stdout.splitlines():
            if "❌" in line:
                print("  ", line)
        sys.exit(1)
    print("✅ 自检通过(0 阻塞), 继续部署。")

    tasks = BATCHES[args.batch]
    print(f"\n=== 第 2 步: 部署 {len(tasks)} 个任务(并行度 {args.parallel}) ===")
    for tag, cmd in tasks:
        print(f"  [{tag}] {PYTHON} {' '.join(cmd)}")
    if args.dry_run:
        print("(dry-run, 不执行)")
        return

    # ── 2. 并行启动(已存在的 checkpoint 跳过) ─────────────────────────
    running = []
    done = []
    t0 = time.time()
    skipped = 0
    for tag, cmd in tasks:
        # 跳过规则: ckpt-dir 参数后的目录已含 .pt 文件 → 该任务已跑过
        if "--ckpt-dir" in cmd:
            ckpt = Path(cmd[cmd.index("--ckpt-dir") + 1])
            if ckpt.exists() and any(ckpt.glob("*.pt")):
                print(f"  跳过 [{tag}](checkpoint 已存在: {ckpt})")
                skipped += 1
                continue
        while len(running) >= args.parallel:
            running = [p for p in running if p.poll() is None]
            time.sleep(5)
        log = OUTPUTS / f"deploy_{tag}.log"
        with open(log, "w") as out, open(str(log) + ".err", "w") as err:
            p = subprocess.Popen([PYTHON, "-u", *cmd], cwd=ROOT,
                                 stdout=out, stderr=err)
        running.append(p)
        done.append((tag, p, log))
        print(f"  启动 [{tag}]")

    # ── 3. 等待完成 ─────────────────────────────────────────────────
    for tag, p, log in done:
        p.wait()
        status = "✅" if p.returncode == 0 else f"❌(exit {p.returncode})"
        print(f"  [{tag}] {status}")

    # ── 4. 汇总 ─────────────────────────────────────────────────────
    print("\n=== 第 3 步: 汇总 ===")
    summary = {"batch": args.batch, "tasks": [], "elapsed_s": round(time.time() - t0)}
    for f in sorted(OUTPUTS.glob("EXP-*-daft-oos.json")):
        try:
            rpt = json.load(open(f, encoding="utf-8"))
        except Exception:
            continue
        if rpt.get("hidden") != 128 or rpt.get("n_layers") != 4:
            continue
        oos = rpt.get("out_of_sample", {})
        bt = rpt.get("backtest", {})
        row = {"ablate": rpt.get("ablate"), "seed": rpt.get("seed"),
               "ic": oos.get("ic_mean"), "t": oos.get("ic_t_stat"),
               "sharpe": bt.get("sharpe_ratio")}
        summary["tasks"].append(row)
        print(f"  ablate={row['ablate']:7s} seed={row['seed']}  "
              f"IC={row['ic']:+.4f}  t={row['t']:+.2f}  Sharpe={row['sharpe']:+.4f}")

    out_path = OUTPUTS / "deploy_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n汇总 → {out_path}")


if __name__ == "__main__":
    main()
