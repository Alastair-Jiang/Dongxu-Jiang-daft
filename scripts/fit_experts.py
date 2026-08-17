"""专家层拟合实验启动器 — README regime 专业化 vs 全量训练 (4 任务并行)。"""
import subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = r"D:\env\python.exe"

tasks = [
    ("reg-mem", ["--ablate", "memory", "--ckpt-dir", "checkpoints/fit-reg-mem"]),
    ("all-mem", ["--ablate", "memory", "--no-regime", "--ckpt-dir", "checkpoints/fit-all-mem"]),
    ("reg-none", ["--ckpt-dir", "checkpoints/fit-reg-none"]),
    ("all-none", ["--no-regime", "--ckpt-dir", "checkpoints/fit-all-none"]),
]

common = ["-u", "scripts/run_full_pipeline_oos.py", "--stocks", "100",
          "--seed", "42", "--hidden", "128", "--n-layers", "4"]

procs = []
for tag, extra in tasks:
    log = open(ROOT / "outputs" / f"fit_{tag}.log", "w")
    err = open(ROOT / "outputs" / f"fit_{tag}.err", "w")
    p = subprocess.Popen([PY, *common, *extra], cwd=ROOT, stdout=log, stderr=err)
    procs.append((tag, p, log, err))
    print(f"started {tag} pid={p.pid}")

for tag, p, log, err in procs:
    p.wait()
    log.close(); err.close()
    print(f"{tag}: exit={p.returncode}")

print("=== expert fitting done ===")
