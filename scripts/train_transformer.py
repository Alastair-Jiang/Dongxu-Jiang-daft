"""Transformer 专家架构实验启动器 v2 (2026-08-17, RTX 5060 Ti 16GB)。

v1 教训: 4 任务并行把显存推到 15.5GB 并溢出共享内存 → 蓝屏。
v2 措施:
  - 每进程 DAFT_CUDA_FRACTION 硬封顶显存(device.py 读取该环境变量),
    超限时干净 CUDA OOM 而非溢出;
  - 每波最多 2 任务: 波1 tf100_128x4 + tf300_128x4 (0.42×16G≈6.9G 各),
    波2 tf100_256x8 单独跑 (0.8×16G≈13G);
  - expandable_segments 减少碎片化 OOM。

对照设计 (全部 seed=42 / memory-ablate / 全量训练):
  tf100_128x4 : Transformer 100 股 hidden=128 4 块 4 头 ← 主对照
                 (MLP 锚点已同批完成: EXP-20260817-51, IC +0.0331)
  tf300_128x4 : Transformer 300 股 hidden=128 4 块 4 头 ← 规模放大
  tf100_256x8 : Transformer 100 股 hidden=256 4 块 8 头 ← 容量扫描
"""
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = r"D:\env\python.exe"

BASE = ["-u", "scripts/run_full_pipeline_oos.py",
        "--seed", "42", "--ablate", "memory", "--no-regime"]

WAVE1 = [
    ("tf100_128x4", BASE + ["--stocks", "100", "--arch", "transformer",
                            "--hidden", "128", "--n-layers", "4", "--n-heads", "4",
                            "--ckpt-dir", "checkpoints/tf100_128x4"], 0.45),
    ("tf300_128x4", BASE + ["--stocks", "300", "--arch", "transformer",
                            "--hidden", "128", "--n-layers", "4", "--n-heads", "4",
                            "--ckpt-dir", "checkpoints/tf300_128x4"], 0.45),
]
WAVE2 = [
    ("tf100_256x8", BASE + ["--stocks", "100", "--arch", "transformer",
                            "--hidden", "256", "--n-layers", "4", "--n-heads", "8",
                            "--ckpt-dir", "checkpoints/tf100_256x8"], 0.85),
]
# 256×8 Stage3 显存可能超 13.5GB → 失败时自动降级 192×8
WAVE2_FALLBACK = [
    ("tf100_192x8", BASE + ["--stocks", "100", "--arch", "transformer",
                            "--hidden", "192", "--n-layers", "4", "--n-heads", "8",
                            "--ckpt-dir", "checkpoints/tf100_192x8"], 0.85),
]


def run_wave(wave, wave_no):
    print(f"--- wave {wave_no}: {len(wave)} task(s) ---")
    procs = []
    for tag, argv, frac in wave:
        env = dict(os.environ)
        env["DAFT_CUDA_FRACTION"] = str(frac)
        env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        log = open(ROOT / "outputs" / f"tf_{tag}.log", "w")
        err = open(ROOT / "outputs" / f"tf_{tag}.err", "w")
        p = subprocess.Popen([PY, *argv], cwd=ROOT, stdout=log, stderr=err, env=env)
        procs.append((tag, p, log, err))
        print(f"  started {tag} pid={p.pid} frac={frac}")
    results = {}
    for tag, p, log, err in procs:
        p.wait()
        log.close()
        err.close()
        results[tag] = p.returncode
        print(f"  {tag}: exit={p.returncode}")
    return results


if __name__ == "__main__":
    run_wave(WAVE1, 1)
    res = run_wave(WAVE2, 2)
    for tag, rc in res.items():
        if rc != 0:
            print(f"  [{tag}] 失败(exit={rc}) → 降级 192×8")
            run_wave(WAVE2_FALLBACK, 3)
            break
    print("=== transformer sweep v3 done ===")
