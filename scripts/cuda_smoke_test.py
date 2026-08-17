"""CUDA smoke test — 验证 DAFT 模型在 RTX 5060 Ti 上前向+反向可行。

用法:
  python scripts/cuda_smoke_test.py
"""
from __future__ import annotations
import sys
import time
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from daft.models.factory import build_model
from daft.utils.device import get_device


def main() -> None:
    device = get_device()
    assert device.type == "cuda", f"expected cuda, got {device}"
    props = torch.cuda.get_device_properties(0)
    print(f"Device: {device}  {torch.cuda.get_device_name(0)}  "
          f"VRAM={props.total_memory / 2**30:.1f}GB  sm={props.major}.{props.minor}")

    model, layer_proj = build_model()
    model = model.to(device)
    layer_proj = layer_proj.to(device)

    N, D = 512, 200
    s_t = torch.randn(N, D, device=device)
    target = torch.randn(N, 1, device=device)

    def step():
        l = [layer_proj[k](s_t) for k in ("l0", "l1", "l2")]
        out = model(s_t, l, mode="train")
        loss = ((out["signal"] - target) ** 2).mean()
        loss.backward()
        model.memory.detach_state()  # 截断 BPTT，与 router_trainer 第 405 行一致
        return out, loss

    # warmup
    step()
    model.zero_grad(set_to_none=True)

    # benchmark fwd+bwd
    torch.cuda.synchronize()
    n_iter = 20
    t0 = time.time()
    for _ in range(n_iter):
        step()
        model.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    dt = (time.time() - t0) / n_iter

    # final grad sanity check
    out, loss = step()
    grad_nan = any(
        p.grad is not None and not torch.isfinite(p.grad).all()
        for p in model.parameters() if p.requires_grad
    )

    print(f"fwd+bwd      : {dt * 1000:.1f} ms/iter @ batch={N}")
    print(f"signal mean  : {out['signal'].mean().item():+.4f}")
    print(f"loss         : {loss.item():.4f}")
    print(f"routing_probs: {out['routing_probs'].shape}")
    print(f"grad NaN     : {grad_nan}")
    print(f"VRAM used    : {torch.cuda.memory_allocated() / 2**30:.2f} GB / "
          f"{torch.cuda.memory_reserved() / 2**30:.2f} GB reserved")
    print("CUDA SMOKE TEST PASSED" if not grad_nan else "CUDA SMOKE TEST FAILED (grad NaN)")


if __name__ == "__main__":
    main()
