"""GPU device auto-detection with multi-backend fallback.

Detection chain (in order of preference):
  1. CUDA      — NVIDIA GPUs
  2. XPU       — Intel Arc / Core Ultra iGPU (PyTorch 2.5+, Python ≤ 3.12)
  3. DirectML  — Any DX12-capable GPU (Intel Arc, AMD, older NVIDIA)
  4. MPS       — Apple Silicon
  5. CPU       — Universal fallback

Usage:
    from daft.utils.device import get_device
    device = get_device()          # auto-detect
    device = get_device("cuda")    # force specific backend
"""

from __future__ import annotations

import logging
import os
import warnings
from typing import Optional

import torch

_logger = logging.getLogger(__name__)

# Module-level cache: only detect once per process.
_cached_device: Optional[torch.device] = None
_cached_backend: Optional[str] = None


def get_device(prefer: str = "auto", verbose: bool = True) -> torch.device:
    """Return the best available torch device.

    Parameters
    ----------
    prefer : str
        One of ``"auto"``, ``"cuda"``, ``"xpu"``, ``"directml"``,
        ``"mps"``, or ``"cpu"``.
    verbose : bool
        If True, log which backend was selected.

    Returns
    -------
    device : torch.device
        A device object ready for ``tensor.to(device)``.
    """
    global _cached_device, _cached_backend

    # Honour explicit preference (non-auto)
    if prefer != "auto":
        device = torch.device(_normalise_name(prefer))
        _apply_cuda_memory_fraction(device)
        if verbose:
            _logger.info("Device: %s (forced)", device)
        return device

    # Return cached result
    if _cached_device is not None:
        return _cached_device

    # --- Detection chain ---
    backend = _detect_backend()
    # 检测名(directml 等)需先归一化为 torch 设备串(privateuseone:0 等)
    device = torch.device(_normalise_name(backend))
    _apply_cuda_memory_fraction(device)

    _cached_device = device
    _cached_backend = backend

    if verbose:
        _logger.info("Device: %s (auto-detected)", device)

    return device


def get_backend_name() -> str:
    """Return the human-readable backend name (e.g. ``"cuda"``, ``"directml"``)."""
    if _cached_backend is not None:
        return _cached_backend
    return _detect_backend()


def device_info() -> dict:
    """Return a dict with device capabilities for logging / config."""
    info = {
        "backend": get_backend_name(),
        "device": str(get_device(verbose=False)),
    }

    backend = info["backend"]
    if backend == "cuda":
        info["cuda_devices"] = torch.cuda.device_count()
        info["cuda_name"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"
    elif backend == "xpu":
        try:
            info["xpu_devices"] = torch.xpu.device_count()
            info["xpu_name"] = torch.xpu.get_device_name(0) if torch.xpu.is_available() else "N/A"
        except Exception:
            pass
    elif backend == "directml":
        try:
            import torch_directml
            info["directml_devices"] = torch_directml.device_count()
        except Exception:
            pass
    elif backend == "mps":
        info["mps_available"] = True

    return info


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _apply_cuda_memory_fraction(device: torch.device) -> None:
    """Limit this process's CUDA memory fraction via DAFT_CUDA_FRACTION env.

    2026-08-17 蓝屏教训: 4 任务并行把物理显存推到 15.5GB 并溢出到 Windows
    共享内存导致系统崩溃。设置 DAFT_CUDA_FRACTION=0.42 可把单进程显存硬
    封顶(0.42×16GB≈6.9GB), 2 任务并行合计 ~13.7GB < 15GB 可用线; 超限时
    PyTorch 会干净地报 CUDA OOM 而不是溢出到共享内存。
    """
    raw = os.environ.get("DAFT_CUDA_FRACTION", "0") or "0"
    try:
        frac = float(raw)
    except ValueError:
        return
    if frac > 0 and device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(frac, device.index or 0)
        _logger.info("CUDA memory fraction capped at %.2f (DAFT_CUDA_FRACTION)", frac)


def _detect_backend() -> str:
    """Run the detection chain and return the best backend name string."""
    # 1. CUDA
    if torch.cuda.is_available():
        return "cuda"

    # 2. Intel XPU (Arc / Core Ultra iGPU)
    #    Requires PyTorch 2.5+ with XPU backend:
    #    pip install torch --index-url https://download.pytorch.org/whl/xpu
    if _check_xpu():
        return "xpu"

    # 3. DirectML (any DX12 GPU: Intel Arc, AMD, older NVIDIA)
    #    pip install torch-directml
    if _check_directml():
        # torch-directml expects device string "privateuseone:0" or similar;
        # we normalise to "directml" and map at use time.
        return "directml"

    # 4. Apple MPS
    if torch.backends.mps.is_available():
        return "mps"

    # 5. CPU fallback
    return "cpu"


def _check_xpu() -> bool:
    """Return True if Intel XPU backend is available."""
    try:
        return torch.xpu.is_available()
    except (AttributeError, RuntimeError):
        return False


def _check_directml() -> bool:
    """Return True if torch-directml is installed and has ≥ 1 device."""
    try:
        import torch_directml
        return torch_directml.device_count() > 0
    except (ImportError, RuntimeError):
        return False


def _normalise_name(name: str) -> str:
    """Map user-friendly names to internal torch device strings."""
    mapping = {
        "cuda":     "cuda",
        "xpu":      "xpu",
        "directml": _directml_device_string(),
        "mps":      "mps",
        "cpu":      "cpu",
        "auto":     _detect_backend(),
    }
    return mapping.get(name.lower(), name)


def _directml_device_string() -> str:
    """Return the torch-directml device string (typically 'privateuseone:0')."""
    try:
        import torch_directml
        return str(torch_directml.device())   # e.g. 'privateuseone:0'
    except ImportError:
        # If torch-directml is requested but not installed, we still return a
        # string that will produce a clear error message from torch.device().
        return "privateuseone:0"
