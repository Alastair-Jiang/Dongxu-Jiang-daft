"""Financial ML metrics: Rank IC, ICIR, and related evaluation utilities.

All functions operate on torch.Tensor and are GPU-compatible.
"""

from __future__ import annotations
from typing import Dict, Optional, Union

import torch


def rank_info_coefficient(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    per_timestep: bool = False,
) -> Union[float, torch.Tensor]:
    """Cross-sectional Spearman rank Information Coefficient.

    At each timestep t, computes the Pearson correlation between the
    cross-sectional ranks of predictions[t] and targets[t].

    Parameters
    ----------
    predictions : (T, N) or (B,) or (B, N)
        Model-predicted signals.
    targets : same shape as predictions
        Actual forward returns.
    mask : same shape as predictions, bool, optional
        True = valid observation. Defaults to all-true.
    per_timestep : bool
        If True, return (T,) tensor of per-timestep IC values.
        If False, return scalar mean IC.

    Returns
    -------
    ic : float or torch.Tensor
        Mean IC or per-timestep IC series.
    """
    if mask is None:
        mask = torch.ones_like(predictions, dtype=torch.bool)

    # --- 2D case: (T, N) cross-sectional ---
    if predictions.ndim == 2:
        return _rank_ic_2d(predictions, targets, mask, per_timestep)

    # --- 1D case: (B,) batch ---
    return _rank_ic_1d(predictions, targets, mask, per_timestep)


def ic_summary(ic_series: torch.Tensor) -> Dict[str, float]:
    """Compute IC statistics from a per-timestep IC series.

    Parameters
    ----------
    ic_series : (T,) tensor of per-timestep IC values.

    Returns
    -------
    stats : dict with keys:
        ic_mean, ic_std, icir (= mean/std), ic_positive_ratio,
        ic_t_stat (= mean * sqrt(T) / std)
    """
    valid = ic_series[ic_series.isfinite()]
    if valid.numel() == 0:
        return {
            "ic_mean": 0.0, "ic_std": 0.0, "icir": 0.0,
            "ic_positive_ratio": 0.0, "ic_t_stat": 0.0,
        }

    mu = valid.mean().item()
    sd = valid.std().item() if valid.numel() > 1 else 1.0

    return {
        "ic_mean": mu,
        "ic_std": sd,
        "icir": mu / sd if sd > 0 else 0.0,
        "ic_positive_ratio": (valid > 0).float().mean().item(),
        "ic_t_stat": mu * (valid.numel() ** 0.5) / sd if sd > 0 else 0.0,
    }


def hit_rate(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> float:
    """Fraction of observations where sign(pred) == sign(target).

    Parameters
    ----------
    predictions : (T, N) or (B,)
    targets : same shape
    mask : same shape, bool, optional

    Returns
    -------
    rate : float, in [0, 1]
    """
    if mask is None:
        mask = torch.ones_like(predictions, dtype=torch.bool)

    pred_sign = torch.sign(predictions)
    target_sign = torch.sign(targets)
    match = (pred_sign == target_sign) & mask & (target_sign != 0)

    total = (mask & (target_sign != 0)).sum()
    if total == 0:
        return 0.5
    return (match.sum() / total).item()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _rank_ic_2d(
    predictions: torch.Tensor,  # (T, N)
    targets: torch.Tensor,      # (T, N)
    mask: torch.Tensor,         # (T, N) bool
    per_timestep: bool,
) -> Union[float, torch.Tensor]:
    """Per-timestep cross-sectional rank IC."""
    T, N = predictions.shape
    ic_vals = torch.zeros(T, device=predictions.device)

    for t in range(T):
        p_t = predictions[t]   # (N,)
        r_t = targets[t]
        m_t = mask[t]

        valid = m_t.sum()
        if valid < 3:
            ic_vals[t] = 0.0
            continue

        # Masked elements get sent to extremes (won't affect ranks of valid)
        p_ranked = _masked_rank(p_t, m_t)
        r_ranked = _masked_rank(r_t, m_t)

        # Pearson r on ranks = Spearman rho
        ic_vals[t] = _pearson_r(p_ranked, r_ranked, m_t)

    if per_timestep:
        return ic_vals
    return ic_vals[ic_vals != 0].mean().item() if ic_vals.any() else 0.0


def _rank_ic_1d(
    predictions: torch.Tensor,  # (B,) or (B, N)
    targets: torch.Tensor,
    mask: torch.Tensor,
    per_timestep: bool,
) -> Union[float, torch.Tensor]:
    """Batch-mode Pearson correlation (differentiable proxy for rank IC)."""
    p = predictions.reshape(-1)
    r = targets.reshape(-1)
    m = mask.reshape(-1)

    valid = m.sum()
    if valid < 3:
        return torch.tensor(0.0) if per_timestep else 0.0

    ic = _pearson_r(p, r, m)
    if per_timestep:
        return ic.unsqueeze(0)
    return ic.item()


def _masked_rank(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Compute fractional rank [0, 1] for valid elements, 0 for masked.

    Parameters
    ----------
    x : (N,)
    mask : (N,) bool

    Returns
    -------
    ranks : (N,) float in [0, 1]
    """
    x_masked = x.clone()
    x_masked[~mask] = float("inf")   # push masked to the end
    sorted_idx = x_masked.argsort()
    ranks = torch.zeros_like(x)
    ranks[sorted_idx] = torch.arange(len(x), device=x.device, dtype=torch.float32)
    n_valid = mask.sum().clamp(min=1)
    ranks = ranks / (n_valid - 1).clamp(min=1)  # normalise to [0, 1]
    ranks[~mask] = 0.0
    return ranks


def _pearson_r(x: torch.Tensor, y: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Masked Pearson correlation between two 1D tensors."""
    x_m = x * mask.float()
    y_m = y * mask.float()
    n = mask.sum().clamp(min=1)

    mx = x_m.sum() / n
    my = y_m.sum() / n
    dx = (x - mx) * mask.float()
    dy = (y - my) * mask.float()

    cov = (dx * dy).sum()
    sx = (dx * dx).sum().sqrt().clamp(min=1e-8)
    sy = (dy * dy).sum().sqrt().clamp(min=1e-8)

    return (cov / (sx * sy)).clamp(-1.0, 1.0)
