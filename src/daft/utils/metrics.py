"""Financial ML metrics: Rank IC, ICIR, and related evaluation utilities.

All functions operate on torch.Tensor and are GPU-compatible.
"""

from __future__ import annotations
from typing import Dict, Optional, Union

import torch


def eligible_mask(mask: torch.Tensor) -> torch.Tensor:
    """双条件入样 mask (A4, 2026-08-18): mask[t] AND mask[t+1]。

    K3 纲领 A4 统一的对决口径: 信号日 t 可交易(能建仓)且收益实现日
    t+1 可交易(收益真实)的样本才入样。修复前 DAFT 侧评估用
    mask[t+1] 单条件、Ridge 侧入样/评估也用 mask[t+1] 单条件——
    涨停日(t 停)但次日复牌的样本被计入 IC, 高估可交易信号预测力;
    停牌恢复日两侧处理不对称。

    与 BacktestEngine.run 内部的 ret_mask = mask[:-1] & mask[1:]
    完全同一公式 —— 评估层与回测层口径对齐。

    Parameters
    ----------
    mask : (T, N) bool
        逐日可交易 mask。

    Returns
    -------
    (T-1, N) bool
        第 t 行 = mask[t] & mask[t+1], 对齐"信号日 t 的信号预测
        t→t+1 收益"的 (signal, return) 配对序列。
    """
    if not mask.dtype == torch.bool:
        mask = mask.bool()
    return mask[:-1] & mask[1:]


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


def rank_ic_by_timestep(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    t_idx: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """按时间步分组的截面 rank IC (2026-08-16 新增)。

    训练期验证的正确 IC: 展平样本按原始时间步分组, 每个时间步在截面
    (各资产)上算 Spearman rank IC, 返回 (T,) 序列供 ic_summary 汇总。
    修复了旧实现把展平样本池化成单个 Pearson 相关、ICIR/t-stat 退化
    (恒等于 IC)的问题。

    Parameters
    ----------
    predictions, targets : (K,) 展平的预测与目标
    t_idx : (K,) long 每行所属的原始时间步索引
    mask : (K,) bool, optional 默认全有效

    Returns
    -------
    ic_vals : (T',) 每个时间步的截面 rank IC(仅含有效样本数 ≥3 的步)
    """
    predictions = predictions.reshape(-1)
    targets = targets.reshape(-1)
    t_idx = t_idx.reshape(-1).long()
    if mask is None:
        mask = torch.ones_like(predictions, dtype=torch.bool)
    else:
        mask = mask.reshape(-1)

    device = predictions.device
    ic_vals = []
    for t in t_idx.unique().tolist():
        sel = t_idx == t
        m_t = mask[sel]
        if m_t.sum() < 3:
            continue
        p_ranked = _masked_rank(predictions[sel], m_t)
        r_ranked = _masked_rank(targets[sel], m_t)
        ic_vals.append(_pearson_r(p_ranked, r_ranked, m_t))

    if not ic_vals:
        return torch.zeros(0, device=device)
    return torch.stack(ic_vals)


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
    valid_t = torch.zeros(T, dtype=torch.bool, device=predictions.device)

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
        valid_t[t] = True

    if per_timestep:
        return ic_vals
    # 修复(2026-08-16): 旧实现 ic_vals[ic_vals != 0] 会把 IC 恰好为 0 的
    # 交易日丢掉, 与 ic_summary(保留 0)口径不一致。改为按"有效时间步"平均。
    return ic_vals[valid_t].mean().item() if valid_t.any() else 0.0


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
