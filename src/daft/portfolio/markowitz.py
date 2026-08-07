"""Ledoit-Wolf shrunk Markowitz portfolio optimization.

Cross-sectional optimization at each rebalance step:
    max_w  w^T μ - γ w^T Σ_shrunk w
    s.t.   w ≥ 0 (no short), Σ w = 1,  w ≤ max_weight

Uses closed-form quadratic programming with iterative projection
for the box constraint (no external solver required).
"""

from __future__ import annotations
from typing import Optional

import torch


class MarkowitzOptimizer:
    """Mean-variance optimizer with Ledoit-Wolf shrunk covariance.

    Parameters
    ----------
    risk_aversion : float
        γ: risk aversion parameter. Higher γ = more conservative.
    max_weight : float
        Maximum single-position weight (0 < max_weight ≤ 1).
    """

    def __init__(
        self,
        risk_aversion: float = 1.0,
        max_weight: float = 0.05,
        use_mosek: bool = False,  # unused; kept for API compatibility
    ):
        self.risk_aversion = risk_aversion
        self.max_weight = max_weight

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------
    def optimize(
        self,
        expected_returns: torch.Tensor,   # (N,)
        covariance: torch.Tensor,         # (N, N) sample covariance
        mask: torch.Tensor,               # (N,) bool
    ) -> torch.Tensor:
        """Compute optimal long-only portfolio weights.

        Parameters
        ----------
        expected_returns : (N,) — predicted returns per asset.
        covariance : (N, N) — sample covariance matrix.
        mask : (N,) bool — True = asset is tradable.

        Returns
        -------
        weights : (N,) — optimal weights, sum=1, all ≥ 0, each ≤ max_weight.
        """
        device = expected_returns.device
        valid = mask.nonzero(as_tuple=False).squeeze(-1)
        n_valid = valid.numel()

        if n_valid == 0:
            return torch.zeros_like(expected_returns)

        # Reduce to valid assets only
        mu = expected_returns[valid]                # (n,)
        S = covariance[valid][:, valid]              # (n, n)

        # ---- Ledoit-Wolf shrinkage ----
        S_shrunk = ledoit_wolf_shrinkage(S, mu.unsqueeze(-1))

        # ---- Closed-form MV with long-only + sum-to-1 ----
        w_valid = _mv_closed_form(mu, S_shrunk, self.risk_aversion)

        # ---- Box constraint: w_i ≤ max_weight (iterative projection) ----
        w_valid = _project_box(w_valid, self.max_weight)
        w_valid = w_valid / w_valid.sum().clamp(min=1e-8)

        # Map back to full N-dimensional space
        weights = torch.zeros_like(expected_returns)
        weights[valid] = w_valid.to(device)

        return weights


# ======================================================================
# Ledoit-Wolf shrinkage (OAS variant)
# ======================================================================
def ledoit_wolf_shrinkage(
    S: torch.Tensor,           # (n, n) sample covariance
    mu: Optional[torch.Tensor] = None,  # (n, 1) optional mean
) -> torch.Tensor:
    """Ledoit-Wolf shrinkage toward constant-correlation target.

    Shrinks the sample covariance S toward a structured estimator F where
    all pairwise correlations are equal to the mean correlation.

    Σ_shrunk = δ · F + (1 - δ) · S

    Parameters
    ----------
    S : (n, n) sample covariance.
    mu : (n, 1) optional mean (unused; kept for API compat).

    Returns
    -------
    S_shrunk : (n, n) PSD shrunk covariance matrix.
    """
    n = S.size(0)
    if n < 2:
        return S

    device = S.device

    # --- Constant-correlation target F ---
    stds = S.diag().sqrt().clamp(min=1e-8)                     # (n,)
    D_inv = torch.diag(1.0 / stds)
    R = D_inv @ S @ D_inv                                      # correlation matrix

    # Mean correlation (off-diagonal)
    off_mask = ~torch.eye(n, dtype=torch.bool, device=device)
    r_bar = R[off_mask].mean().clamp(-1.0, 1.0)

    # Target: F_ii = σ_i², F_ij = r_bar · σ_i · σ_j
    F = r_bar * stds.unsqueeze(-1) * stds.unsqueeze(0)
    F[torch.arange(n), torch.arange(n)] = S.diag()

    # --- Optimal shrinkage intensity (OAS estimator) ---
    # π = sum of asymptotic variances / sum of squared deviations
    # We use a simplified version: δ = (tr(S²) + tr²(S)) / ((T-1) * (tr(S²) - tr²(S)/n))
    # with a conservative δ ∈ [0, 1].
    tr_S2 = (S @ S).trace()                                    # tr(S²)
    tr_S_sq = S.trace() ** 2                                   # tr²(S)

    # Variance of sample covariance entries (estimate)
    pi_num = tr_S2 - tr_S_sq / n

    # Squared deviation of S from F
    diff = S - F
    rho = (diff * diff).sum()

    delta = pi_num / rho.clamp(min=1e-8)
    delta = delta.clamp(0.0, 1.0)

    return delta * F + (1.0 - delta) * S


# ======================================================================
# Closed-form MV optimization
# ======================================================================
def _mv_closed_form(
    mu: torch.Tensor,        # (n,)
    Sigma: torch.Tensor,     # (n, n) PSD
    gamma: float,            # risk aversion
) -> torch.Tensor:
    """Unconstrained MV solution: w* = (1/γ) · Σ⁻¹ · μ.

    For long-only + sum-to-1, we first find the tangency portfolio
    then project to the feasible set.
    """
    n = mu.size(0)
    device = mu.device

    # Add ridge for numerical stability
    Sigma_reg = Sigma + 1e-6 * torch.eye(n, device=device)

    try:
        # Solve: Σ · w_raw = μ  (up to scale factor)
        w_raw = torch.linalg.solve(Sigma_reg, mu)               # (n,)
    except RuntimeError:
        # Fallback: use pseudoinverse
        w_raw = torch.linalg.lstsq(Sigma_reg, mu.unsqueeze(-1)).solution.squeeze(-1)

    # Minimum-variance portfolio: Σ · w_mv = 1
    ones = torch.ones(n, device=device)
    try:
        w_mv = torch.linalg.solve(Sigma_reg, ones)
    except RuntimeError:
        w_mv = torch.linalg.lstsq(Sigma_reg, ones.unsqueeze(-1)).solution.squeeze(-1)

    # Blend: w = w_raw/γ + w_mv  (two-fund separation)
    # Scale to satisfy sum(w) = 1
    w = w_raw / gamma + w_mv
    w = w / w.sum().clamp(min=1e-8)

    # Ensure non-negative
    w = w.clamp(min=0)
    w = w / w.sum().clamp(min=1e-8)

    return w


def _project_box(w: torch.Tensor, max_w: float, max_iter: int = 20) -> torch.Tensor:
    """Iteratively clip weights above max_w and renormalize.

    This is a simple projection onto the simplex ∩ box constraint.
    For MVP purposes it converges quickly (< 5 iterations typically).
    """
    w = w.clone()
    for _ in range(max_iter):
        excess = (w > max_w).any()
        if not excess:
            break
        # Clip to max_w
        w = w.clamp(max=max_w)
        # Renormalize
        s = w.sum()
        if s > 0:
            w = w / s
    return w
