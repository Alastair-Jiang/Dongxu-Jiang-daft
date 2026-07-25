"""Ledoit-Wolf shrunk Markowitz portfolio optimization.

Cross-sectional optimization at each rebalance step:
    max_w  w^T μ - γ w^T Σ_shrunk w
    s.t.   w ≥ 0 (no short), Σ w = 1

The Ledoit-Wolf shrinkage estimator blends the sample covariance matrix
with a structured estimator (constant correlation target), reducing
estimation error in high-dimensional settings (N >> T).

Derived from ml-quant-trading.
"""

import torch


class MarkowitzOptimizer:
    """Mean-variance optimizer with Ledoit-Wolf shrunk covariance.

    Parameters
    ----------
    risk_aversion : float
        γ: risk aversion parameter. Higher γ = more conservative.
    max_weight : float
        Maximum single-position weight.
    use_mosek : bool
        Use MOSEK solver (faster, requires license) vs. CVXPY default.
    """

    def __init__(
        self,
        risk_aversion: float = 1.0,
        max_weight: float = 0.05,
        use_mosek: bool = False,
    ):
        self.risk_aversion = risk_aversion
        self.max_weight = max_weight
        self.use_mosek = use_mosek

    def optimize(
        self,
        expected_returns: torch.Tensor,
        covariance: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute optimal portfolio weights.

        Parameters
        ----------
        expected_returns : torch.Tensor, shape (N,)
            Predicted returns for each asset.
        covariance : torch.Tensor, shape (N, N)
            Shrunk covariance matrix.
        mask : torch.Tensor, shape (N,), bool
            Tradability mask for current time step.

        Returns
        -------
        weights : torch.Tensor, shape (N,)
            Optimal portfolio weights (sum=1, all ≥ 0).
        """
        raise NotImplementedError(
            "MarkowitzOptimizer to be implemented after model signal generation "
            "integration. Will use CVXPY with optional MOSEK backend."
        )
