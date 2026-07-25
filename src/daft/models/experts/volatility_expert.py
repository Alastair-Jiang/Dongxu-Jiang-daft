"""Volatility regime expert.

Trained on periods with elevated volatility (ATR above rolling 80th percentile).
Loss = volatility forecast MSE + directional hedging signal.

Inspired by K3's multi-expert design: each expert captures a different
facet of the input distribution.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from daft.models.experts.base_expert import BaseExpert


class VolatilityExpert(BaseExpert):
    """Expert specialized in volatility regimes and tail-risk hedging.

    Competence region: periods with ATR above rolling 80th percentile,
    VIX spikes, or clustered high-volatility episodes.
    Output: volatility-adjusted signal (reduced position sizing in high-vol).
    """

    def __init__(self, input_dim=200, hidden_dim=48, n_layers=2):
        super().__init__(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            name="volatility",
        )

    def _regime_filter(self, panel) -> torch.Tensor:
        """Select high-volatility regime samples.

        Filter: rolling volatility above 80th percentile.
        """
        raise NotImplementedError(
            "Regime filter requires computed regime features. "
            "Will be implemented after Feature Engine integration."
        )

    def compute_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Volatility expert loss: MSE on returns + regularization on variance.

        Penalizes overconfidence in high-volatility environments.

        L = MSE(pred, target) + λ · Var(pred)
        """
        pred = pred.squeeze(-1)
        target = target.squeeze(-1)
        mask = mask.squeeze(-1)

        mse = ((target - pred) ** 2 * mask).sum() / mask.sum().clamp(min=1)

        # Variance penalty: discourage extreme predictions in volatile regimes
        pred_var = ((pred - pred.mean()) ** 2 * mask).sum() / mask.sum().clamp(min=1)
        lambda_reg = 0.01

        return mse + lambda_reg * pred_var
