"""Trend-following expert.

Trained on periods with sustained directional movement (ADX > 25).
Loss = directional accuracy + Sharpe ratio of positions.

Inspired by K3's Stable LatentMoE expert specialization philosophy:
each expert has a clearly defined competence region, and the router
learns to assign inputs to the right specialist.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from daft.models.experts.base_expert import BaseExpert


class TrendExpert(BaseExpert):
    """Expert specialized in trend-following strategies.

    Competence region: periods with ADX > 25 and sustained directional movement.
    Output: expected return direction and magnitude for trend continuation.
    """

    def __init__(self, input_dim=200, hidden_dim=64, n_layers=2):
        super().__init__(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            name="trend",
        )

    def _regime_filter(self, panel) -> torch.Tensor:
        """Select trend regime timesteps: mean cross-sectional ADX > 25.

        Computes Wilder's ADX (14-period) per asset, then aggregates
        across assets via mean. Returns True for time steps where the
        average ADX exceeds 25 — indicating sustained directional movement.
        """
        from daft.models.experts.base_expert import _compute_adx_mask

        return _compute_adx_mask(panel, threshold=25.0, above=True)

    def compute_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Trend expert loss: direction-weighted MSE.

        Penalizes wrong-sign predictions 11x more than magnitude errors,
        following the Adjusted-MSE approach from ml-quant-trading.

        L = Σ masked [ (y - ŷ)² · (1 + 10 · 1[sign(ŷ) ≠ sign(y)]) ]
        """
        pred = pred.squeeze(-1)
        target = target.squeeze(-1)
        mask = mask.squeeze(-1)

        se = (target - pred) ** 2

        # Heavier penalty for directional errors
        sign_mismatch = (torch.sign(pred) != torch.sign(target)).float()
        adjusted_se = se * (1.0 + 10.0 * sign_mismatch)

        # Mean over valid (masked) elements
        mask_f = mask.float()  # DirectML 上 bool.sum() 返回 bool, 必须转 float
        loss = (adjusted_se * mask_f).sum() / mask_f.sum().clamp(min=1)
        return loss
