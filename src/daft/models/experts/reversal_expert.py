"""Mean-reversion expert.

Trained on periods with low ADX (oscillating within bands).
Loss = rank IC (Information Coefficient) of predicted returns.

In multi-regime markets, the reversion expert captures the tendency
of prices to revert to the mean after short-term deviations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from daft.models.experts.base_expert import BaseExpert


class ReversalExpert(BaseExpert):
    """Expert specialized in mean-reversion strategies.

    Competence region: periods with ADX < 20, oscillating within Bollinger Bands.
    Output: expected return for mean reversion (sign opposite to recent deviation).
    """

    def __init__(self, input_dim=200, hidden_dim=64, n_layers=2):
        super().__init__(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            name="reversal",
        )

    def _regime_filter(self, panel) -> torch.Tensor:
        """Select mean-reversion regime timesteps: mean cross-sectional ADX < 20.

        Low ADX indicates range-bound / oscillating markets where
        mean-reversion strategies are most effective.
        """
        from daft.models.experts.base_expert import _compute_adx_mask
        return _compute_adx_mask(panel, threshold=20.0, above=False)

    def compute_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Reversal expert loss: negative rank IC.

        For mean reversion, we want high rank correlation between predicted
        and actual returns (spearman-like loss implemented via differentiable
        ranking of cross-sectional predictions).
        """
        pred = pred.squeeze(-1)    # (B,)
        target = target.squeeze(-1) # (B,)
        mask = mask.squeeze(-1)     # (B,)

        # Masked rank correlation approximation
        pred_masked = pred * mask
        target_masked = target * mask

        # Pearson correlation as a differentiable proxy for rank IC
        pred_centered = pred_masked - pred_masked.sum() / mask.sum().clamp(min=1)
        target_centered = target_masked - target_masked.sum() / mask.sum().clamp(min=1)

        cov = (pred_centered * target_centered * mask).sum()
        pred_std = ((pred_centered ** 2 * mask).sum() + 1e-8).sqrt()
        target_std = ((target_centered ** 2 * mask).sum() + 1e-8).sqrt()

        ic = cov / (pred_std * target_std).clamp(min=1e-8)
        return -ic  # Negate: maximize IC
