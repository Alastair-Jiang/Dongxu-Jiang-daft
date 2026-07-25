"""Event-driven expert.

Trained on ±3-day windows around earnings announcements, FOMC meetings,
and macro data releases. Loss = post-event directional accuracy.

Event-driven trading requires fundamentally different feature processing
than continuous price-based strategies. This expert is the most sparse
in activation frequency but the highest in per-trade conviction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from daft.models.experts.base_expert import BaseExpert


class EventExpert(BaseExpert):
    """Expert specialized in event-driven trading.

    Competence region: ±3-day windows around scheduled events (earnings,
    FOMC, macro releases). The expert learns pre-event positioning and
    post-event momentum patterns.
    """

    def __init__(self, input_dim=200, hidden_dim=48, n_layers=2):
        super().__init__(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            name="event",
        )

    def _regime_filter(self, panel) -> torch.Tensor:
        """Select event-driven regime samples.

        Filter: time steps within ±n days of a known event.
        Requires an external event calendar.
        """
        raise NotImplementedError(
            "Regime filter requires event calendar integration. "
            "Will be implemented after data source configuration."
        )

    def compute_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Event expert loss: directional binary cross-entropy.

        Events have discrete outcomes — the model should predict direction
        with high conviction rather than precise magnitude.

        L = BCE(sigmoid(ŷ), 1[target > 0])
        """
        pred = pred.squeeze(-1)
        target = target.squeeze(-1)
        mask = mask.squeeze(-1)

        # Binarize target: care about direction, not magnitude
        target_binary = (target > 0).float()

        # Weighted BCE: penalize missed events more than false alarms
        pos_weight = torch.tensor(2.0, device=pred.device)
        bce = F.binary_cross_entropy_with_logits(
            pred, target_binary, pos_weight=pos_weight, reduction='none'
        )

        loss = (bce * mask).sum() / mask.sum().clamp(min=1)
        return loss
