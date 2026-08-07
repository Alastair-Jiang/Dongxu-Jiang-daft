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
        """Select event-driven regime timesteps.

        Without an external event calendar, the event expert trains on ALL
        data as a catch-all generalist. If ``panel.metadata`` contains an
        ``"event_dates"`` key (list of date indices), only ±3-day windows
        around those events are selected.
        """
        T = panel.values.size(0)

        # Check for event calendar in metadata
        event_dates = (panel.metadata or {}).get("event_dates", None)
        if event_dates is not None:
            mask = torch.zeros(T, dtype=torch.bool)
            for ed in event_dates:
                lo = max(0, ed - 3)
                hi = min(T, ed + 4)
                mask[lo:hi] = True
            return mask

        # No calendar → all timesteps (generalist fallback)
        return torch.ones(T, dtype=torch.bool)

    def compute_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Event expert loss: direction-weighted MSE.

        Since the expert's output is SiTU-activated (range [-1, 1]),
        we use MSE with a 5× penalty on directional errors rather than
        BCE (which expects either raw logits or [0,1] probabilities).

        This is consistent with how the ensemble fuses expert signals:
        all experts produce magnitude-comparable outputs in [-1, 1].

        L = Σ masked [ (y - ŷ)² · (1 + 4 · 1[sign(ŷ) ≠ sign(y)]) ]
        """
        pred = pred.squeeze(-1)
        target = target.squeeze(-1)
        mask = mask.squeeze(-1)

        se = (target - pred) ** 2
        sign_mismatch = (torch.sign(pred) != torch.sign(target)).float()
        adjusted_se = se * (1.0 + 4.0 * sign_mismatch)

        loss = (adjusted_se * mask).sum() / mask.sum().clamp(min=1)
        return loss
