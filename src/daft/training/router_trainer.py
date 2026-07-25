"""Stage 2: Router + Memory training (experts frozen).

With expert weights frozen from Stage 1, train the Regime Router and
KDA Market Memory to:
1. Route each market state to the best expert(s)
2. Maintain a predictive memory of historical patterns
3. CDAP connections operate at low modulation strength (δ = 0.1)

Loss = weighted sum of expert prediction quality, weighted by routing probs.
"""

import torch
import torch.nn as nn


class RouterTrainer:
    """Train router and memory with frozen experts.

    Parameters
    ----------
    model : ExpertEnsemble
        Full DAFT model with frozen experts.
    config : dict
        Stage 2 config.
    device : torch.device
    """

    def __init__(self, model, config: dict, device: torch.device):
        self.model = model
        self.config = config
        self.device = device

    def train(self, train_panel, val_panel) -> dict:
        raise NotImplementedError(
            "RouterTrainer to be implemented after Stage 1 integration."
        )
