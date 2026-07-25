"""Stage 1: Independent expert training.

Each strategy expert is trained on its regime-specific data subset
with its own specialized loss function. This stage establishes expert
competence before the router learns to assign inputs.

Training recipe:
- TrendExpert: directional accuracy + Sharpe (AdjMSE loss)
- ReversalExpert: rank IC (negative-IC loss)
- VolatilityExpert: volatility forecast MSE + variance penalty
- EventExpert: directional BCE (binary cross-entropy)

After Stage 1, all expert weights are frozen for Stage 2.
"""

from typing import Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class ExpertTrainer:
    """Train each expert independently on its regime-filtered data.

    Parameters
    ----------
    experts : list of BaseExpert
        List of expert modules to train.
    config : dict
        Stage 1 training config (epochs, lr, batch_size, etc.).
    device : torch.device
    """

    def __init__(self, experts, config: dict, device: torch.device):
        self.experts = experts
        self.config = config
        self.device = device

    def train(
        self,
        train_panel,
        val_panel,
    ) -> Dict[str, float]:
        """Execute Stage 1 training for all experts.

        Returns
        -------
        metrics : dict
            Validation metrics per expert.
        """
        # PLACEHOLDER — to be implemented with full data pipeline
        raise NotImplementedError(
            "ExpertTrainer to be implemented after data pipeline integration. "
            "See docs/architecture.md for the training protocol specification."
        )
