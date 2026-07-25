"""Stage 3: Joint fine-tuning.

All parameters unfrozen. Full CDAP modulation (δ = 1.0).
Low learning rate (η = 1e-5) to prevent catastrophic forgetting.
Early stopping on validation IC degradation.

After Stage 3, run HardeningEngine statistics collection (Stage 4).
"""

import torch
import torch.nn as nn


class JointTrainer:
    """Joint fine-tuning with full CDAP modulation.

    Parameters
    ----------
    model : ExpertEnsemble
        Full DAFT model (all parameters trainable).
    config : dict
        Stage 3 config.
    device : torch.device
    """

    def __init__(self, model, config: dict, device: torch.device):
        self.model = model
        self.config = config
        self.device = device

    def train(self, train_panel, val_panel) -> dict:
        raise NotImplementedError(
            "JointTrainer to be implemented after Stage 2 integration."
        )
