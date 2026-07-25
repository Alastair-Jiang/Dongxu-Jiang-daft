"""Abstract base class for all strategy experts.

Each expert is a specialized neural forecaster trained on a specific
market regime subset. Experts share a common interface for the MoE
ensemble to route through them uniformly.
"""

from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class BaseExpert(nn.Module, ABC):
    """Abstract strategy expert.

    Parameters
    ----------
    input_dim : int
        Dimension of the input feature vector (market state s_t).
    hidden_dim : int
        Hidden layer dimension.
    n_layers : int
        Number of hidden layers.
    name : str
        Expert type identifier (e.g., "trend", "reversal").

    Subclasses must implement:
        _regime_filter(panel) → mask
            Return a boolean mask selecting regime-appropriate training samples.
        compute_loss(pred, target, mask) → Tensor
            Return the expert-specific loss for a batch.
    """

    def __init__(
        self,
        input_dim: int = 200,
        hidden_dim: int = 64,
        n_layers: int = 2,
        name: str = "base",
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.name = name

        # Build MLP backbone
        layers = []
        in_dim = input_dim
        for i in range(n_layers):
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),    # SiLU = Swish, consistent with K3 activation choice
                nn.Dropout(0.1),
            ])
            in_dim = hidden_dim
        self.backbone = nn.Sequential(*layers)

        # Prediction head: hidden_dim → 1 (expected return for next bar)
        self.head = nn.Linear(hidden_dim, 1)

        # SiTU-style activation for bounded, magnitude-comparable output
        # σ(x) · tanh(x) → naturally bounded in [-1, 1]
        self.output_activation = SiTU()

    def forward(self, s_t: torch.Tensor, return_hidden: bool = False):
        """Forward pass.

        Parameters
        ----------
        s_t : torch.Tensor, shape (B, input_dim)
            Market state vector at the current timestep.
        return_hidden : bool
            If True, also return the hidden representation before the head.

        Returns
        -------
        signal : torch.Tensor, shape (B, 1)
            Predicted signal (e.g., expected return for next bar).
        hidden : torch.Tensor, optional, shape (B, hidden_dim)
            Hidden representation, returned only if ``return_hidden=True``.
        """
        hidden = self.backbone(s_t)          # (B, hidden_dim)
        raw = self.head(hidden)              # (B, 1)
        signal = self.output_activation(raw) # (B, 1), bounded in [-1, 1]

        if return_hidden:
            return signal, hidden
        return signal

    @abstractmethod
    def _regime_filter(self, panel) -> torch.Tensor:
        """Return a boolean mask over time steps for regime-appropriate training.

        Each expert subclass defines what "its" regime looks like and trains
        only on those samples. This is the key to expert specialization.

        Returns
        -------
        mask : torch.Tensor, shape (T,), dtype bool
            True for time steps where this expert is active.
        """
        ...

    @abstractmethod
    def compute_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the expert-specific training loss.

        Parameters
        ----------
        pred : torch.Tensor, shape (B, 1)
            Predicted signal.
        target : torch.Tensor, shape (B, 1)
            Ground truth (e.g., actual next-bar return).
        mask : torch.Tensor, shape (B, 1)
            Tradability mask.

        Returns
        -------
        loss : torch.Tensor, scalar
        """
        ...


class SiTU(nn.Module):
    """Sigmoid Tanh Unit — adapted from Kimi K3's custom activation.

    σ(x) · tanh(x) ensures output naturally bounded in [-1, 1] with smooth
    gradients. This is critical for MoE architectures where expert outputs
    must be magnitude-comparable before gated fusion.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(x) * torch.tanh(x)
