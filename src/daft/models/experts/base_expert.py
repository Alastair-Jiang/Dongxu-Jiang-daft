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


# ======================================================================
# Shared regime filter helpers
# ======================================================================

def _compute_adx_mask(panel, threshold: float, above: bool) -> "torch.Tensor":
    """Compute (T,) bool mask where mean-cross-sectional ADX crosses threshold.

    Parameters
    ----------
    panel : Panel  with values (T, N, 5) [open, high, low, close, volume].
    threshold : float  ADX decision boundary.
    above : bool  True → ADX > threshold; False → ADX < threshold.

    Returns
    -------
    mask : (T,) bool
    """
    close = panel.values[..., 3]   # (T, N)
    high = panel.values[..., 1]
    low = panel.values[..., 2]

    T, N = close.shape
    if T < 16:
        return torch.ones(T, dtype=torch.bool)

    # Directional movement
    up_move = high[1:] - high[:-1]           # (T-1, N)
    dn_move = low[:-1] - low[1:]
    plus_dm = up_move.clamp(min=0) * (up_move > dn_move).float()
    minus_dm = dn_move.clamp(min=0) * (dn_move > up_move).float()

    tr = torch.maximum(high[1:] - low[1:],
           torch.maximum((high[1:] - close[:-1]).abs(),
                         (low[1:] - close[:-1]).abs()))

    span = 14
    alpha = 2.0 / (span + 1)
    atr = torch.zeros_like(tr)
    pdi = torch.zeros_like(tr)
    mdi = torch.zeros_like(tr)
    for t in range(tr.size(0)):
        if t == 0:
            atr[t] = tr[t]; pdi[t] = plus_dm[t]; mdi[t] = minus_dm[t]
        else:
            atr[t] = alpha * tr[t] + (1 - alpha) * atr[t - 1]
            pdi[t] = alpha * plus_dm[t] + (1 - alpha) * pdi[t - 1]
            mdi[t] = alpha * minus_dm[t] + (1 - alpha) * mdi[t - 1]

    dx = (pdi - mdi).abs() / (pdi + mdi).clamp(min=1e-8) * 100.0
    adx = torch.zeros_like(dx)
    for t in range(dx.size(0)):
        adx[t] = alpha * (dx[t] if t == 0 else dx[t]) + \
                 (1 - alpha) * (adx[t - 1] if t > 0 else dx[t])

    # Per-timestep mean ADX across assets
    mean_adx = adx.nan_to_num(0).mean(dim=-1)   # (T-1,)
    # Pad first element (no ADX yet) with threshold value
    mean_adx = torch.cat([mean_adx[:1], mean_adx])

    if above:
        return mean_adx > threshold
    return mean_adx < threshold


def _compute_vol_mask(panel, quantile: float = 0.80) -> "torch.Tensor":
    """Compute (T,) bool mask where mean-cross-sectional vol exceeds quantile.

    Uses rolling 20-day std of log-returns, averaged across assets.
    """
    close = panel.values[..., 3]   # (T, N)
    log_c = torch.log(close.clamp(min=1e-8))
    returns = log_c[1:] - log_c[:-1]     # (T-1, N)
    T_ret, N = returns.shape

    if T_ret < 21:
        return torch.ones(panel.T, dtype=torch.bool)

    # Rolling 20d std per asset
    vol20 = torch.zeros(T_ret, N)
    for t in range(T_ret):
        lo = max(0, t - 19)
        n_obs = t - lo + 1
        if n_obs >= 2:
            vol20[t] = returns[lo:t + 1].std(dim=0)
        # else: stays 0

    vol20 = vol20.nan_to_num(0)
    mean_vol = vol20.mean(dim=-1)          # (T-1,)
    threshold = mean_vol[mean_vol > 0].quantile(quantile) if mean_vol.any() else 0.0
    vol_mask = mean_vol > threshold

    # Pad to length T
    return torch.cat([vol_mask[:1], vol_mask])
