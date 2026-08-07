"""Momentum expert.

Trained on periods with persistent, cross-sectional momentum — a regime
that is *distinct* from pure trend-following. Momentum captures the
cross-sectional persistence of returns (winners keep winning), while the
trend expert captures directional ADX-filtered movement. The two regimes
partially overlap but are not identical; the router learns to assign
inputs to whichever specialist is most predictive.

Research grounding
------------------
- "Understanding momentum and reversal" (Chichernea, Holderness,
  Petkevich — Journal of Financial Economics 140(3):726–743, 2021).
  Establishes that momentum and reversal are *separate, regime-dependent*
  phenomena and that the sign of the effect flips with the formation
  horizon — motivating a dedicated momentum specialist rather than folding
  it into the trend expert.
- DHMoE: Diffusion Generated Hierarchical Multi-Granular Expertise for
  Stock Prediction (Chen & Wang, AAAI 2025) — hierarchical, granular
  expert specialization improves stock prediction; each expert owns a
  well-defined competence slice.

Competence region: 20-day (formation) cross-sectional momentum is
significantly non-zero AND return spread between top/bottom deciles is
wide. Output: expected next-bar return conditioned on momentum persistence.
"""

import torch

from daft.models.experts.base_expert import BaseExpert


def _compute_momentum_mask(panel, formation: int = 20, atol: float = 1e-6):
    """Return a (T,) bool mask selecting momentum-regime timesteps.

    Momentum regime is detected when the mean cross-sectional 20-day
    momentum exceeds a rolling significance bound (its own rolling mean +
    0.5× rolling std), i.e. momentum is *persistently* present rather than
    noise. This keeps the expert specialized to genuine momentum episodes.
    """
    close = panel.values[..., 3]          # (T, N)
    T, N = close.shape
    if T < formation + 5:
        return torch.ones(T, dtype=torch.bool)

    # Simple 20-day momentum: log(close_t / close_{t-formation})
    log_c = torch.log(close.clamp(min=1e-8))
    mom = log_c[formation:] - log_c[:-formation]   # (T-formation, N)
    cross_mom = mom.mean(dim=-1)                     # (T-formation,)

    # Rolling significance band (window 40)
    sig = torch.zeros_like(cross_mom)
    inc = torch.zeros_like(cross_mom)
    for t in range(cross_mom.size(0)):
        lo = max(0, t - 39)
        window = cross_mom[lo:t + 1]
        if window.numel() >= 5:
            sig[t] = window.mean() + 0.5 * window.std().clamp(min=atol)

    active = cross_mom.abs() > sig.abs().clamp(min=atol)
    # Pad front (formation) steps to first available value
    active = torch.cat([active[:1].expand(formation), active])
    # Ensure length T
    return active[:T]


class MomentumExpert(BaseExpert):
    """Expert specialized in cross-sectional momentum regimes.

    Competence region: persistent, significant 20-day cross-sectional
    momentum. Output: expected next-bar return, momentum-persistence
    conditioned.
    """

    def __init__(self, input_dim=200, hidden_dim=64, n_layers=2):
        super().__init__(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            name="momentum",
        )

    def _regime_filter(self, panel) -> torch.Tensor:
        return _compute_momentum_mask(panel)

    def compute_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Momentum expert loss: direction-weighted MSE.

        Momentum persistence means *sign consistency* matters most — a
        wrong sign is costed 8x a pure magnitude error. Tuned between the
        trend expert (11x) and event expert (4x) since momentum is
        noisier than trend but stronger than event signals.
        """
        pred = pred.squeeze(-1)
        target = target.squeeze(-1)
        mask = mask.squeeze(-1)

        se = (target - pred) ** 2
        sign_mismatch = (torch.sign(pred) != torch.sign(target)).float()
        adjusted_se = se * (1.0 + 8.0 * sign_mismatch)
        loss = (adjusted_se * mask).sum() / mask.sum().clamp(min=1)
        return loss