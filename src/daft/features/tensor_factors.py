"""GPU-vectorized factor computation with mask propagation.

Implements the six core primitives on (T, N) tensors. All operations
respect the boolean tradability mask.

Adapted from ml-quant-trading (Yimin Du, 2025, MIT License).
"""

import torch
import torch.nn.functional as F


class TensorFactorEngine:
    """Mask-aware, GPU-accelerated factor computation."""

    # ------------------------------------------------------------------
    @staticmethod
    def rank(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Cross-sectional percentile rank within each timestep.

        x    : (T, N)  float
        mask : (T, N)  bool  (True = valid)
        →    : (T, N)  float ∈ [0, 1]
        """
        x_m = x.clone()
        x_m[~mask] = float("-inf")
        ranked = x_m.argsort(dim=-1).argsort(dim=-1).float()
        valid = mask.sum(dim=-1, keepdim=True).clamp(min=1)
        ranked = ranked / (valid - 1).clamp(min=1)
        ranked[~mask] = 0.0
        return ranked

    # ------------------------------------------------------------------
    @staticmethod
    def corr(
        x: torch.Tensor, y: torch.Tensor, window: int, mask: torch.Tensor
    ) -> torch.Tensor:
        """Rolling Pearson correlation over `window` timesteps.

        x, y   : (T, N)
        window : int
        mask   : (T, N)  bool
        →      : (T, N)  float ∈ [-1, 1]
        """
        x_m = x * mask.float()
        y_m = y * mask.float()
        m_f = mask.float()

        x_w = x_m.unfold(0, window, 1)
        y_w = y_m.unfold(0, window, 1)
        m_w = m_f.unfold(0, window, 1)

        n = m_w.sum(dim=-1).clamp(min=1)
        mu_x = x_w.sum(dim=-1) / n
        mu_y = y_w.sum(dim=-1) / n
        dx = (x_w - mu_x.unsqueeze(-1)) * m_w
        dy = (y_w - mu_y.unsqueeze(-1)) * m_w

        cov = (dx * dy).sum(dim=-1)
        sx = (dx * dx).sum(dim=-1).sqrt().clamp(min=1e-8)
        sy = (dy * dy).sum(dim=-1).sqrt().clamp(min=1e-8)
        c = torch.clamp(cov / (sx * sy), -1.0, 1.0)

        pad = torch.zeros(window - 1, x.size(1), device=x.device)
        return torch.cat([pad, c], dim=0)

    # ------------------------------------------------------------------
    @staticmethod
    def ewma(x: torch.Tensor, span: int, mask: torch.Tensor) -> torch.Tensor:
        """EWMA with span.  x : (T, N), mask : (T, N) bool  →  (T, N)."""
        alpha = 2.0 / (span + 1.0)
        x_m = x * mask.float()
        out = torch.zeros_like(x)
        run = x_m[0].clone()
        out[0] = run
        for t in range(1, x.size(0)):
            run = alpha * x_m[t] + (1 - alpha) * run
            out[t] = run
        return out

    # ------------------------------------------------------------------
    @staticmethod
    def ts_delta(x: torch.Tensor, d: int, mask: torch.Tensor) -> torch.Tensor:
        """Lagged diff  x_t − x_{t−d}.  x : (T, N)."""
        delta = torch.zeros_like(x)
        delta[d:] = x[d:] - x[:-d]
        delta[~mask] = 0.0
        return delta

    # ------------------------------------------------------------------
    @staticmethod
    def ts_sum(x: torch.Tensor, d: int, mask: torch.Tensor) -> torch.Tensor:
        """Rolling sum over d steps.  O(T) via cumsum."""
        x_m = x * mask.float()
        cs = torch.cumsum(x_m, dim=0)
        out = cs.clone()
        out[d:] = cs[d:] - cs[:-d]
        return out

    # ------------------------------------------------------------------
    @staticmethod
    def ts_std(x: torch.Tensor, d: int, mask: torch.Tensor) -> torch.Tensor:
        """Rolling std over d steps (masked, Welford-style)."""
        x_m = x * mask.float()
        m_f = mask.float()
        x_w = x_m.unfold(0, d, 1)
        m_w = m_f.unfold(0, d, 1)
        n = m_w.sum(dim=-1).clamp(min=2)
        mu = x_w.sum(dim=-1) / n
        dx = (x_w - mu.unsqueeze(-1)) * m_w
        var = (dx * dx).sum(dim=-1) / (n - 1).clamp(min=1)
        s = var.sqrt()
        pad = torch.zeros(d - 1, x.size(1), device=x.device)
        return torch.cat([pad, s], dim=0)
