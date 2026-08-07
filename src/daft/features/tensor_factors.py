"""GPU-vectorized factor computation engine.

Provides masked primitives for computing common factor operations on
Panel tensors without leaking information from non-tradable periods.

Derived from ml-quant-trading (Yimin Du, 2025, MIT License).

Primitives:
    rank(x, mask)          — cross-sectional ranking
    corr(x, y, window, mask) — rolling Pearson correlation
    ewma(x, span, mask)    — exponentially weighted moving average
    ts_delta(x, d, mask)   — lagged difference
    ts_sum(x, d, mask)     — rolling sum
    ts_std(x, d, mask)     — rolling standard deviation
"""

import torch
import torch.nn.functional as F


class TensorFactorEngine:
    """GPU-accelerated factor computation with mask propagation.

    All operations respect the boolean tradability mask to prevent
    upstream contamination from limit-up, limit-down, or suspended periods.

    The mask is threaded through every rolling operation: masked values
    are excluded from the computation window.

    All methods accept and return (T, N) tensors where T = time steps
    and N = number of assets.
    """

    def __init__(self):
        pass

    # ── Cross-Sectional Rank ────────────────────────────────────────────

    def rank(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Cross-sectional rank (percentile) within each time step.

        For each time step t, assets are ranked against each other.
        Masked (non-tradable) assets are excluded from the ranking pool
        and assigned a neutral rank of 0.5.

        Uses the double-argsort method: the first argsort gives positions,
        the second argsort of those positions gives ranks in [0, M-1] where
        M is the number of tradable assets at time t.

        Parameters
        ----------
        x : torch.Tensor, shape (T, N)
            Feature values.
        mask : torch.Tensor, shape (T, N), bool
            Tradability mask. ``True`` = tradable.

        Returns
        -------
        ranked : torch.Tensor, shape (T, N)
            Rank percentiles in [0, 1]. Masked assets = 0.5.
        """
        T, N = x.shape
        device = x.device
        ranked = torch.full((T, N), 0.5, device=device, dtype=x.dtype)

        for t in range(T):
            valid = mask[t]  # (N,), bool
            n_valid = valid.sum().item()
            if n_valid <= 1:
                continue

            x_t = x[t, valid]  # (M,) where M = n_valid
            # Double-argsort: ranks in [0, M-1]
            order = torch.argsort(x_t)
            ranks_raw = torch.argsort(order).float()  # (M,)
            # Normalize to [0, 1]
            ranks_norm = ranks_raw / (n_valid - 1)
            ranked[t, valid] = ranks_norm.to(dtype=x.dtype)

        return ranked

    # ── Rolling Pearson Correlation ─────────────────────────────────────

    def corr(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        window: int,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Rolling Pearson correlation with mask propagation.

        For each asset n and time step t, computes Pearson correlation
        between x[t-window:t, n] and y[t-window:t, n], using only
        timesteps where both observations are valid (mask=True).

        Parameters
        ----------
        x, y : torch.Tensor, shape (T, N)
            Input series.
        window : int
            Rolling window length.
        mask : torch.Tensor, shape (T, N), bool
            Tradability mask.

        Returns
        -------
        corr : torch.Tensor, shape (T, N)
            Rolling correlation in [-1, 1]. Insufficient data → 0.
        """
        T, N = x.shape
        device = x.device
        min_periods = max(window // 3, 3)
        corr_out = torch.zeros(T, N, device=device, dtype=x.dtype)

        # Combine mask: both x and y must be valid
        joint_mask = mask  # same mask for both in typical usage

        # Pre-compute masked series (zero-out invalid values for unfolding)
        x_masked = x.clone()
        y_masked = y.clone()
        x_masked[~joint_mask] = 0.0
        y_masked[~joint_mask] = 0.0

        # Pad at the beginning for unfolding
        pad = torch.zeros(window - 1, N, device=device, dtype=x.dtype)
        x_pad = torch.cat([pad, x_masked], dim=0)  # (T+window-1, N)
        y_pad = torch.cat([pad, y_masked], dim=0)
        m_pad = torch.cat([
            torch.zeros(window - 1, N, device=device, dtype=torch.bool),
            joint_mask,
        ], dim=0)

        # Unfold into windows: (T, N, window)
        x_win = x_pad.unfold(0, window, 1)  # (T+window-1, N, window) → we only need first T
        y_win = y_pad.unfold(0, window, 1)
        m_win = m_pad.unfold(0, window, 1)

        # Trim to exactly T
        x_win = x_win[-T:]  # (T, N, window)
        y_win = y_win[-T:]
        m_win = m_win[-T:]

        # Per-window: count valid observations
        n_valid = m_win.float().sum(dim=-1)  # (T, N)

        # Masked mean
        sum_x = x_win.sum(dim=-1)  # (T, N)
        sum_y = y_win.sum(dim=-1)
        mean_x = sum_x / n_valid.clamp(min=1)
        mean_y = sum_y / n_valid.clamp(min=1)

        # Masked demean
        dm_x = (x_win - mean_x.unsqueeze(-1)) * m_win.float()  # (T, N, window)
        dm_y = (y_win - mean_y.unsqueeze(-1)) * m_win.float()

        # Covariance and std
        cov = (dm_x * dm_y).sum(dim=-1) / n_valid.clamp(min=1)  # (T, N)
        var_x = (dm_x * dm_x).sum(dim=-1) / n_valid.clamp(min=1)
        var_y = (dm_y * dm_y).sum(dim=-1) / n_valid.clamp(min=1)

        # Pearson correlation
        denom = (var_x * var_y).sqrt()
        valid_denom = denom > 1e-10
        corr_out[valid_denom] = (cov[valid_denom] / denom[valid_denom]).clamp(-1.0, 1.0)

        # Zero-out where insufficient data
        corr_out[n_valid < min_periods] = 0.0

        return corr_out

    # ── Exponentially Weighted Moving Average ────────────────────────────

    def ewma(
        self,
        x: torch.Tensor,
        span: int,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Exponentially weighted moving average with mask-aware carry-forward.

        When mask[t, n] is False, the EWMA carries forward the previous
        value (no update). When True, the standard EWMA update applies.

        Vectorized over assets: processes all N assets in parallel per timestep.

        Parameters
        ----------
        x : torch.Tensor, shape (T, N)
            Input series.
        span : int
            EWMA span (center of mass). α = 2 / (span + 1).
        mask : torch.Tensor, shape (T, N), bool
            Tradability mask.

        Returns
        -------
        ewma : torch.Tensor, shape (T, N)
            Exponentially smoothed series.
        """
        T, N = x.shape
        device = x.device
        alpha = 2.0 / (span + 1.0)
        ewma_out = torch.zeros(T, N, device=device, dtype=x.dtype)

        # Vectorized over N: one Python loop over T, torch.where for masking
        for t in range(T):
            if t == 0:
                ewma_out[t] = torch.where(mask[t], x[t], ewma_out[t])
            else:
                update = alpha * x[t] + (1.0 - alpha) * ewma_out[t - 1]
                ewma_out[t] = torch.where(mask[t], update, ewma_out[t - 1])

        return ewma_out

    # ── Time-Series Delta ────────────────────────────────────────────────

    def ts_delta(
        self,
        x: torch.Tensor,
        d: int,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Lagged difference: x[t] - x[t-d].

        Mask propagation: result at t is valid only if both
        mask[t] and mask[t-d] are True.

        Parameters
        ----------
        x : torch.Tensor, shape (T, N)
            Input series.
        d : int
            Lag period.
        mask : torch.Tensor, shape (T, N), bool
            Tradability mask.

        Returns
        -------
        delta : torch.Tensor, shape (T, N)
            Lagged difference. t < d → 0.
        """
        T, N = x.shape
        device = x.device
        delta = torch.zeros(T, N, device=device, dtype=x.dtype)

        if d >= T:
            return delta

        delta[d:] = x[d:] - x[:-d]
        # Mask propagation: both t and t-d must be valid
        result_mask = mask.clone()
        result_mask[d:] = mask[d:] & mask[:-d]
        delta[~result_mask] = 0.0

        return delta

    # ── Rolling Sum ──────────────────────────────────────────────────────

    def ts_sum(
        self,
        x: torch.Tensor,
        d: int,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Rolling sum over d steps, excluding masked values.

        Parameters
        ----------
        x : torch.Tensor, shape (T, N)
            Input series.
        d : int
            Rolling window length.
        mask : torch.Tensor, shape (T, N), bool
            Tradability mask.

        Returns
        -------
        rolling_sum : torch.Tensor, shape (T, N)
            Rolling sum. t < d-1 → partial sum based on available data.
        """
        T, N = x.shape
        device = x.device

        # Masked x: invalid values → 0
        x_masked = x.clone()
        x_masked[~mask] = 0.0

        # Pad and unfold
        pad = torch.zeros(d - 1, N, device=device, dtype=x.dtype)
        x_pad = torch.cat([pad, x_masked], dim=0)

        windows = x_pad.unfold(0, d, 1)  # (T+d-1, N, d)
        windows = windows[:T]  # (T, N, d)

        rolling_sum = windows.sum(dim=-1)  # (T, N)
        return rolling_sum

    # ── Rolling Standard Deviation ───────────────────────────────────────

    def ts_std(
        self,
        x: torch.Tensor,
        d: int,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Rolling standard deviation over d steps, excluding masked values.

        Uses the Welford-style two-pass approach via masked mean.
        min_periods = max(d // 3, 2).

        Parameters
        ----------
        x : torch.Tensor, shape (T, N)
            Input series.
        d : int
            Rolling window length.
        mask : torch.Tensor, shape (T, N), bool
            Tradability mask.

        Returns
        -------
        rolling_std : torch.Tensor, shape (T, N)
            Rolling standard deviation. Insufficient data → 0.
        """
        T, N = x.shape
        device = x.device
        min_periods = max(d // 3, 2)

        x_masked = x.clone()
        x_masked[~mask] = 0.0

        # Pad and unfold
        pad = torch.zeros(d - 1, N, device=device, dtype=x.dtype)
        x_pad = torch.cat([pad, x_masked], dim=0)
        m_pad = torch.cat([
            torch.zeros(d - 1, N, device=device, dtype=torch.bool),
            mask,
        ], dim=0)

        x_win = x_pad.unfold(0, d, 1)[:T]  # (T, N, d)
        m_win = m_pad.unfold(0, d, 1)[:T]

        n_valid = m_win.float().sum(dim=-1)  # (T, N)

        # Masked mean
        sum_x = x_win.sum(dim=-1)
        mean_x = sum_x / n_valid.clamp(min=1)

        # Masked variance
        dm = (x_win - mean_x.unsqueeze(-1)) * m_win.float()
        var = (dm * dm).sum(dim=-1) / n_valid.clamp(min=1)

        std = var.sqrt()
        std[n_valid < min_periods] = 0.0

        return std

    # ── Rolling Mean (convenience) ───────────────────────────────────────

    def ts_mean(
        self,
        x: torch.Tensor,
        d: int,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Rolling mean over d steps, excluding masked values.

        Convenience wrapper combining ts_sum with count normalization.

        Parameters
        ----------
        x : torch.Tensor, shape (T, N)
            Input series.
        d : int
            Rolling window length.
        mask : torch.Tensor, shape (T, N), bool
            Tradability mask.

        Returns
        -------
        rolling_mean : torch.Tensor, shape (T, N)
        """
        T, N = x.shape
        device = x.device
        min_periods = max(d // 3, 2)

        x_masked = x.clone()
        x_masked[~mask] = 0.0

        pad = torch.zeros(d - 1, N, device=device, dtype=x.dtype)
        x_pad = torch.cat([pad, x_masked], dim=0)
        m_pad = torch.cat([
            torch.zeros(d - 1, N, device=device, dtype=torch.bool),
            mask,
        ], dim=0)

        x_win = x_pad.unfold(0, d, 1)[:T]
        m_win = m_pad.unfold(0, d, 1)[:T]

        n_valid = m_win.float().sum(dim=-1)
        sum_x = x_win.sum(dim=-1)
        mean = sum_x / n_valid.clamp(min=1)
        mean[n_valid < min_periods] = 0.0

        return mean
