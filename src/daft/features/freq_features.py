"""FFT-based frequency-domain feature extraction.

Inspired by Super-Linear's FFT-gated MoE router. Computes spectral
signatures of price and volume series for frequency-aware regime detection.

Different market regimes have distinct spectral signatures:
- Trending: dominant low-frequency power
- Mean-reverting: dominant mid-frequency power
- High-volatility: elevated broadband power
- Event-driven: transient high-frequency spikes
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from daft.features.base_features import ensure_base_panel


class FreqFeatureExtractor(nn.Module):
    """FFT-based spectral feature extractor.

    Computes periodogram features from price/volume time series for
    frequency-aware regime identification.

    Parameters
    ----------
    lookback : int
        Number of historical bars for FFT computation. Default: 512.
    n_freq_bins : int
        Number of frequency bins to retain. Default: 64.
    stride : int
        Step size for sliding the FFT window. Default: 1.
        Higher values reduce computation at the cost of temporal resolution.
    """

    def __init__(self, lookback: int = 512, n_freq_bins: int = 64, stride: int = 1):
        super().__init__()
        self.lookback = lookback
        self.n_freq_bins = n_freq_bins
        self.stride = stride

    def compute_periodogram(self, x: torch.Tensor) -> torch.Tensor:
        """Compute normalized power spectral density.

        Following the Super-Linear approach:
        1. Remove DC component
        2. FFT
        3. Power spectral density = |DFT|^2
        4. Normalize to sum=1

        Parameters
        ----------
        x : torch.Tensor, shape (..., lookback)
            Time series values.

        Returns
        -------
        psd : torch.Tensor, shape (..., n_freq_bins)
            Normalized power spectral density.
        """
        # Remove DC component
        x = x - torch.mean(x, dim=-1, keepdim=True)

        # FFT
        dft = torch.fft.fft(x, dim=-1)

        # First n//2 frequency bins (positive frequencies only)
        dft = dft[..., :x.shape[-1] // 2]

        # Power spectral density
        psd = (dft.abs() ** 2)

        # Normalize to sum=1
        psd = psd / (psd.sum(dim=-1, keepdim=True) + 1e-8)

        # Truncate or pad to n_freq_bins
        if psd.shape[-1] > self.n_freq_bins:
            psd = psd[..., :self.n_freq_bins]
        elif psd.shape[-1] < self.n_freq_bins:
            psd = F.pad(psd, (0, self.n_freq_bins - psd.shape[-1]))

        return psd

    def _compute_band_powers(self, psd: torch.Tensor) -> torch.Tensor:
        """Aggregate PSD into low/mid/high frequency band powers.

        Parameters
        ----------
        psd : torch.Tensor, shape (..., n_freq_bins)
            Normalized power spectral density.

        Returns
        -------
        bands : torch.Tensor, shape (..., 3)
            [low_freq_power, mid_freq_power, high_freq_power] each in [0, 1].
        """
        n = psd.shape[-1]
        low_end = max(1, int(n * 0.10))
        mid_end = max(low_end + 1, int(n * 0.40))

        low_power = psd[..., :low_end].sum(dim=-1)
        mid_power = psd[..., low_end:mid_end].sum(dim=-1)
        high_power = psd[..., mid_end:].sum(dim=-1)

        return torch.stack([low_power, mid_power, high_power], dim=-1)

    def forward(self, panel) -> torch.Tensor:
        """Compute sliding-window FFT features for all assets in the panel.

        For each asset, slides a window of ``lookback`` bars across time,
        computes the normalized periodogram, and extracts low/mid/high
        frequency band powers alongside the full PSD.

        Parameters
        ----------
        panel : Panel
            Market data panel with values of shape (T, N, F).
            接受 OHLCV 布局或基础布局(见 features/base_features.py)。
            FFT 使用基础布局里的 log_return(索引 1)。

        Returns
        -------
        freq_features : torch.Tensor, shape (T, N, n_freq_bins + 3)
            Per-timestep spectral features. The first ``n_freq_bins`` columns
            are the PSD bins; the last 3 are low/mid/high band powers.
            For t < lookback, all values are 0.
        """
        # ── 通道契约: 统一为基础特征布局 (2026-08-16 修复) ──
        panel = ensure_base_panel(panel)

        T, N, F = panel.values.shape
        device = panel.device

        # Use log_return (index 1) as primary signal; fallback to close diffs
        if F >= 2:
            signal = panel.values[:, :, 1]  # (T, N): log_return
        else:
            # Compute returns from close price
            close = panel.values[:, :, 0]
            signal = torch.diff(close, dim=0, prepend=close[:1].clone()) / close.clamp(min=1e-6)

        # Pad signal at the beginning so we can produce output at every t
        # For t < lookback: output zeros (insufficient history)
        # Unfold into windows: (T - lookback + 1, N, lookback)
        pad = torch.zeros(self.lookback - 1, N, device=device, dtype=signal.dtype)
        signal_padded = torch.cat([pad, signal], dim=0)  # (T + lookback - 1, N)

        # Unfold windows
        windows = signal_padded.unfold(0, self.lookback, 1)  # (T + L - 1, N, L) → actually (Q, N, L) where Q = T+L-1-L+1 = T
        windows = windows[:T]  # (T, N, lookback)

        # Output tensor: (T, N, n_freq_bins + 3)
        out = torch.zeros(T, N, self.n_freq_bins + 3, device=device, dtype=signal.dtype)

        # Only compute for t >= lookback - 1 (i.e., windows that have enough history)
        # Actually windows[:lookback-1] contain padded zeros, windows[lookback-1:] are real
        start = self.lookback - 1
        if start >= T:
            return out

        # Vectorized: reshape to (B, lookback), compute once
        valid_windows = windows[start:]  # (T-start, N, lookback)
        n_valid, n_assets = valid_windows.shape[0], valid_windows.shape[1]

        # Reshape to 2D for batched FFT
        flat = valid_windows.reshape(-1, self.lookback)  # (n_valid * N, lookback)

        # Compute periodograms in chunks to avoid OOM for large (T×N)
        chunk_size = 1024
        psd_chunks = []
        for i in range(0, flat.shape[0], chunk_size):
            chunk = flat[i:i + chunk_size]
            psd_chunks.append(self.compute_periodogram(chunk))
        psd_all = torch.cat(psd_chunks, dim=0)  # (n_valid * N, n_freq_bins)

        # Reshape back
        psd_all = psd_all.reshape(n_valid, n_assets, self.n_freq_bins)  # (n_valid, N, n_freq_bins)
        band_powers = self._compute_band_powers(psd_all)  # (n_valid, N, 3)

        # Write back
        out[start:, :, :self.n_freq_bins] = psd_all
        out[start:, :, self.n_freq_bins:] = band_powers

        return out
