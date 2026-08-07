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
    """

    def __init__(self, lookback: int = 512, n_freq_bins: int = 64):
        super().__init__()
        self.lookback = lookback
        self.n_freq_bins = n_freq_bins

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

    def forward(self, panel) -> torch.Tensor:
        """Compute FFT features for all assets.

        Returns
        -------
        freq_features : torch.Tensor, shape (T, N, n_freq_bins)
        """
        raise NotImplementedError(
            "FreqFeatureExtractor forward() to be implemented after "
            "data source configuration."
        )
