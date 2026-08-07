"""Panel dataclass: T×N×F tensor with tradability mask.

Adapted from ml-quant-trading (Yimin Du, 2025, MIT License).
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class Panel:
    """Three-dimensional panel data container for financial time series.

    Parameters
    ----------
    values : torch.Tensor, shape (T, N, F)
        Feature tensor. T = timesteps, N = assets, F = features.
        For OHLCV data, F=5: [open, high, low, close, volume].
    mask : torch.Tensor, shape (T, N), dtype bool
        Tradability mask. True = tradable, False = suspended / limit-up / limit-down.
    dates : list[str] or None
        Date/time labels for the T timesteps.
    asset_ids : list[str] or None
        Asset identifiers for the N assets.
    feature_names : list[str] or None
        Feature names for the F dimensions.
    metadata : dict
        Arbitrary metadata (data source, frequency, generation params, etc.).
    """

    values: torch.Tensor  # (T, N, F)
    mask: torch.Tensor    # (T, N), bool
    dates: Optional[list] = None
    asset_ids: Optional[list] = None
    feature_names: Optional[list] = None
    metadata: Optional[dict] = None

    @property
    def shape(self) -> tuple:
        return tuple(self.values.shape)

    @property
    def device(self) -> torch.device:
        return self.values.device

    @property
    def T(self) -> int:
        return self.values.size(0)

    @property
    def N(self) -> int:
        return self.values.size(1)

    @property
    def F(self) -> int:
        return self.values.size(2)

    def to(self, device: torch.device) -> "Panel":
        return Panel(
            values=self.values.to(device),
            mask=self.mask.to(device),
            dates=self.dates,
            asset_ids=self.asset_ids,
            feature_names=self.feature_names,
            metadata=self.metadata,
        )

    def slice_time(self, start: int, end: int) -> "Panel":
        """Return a time-sliced view of the panel.

        Parameters
        ----------
        start : int  Start timestep index (inclusive).
        end : int    End timestep index (exclusive).

        Returns
        -------
        Panel  with the same N and F, but T = end - start.
        """
        return Panel(
            values=self.values[start:end],
            mask=self.mask[start:end],
            dates=self.dates[start:end] if self.dates else None,
            asset_ids=self.asset_ids,
            feature_names=self.feature_names,
            metadata=self.metadata,
        )

    def train_val_test_split(
        self, train_frac=0.7, val_frac=0.15
    ) -> tuple["Panel", "Panel", "Panel"]:
        """Chronological split (no shuffle, preserves temporal order)."""
        T = self.T
        train_end = int(T * train_frac)
        val_end = int(T * (train_frac + val_frac))

        def _slice(start, end):
            return Panel(
                values=self.values[start:end],
                mask=self.mask[start:end],
                dates=self.dates[start:end] if self.dates else None,
                asset_ids=self.asset_ids,
                feature_names=self.feature_names,
                metadata=self.metadata,
            )

        return _slice(0, train_end), _slice(train_end, val_end), _slice(val_end, T)

    def __repr__(self) -> str:
        return (
            f"Panel(T={self.T}, N={self.N}, F={self.F}, "
            f"mask_coverage={self.mask.float().mean().item():.1%})"
        )
