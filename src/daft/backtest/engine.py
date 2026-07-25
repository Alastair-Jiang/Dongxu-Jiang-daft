"""Vectorized backtesting engine.

Computes strategy returns from model signals with configurable:
- Rebalance frequency (daily, hourly, minutely)
- Transaction cost model (fixed + proportional)
- Slippage model (linear in trade size)
- Performance metrics (Sharpe, Max Drawdown, Calmar, IC, IR)
"""

from typing import Dict

import torch


class BacktestEngine:
    """Vectorized backtesting with performance metrics.

    Parameters
    ----------
    config : dict
        Evaluation section of YAML config.
    """

    def __init__(self, config: dict):
        self.config = config

    def run(
        self,
        signals: torch.Tensor,
        prices: torch.Tensor,
        mask: torch.Tensor,
    ) -> Dict[str, float]:
        """Execute backtest on model signals.

        Parameters
        ----------
        signals : torch.Tensor, shape (T, N)
            Model-predicted signals (expected returns or position scores).
        prices : torch.Tensor, shape (T, N)
            Asset prices for return computation.
        mask : torch.Tensor, shape (T, N), bool
            Tradability mask.

        Returns
        -------
        metrics : dict
            Sharpe ratio, max drawdown, Calmar ratio, IC, ICIR, hit rate, etc.
        """
        raise NotImplementedError(
            "BacktestEngine to be implemented after model signal generation "
            "integration. Will produce Sharpe, MaxDD, Calmar, IC, ICIR, hit rate."
        )

    @staticmethod
    def sharpe_ratio(returns: torch.Tensor, annualization: float = 252.0) -> float:
        """Annualized Sharpe ratio.

        Sharpe = sqrt(T) · mean(r) / std(r)
        """
        if returns.std() == 0:
            return 0.0
        return (returns.mean() / returns.std()).item() * (annualization ** 0.5)

    @staticmethod
    def max_drawdown(cumulative_returns: torch.Tensor) -> float:
        """Maximum drawdown from peak.

        MaxDD = min_t (C_t / max_{τ≤t} C_τ - 1)
        """
        running_max = cumulative_returns.cummax(dim=0).values
        drawdowns = cumulative_returns / running_max - 1
        return drawdowns.min().item()

    @staticmethod
    def info_coefficient(
        predictions: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor,
    ) -> float:
        """Rank IC (Spearman correlation of cross-sectional predictions)."""
        raise NotImplementedError("Rank IC to be implemented.")
