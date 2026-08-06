"""Synthetic and real data loaders for the DAFT training pipeline.

Synthetic data: Hidden Markov Model with 3 market regimes (bull/bear/sideways),
multi-factor return generation (CAPM-style), realistic OHLCV synthesis.

Real data: placeholder adapters for baostock / yfinance (to be wired).
"""

from __future__ import annotations
from typing import Dict, Optional
import math

import torch

from daft.data.panel import Panel


class DataLoader:
    """Unified data loader dispatching on source type.

    Usage:
        loader = DataLoader({"source": "synthetic", "n_stocks": 200, "n_days": 500, "frequency": "1d"})
        panel = loader.load()
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def __init__(self, config: Dict):
        self.config = config
        self.source = config.get("source", "synthetic")

    def load(self) -> Panel:
        if self.source == "synthetic":
            gen = SyntheticDataGenerator(
                n_stocks=self.config.get("n_stocks", 200),
                n_days=self.config.get("n_days", 500),
                seed=self.config.get("seed", 42),
            )
            return gen.generate()

        elif self.source == "baostock":
            from daft.data.adapters.baostock_adapter import BaostockAdapter
            adapter = BaostockAdapter(self.config)
            return adapter.load()

        elif self.source == "yfinance":
            from daft.data.adapters.yfinance_adapter import YFinanceAdapter
            adapter = YFinanceAdapter(self.config)
            return adapter.load()

        else:
            raise NotImplementedError(
                f"Data source '{self.source}' not supported. "
                "Available: 'synthetic', 'baostock', 'yfinance'."
            )


# ======================================================================
# Synthetic data generator
# ======================================================================
class SyntheticDataGenerator:
    """Generate realistic synthetic market data with hidden regime structure.

    Data-generating process
    -----------------------
    1. **Hidden regime** — 3-state Markov chain:
       - Bull  (regime=0):  μ = +12 bps/day,  σ = 15 % ann
       - Bear  (regime=1):  μ =  −8 bps/day,  σ = 25 % ann
       - Choppy (regime=2): μ =   0 bps/day,  σ = 10 % ann
       Transition matrix has 95 % diagonal persistence.

    2. **Factor model** (K=3 common factors + idiosyncratic):
       r_i = β_i,1·F₁ + β_i,2·F₂ + β_i,3·F₃ + ε_i
       where each F_k ∼ N(μ_regime_k, σ_regime_k).

    3. **OHLCV synthesis**:
       - Close  = previous close · exp(r_i)
       - Open   = previous close · exp(small overnight noise)
       - High   = max(open, close) + |HalfNormal(σ_hl)|
       - Low    = min(open, close) − |HalfNormal(σ_hl)|
       - Volume = LogNormal(μ_vol_regime, σ_vol), scaled by market cap proxy.

    Output Panel
    ------------
    values : (T, N, 5)   [open, high, low, close, volume]
    mask   : (T, N)      all True  (synthetic data has no suspensions)
    metadata["regime_ids"] : (T,) ground-truth regime label (for eval)
    """

    # Regime parameters: (μ_daily_bps, σ_annual_pct, label)
    REGIME_PARAMS = [
        (12.0, 0.15, "bull"),
        (-8.0, 0.25, "bear"),
        (0.0, 0.10, "choppy"),
    ]

    # Markov transition matrix (row-stochastic)
    # stay 95 %, jump to each other regime 2.5 %
    TRANS_MAT = torch.tensor([
        [0.950, 0.025, 0.025],
        [0.025, 0.950, 0.025],
        [0.025, 0.025, 0.950],
    ])

    def __init__(
        self,
        n_stocks: int = 200,
        n_days: int = 500,
        n_factors: int = 3,
        seed: int = 42,
    ):
        self.n_stocks = n_stocks
        self.n_days = n_days
        self.n_factors = n_factors
        self.seed = seed
        self._rng = torch.Generator()
        self._rng.manual_seed(seed)

    # ------------------------------------------------------------------
    # Top-level entry point
    # ------------------------------------------------------------------
    def generate(self) -> Panel:
        torch.manual_seed(self.seed)

        # 1. Regime sequence
        regime_ids = self._generate_regime_sequence()          # (T,)

        # 2. Factor returns + stock betas + idiosyncratic
        returns = self._generate_returns(regime_ids)           # (T, N)

        # 3. OHLCV from returns
        prices = self._returns_to_prices(returns)              # (T, N, 5)
        # prices[..., 3] = close, prices[..., 4] = volume

        # 4. Assemble Panel
        mask = torch.ones(self.n_days, self.n_stocks, dtype=torch.bool)
        feature_names = ["open", "high", "low", "close", "volume"]

        return Panel(
            values=prices,
            mask=mask,
            feature_names=feature_names,
            metadata={
                "source": "synthetic",
                "n_stocks": self.n_stocks,
                "n_days": self.n_days,
                "regime_ids": regime_ids,
                "frequency": "1d",
            },
        )

    # ------------------------------------------------------------------
    # Step 1 — hidden Markov regime
    # ------------------------------------------------------------------
    def _generate_regime_sequence(self) -> torch.Tensor:
        """Sample a regime path from the Markov transition matrix."""
        n_regimes = len(self.REGIME_PARAMS)
        regimes = torch.zeros(self.n_days, dtype=torch.long)

        # Start uniformly
        regimes[0] = torch.randint(0, n_regimes, (1,), generator=self._rng).item()

        for t in range(1, self.n_days):
            probs = self.TRANS_MAT[regimes[t - 1]]
            regimes[t] = torch.multinomial(
                probs, 1, generator=self._rng
            ).item()

        return regimes

    # ------------------------------------------------------------------
    # Step 2 — multi-factor returns
    # ------------------------------------------------------------------
    def _generate_returns(self, regime_ids: torch.Tensor) -> torch.Tensor:
        """Generate (T, N) return matrix via 3-factor model with regime drift."""
        T, N = self.n_days, self.n_stocks

        # Factor loadings — each stock loads on 3 common factors + market
        # β_i,k ∼ Uniform(0.3, 1.7) for factor 1 ("market"), smaller for others
        beta_market = 0.5 + 1.2 * torch.rand(N, generator=self._rng)          # (N,)
        beta_factor = 0.3 + 0.7 * torch.rand(N, self.n_factors - 1, generator=self._rng)  # (N, K-1)
        beta = torch.cat([beta_market.unsqueeze(-1), beta_factor], dim=-1)     # (N, K)

        # Idiosyncratic vol (annual %), uniform [5 %, 40 %]
        idio_vol_annual = 0.05 + 0.35 * torch.rand(N, generator=self._rng)    # (N,)
        idio_vol_daily = idio_vol_annual / math.sqrt(252)

        # Generate returns day-by-day  (vectorised over stocks, loop over time)
        returns = torch.zeros(T, N)
        for t in range(T):
            regime = regime_ids[t].item()
            mu_bps, sigma_annual, _ = self.REGIME_PARAMS[regime]
            sigma_daily = sigma_annual / math.sqrt(252)

            # Common factors
            factors = torch.randn(self.n_factors, generator=self._rng)
            factors = factors * sigma_daily + (mu_bps / 10000.0) / self.n_factors

            # Stock returns = β·f + ε
            systematic = beta @ factors                                  # (N,)
            idio = torch.randn(N, generator=self._rng) * idio_vol_daily   # (N,)
            returns[t] = systematic + idio

        return returns

    # ------------------------------------------------------------------
    # Step 3 — returns → OHLCV
    # ------------------------------------------------------------------
    def _returns_to_prices(self, returns: torch.Tensor) -> torch.Tensor:
        """Convert log-returns to OHLCV panel.

        Returns
        -------
        panel : (T, N, 5)  [open, high, low, close, volume]
        """
        T, N = returns.shape
        panel = torch.zeros(T, N, 5)

        # Initial close price = 10.0 (arbitrary)
        prev_close = torch.full((N,), 10.0)

        # Volume parameters (log-normal, regime-dependent)
        base_vol = 1_000_000.0  # base daily volume
        vol_std = 0.4           # log-normal std

        for t in range(T):
            r_t = returns[t]  # (N,)

            # Close
            close = prev_close * torch.exp(r_t)

            # Open = previous close * exp(small overnight noise ~ 20% of daily vol)
            overnight_std = r_t.std().item() * 0.2 + 0.001
            open_price = prev_close * torch.exp(
                overnight_std * torch.randn(N, generator=self._rng)
            )

            # High / Low
            intraday_range_std = torch.abs(r_t) * 0.5 + 0.005
            half_range = torch.abs(
                intraday_range_std * torch.randn(N, generator=self._rng)
            )
            high = torch.maximum(open_price, close) + half_range
            low = torch.minimum(open_price, close) - half_range
            low = torch.clamp(low, min=0.01)  # price floor

            # Volume — log-normal
            log_volume = math.log(base_vol) + vol_std * torch.randn(
                N, generator=self._rng
            )
            volume = torch.exp(log_volume)

            panel[t, :, 0] = open_price
            panel[t, :, 1] = high
            panel[t, :, 2] = low
            panel[t, :, 3] = close
            panel[t, :, 4] = volume

            prev_close = close

        return panel
