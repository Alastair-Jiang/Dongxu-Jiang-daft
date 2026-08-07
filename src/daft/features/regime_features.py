"""Market regime feature extraction.

Constructs the 200-dimensional market state vector s_t from raw OHLCV data
and computed base factors.

s_t encodes:
- Price dynamics: returns at multiple horizons, price relatives to MAs
- Volume dynamics: volume ratios, turnover anomalies
- Volatility structure: rolling vol at multiple windows, vol-of-vol
- Microstructure: bid-ask spread proxy, price impact proxy, Amihud illiquidity
- Cross-sectional: rank position within universe, dispersion
- Momentum/factor: short/medium/long-term momentum, factor exposures
"""

import torch
import torch.nn as nn

from daft.features.tensor_factors import TensorFactorEngine


class RegimeFeatureExtractor(nn.Module):
    """Extract the 200-dimensional market state vector s_t.

    This is the input to the Regime Router and all downstream components.
    The design follows the principle: capture enough information for regime
    identification without overfitting to noise.

    The 200 dimensions are grouped into 6 families:
        Price Dynamics      (45 dims): multi-horizon returns, MA devs, acceleration
        Volume Dynamics     (35 dims): volume ratios, trends, vol-price corr
        Volatility Structure(40 dims): multi-window vol, vol-of-vol, ret/vol ratio
        Microstructure      (20 dims): Amihud, price impact, spread proxies
        Cross-Sectional     (30 dims): rank, dispersion, z-scores
        Momentum & Factor   (30 dims): momentum, reversal, RSI, BB position

    Parameters
    ----------
    n_base_factors : int
        Number of base factors expected in the panel. Default: 50.
        Currently the extractor uses 5 base features from the panel
        (close, log_return, volume, volume_ratio, volatility_20) and
        derives the remaining 195 dimensions.
    """

    # Expected panel feature indices
    IDX_CLOSE = 0
    IDX_RETURN = 1
    IDX_VOLUME = 2
    IDX_VOLUME_RATIO = 3
    IDX_VOLATILITY = 4

    def __init__(self, n_base_factors: int = 50, output_dim: int = 200):
        super().__init__()
        self.n_base_factors = n_base_factors
        self.output_dim = output_dim
        self.engine = TensorFactorEngine()

    def _get_series(self, panel, idx: int):
        """Extract a named feature series from the panel."""
        return panel.values[:, :, idx]

    def _panel_mask_2d(self, panel) -> torch.Tensor:
        """Get (T, N) boolean mask from panel.

        Uses the first feature's mask as the asset-level tradability mask,
        since all features in a Panel share the same mask shape.
        """
        if panel.mask.ndim == 3:
            return panel.mask[:, :, 0]
        return panel.mask

    # ════════════════════════════════════════════════════════════════════
    # Group 1: Price Dynamics (45 dims)
    # ════════════════════════════════════════════════════════════════════

    def _price_dynamics(self, panel, mask_2d):
        """Multi-horizon returns, price-to-MA, acceleration, cumulative returns."""
        ret = self._get_series(panel, self.IDX_RETURN)  # (T, N)
        close = self._get_series(panel, self.IDX_CLOSE)
        engine = self.engine
        feats = []

        # --- Multi-horizon returns (7 dims) ---
        for d in [1, 2, 3, 5, 10, 20, 60]:
            # Cumulative return over d steps
            cum_ret = engine.ts_sum(ret, d, mask_2d)
            feats.append(cum_ret)

        # --- Price relative to moving averages (4 dims) ---
        for w in [5, 10, 20, 60]:
            ma = engine.ts_mean(close, w, mask_2d)
            dev = (close - ma) / ma.clamp(min=1e-6)
            feats.append(dev)

        # --- Log price (1 dim) ---
        log_price = close.clamp(min=1e-6).log()
        feats.append(log_price)

        # --- High-Low range proxy: use volatility as proxy (1 dim) ---
        vol_20 = self._get_series(panel, self.IDX_VOLATILITY)
        hl_range = vol_20 / close.clamp(min=1e-6)
        feats.append(hl_range)

        # --- Return acceleration: ret_t - ret_{t-d} (2 dims) ---
        for d in [5, 20]:
            accel = engine.ts_delta(ret, d, mask_2d)
            feats.append(accel)

        # --- Cumulative returns (2 dims) ---
        for d in [5, 20]:
            cumret_long = engine.ts_sum(ret, d, mask_2d)
            feats.append(cumret_long)

        # --- Additional price-derived features (28 dims) ---
        # Multi-scale return volatility (std of returns)
        for d in [5, 10, 20, 60]:
            feats.append(engine.ts_std(ret, d, mask_2d))

        # Return EWMA with different spans
        for s in [3, 10, 30]:
            feats.append(engine.ewma(ret, s, mask_2d))

        # Price path signature: ratio of short-term to long-term MA deviation
        for w_short, w_long in [(5, 20), (10, 60), (20, 60)]:
            ma_s = engine.ts_mean(close, w_short, mask_2d)
            ma_l = engine.ts_mean(close, w_long, mask_2d)
            feats.append(ma_s / ma_l.clamp(min=1e-6))

        # Return skewness proxy: (ret - ret_ewma) / ret_std
        ret_ewma_30 = engine.ewma(ret, 30, mask_2d)
        ret_std_20 = engine.ts_std(ret, 20, mask_2d)
        skew = (ret - ret_ewma_30) / ret_std_20.clamp(min=1e-8)
        feats.append(skew)

        # Upside/downside volatility ratio (5 dims)
        for d in [10, 20]:
            ups = ret.clamp(min=0)
            downs = ret.clamp(max=0).abs()
            up_std = engine.ts_std(ups, d, mask_2d)
            down_std = engine.ts_std(downs, d, mask_2d)
            feats.append(up_std / down_std.clamp(min=1e-8))
            feats.append((up_std - down_std) / (up_std + down_std + 1e-8))

        # Return autocorrelation proxy (1 dim): ts_delta of EWMA
        feats.append(engine.ts_delta(ret_ewma_30, 1, mask_2d))

        # Fill remaining slots with interactions
        while len(feats) < 45:
            # Cross-scale return products
            ret_5 = engine.ts_sum(ret, 5, mask_2d)
            ret_20 = engine.ts_sum(ret, 20, mask_2d)
            feats.append(ret_5 * ret_20)

        # Stack and truncate to exactly 45
        stacked = torch.stack(feats[:45], dim=-1)  # (T, N, 45)
        return stacked

    # ════════════════════════════════════════════════════════════════════
    # Group 2: Volume Dynamics (35 dims)
    # ════════════════════════════════════════════════════════════════════

    def _volume_dynamics(self, panel, mask_2d):
        """Volume ratios, volume trends, volume-price correlation, turnover."""
        volume = self._get_series(panel, self.IDX_VOLUME)
        vol_ratio = self._get_series(panel, self.IDX_VOLUME_RATIO)
        ret = self._get_series(panel, self.IDX_RETURN)
        engine = self.engine
        feats = []

        # --- Volume ratio at multiple horizons (3 dims) ---
        for w in [5, 20, 60]:
            vol_ma = engine.ts_mean(volume, w, mask_2d)
            feats.append(volume / vol_ma.clamp(min=1e-6))

        # --- Volume trend: EWMA of volume (3 dims) ---
        for s in [5, 20, 60]:
            feats.append(engine.ewma(volume, s, mask_2d))

        # --- Volume acceleration: delta of volume (2 dims) ---
        for d in [5, 20]:
            feats.append(engine.ts_delta(volume, d, mask_2d))

        # --- Volume-price correlation (1 dim) ---
        feats.append(engine.corr(volume, ret, 20, mask_2d))

        # --- Turnover proxy: volume_ratio (already in panel) + derived (3 dims) ---
        feats.append(vol_ratio)
        feats.append(engine.ewma(vol_ratio, 10, mask_2d))
        feats.append(engine.ts_std(vol_ratio, 20, mask_2d))

        # --- Volume volatility (3 dims) ---
        for d in [5, 20, 60]:
            feats.append(engine.ts_std(volume, d, mask_2d))

        # --- Volume rank (cross-sectional volume position) (3 dims) ---
        for w in [1, 5, 20]:
            vol_smoothed = engine.ewma(volume, w, mask_2d)
            feats.append(engine.rank(vol_smoothed, mask_2d))

        # --- Volume-return interaction (3 dims) ---
        feats.append(ret * vol_ratio)
        feats.append(ret.abs() * vol_ratio)
        feats.append(engine.ewma(ret * vol_ratio, 10, mask_2d))

        # Fill remaining slots
        while len(feats) < 35:
            feats.append(engine.ts_delta(vol_ratio, 3, mask_2d))

        stacked = torch.stack(feats[:35], dim=-1)
        return stacked

    # ════════════════════════════════════════════════════════════════════
    # Group 3: Volatility Structure (40 dims)
    # ════════════════════════════════════════════════════════════════════

    def _volatility_structure(self, panel, mask_2d):
        """Multi-window vol, vol-of-vol, return/vol ratios, higher moments."""
        ret = self._get_series(panel, self.IDX_RETURN)
        engine = self.engine
        feats = []

        # --- Multi-window volatility (4 dims) ---
        for d in [5, 10, 20, 60]:
            feats.append(engine.ts_std(ret, d, mask_2d))

        # --- Vol-of-vol (2 dims) ---
        vol_20 = engine.ts_std(ret, 20, mask_2d)
        vol_60 = engine.ts_std(ret, 60, mask_2d)
        feats.append(engine.ts_std(vol_20, 60, mask_2d))  # vol-of-vol
        feats.append(engine.ts_std(vol_60, 60, mask_2d))

        # --- Parkinson-like volatility: use high-low proxy from vol (2 dims) ---
        # (In full implementation, would use actual high/low; here we use vol as proxy)
        feats.append(vol_20)  # short-term vol
        feats.append(vol_60 / vol_20.clamp(min=1e-8))  # vol term structure

        # --- Return-to-volatility ratios (4 dims) ---
        for d in [5, 10, 20, 60]:
            ret_cum = engine.ts_sum(ret, d, mask_2d)
            vol_d = engine.ts_std(ret, d, mask_2d)
            feats.append(ret_cum / vol_d.clamp(min=1e-8))

        # --- Volatility EWMA (3 dims) ---
        for s in [5, 20, 60]:
            vol_s = engine.ts_std(ret, s, mask_2d)
            feats.append(engine.ewma(vol_s, 30, mask_2d))

        # --- Volatility acceleration (2 dims) ---
        feats.append(engine.ts_delta(vol_20, 5, mask_2d))
        feats.append(engine.ts_delta(vol_20, 20, mask_2d))

        # --- Volatility regime: above/below long-term average (3 dims) ---
        vol_lt_mean = engine.ewma(vol_20, 120, mask_2d)
        feats.append(vol_20 / vol_lt_mean.clamp(min=1e-8))
        feats.append((vol_20 > vol_lt_mean).float())
        feats.append(torch.sigmoid((vol_20 - vol_lt_mean) / vol_lt_mean.clamp(min=1e-8)))

        # --- Cross-sectional vol rank (1 dim) ---
        feats.append(engine.rank(vol_20, mask_2d))

        # Fill remaining slots with vol-derived features
        while len(feats) < 40:
            feats.append(vol_20 * ret)

        stacked = torch.stack(feats[:40], dim=-1)
        return stacked

    # ════════════════════════════════════════════════════════════════════
    # Group 4: Microstructure (20 dims)
    # ════════════════════════════════════════════════════════════════════

    def _microstructure(self, panel, mask_2d):
        """Amihud illiquidity, price impact proxy, spread proxies."""
        ret = self._get_series(panel, self.IDX_RETURN)
        volume = self._get_series(panel, self.IDX_VOLUME)
        close = self._get_series(panel, self.IDX_CLOSE)
        vol_20 = self._get_series(panel, self.IDX_VOLATILITY)
        engine = self.engine
        feats = []

        # --- Amihud illiquidity: |ret| / volume (3 dims) ---
        for d in [5, 20, 60]:
            amihud = engine.ts_sum(ret.abs(), d, mask_2d) / (
                engine.ts_sum(volume, d, mask_2d).clamp(min=1e-6)
            )
            feats.append(amihud)

        # --- Amihud EWMA-smoothed (2 dims) ---
        amihud_20 = feats[-2]  # the 20-day Amihud
        feats.append(engine.ewma(amihud_20, 10, mask_2d))
        feats.append(engine.ts_delta(amihud_20, 5, mask_2d))

        # --- Price impact proxy: |ret| / sqrt(volume) (2 dims) ---
        for d in [5, 20]:
            impact = engine.ts_sum(ret.abs(), d, mask_2d) / (
                engine.ts_sum(volume, d, mask_2d).sqrt().clamp(min=1e-6)
            )
            feats.append(impact)

        # --- Spread proxy: vol / close (2 dims) ---
        spread = vol_20 / close.clamp(min=1e-6)
        feats.append(spread)
        feats.append(engine.ewma(spread, 10, mask_2d))

        # --- Realized spread: |ret| / close (2 dims) ---
        realized = ret.abs() / close.clamp(min=1e-6)
        feats.append(realized)
        feats.append(engine.ewma(realized, 10, mask_2d))

        # --- Volume imbalance proxy: sign(ret) * volume / avg_volume (2 dims) ---
        avg_vol = engine.ewma(volume, 60, mask_2d)
        imbalance = ret.sign() * volume / avg_vol.clamp(min=1e-6)
        feats.append(imbalance)
        feats.append(engine.ewma(imbalance, 10, mask_2d))

        # --- Roll model proxy: auto-correlation of returns (2 dims) ---
        # Use corr(ret_t, ret_{t-1}) approximation
        ret_lag1 = torch.cat([ret[:1], ret[:-1]], dim=0)
        roll_corr = engine.corr(ret, ret_lag1, 20, mask_2d)
        feats.append(roll_corr)
        feats.append(engine.ewma(roll_corr, 20, mask_2d))

        # Fill remaining slots
        while len(feats) < 20:
            feats.append(amihud_20 * spread)

        stacked = torch.stack(feats[:20], dim=-1)
        return stacked

    # ════════════════════════════════════════════════════════════════════
    # Group 5: Cross-Sectional (30 dims)
    # ════════════════════════════════════════════════════════════════════

    def _cross_sectional(self, panel, mask_2d):
        """Rank, dispersion, z-score features across the asset universe."""
        ret = self._get_series(panel, self.IDX_RETURN)
        volume = self._get_series(panel, self.IDX_VOLUME)
        vol_20 = self._get_series(panel, self.IDX_VOLATILITY)
        engine = self.engine
        feats = []

        # --- Return rank (3 dims) ---
        for d in [1, 5, 20]:
            cum_ret = engine.ts_sum(ret, d, mask_2d)
            feats.append(engine.rank(cum_ret, mask_2d))

        # --- Volume rank (2 dims) ---
        feats.append(engine.rank(volume, mask_2d))
        feats.append(engine.rank(engine.ewma(volume, 20, mask_2d), mask_2d))

        # --- Volatility rank (2 dims) ---
        feats.append(engine.rank(vol_20, mask_2d))

        # --- Cross-sectional dispersion (3 dims) ---
        for d in [1, 5, 20]:
            cum_ret = engine.ts_sum(ret, d, mask_2d)
            # Dispersion = std of returns across assets at each t
            # Replace masked values with mean for dispersion computation
            cum_ret_filled = cum_ret.clone()
            for t in range(cum_ret.shape[0]):
                valid = mask_2d[t]
                if valid.sum() > 1:
                    mean_val = cum_ret[t, valid].mean()
                    cum_ret_filled[t, ~valid] = mean_val
            disp = cum_ret_filled.std(dim=-1, keepdim=True).expand_as(cum_ret)
            feats.append(disp)

        # --- Z-score (normalized position within cross-section) (3 dims) ---
        for w in [1, 5, 20]:
            cum_ret = engine.ts_sum(ret, w, mask_2d)
            zscore = torch.zeros_like(cum_ret)
            for t in range(cum_ret.shape[0]):
                valid = mask_2d[t]
                if valid.sum() > 1:
                    m = cum_ret[t, valid].mean()
                    s = cum_ret[t, valid].std()
                    if s > 1e-8:
                        zscore[t, valid] = (cum_ret[t, valid] - m) / s
            feats.append(zscore)

        # --- Momentum rank (3 dims) ---
        for d in [5, 20, 60]:
            mom = engine.ts_sum(ret, d, mask_2d)
            feats.append(engine.rank(mom, mask_2d))

        # --- Amihud rank (1 dim) ---
        amihud_20 = engine.ts_sum(ret.abs(), 20, mask_2d) / (
            engine.ts_sum(volume, 20, mask_2d).clamp(min=1e-6)
        )
        feats.append(engine.rank(amihud_20, mask_2d))

        # --- Cross-asset correlation proxy (2 dims) ---
        # Average pairwise correlation = dispersion of cumret / average vol
        for d in [5, 20]:
            cum_ret = engine.ts_sum(ret, d, mask_2d)
            cross_disp = torch.zeros_like(cum_ret)
            for t in range(cum_ret.shape[0]):
                valid = mask_2d[t]
                if valid.sum() > 1:
                    cross_disp[t, valid] = cum_ret[t, valid].std()
            avg_vol = vol_20.clamp(min=1e-8)
            feats.append(cross_disp / avg_vol)

        # Fill remaining slots
        while len(feats) < 30:
            feats.append(engine.rank(ret, mask_2d))

        stacked = torch.stack(feats[:30], dim=-1)
        return stacked

    # ════════════════════════════════════════════════════════════════════
    # Group 6: Momentum & Factor (30 dims)
    # ════════════════════════════════════════════════════════════════════

    def _momentum_factor(self, panel, mask_2d):
        """Momentum, reversal, RSI, Bollinger Band position, MACD."""
        ret = self._get_series(panel, self.IDX_RETURN)
        close = self._get_series(panel, self.IDX_CLOSE)
        engine = self.engine
        feats = []

        # --- Momentum (3 dims) ---
        for d in [5, 20, 60]:
            feats.append(engine.ts_sum(ret, d, mask_2d))

        # --- Short-term reversal (2 dims) ---
        for d in [1, 3]:
            feats.append(-engine.ts_sum(ret, d, mask_2d))  # negative → reversal

        # --- RSI proxy: up/down EWMA ratio (3 dims) ---
        for w in [5, 14, 20]:
            ups = ret.clamp(min=0)
            downs = ret.clamp(max=0).abs()
            up_ewma = engine.ewma(ups, w, mask_2d)
            down_ewma = engine.ewma(downs, w, mask_2d)
            rs = up_ewma / down_ewma.clamp(min=1e-8)
            rsi = 100.0 - 100.0 / (1.0 + rs)
            feats.append(rsi / 100.0)  # normalize to [0, 1]

        # --- Bollinger Band position (2 dims) ---
        for w in [20, 60]:
            ma = engine.ts_mean(close, w, mask_2d)
            std = engine.ts_std(close, w, mask_2d)
            bb_pos = (close - ma) / (2.0 * std.clamp(min=1e-8))
            feats.append(bb_pos.clamp(-3, 3))
            feats.append(std / ma.clamp(min=1e-6))  # BB width

        # --- MACD signal (2 dims) ---
        ema_12 = engine.ewma(close, 12, mask_2d)
        ema_26 = engine.ewma(close, 26, mask_2d)
        macd_line = ema_12 - ema_26
        macd_signal = engine.ewma(macd_line, 9, mask_2d)
        macd_hist = macd_line - macd_signal
        feats.append(macd_line / close.clamp(min=1e-6))
        feats.append(macd_hist / close.clamp(min=1e-6))

        # --- Rate of change (ROC) (3 dims) ---
        for d in [5, 10, 20]:
            roc = engine.ts_delta(close, d, mask_2d) / close.clamp(min=1e-6)
            feats.append(roc)

        # --- Moving average crossovers (3 dims) ---
        for s, l in [(5, 20), (10, 30), (20, 60)]:
            ma_s = engine.ts_mean(close, s, mask_2d)
            ma_l = engine.ts_mean(close, l, mask_2d)
            feats.append((ma_s - ma_l) / ma_l.clamp(min=1e-6))

        # --- Trend strength: abs(MA_diff) / vol (2 dims) ---
        for s, l in [(5, 20), (20, 60)]:
            ma_s = engine.ts_mean(close, s, mask_2d)
            ma_l = engine.ts_mean(close, l, mask_2d)
            trend = (ma_s - ma_l).abs() / engine.ts_std(ret, 20, mask_2d).clamp(min=1e-8)
            feats.append(trend)

        # Fill remaining slots
        while len(feats) < 30:
            feats.append(engine.ts_sum(ret, 10, mask_2d))

        stacked = torch.stack(feats[:30], dim=-1)
        return stacked

    # ════════════════════════════════════════════════════════════════════
    # Forward
    # ════════════════════════════════════════════════════════════════════

    def forward(self, panel) -> torch.Tensor:
        """Compute the 200-dimensional market state vector for all time steps.

        Parameters
        ----------
        panel : Panel
            Raw market data panel with values of shape (T, N, F).
            Minimum F = 5: [close, log_return, volume, volume_ratio, volatility_20].

        Returns
        -------
        s_t : torch.Tensor, shape (T, N, 200)
            Market state vectors. T = time steps, N = assets.
            Values at masked (non-tradable) positions are zeroed.
        """
        T, N, F = panel.values.shape
        device = panel.device
        mask_2d = self._panel_mask_2d(panel)  # (T, N)

        # Compute each feature group
        groups = [
            self._price_dynamics(panel, mask_2d),       # (T, N, 45)
            self._volume_dynamics(panel, mask_2d),       # (T, N, 35)
            self._volatility_structure(panel, mask_2d),  # (T, N, 40)
            self._microstructure(panel, mask_2d),        # (T, N, 20)
            self._cross_sectional(panel, mask_2d),       # (T, N, 30)
            self._momentum_factor(panel, mask_2d),       # (T, N, 30)
        ]

        # Concatenate along feature dimension
        s_t = torch.cat(groups, dim=-1)  # (T, N, 200)

        # Safety: clip extreme values and replace NaN/Inf
        s_t = torch.nan_to_num(s_t, nan=0.0, posinf=10.0, neginf=-10.0)
        s_t = s_t.clamp(-50.0, 50.0)

        # Apply mask: zero-out non-tradable positions
        s_t[~mask_2d] = 0.0

        return s_t
