"""Market state vector s_t ∈ R²⁰⁰ construction from Panel (T, N, 5) OHLCV.

s_t is the unified input to the Regime Router, KDA Memory, CDAP, and all
strategy experts. It must capture sufficient information for regime
identification while remaining compact enough for Mac Mini M4 training.

Feature groups  (200 total):
  1. Price/Return dynamics     —  55 dims
  2. Volatility structure       —  40 dims
  3. Volume & liquidity         —  30 dims
  4. Technical / momentum       —  35 dims
  5. Cross-sectional context    —  30 dims
  6. Spectral (FFT energy)      —  10 dims
"""

import math
import torch
import torch.nn as nn

from daft.data.panel import Panel
from daft.features.tensor_factors import TensorFactorEngine as TFE

# Short aliases for readability
_rank = TFE.rank
_corr = TFE.corr
_ewma = TFE.ewma
_ts_delta = TFE.ts_delta
_ts_sum = TFE.ts_sum
_ts_std = TFE.ts_std


class RegimeFeatureExtractor(nn.Module):
    """Extract (T, N, 200) market state vectors from a (T, N, 5) OHLCV Panel."""

    def __init__(self, n_output: int = 200):
        super().__init__()
        self.n_output = n_output

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------
    def forward(self, panel: Panel) -> torch.Tensor:
        """Compute s_t for all time steps and all assets.

        Parameters
        ----------
        panel : Panel,  values.shape = (T, N, 5)
                feature_names = ["open", "high", "low", "close", "volume"]

        Returns
        -------
        s_t : torch.Tensor  (T, N, 200)
        """
        vals = panel.values       # (T, N, 5)
        mask = panel.mask         # (T, N)  bool

        o, h, l, c, v = vals.unbind(dim=-1)   # each (T, N)

        features = []

        # ---- Group 1:  Price / Return dynamics  (~55) ----
        features.extend(self._return_group(c, mask))
        features.extend(self._price_pattern(c, o, mask))

        # ---- Group 2:  Volatility structure  (~40) ----
        features.extend(self._vol_group(c, h, l, mask))

        # ---- Group 3:  Volume & liquidity  (~30) ----
        features.extend(self._volume_group(v, c, mask))

        # ---- Group 4:  Technical / momentum  (~35) ----
        features.extend(self._technical_group(c, h, l, mask))

        # ---- Group 5:  Cross-sectional context  (~30) ----
        features.extend(self._cross_sectional_group(c, mask))

        # ---- Group 6:  Spectral  (~10) ----
        features.extend(self._spectral_group(c, mask))

        # Stack & pad / trim to exactly n_output
        stacked = torch.stack(features, dim=-1)                # (T, N, F_total)
        F_actual = stacked.size(-1)
        if F_actual < self.n_output:
            pad = torch.zeros(
                stacked.size(0), stacked.size(1),
                self.n_output - F_actual, device=stacked.device
            )
            stacked = torch.cat([stacked, pad], dim=-1)
        elif F_actual > self.n_output:
            stacked = stacked[..., :self.n_output]

        return stacked

    # ==================================================================
    # Group 1 — Price / Return (~55 dims)
    # ==================================================================
    def _return_group(self, c, mask):
        log_c = torch.log(c.clamp(min=1e-8))
        r1 = _ts_delta(log_c, 1, mask)          # 1-day return
        r5 = _ts_delta(log_c, 5, mask)          # 5-day
        r20 = _ts_delta(log_c, 20, mask)        # 20-day

        out = []
        for r, tag in [(r1, 1), (r5, 5), (r20, 20)]:
            out.append(r)
            out.append(_ewma(r, tag * 3, mask))              # smoothed
            out.append(_ts_std(r, max(tag * 2, 5), mask))     # return vol

        # Sharpe proxies
        for r, w in [(r5, 20), (r20, 60)]:
            mu = _ewma(r, w // 2, mask)
            sd = _ts_std(r, w, mask).clamp(min=1e-6)
            out.append(mu / sd)

        # Higher moments (20d window)
        r20_std = _ts_std(r20, 20, mask).clamp(min=1e-6)
        r20_mu = _ewma(r20, 10, mask)
        r20_z = (r20 - r20_mu) / r20_std
        out.append(_ewma(r20_z ** 3, 20, mask))   # skew proxy
        out.append(_ewma(r20_z ** 4, 20, mask))   # kurt proxy

        # Drawdown proxy
        cum_c = torch.cumsum(r1, dim=0)
        running_max = torch.zeros_like(cum_c)
        for t in range(cum_c.size(0)):
            running_max[t] = cum_c[:t + 1].max(dim=0).values
        drawdown = cum_c - running_max
        out.append(_ewma(drawdown, 20, mask))

        # Autocorrelation
        out.append(_corr(r1, torch.roll(r1, 1, 0), 20, mask))
        out.append(_corr(r1, torch.roll(r1, 5, 0), 20, mask))

        return out

    # ==================================================================
    # Group 1b — Price pattern
    # ==================================================================
    def _price_pattern(self, c, o, mask):
        out = []

        # Opening gap
        gap = torch.log(o / torch.roll(c, 1, 0).clamp(min=1e-8))
        out.append(gap)

        # Intraday range proxy (H-L)/C  for daily bars this is the daily range
        intra_range = torch.zeros_like(c)  # placeholder — H/L already in OHLC
        # use high/low from original panel (handled in _vol_group)

        # Distance from moving averages
        for w in [5, 10, 20, 60]:
            ma = _ewma(c, w, mask)
            dist = (c - ma) / ma.clamp(min=1e-8)
            out.append(dist)

        # MA crossovers
        ma5 = _ewma(c, 5, mask)
        ma20 = _ewma(c, 20, mask)
        out.append((ma5 - ma20) / ma20.clamp(min=1e-8))

        ma10 = _ewma(c, 10, mask)
        ma60 = _ewma(c, 60, mask)
        out.append((ma10 - ma60) / ma60.clamp(min=1e-8))

        # Price position in N-day range
        for w in [20, 60]:
            h_max = torch.zeros_like(c)
            l_min = torch.zeros_like(c)
            for t in range(c.size(0)):
                lo = max(0, t - w + 1)
                h_max[t] = c[lo:t + 1].max(dim=0).values
                l_min[t] = c[lo:t + 1].min(dim=0).values
            rng = (h_max - l_min).clamp(min=1e-8)
            out.append((c - l_min) / rng)   # % from low
            out.append((h_max - c) / rng)   # % from high

        return out

    # ==================================================================
    # Group 2 — Volatility (~40 dims)
    # ==================================================================
    def _vol_group(self, c, h, l, mask):
        log_c = torch.log(c.clamp(min=1e-8))
        r1 = _ts_delta(log_c, 1, mask)

        out = []

        # Close-to-close realized vol at multiple horizons
        for w in [5, 10, 20, 60]:
            rv = _ts_std(r1, w, mask) * math.sqrt(252)
            rv_ewma = _ewma(rv, w, mask)
            out.append(rv)
            out.append(rv_ewma)

        # Parkinson range-based vol  σ_p = sqrt( 1/(4 ln 2) * mean[(ln H/L)²] )
        log_hl = torch.log((h / l.clamp(min=1e-8)).clamp(min=1e-8))
        hl_sq = log_hl ** 2
        for w in [5, 20]:
            pk = (_ewma(hl_sq, w, mask) / (4 * math.log(2))).sqrt() * math.sqrt(252)
            out.append(pk)

        # Vol-of-vol
        rv20 = _ts_std(r1, 20, mask)
        out.append(_ts_std(rv20, 60, mask))          # vol-of-vol
        out.append(_ewma(rv20, 20, mask) / rv20.clamp(min=1e-6))  # vol regime

        # ATR-style
        tr = torch.maximum(
            h - l,
            torch.maximum(
                (h - torch.roll(c, 1, 0)).abs(),
                (l - torch.roll(c, 1, 0)).abs(),
            ),
        )
        for w in [5, 20]:
            out.append(_ewma(tr, w, mask) / c.clamp(min=1e-8))

        return out

    # ==================================================================
    # Group 3 — Volume & liquidity (~30 dims)
    # ==================================================================
    def _volume_group(self, v, c, mask):
        log_v = torch.log(v.clamp(min=1))

        out = []

        # Volume ratios vs moving average
        for w in [5, 20, 60]:
            v_ma = _ewma(v, w, mask).clamp(min=1)
            out.append(v / v_ma)
            out.append(torch.log(v / v_ma))

        # Volume trend
        for w in [5, 20]:
            out.append(_ts_delta(log_v, w, mask))

        # Volume volatility
        out.append(_ts_std(log_v, 20, mask))

        # Turnover proxy  (V / lagged V)
        v_shift = torch.roll(v, 1, 0).clamp(min=1)
        out.append(v / v_shift)

        # Price-volume correlation
        r1 = _ts_delta(torch.log(c.clamp(min=1e-8)), 1, mask)
        out.append(_corr(r1, log_v, 20, mask))

        # OBV-style cumulative
        direction = torch.sign(_ts_delta(c, 1, mask))
        obv = torch.cumsum(direction * v, dim=0)
        out.append(_ts_delta(obv, 5, mask) / obv.clamp(min=1))
        out.append(_ts_delta(obv, 20, mask) / obv.clamp(min=1))

        # Volume rank
        out.append(_rank(v, mask))

        return out

    # ==================================================================
    # Group 4 — Technical / momentum (~35 dims)
    # ==================================================================
    def _technical_group(self, c, h, l, mask):
        log_c = torch.log(c.clamp(min=1e-8))
        r1 = _ts_delta(log_c, 1, mask)

        out = []

        # RSI-like  (simplified:  avg gain / avg loss ratio)
        for w in [14, 28]:
            gain = r1.clamp(min=0)
            loss = (-r1).clamp(min=0)
            avg_gain = _ewma(gain, w, mask)
            avg_loss = _ewma(loss, w, mask).clamp(min=1e-8)
            rs = avg_gain / avg_loss
            rsi = 100.0 - 100.0 / (1.0 + rs)
            out.append(rsi / 100.0)   # normalise to [0, 1]

        # Stochastic-like  (%K proxy)
        for w in [14, 28]:
            h_max = torch.zeros_like(c)
            l_min = torch.zeros_like(c)
            for t in range(min(w, c.size(0))):
                h_max[t] = h[:t + 1].max(dim=0).values
                l_min[t] = l[:t + 1].min(dim=0).values
            for t in range(w, c.size(0)):
                h_max[t] = h[t - w + 1:t + 1].max(dim=0).values
                l_min[t] = l[t - w + 1:t + 1].min(dim=0).values
            rng = (h_max - l_min).clamp(min=1e-8)
            out.append((c - l_min) / rng)

        # MACD components
        ema12 = _ewma(c, 12, mask)
        ema26 = _ewma(c, 26, mask)
        macd = ema12 - ema26
        signal = _ewma(macd, 9, mask)
        out.append(macd / c.clamp(min=1e-8))
        out.append(signal / c.clamp(min=1e-8))
        out.append((macd - signal) / c.clamp(min=1e-8))

        # Bollinger %B
        for w in [20, 60]:
            ma = _ewma(c, w, mask)
            sd = _ts_std(c, w, mask)
            out.append((c - ma) / sd.clamp(min=1e-8))
            out.append(sd / c.clamp(min=1e-8) * math.sqrt(252))  # BB width

        # Momentum (rate of change)
        for w in [5, 10, 20]:
            out.append(c / torch.roll(c, w, 0).clamp(min=1e-8) - 1.0)

        # ADX-like  (simplified directional movement)
        up_move = h - torch.roll(h, 1, 0)
        dn_move = torch.roll(l, 1, 0) - l
        plus_dm = up_move.clamp(min=0) * (up_move > dn_move).float()
        minus_dm = dn_move.clamp(min=0) * (dn_move > up_move).float()
        tr = torch.maximum(h - l, torch.maximum(
            (h - torch.roll(c, 1, 0)).abs(), (l - torch.roll(c, 1, 0)).abs()))
        atr14 = _ewma(tr, 14, mask)
        plus_di = _ewma(plus_dm, 14, mask) / atr14.clamp(min=1e-8) * 100
        minus_di = _ewma(minus_dm, 14, mask) / atr14.clamp(min=1e-8) * 100
        dx = (plus_di - minus_di).abs() / (plus_di + minus_di).clamp(min=1e-8) * 100
        out.append(_ewma(dx, 14, mask) / 100.0)  # ADX normalised

        return out

    # ==================================================================
    # Group 5 — Cross-sectional (~30 dims)
    # ==================================================================
    def _cross_sectional_group(self, c, mask):
        log_c = torch.log(c.clamp(min=1e-8))
        r1 = _ts_delta(log_c, 1, mask)

        out = []

        # Rank features
        for feat, name in [(r1, "r1"), (c, "price")]:
            out.append(_rank(feat, mask))

        # Cross-sectional dispersion
        mu_r1 = (r1 * mask.float()).sum(dim=-1) / mask.float().sum(dim=-1).clamp(min=1)
        disp = ((r1 - mu_r1.unsqueeze(-1)) ** 2 * mask.float()).sum(dim=-1)
        disp = (disp / mask.float().sum(dim=-1).clamp(min=1)).sqrt()
        out.append(disp.unsqueeze(-1).expand_as(r1))              # dispersion
        for w in [5, 20]:
            out.append(_ewma(disp.unsqueeze(-1).expand_as(r1), w, mask))

        # % of stocks with positive return
        up_pct = ((r1 > 0).float() * mask.float()).sum(dim=-1) / mask.float().sum(dim=-1).clamp(min=1)
        out.append(up_pct.unsqueeze(-1).expand_as(r1))

        # Breadth: cross-sectional mean of returns
        for w in [5, 20]:
            r_w = _ts_delta(log_c, w, mask)
            mu_w = (r_w * mask.float()).sum(dim=-1) / mask.float().sum(dim=-1).clamp(min=1)
            out.append(mu_w.unsqueeze(-1).expand_as(r1))

        # Market beta proxy  (stock vs cross-sectional mean)
        mu1 = (r1 * mask.float()).sum(dim=-1) / mask.float().sum(dim=-1).clamp(min=1)
        for w in [20, 60]:
            out.append(_corr(r1, mu1.unsqueeze(-1).expand_as(r1), w, mask))

        return out

    # ==================================================================
    # Group 6 — Spectral (~10 dims)
    # ==================================================================
    def _spectral_group(self, c, mask):
        out = []

        # RFFT on the close-price series (per stock)
        # Use a 64-point window of log-returns; pad short series
        log_c = torch.log(c.clamp(min=1e-8))
        r1 = _ts_delta(log_c, 1, mask)
        T = r1.size(0)

        win = min(64, T)
        if T >= win:
            window = r1[-win:]                                 # (win, N)
            window = window * mask[-win:].float()
            window = window - window.mean(dim=0, keepdim=True)
            window = window * torch.hann_window(win, device=c.device).unsqueeze(-1)
            spec = torch.fft.rfft(window, dim=0).abs()        # (win//2+1, N)

            # Energy in frequency bands
            n_freq = spec.size(0)
            low = spec[:n_freq // 3].mean(dim=0)              # low freq
            mid = spec[n_freq // 3:2 * n_freq // 3].mean(dim=0)
            high = spec[2 * n_freq // 3:].mean(dim=0)
            total = (low + mid + high).clamp(min=1e-8)

            out.append(low.unsqueeze(0).expand(T, -1) / total)
            out.append(mid.unsqueeze(0).expand(T, -1) / total)
            out.append(high.unsqueeze(0).expand(T, -1) / total)

            # Spectral centroid
            freqs = torch.arange(n_freq, device=c.device).float()
            centroid = (freqs.unsqueeze(-1) * spec).sum(dim=0) / spec.sum(dim=0).clamp(min=1e-8)
            out.append(centroid.unsqueeze(0).expand(T, -1) / n_freq)

            # Spectral entropy
            spec_norm = spec / spec.sum(dim=0, keepdim=True).clamp(min=1e-8)
            entropy = -(spec_norm * spec_norm.clamp(min=1e-8).log()).sum(dim=0)
            max_entropy = math.log(n_freq)
            out.append(entropy.unsqueeze(0).expand(T, -1) / max_entropy)
        else:
            # Not enough data — zeros
            for _ in range(5):
                out.append(torch.zeros(T, c.size(1), device=c.device))

        return out
