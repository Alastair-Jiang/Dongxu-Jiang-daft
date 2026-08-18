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

A5 (2026-08-18, K3 纲领): 特征清单改为命名注册表 + 显式零填充。
原实现各组在特征数不足槽位宽度时用循环反复追加同一表达式补齐，
200 维中 70 维是纯复制列、另有多处跨组/组内同值重复——虚假放大特征
维度并放大共线性。修复后: 每个特征具名(feature_names), 跨组与组内
同值重复列删除, 不足槽位填零并命名 g{k}_pad_XX, 由 real_feature_mask
显式标注。填充列恒为 0(不携带信息、不放大共线性), 下游 200 维契约不变。
"""

import torch
import torch.nn as nn

from daft.features.tensor_factors import TensorFactorEngine
from daft.features.base_features import ensure_base_panel


# A5: 各组真实特征数(日频 lookback_scale=1.0 口径; 周线缩放下窗口塌缩
# 会进一步减少, 由注册表按名去重自动处理)
A5_GROUP_SIZES = [45, 35, 40, 20, 30, 30]  # 槽位宽度(下游契约, 不变)
A5_REAL_COUNTS_DAILY = {"g1": 27, "g2": 20, "g3": 20,
                        "g4": 14, "g5": 15, "g6": 19}  # 共 115


class RegimeFeatureExtractor(nn.Module):
    """Extract the 200-dimensional market state vector s_t.

    This is the input to the Regime Router and all downstream components.
    The design follows the principle: capture enough information for regime
    identification without overfitting to noise.

    The 200 dimension slots are grouped into 6 families
    (A5 修复后: 真实特征数 + 显式零填充, 替代原 while 重复列填充):
        Price Dynamics       (45 slots): 27 real + 18 pad
        Volume Dynamics      (35 slots): 20 real + 15 pad
        Volatility Structure (40 slots): 20 real + 20 pad
        Microstructure       (20 slots): 14 real +  6 pad
        Cross-Sectional      (30 slots): 15 real + 15 pad
        Momentum & Factor    (30 slots): 19 real + 11 pad
    日频共 115 个真实特征 + 85 个零填充列。填充列名 g{k}_pad_XX,
    由 self.real_feature_mask 标注(True=真实特征)。

    Parameters
    ----------
    n_base_factors : int
        Number of base factors expected in the panel. Default: 50.
        Currently the extractor uses 5 base features from the panel
        (close, log_return, volume, volume_ratio, volatility_20) and
        derives the remaining features.
    lookback_scale : float
        窗口缩放系数(K3 周线验证, 2026-08-17)。日频 1.0 不变;
        周频 0.2 → 所有窗口类 lookback ÷5(20 日动量→4 周)。
        窗口塌缩产生的同名特征按注册表自动去重(首现保留)。

    Attributes (A5, 每次 forward 后更新)
    ------------------------------------
    feature_names : list[str]
        200 个特征名(g{组}_{语义}_{窗口} 或 g{k}_pad_XX)。
    real_feature_mask : torch.Tensor
        (200,) bool, True=真实特征列, False=零填充列。
    n_real_features / n_padding : int
        真实/填充列数(之和恒为 200)。
    group_real_counts : dict[str, int]
        各组真实特征数(诊断/消融复核用)。
    """

    # Expected panel feature indices
    IDX_CLOSE = 0
    IDX_RETURN = 1
    IDX_VOLUME = 2
    IDX_VOLUME_RATIO = 3
    IDX_VOLATILITY = 4

    def __init__(
        self,
        n_base_factors: int = 50,
        output_dim: int = 200,
        lookback_scale: float = 1.0,
    ):
        super().__init__()
        self.n_base_factors = n_base_factors
        self.output_dim = output_dim
        self.lookback_scale = lookback_scale
        self.engine = TensorFactorEngine()

        # A5: 特征注册表(每次 forward 后重算; 由配置决定, 无随机性)
        self.feature_names: list = []
        self.real_feature_mask = None
        self.n_real_features = 0
        self.n_padding = 0
        self.group_real_counts: dict = {}

    def _w(self, d: int) -> int:
        """窗口缩放：日频 scale=1.0 不变；周频 scale=0.2 → 20 日→4 周。

        短窗口(min 后为 1)在周频下退化为「1 周」, 与日频的「1 日」语义对齐。
        """
        return max(1, int(round(d * self.lookback_scale)))

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

    def _pad_and_stack(self, feats, target: int, tag: str):
        """A5: 命名注册表 + 显式零填充(替代 while 重复列填充)。

        - 组内同名特征去重(首现保留)——兜底周线窗口塌缩产生的
          同名同值列(如 _w(1)=_w(2)=_w(3)=1 时 ret_cum_1 重复)。
        - 不足槽位填零列并命名 {tag}_pad_XX, 不再复制已有表达式。
        - 真实特征超出槽位时截断(防御, 当前配置不会触发)。
        """
        seen = set()
        unique = []
        for name, tensor in feats:
            if name in seen:
                continue
            seen.add(name)
            unique.append((name, tensor))
        n_real = len(unique)
        if n_real > target:
            unique = unique[:target]
            n_real = target
        names = [f"{tag}_{n}" for n, _ in unique]
        tensors = [t for _, t in unique]
        ref = tensors[0]
        for i in range(target - n_real):
            names.append(f"{tag}_pad_{i:02d}")
            tensors.append(torch.zeros_like(ref))
        stacked = torch.stack(tensors[:target], dim=-1)  # (T, N, target)
        return stacked, names

    # ════════════════════════════════════════════════════════════════════
    # Group 1: Price Dynamics (45 slots, 27 real)
    # ════════════════════════════════════════════════════════════════════

    def _price_dynamics(self, panel, mask_2d):
        """Multi-horizon returns, price-to-MA, acceleration."""
        ret = self._get_series(panel, self.IDX_RETURN)  # (T, N)
        close = self._get_series(panel, self.IDX_CLOSE)
        engine = self.engine
        feats = []

        # --- Multi-horizon returns (7) ---
        for d in (self._w(1), self._w(2), self._w(3), self._w(5), self._w(10),
                  self._w(20), self._w(60)):
            feats.append((f"ret_cum_{d}", engine.ts_sum(ret, d, mask_2d)))

        # --- Price relative to moving averages (4) ---
        for w in (self._w(5), self._w(10), self._w(20), self._w(60)):
            ma = engine.ts_mean(close, w, mask_2d)
            dev = (close - ma) / ma.clamp(min=1e-6)
            feats.append((f"ma_dev_{w}", dev))

        # --- Log price (1) ---
        feats.append(("log_price", close.clamp(min=1e-6).log()))

        # --- High-Low range proxy: vol / close (1) ---
        vol_20 = self._get_series(panel, self.IDX_VOLATILITY)
        feats.append(("hl_range", vol_20 / close.clamp(min=1e-6)))

        # --- Return acceleration: ret_t - ret_{t-d} (2) ---
        for d in (self._w(5), self._w(20)):
            feats.append((f"ret_accel_{d}", engine.ts_delta(ret, d, mask_2d)))

        # (A5 删除: 原"Cumulative returns" d=5/20 与 ret_cum_5/ret_cum_20
        #  同值重复; 原"multi-scale return volatility" 4 列与 G3 的
        #  ret_vol_{d} 跨组同值重复——统一由 G3 保留)

        # --- Return EWMA with different spans (3) ---
        for s in (self._w(3), self._w(10), self._w(30)):
            feats.append((f"ret_ewma_{s}", engine.ewma(ret, s, mask_2d)))

        # --- Price path signature: short/long MA ratio (3) ---
        for w_short, w_long in (
            (self._w(5), self._w(20)),
            (self._w(10), self._w(60)),
            (self._w(20), self._w(60)),
        ):
            ma_s = engine.ts_mean(close, w_short, mask_2d)
            ma_l = engine.ts_mean(close, w_long, mask_2d)
            feats.append((f"path_sig_{w_short}x{w_long}",
                          ma_s / ma_l.clamp(min=1e-6)))

        # --- Return skewness proxy (1) ---
        ret_ewma_30 = engine.ewma(ret, self._w(30), mask_2d)
        ret_std_20 = engine.ts_std(ret, self._w(20), mask_2d)
        feats.append(("ret_skew",
                      (ret - ret_ewma_30) / ret_std_20.clamp(min=1e-8)))

        # --- Upside/downside volatility ratio (4) ---
        for d in (self._w(10), self._w(20)):
            ups = ret.clamp(min=0)
            downs = ret.clamp(max=0).abs()
            up_std = engine.ts_std(ups, d, mask_2d)
            down_std = engine.ts_std(downs, d, mask_2d)
            feats.append((f"updown_ratio_{d}",
                          up_std / down_std.clamp(min=1e-8)))
            feats.append((f"updown_diff_{d}",
                          (up_std - down_std) / (up_std + down_std + 1e-8)))

        # --- Return autocorrelation proxy (1) ---
        feats.append(("ret_autocorr",
                      engine.ts_delta(ret_ewma_30, 1, mask_2d)))

        return self._pad_and_stack(feats, 45, "g1")

    # ════════════════════════════════════════════════════════════════════
    # Group 2: Volume Dynamics (35 slots, 20 real)
    # ════════════════════════════════════════════════════════════════════

    def _volume_dynamics(self, panel, mask_2d):
        """Volume ratios, volume trends, volume-price correlation, turnover."""
        volume = self._get_series(panel, self.IDX_VOLUME)
        vol_ratio = self._get_series(panel, self.IDX_VOLUME_RATIO)
        ret = self._get_series(panel, self.IDX_RETURN)
        engine = self.engine
        feats = []

        # --- Volume ratio at multiple horizons (3) ---
        for w in (self._w(5), self._w(20), self._w(60)):
            vol_ma = engine.ts_mean(volume, w, mask_2d)
            feats.append((f"vol_ratio_{w}", volume / vol_ma.clamp(min=1e-6)))

        # --- Volume trend: EWMA of volume (3) ---
        for s in (self._w(5), self._w(20), self._w(60)):
            feats.append((f"volume_ewma_{s}", engine.ewma(volume, s, mask_2d)))

        # --- Volume acceleration: delta of volume (2) ---
        for d in (self._w(5), self._w(20)):
            feats.append((f"volume_delta_{d}", engine.ts_delta(volume, d, mask_2d)))

        # --- Volume-price correlation (1) ---
        feats.append(("volpx_corr", engine.corr(volume, ret, self._w(20), mask_2d)))

        # --- Turnover proxy: volume_ratio + derived (3) ---
        feats.append(("turnover_ratio", vol_ratio))
        feats.append(("turnover_ewma", engine.ewma(vol_ratio, self._w(10), mask_2d)))
        feats.append(("turnover_std", engine.ts_std(vol_ratio, self._w(20), mask_2d)))

        # --- Volume volatility (3) ---
        for d in (self._w(5), self._w(20), self._w(60)):
            feats.append((f"volume_std_{d}", engine.ts_std(volume, d, mask_2d)))

        # --- Volume rank (2); A5: 删 w=1(ewma span-1=恒等, 与 G5 volume_rank 同值) ---
        for w in (self._w(5), self._w(20)):
            vol_smoothed = engine.ewma(volume, w, mask_2d)
            feats.append((f"volume_rank_{w}", engine.rank(vol_smoothed, mask_2d)))

        # --- Volume-return interaction (3) ---
        feats.append(("ret_x_turnover", ret * vol_ratio))
        feats.append(("absret_x_turnover", ret.abs() * vol_ratio))
        feats.append(("retturn_ewma",
                      engine.ewma(ret * vol_ratio, self._w(10), mask_2d)))

        return self._pad_and_stack(feats, 35, "g2")

    # ════════════════════════════════════════════════════════════════════
    # Group 3: Volatility Structure (40 slots, 20 real)
    # ════════════════════════════════════════════════════════════════════

    def _volatility_structure(self, panel, mask_2d):
        """Multi-window vol, vol-of-vol, return/vol ratios, higher moments."""
        ret = self._get_series(panel, self.IDX_RETURN)
        engine = self.engine
        feats = []

        # --- Multi-window volatility (4) — 本组为波动率特征的唯一出处
        #     (A5: G1 原重复的 4 列已删, 由这里保留) ---
        for d in (self._w(5), self._w(10), self._w(20), self._w(60)):
            feats.append((f"ret_vol_{d}", engine.ts_std(ret, d, mask_2d)))

        vol_20 = engine.ts_std(ret, self._w(20), mask_2d)
        vol_60 = engine.ts_std(ret, self._w(60), mask_2d)

        # --- Vol-of-vol (2) ---
        feats.append(("vol_of_vol_20", engine.ts_std(vol_20, self._w(60), mask_2d)))
        feats.append(("vol_of_vol_60", engine.ts_std(vol_60, self._w(60), mask_2d)))

        # (A5 删除: 原 vol_20 裸水平列与 ret_vol_20 同值重复)

        # --- Vol term structure (1) ---
        feats.append(("vol_term", vol_60 / vol_20.clamp(min=1e-8)))

        # --- Return-to-volatility ratios (4) ---
        for d in (self._w(5), self._w(10), self._w(20), self._w(60)):
            ret_cum = engine.ts_sum(ret, d, mask_2d)
            vol_d = engine.ts_std(ret, d, mask_2d)
            feats.append((f"sharpe_{d}", ret_cum / vol_d.clamp(min=1e-8)))

        # --- Volatility EWMA (3) ---
        for s in (self._w(5), self._w(20), self._w(60)):
            vol_s = engine.ts_std(ret, s, mask_2d)
            feats.append((f"vol_ewma_{s}",
                          engine.ewma(vol_s, self._w(30), mask_2d)))

        # --- Volatility acceleration (2) ---
        feats.append(("vol_accel_5", engine.ts_delta(vol_20, self._w(5), mask_2d)))
        feats.append(("vol_accel_20", engine.ts_delta(vol_20, self._w(20), mask_2d)))

        # --- Volatility regime: above/below long-term average (3) ---
        vol_lt_mean = engine.ewma(vol_20, self._w(120), mask_2d)
        feats.append(("vol_regime_ratio",
                      vol_20 / vol_lt_mean.clamp(min=1e-8)))
        feats.append(("vol_regime_flag", (vol_20 > vol_lt_mean).float()))
        feats.append(("vol_regime_sigmoid",
                      torch.sigmoid((vol_20 - vol_lt_mean)
                                    / vol_lt_mean.clamp(min=1e-8))))

        # --- Cross-sectional vol rank (1) ---
        feats.append(("vol_rank", engine.rank(vol_20, mask_2d)))

        return self._pad_and_stack(feats, 40, "g3")

    # ════════════════════════════════════════════════════════════════════
    # Group 4: Microstructure (20 slots, 14 real)
    # ════════════════════════════════════════════════════════════════════

    def _microstructure(self, panel, mask_2d):
        """Amihud illiquidity, price impact proxy, spread proxies."""
        ret = self._get_series(panel, self.IDX_RETURN)
        volume = self._get_series(panel, self.IDX_VOLUME)
        close = self._get_series(panel, self.IDX_CLOSE)
        vol_20 = self._get_series(panel, self.IDX_VOLATILITY)
        engine = self.engine
        feats = []

        # --- Amihud illiquidity: |ret| / volume (3) ---
        amihud = {}
        for d in (self._w(5), self._w(20), self._w(60)):
            amihud[d] = engine.ts_sum(ret.abs(), d, mask_2d) / (
                engine.ts_sum(volume, d, mask_2d).clamp(min=1e-6)
            )
            feats.append((f"amihud_{d}", amihud[d]))
        amihud_20 = amihud[self._w(20)]

        # --- Amihud EWMA-smoothed (2) ---
        feats.append(("amihud_ewma", engine.ewma(amihud_20, self._w(10), mask_2d)))
        feats.append(("amihud_delta", engine.ts_delta(amihud_20, self._w(5), mask_2d)))

        # --- Price impact proxy: |ret| / sqrt(volume) (2) ---
        for d in (self._w(5), self._w(20)):
            impact = engine.ts_sum(ret.abs(), d, mask_2d) / (
                engine.ts_sum(volume, d, mask_2d).sqrt().clamp(min=1e-6)
            )
            feats.append((f"impact_{d}", impact))

        # (A5 删除: 原 spread = vol_20/close 裸列与 G1 hl_range 同值重复;
        #  其 EWMA 保留为独立特征)
        spread = vol_20 / close.clamp(min=1e-6)
        feats.append(("spread_ewma", engine.ewma(spread, self._w(10), mask_2d)))

        # --- Realized spread: |ret| / close (2) ---
        realized = ret.abs() / close.clamp(min=1e-6)
        feats.append(("realized_spread", realized))
        feats.append(("realized_ewma", engine.ewma(realized, self._w(10), mask_2d)))

        # --- Volume imbalance proxy (2) ---
        avg_vol = engine.ewma(volume, self._w(60), mask_2d)
        imbalance = ret.sign() * volume / avg_vol.clamp(min=1e-6)
        feats.append(("imbalance", imbalance))
        feats.append(("imbalance_ewma", engine.ewma(imbalance, self._w(10), mask_2d)))

        # --- Roll model proxy: autocorrelation of returns (2) ---
        ret_lag1 = torch.cat([ret[:1], ret[:-1]], dim=0)
        roll_corr = engine.corr(ret, ret_lag1, self._w(20), mask_2d)
        feats.append(("roll_corr", roll_corr))
        feats.append(("roll_ewma", engine.ewma(roll_corr, self._w(20), mask_2d)))

        return self._pad_and_stack(feats, 20, "g4")

    # ════════════════════════════════════════════════════════════════════
    # Group 5: Cross-Sectional (30 slots, 15 real)
    # ════════════════════════════════════════════════════════════════════

    def _cross_sectional(self, panel, mask_2d):
        """Rank, dispersion, z-score features across the asset universe."""
        ret = self._get_series(panel, self.IDX_RETURN)
        volume = self._get_series(panel, self.IDX_VOLUME)
        vol_20 = self._get_series(panel, self.IDX_VOLATILITY)
        engine = self.engine
        feats = []

        # --- Return rank (3) ---
        for d in (self._w(1), self._w(5), self._w(20)):
            cum_ret = engine.ts_sum(ret, d, mask_2d)
            feats.append((f"ret_rank_{d}", engine.rank(cum_ret, mask_2d)))

        # --- Volume rank (1); A5: 删 rank(ewma(vol,20)) (与 G2 volume_rank_20
        #     跨组同值重复) ---
        feats.append(("volume_rank", engine.rank(volume, mask_2d)))

        # --- Volatility rank (1) ---
        feats.append(("vol20_rank", engine.rank(vol_20, mask_2d)))

        # --- Cross-sectional dispersion (3) ---
        for d in (self._w(1), self._w(5), self._w(20)):
            cum_ret = engine.ts_sum(ret, d, mask_2d)
            cum_ret_filled = cum_ret.clone()
            for t in range(cum_ret.shape[0]):
                valid = mask_2d[t]
                if valid.sum() > 1:
                    mean_val = cum_ret[t, valid].mean()
                    cum_ret_filled[t, ~valid] = mean_val
            disp = cum_ret_filled.std(dim=-1, keepdim=True).expand_as(cum_ret)
            feats.append((f"xsec_disp_{d}", disp))

        # --- Z-score: normalized position within cross-section (3) ---
        for w in (self._w(1), self._w(5), self._w(20)):
            cum_ret = engine.ts_sum(ret, w, mask_2d)
            zscore = torch.zeros_like(cum_ret)
            for t in range(cum_ret.shape[0]):
                valid = mask_2d[t]
                if valid.sum() > 1:
                    m = cum_ret[t, valid].mean()
                    s = cum_ret[t, valid].std()
                    if s > 1e-8:
                        zscore[t, valid] = (cum_ret[t, valid] - m) / s
            feats.append((f"zscore_{w}", zscore))

        # --- Momentum rank (1); A5: 删 d=5/20 (与 ret_rank_5/20 同值重复) ---
        mom = engine.ts_sum(ret, self._w(60), mask_2d)
        feats.append(("mom_rank_60", engine.rank(mom, mask_2d)))

        # --- Amihud rank (1) ---
        amihud_20 = engine.ts_sum(ret.abs(), self._w(20), mask_2d) / (
            engine.ts_sum(volume, self._w(20), mask_2d).clamp(min=1e-6)
        )
        feats.append(("amihud_rank", engine.rank(amihud_20, mask_2d)))

        # --- Cross-asset correlation proxy (2) ---
        for d in (self._w(5), self._w(20)):
            cum_ret = engine.ts_sum(ret, d, mask_2d)
            cross_disp = torch.zeros_like(cum_ret)
            for t in range(cum_ret.shape[0]):
                valid = mask_2d[t]
                if valid.sum() > 1:
                    cross_disp[t, valid] = cum_ret[t, valid].std()
            avg_vol = vol_20.clamp(min=1e-8)
            feats.append((f"corr_proxy_{d}", cross_disp / avg_vol))

        return self._pad_and_stack(feats, 30, "g5")

    # ════════════════════════════════════════════════════════════════════
    # Group 6: Momentum & Factor (30 slots, 19 real)
    # ════════════════════════════════════════════════════════════════════

    def _momentum_factor(self, panel, mask_2d):
        """Reversal, RSI, Bollinger Band position, MACD, ROC, MA crossovers."""
        ret = self._get_series(panel, self.IDX_RETURN)
        close = self._get_series(panel, self.IDX_CLOSE)
        engine = self.engine
        feats = []

        # (A5 删除: 原 momentum d=5/20/60 三列 = ts_sum(ret,d),
        #  与 G1 ret_cum_{d} 跨组同值重复; 动量信息由 G1 承载)

        # --- Short-term reversal (2); 与 G1 ret_cum 同窗反号, 信息独立保留 ---
        for d in (self._w(1), self._w(3)):
            feats.append((f"rev_{d}", -engine.ts_sum(ret, d, mask_2d)))

        # --- RSI proxy: up/down EWMA ratio (3) ---
        for w in (self._w(5), self._w(14), self._w(20)):
            ups = ret.clamp(min=0)
            downs = ret.clamp(max=0).abs()
            up_ewma = engine.ewma(ups, w, mask_2d)
            down_ewma = engine.ewma(downs, w, mask_2d)
            rs = up_ewma / down_ewma.clamp(min=1e-8)
            rsi = 100.0 - 100.0 / (1.0 + rs)
            feats.append((f"rsi_{w}", rsi / 100.0))

        # --- Bollinger Band position (4) ---
        for w in (self._w(20), self._w(60)):
            ma = engine.ts_mean(close, w, mask_2d)
            std = engine.ts_std(close, w, mask_2d)
            bb_pos = (close - ma) / (2.0 * std.clamp(min=1e-8))
            feats.append((f"bb_pos_{w}", bb_pos.clamp(-3, 3)))
            feats.append((f"bb_width_{w}", std / ma.clamp(min=1e-6)))

        # --- MACD signal (2) ---
        ema_12 = engine.ewma(close, self._w(12), mask_2d)
        ema_26 = engine.ewma(close, self._w(26), mask_2d)
        macd_line = ema_12 - ema_26
        macd_signal = engine.ewma(macd_line, self._w(9), mask_2d)
        macd_hist = macd_line - macd_signal
        feats.append(("macd_line", macd_line / close.clamp(min=1e-6)))
        feats.append(("macd_hist", macd_hist / close.clamp(min=1e-6)))

        # --- Rate of change (ROC) (3) ---
        for d in (self._w(5), self._w(10), self._w(20)):
            roc = engine.ts_delta(close, d, mask_2d) / close.clamp(min=1e-6)
            feats.append((f"roc_{d}", roc))

        # --- Moving average crossovers (3) ---
        for s, l in (
            (self._w(5), self._w(20)),
            (self._w(10), self._w(30)),
            (self._w(20), self._w(60)),
        ):
            ma_s = engine.ts_mean(close, s, mask_2d)
            ma_l = engine.ts_mean(close, l, mask_2d)
            feats.append((f"ma_cross_{s}x{l}",
                          (ma_s - ma_l) / ma_l.clamp(min=1e-6)))

        # --- Trend strength: |MA_diff| / vol (2) ---
        for s, l in ((self._w(5), self._w(20)), (self._w(20), self._w(60))):
            ma_s = engine.ts_mean(close, s, mask_2d)
            ma_l = engine.ts_mean(close, l, mask_2d)
            trend = (ma_s - ma_l).abs() / engine.ts_std(
                ret, self._w(20), mask_2d).clamp(min=1e-8)
            feats.append((f"trend_{s}x{l}", trend))

        return self._pad_and_stack(feats, 30, "g6")

    # ════════════════════════════════════════════════════════════════════
    # Forward
    # ════════════════════════════════════════════════════════════════════

    def forward(self, panel) -> torch.Tensor:
        """Compute the 200-dimensional market state vector for all time steps.

        Parameters
        ----------
        panel : Panel
            Raw market data panel with values of shape (T, N, F).
            接受 OHLCV 布局或基础布局
            [close, log_return, volume, volume_ratio, volatility_20];
            其他布局抛错(通道契约见 features/base_features.py)。

        Returns
        -------
        s_t : torch.Tensor, shape (T, N, 200)
            Market state vectors. T = time steps, N = assets.
            Values at masked (non-tradable) positions are zeroed.
            A5: 未被真实特征占据的槽位为显式零填充列(见
            self.feature_names / self.real_feature_mask)。
        """
        # ── 通道契约: 统一为基础特征布局 (2026-08-16 修复) ──
        # 修复前数据源的 OHLCV 列被直接当作 [close, log_return, ...] 读取,
        # 所有 s_t 建立在错列上。这里先做唯一转换。
        # 周线(2026-08-17): vol_window 随 lookback_scale 缩放(20 日→4 周)。
        panel = ensure_base_panel(panel, vol_window=self._w(20))

        T, N, F = panel.values.shape
        device = panel.device
        mask_2d = self._panel_mask_2d(panel)  # (T, N)

        # Compute each feature group → (tensor, names)
        results = [
            self._price_dynamics(panel, mask_2d),        # (T, N, 45)
            self._volume_dynamics(panel, mask_2d),       # (T, N, 35)
            self._volatility_structure(panel, mask_2d),  # (T, N, 40)
            self._microstructure(panel, mask_2d),        # (T, N, 20)
            self._cross_sectional(panel, mask_2d),       # (T, N, 30)
            self._momentum_factor(panel, mask_2d),       # (T, N, 30)
        ]
        groups = [t for t, _ in results]
        all_names = [n for _, ns in results for n in ns]

        # Concatenate along feature dimension
        s_t = torch.cat(groups, dim=-1)  # (T, N, 200)

        # Safety: clip extreme values and replace NaN/Inf
        s_t = torch.nan_to_num(s_t, nan=0.0, posinf=10.0, neginf=-10.0)
        s_t = s_t.clamp(-50.0, 50.0)

        # Apply mask: zero-out non-tradable positions
        s_t[~mask_2d] = 0.0

        # ── A5: 更新特征注册表(配置决定, 无随机性; 每次 forward 幂等) ──
        self.feature_names = all_names
        self.real_feature_mask = torch.tensor(
            [("_pad_" not in n) for n in all_names], dtype=torch.bool)
        self.n_real_features = int(self.real_feature_mask.sum().item())
        self.n_padding = len(all_names) - self.n_real_features
        counts = {}
        for n in all_names:
            tag = n.split("_", 1)[0]
            if "_pad_" not in n:
                counts[tag] = counts.get(tag, 0) + 1
        self.group_real_counts = counts

        return s_t
