"""Vectorized backtesting engine.

Computes strategy returns from model signals with configurable:
- Rebalance frequency
- Signal → position conversion (quantile-based)
- Transaction cost model (fixed + proportional + slippage)
- Performance metrics (Sharpe, MaxDD, Calmar, IC, ICIR, hit rate)
"""

from __future__ import annotations
import math
from typing import Dict, List, Optional

import torch

from daft.utils.metrics import rank_info_coefficient, ic_summary, hit_rate


class BacktestEngine:
    """Vectorized backtesting with performance metrics.

    Parameters
    ----------
    config : dict
        Evaluation section of YAML config. Keys:
        - transaction_cost_bps : float  (default 2.0)
        - slippage_bps : float          (default 1.0)
        - top_quantile : float          (default 0.2, for signal→position)
        - long_only : bool              (default True)
        - annualization : int           (default 252)
        - rebalance_freq : int          (default 1, rebalance every N bars)
        - signal_smoothing : float      (default 0.0, EMA decay in [0,1);
                                        0 = no smoothing, 0.5 = strong)
    """

    def __init__(self, config: Optional[dict] = None):
        self.config = config or {}

        # Cost params
        self.tc_bps = self.config.get("transaction_cost_bps", 2.0)
        self.slippage_bps = self.config.get("slippage_bps", 1.0)

        # Position params
        self.top_quantile = self.config.get("top_quantile", 0.2)
        self.long_only = self.config.get("long_only", True)
        # 仓位模式(2026-08-16 新增): "equal" = top-k 等权(原行为);
        # "signal_zscore" = 按信号 z 分数加权(候选内)
        self.weight_mode = self.config.get("weight_mode", "equal")

        # 成交约束(2026-08-16 新增): True → 停牌/涨跌停日(mask=False)
        # 禁止开平仓(已持仓维持昨收、不计换手成本)。
        self.respect_mask_no_trade = self.config.get("respect_mask_no_trade", True)

        # Other
        self.annualization = self.config.get("annualization", 252)
        self.rebalance_freq = self.config.get("rebalance_freq", 1)
        # 信号平滑 (Kimi K3 评审 2026-08-09): DAFT 样本外输给 Ridge 的
        # 直接原因是换手 1.7%/bar vs 0.03% —— 信号抖动在稳定交手续费。
        # EMA 平滑信号可降换手; 0 = 不启用 (保持原行为)。
        self.signal_smoothing = self.config.get("signal_smoothing", 0.0)

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------
    def run(
        self,
        signals: torch.Tensor,
        prices: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        metrics_list: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """Execute walk-forward backtest on model signals.

        Parameters
        ----------
        signals : (T, N)
            Model-predicted signals (expected returns or position scores).
        prices : (T, N)
            Close prices for return computation.
        mask : (T, N), bool, optional
            Tradability mask. Defaults to all-true.
        metrics_list : list of str, optional
            Subset of metrics to compute. None → compute all.

        Returns
        -------
        metrics : dict
            sharpe_ratio, max_drawdown, calmar_ratio, annual_return,
            annual_volatility, ic_rank, icir, hit_rate, rmse, turnover.

        Notes
        -----
        Works on CPU or GPU. All computations are vectorized over assets
        but sequential over time (due to transaction costs).

        交易制度约束(2026-08-16):
        - T+1: 日线收盘-收盘持仓结构下, t 日收盘建立的头寸最早于 t+1 日
          收盘平仓(持有满 1 日), A 股 T+1 约束结构性满足。
        - 停牌/涨跌停: mask=False 的交易日既不能开仓也不能平仓 —
          respect_mask_no_trade=True(默认)时该资产持仓维持昨收、不计换手;
          信号选择阶段亦排除 masked 资产建仓。
        """
        T, N = signals.shape
        if mask is None:
            mask = torch.ones(T, N, dtype=torch.bool, device=signals.device)
        device = signals.device

        # --- Optional EMA signal smoothing (Kimi K3 评审 2026-08-09) ---
        # 降换手: 平滑后的信号 s'_t = (1-λ)·s_t + λ·s'_{t-1}, λ=signal_smoothing
        # 因果(只用过去), 不引入 look-ahead。λ=0 时完全等价于原信号。
        if self.signal_smoothing > 0:
            lam = min(max(self.signal_smoothing, 0.0), 0.99)
            smoothed = torch.zeros_like(signals)
            smoothed[0] = signals[0]
            for t in range(1, T):
                smoothed[t] = (1 - lam) * signals[t] + lam * smoothed[t - 1]
            signals = smoothed

        # --- Daily log-returns ---
        log_p = torch.log(prices.clamp(min=1e-8))
        returns = log_p[1:] - log_p[:-1]         # (T-1, N)
        ret_mask = mask[:-1] & mask[1:]            # both t and t-1 valid
        T_ret = returns.size(0)

        # --- Signal → position weights ---
        positions = self._signals_to_positions(
            signals[:-1], ret_mask, device  # use t's signal for t+1's return
        )  # (T-1, N)

        # --- Walk-forward ---
        daily_returns = torch.zeros(T_ret, device=device)
        prev_positions = torch.zeros(N, device=device)
        turnovers = torch.zeros(T_ret, device=device)

        for t in range(T_ret):
            # 调仓频率(2026-08-16 实现): 仅每 rebalance_freq 根 bar 换仓,
            # 其间持仓不变; 成本只在换仓日计提。
            if t % self.rebalance_freq == 0:
                w_t = positions[t].clone()                     # (N,)
                # 成交约束(2026-08-16 新增): 停牌/涨跌停日(mask=False)无法
                # 开仓也无法平仓 — 该资产的持仓维持昨收, 不产生换手与成本。
                # 注: 信号选择阶段已把 masked 资产排除在新建仓之外; 这里
                # 补的是"已持仓资产在跌停日卖不掉"这一半。
                if self.respect_mask_no_trade:
                    blocked = ~ret_mask[t]
                    if blocked.any():
                        w_t[blocked] = prev_positions[blocked]
                # Turnover (L1 distance from previous weights) — 真实仓位换手
                turnover = (w_t - prev_positions).abs().sum()
                turnovers[t] = turnover
                # Transaction cost
                tc = (self.tc_bps / 10000.0) * turnover
                sl = (self.slippage_bps / 10000.0) * turnover
                prev_positions = w_t
            else:
                w_t = prev_positions                            # 持仓不变
                turnover = 0.0
                tc, sl = 0.0, 0.0

            r_t = returns[t]                                    # (N,)
            m_t = ret_mask[t]                                   # (N,)

            # Portfolio return
            port_r = (w_t * r_t * m_t.float()).sum()
            daily_returns[t] = port_r - tc - sl

        # --- Metrics ---
        metrics = self._compute_metrics(
            daily_returns, signals[:-1], returns, ret_mask, turnovers,
            metrics_list,
        )

        # Add IC-specific summary
        ic_series = rank_info_coefficient(
            signals[:-1], returns, ret_mask, per_timestep=True,
        )
        ic_stats = ic_summary(ic_series)
        metrics.update({
            "ic_rank": ic_stats["ic_mean"],
            "icir": ic_stats["icir"],
        })

        return metrics

    # ------------------------------------------------------------------
    # Signal → position conversion
    # ------------------------------------------------------------------
    def _signals_to_positions(
        self,
        signals: torch.Tensor,   # (T, N)
        mask: torch.Tensor,       # (T, N) bool
        device: torch.device,
    ) -> torch.Tensor:
        """Convert raw signals to portfolio weights using quantile selection.

        Long-only (default): top ``top_quantile`` fraction receives weights
        proportional to signal strength, normalized to sum to 1.

        Long-short: top quantile → long, bottom quantile → short,
        weights proportional to signal z-score, normalized to sum to 0
        (i.e. longs sum to +1, shorts sum to -1).
        """
        T, N = signals.shape
        positions = torch.zeros(T, N, device=device)

        for t in range(T):
            s_t = signals[t].clone()
            m_t = mask[t]
            s_t[~m_t] = float("-inf")

            n_valid = m_t.sum().item()
            if n_valid == 0:
                continue

            k = max(1, int(n_valid * self.top_quantile))

            if self.long_only:
                _, top_idx = s_t.topk(k)
                w = torch.zeros(N, device=device)
                if self.weight_mode == "signal_zscore":
                    z = _zscore_pos(s_t[top_idx])
                    w[top_idx] = z
                else:
                    w[top_idx] = 1.0 / k
            else:
                # Long top-k, short bottom-k
                # 修复(2026-08-16): 空头候选必须排除 masked 资产。
                # 旧实现用 (-s_t).topk(k), 而 s_t[masked]=-inf 取反后变 +inf,
                # 停牌股反而优先被做空(已复现)。
                _, top_idx = s_t.topk(k)
                s_short = s_t.clone()
                s_short[~m_t] = float("inf")     # masked 资产永不做空
                _, bot_idx = s_short.topk(k, largest=False)  # 最差 k 个有效信号
                w = torch.zeros(N, device=device)
                if self.weight_mode == "signal_zscore":
                    z_long = _zscore_pos(s_t[top_idx])
                    z_short = _zscore_pos(-s_t[bot_idx])
                    w[top_idx] = z_long
                    w[bot_idx] = -z_short
                else:
                    w[top_idx] = 1.0 / k
                    w[bot_idx] = -1.0 / k

            positions[t] = w

        return positions

    # ------------------------------------------------------------------
    # Metric computation
    # ------------------------------------------------------------------
    def _compute_metrics(
        self,
        daily_returns: torch.Tensor,    # (T,)
        signals: torch.Tensor,          # (T, N)
        returns: torch.Tensor,          # (T, N)
        mask: torch.Tensor,             # (T, N) bool
        turnovers: Optional[torch.Tensor] = None,  # (T,) 真实仓位换手
        metrics_list: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """Compute all standard backtest metrics."""
        want = set(metrics_list) if metrics_list else None

        def _want(name: str) -> bool:
            return want is None or name in want

        out: Dict[str, float] = {}

        # Cumulative returns (log space)
        cum_ret = torch.cumsum(daily_returns, dim=0)  # log-cumulative

        # Annualized return
        ann_ret = daily_returns.mean().item() * self.annualization
        out["annual_return"] = ann_ret

        # Annualized volatility
        ann_vol = daily_returns.std().item() * math.sqrt(self.annualization)
        out["annual_volatility"] = ann_vol

        # Sharpe
        out["sharpe_ratio"] = self.sharpe_ratio(daily_returns, self.annualization)

        # Max drawdown — 百分比口径(2026-08-16):
        # equity = exp(对数累计收益), drawdown = equity/running_max - 1
        out["max_drawdown"] = self.max_drawdown(cum_ret)

        # Calmar = annual return / |max drawdown|
        mdd = out["max_drawdown"]
        out["calmar_ratio"] = ann_ret / abs(mdd) if abs(mdd) > 1e-8 else 0.0

        # Hit rate
        out["hit_rate"] = hit_rate(signals, returns, mask)

        # Turnover — 真实仓位换手 (2026-08-16 修复口径):
        # 之前上报的是原始信号 L1 代理, 与成本驱动因子(分位数仓位 Σ|Δw|)
        # 不同刻度, 导致"换手降 7 倍"等归因失真。现在直接上报实际仓位换手。
        if turnovers is not None and turnovers.numel() > 0:
            out["turnover"] = turnovers.mean().item()
        else:
            out["turnover"] = self._compute_turnover(signals, mask, daily_returns.size(0))

        # RMSE of normalized predictions vs targets
        out["rmse"] = _compute_rmse(signals, returns, mask)

        # Max drawdown duration
        out["max_drawdown_duration"] = _max_dd_duration(daily_returns)

        return {k: round(v, 6) if isinstance(v, float) else v
                for k, v in out.items()}

    def _compute_turnover(
        self, signals: torch.Tensor, mask: torch.Tensor, n_ret: int,
    ) -> float:
        """Mean per-step turnover from signals. ⚠️ 已弃用(2026-08-16)。

        这是原始信号的 L1 代理, 不是实际仓位换手 — 仅当 run() 未提供
        真实仓位换手时回退使用。新代码请用 run() 上报的 turnover。
        """
        T = min(signals.size(0) - 1, n_ret)
        if T < 2:
            return 0.0
        # Use signals as a proxy for positions (actual positions depend on quantile)
        pos_proxy = signals[:T] * mask[:T].float()
        turnover = (pos_proxy[1:] - pos_proxy[:-1]).abs().sum(dim=-1)  # (T-1,)
        return (turnover / mask[1:T].sum(dim=-1).clamp(min=1)).mean().item()

    # ------------------------------------------------------------------
    # Static metric methods
    # ------------------------------------------------------------------
    @staticmethod
    def sharpe_ratio(returns: torch.Tensor, annualization: float = 252.0) -> float:
        """Annualized Sharpe ratio."""
        if returns.numel() < 2:
            return 0.0
        std = returns.std()
        if std == 0:
            return 0.0
        return (returns.mean() / std).item() * (annualization ** 0.5)

    @staticmethod
    def max_drawdown(cumulative_returns: torch.Tensor) -> float:
        """Maximum drawdown in percentage terms (2026-08-16 改口径)。

        ``cumulative_returns`` 为对数累计收益; 转成净值曲线后计算:
            equity = exp(cumsum(r)), dd = equity / running_max - 1
        返回最深的负值(如 -0.25 = 回撤 25%), 可直接与年化收益对比。

        Parameters
        ----------
        cumulative_returns : (T,) 对数累计收益曲线。
        """
        if cumulative_returns.numel() < 2:
            return 0.0
        equity = torch.exp(cumulative_returns)
        running_max = equity.cummax(dim=0).values
        drawdowns = equity / running_max.clamp(min=1e-12) - 1.0   # ≤ 0
        return drawdowns.min().item()

    @staticmethod
    def info_coefficient(
        predictions: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor,
    ) -> float:
        """Rank IC = cross-sectional Spearman correlation, averaged over time."""
        if mask is None:
            mask = torch.ones_like(predictions, dtype=torch.bool)
        return rank_info_coefficient(predictions, targets, mask, per_timestep=False)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _zscore_pos(x: torch.Tensor) -> torch.Tensor:
    """信号 → 正仓位权重: softplus(z-score) 归一化。

    候选内权重单调于信号且恒为正(softplus 无零点), 用于
    weight_mode="signal_zscore" 的分数仓位。
    """
    mu = x.mean()
    sd = x.std().clamp(min=1e-8)
    z = (x - mu) / sd
    pos = torch.nn.functional.softplus(z)
    return pos / pos.sum().clamp(min=1e-12)


def _compute_rmse(
    predictions: torch.Tensor,  # (T, N)
    targets: torch.Tensor,      # (T, N)
    mask: torch.Tensor,         # (T, N) bool
) -> float:
    """Normalized RMSE of predictions vs targets."""
    p = predictions[mask]
    t = targets[mask]
    if p.numel() < 2:
        return 0.0

    # Normalize to unit variance per time step
    p_z = (p - p.mean()) / p.std().clamp(min=1e-8)
    t_z = (t - t.mean()) / t.std().clamp(min=1e-8)
    return ((p_z - t_z) ** 2).mean().sqrt().item()


def _max_dd_duration(daily_returns: torch.Tensor) -> int:
    """Longest consecutive number of bars spent in drawdown (below peak)."""
    if daily_returns.numel() < 2:
        return 0

    cum = torch.cumsum(daily_returns, dim=0)
    running_max = cum.cummax(dim=0).values
    in_dd = (cum < running_max).int()

    max_dur = 0
    cur_dur = 0
    for v in in_dd.tolist():
        if v:
            cur_dur += 1
            max_dur = max(max_dur, cur_dur)
        else:
            cur_dur = 0
    return max_dur
