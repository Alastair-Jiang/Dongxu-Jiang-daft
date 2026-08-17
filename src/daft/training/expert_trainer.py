"""Stage 1: Independent strategy expert training.

Each of the 8 experts is trained on its regime-appropriate data subset using
its own specialized loss function. Experts are frozen after Stage 1.

Training data selection per expert type:
  - Trend (×2):      ADX > 25  (sustained directional movement)
  - Reversal (×2):   ADX < 20  (range-bound / oscillating)
  - Volatility (×2): vol > 80th percentile (high-vol regime)
  - Event (×2):      all data  (catch-all; real event filter needs calendar)

Loss functions:
  - TrendExpert:      direction-weighted MSE (11× penalty on sign errors)
  - ReversalExpert:   negative rank IC (maximise Spearman correlation)
  - VolatilityExpert: MSE + 0.01·Var(pred) regularisation
  - EventExpert:      binary cross-entropy on return direction
"""

from __future__ import annotations
import math
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import TensorDataset, DataLoader

from daft.data.panel import Panel
from daft.features.regime_features import RegimeFeatureExtractor


class Stage1ExpertTrainer:
    """Independent training of strategy experts on regime-specific data.

    Parameters
    ----------
    experts : nn.ModuleList
        List of 8 expert modules (2 per type, ordered: T,T,R,R,V,V,E,E).
    panel : Panel
        Raw OHLCV Panel — used to build s_t, targets, and regime masks.
    config : dict, optional
        Training hyperparameters. Keys:
        - lr (float, default 1e-3)
        - batch_size (int, default 2048)
        - epochs (int, default 50)
        - weight_decay (float, default 1e-5)
        - early_stop_patience (int, default 10)
        - grad_clip_norm (float, default 1.0)
        - val_frac (float, default 0.1)
    device : torch.device, optional
    """

    # Map expert index → regime group label
    EXPERT_REGIME_GROUP = {
        0: "trend",      1: "trend",
        2: "reversal",   3: "reversal",
        4: "volatility", 5: "volatility",
        6: "event",      7: "event",
        8: "event",      9: "event",           # MomentumExperts, train on all data like event
    }

    def __init__(
        self,
        experts: nn.ModuleList,
        panel: Panel,
        config: Optional[Dict] = None,
        device: Optional[torch.device] = None,
        lookback_scale: float = 1.0,
    ):
        self.experts = experts
        self.panel = panel
        self.config = config or {}
        self.device = device or torch.device("cpu")
        self.lookback_scale = lookback_scale

        # --- Build s_t: market state vectors via RegimeFeatureExtractor ---
        extractor = RegimeFeatureExtractor(
            n_base_factors=50, output_dim=200, lookback_scale=self.lookback_scale
        )
        with torch.no_grad():
            s_t_raw = extractor(panel)                     # (T, N, 200)

        # --- Z-score normalise s_t across (T, N) for stable training ---
        # RegimeFeatureExtractor can produce extreme values (rank explosions,
        # log-of-ratios, etc.). We clip extreme outliers and normalise per
        # feature dimension to μ=0, σ=1.
        s_t_raw = torch.nan_to_num(s_t_raw, nan=0.0, posinf=1e6, neginf=-1e6)
        s_t_raw = s_t_raw.clamp(-1e6, 1e6)
        # Per-feature statistics over (T, N) — avoid in-place to preserve autograd readiness
        s_flat = s_t_raw.reshape(-1, 200)                  # (T*N, 200)
        s_mean = s_flat.mean(dim=0, keepdim=True)           # (1, 200)
        s_std  = s_flat.std(dim=0, keepdim=True).clamp(min=1e-4)  # (1, 200)
        self.s_t = ((s_t_raw - s_mean) / s_std).clamp(-10.0, 10.0)  # (T, N, 200)

        # --- Build targets: forward 1-bar log-return ---
        close = panel.values[..., 3]                         # (T, N)
        log_c = torch.log(close.clamp(min=1e-8))
        self.targets = (log_c[1:] - log_c[:-1]).clamp(-0.5, 0.5)  # (T-1, N)

        # Align s_t[:-1] with targets
        self.s_t_aligned = self.s_t[:-1]                    # (T-1, N, 200)
        assert self.s_t_aligned.size(0) == self.targets.size(0), \
            f"Alignment error: s_t={self.s_t_aligned.size(0)}, target={self.targets.size(0)}"

        # --- Build regime masks from panel heuristics ---
        self.regime_masks = self._build_regime_masks()

    # ------------------------------------------------------------------
    # Regime mask construction
    # ------------------------------------------------------------------
    def _build_regime_masks(self) -> Dict[str, torch.Tensor]:
        """Build boolean regime masks of shape (T-1, N) using ADX/volatility.

        These heuristics match each expert's documented competence region
        and replace the currently-unimplemented per-expert _regime_filter().
        """
        T_m1 = self.targets.size(0)
        N = self.panel.N

        close = self.panel.values[:-1, :, 3]    # (T-1, N)
        high  = self.panel.values[:-1, :, 1]
        low   = self.panel.values[:-1, :, 2]
        mask  = self.panel.mask[:-1]             # (T-1, N), bool

        # ----- ADX proxy via directional movement (Wilder-style smoothed) -----
        up_move   = high[1:] - high[:-1]                           # (T-2, N)
        dn_move   = low[:-1] - low[1:]
        plus_dm   = up_move.clamp(min=0) * (up_move > dn_move).float()
        minus_dm  = dn_move.clamp(min=0) * (dn_move > up_move).float()
        tr = torch.maximum(
            high[1:] - low[1:],
            torch.maximum(
                (high[1:] - close[:-1]).abs(),
                (low[1:]  - close[:-1]).abs(),
            ),
        )

        span = 14
        alpha = 2.0 / (span + 1.0)
        atr_smooth = torch.zeros_like(tr)
        pdi_smooth = torch.zeros_like(tr)
        mdi_smooth = torch.zeros_like(tr)
        for t in range(tr.size(0)):
            if t == 0:
                atr_smooth[t] = tr[t]
                pdi_smooth[t] = plus_dm[t]
                mdi_smooth[t] = minus_dm[t]
            else:
                atr_smooth[t] = alpha * tr[t] + (1 - alpha) * atr_smooth[t - 1]
                pdi_smooth[t] = alpha * plus_dm[t] + (1 - alpha) * pdi_smooth[t - 1]
                mdi_smooth[t] = alpha * minus_dm[t] + (1 - alpha) * mdi_smooth[t - 1]

        dx = (pdi_smooth - mdi_smooth).abs() / (pdi_smooth + mdi_smooth).clamp(min=1e-8) * 100.0
        adx = torch.zeros_like(dx)
        for t in range(dx.size(0)):
            if t == 0:
                adx[t] = dx[t]
            else:
                adx[t] = alpha * dx[t] + (1 - alpha) * adx[t - 1]
        # Pad first row so shape is (T-1, N)
        adx = torch.cat([adx[:1], adx], dim=0)                     # (T-1, N)
        adx = adx * mask.float()

        # ----- Volatility proxy: rolling 20d std of returns -----
        # Use unbiased std only when n ≥ 2; for single-element windows use 0.
        # NaNs from single-element std would propagate through quantile and
        # zero out the volatility mask (NaN > x → False for all x).
        r_aligned = self.targets                                       # (T-1, N)
        vol20 = torch.zeros_like(r_aligned)
        for t in range(T_m1):
            lo = max(0, t - 19)
            n_obs = t - lo + 1
            if n_obs >= 2:
                vol20[t] = r_aligned[lo:t + 1].std(dim=0)             # unbiased
            else:
                vol20[t] = 0.0                                         # 1 obs → no variance
        # Compute threshold from valid (non-NaN) values only
        vol_valid = vol20[mask]
        vol_valid = vol_valid[vol_valid.isfinite()]
        vol_threshold = vol_valid.quantile(0.80) if vol_valid.numel() > 0 else 0.0

        # ----- Regime masks -----
        trend_mask      = (adx > 25.0) & mask
        reversal_mask   = (adx < 20.0) & mask
        volatility_mask = (vol20 > vol_threshold) & mask
        event_mask      = mask.clone()                                 # all data

        # Safety: if volatility mask is empty, fall back to top-20% by vol20
        if volatility_mask.sum() == 0:
            vol_sorted = vol_valid.sort().values
            if vol_sorted.numel() > 0:
                vol_threshold = vol_sorted[int(vol_sorted.numel() * 0.80)]
                volatility_mask = (vol20 > vol_threshold) & mask

        return {
            "trend":      trend_mask,
            "reversal":   reversal_mask,
            "volatility": volatility_mask,
            "event":      event_mask,
        }

    # ------------------------------------------------------------------
    # Main training entry point
    # ------------------------------------------------------------------
    def train_all(
        self,
        epochs: int = 50,
        batch_size: int = 2048,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        early_stop_patience: int = 10,
        verbose: bool = True,
        use_regime: bool = True,
    ) -> Dict[str, List[float]]:
        """Train all 8 experts independently.

        use_regime=False → 每个专家用全部数据训练(README regime 专业化对照,
        2026-08-17 研究实验)。
        """
        histories: Dict[str, List[float]] = {}

        for i, expert in enumerate(self.experts):
            if use_regime:
                regime_label = self.EXPERT_REGIME_GROUP[i]
                regime_mask = self.regime_masks[regime_label]
                tag = regime_label
            else:
                regime_mask = torch.ones_like(self.targets, dtype=torch.bool)
                tag = "all"
            n_samples = regime_mask.sum().item()

            if verbose:
                print(f"\n--- Expert [{i}] {expert.name} ({tag}) "
                      f"— {n_samples:,} samples ---")

            if n_samples < batch_size:
                if verbose:
                    print(f"  [SKIP] Only {n_samples} samples (< batch_size {batch_size})")
                continue

            history = self._train_one_expert(
                expert=expert,
                expert_idx=i,
                regime_mask=regime_mask,
                epochs=epochs,
                batch_size=batch_size,
                lr=lr,
                weight_decay=weight_decay,
                patience=early_stop_patience,
                verbose=verbose,
            )
            histories[f"expert_{i}_{expert.name}"] = history

        return histories

    # ------------------------------------------------------------------
    # Single-expert training loop
    # ------------------------------------------------------------------
    def _train_one_expert(
        self,
        expert: nn.Module,
        expert_idx: int,
        regime_mask: torch.Tensor,    # (T-1, N) bool
        epochs: int,
        batch_size: int,
        lr: float,
        weight_decay: float,
        patience: int,
        verbose: bool,
    ) -> List[float]:
        """Train a single expert on its regime-filtered data subset."""
        expert = expert.to(self.device)

        # --- Extract regime-specific samples ---
        s_flat = self.s_t_aligned[regime_mask]                 # (K, 200)
        t_flat = self.targets[regime_mask].unsqueeze(-1)       # (K, 1)
        loss_mask = torch.ones_like(t_flat)                     # all valid in regime

        # --- Train / validation split ---
        n_total = s_flat.size(0)
        n_train = int(n_total * (1.0 - self.config.get("val_frac", 0.1)))
        perm = torch.randperm(n_total)
        idx_train, idx_val = perm[:n_train], perm[n_train:]

        s_train, s_val = s_flat[idx_train], s_flat[idx_val]
        t_train, t_val = t_flat[idx_train], t_flat[idx_val]
        m_train, m_val = loss_mask[idx_train], loss_mask[idx_val]

        train_ds = TensorDataset(s_train, t_train, m_train)
        val_ds   = TensorDataset(s_val,   t_val,   m_val)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)

        # --- Optimiser ---
        optimizer = Adam(expert.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
        grad_clip = self.config.get("grad_clip_norm", 1.0)

        # --- Training loop ---
        val_history: List[float] = []
        best_val_loss = float("inf")
        best_state: Optional[Dict] = None
        stall = 0

        for epoch in range(epochs):
            # ---- Train ----
            expert.train()
            train_loss = 0.0
            n_batches = 0
            for s_b, t_b, m_b in train_loader:
                s_b = s_b.to(self.device)
                t_b = t_b.to(self.device)
                m_b = m_b.to(self.device)

                optimizer.zero_grad()
                pred = expert(s_b)                                    # (B, 1)
                loss = expert.compute_loss(pred, t_b, m_b)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(expert.parameters(), grad_clip)
                optimizer.step()

                train_loss += loss.item()
                n_batches += 1
            train_loss /= max(n_batches, 1)

            # ---- Validate ----
            expert.eval()
            val_loss = 0.0
            n_val = 0
            with torch.no_grad():
                for s_b, t_b, m_b in val_loader:
                    s_b = s_b.to(self.device)
                    t_b = t_b.to(self.device)
                    m_b = m_b.to(self.device)
                    pred = expert(s_b)
                    val_loss += expert.compute_loss(pred, t_b, m_b).item()
                    n_val += 1
            val_loss /= max(n_val, 1)
            val_history.append(val_loss)

            scheduler.step()

            # ---- Early stopping ----
            if val_loss < best_val_loss - 1e-6:
                best_val_loss = val_loss
                best_state = {k: v.clone() for k, v in expert.state_dict().items()}
                stall = 0
            else:
                stall += 1

            if verbose and (epoch % max(epochs // 10, 1) == 0 or epoch == epochs - 1):
                lr_now = scheduler.get_last_lr()[0]
                print(f"  epoch {epoch:3d}/{epochs}  "
                      f"train={train_loss:.6f}  val={val_loss:.6f}  "
                      f"lr={lr_now:.1e}  stall={stall}")

            if stall >= patience:
                if verbose:
                    print(f"  Early stop at epoch {epoch}  (best val={best_val_loss:.6f})")
                break

        # Restore best weights
        if best_state is not None:
            expert.load_state_dict(best_state)

        expert.to("cpu")
        return val_history

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save_checkpoints(self, out_dir: str) -> None:
        """Save all expert state dicts to disk."""
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        for i, expert in enumerate(self.experts):
            torch.save(expert.state_dict(), out / f"expert_{i}_{expert.name}.pt")

    @staticmethod
    def load_checkpoints(
        experts: nn.ModuleList, ckpt_dir: str
    ) -> nn.ModuleList:
        """Restore expert weights from disk."""
        ckpt = Path(ckpt_dir)
        for i, expert in enumerate(experts):
            path = ckpt / f"expert_{i}_{expert.name}.pt"
            if path.exists():
                expert.load_state_dict(torch.load(path, map_location="cpu"))
        return experts
