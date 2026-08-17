"""Stage 3: Joint fine-tuning.

All parameters unfrozen. Full CDAP modulation (δ = 1.0).
Low learning rate (η = 1e-5) to prevent catastrophic forgetting.
Early stopping on validation IC degradation.

After Stage 3, run HardeningEngine statistics collection (Stage 4).
"""

from __future__ import annotations
import copy
import math
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import TensorDataset, DataLoader

from daft.data.panel import Panel
from daft.features.regime_features import RegimeFeatureExtractor
from daft.utils.metrics import rank_info_coefficient, ic_summary, rank_ic_by_timestep


class JointTrainer:
    """Joint fine-tuning with full CDAP modulation.

    Parameters
    ----------
    model : ExpertEnsemble
        Full DAFT model (all parameters trainable).
    layer_proj : nn.ModuleDict
        Layer hierarchy projectors from Stage 2.
    config : dict
        Stage 3 config.
    device : torch.device
    """

    def __init__(
        self,
        model,
        layer_proj: nn.ModuleDict,
        config: dict,
        device: torch.device,
        lookback_scale: float = 1.0,
    ):
        self.model = model
        self.layer_proj = layer_proj
        self.config = config
        self.device = device
        self.lookback_scale = lookback_scale
        # A2 修复 (2026-08-18): 标准化统计量 (mean, std) 只在训练段拟合一次,
        # val/推理共用同一份 —— 修复 val 段自算统计量与 OOS 推理(train-only)
        # 分布不一致的问题(早停/选型依据失真)。
        self.norm_stats = None

    # ------------------------------------------------------------------
    def train(self, train_panel: Panel, val_panel: Panel) -> dict:
        """Stage 3: Joint fine-tuning with full CDAP.

        1. Unfreeze all parameters
        2. Full CDAP modulation_strength = 1.0
        3. Very low LR = 1e-5 (prevent catastrophic forgetting)
        4. Gradient clipping norm = 0.5
        5. Early stopping on validation IC degradation
        """
        cfg = self.config
        epochs = cfg.get("epochs", 20)
        batch_size = cfg.get("batch_size", 1024)
        lr = cfg.get("lr", 1e-5)
        weight_decay = cfg.get("weight_decay", 1e-6)
        patience = cfg.get("early_stop_patience", 5)
        grad_clip_norm = cfg.get("grad_clip_norm", 0.5)
        expert_lr_ratio = cfg.get("expert_lr_ratio", 0.1)  # experts get even lower LR

        # --- Unfreeze all parameters ---
        for expert in self.model.experts:
            for p in expert.parameters():
                p.requires_grad = True

        # --- Full CDAP modulation ---
        self.model.cross_dim_attn.modulation_strength = 1.0
        self.model.router.temperature = 0.1   # near-discrete routing at inference temp

        # --- Build s_t and targets ---
        # A2 (2026-08-18): 训练段拟合统计量并记录, val 段强制复用
        # (与 OOS 推理的 train-only 标准化同口径)
        train_s, train_t, train_m, train_tidx = self._build_dataset(train_panel)
        val_s, val_t, val_m, val_tidx = self._build_dataset(
            val_panel, norm_stats=self.norm_stats
        )

        # --- Move to device ---
        self.layer_proj.to(self.device)
        self.model.to(self.device)

        # --- Data loaders (按日对齐批次, 2026-08-16) ---
        N_stocks = train_panel.N
        eff_batch = max(N_stocks, (batch_size // N_stocks) * N_stocks)
        train_ds = TensorDataset(train_s, train_t, train_m, train_tidx)
        val_ds = TensorDataset(val_s, val_t, val_m, val_tidx)
        train_loader = DataLoader(train_ds, batch_size=eff_batch, shuffle=False)
        val_loader = DataLoader(val_ds, batch_size=eff_batch, shuffle=False)

        # --- Parameter groups: experts get lower LR ---
        expert_params = []
        other_params = []
        for expert in self.model.experts:
            expert_params.extend(list(expert.parameters()))
        other_params.extend(list(self.model.router.parameters()))
        other_params.extend(list(self.model.memory.parameters()))
        other_params.extend(list(self.model.cross_dim_attn.parameters()))
        other_params.extend(list(self.layer_proj.parameters()))

        param_groups = [
            {"params": other_params, "lr": lr},
            {"params": expert_params, "lr": lr * expert_lr_ratio},
        ]
        optimizer = Adam(param_groups, weight_decay=weight_decay)
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

        # --- Training loop ---
        history: Dict[str, List[float]] = {
            "train_loss": [], "val_loss": [],
            "val_ic_mean": [], "val_icir": [],
            "routing_entropy": [],
        }
        best_val_ic = -float("inf")
        best_state: Optional[Dict] = None
        stall = 0

        for epoch in range(epochs):
            # ---- Train ----
            self.model.train()
            self.layer_proj.train()
            train_loss, train_entropy = self._run_epoch(
                train_loader, optimizer, True, grad_clip_norm=grad_clip_norm,
            )

            # ---- Validate ----
            self.model.eval()
            self.layer_proj.eval()
            val_loss, val_signals, val_targets, val_tidx_c = self._run_epoch(
                val_loader, None, False, return_predictions=True,
            )

            # 逐时步截面 rank IC (2026-08-16 修复: 旧实现为退化指标)
            val_ic = rank_ic_by_timestep(
                val_signals.squeeze(-1), val_targets.squeeze(-1), val_tidx_c,
            )
            val_ic_summary = ic_summary(val_ic) if val_ic.numel() > 0 else {
                "ic_mean": 0.0, "ic_std": 0.0, "icir": 0.0,
                "ic_positive_ratio": 0.0, "ic_t_stat": 0.0,
            }

            scheduler.step()

            # ---- Logging ----
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["val_ic_mean"].append(val_ic_summary["ic_mean"])
            history["val_icir"].append(val_ic_summary["icir"])
            history["routing_entropy"].append(train_entropy)

            print(
                f"  epoch {epoch:3d}/{epochs}  "
                f"train_loss={train_loss:.6f}  val_loss={val_loss:.6f}  "
                f"val_ic={val_ic_summary['ic_mean']:+.4f}  "
                f"ICIR={val_ic_summary['icir']:+.3f}  "
                f"entropy={train_entropy:.3f}"
            )

            # ---- Early stopping on validation IC ----
            val_ic_mean = val_ic_summary["ic_mean"]
            if val_ic_mean > best_val_ic + 1e-5:
                best_val_ic = val_ic_mean
                best_state = {
                    "experts": [
                        copy.deepcopy(e.state_dict()) for e in self.model.experts
                    ],
                    "router": copy.deepcopy(self.model.router.state_dict()),
                    "memory": copy.deepcopy(self.model.memory.state_dict()),
                    "cdap": copy.deepcopy(self.model.cross_dim_attn.state_dict()),
                    "layer_proj": copy.deepcopy(self.layer_proj.state_dict()),
                }
                stall = 0
            else:
                stall += 1

            if stall >= patience:
                print(f"  Early stop at epoch {epoch}  (best val_ic={best_val_ic:+.4f})")
                break

        # --- Restore best ---
        if best_state is not None:
            for i, expert in enumerate(self.model.experts):
                expert.load_state_dict(best_state["experts"][i])
            self.model.router.load_state_dict(best_state["router"])
            self.model.memory.load_state_dict(best_state["memory"])
            self.model.cross_dim_attn.load_state_dict(best_state["cdap"])
            self.layer_proj.load_state_dict(best_state["layer_proj"])

        return history

    # ------------------------------------------------------------------
    def _build_dataset(self, panel: Panel, norm_stats=None):
        """Build s_t, targets, mask from panel.

        norm_stats : (mean, std) 或 None (A2 修复, 2026-08-18)
            None → 从当前段拟合并记录到 self.norm_stats(训练段用法);
            给定 → 复用该统计量且**不**覆盖 self.norm_stats(val 段用法,
            与 OOS 推理的 train-only 标准化同口径)。
        """
        extractor = RegimeFeatureExtractor(
            n_base_factors=50, output_dim=200, lookback_scale=self.lookback_scale
        )
        with torch.no_grad():
            s_t_raw = extractor(panel)

        s_t_raw = torch.nan_to_num(s_t_raw, nan=0.0, posinf=1e6, neginf=-1e6)
        s_t_raw = s_t_raw.clamp(-1e6, 1e6)
        # A2: 统计量来源可注入; 默认本段拟合
        if norm_stats is None:
            s_flat = s_t_raw.reshape(-1, 200)
            s_mean = s_flat.mean(dim=0, keepdim=True)
            s_std = s_flat.std(dim=0, keepdim=True).clamp(min=1e-4)
            self.norm_stats = (s_mean, s_std)
        else:
            s_mean, s_std = norm_stats
        s_t = ((s_t_raw - s_mean) / s_std).clamp(-10.0, 10.0)

        close = panel.values[..., 3]
        log_c = torch.log(close.clamp(min=1e-8))
        targets = (log_c[1:] - log_c[:-1]).clamp(-0.5, 0.5)
        s_aligned = s_t[:-1]

        T, N = targets.shape
        s_2d = s_aligned.reshape(T * N, 200)
        t_1d = targets.reshape(T * N, 1)
        m_1d = panel.mask[:-1].reshape(T * N, 1)
        t_idx = torch.arange(T, device=s_2d.device).repeat_interleave(N)

        valid = m_1d.squeeze(-1)
        if valid.any():
            s_2d = s_2d[valid]
            t_1d = t_1d[valid]
            m_1d = m_1d[valid]
            t_idx = t_idx[valid]

        return s_2d, t_1d, m_1d, t_idx

    # ------------------------------------------------------------------
    def _run_epoch(
        self,
        loader: DataLoader,
        optimizer: Optional[torch.optim.Optimizer],
        training: bool,
        grad_clip_norm: float = 0.5,
        return_predictions: bool = False,
    ):
        """Run one epoch of joint training."""
        total_loss = 0.0
        total_entropy = 0.0
        n_batches = 0
        all_signals = []
        all_targets = []
        all_tidx = []

        self.model.memory.reset_state(1, self.device)

        for s_b, t_b, m_b, ti_b in loader:
            s_b = s_b.to(self.device)
            t_b = t_b.to(self.device)
            m_b = m_b.to(self.device)
            B = s_b.size(0)

            if self.model.memory.M is None or self.model.memory.M.size(0) != B:
                self.model.memory.reset_state(B, self.device)

            l0 = self.layer_proj["l0"](s_b)
            l1 = self.layer_proj["l1"](s_b)
            l2 = self.layer_proj["l2"](s_b)
            layer_outputs = [l0, l1, l2]

            if training:
                outputs = self.model(s_b, layer_outputs, mode="train")
                signal = outputs["signal"]          # (B, 1)
                routing_probs = outputs["routing_probs"]

                # --- Joint loss: MSE on final signal + expert consistency ---
                mse = ((signal - t_b) ** 2 * m_b.float()).sum() / m_b.float().sum().clamp(min=1)

                # Expert consistency: lightly regularize each expert
                expert_reg = 0.0
                for i, expert in enumerate(self.model.experts):
                    pred_i = expert(s_b)
                    loss_i = expert.compute_loss(pred_i, t_b, m_b)
                    # Weight by routing probability to preserve specialization
                    w_i = routing_probs[:, i].mean()
                    expert_reg = expert_reg + w_i * loss_i

                loss = mse + 0.1 * expert_reg

                routing_entropy = -(routing_probs * (routing_probs + 1e-8).log()
                                    ).sum(dim=-1).mean()

                optimizer.zero_grad()
                loss.backward()
                # Clip ALL parameters
                all_params = []
                for expert in self.model.experts:
                    all_params.extend(list(expert.parameters()))
                all_params.extend(list(self.model.router.parameters()))
                all_params.extend(list(self.model.memory.parameters()))
                all_params.extend(list(self.model.cross_dim_attn.parameters()))
                all_params.extend(list(self.layer_proj.parameters()))
                torch.nn.utils.clip_grad_norm_(all_params, grad_clip_norm)
                optimizer.step()

                total_loss += loss.item()
                total_entropy += routing_entropy.item()

            else:
                with torch.no_grad():
                    outputs = self.model(s_b, layer_outputs, mode="val")
                    signal = outputs["signal"]
                    mse = ((signal - t_b) ** 2 * m_b.float()).sum() / m_b.float().sum().clamp(min=1)
                    total_loss += mse.item()

                if return_predictions:
                    all_signals.append(signal.cpu())
                    all_targets.append(t_b.cpu())
                    all_tidx.append(ti_b.cpu())

            self.model.memory.detach_state()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        avg_entropy = total_entropy / max(n_batches, 1) if training else 0.0

        if return_predictions:
            signals = torch.cat(all_signals, dim=0) if all_signals else torch.zeros(0, 1)
            targets = torch.cat(all_targets, dim=0) if all_targets else torch.zeros(0, 1)
            t_idx = torch.cat(all_tidx, dim=0) if all_tidx else torch.zeros(0, dtype=torch.long)
            return avg_loss, signals, targets, t_idx

        return avg_loss, avg_entropy

    # ------------------------------------------------------------------
    def save_checkpoints(self, out_dir: str) -> None:
        """Save all model components (final trained weights)."""
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        for i, expert in enumerate(self.model.experts):
            torch.save(expert.state_dict(), out / f"expert_{i}_{expert.name}.pt")
        torch.save(self.model.router.state_dict(), out / "router.pt")
        torch.save(self.model.memory.state_dict(), out / "memory.pt")
        torch.save(self.model.cross_dim_attn.state_dict(), out / "cdap.pt")
        torch.save(self.layer_proj.state_dict(), out / "layer_proj.pt")

    @staticmethod
    def load_checkpoints(
        model, layer_proj: nn.ModuleDict, ckpt_dir: str
    ) -> None:
        """Restore Stage 3 weights from disk."""
        ckpt = Path(ckpt_dir)
        for i, expert in enumerate(model.experts):
            path = ckpt / f"expert_{i}_{expert.name}.pt"
            if path.exists():
                expert.load_state_dict(torch.load(path, map_location="cpu"))
        for name, component in [
            ("router.pt", model.router),
            ("memory.pt", model.memory),
            ("cdap.pt", model.cross_dim_attn),
        ]:
            path = ckpt / name
            if path.exists():
                component.load_state_dict(torch.load(path, map_location="cpu"))
        path = ckpt / "layer_proj.pt"
        if path.exists():
            layer_proj.load_state_dict(torch.load(path, map_location="cpu"))
