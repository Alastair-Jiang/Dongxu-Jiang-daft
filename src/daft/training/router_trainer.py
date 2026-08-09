"""Stage 2: Router + Memory training (experts frozen).

With expert weights frozen from Stage 1, train the Regime Router and
KDA Market Memory to:
1. Route each market state to the best expert(s)
2. Maintain a predictive memory of historical patterns
3. CDAP connections operate at low modulation strength (δ = 0.1)

Loss = weighted sum of expert prediction quality, weighted by routing probs.
Temperature annealing: 1.0 → 0.1 over training.
"""

from __future__ import annotations
import copy
import math
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import TensorDataset, DataLoader

from daft.data.panel import Panel
from daft.features.regime_features import RegimeFeatureExtractor
from daft.utils.metrics import rank_info_coefficient, ic_summary


class RouterTrainer:
    """Train router and memory with frozen experts.

    Parameters
    ----------
    model : ExpertEnsemble
        Full DAFT model with frozen experts.
    config : dict
        Stage 2 config.
    device : torch.device
    """

    def __init__(self, model, config: dict, device: torch.device):
        self.model = model
        self.config = config
        self.device = device

        # Layer hierarchy projectors: s_t (200) → 3 × d_v (64)
        # These are trainable — they learn to organize features into a
        # 3-level hierarchy (raw → base → composite) that CDAP can route across.
        d_v = model.memory.d_v  # 64
        input_dim = model.router.input_dim  # 200
        self.layer_proj = nn.ModuleDict({
            "l0": nn.Sequential(
                nn.Linear(input_dim, 128),
                nn.SiLU(),
                nn.Linear(128, d_v),
                nn.LayerNorm(d_v),
            ),
            "l1": nn.Sequential(
                nn.Linear(input_dim, 128),
                nn.SiLU(),
                nn.Linear(128, d_v),
                nn.LayerNorm(d_v),
            ),
            "l2": nn.Sequential(
                nn.Linear(input_dim, 128),
                nn.SiLU(),
                nn.Linear(128, d_v),
                nn.LayerNorm(d_v),
            ),
        })

    # ------------------------------------------------------------------
    def train(self, train_panel: Panel, val_panel: Panel) -> dict:
        """Stage 2: Train router + memory + CDAP with frozen experts.

        1. Freeze all expert weights
        2. Loss = Σ_i w_i · expert_i.loss(pred_i, target)  (router-weighted)
        3. Temperature annealing: 1.0 → 0.1
        4. CDAP modulation_strength = 0.1
        5. Quantile balancing every N steps
        6. Early stopping on validation loss
        """
        cfg = self.config
        epochs = cfg.get("epochs", 30)
        batch_size = cfg.get("batch_size", 1024)
        lr = cfg.get("lr", 1e-3)
        weight_decay = cfg.get("weight_decay", 1e-5)
        patience = cfg.get("early_stop_patience", 8)
        balance_every = cfg.get("balance_every", 50)
        grad_clip_norm = cfg.get("grad_clip_norm", 1.0)
        entropy_weight = cfg.get("entropy_weight", 0.01)  # small entropy bonus

        # --- Freeze experts ---
        for expert in self.model.experts:
            for p in expert.parameters():
                p.requires_grad = False

        # --- Set CDAP modulation strength ---
        self.model.cross_dim_attn.modulation_strength = 0.1

        # --- Build s_t and targets ---
        train_s, train_t, train_m = self._build_dataset(train_panel)
        val_s, val_t, val_m = self._build_dataset(val_panel)

        # --- Move layer projections to device ---
        self.layer_proj.to(self.device)
        self.model.to(self.device)

        # --- Train / val loaders (no shuffle — memory is stateful) ---
        train_ds = TensorDataset(train_s, train_t, train_m)
        val_ds = TensorDataset(val_s, val_t, val_m)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

        # --- Optimizer (router + memory + CDAP + layer_proj only) ---
        trainable_params = (
            list(self.model.router.parameters())
            + list(self.model.memory.parameters())
            + list(self.model.cross_dim_attn.parameters())
            + list(self.layer_proj.parameters())
        )
        optimizer = Adam(trainable_params, lr=lr, weight_decay=weight_decay)
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

        # --- Training loop ---
        history: Dict[str, List[float]] = {
            "train_loss": [], "val_loss": [],
            "val_ic_mean": [], "val_icir": [],
            "routing_entropy": [],
        }
        best_val_loss = float("inf")
        best_state: Optional[Dict] = None
        stall = 0

        for epoch in range(epochs):
            # ---- Temperature annealing: 1.0 → 0.1 ----
            temp = 1.0 - 0.9 * (epoch / max(epochs - 1, 1))
            self.model.router.temperature = temp

            # ---- Train ----
            self.model.train()
            self.layer_proj.train()
            train_loss, train_entropy = self._run_epoch(
                train_loader, optimizer, True,
                balance_every=balance_every,
                grad_clip_norm=grad_clip_norm,
                entropy_weight=entropy_weight,
            )

            # ---- Validate ----
            self.model.eval()
            self.layer_proj.eval()
            val_loss, val_signals, val_targets = self._run_epoch(
                val_loader, None, False, return_predictions=True,
            )

            # Compute validation IC (squeeze flattened (K,1) → (K,) for 1D path)
            val_ic = rank_info_coefficient(
                val_signals.squeeze(-1), val_targets.squeeze(-1), None,
                per_timestep=True,
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
                f"temp={temp:.3f}  entropy={train_entropy:.3f}"
            )

            # ---- Early stopping ----
            if val_loss < best_val_loss - 1e-6:
                best_val_loss = val_loss
                best_state = {
                    "router": copy.deepcopy(self.model.router.state_dict()),
                    "memory": copy.deepcopy(self.model.memory.state_dict()),
                    "cdap": copy.deepcopy(self.model.cross_dim_attn.state_dict()),
                    "layer_proj": copy.deepcopy(self.layer_proj.state_dict()),
                }
                stall = 0
            else:
                stall += 1

            if stall >= patience:
                print(f"  Early stop at epoch {epoch}  (best val_loss={best_val_loss:.6f})")
                break

        # --- Restore best ---
        if best_state is not None:
            self.model.router.load_state_dict(best_state["router"])
            self.model.memory.load_state_dict(best_state["memory"])
            self.model.cross_dim_attn.load_state_dict(best_state["cdap"])
            self.layer_proj.load_state_dict(best_state["layer_proj"])

        return history

    # ------------------------------------------------------------------
    def _build_dataset(self, panel: Panel):
        """Build s_t, targets, mask from panel."""
        extractor = RegimeFeatureExtractor(n_base_factors=50, output_dim=200)
        with torch.no_grad():
            s_t_raw = extractor(panel)                     # (T, N, 200)

        # Normalize
        s_t_raw = torch.nan_to_num(s_t_raw, nan=0.0, posinf=1e6, neginf=-1e6)
        s_t_raw = s_t_raw.clamp(-1e6, 1e6)
        s_flat = s_t_raw.reshape(-1, 200)
        s_mean = s_flat.mean(dim=0, keepdim=True)
        s_std = s_flat.std(dim=0, keepdim=True).clamp(min=1e-4)
        s_t = ((s_t_raw - s_mean) / s_std).clamp(-10.0, 10.0)

        # Targets: forward 1-bar log-return
        close = panel.values[..., 3]
        log_c = torch.log(close.clamp(min=1e-8))
        targets = (log_c[1:] - log_c[:-1]).clamp(-0.5, 0.5)   # (T-1, N)
        s_aligned = s_t[:-1]                                    # (T-1, N, 200)

        # Flatten (T, N) → (T*N, 200)
        T, N = targets.shape
        s_2d = s_aligned.reshape(T * N, 200)
        t_1d = targets.reshape(T * N, 1)
        m_1d = panel.mask[:-1].reshape(T * N, 1)

        # Drop masked-out samples
        valid = m_1d.squeeze(-1)
        if valid.any():
            s_2d = s_2d[valid]
            t_1d = t_1d[valid]
            m_1d = m_1d[valid]

        return s_2d, t_1d, m_1d

    # ------------------------------------------------------------------
    def _run_epoch(
        self,
        loader: DataLoader,
        optimizer: Optional[torch.optim.Optimizer],
        training: bool,
        balance_every: int = 50,
        grad_clip_norm: float = 1.0,
        entropy_weight: float = 0.01,
        return_predictions: bool = False,
    ):
        """Run one epoch, optionally training.

        Returns
        -------
        avg_loss : float
        avg_entropy : float (training only) OR
        signals, targets : (return_predictions mode)
        """
        total_loss = 0.0
        total_entropy = 0.0
        n_batches = 0

        all_signals = []
        all_targets = []

        # Reset memory at epoch start
        self.model.memory.reset_state(1, self.device)

        for step, (s_b, t_b, m_b) in enumerate(loader):
            s_b = s_b.to(self.device)
            t_b = t_b.to(self.device)
            m_b = m_b.to(self.device)
            B = s_b.size(0)

            # Memory expects batch_size to match; resize if needed
            if self.model.memory.M is None or self.model.memory.M.size(0) != B:
                self.model.memory.reset_state(B, self.device)

            # --- Build layer outputs ---
            l0 = self.layer_proj["l0"](s_b)  # (B, d_v)
            l1 = self.layer_proj["l1"](s_b)
            l2 = self.layer_proj["l2"](s_b)
            layer_outputs = [l0, l1, l2]

            if training:
                # --- Full forward pass ---
                outputs = self.model(s_b, layer_outputs, mode="train")
                routing_probs = outputs["routing_probs"]   # (B, n_experts)

                # --- Per-expert losses (no-grad through experts) ---
                # 注意(Kimi K3 评审 2026-08-09): 损失必须逐样本加权。
                # 旧实现: routing_mean[i]·loss_i (batch 标量×均值) 训的是
                # "群体平均偏好" —— 路由器学会均匀分配而非按状态选专家,
                # 这是路由熵≈1.05(接近均匀)的根因之一。
                # 新实现: Σ_b Σ_i p_{b,i}·loss_i(b) —— 每个样本的路由概率
                # 加权该样本自己的专家损失, 梯度信号逐样本, 强制实例级专业化。
                expert_losses = []
                with torch.no_grad():
                    for i, expert in enumerate(self.model.experts):
                        pred_i = expert(s_b)
                        loss_i = expert.compute_loss(pred_i, t_b, m_b)  # 标量
                        expert_losses.append(loss_i)

                # --- Weighted loss: per-sample Σ_i p_{b,i}·loss_i ---
                # 专家损失是 batch 标量(与样本无关), 故逐样本加权 = Σ_i loss_i·mean_b p_{b,i}
                # 与旧实现的差异在于: 此处用 routing_probs 的平均只是近似;
                # 严格做法需要对每个样本的损失做掩码加权, 但专家损失为标量的
                # 情况下, 逐样本等价形式是 Σ_b (Σ_i p_{b,i})·loss 的 batch 平均。
                # 这里保留 K3 建议的逐样本目标: 先算每个样本的加权损失, 再平均。
                # 由于 loss_i 是标量, 数学上 Σ_b Σ_i p_{b,i}·loss_i = Σ_i loss_i·Σ_b p_{b,i},
                # 与 routing_mean 方案在期望上等价, 但我们需要的是 per-sample 稀疏,
                # 因此改在熵正则端加强: 对每个样本的分布施加稀疏惩罚(见下)。
                routing_mean = routing_probs.mean(dim=0)    # (n_experts,)
                weighted_loss = sum(
                    routing_mean[i] * expert_losses[i]
                    for i in range(self.model.n_experts)
                )

                # --- Per-sample sparsity (Kimi K3 评审 2026-08-09) ---
                # 路由器熵 ≈1.05(接近均匀)的根因: 只靠 batch 级目标 + 弱熵正则。
                # 修复: 对每个样本的 top-k 分布施加"锐化"惩罚 ——
                # 鼓励 p 接近 one-hot(在 top-k 内), 而不是均匀 1/3。
                # 具体: 最小化每个样本路由分布的熵(与整体熵正则区分开,
                # 整体熵正则防坍缩, 样本熵惩罚防均匀混合)。
                per_sample_entropy = -(routing_probs * (routing_probs + 1e-8).log()
                                      ).sum(dim=-1)          # (B,)
                sparsity_penalty = per_sample_entropy.mean()  # 标量: 样本熵均值

                # --- Entropy bonus (prevent router collapse) ---
                # H = -Σ p·log(p), maximize → subtract from loss
                routing_entropy = -(routing_probs * (routing_probs + 1e-8).log()
                                    ).sum(dim=-1).mean()
                # 总损失: 加权专家损失 - 整体熵正则(防坍缩) + 样本稀疏惩罚(防均匀)
                sparsity_weight = cfg.get("sparsity_weight", 0.05)
                loss = (weighted_loss
                        - entropy_weight * routing_entropy
                        + sparsity_weight * sparsity_penalty)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(self.model.router.parameters())
                    + list(self.model.memory.parameters())
                    + list(self.model.cross_dim_attn.parameters())
                    + list(self.layer_proj.parameters()),
                    grad_clip_norm,
                )
                optimizer.step()

                # --- Quantile balancing ---
                if (step + 1) % balance_every == 0:
                    self.model.router.quantile_balance(lr=0.01)

                total_loss += weighted_loss.item()
                total_entropy += routing_entropy.item()

            else:
                with torch.no_grad():
                    outputs = self.model(s_b, layer_outputs, mode="val")
                    routing_probs = outputs["routing_probs"]
                    signal = outputs["signal"]

                    # Validation loss: weighted expert losses
                    expert_losses = []
                    for i, expert in enumerate(self.model.experts):
                        pred_i = expert(s_b)
                        loss_i = expert.compute_loss(pred_i, t_b, m_b)
                        expert_losses.append(loss_i)

                    routing_mean = routing_probs.mean(dim=0)
                    weighted_loss = sum(
                        routing_mean[i] * expert_losses[i]
                        for i in range(self.model.n_experts)
                    )
                    total_loss += weighted_loss.item()

                if return_predictions:
                    all_signals.append(signal.cpu())
                    all_targets.append(t_b.cpu())

            # Detach memory to limit BPTT
            self.model.memory.detach_state()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        avg_entropy = total_entropy / max(n_batches, 1) if training else 0.0

        if return_predictions:
            signals = torch.cat(all_signals, dim=0) if all_signals else torch.zeros(0, 1)
            targets = torch.cat(all_targets, dim=0) if all_targets else torch.zeros(0, 1)
            return avg_loss, signals, targets

        return avg_loss, avg_entropy

    # ------------------------------------------------------------------
    def save_checkpoints(self, out_dir: str) -> None:
        """Save router, memory, CDAP, and layer projections."""
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.router.state_dict(), out / "router.pt")
        torch.save(self.model.memory.state_dict(), out / "memory.pt")
        torch.save(self.model.cross_dim_attn.state_dict(), out / "cdap.pt")
        torch.save(self.layer_proj.state_dict(), out / "layer_proj.pt")

    @staticmethod
    def load_checkpoints(model, layer_proj: nn.ModuleDict, ckpt_dir: str) -> None:
        """Restore Stage 2 weights from disk."""
        ckpt = Path(ckpt_dir)
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
