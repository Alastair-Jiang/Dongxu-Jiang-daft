"""Component 2: KDA Market Memory — Delta-attention memory for financial time series.

Inspired by Kimi Delta Attention (KDA):

Core formula (from KDA paper, arXiv:2510.26692):
    S_t = (I - β_t · k_t · k_t^T) · Diag(α_t) · S_{t-1} + β_t · k_t · v_t^T

DAFT adaptation for financial time series:

    M_t = M_{t-1} - β_t · k_t ⊗ (M_{t-1} · k_t) + β_t · k_t ⊗ v_t

with the CDAP extension: the per-channel forget gate α_t is modulated by
the routing latent vector z_t from the Regime Router.

    α'_t = α_t ⊙ σ(W_route · z_t)

This is the Router → Memory connection in the CDAP triad.

Key properties:
- O(d_k · d_v) per step, independent of sequence length
- No KV-cache: state matrix M_t has fixed size (128 × 64 = 8192 floats)
- Per-channel forgetting: each of the 128 memory slots has its own decay rate
- Route-aware: forgetting policy adapts to the current market regime
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class SiTU(nn.Module):
    """Sigmoid Tanh Unit — K3 activation: σ(x) ⊙ tanh(x).

    Naturally bounded in [-1, 1] with smooth gradients.
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(x) * torch.tanh(x)


class KDAMarketMemory(nn.Module):
    """KDA-style market memory with route-modulated forgetting.

    Upgraded to full K3 spec (arXiv:2510.26692):
    - SiTU activation (replaces SiLU)
    - Safe forget gate with A_log / dt_bias learnable parameters
    - Output gate (low-rank sigmoid) on retrieved memory
    - Route-modulated forgetting (CDAP connection)

    Maintains a fixed-size memory matrix M ∈ R^{d_k × d_v} updated
    at each timestep via delta-rule online gradient descent.

    Parameters
    ----------
    d_k : int  — Key dim / number of memory slots. Default: 128.
    d_v : int  — Value dim / stored info per slot. Default: 64.
    d_feature : int  — Input feature dim. Default: 200.
    bottleneck_ratio : int  — Forget gate low-rank compression. Default: 4.
    use_route_modulation : bool  — Enable Router → Memory CDAP. Default: True.
    safe_gate_lower_bound : float  — K3 safe gate floor. Default: 0.001.
    """

    def __init__(
        self,
        d_k: int = 128,
        d_v: int = 64,
        d_feature: int = 200,
        bottleneck_ratio: int = 4,
        use_route_modulation: bool = True,
        safe_gate_lower_bound: float = 0.001,
    ):
        super().__init__()
        self.d_k = d_k
        self.d_v = d_v
        self.d_feature = d_feature
        self.use_route_modulation = use_route_modulation
        self.lower_bound = safe_gate_lower_bound

        bottleneck_dim = d_k // bottleneck_ratio

        # === Per-channel forget gate (KDA FineGrainedGating) ===
        self.forget_down = nn.Linear(d_feature, bottleneck_dim)
        self.forget_up = nn.Linear(bottleneck_dim, d_k)

        # === K3 Safe Gate: learnable decay parameters ===
        self.A_log = nn.Parameter(torch.randn(d_k) * 0.1)   # per-channel log decay
        self.dt_bias = nn.Parameter(torch.zeros(d_k))        # per-channel dt bias

        # === Learnable per-step learning rate β_t ===
        self.beta_proj = nn.Sequential(
            nn.Linear(d_feature, 1),
            nn.Sigmoid(),
        )

        # === Q / K / V projections ===
        self.q_proj = nn.Linear(d_feature, d_k)
        self.k_proj = nn.Linear(d_feature, d_k)
        self.v_proj = nn.Linear(d_feature, d_v)

        # === Output gate (K3 spec) — low-rank sigmoid on retrieved content ===
        out_bottleneck = max(d_v // 4, 4)
        self.out_gate_down = nn.Linear(d_feature, out_bottleneck)
        self.out_gate_up = nn.Linear(out_bottleneck, d_v)

        # === CDAP: Router → Memory modulation ===
        if use_route_modulation:
            self.route_modulate = nn.Linear(16, d_k)

        # State matrix (created dynamically per batch)
        self.M: Optional[torch.Tensor] = None

    def reset_state(self, batch_size: int, device: torch.device):
        """Re-initialize the memory matrix to zeros."""
        self.M = torch.zeros(batch_size, self.d_k, self.d_v, device=device)

    def detach_state(self):
        """Detach memory state from computation graph."""
        if self.M is not None:
            self.M = self.M.detach()

    def forward(
        self,
        s_t: torch.Tensor,
        z_t: Optional[torch.Tensor] = None,
        reset: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Process one timestep through the KDA market memory."""
        B = s_t.size(0)
        device = s_t.device

        if self.M is None or reset or self.M.size(0) != B:
            self.reset_state(B, device)

        # === Step 1: Per-channel forget gate (KDA FineGrainedGating) ===
        alpha = self.forget_down(s_t)            # (B, bottleneck)
        alpha = SiTU()(alpha)                    # SiTU (K3 spec, was SiLU)
        alpha = self.forget_up(alpha)            # (B, d_k)

        # K3 Safe Gate:  α = lower_bound · σ(exp(A_log) · (input + dt_bias))
        alpha = self.lower_bound * torch.sigmoid(
            torch.exp(self.A_log) * (alpha + self.dt_bias)
        )  # α ∈ (lower_bound, 1)^{d_k}

        # === CDAP: Router → Memory modulation ===
        if self.use_route_modulation and z_t is not None:
            route_mod = torch.sigmoid(self.route_modulate(z_t))   # (B, d_k)
            alpha = alpha * route_mod

        # === Step 2: β_t (learnable learning rate) ===
        beta = self.beta_proj(s_t)   # (B, 1), β ∈ (0, 1)

        # === Step 3: Q / K / V projections ===
        k = self.k_proj(s_t)
        k = F.normalize(k, dim=-1)    # L2-norm  (KDA stability)
        v = self.v_proj(s_t)
        q = self.q_proj(s_t)

        # === Step 4: KDA Delta-Rule Update ===

        # 4a: Per-channel forget
        self.M = alpha.unsqueeze(-1) * self.M

        # 4b: Delta correction  M -= β · k ⊗ (M · k)
        M_k = torch.einsum('bkv,bk->bv', self.M, k)
        self.M = self.M - beta.unsqueeze(-1) * torch.einsum(
            'bk,bv->bkv', k, M_k
        )

        # 4c: KV write  M += β · k ⊗ v
        self.M = self.M + beta.unsqueeze(-1) * torch.einsum(
            'bk,bv->bkv', k, v
        )

        # === Step 5: Memory retrieval  o = M^T · q ===
        retrieved = torch.einsum('bkv,bk->bv', self.M, q)

        # === Output gate (K3 spec): y = σ(W_up · W_down · s_t) ⊙ o ===
        out_gate = self.out_gate_down(s_t)
        out_gate = SiTU()(out_gate)
        out_gate = self.out_gate_up(out_gate)
        out_gate = torch.sigmoid(out_gate)          # (B, d_v), ∈ (0, 1)
        retrieved = retrieved * out_gate

        # RMSNorm on output
        retrieved = self._rms_norm(retrieved)

        return retrieved, self.M.clone()

    @staticmethod
    def _rms_norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + eps)
        return x / rms

    def get_memory_summary(self) -> torch.Tensor:
        if self.M is None:
            raise RuntimeError("Memory not initialized. Call forward() first.")
        return self.M.reshape(self.M.size(0), -1)
