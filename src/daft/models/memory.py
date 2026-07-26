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


class KDAMarketMemory(nn.Module):
    """KDA-style market memory with route-modulated forgetting.

    Maintains a fixed-size memory matrix M ∈ R^{d_k × d_v} that is updated
    at each timestep via a delta-rule (online gradient descent on the
    reconstruction objective).

    Parameters
    ----------
    d_k : int
        Key dimension = number of memory slots. Default: 128.
    d_v : int
        Value dimension = information stored per slot. Default: 64.
    d_feature : int
        Input feature dimension. Default: 200.
    bottleneck_ratio : int
        Forget gate low-rank compression ratio (KDA FineGrainedGating).
        Default: 4 (200 → 32 → 128 for the forget gate).
    use_route_modulation : bool
        Enable Router → Memory CDAP connection. Default: True.
    """

    def __init__(
        self,
        d_k: int = 128,
        d_v: int = 64,
        d_feature: int = 200,
        bottleneck_ratio: int = 4,
        use_route_modulation: bool = True,
    ):
        super().__init__()
        self.d_k = d_k
        self.d_v = d_v
        self.d_feature = d_feature
        self.use_route_modulation = use_route_modulation

        bottleneck_dim = d_k // bottleneck_ratio

        # === Per-channel forget gate (KDA FineGrainedGating) ===
        # Low-rank bottleneck: d_feature → bottleneck → d_k
        self.forget_down = nn.Linear(d_feature, bottleneck_dim)
        self.forget_up = nn.Linear(bottleneck_dim, d_k)

        # === Learnable per-step learning rate β_t ===
        self.beta_proj = nn.Sequential(
            nn.Linear(d_feature, 1),
            nn.Sigmoid(),
        )

        # === Query / Key / Value projections ===
        self.q_proj = nn.Linear(d_feature, d_k)
        self.k_proj = nn.Linear(d_feature, d_k)
        self.v_proj = nn.Linear(d_feature, d_v)

        # === CDAP: Router → Memory modulation ===
        if use_route_modulation:
            # z_t (latent_dim=16) → d_k forget gate modulation
            self.route_modulate = nn.Linear(16, d_k)

        # State matrix (created dynamically per batch)
        self.M: Optional[torch.Tensor] = None

        # === CDAP: external gate modulation (set by ensemble after CDAP forward) ===
        # memory_gate ∈ (0,1)^{d_k} from CDAP's joint→memory reverse projection.
        # Applied at the NEXT timestep's forget-gate computation to close the
        # Depth→Memory feedback loop in the CDAP triad.
        self._external_gate: Optional[torch.Tensor] = None  # (B, d_k)

    def reset_state(self, batch_size: int, device: torch.device):
        """(Re)initialize the memory matrix to zeros.

        Parameters
        ----------
        batch_size : int
            Number of independent sequences in the batch.
        device : torch.device
        """
        self.M = torch.zeros(batch_size, self.d_k, self.d_v, device=device)
        self._external_gate = None  # Stale gate instructions invalid with new state

    def detach_state(self):
        """Detach memory state from computation graph (call after backward)."""
        if self.M is not None:
            self.M = self.M.detach()
        if self._external_gate is not None:
            self._external_gate = self._external_gate.detach()

    def forward(
        self,
        s_t: torch.Tensor,
        z_t: Optional[torch.Tensor] = None,
        reset: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Process one timestep through the KDA market memory.

        Parameters
        ----------
        s_t : torch.Tensor, shape (B, d_feature)
            Market state vector at current timestep.
        z_t : torch.Tensor, shape (B, 16), optional
            Routing latent vector. If provided and use_route_modulation=True,
            modulates the forget gate. This is the CDAP Router → Memory link.
        reset : bool
            If True, reinitialize memory before processing.

        Returns
        -------
        retrieved : torch.Tensor, shape (B, d_v)
            Retrieved memory content for the current query.
        M_t : torch.Tensor, shape (B, d_k, d_v)
            Updated memory matrix state.
        """
        B = s_t.size(0)
        device = s_t.device

        # Initialize memory if needed
        if self.M is None or reset or self.M.size(0) != B:
            self.reset_state(B, device)

        # === Step 1: Per-channel forget gate (KDA FineGrainedGating) ===
        alpha = self.forget_down(s_t)        # (B, bottleneck)
        alpha = F.silu(alpha)                # SiLU activation
        alpha = self.forget_up(alpha)        # (B, d_k)
        alpha = torch.sigmoid(alpha)         # α ∈ (0, 1)^{d_k}

        # === CDAP: Router → Memory modulation ===
        if self.use_route_modulation and z_t is not None:
            route_mod = torch.sigmoid(self.route_modulate(z_t))  # (B, d_k)
            alpha = alpha * route_mod  # Element-wise modulation

        # === CDAP: Depth → Memory modulation (external gate from CDAP) ===
        if self._external_gate is not None:
            alpha = alpha * self._external_gate
            self._external_gate = None  # Consume: single-use, applied once

        # === Step 2: Compute β_t (learnable learning rate) ===
        beta = self.beta_proj(s_t)  # (B, 1), β ∈ (0, 1)

        # === Step 3: Query / Key / Value projections ===
        k = self.k_proj(s_t)  # (B, d_k)
        k = F.normalize(k, dim=-1)  # L2-normalize (KDA stability requirement)

        v = self.v_proj(s_t)  # (B, d_v)
        q = self.q_proj(s_t)  # (B, d_k)

        # === Step 4: KDA Delta-Rule Update ===

        # 4a: Apply per-channel forget (diagonal decay)
        # α ∈ (B, d_k) → broadcast to (B, d_k, d_v)
        self.M = alpha.unsqueeze(-1) * self.M

        # 4b: Delta correction: remove conflicting old information
        # M_k = M · k, shape (B, d_v)
        M_k = torch.einsum('bkv,bk->bv', self.M, k)

        # M = M - β · k ⊗ (M · k)
        self.M = self.M - beta.unsqueeze(-1) * torch.einsum(
            'bk,bv->bkv', k, M_k
        )

        # 4c: KV write: add new information
        # M = M + β · k ⊗ v
        self.M = self.M + beta.unsqueeze(-1) * torch.einsum(
            'bk,bv->bkv', k, v
        )

        # === Step 5: Memory retrieval ===
        # o = M^T · q, shape (B, d_v)
        retrieved = torch.einsum('bkv,bk->bv', self.M, q)

        # RMSNorm on output (KDA practice)
        retrieved = self._rms_norm(retrieved)

        return retrieved, self.M.clone()

    def set_external_gate(self, gate: Optional[torch.Tensor]) -> None:
        """Set an external forget-gate modulation from CDAP.

        Called by ExpertEnsemble after CDAP forward to close the
        Depth→Memory feedback loop. The gate is applied in the NEXT
        forward() call and then cleared.

        Parameters
        ----------
        gate : torch.Tensor, shape (B, d_k), optional
            Memory gate from CDAP joint→memory reverse projection.
            Values in (0, 1). Set to None to clear.
        """
        self._external_gate = gate

    @staticmethod
    def _rms_norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        """Root Mean Square Layer Normalization (used in KDA)."""
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + eps)
        return x / rms

    def get_memory_summary(self) -> torch.Tensor:
        """Return a flattened summary of the current memory state.

        Used by the Cross-Dimension Attention Protocol to incorporate
        memory state into joint modulation.

        Returns
        -------
        summary : torch.Tensor, shape (B, d_k * d_v)
            Flattened memory matrix.
        """
        if self.M is None:
            raise RuntimeError("Memory not initialized. Call forward() first.")
        return self.M.reshape(self.M.size(0), -1)
