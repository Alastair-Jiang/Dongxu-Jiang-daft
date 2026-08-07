"""Component 3: Cross-Dimension Attention Protocol (CDAP). ★ ORIGINAL CONTRIBUTION ★

The core methodological innovation of DAFT. Establishes bidirectional
modulation among three information dimensions:

    Dimension 1 (Spatial / Expert):  routing distribution p_t
    Dimension 2 (Temporal / Memory): memory state M_t
    Dimension 3 (Depth / Feature):   layer-wise features [h0, h1, h2]

These are NOT treated as independent streams. They project into a shared
joint latent space where they modulate one another through element-wise
multiplication (strong inductive bias for sparse, regime-specific computation),
then project back as corrected signals.

Joint Space Fusion (key design choice):
    j = e ⊙ m ⊙ d   ∈  R^{64}

    WHY element-wise product, not addition?
    → Addition assumes orthogonal contributions. In financial markets,
      routing, memory, and depth are inherently coupled — a trending
      market discovered by the router should change what the memory retains.
      Multiplication ensures: if any dimension has near-zero activation
      (e.g., memory is uncertain), it silences cross-modulation rather
      than adding noise.

Reverse Projections:
    → Router:  p'_t = softmax(log p_t + δ · W_expert · j)
    → Memory:  g_t = σ(W_mem · j)          [additional forget-gate modulation]
    → Depth:   w_t = softmax(W_depth · j)   [cross-layer retrieval weights]

Design derivation:
    Inspired by K3's AttnRes (cross-layer attention) and the observation
    that K3's KDA forget gates, MoE routing, and AttnRes retrieval are
    architecturally isolated — each makes decisions without awareness
    of the other two. CDAP connects them.
"""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossDimensionAttention(nn.Module):
    """Cross-Dimension Attention Protocol: three-way bidirectional modulation.

    Parameters
    ----------
    n_experts : int
        Number of strategy experts (dimension of routing distribution).
    d_k : int
        Memory key dimension.
    d_v : int
        Memory value dimension.
    n_layers : int
        Number of feature hierarchy layers. Default: 3 (L0 raw, L1 base, L2 composite).
    joint_dim : int
        Dimension of the shared joint latent space. Default: 64.
    modulation_strength : float
        Scale factor δ for back-projection signals. Lower = more conservative
        modulation. Default: 1.0 (full modulation during joint training),
        0.1 during Stage 2 (router+memory only).
    """

    def __init__(
        self,
        n_experts: int = 10,
        d_k: int = 128,
        d_v: int = 64,
        n_layers: int = 3,
        joint_dim: int = 64,
        modulation_strength: float = 1.0,
    ):
        super().__init__()
        self.n_experts = n_experts
        self.d_k = d_k
        self.d_v = d_v
        self.n_layers = n_layers
        self.joint_dim = joint_dim
        self.modulation_strength = modulation_strength

        # === Forward projections: each dimension -> joint space ===
        self.expert_to_joint = nn.Sequential(
            nn.Linear(n_experts, joint_dim),
            nn.LayerNorm(joint_dim),
        )
        # Memory: pool over d_k slots first, then project (d_v -> joint_dim)
        # This avoids a 1M+ param flatten-and-project bottleneck
        self.memory_to_joint = nn.Sequential(
            nn.Linear(d_v, joint_dim * 2),
            nn.SiLU(),
            nn.Linear(joint_dim * 2, joint_dim),
            nn.LayerNorm(joint_dim),
        )
        self.depth_to_joint = nn.Sequential(
            nn.Linear(d_v * n_layers, joint_dim * 2),
            nn.SiLU(),
            nn.Linear(joint_dim * 2, joint_dim),
            nn.LayerNorm(joint_dim),
        )

        # === Reverse projections: joint space → each dimension ===
        # → Router bias correction (additive in logit space)
        self.joint_to_expert_bias = nn.Linear(joint_dim, n_experts)

        # → Memory gate modulation (additional forget signal)
        self.joint_to_memory_gate = nn.Sequential(
            nn.Linear(joint_dim, d_k // 4),
            nn.SiLU(),
            nn.Linear(d_k // 4, d_k),
        )

        # → Depth weight recalibration
        self.joint_to_depth_weights = nn.Linear(joint_dim, n_layers)

        # Learnable initial bias for residual-style modulation
        self.expert_bias_scale = nn.Parameter(torch.zeros(1))
        self.memory_gate_scale = nn.Parameter(torch.zeros(1))
        self.depth_weight_scale = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        routing_probs: torch.Tensor,
        memory_matrix: torch.Tensor,
        layer_outputs: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Execute the full CDAP modulation cycle.

        Parameters
        ----------
        routing_probs : torch.Tensor, shape (B, n_experts)
            Current expert routing distribution (post-softmax).
        memory_matrix : torch.Tensor, shape (B, d_k, d_v)
            Current memory state matrix.
        layer_outputs : list of torch.Tensor, each shape (B, d_v)
            Feature hierarchy: [L0 (raw), L1 (base factors), L2 (composite)].
            Each layer output should be a d_v-dimensional representation.

        Returns
        -------
        routing_modulated : torch.Tensor, shape (B, n_experts)
            Corrected routing distribution.
        memory_gate : torch.Tensor, shape (B, d_k)
            Additional forget-gate modulation for KDA Memory.
        depth_weights : torch.Tensor, shape (B, n_layers)
            Cross-layer retrieval weights (sum to 1).
        fused_layers : torch.Tensor, shape (B, d_v)
            Depth-weighted fusion of layer outputs.
        """
        B = routing_probs.size(0)
        device = routing_probs.device

        # === Step 1: Project each dimension into joint space ===
        e = self.expert_to_joint(routing_probs)                    # (B, joint_dim)

        # Pool memory over d_k slots (mean), giving (B, d_v), then project
        m = memory_matrix.mean(dim=1)                               # (B, d_v)
        m = self.memory_to_joint(m)                                 # (B, joint_dim)

        h_stacked = torch.stack(layer_outputs, dim=-1)             # (B, d_v, n_layers)
        d = h_stacked.reshape(B, -1)                               # (B, d_v * n_layers)
        d = self.depth_to_joint(d)                                 # (B, joint_dim)

        # === Step 2: Joint space fusion via element-wise multiplication ===
        # Each dimension's activation gates the others.
        # If the memory is uncertain (low activations in m), it cannot
        # distort the routing signal. This is the key inductive bias.
        joint = e * m * d  # (B, joint_dim)

        # === Step 3: Reverse projections — from joint back to each dimension ===

        # → Router: additive bias in logit space
        expert_bias = self.joint_to_expert_bias(joint)             # (B, n_experts)
        expert_bias = expert_bias * self.expert_bias_scale.tanh()  # Learned scale
        routing_modulated = routing_probs + (
            self.modulation_strength * expert_bias
        )
        # Ensure valid probability distribution
        routing_modulated = F.softmax(routing_modulated, dim=-1)

        # → Memory: forget gate modulation signal
        memory_gate_raw = self.joint_to_memory_gate(joint)         # (B, d_k)
        memory_gate_raw = memory_gate_raw * self.memory_gate_scale.tanh()
        memory_gate = torch.sigmoid(memory_gate_raw)               # (B, d_k), ∈ (0,1)

        # → Depth: cross-layer weights
        depth_raw = self.joint_to_depth_weights(joint)             # (B, n_layers)
        depth_raw = depth_raw * self.depth_weight_scale.tanh()
        depth_weights = F.softmax(depth_raw, dim=-1)               # (B, n_layers), sum=1

        # === Step 4: Depth-weighted layer fusion ===
        fused_layers = sum(
            depth_weights[:, k:k+1] * layer_outputs[k]
            for k in range(self.n_layers)
        )  # (B, d_v)

        return routing_modulated, memory_gate, depth_weights, fused_layers
