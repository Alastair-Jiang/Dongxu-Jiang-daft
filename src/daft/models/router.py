"""Component 1: Regime Router — Stable LatentMoE for financial time series.

Inspired by Kimi K3's Stable LatentMoE:
- Raw market state is projected into a low-dimensional latent regime space
- Routing decisions are made in this latent space (not the raw feature space)
- Quantile Balancing eliminates the need for auxiliary load-balancing losses
- Temperature schedule: soft routing during training → near-discrete after hardening

Key architectural insight from K3:
  896 experts, but only 16 activated per token. DAFT scales down: 8 experts,
  top-3 activated. The design principle is identical — broad knowledge via
  many specialists, low compute via sparse activation.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class SiTU(nn.Module):
    """Sigmoid Tanh Unit — K3 activation: σ(x) ⊙ tanh(x).

    Naturally bounded in [-1, 1] with smooth gradients.
    Critical for MoE routing stability — prevents expert activation
    magnitude drift that would distort router gradients.
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(x) * torch.tanh(x)


class RegimeRouter(nn.Module):
    """Market regime router using Stable LatentMoE design.

    Projects the 200-dimensional market state vector into a 16-dimensional
    latent regime space, then computes expert selection probabilities via
    temperature-scaled softmax with Top-K sparsification.

    Parameters
    ----------
    input_dim : int
        Dimension of the raw market state vector s_t. Default: 200.
    latent_dim : int
        Dimension of the regime latent space z_t. Default: 16.
        (K3 uses a latent bottleneck; 16 is sufficient for ~5-8 market regimes.)
    n_experts : int
        Total number of strategy experts. Default: 8.
    top_k : int
        Number of experts activated per forward pass. Default: 3.
    temperature : float
        Softmax temperature. Higher → softer routing. Default: 1.0.
    noisy_gating_std : float
        Standard deviation of Gaussian noise added during training for
        exploration (Super-Linear approach). Default: 0.1.
    """

    def __init__(
        self,
        input_dim: int = 200,
        latent_dim: int = 16,
        n_experts: int = 10,
        top_k: int = 3,
        temperature: float = 1.0,
        noisy_gating_std: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.n_experts = n_experts
        self.top_k = top_k
        self.temperature = temperature
        self.noisy_gating_std = noisy_gating_std

        # Latent space projection: 200 → 100 → 16 (two-layer bottleneck)
        # Matches K3's Stable LatentMoE design: project into a compact
        # regime space where routing is more stable
        self.proj_down = nn.Linear(input_dim, input_dim // 2)
        self.proj_up = nn.Linear(input_dim // 2, latent_dim)
        self.proj_norm = nn.LayerNorm(latent_dim)

        # Route from latent space to expert logits
        self.route = nn.Linear(latent_dim, n_experts)

        # Trainable bias for Quantile Balancing (K3 approach)
        # Initialized to zero — learns to balance expert utilization
        self.register_buffer("expert_bias", torch.zeros(n_experts))
        self.register_buffer("activation_counts", torch.zeros(n_experts))

        # SiTU activation (K3 spec): σ(x)·tanh(x), naturally bounded in [-1, 1]
        # Critical for MoE: prevents expert activation magnitude drift
        self.activation = SiTU()

    def forward(
        self,
        s_t: torch.Tensor,
        mode: str = "train",
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Route a batch of market state vectors to strategy experts.

        Parameters
        ----------
        s_t : torch.Tensor, shape (B, input_dim)
            Market state vectors at current timestep.
        mode : str
            "train": soft routing with exploration noise.
            "val": deterministic soft routing.
            "inference": near-discrete routing (use temp=0.1).

        Returns
        -------
        topk_probs : torch.Tensor, shape (B, top_k)
            Renormalized routing probabilities for activated experts.
        topk_indices : torch.Tensor, shape (B, top_k)
            Indices of activated experts.
        z_t : torch.Tensor, shape (B, latent_dim)
            Latent regime vector -- passed to CDAP for cross-dimension modulation.
        probs : torch.Tensor, shape (B, n_experts)
            Full routing distribution (for CDAP joint-space projection).
        """
        B = s_t.size(0)

        # === Step 1: Latent space projection (Stable LatentMoE) ===
        z_t = self.proj_down(s_t)
        z_t = self.activation(z_t)
        z_t = self.proj_up(z_t)
        z_t = self.proj_norm(z_t)  # (B, latent_dim)

        # === Step 2: Route from latent space (not raw features) ===
        logits = self.route(z_t)  # (B, n_experts)

        # Add Quantile Balancing bias
        logits = logits + self.expert_bias

        # Exploration noise during training (Super-Linear approach)
        if mode == "train" and self.noisy_gating_std > 0:
            noise = torch.randn_like(logits) * self.noisy_gating_std
            logits = logits + noise

        # === Step 3: Temperature-scaled softmax ===
        temp = self.temperature if mode != "inference" else 0.1
        probs = F.softmax(logits / temp, dim=-1)  # (B, n_experts)

        # === Step 4: Top-K sparsification ===
        topk_probs, topk_indices = torch.topk(probs, self.top_k, dim=-1)

        # Renormalize selected probabilities to sum to 1
        topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True)

        # === Step 5: Update activation statistics (for Quantile Balancing) ===
        if mode == "train":
            with torch.no_grad():
                counts = torch.bincount(
                    topk_indices.flatten(),
                    minlength=self.n_experts,
                ).float()
                self.activation_counts = (
                    0.99 * self.activation_counts + 0.01 * counts
                )

        return topk_probs, topk_indices, z_t, probs

    def quantile_balance(self, lr: float = 0.01) -> None:
        """Apply Quantile Balancing to expert bias (K3 approach).

        Called periodically during training. Adjusts bias to balance expert
        utilization — overloaded experts get lower bias, underutilized get
        higher bias. No auxiliary loss required.

        Parameters
        ----------
        lr : float
            Bias adjustment learning rate. Default: 0.01.
        """
        if self.activation_counts.sum() == 0:
            return

        current_frac = self.activation_counts / self.activation_counts.sum()
        target_frac = 1.0 / self.n_experts
        delta = lr * (target_frac - current_frac)
        self.expert_bias += delta

    @torch.no_grad()
    def get_regime_id(self, z_t: torch.Tensor) -> torch.Tensor:
        """Discretize the latent regime vector into a regime ID.

        Uses the argmax of the expert routing distribution as a proxy
        for the discrete regime cluster.

        Parameters
        ----------
        z_t : torch.Tensor, shape (B, latent_dim)

        Returns
        -------
        regime_id : torch.Tensor, shape (B,), dtype long
            Discrete regime identifier (0 … n_experts-1).
        """
        logits = self.route(z_t) + self.expert_bias
        return logits.argmax(dim=-1)
