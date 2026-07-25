"""Expert Ensemble: fuses strategy expert signals into a single trading signal.

The ensemble is the final stage of the DAFT pipeline. It:
1. Collects predictions from activated experts (via Router Top-K)
2. Applies CDAP-modulated routing weights
3. Fuses multi-layer features via CDAP depth weights
4. Optionally uses hardened fast-path weights (bypassing CDAP)
5. Produces a final signal: expected return for the next bar
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ExpertEnsemble(nn.Module):
    """Top-level model that combines all DAFT components.

    Parameters
    ----------
    experts : nn.ModuleList
        List of strategy expert modules.
    router : RegimeRouter
        Regime-aware expert router.
    memory : KDAMarketMemory
        KDA-style market memory.
    cross_dim_attn : CrossDimensionAttention
        Cross-Dimension Attention Protocol module.
    hardening : HardeningEngine
        Adaptive Hardening mechanism.
    """

    def __init__(
        self,
        experts: nn.ModuleList,
        router: nn.Module,
        memory: nn.Module,
        cross_dim_attn: nn.Module,
        hardening,
    ):
        super().__init__()
        self.experts = experts
        self.router = router
        self.memory = memory
        self.cross_dim_attn = cross_dim_attn
        self.hardening = hardening

        self.n_experts = len(experts)

    def forward(
        self,
        s_t: torch.Tensor,
        layer_outputs: List[torch.Tensor],
        mode: str = "train",
        use_hardening: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Full DAFT forward pass.

        Parameters
        ----------
        s_t : torch.Tensor, shape (B, 200)
            Market state vector.
        layer_outputs : list of torch.Tensor
            [L0_raw, L1_base, L2_composite], each shape (B, d_v).
        mode : str
            "train", "val", or "inference".
        use_hardening : bool
            Enable hardened fast-path lookup.

        Returns
        -------
        outputs : dict
            signal : torch.Tensor, shape (B, 1) — final trading signal
            routing_probs : torch.Tensor, shape (B, n_experts) — expert weights
            regime_id : torch.Tensor, shape (B,) — discrete regime
            depth_weights : torch.Tensor, shape (B, 3) — layer weights
            metadata : dict — hardening stats, CDAP modulation values
        """
        B = s_t.size(0)
        device = s_t.device

        # === Step 1: Route to experts ===
        topk_probs, topk_indices, z_t, full_probs = self.router(s_t, mode=mode)

        # === Step 2: Update and query memory ===
        retrieved, _ = self.memory(s_t, z_t=z_t)

        # === Step 3: Expert forward passes ===
        expert_outputs = []
        expert_hiddens = []
        for i, expert in enumerate(self.experts):
            signal_i, hidden_i = expert(s_t, return_hidden=True)
            expert_outputs.append(signal_i.squeeze(-1))    # (B,)
            expert_hiddens.append(hidden_i)                 # (B, hidden_dim)

        expert_outputs = torch.stack(expert_outputs, dim=-1)   # (B, n_experts)
        # Note: expert_hiddens have different dims (heterogeneous experts),
        # so we don't stack them. They're available per-expert if needed.

        # === Step 4: Cross-Dimension Attention Protocol ===
        if mode != "inference" or not use_hardening:
            # Full CDAP modulation
            memory_matrix = self.memory.M.clone()

            routing_mod, memory_gate, depth_weights, fused_layers = \
                self.cross_dim_attn(
                    routing_probs=full_probs,
                    memory_matrix=memory_matrix,
                    layer_outputs=layer_outputs,
                )

            # Apply memory gate to memory (additional forget modulation)
            # This will take effect in the NEXT timestep's memory update
            # (stored as a side-channel for the memory module)

            # Use modulated routing weights
            final_routing = routing_mod
        else:
            # Try hardened fast path
            regime_id = self.router.get_regime_id(z_t)
            routing_avg = full_probs.mean(dim=0)  # batch-average for hardening check

            if self.hardening.should_use_fast_path(
                regime_id[0].item(), routing_avg
            ):
                # Use cached weights (detached, no gradient)
                cached_weights = self.hardening.get_cached_weights(
                    regime_id[0].item(), routing_avg
                )
                final_routing = cached_weights.unsqueeze(0).expand(B, -1)
                depth_weights = F.softmax(
                    torch.ones(B, 3, device=device), dim=-1
                )  # Uniform depth weights in fast path
                fused_layers = sum(
                    depth_weights[:, k:k+1] * layer_outputs[k] for k in range(3)
                )
            else:
                # Regime shift detected → full CDAP
                memory_matrix = self.memory.M.clone()
                routing_mod, memory_gate, depth_weights, fused_layers = \
                    self.cross_dim_attn(
                        routing_probs=full_probs,
                        memory_matrix=memory_matrix,
                        layer_outputs=layer_outputs,
                    )
                final_routing = routing_mod

        # === Step 5: Weighted expert fusion ===
        # signal = Σ_i w_i · expert_i(s_t)
        signal = (final_routing * expert_outputs).sum(dim=-1, keepdim=True)  # (B, 1)

        # === Step 6: Depth-weighted layer fusion ===
        signal = signal + 0.1 * fused_layers.mean(dim=-1, keepdim=True)

        # === Step 7: Assemble outputs ===
        regime_id = self.router.get_regime_id(z_t) if mode == "inference" else None

        outputs = {
            "signal": signal,
            "routing_probs": final_routing,
            "regime_id": regime_id,
            "depth_weights": depth_weights,
            "fused_layers": fused_layers,
            "metadata": {
                "mode": mode,
                "fast_path_used": (
                    use_hardening and
                    self.hardening.n_fast_path > 0
                ),
                "hardening_stats": self.hardening.get_stats(),
            },
        }

        return outputs
