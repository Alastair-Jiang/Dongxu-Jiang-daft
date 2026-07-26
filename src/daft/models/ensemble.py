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

            # Close the CDAP feedback loop: Depth → Memory modulation
            # memory_gate will be applied at the NEXT timestep's forget-gate
            self.memory.set_external_gate(memory_gate)

            # Use modulated routing weights
            final_routing = routing_mod
        else:
            # Try hardened fast path — per-sample decision
            # Each sample in the batch may be in a different regime and
            # should independently decide whether to use a cached fast path.
            regime_id = self.router.get_regime_id(z_t)  # (B,)

            # Classify each sample: fast path (cached) or slow path (full CDAP)
            fast_mask = torch.zeros(B, dtype=torch.bool, device=device)
            fast_cached_weights: list = []  # one cached weight tensor per fast sample

            for i in range(B):
                rid = regime_id[i].item()
                rprob = full_probs[i]  # (n_experts,) — per-sample routing distribution

                if self.hardening.should_use_fast_path(rid, rprob):
                    fast_mask[i] = True
                    cached = self.hardening.get_cached_weights(rid, rprob)
                    fast_cached_weights.append(cached.to(device))

            # --- Route each group appropriately ---
            final_routing = torch.zeros(B, self.n_experts, device=device)
            depth_weights = torch.zeros(B, 3, device=device)
            fused_layers = torch.zeros(B, self.memory.d_v, device=device)

            n_fast = fast_mask.sum().item()
            n_slow = B - n_fast

            # Fast-path samples: use hardened cached weights
            if n_fast > 0:
                fast_idx = fast_mask.nonzero(as_tuple=True)[0]  # indices into batch
                fast_weights_t = torch.stack(fast_cached_weights, dim=0)  # (n_fast, n_experts)
                final_routing[fast_idx] = fast_weights_t
                # Uniform depth weights in fast path (no CDAP modulation)
                depth_weights[fast_idx] = 1.0 / 3
                # Uniform layer fusion
                for k in range(3):
                    fused_layers[fast_idx] += (1.0 / 3) * layer_outputs[k][fast_idx]

            # Slow-path samples: full CDAP modulation
            if n_slow > 0:
                slow_idx = (~fast_mask).nonzero(as_tuple=True)[0]

                # Slice memory and inputs to slow-path samples only
                memory_matrix_slow = self.memory.M[slow_idx].clone()
                routing_probs_slow = full_probs[slow_idx]
                layer_outputs_slow = [h[slow_idx] for h in layer_outputs]

                routing_mod_slow, memory_gate_slow, dw_slow, fl_slow = \
                    self.cross_dim_attn(
                        routing_probs=routing_probs_slow,
                        memory_matrix=memory_matrix_slow,
                        layer_outputs=layer_outputs_slow,
                    )

                final_routing[slow_idx] = routing_mod_slow
                depth_weights[slow_idx] = dw_slow
                fused_layers[slow_idx] = fl_slow

                # Close CDAP feedback loop for slow-path samples
                # (fast-path samples have no CDAP → no gate modulation → pass None)
                # Note: set_external_gate applies batch-wide; for mixed batches
                # we write the slow-path gates into a full-size tensor and
                # leave fast-path slots as ones (no modulation).
                full_gate = torch.ones(B, self.memory.d_k, device=device)
                full_gate[slow_idx] = memory_gate_slow
                self.memory.set_external_gate(full_gate)

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
