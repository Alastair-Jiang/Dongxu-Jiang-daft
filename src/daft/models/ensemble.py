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
        ablate: str = "none",
    ):
        super().__init__()
        self.experts = experts
        self.router = router
        self.memory = memory
        self.cross_dim_attn = cross_dim_attn
        self.hardening = hardening
        # 消融开关(2026-08-17 研究项目): none | cdap | memory | router
        self.ablate = ablate

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

        Notes
        -----
        AHM 决策(2026-08-09, Kimi K3 评审): 默认禁用且不推荐启用。
        理由: (1) 推理优化不应先于信号验证 —— 未验证的策略没有"加速"
        的需求; (2) fast path 绕过 CDAP, 与 CDAP 的调制增益自相矛盾;
        (3) 缓存的路由快照在市场漂移后过时。保留实现供研究参考,
        但所有训练/评估路径均不启用。

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

        # 消融: router → 均匀路由(CDAP 与 memory 仍跑, 只去掉"学到的"路由)
        if self.ablate == "router":
            full_probs = torch.full_like(full_probs, 1.0 / self.n_experts)

        # === Step 2: Expert forward passes ===
        expert_outputs = []
        expert_hiddens = []
        for i, expert in enumerate(self.experts):
            signal_i, hidden_i = expert(s_t, return_hidden=True)
            expert_outputs.append(signal_i.squeeze(-1))    # (B,)
            expert_hiddens.append(hidden_i)                 # (B, hidden_dim)

        expert_outputs = torch.stack(expert_outputs, dim=-1)   # (B, n_experts)
        # Note: expert_hiddens have different dims (heterogeneous experts),
        # so we don't stack them. They're available per-expert if needed.

        # === Step 3: Cross-Dimension Attention Protocol (BEFORE memory) ===
        # CDAP runs before memory so the memory_gate can be passed directly
        # to memory.forward(), keeping the gradient path intact:
        #   loss → memory(α·gate) → gate → CDAP → memory_gate_scale
        # This replaces the old side-channel approach where the gate was
        # stored for the NEXT step and detach_state() killed the gradient.
        # Initialize M placeholder for first call (CDAP needs it before
        # memory.forward() has run for the first time).
        if self.memory.M is None or self.memory.M.size(0) != B:
            self.memory.reset_state(B, device)

        if self.ablate == "cdap":
            # 消融: 跳过 CDAP 三向调制, 路由=原始 softmax, 深度=均匀平均
            final_routing = full_probs
            memory_gate = None
            depth_weights = F.softmax(torch.ones(B, 3, device=device), dim=-1)
            fused_layers = sum(
                depth_weights[:, k:k+1] * layer_outputs[k] for k in range(3)
            )
        elif mode != "inference" or not use_hardening:
            # Full CDAP modulation
            memory_matrix = self.memory.M.clone()

            routing_mod, memory_gate, depth_weights, fused_layers = \
                self.cross_dim_attn(
                    routing_probs=full_probs,
                    memory_matrix=memory_matrix,
                    layer_outputs=layer_outputs,
                )
            final_routing = routing_mod
        else:
            # Try hardened fast path (per-sample independent decisions)
            regime_ids = self.router.get_regime_id(z_t)  # (B,) long

            fast_decisions = []
            for b in range(B):
                fast_decisions.append(
                    self.hardening.should_use_fast_path(
                        regime_ids[b].item(), full_probs[b]
                    )
                )
            use_fast = all(fast_decisions)

            if use_fast:
                cached_weights_list = []
                for b in range(B):
                    cw = self.hardening.get_cached_weights(
                        regime_ids[b].item(), full_probs[b]
                    )
                    cached_weights_list.append(cw)
                final_routing = torch.stack(cached_weights_list, dim=0)  # (B, n_experts)
                depth_weights = F.softmax(
                    torch.ones(B, 3, device=device), dim=-1
                )
                fused_layers = sum(
                    depth_weights[:, k:k+1] * layer_outputs[k] for k in range(3)
                )
                # Fast path: no CDAP → no memory_gate modulation
                memory_gate = None
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

        # === Step 4: Update and query memory (with CDAP gate, same graph) ===
        # memory_gate is passed directly — gradients flow from α modulation
        # back through the gate to CDAP's memory_gate_scale parameter.
        # The retrieved memory vector contributes to the signal so that
        # the memory pathway receives meaningful gradients.
        if self.ablate == "memory":
            # 消融: 跳过 KDA 记忆, 检索输出置零(记忆路径不参与信号)
            retrieved = torch.zeros(B, self.memory.d_v, device=device)
        else:
            retrieved, _ = self.memory(s_t, z_t=z_t, cdap_gate=memory_gate)

        # === Step 5: Weighted expert fusion ===
        # signal = Σ_i w_i · expert_i(s_t)
        signal = (final_routing * expert_outputs).sum(dim=-1, keepdim=True)  # (B, 1)

        # === Step 6: Depth + Memory fusion ===
        # Both fused_layers (CDAP depth output) and retrieved (KDA memory output)
        # contribute to the signal, ensuring both CDAP and memory pathways
        # receive gradient signal through the loss.
        signal = signal + 0.1 * (fused_layers + retrieved).mean(dim=-1, keepdim=True)

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
