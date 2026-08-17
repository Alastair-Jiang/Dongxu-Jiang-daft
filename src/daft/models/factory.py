"""共享模型工厂 (2026-08-16 新增 — 代码整理)。

历史上 build_experts / build_ensemble / build_layer_proj 在 7 个脚本里
各复制了一份, n_experts 8 vs 10 的崩溃正是这样漂出来的。本模块提供
单一权威入口, 所有脚本必须从这里构建模型。

注意: 此处仅承载"标准 DAFT 架构"(10 专家 / d_k=128 / d_v=64 /
latent 16 / joint 64), 与 tests/conftest.py 的常量保持一致。
"""

from __future__ import annotations

import torch.nn as nn

from daft.models.cross_dim_attn import CrossDimensionAttention
from daft.models.ensemble import ExpertEnsemble
from daft.models.experts import (
    EventExpert, MomentumExpert, ReversalExpert, TrendExpert, VolatilityExpert,
    TransformerExpert,
)
from daft.models.hardening import HardeningEngine
from daft.models.memory import KDAMarketMemory
from daft.models.router import RegimeRouter

# ── 标准架构常量(与 tests/conftest.py 一致) ─────────────────────────────
INPUT_DIM = 200
LATENT_DIM = 16
D_K = 128
D_V = 64
N_LAYERS = 3
JOINT_DIM = 64
N_EXPERTS = 10
TOP_K = 3


def build_experts(hidden: int = 64, n_layers: int = 2,
                  arch: str = "mlp", n_heads: int = 4) -> nn.ModuleList:
    """10 专家池。

    arch="mlp" → 5 类 regime 专家 × 2 实例(顺序与 conftest 一致)。
    arch="transformer" → 10 个 TransformerExpert(特征自注意力, 2026-08-17
    研究项目: 独立初始化、全量训练、路由器分工)。

    hidden / n_layers: 容量扫描参数(2026-08-17 研究项目)。
    volatility/event 用 0.75× hidden(历史口径 64/48)。
    """
    if arch == "transformer":
        return nn.ModuleList([
            TransformerExpert(input_dim=INPUT_DIM, hidden_dim=hidden,
                              n_layers=n_layers, n_heads=n_heads)
            for _ in range(N_EXPERTS)
        ])
    h_small = max(32, int(hidden * 0.75))
    return nn.ModuleList([
        TrendExpert(input_dim=INPUT_DIM, hidden_dim=hidden, n_layers=n_layers),
        TrendExpert(input_dim=INPUT_DIM, hidden_dim=hidden, n_layers=n_layers),
        ReversalExpert(input_dim=INPUT_DIM, hidden_dim=hidden, n_layers=n_layers),
        ReversalExpert(input_dim=INPUT_DIM, hidden_dim=hidden, n_layers=n_layers),
        VolatilityExpert(input_dim=INPUT_DIM, hidden_dim=h_small, n_layers=n_layers),
        VolatilityExpert(input_dim=INPUT_DIM, hidden_dim=h_small, n_layers=n_layers),
        EventExpert(input_dim=INPUT_DIM, hidden_dim=h_small, n_layers=n_layers),
        EventExpert(input_dim=INPUT_DIM, hidden_dim=h_small, n_layers=n_layers),
        MomentumExpert(input_dim=INPUT_DIM, hidden_dim=hidden, n_layers=n_layers),
        MomentumExpert(input_dim=INPUT_DIM, hidden_dim=hidden, n_layers=n_layers),
    ])


def build_layer_proj(input_dim: int = INPUT_DIM, d_v: int = D_V) -> nn.ModuleDict:
    """3 层特征层级投影 (L0/L1/L2)。"""
    def _block():
        return nn.Sequential(
            nn.Linear(input_dim, 128), nn.SiLU(),
            nn.Linear(128, d_v), nn.LayerNorm(d_v),
        )
    return nn.ModuleDict({"l0": _block(), "l1": _block(), "l2": _block()})


def build_ensemble(
    experts: nn.ModuleList,
    cdap_strength: float = 0.1,
    router_temperature: float = 1.0,
    noisy_gating_std: float = 0.1,
    ablate: str = "none",
) -> ExpertEnsemble:
    """组装标准 DAFT 模型(含 n_experts 一致性守卫)。

    ablate: none | cdap | memory | router — 消融开关(2026-08-17 研究项目)。
    """
    router = RegimeRouter(
        input_dim=INPUT_DIM, latent_dim=LATENT_DIM, n_experts=N_EXPERTS,
        top_k=TOP_K, temperature=router_temperature,
        noisy_gating_std=noisy_gating_std,
    )
    memory = KDAMarketMemory(
        d_k=D_K, d_v=D_V, d_feature=INPUT_DIM,
        bottleneck_ratio=4, use_route_modulation=True,
    )
    cdap = CrossDimensionAttention(
        n_experts=N_EXPERTS, d_k=D_K, d_v=D_V, n_layers=N_LAYERS,
        joint_dim=JOINT_DIM, modulation_strength=cdap_strength,
    )
    hardening = HardeningEngine(n_regimes=N_EXPERTS, n_experts=N_EXPERTS)
    assert len(experts) == router.n_experts == cdap.n_experts, (
        f"n_experts 不一致: experts={len(experts)}, "
        f"router={router.n_experts}, cdap={cdap.n_experts}"
    )
    return ExpertEnsemble(experts, router, memory, cdap, hardening, ablate=ablate)


def build_model(
    cdap_strength: float = 0.1,
    router_temperature: float = 1.0,
    noisy_gating_std: float = 0.1,
    ablate: str = "none",
    hidden: int = 64,
    n_layers: int = 2,
    arch: str = "mlp",
    n_heads: int = 4,
):
    """一键: experts + ensemble + layer_proj。"""
    experts = build_experts(hidden=hidden, n_layers=n_layers,
                            arch=arch, n_heads=n_heads)
    model = build_ensemble(
        experts,
        cdap_strength=cdap_strength,
        router_temperature=router_temperature,
        noisy_gating_std=noisy_gating_std,
        ablate=ablate,
    )
    return model, build_layer_proj()
