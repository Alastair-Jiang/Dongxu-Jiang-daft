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


def build_experts() -> nn.ModuleList:
    """10 专家池: 5 类 × 2 实例(顺序与 conftest 一致)。"""
    return nn.ModuleList([
        TrendExpert(input_dim=INPUT_DIM, hidden_dim=64),
        TrendExpert(input_dim=INPUT_DIM, hidden_dim=64),
        ReversalExpert(input_dim=INPUT_DIM, hidden_dim=64),
        ReversalExpert(input_dim=INPUT_DIM, hidden_dim=64),
        VolatilityExpert(input_dim=INPUT_DIM, hidden_dim=48),
        VolatilityExpert(input_dim=INPUT_DIM, hidden_dim=48),
        EventExpert(input_dim=INPUT_DIM, hidden_dim=48),
        EventExpert(input_dim=INPUT_DIM, hidden_dim=48),
        MomentumExpert(input_dim=INPUT_DIM, hidden_dim=64),
        MomentumExpert(input_dim=INPUT_DIM, hidden_dim=64),
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
) -> ExpertEnsemble:
    """组装标准 DAFT 模型(含 n_experts 一致性守卫)。"""
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
    return ExpertEnsemble(experts, router, memory, cdap, hardening)


def build_model(
    cdap_strength: float = 0.1,
    router_temperature: float = 1.0,
    noisy_gating_std: float = 0.1,
):
    """一键: experts + ensemble + layer_proj。"""
    experts = build_experts()
    model = build_ensemble(
        experts,
        cdap_strength=cdap_strength,
        router_temperature=router_temperature,
        noisy_gating_std=noisy_gating_std,
    )
    return model, build_layer_proj()
