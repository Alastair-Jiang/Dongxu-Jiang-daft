"""Shared fixtures and configuration for the DAFT test suite.

This file consolidates constants and fixtures that are used across
multiple test modules, reducing duplication and ensuring consistency.
"""

import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

# ── 源码守卫: 确保测试导入的是本仓库的 daft, 而不是 site-packages 里
# editable 安装指向的其他克隆(历史事故: C:\Users\26843\daft 的旧版
# 被导入, 导致测试在错误的代码上运行)。
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from daft.models.router import RegimeRouter
from daft.models.memory import KDAMarketMemory
from daft.models.cross_dim_attn import CrossDimensionAttention
from daft.models.hardening import HardeningEngine
from daft.models.ensemble import ExpertEnsemble
from daft.models.experts import (
    TrendExpert, ReversalExpert, VolatilityExpert, EventExpert, MomentumExpert,
)

# ── 导入后校验: daft 必须来自本仓库的 src/ ──────────────────────────────
import daft as _daft_pkg  # noqa: E402

_daft_src = Path(_daft_pkg.__file__).resolve()
if not _daft_src.is_relative_to(PROJECT_ROOT):
    raise RuntimeError(
        f"daft 导入自错误的路径: {_daft_src}\n"
        f"预期在本仓库内: {PROJECT_ROOT / 'src'}\n"
        "检查 editable 安装(pip show daft)是否指向了其他克隆, "
        "或在 pyproject 里配置 pytest pythonpath=['src']。"
    )


# ── Shared constants ────────────────────────────────────────────────────
# These values match the DAFT architecture defaults and are used
# consistently across all test files.

INPUT_DIM = 200       # Market state vector dimension
D_K = 128             # Memory key dimension (number of slots)
D_V = 64              # Memory value dimension
N_EXPERTS = 10        # Total expert pool size (5 types × 2 instances)
TOP_K = 3             # Experts activated per forward pass
N_LAYERS = 3          # Feature hierarchy depth
JOINT_DIM = 64        # CDAP joint latent space dimension
LATENT_DIM = 16       # Router latent regime space dimension
N_REGIMES = 10        # Regime count (= n_experts, one per expert instance)
BATCH_SIZES = [1, 4, 32]  # Standard batch sizes for shape tests


# ── Component fixtures ──────────────────────────────────────────────────
# Each fixture returns a freshly-initialized component with sensible
# defaults, usable directly in tests without duplicating init code.


@pytest.fixture
def router():
    """RegimeRouter with standard architecture."""
    return RegimeRouter(
        input_dim=INPUT_DIM, latent_dim=LATENT_DIM, n_experts=N_EXPERTS,
        top_k=TOP_K, temperature=1.0, noisy_gating_std=0.1,
    )


@pytest.fixture
def memory():
    """KDAMarketMemory with standard architecture."""
    return KDAMarketMemory(
        d_k=D_K, d_v=D_V, d_feature=INPUT_DIM,
        bottleneck_ratio=4, use_route_modulation=True,
    )


@pytest.fixture
def cdap():
    """CrossDimensionAttention with standard architecture."""
    return CrossDimensionAttention(
        n_experts=N_EXPERTS, d_k=D_K, d_v=D_V,
        n_layers=N_LAYERS, joint_dim=JOINT_DIM,
        modulation_strength=1.0,
    )


@pytest.fixture
def hardening():
    """HardeningEngine with test-friendly low threshold."""
    return HardeningEngine(
        n_regimes=N_REGIMES, n_experts=N_EXPERTS,
        threshold=20, min_confidence=0.1, entropy_multiplier=2.0,
    )


# ── Ensemble fixture ────────────────────────────────────────────────────


def _make_expert_pool(n_experts: int = 10) -> nn.ModuleList:
    """Create a balanced pool of heterogeneous strategy experts.

    Order: 2×Trend → 2×Reversal → 2×Volatility → 2×Event → 2×Momentum.
    """
    expert_types = [
        (TrendExpert, 64), (TrendExpert, 64),
        (ReversalExpert, 64), (ReversalExpert, 64),
        (VolatilityExpert, 48), (VolatilityExpert, 48),
        (EventExpert, 48), (EventExpert, 48),
        (MomentumExpert, 64), (MomentumExpert, 64),
    ]
    experts = []
    for i in range(min(n_experts, len(expert_types))):
        cls, hidden_dim = expert_types[i]
        experts.append(cls(input_dim=INPUT_DIM, hidden_dim=hidden_dim))
    return nn.ModuleList(experts)


@pytest.fixture
def ensemble():
    """Full ExpertEnsemble with all DAFT components (10 experts)."""
    return ExpertEnsemble(
        experts=_make_expert_pool(10),
        router=RegimeRouter(input_dim=INPUT_DIM, latent_dim=LATENT_DIM,
                            n_experts=N_EXPERTS, top_k=TOP_K),
        memory=KDAMarketMemory(d_k=D_K, d_v=D_V, d_feature=INPUT_DIM,
                               use_route_modulation=True),
        cross_dim_attn=CrossDimensionAttention(
            n_experts=N_EXPERTS, d_k=D_K, d_v=D_V,
            n_layers=N_LAYERS, joint_dim=JOINT_DIM,
        ),
        hardening=HardeningEngine(n_regimes=N_REGIMES, n_experts=N_EXPERTS,
                                  threshold=100),
    )


@pytest.fixture
def ensemble_low_threshold():
    """Ensemble with low hardening threshold for fast-path testing."""
    return ExpertEnsemble(
        experts=_make_expert_pool(10),
        router=RegimeRouter(input_dim=INPUT_DIM, latent_dim=LATENT_DIM,
                            n_experts=N_EXPERTS, top_k=TOP_K),
        memory=KDAMarketMemory(d_k=D_K, d_v=D_V, d_feature=INPUT_DIM,
                               use_route_modulation=True),
        cross_dim_attn=CrossDimensionAttention(
            n_experts=N_EXPERTS, d_k=D_K, d_v=D_V,
            n_layers=N_LAYERS, joint_dim=JOINT_DIM,
        ),
        hardening=HardeningEngine(n_regimes=N_REGIMES, n_experts=N_EXPERTS,
                                  threshold=3, min_confidence=0.01),
    )
