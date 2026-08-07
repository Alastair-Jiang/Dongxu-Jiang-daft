"""
DAFT: Dimension-Aware Financial Trading.

A cross-dimensional attention architecture for medium-frequency quantitative
trading, inspired by Kimi K3's Stable LatentMoE, KDA, and AttnRes.
"""

__version__ = "0.1.0"
__author__ = "Alastair(Dongxu-Jiang)"
__license__ = "MIT"

from daft.models.router import RegimeRouter
from daft.models.memory import KDAMarketMemory
from daft.models.cross_dim_attn import CrossDimensionAttention
from daft.models.hardening import HardeningEngine
from daft.models.ensemble import ExpertEnsemble

__all__ = [
    "RegimeRouter",
    "KDAMarketMemory",
    "CrossDimensionAttention",
    "HardeningEngine",
    "ExpertEnsemble",
]
