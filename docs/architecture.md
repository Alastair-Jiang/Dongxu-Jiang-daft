# DAFT: Dimension-Aware Financial Trading
#
# Architecture Specification
# =========================
#
# This document provides the detailed architectural specification for DAFT,
# including component interfaces, data flow diagrams, and design rationale.
#
# See README.md for the high-level overview and Quick Start guide.
#
# ---
#
# ## Table of Contents
#
# 1. Data Flow
# 2. Component Interfaces
# 3. Training Protocol
# 4. Evaluation Protocol
# 5. Extension Points
#
# ---
#
# ## 1. Data Flow
#
# ```
# [Raw Market Data] → Panel(T×N×F) → Feature Engine → s_t(200) ─┐
#                                                               │
#     ┌──────────────────────────────────────────────────────────┘
#     │
#     ├─→ RegimeRouter → routing_probs, z_t ──┐
#     ├─→ KDAMemory → retrieved, M_t ─────────┤
#     └─→ FactorLayers → [h0, h1, h2] ────────┤
#                                              │
#                    ┌─────────────────────────┘
#                    ▼
#            CrossDimensionAttention
#                    │
#        ┌───────────┼───────────┐
#        ▼           ▼           ▼
#   routing_mod  memory_gate  depth_weights
#        │           │           │
#        └───────────┼───────────┘
#                    ▼
#            ExpertEnsemble → signal(B,1)
#                    │
#                    ▼
#            PortfolioOptimizer → weights(N,)
#                    │
#                    ▼
#            BacktestEngine → metrics
# ```
#
# ## 2. Component Interfaces
#
# (Detailed API docs — see docstrings in each module.)
#
# ## 3. Training Protocol
#
# (See README Training Pipeline section.)
#
# ## 4. Evaluation Protocol
#
# ### Primary Metrics
# - Sharpe Ratio (annualized)
# - Maximum Drawdown
# - Calmar Ratio
# - Rank Information Coefficient (IC)
# - IC Information Ratio (ICIR)
#
# ### Ablation Metrics
# - Component ablation: disable each CDAP connection, measure ΔSharpe
# - Hardening ablation: fast vs slow path accuracy delta
# - Inference speedup: hardened / full path latency ratio
#
# ## 5. Extension Points
#
# - **New experts**: subclass BaseExpert, implement _regime_filter and compute_loss
# - **New data sources**: implement DataLoader backend in data/loaders.py
# - **Custom routing**: override RegimeRouter with alternative gating strategies
# - **Alternative memory**: swap KDAMarketMemory with e.g., Mamba-style SSM
