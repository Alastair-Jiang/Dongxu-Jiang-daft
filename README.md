# DAFT: Dimension-Aware Financial Trading

<div align="center">

**[English](README.md) · [中文](#)**

*A cross-dimensional attention architecture for medium-frequency quantitative trading, inspired by **Kimi K3**'s Stable LatentMoE, KDA, and AttnRes.*

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Status: Alpha](https://img.shields.io/badge/Status-Alpha-orange.svg)]()

</div>

---

## Abstract

Most ML-for-trading systems treat model components — routing, memory, and feature hierarchy — as independent modules with unidirectional data flow. We argue that **these components should bidirectionally modulate one another** to form a coherent decision-making engine.

**DAFT** introduces two architectural innovations:

1. **Cross-Dimension Attention Protocol (CDAP)** — A unified modulation framework where MoE routing decisions influence memory retention policies, memory state informs cross-layer retrieval weights, and retrieval results feed back into routing bias correction. The three dimensions (expert space, temporal memory, feature depth) interact through a shared latent joint-space.

2. **Adaptive Hardening Mechanism (AHM)** — After training, frequently traversed $(regime, expert\_pattern, memory\_state, depth\_weights)$ tuples are cached as fast-path lookups. Routine market conditions follow $\mathcal{O}(1)$ hardened paths, while anomalous conditions automatically degrade to full exploration. This generalizes Kimi K3's static 3:1 KDA-to-full-attention ratio into a **data-driven, regime-adaptive routing policy**.

The architecture is systematically derived from **Kimi K3** (Moonshot AI, July 2026), currently the world's largest open-weight model (2.8T parameters, 896 experts, 16 activated), whose design principles — Stable LatentMoE, Kimi Delta Attention (KDA), and Attention Residuals (AttnRes) — are mapped to the financial time-series domain and extended with bidirectional cross-component modulation.

---

## Table of Contents

- [Why This Exists](#why-this-exists)
- [Architecture](#architecture)
  - [High-Level Design](#high-level-design)
  - [Component 1: Regime Router](#component-1-regime-router)
  - [Component 2: KDA Market Memory](#component-2-kda-market-memory)
  - [Component 3: Cross-Dimension Attention Protocol](#component-3-cross-dimension-attention-protocol)
  - [Component 4: Adaptive Hardening Mechanism](#component-4-adaptive-hardening-mechanism)
- [Design Derivation: Kimi K3 → Finance](#design-derivation-kimi-k3--finance)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Training Pipeline](#training-pipeline)
- [Experiments](#experiments)
  - [Benchmarks](#benchmarks)
  - [Ablation Studies](#ablation-studies)
  - [Hardening Analysis](#hardening-analysis)
- [Performance](#performance)
  - [Forecast Accuracy](#forecast-accuracy)
  - [Inference Efficiency](#inference-efficiency)
- [Limitations](#limitations)
- [Citation](#citation)
- [License](#license)
- [Acknowledgements](#acknowledgements)

---

## Why This Exists

### The Gap

| Existing Approach | What It Misses |
|---|---|
| **Time-MoE** (ICLR 2025), **Super-Linear** (2025) | MoE routing treats memory as passive state; no cross-layer retrieval |
| **Dynamic TMoE** (ICML 2026) | Expert pool adapts to distribution drift, but expert selection never informs memory strategy |
| **KDA** (Kimi, 2025) | Per-channel forget gates are input-driven only — no routing signal, no depth signal |
| **AttnRes** (K3, 2026) | Cross-layer attention weights ignore routing distribution and memory state |
| **PatchTST, TimesNet, iTransformer** | Single-model, single-regime — no expert specialization |

**None of these systems allow the routing decision to change what is remembered, or the memory state to change which feature layers are trusted.**

### The Intuition

In financial markets, regime identification (routing), historical pattern matching (memory), and multi-scale feature extraction (depth) are **not independent problems**. They are three facets of the same problem. When the router identifies a trending regime:

- The memory should **retain trend-related patterns and forget mean-reversion noise** (routing → memory modulation)
- Deep abstract features (regime labels) should be **trusted more than raw prices** (memory → depth modulation)
- If the trend suddenly breaks, the depth signal should **override the router's prior** (depth → routing feedback)

DAFT formalizes these intuitions as a trainable, three-way modulation protocol.

### Target Audience

- **Researchers** exploring architectural co-design for structured time-series domains
- **Quantitative finance practitioners** building regime-adaptive trading systems
- **Students** studying the intersection of modern LLM architectures and financial ML

---

## Architecture

### High-Level Design

```
═══════════════════════════════════════════════════════════════════════
                          DAFT System Architecture
═══════════════════════════════════════════════════════════════════════

                            ┌──────────────────┐
                            │   Market Data     │
                            │   (OHLCV, min)    │
                            └────────┬──────────┘
                                     │
                            ┌────────▼──────────┐
                            │  Feature Engine    │
                            │  • 213 base factors│
                            │  • Regime features │
                            │  • FFT spectral    │
                            │  • s_t ∈ R^200     │
                            └────────┬──────────┘
                                     │
              ┌──────────────────────┼──────────────────────┐
              │                      │                      │
              ▼                      ▼                      ▼
    ┌─────────────────┐   ┌─────────────────┐   ┌─────────────────┐
    │  L0: Raw Data   │   │  L1: Base Factors│   │  L2: Composite  │
    │  (Price/Volume)  │   │  (MA/Vol/RSI)   │   │  (Regime/Risk)  │
    └────────┬────────┘   └────────┬────────┘   └────────┬────────┘
             │                     │                     │
             └─────────────────────┼─────────────────────┘
                                   │
                    ┌──────────────┼──────────────┐
                    │              │              │
                    ▼              ▼              ▼
    ┌───────────────────────────────────────────────────────┐
    │              CROSS-DIMENSION ATTENTION                 │
    │                                                       │
    │   ┌─────────┐      ┌──────────┐      ┌────────────┐  │
    │   │ Router  │◄────►│  Memory  │◄────►│   Depth    │  │
    │   │(Regime) │      │  (KDA)   │      │ (AttnRes)  │  │
    │   └────┬────┘      └────┬─────┘      └─────┬──────┘  │
    │        │                │                   │         │
    │        └────────────────┼───────────────────┘         │
    │                         │                             │
    │               Joint Latent Space                      │
    │               (mutual modulation)                     │
    └─────────────────────────┬─────────────────────────────┘
                              │
                    ┌─────────▼─────────┐
                    │  Hardening Engine  │
                    │  Fast Path / Slow  │
                    │  Path Router       │
                    └─────────┬─────────┘
                              │
                    ┌─────────▼─────────┐
                    │  Expert Ensemble  │
                    │  Weighted Signal  │
                    └─────────┬─────────┘
                              │
                    ┌─────────▼─────────┐
                    │  Portfolio Optim  │
                    │  (Markowitz)      │
                    └─────────┬─────────┘
                              │
                    ┌─────────▼─────────┐
                    │  Backtest Engine  │
                    │  (Vectorized)     │
                    └───────────────────┘
```

### Component 1: Regime Router

**Inspiration**: Kimi K3 Stable LatentMoE (896 experts, 16 active, latent-space routing)

**Mathematical Formulation**:

The raw market state vector $\mathbf{s}_t \in \mathbb{R}^{200}$ is projected into a low-dimensional latent regime space:

$$\mathbf{z}_t = \text{LayerNorm}\big(W_{\text{down}} \cdot \text{SiLU}(W_{\text{up}} \cdot \mathbf{s}_t)\big) \in \mathbb{R}^{16}$$

Expert routing with temperature-scaled softmax:

$$p(\text{expert}_i \mid \mathbf{z}_t) = \frac{\exp\big((W_i^{\text{route}} \cdot \mathbf{z}_t + b_i) / \tau\big)}{\sum_{j=1}^{N} \exp\big((W_j^{\text{route}} \cdot \mathbf{z}_t + b_j) / \tau\big)}$$

$$\text{activated} = \text{TopK}(p, k = 3)$$

**Load Balancing** (Quantile Balancing, derived from K3):

$$b_i \leftarrow b_i + \eta \cdot \left(\frac{1}{N} - \frac{\text{count}_i}{\sum_j \text{count}_j}\right)$$

Zero auxiliary loss. Bias adjustment based on activation frequency quantiles.

| Parameter | Value | Description |
|-----------|-------|-------------|
| `input_dim` | 200 | Market state vector dimension |
| `latent_dim` | 16 | Regime latent space dimension |
| `n_experts` | 8 | Number of strategy experts |
| `top_k` | 3 | Experts activated per forward pass |
| `τ_train` | 1.0 | Temperature during training (soft routing) |
| `τ_hardened` | 0.1 | Temperature after hardening (near-discrete) |

---

### Component 2: KDA Market Memory

**Inspiration**: Kimi Delta Attention — per-channel forget gates + delta-rule state updates

**Mathematical Formulation**:

The memory state $M_t \in \mathbb{R}^{d_k \times d_v}$ ($d_k = 128$, $d_v = 64$) is maintained as a fixed-size recurrent matrix, independent of sequence length.

**Per-Channel Forget Gate** (low-rank bottleneck, adapted from KDA FineGrainedGating):

$$\boldsymbol{\alpha}_t = \sigma\big(W_{\text{up}} \cdot \text{SiLU}(W_{\text{down}} \cdot \mathbf{s}_t)\big) \in (0, 1)^{d_k}$$

**Route-Modulated Forgetting** (CDAP connection: Router → Memory):

$$\boldsymbol{\alpha}'_t = \boldsymbol{\alpha}_t \odot \sigma(W_{\text{route}} \cdot \mathbf{z}_t)$$

where $\mathbf{z}_t$ is the routing latent vector from Component 1.

**Delta-Rule State Update** (derived from KDA's online learning formulation):

$$M_t = M_{t-1} - \beta_t \cdot k_t \otimes (M_{t-1} \cdot k_t) + \beta_t \cdot k_t \otimes v_t$$

where:
- $k_t = \text{L2Norm}(W_k \cdot \mathbf{s}_t)$ — L2-normalized key (critical for stability)
- $v_t = W_v \cdot \mathbf{s}_t$ — Value vector
- $\beta_t = \sigma(W_\beta \cdot \mathbf{s}_t)$ — Learned per-step learning rate

**Memory Retrieval**:

$$o_t = M_t^\top \cdot q_t, \quad q_t = W_q \cdot \mathbf{s}_t$$

$$\text{retrieved}_t = \text{RMSNorm}(o_t)$$

**Complexity**: $\mathcal{O}(d_k \cdot d_v)$ per step, independent of sequence length. No KV-cache. Total state footprint: $128 \times 64 = 8192$ floats ≈ 32 KB.

| Parameter | Value | Description |
|-----------|-------|-------------|
| `d_k` | 128 | Key dimension (= number of memory slots) |
| `d_v` | 64 | Value dimension (stored information per slot) |
| `bottleneck_ratio` | 4 | Forget gate low-rank compression ratio |
| `use_route_modulation` | True | Enable Router → Memory CDAP connection |

---

### Component 3: Cross-Dimension Attention Protocol

**The core methodological contribution of DAFT.**

Three information streams — routing distribution $\mathbf{p}_t$, memory state $M_t$, and layer-wise features $\{h_0, h_1, h_2\}$ — are projected into a shared **joint latent space**, where they modulate one another before being projected back as corrected signals.

**Joint Space Projection**:

$$\mathbf{e} = f_{\text{expert} \to \text{joint}}(\mathbf{p}_t) \in \mathbb{R}^{64}$$
$$\mathbf{m} = f_{\text{memory} \to \text{joint}}(\text{flatten}(M_t)) \in \mathbb{R}^{64}$$
$$\mathbf{d} = f_{\text{depth} \to \text{joint}}(\text{concat}(h_0, h_1, h_2)) \in \mathbb{R}^{64}$$

**Fusion via Mutual Modulation**:

$$\mathbf{j} = \mathbf{e} \odot \mathbf{m} \odot \mathbf{d} \in \mathbb{R}^{64}$$

The element-wise product ensures that **any dimension with near-zero activation silences the cross-modulation signal** — a strong inductive bias for sparse, regime-specific computation.

**Reverse Projections** (Joint → Components):

| Direction | Formula | Effect |
|-----------|---------|--------|
| → Router | $\mathbf{p}'_t = \text{softmax}(\log \mathbf{p}_t + \delta \cdot W_{\text{out}}^{\text{router}} \cdot \mathbf{j})$ | Memory + Depth bias routing |
| → Memory | $\mathbf{g}_t = \sigma(W_{\text{out}}^{\text{memory}} \cdot \mathbf{j}) \in (0,1)^{d_k}$ | Additional forget-gate modulation |
| → Depth | $\mathbf{w}_t = \text{softmax}(W_{\text{out}}^{\text{depth}} \cdot \mathbf{j}) \in \Delta^2$ | Cross-layer retrieval weights |

**Fused Output**:

$$h_t^{\text{fused}} = \sum_{k=0}^{2} w_t^{(k)} \cdot h_k$$

**Design Rationale**: The element-wise product in joint space is deliberate — not additive fusion. Addition assumes orthogonal contributions, but routing, memory, and depth are inherently coupled. Multiplication ensures that when the memory is uncertain (near-zero activations), it cannot distort the routing signal, and vice versa.

---

### Component 4: Adaptive Hardening Mechanism

**Inspiration**: K3's 3:1 KDA-to-full-attention static ratio → generalized as data-driven dynamic routing

**Core Idea**: After training, frequently traversed $(regime, expert\_pattern)$ tuples are cached. Routine regimes follow $\mathcal{O}(1)$ fast paths; novel regimes fall back to full computation.

**Hardening Criterion**:

A pattern $(r, \mathcal{E})$ is eligible for hardening when:

$$\text{count}(r, \mathcal{E}) \geq \theta_{\text{harden}} \quad \text{and} \quad \text{confidence}(r, \mathcal{E}) > \rho_{\text{min}}$$

where $\theta_{\text{harden}} = 100$ and $\rho_{\text{min}} = 0.95$.

**Regime Shift Detection**:

$$\text{if } H(\mathbf{p}_{\text{recent}}) > \lambda \cdot H_{\text{baseline}} \quad \text{→ degrade to full exploration}$$

where $H(\cdot)$ is the entropy of the routing distribution, and $\lambda = 2.0$.

**Hardening Statistic**:

$$C_{\text{hardened}}(r, \mathcal{E}) = \big\{ \mathbf{w}^*_{\text{expert}}, \mathbf{g}^*_{\text{memory}}, \mathbf{w}^*_{\text{depth}} \big\}$$

Three cached vectors for the hardened $(regime, expert\_pattern)$ combination.

| Parameter | Value | Description |
|-----------|-------|-------------|
| `θ_harden` | 100 | Minimum observations before hardening |
| `ρ_min` | 0.95 | Minimum confidence for cache validity |
| `λ_entropy` | 2.0 | Entropy multiplier threshold for degradation |
| `n_regimes_tracked` | 8 | Maximum discrete regime clusters |

---

## Design Derivation: Kimi K3 → Finance

| K3 Component | K3 Implementation | DAFT Mapping | Key Modification |
|---|---|---|---|
| **Stable LatentMoE** | 896 experts, 16 active, latent-space routing with Quantile Balancing | 8 strategy experts, top-3 active, regime latent space ($\mathbb{R}^{16}$) | Financial expert semantics: trend, reversal, volatility, event-driven |
| **KDA (Kimi Delta Attention)** | Per-channel forget gates ($\boldsymbol{\alpha}_t$), delta-rule state update ($S_t$), 3:1 KDA-to-MLA layer ratio | Per-slot forget gates, route-modulated forgetting ($\boldsymbol{\alpha}'_t$), fixed-size market memory ($M_t \in \mathbb{R}^{128 \times 64}$) | Router signal modulates forget gate; NoPE by design (market time is non-uniform) |
| **AttnRes (Attention Residuals)** | Cross-layer attention over hidden states $[h_0, \ldots, h_{l-1}]$ | Cross-layer retrieval over factor hierarchy (L0 raw → L1 base → L2 composite) | Depth weights are memory-state-aware (CDAP connection) |
| **SiTU Activation** | $\sigma(x) \odot \tanh(x)$, natural output bound $[-1,1]$ | Expert output activation for natural weight alignment | Ensures expert signals are magnitude-comparable before gated fusion |
| **3:1 Hybrid Ratio** | Static: 3 KDA layers per 1 MLA layer | **Dynamic: AHM-learned fast/slow path ratio** | **Our extension**: ratio adapts to market regime |

---

## Quick Start

```bash
# Clone
git clone https://github.com/Dongxu-Jiang/daft.git
cd daft

# Install (CPU)
pip install -e ".[dev]"

# Install (GPU / Apple M-series)
pip install -e ".[gpu]"    # CUDA
pip install -e ".[mps]"    # Apple Silicon

# Smoke test: synthetic 200 stocks × 500 days, ~30 seconds
make paper CONFIG=configs/small.yaml

# Full experiment
make paper CONFIG=configs/paper.yaml

# Hardening analysis
make paper CONFIG=configs/hardening.yaml
```

---

## Installation

### Requirements

| Dependency | Version | Purpose |
|---|---|---|
| Python | ≥ 3.10 | Runtime |
| PyTorch | ≥ 2.0 | Core ML framework |
| NumPy | ≥ 1.24 | Numerical ops |
| Pandas | ≥ 2.0 | Data manipulation |
| Polars | ≥ 0.20 | High-performance data loading |
| einops | ≥ 0.7 | Tensor operations |
| PyYAML | ≥ 6.0 | Configuration |
| tqdm | ≥ 4.65 | Progress bars |
| Matplotlib | ≥ 3.7 | Visualization |
| pytest | ≥ 7.0 | Testing |
| CVXPY | ≥ 1.4 | Portfolio optimization |
| MOSEK | ≥ 10.0 (optional) | Fast QP solver |

### Hardware Guidance

| Hardware | Capability |
|---|---|
| Mac Mini M4 (16 GB) | Full training on ≤ 200 stocks, smoke tests, experimentation |
| NVIDIA GPU (≥ 8 GB VRAM) | Full training on ≥ 500 stocks, hyperparameter sweeps |
| CPU-only (16 GB) | Inference, smoke tests, small-scale training |

The model has **< 200K total parameters** (deliberately lightweight for research iteration).

---

## Project Structure

```
daft/
├── README.md                         # This document
├── LICENSE                           # MIT
├── pyproject.toml                    # Build config + dependencies
├── .gitignore
│
├── configs/                          # YAML experiment configs
│   ├── small.yaml                    #   Smoke test (synthetic, 30s)
│   ├── paper.yaml                    #   Full experiment
│   └── hardening.yaml                #   Hardening-specific ablation
│
├── src/daft/                         # Main package
│   ├── __init__.py                   #   Version, public API
│   │
│   ├── data/                         # Data pipeline
│   │   ├── __init__.py
│   │   ├── panel.py                  #   Panel dataclass (T×N×F tensor + masks)
│   │   └── loaders.py                #   Data source adapters ( → )
│   │
│   ├── features/                     # Feature engineering
│   │   ├── __init__.py
│   │   ├── tensor_factors.py         #   GPU-vectorized primitives (rank, corr, ewma, ts_*)
│   │   ├── legacy_factors.py         #   204 hand-crafted alpha factors
│   │   ├── regime_features.py        #   Market state vector s_t construction
│   │   └── freq_features.py          #   FFT spectral features (Super-Linear style)
│   │
│   ├── models/                       # Core architecture (MAIN CONTRIBUTION)
│   │   ├── __init__.py
│   │   ├── experts/                  #   Strategy expert pool
│   │   │   ├── __init__.py
│   │   │   ├── base_expert.py        #     Abstract expert interface
│   │   │   ├── trend_expert.py       #     Trend-following (moving average crossover, MACD)
│   │   │   ├── reversal_expert.py    #     Mean-reversion (Bollinger, RSI, cointegration)
│   │   │   ├── volatility_expert.py  #     Volatility regime (GARCH signal, VIX-related)
│   │   │   └── event_expert.py       #     Event-driven (earnings, macro announcements)
│   │   ├── router.py                 #   [C1] Regime Router (Stable LatentMoE)
│   │   ├── memory.py                 #   [C2] KDA Market Memory
│   │   ├── cross_dim_attn.py         #   [C3] Cross-Dimension Attention Protocol ★
│   │   ├── hardening.py              #   [C4] Adaptive Hardening Mechanism ★
│   │   └── ensemble.py               #   Expert fusion + signal generation
│   │
│   ├── training/                     # Staged training pipeline
│   │   ├── __init__.py
│   │   ├── expert_trainer.py         #   Stage 1: Independent expert training
│   │   ├── router_trainer.py         #   Stage 2: Router + Memory training (experts frozen)
│   │   └── joint_trainer.py          #   Stage 3: Joint fine-tuning (all parameters)
│   │
│   ├── portfolio/                    # Portfolio construction
│   │   ├── __init__.py
│   │   └── markowitz.py              #   Ledoit-Wolf shrunk Markowitz optimization
│   │
│   └── backtest/                     # Evaluation
│       ├── __init__.py
│       └── engine.py                 #   Vectorized backtesting + metrics (Sharpe, IC, IR, DD)
│
├── notebooks/                        # Analysis notebooks
│   ├── 01_data_exploration.ipynb     #     (placeholder)
│   ├── 02_regime_analysis.ipynb      #     (placeholder)
│   ├── 03_ablation_results.ipynb     #     (placeholder)
│   └── 04_hardening_analysis.ipynb   #     (placeholder)
│
├── tests/                            # Test suite
│   ├── __init__.py
│   ├── test_router.py                #     (placeholder)
│   ├── test_memory.py                #     (placeholder)
│   ├── test_cross_dim_attn.py        #     (placeholder)
│   ├── test_hardening.py             #     (placeholder)
│   └── test_ensemble.py              #     (placeholder)
│
├── docs/                             # Extended documentation
│   ├── architecture.md               #   Detailed architecture specification
│   └── experiments.md                #   Experiment log template
│
└── scripts/                          # Utility scripts
    ├── fetch_data.py                 #     (placeholder)
    └── run_ablation.py               #     (placeholder)
```

---

## Training Pipeline

### Stage 1: Independent Expert Training

Each expert is trained on its regime-specific subset of the data.

| Expert | Training Data Selection | Loss Function |
|--------|------------------------|---------------|
| Trend Expert | Periods with ADX > 25, sustained directional movement | Directional accuracy + Sharpe |
| Reversal Expert | Periods with low ADX, oscillating within bands | IC (Information Coefficient) |
| Volatility Expert | Periods with VIX/ATR above rolling 80th percentile | Volatility forecast MSE |
| Event Expert | ±3-day windows around earnings, FOMC, macro releases | Post-event directional accuracy |

### Stage 2: Router + Memory Training

- **Experts frozen** (Stage 1 weights)
- **Router trained** to select the best expert(s) for each market state
- **Memory trained** to retain the most predictive historical patterns
- **CDAP connections** enabled at low modulation strength ($\delta = 0.1$)
- **Loss**: Weighted sum of expert prediction quality, weighted by routing probabilities

### Stage 3: Joint Fine-Tuning

- **All parameters unfrozen**
- **Full CDAP modulation** ($\delta = 1.0$)
- **Low learning rate** ($\eta = 10^{-5}$) to prevent catastrophic forgetting
- **Early stopping** on validation IC degradation

### Stage 4: Hardening

- Stage 3 model run in **inference mode** over the full training set
- Hardening engine **counts** $(regime, expert\_pattern)$ co-occurrence frequencies
- Patterns with count $\geq \theta_{\text{harden}}$ are **cached**
- Validation on held-out period: **fast-path vs. full-path accuracy delta < 2%** required for hardening acceptance

---

## Experiments

> **⚠️ PLACEHOLDER — Experimental results will be populated as training completes.**

### Benchmarks

| Benchmark | Description | Data Source |
|-----------|-------------|-------------|
| **CSI 500 Minutes** | 500 A-share stocks, 5-year 1-minute bars | (to be configured) |
| **CSI 300 Multi-Freq** | 300 stocks, daily/weekly/minutely factors | (to be configured) |
| **S&P 500 ETF Universe** | US-listed ETFs, multi-asset | (to be configured) |

### Ablation Studies

To isolate the contribution of each architectural component, we disable one at a time:

| Experiment | CDAP | AHM | Router → Mem | Mem → Depth | Depth → Router |
|------------|:----:|:---:|:------------:|:-----------:|:--------------:|
| Full DAFT | ✅ | ✅ | ✅ | ✅ | ✅ |
| – CDAP | ❌ | ✅ | ❌ | ❌ | ❌ |
| – AHM | ✅ | ❌ | ✅ | ✅ | ✅ |
| – R→M only | ✅ | ✅ | ❌ | ✅ | ✅ |
| – M→D only | ✅ | ✅ | ✅ | ❌ | ✅ |
| – D→R only | ✅ | ✅ | ✅ | ✅ | ❌ |
| Baseline (MLP) | ❌ | ❌ | ❌ | ❌ | ❌ |
| Baseline (Transformer) | ❌ | ❌ | ❌ | ❌ | ❌ |
| Time-MoE (ICLR 2025) | ❌ | ❌ | ❌ | ❌ | ❌ |

**Expected Outcome**: Each CDAP connection contributes a marginal improvement; the full three-way modulation is more than the sum of its parts (synergy hypothesis). AHM should reduce average inference latency by 60–80% with < 2% accuracy degradation.

### Hardening Analysis

| Metric | Full Path | Hardened Path | Δ |
|--------|-----------|---------------|---|
| Avg. Inference Time | (ms) | (ms) | –% |
| Sharpe Ratio | — | — | — |
| Max Drawdown | — | — | — |
| IC (Rank) | — | — | — |
| % Fast-Path Eligible | — | — | — |
| Regime Degradation Rate | — | — | — |

---

## Performance

> **⚠️ PLACEHOLDER — Performance metrics will be populated as training completes.**

### Forecast Accuracy

| Model | IC (Rank) | ICIR | RMSE |
|-------|-----------|------|------|
| DAFT (Ours) | — | — | — |
| Transformer | — | — | — |
| MLP | — | — | — |
| XGBoost | — | — | — |
| Time-MoE | — | — | — |

### Inference Efficiency

| Mode | Tokens/s-equiv* | Latency (ms/step) | Memory (MB) |
|------|-----------------|-------------------|-------------|
| Full Path (3 experts) | — | — | — |
| Hardened Path (cached) | — | — | — |
| Speedup Ratio | — | — | — |

*\*"Tokens/s-equiv": financial time-step equivalents processed per second*

---

## Limitations

1. **Single-market validation**. Initial experiments focus on Chinese A-shares. Cross-market generalization (US equities, crypto, FX) remains unverified.

2. **Mid-frequency only**. The architecture assumes minute- to hour-scale decision horizons. Extending to tick-level (HFT) or monthly (factor investing) requires architectural adaptation.

3. **8 experts may underfit regime diversity**. K3 uses 896 experts; DAFT uses 8. This is a deliberate trade-off for training feasibility on consumer hardware, but may miss nuanced market sub-regimes.

4. **Hardening assumes regime stationarity within cache window**. If markets undergo a structural break (e.g., T+0 reform, circuit breaker introduction), cached fast paths may become stale. The entropy-based degradation trigger is a heuristic — it can fail if the break is gradual rather than abrupt.

5. **Transaction costs not fully modeled**. Current backtesting uses simplified cost assumptions. Production deployment would require bid-ask spread, market impact, and capacity constraint modeling.

6. **No live trading validation**. All results are historical backtests. Live forward-testing with paper trading is needed to confirm robustness against overfitting.

---

## Citation

If you use DAFT in your research, please cite:

```bibtex
@software{daft2026,
  author       = {Alastair(Dongxu-Jiang)},
  title        = {DAFT: Dimension-Aware Financial Trading},
  year         = {2026},
  publisher    = {GitHub},
  journal      = {GitHub Repository},
  url          = {https://github.com/[Dongxu-Jiang]/daft},
  note         = {A cross-dimensional attention architecture for medium-frequency
                  quantitative trading, inspired by Kimi K3},
}
```

---

## License

This project is licensed under the **MIT License**. See [LICENSE](LICENSE) for details.

The design is inspired by publicly available descriptions of Kimi K3 (Moonshot AI, July 2026), Super-Linear (Azencot Group, 2025), and ml-quant-trading (Yimin Du, 2025). All referenced projects are used under their respective open-source licenses.

---

## Acknowledgements

- **Moonshot AI** — Kimi K3 architecture (Stable LatentMoE, KDA, AttnRes), whose design principles inspired this work
- **Yimin Du** — [ml-quant-trading](https://github.com/initial-d/ml-quant-trading) (MIT), which provides the foundational data pipeline and backtesting infrastructure
- **Azencot Group** — [Super-Linear](https://github.com/azencot-group/SuperLinear) (MIT), which demonstrates the FFT-gated sparse MoE paradigm for time series
- **DeepSeek AI** — DeepSeek V4 technical report, whose mHC (manifold-constrained hyper-connections) and Engram proposals provide complementary perspectives
- **Time-MoE Team** — ICLR 2025 Spotlight, the first billion-scale MoE time series foundation model

---

<div align="center">

**[⬆ Back to Top](#daft-dimension-aware-financial-trading)**

*Built with PyTorch · Inspired by Kimi K3 · Designed for research*

</div>
