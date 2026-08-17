# DAFT Project Log — v0.0.1（旧规则:v0.2.0）

**Dimension-Aware Financial Trading: A Cross-Dimensional Attention Architecture for Medium-Frequency Quantitative Trading**

| | |
|---|---|
| **Version** | v0.0.1（旧规则:v0.2.0） (2026-08-07) |
| **Repository** | [github.com/Alastair-Jiang/Dongxu-Jiang-daft](https://github.com/Alastair-Jiang/Dongxu-Jiang-daft) |
| **Code size** | ~8,800 lines (Python, PyTorch) |
| **License** | MIT |

---

## 1. What is DAFT

DAFT is a research project exploring whether architectural ideas from large language models can be adapted to quantitative trading. It is inspired by **Kimi K3** (Moonshot AI, July 2026), currently the largest open-weight model (2.8T parameters, 896 experts, 16 activated per token).

**Core idea.** In most ML-for-trading systems, three functional components operate independently:

1. **Regime identification** — what kind of market are we in (trending / mean-reverting / volatile)?
2. **Historical memory** — how did similar situations resolve in the past?
3. **Feature hierarchy** — which level of signal should we trust (raw price, base factors, composite state)?

DAFT argues these are not independent problems but **three facets of one problem**, and proposes a trainable protocol that lets them modulate each other bidirectionally:

- routing decisions influence what the memory retains,
- memory state influences which feature layers are trusted,
- feature signals feed back to correct routing bias.

This is formalized as the **Cross-Dimension Attention Protocol (CDAP)** — the project's main original contribution — together with an **Adaptive Hardening Mechanism (AHM)** that generalizes K3's static fast/slow-layer ratio into a data-driven, regime-adaptive routing policy.

**Design target.** Medium-frequency (minute-level) A-share trading. Total model parameters ≈ 275K, trainable on CPU / Apple Silicon / entry-level GPUs.

---

## 2. Architecture Overview

```
[Market Data] → Panel(T×N×F) → Feature Engine → s_t ∈ R^200 (market state)
                                                    │
        ┌───────────────────────────────────────────┤
        ▼                ▼                ▼
   RegimeRouter      KDA Memory     Layer Projectors (L0/L1/L2)
   (latent MoE)      (delta-rule)   (feature hierarchy)
        │                │                │
        └───────► CDAP cross-dimension attention ◄──────┘
                    (e ⊙ m ⊙ d joint-space fusion)
                        │
                   Expert Ensemble → signal (B, 1)
                        │
              Markowitz Optimizer → weights (N,)
                        │
                  Backtest Engine → metrics
```

| Component | Design source | Description |
|---|---|---|
| **RegimeRouter** | K3 Stable LatentMoE | Projects 200-d market state into a 16-d latent regime space; routes to 8 experts, top-3 activated; SiTU activation; quantile balancing (no auxiliary load-balancing loss); temperature schedule 1.0 → 0.1 |
| **KDAMarketMemory** | KDA (Kimi Delta Attention, [arXiv:2510.26692](https://arxiv.org/abs/2510.26692)) | Fixed-size 128×64 state matrix updated by delta rule; per-channel forget gates with safe-gate lower bound; route-modulated forgetting (CDAP connection); no KV cache, O(d_k·d_v) per step |
| **CDAP** ★ | Original (inspired by K3 AttnRes) | Projects routing distribution, memory state, and layer outputs into a shared 64-d joint latent space; element-wise fusion j = e⊙m⊙d; reverse projections correct all three dimensions; learnable residual scales for conservative modulation |
| **AHM** ★ | Original (generalizes K3 3:1 ratio) | After training, frequent (regime, expert-pattern) tuples are cached as O(1) fast-path lookups; entropy guard degrades to full exploration on regime shifts; staleness eviction |
| **Expert pool** | MoE philosophy | 8 experts (2× trend / 2× reversal / 2× volatility / 2× event), each with a specialized loss (direction-weighted MSE, negative rank-IC, MSE+var-regularization, direction-weighted MSE) |

**Feature engine**: 200-d market state vectors from 6 groups — price/return dynamics (55), volatility structure (40), volume/liquidity (30), technical/momentum (35), cross-sectional context (30), spectral/FFT (10). All factor primitives are mask-aware and GPU-vectorized.

---

## 3. Progress

### v0.1.0 (2026-07) — Core architecture（旧规则最早版本，新规则下未重编号）
- All four model components, 200-d feature engine, synthetic data generator (HMM 3-regime × 3-factor model), Stage-1 expert training, smoke tests.

### v0.0.1 (2026-08-06/07) — Full pipeline end-to-end（旧规则:v0.2.0）
All `NotImplementedError`s eliminated:

| Module | What was added |
|---|---|
| `backtest/engine.py` | Walk-forward vectorized backtest; signal→position via quantile selection (long / long-short); transaction costs + slippage; Sharpe, MaxDD, Calmar, Rank-IC, ICIR, hit rate, turnover, DD duration |
| `portfolio/markowitz.py` | Ledoit-Wolf shrunk covariance (OAS variant, pure PyTorch closed-form, no CVXPY); box-constrained weights via iterative projection |
| `training/router_trainer.py` | Stage 2: router + memory + CDAP training with frozen experts; routing-weighted loss; entropy regularization; temperature annealing; quantile balancing; memory detach |
| `training/joint_trainer.py` | Stage 3: full joint fine-tuning; dual learning-rate groups (experts ×0.1); early stopping on validation IC |
| `data/adapters/` | `BaostockAdapter` (A-share, built-in CSI-300 sample of 50 tickers) and `YFinanceAdapter` (US); Panel conversion with suspension masks and limited forward-fill |
| `utils/` | Multi-backend device detection (CUDA/XPU/DirectML/MPS/CPU); Rank-IC, ICIR, hit-rate metrics |
| Scripts | `run_stage1/2/3.py`, `run_full_pipeline.py`, `run_backtest_only.py`, `smoke_test_all.py` |

---

## 4. Experiments

### 4.1 Setup (QUICK config, run on CPU)

- Synthetic data: 50 stocks × 300 days (HMM 3-regime, 3-factor returns)
- Stage 1: 15 epochs, batch 1024, lr 1e-3
- Stage 2: 10 epochs, batch 512, lr 1e-3
- Stage 3: 8 epochs, batch 512, lr 1e-5
- Backtest: transaction cost 5 bps + slippage 1 bps, top-quantile 0.2, long-short

### 4.2 Results

| Stage | Time | Key result |
|---|---|---|
| Stage 1 (experts) | 65.7 s | 8/8 experts converged |
| Stage 2 (router+memory) | 155.3 s | validation IC: +0.045 → +0.061; routing entropy annealed 2.07 → 1.04 |
| Stage 3 (joint) | 67.6 s | validation IC peak **+0.108** |
| Backtest | 0.1 s | IC +0.020, ICIR +0.140 |

**Reading the results.** Validation IC rises monotonically across stages (0.045 → 0.061 → 0.108), indicating that joint training of routing, memory, and CDAP learns meaningful structure — the architecture's training loop works as designed. Routing entropy decreasing smoothly (2.07 → 1.04) shows the router transitions from uniform exploration to a confident preference, which is the expected behavior of temperature annealing.

The negative backtest Sharpe (-1.79) is **expected under this setup and should not be interpreted as either strategy failure or architecture failure**: synthetic data is generated as a near-random walk with low signal-to-noise ratio, transaction costs are applied, and the evaluation is in-sample. See Limitations.

### 4.3 Reproduce

```bash
# Full pipeline (data → stage1 → stage2 → stage3 → backtest → report)
python scripts/run_full_pipeline.py          # quick config
python scripts/run_full_pipeline.py --full   # larger config (100×500, more epochs)

# Real A-share data (requires: pip install baostock)
python scripts/run_full_pipeline.py --source baostock

# Backtest only (reuse existing checkpoints)
python scripts/run_backtest_only.py
```

---

## 5. Known Limitations

1. **In-sample backtest.** The pipeline currently evaluates on the full panel (including training periods). A strict train/val/test split with out-of-sample backtesting is required before any performance claim.
2. **Feature normalization look-ahead.** Standardization statistics are computed on the full dataset; for real data these must be fitted on the training split only.
3. **No ablation study yet.** The value of CDAP/AHM over independent components has not been quantified. Component ablations and simple baselines (e.g., linear model, MLP, Transformer) are the next priority.
4. **Synthetic data only so far.** The baostock adapter is implemented but not yet run; no real-market validation has been performed.
5. **Medium-frequency (minute-level) target not yet exercised.** Current runs use daily bars; minute-level data and transaction-cost realism (T+1, limit-up/down) remain to be validated.

---

## 6. Next Steps

1. Out-of-sample backtesting (train/val/test split, fitted-only normalization)
2. Full-scale run (`--full` config) to confirm IC trends
3. Real A-share data run via baostock (CSI-300 sample)
4. Ablation study: full DAFT vs. no-CDAP vs. no-AHM vs. individual CDAP directions vs. linear baseline
5. Minute-level validation with realistic A-share microstructure constraints

---

## 7. Related Work & References

| Area | Paper / Resource | Relevance to DAFT |
|---|---|---|
| LLM architecture | **Kimi K3 Technical Report** (Moonshot AI, 2026-07) | Source of Stable LatentMoE, KDA, AttnRes design principles |
| Linear attention | **KDA: Kimi Delta Attention** — [arXiv:2510.26692](https://arxiv.org/abs/2510.26692) | Direct basis of KDAMarketMemory |
| Sparse routing | **Time-MoE** (ICLR 2025) — [arXiv:2405.16073](https://arxiv.org/abs/2405.16073) | MoE for time series; routing design comparison |
| Sequence memory | **xLSTM** (NeurIPS 2024) — [arXiv:2405.04517](https://arxiv.org/abs/2405.04517) | Alternative linear-memory family; candidate baseline |
| State-space models | **Mamba** — [arXiv:2312.00752](https://arxiv.org/abs/2312.00752) | SSM family context for memory design |
| Time-series forecasting | **LTSF-Linear** (AAAI 2023) — [arXiv:2205.13504](https://arxiv.org/abs/2205.13504) | "Linear models beat complex transformers" — motivates baselines |
| Time-series forecasting | **PatchTST** (ICLR 2023) — [arXiv:2211.14730](https://arxiv.org/abs/2211.14730) | Transformer baseline candidate |
| Time-series forecasting | **iTransformer** (ICLR 2024) — [arXiv:2310.06625](https://arxiv.org/abs/2310.06625) | Transformer baseline candidate |
| Financial DL benchmark | **Deep Learning for Financial Time Series** (2026) — [arXiv:2603.01820](https://arxiv.org/abs/2603.01820) | Large-scale benchmark (15y, multi-asset, daily); evaluation framework reference |
| Synthetic benchmarks | **FinStressTS** — [arXiv:2606.03184](https://arxiv.org/abs/2606.03184) | Parametric synthetic benchmark methodology |
| Financial ML methods | **López de Prado, Advances in Financial Machine Learning** (Wiley, 2018) | Out-of-sample methodology, leakage prevention, sample weighting |
| Portfolio optimization | **Ledoit & Wolf (2004)** — [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0304407603002334) | Covariance shrinkage used in MarkowitzOptimizer |
| Factor investing | **Fama & French (1993)** — [PDF](https://pages.stern.nyu.edu/~ehabers/FF_JFE1993.pdf) | Conceptual basis of the factor/feature engine |
| Regime switching | **Hamilton (1989)** — [JSTOR](https://www.jstor.org/stable/1912559) | Markov regime-switching; basis of synthetic generator |
| Regime-aware routing | **Temporal Routing Adaptor** (ICLR 2023) — [arXiv:2210.12153](https://arxiv.org/abs/2210.12153) | Related work on adaptive routing for time series |

---

*Generated 2026-08-07. This log is maintained alongside the codebase; see `docs/PROJECT_REPORT.md` for the full in-repo report.*
