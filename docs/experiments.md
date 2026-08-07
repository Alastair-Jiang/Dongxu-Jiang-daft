# DAFT Experiment Log

> **⚠️ This document is a TEMPLATE. Populate with actual results as experiments complete.**

---

## Experiment 001: Smoke Test (2026-07-25)
| Field | Value |
|-------|-------|
| **Date** | 2026-07-25 |
| **Config** | `scripts/smoke_test.py` |
| **Data** | Synthetic, 200 stocks x 500 days (120,000 bars) |
| **Device** | CPU (Windows 11, Python 3.13, PyTorch 2.13) |
| **Result** | ALL 8 TESTS PASSED |
| **Router** | Stable LatentMoE: 200->16 latent, top-3/8 experts |
| **Memory** | KDA: 128x64 state, per-channel forget, route-modulated |
| **CDAP** | 3-way mutual modulation (routing<->memory<->depth) |
| **AHM** | 95% fast-path after warmup on consistent patterns |
| **Ensemble** | Signal (B,1), gradient flowing to 90/129 params |
| **Parameters** | 275,099 (M4-friendly, under 500K threshold) |
| **Notes** | Fixed memory_to_joint bottleneck (1M->16K params) |

---

## Experiment 002: Synthetic Training Demo (2026-07-25)
| Field | Value |
|-------|-------|
| **Date** | 2026-07-25 |
| **Config** | `scripts/training_loop.py` |
| **Data** | Synthetic, 50 stocks x 300 days (14,400 bars, 5min) |
| **Device** | CPU (Windows 11, Python 3.13, PyTorch 2.13) |
| **Stage 1** | 5 epochs, final loss=0.000122, 20.9s |
| **Stage 2** | 10 epochs, final loss=0.000117, 29.6s |
| **Stage 3** | 5 epochs, final loss=0.000961, 20.5s |
| **Stage 4** | 50 batches, 1 cached pattern, 40% fast-path ratio |
| **Total Time** | 71.4s |
| **Notes** | Stage 3 loss spike expected (CDAP unfreeze + modulation 0.1->1.0). Converging toward Stage 2 levels. Hardening functional (20 fast/30 slow paths). | |

---

## Ablation Results

### Component Ablation

| Experiment | Sharpe | MaxDD | IC (Rank) | ICIR | Latency (ms) |
|------------|--------|-------|-----------|------|--------------|
| Full DAFT | — | — | — | — | — |
| – CDAP (all connections disabled) | — | — | — | — | — |
| – AHM (hardening disabled) | — | — | — | — | — |
| – R→M (router→memory only) | — | — | — | — | — |
| – M→D (memory→depth only) | — | — | — | — | — |
| – D→R (depth→router only) | — | — | — | — | — |
| Baseline: MLP (single model) | — | — | — | — | — |
| Baseline: Transformer | — | — | — | — | — |
| Baseline: Time-MoE | — | — | — | — | — |

### Hardening Analysis

| θ (threshold) | Fast Path % | Speedup | Δ Accuracy | Regime Degradations |
|---------------|-------------|---------|------------|---------------------|
| 50 | — | — | — | — |
| 100 | — | — | — | — |
| 200 | — | — | — | — |
| 500 | — | — | — | — |

### Regime Distribution (Qualitative)

(To be filled: plot of regime cluster assignments over time, overlaid
with known market events for qualitative validation.)

---

## Reproducibility Checklist

- [ ] Random seed fixed (42)
- [ ] Data splits documented (train/val/test ratios)
- [ ] Hyperparameters committed to config YAML
- [ ] Model checkpoints saved
- [ ] Environment (`pip freeze`) archived
- [ ] All metrics computed on held-out test set ONLY
