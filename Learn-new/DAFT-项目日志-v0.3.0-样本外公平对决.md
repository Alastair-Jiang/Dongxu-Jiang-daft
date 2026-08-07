# DAFT Project Log — v0.3.0

**Baseline & Out-of-Sample Evaluation: DAFT vs. Ridge on Real A-Share Data**

| | |
|---|---|
| **Version** | v0.3.0 (2026-08-07) |
| **Repository** | [github.com/Alastair-Jiang/Dongxu-Jiang-daft](https://github.com/Alastair-Jiang/Dongxu-Jiang-daft) |
| **Milestone** | First strict out-of-sample comparison of the full DAFT pipeline against a linear baseline on real Chinese A-share data |
| **License** | MIT |

---

## 1. What happened this iteration

Two additions move DAFT from "architecture demonstration" toward "empirically accountable research":

1. **A baseline and an evaluation discipline.** `scripts/run_baseline_ridge.py` and `scripts/run_baseline_ridge_real.py` implement a ridge-regression baseline (200 features, 200 parameters) evaluated under strict out-of-sample rules: chronological train/val/test split, normalization statistics fitted on the training segment only, transaction costs applied.

2. **An out-of-sample version of the full DAFT pipeline.** `scripts/run_full_pipeline_oos.py` + `scripts/run_oos_backtest_only.py` run the complete DAFT training protocol (Stage 1 experts → Stage 2 router/memory/CDAP → Stage 3 joint fine-tuning) **without ever exposing the test segment to training**, with causal memory warm-up and train-only normalization. This closes the in-sample-evaluation gap identified in the v0.2.0 review.

### New scripts
```
scripts/run_baseline_ridge.py           # ridge baseline, synthetic data (framework validation)
scripts/run_baseline_ridge_real.py      # ridge baseline, real A-shares (baostock)
scripts/run_full_pipeline_oos.py        # DAFT end-to-end, out-of-sample, real data
scripts/run_oos_backtest_only.py        # reuse checkpoints: signals → backtest (no retraining)
```

### New outputs
```
outputs/baseline_ridge_report.json          # synthetic baseline
outputs/baseline_ridge_real_report.json     # real-data baseline (30 stocks)
outputs/full_pipeline_oos_report.json       # DAFT, real data, out-of-sample (20 stocks)
```

---

## 2. Why a baseline matters (research rationale)

A recurring finding in time-series forecasting is that complex architectures frequently fail to beat simple linear models on noisy, low-SNR series (see LTSF-Linear, AAAI 2023, [arXiv:2205.13504](https://arxiv.org/abs/2205.13504)). DAFT's core claims — that routing, memory, and feature-hierarchy should mutually modulate (CDAP), and that adaptive hardening (AHM) adds value — can only be attributed if the full model outperforms a simple baseline under identical evaluation conditions. This iteration establishes that measurement apparatus.

---

## 3. Experimental setup

All real-data experiments use the same evaluation protocol:

| Setting | Value |
|---|---|
| Data | baostock, CSI-300 constituent sample, daily bars, forward-adjusted |
| Period | 2021-01-01 → 2025-12-31 (1,212 trading days) |
| Split | train 60% / val 20% / test 20% (chronological) |
| Normalization | fit on train only, frozen for val/test |
| Transaction costs | 5 bps + 1 bps slippage, per-unit turnover |
| Position sizing | top-quantile 0.2, long-short |
| Evaluation | test-segment only: Rank IC, ICIR, IC t-stat, hit rate, Sharpe (net of costs) |

**Caveat on comparability:** the ridge baseline was run on 30 stocks; the DAFT OOS run used 20 stocks (same period, same protocol). The comparison below is indicative rather than perfectly matched; a 30-stock DAFT run is planned to remove this residual difference.

---

## 4. Results

### 4.1 Synthetic data sanity check

The framework was first validated on synthetic data (50 stocks × 300 days, same protocol):

| Metric | Ridge (synthetic) |
|---|---|
| Rank IC | +0.009 |
| IC t-stat | +0.36 |
| IC>0 ratio | 50.0% |

Synthetic HMM data is near-random-walk: the baseline finds no signal, confirming that the evaluation framework does not fabricate IC. Any positive number reported later is therefore meaningful.

### 4.2 Real A-shares: DAFT vs. Ridge (strict out-of-sample)

| Metric | Ridge (linear, 200 params) | DAFT (full, ~275K params) |
|---|---|---|
| **Rank IC** | **+0.0285** | +0.0252 |
| **ICIR** | +0.122 | +0.110 |
| **IC t-stat** | **+1.89** | +1.71 |
| IC>0 ratio | 53.5% | **55.4%** |
| Hit rate | 50.5% | 50.0% |
| **Sharpe (net)** | **+0.69** | −0.70 |
| Annual return (net) | +11.3% | −10.0% |
| Max drawdown | −18.2% | −15.0% |
| Turnover | 0.03% | 1.7% |

### 4.3 Honest interpretation

1. **Predictive power: DAFT ≈ Ridge.** IC (0.025 vs 0.029) and t-stat (1.71 vs 1.89) are statistically indistinguishable. At the current training budget, the full architecture does **not** demonstrate superior predictive ability over a 200-parameter linear model. This is consistent with the "linear baseline paradox" observed broadly in time-series ML.

2. **Trading layer: DAFT materially worse.** Net Sharpe −0.70 vs +0.69. The primary driver is turnover (1.7% vs 0.03% per step): DAFT pays transaction costs on a signal that is not stronger than the baseline's. This is a second, model-level demonstration that costs dominate medium-frequency results.

3. **Positive signals in the DAFT run:** IC>0 ratio 55.4% (vs 53.5%) and a positive training-validation IC trajectory indicate the pipeline learns real structure from A-share data; the out-of-sample framework now produces numbers that can be defended.

4. **Known confounds:** (a) different stock counts (30 vs 20); (b) Stage 3 was early-stopped at epoch 5 with val-IC peak +0.0049 — well below the synthetic-data peak (0.108), suggesting underfitting on real data; (c) daily frequency only — the medium-frequency (minute-level) target remains untested.

---

## 5. What this means for the project

- The out-of-sample evaluation framework is now the project's standard of evidence. All future claims should be reported against it.
- The feature engine + linear model demonstrably extract weak but real signal from A-shares (IC 0.029, t≈1.9) — the feature engineering is validated.
- Current evidence does **not** support the claim that CDAP/AHM complexity pays for itself. This is a research finding, not a failure: the next iteration should either improve DAFT's real-data training (longer Stage 3, hyperparameter search, feature selection) or explicitly frame the complexity-vs-baseline question as the project's open research problem.
- Recommended posture: keep the ridge baseline as the production-usable reference, and treat DAFT's architectural claims as hypotheses under test.

---

## 6. Next steps

1. Match stock counts (30-stock DAFT OOS run) for a clean head-to-head
2. Longer Stage 3 / hyperparameter tuning on real data; check whether val-IC can approach meaningful levels (≥0.03)
3. Feature selection / dimensionality reduction before routing (213 → ~50 features)
4. T+1 and limit-up/down constraints in the backtest (A-share microstructure)
5. Ablation study once DAFT and baseline are properly matched

---

## 7. References

| Area | Paper / Resource | Link |
|---|---|---|
| Linear baseline paradox | LTSF-Linear (AAAI 2023) | https://arxiv.org/abs/2205.13504 |
| Memory component | KDA (Kimi Delta Attention) | https://arxiv.org/abs/2510.26692 |
| Sparse routing | Time-MoE (ICLR 2025) | https://arxiv.org/abs/2405.16073 |
| Financial DL benchmark | Deep Learning for Financial Time Series (2026) | https://arxiv.org/abs/2603.01820 |
| Evaluation methodology | López de Prado, *Advances in Financial Machine Learning* (2018) | https://www.wiley.com/en-us/Advances+in+Financial+Machine+Learning-p-9781119482086 |

---

*Generated 2026-08-07. All results verified against `outputs/*.json` run artifacts; see `docs/PROJECT_REPORT.md` for the in-repo developer-facing report.*
