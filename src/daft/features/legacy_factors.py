"""Legacy alpha factors (204 hand-crafted factors).

Imported and adapted from ml-quant-trading (Yimin Du, 2025, MIT License).

Factor families:
    better_001–028 (28): VWAP deviation + volume-weighted momentum
    best_001–021   (21): Close-location momentum variants
    old_027–076    (50): Classic alpha signals (corr/rank composites)
    stock_001–022  (22): Per-stock derived series
    extra_001–014  (14): Turnover + amount features
    add_001–030    (30): Additional composite factors
    change_001–005  (5): Short-window velocity changes
    original_001–028 (28): Close/volume direct statistics
    cs_rank_*        (6): Market breadth indicators
"""

# PLACEHOLDER — to be populated by importing factor definitions from
# ml-quant-trading/src/mlquant/features/legacy_factors.py
#
# All factors use the mask-aware Panel interface:
#     factor(panel: Panel) → torch.Tensor, shape (T, N)
#
# See docs/architecture.md for the factor integration plan.
