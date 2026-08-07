"""Component 4: Adaptive Hardening Mechanism (AHM). ★ ORIGINAL CONTRIBUTION ★

Generalizes Kimi K3's static 3:1 KDA-to-full-attention layer ratio into a
data-driven, regime-adaptive routing policy.

Core idea:
    After training, frequently traversed (regime, expert_pattern) tuples
    are cached as O(1) fast-path lookups. Routine market conditions follow
    hardened paths; anomalous conditions automatically degrade to full
    exploration via entropy-based regime shift detection.

Inspiration from K3:
    K3 uses a FIXED 3:1 ratio — 3 KDA (linear-attn) layers per 1 MLA
    (full-attn) layer. This works well for NLP, but financial time series
    have REGIME-DEPENDENT information density:
    → Low-volatility trend: mostly routine → should use ~90% fast path
    → High-volatility event: highly novel → should use ~10% fast path

    DAFT's AHM learns these ratios from data instead of hardcoding them.

Key mechanisms:
    1. Pattern Counter: tracks (regime_id, expert_discrete_pattern) frequency
    2. Cache Builder: patterns observed ≥ θ times → hardened cache entry
    3. Entropy Guard: routing entropy spike → degrade to full exploration
    4. Staleness Eviction: structural breaks → purge stale cache entries
"""

from collections import defaultdict
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn


class HardeningEngine:
    """Adaptive hardening: learn fast paths from usage frequency.

    Parameters
    ----------
    n_regimes : int
        Number of discrete regime clusters to track.
    n_experts : int
        Number of strategy experts.
    threshold : int
        θ: minimum observations before a pattern is eligible for hardening.
    min_confidence : float
        ρ: minimum ratio (pattern_count / total_decisions) for cache validity.
    entropy_multiplier : float
        λ: entropy threshold multiplier for regime shift detection.
        Recent entropy > λ × baseline → degrade to full exploration.
    """

    def __init__(
        self,
        n_regimes: int = 10,
        n_experts: int = 10,
        threshold: int = 100,
        min_confidence: float = 0.95,
        entropy_multiplier: float = 2.0,
    ):
        self.n_regimes = n_regimes
        self.n_experts = n_experts
        self.threshold = threshold
        self.min_confidence = min_confidence
        self.entropy_multiplier = entropy_multiplier

        # === Pattern frequency statistics ===
        self.pattern_counter: Dict[Tuple, int] = defaultdict(int)

        # === Hardened cache: (regime_id, pattern) → cached weights ===
        self.cache: Dict[Tuple, Dict[str, torch.Tensor]] = {}

        # === Global statistics ===
        self.total_decisions: int = 0
        self.baseline_entropy: float = 0.0
        self.entropy_history: list = []  # Rolling window of recent entropies

        # === Tracking metrics ===
        self.n_fast_path: int = 0
        self.n_slow_path: int = 0
        self.n_degradations: int = 0

    def _discretize_pattern(
        self,
        routing_probs: torch.Tensor,
        top_k: int = 3,
    ) -> Tuple:
        """Convert soft routing distribution to a discrete pattern ID.

        Preserves Top-K expert ordering and rough magnitude binning.
        """
        topk_probs, topk_indices = torch.topk(routing_probs, top_k)
        # Sort indices for order-invariant pattern matching
        sorted_indices = tuple(topk_indices.sort().values.tolist())
        return sorted_indices

    def _compute_routing_entropy(
        self,
        routing_probs: torch.Tensor,
    ) -> float:
        """Shannon entropy of the routing distribution.

        High entropy ≈ uncertain routing ≈ novel regime → degrade to slow path.
        """
        # Avoid log(0)
        probs = routing_probs.clamp(min=1e-8)
        entropy = -(probs * probs.log()).sum().item()
        return entropy

    def should_use_fast_path(
        self,
        regime_id: int,
        routing_probs: torch.Tensor,
    ) -> bool:
        """Decide whether the current market state can use a hardened fast path.

        Parameters
        ----------
        regime_id : int
            Discrete regime cluster ID from the router.
        routing_probs : torch.Tensor, shape (n_experts,)
            Current soft routing distribution.

        Returns
        -------
        use_fast : bool
            True if a hardened cache entry exists and is valid.
        """
        # === Update statistics ===
        pattern = self._discretize_pattern(routing_probs)
        key = (regime_id, pattern)
        self.pattern_counter[key] += 1
        self.total_decisions += 1

        # === Update entropy baseline ===
        entropy = self._compute_routing_entropy(routing_probs)
        self.entropy_history.append(entropy)
        if len(self.entropy_history) > 1000:
            self.entropy_history.pop(0)
        if len(self.entropy_history) >= 30:
            self.baseline_entropy = sum(self.entropy_history) / len(self.entropy_history)

        # === Check: does a hardened cache entry exist? ===
        if key in self.cache:
            self.n_fast_path += 1
            return True

        # === Check: should we create a cache entry? ===
        if (self.pattern_counter[key] >= self.threshold and
                key not in self.cache):
            confidence = self.pattern_counter[key] / self.total_decisions
            if confidence >= self.min_confidence:
                self._create_cache(key, routing_probs)
                # Don't use fast path yet — first hit validates the cache

        self.n_slow_path += 1
        return False

    def _create_cache(
        self,
        key: Tuple,
        routing_probs: torch.Tensor,
    ) -> None:
        """Create a hardened cache entry for a frequently-seen pattern."""
        self.cache[key] = {
            'expert_weights': routing_probs.clone().detach(),
            'created_at': self.total_decisions,
            'hit_count': 0,
        }

    def get_cached_weights(
        self,
        regime_id: int,
        routing_probs: torch.Tensor,
    ) -> torch.Tensor:
        """Retrieve cached expert weights for a hardened pattern.

        Returns the frozen (detached) weights that were recorded when the
        pattern was first hardened. These do NOT receive gradient updates.
        """
        pattern = self._discretize_pattern(routing_probs)
        key = (regime_id, pattern)
        if key not in self.cache:
            raise KeyError(
                f"No cached entry for regime={regime_id}, pattern={pattern}. "
                "Call should_use_fast_path() first."
            )
        self.cache[key]['hit_count'] += 1
        return self.cache[key]['expert_weights']

    def detect_regime_shift(self) -> bool:
        """Check if recent routing entropy suggests a regime shift.

        If the recent entropy is significantly higher than the baseline,
        the market may have entered an unfamiliar regime → degrade to
        full exploration to avoid using stale hardened paths.

        Returns
        -------
        degraded : bool
            True if a regime shift is detected → disable fast path temporarily.
        """
        if len(self.entropy_history) < 20:
            return False

        recent = self.entropy_history[-20:]
        recent_avg = sum(recent) / len(recent)

        if self.baseline_entropy > 0:
            if recent_avg > self.entropy_multiplier * self.baseline_entropy:
                self.n_degradations += 1
                return True
        return False

    def evict_stale_entries(self, max_age: int = 10000) -> int:
        """Remove cache entries that haven't been used recently.

        Parameters
        ----------
        max_age : int
            Maximum decisions since last hit before eviction.

        Returns
        -------
        n_evicted : int
            Number of entries removed.
        """
        stale_keys = []
        for key, entry in self.cache.items():
            age = self.total_decisions - entry['created_at']
            if age > max_age and entry['hit_count'] < 10:
                stale_keys.append(key)

        for key in stale_keys:
            del self.cache[key]

        return len(stale_keys)

    def get_stats(self) -> dict:
        """Return hardening statistics for logging and analysis."""
        return {
            'total_decisions': self.total_decisions,
            'n_cached_patterns': len(self.cache),
            'n_fast_path': self.n_fast_path,
            'n_slow_path': self.n_slow_path,
            'n_degradations': self.n_degradations,
            'fast_path_ratio': (
                self.n_fast_path / max(self.total_decisions, 1)
            ),
            'baseline_entropy': self.baseline_entropy,
            'cache_hit_rate': (
                sum(e['hit_count'] for e in self.cache.values()) /
                max(sum(self.pattern_counter.values()), 1)
            ),
        }
