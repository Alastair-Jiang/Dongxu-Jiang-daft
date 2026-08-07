"""Test Component 4: Adaptive Hardening Mechanism (AHM). ★ ORIGINAL ★"""

import pytest
import torch

from daft.models.hardening import HardeningEngine


# Fixture `hardening` is defined in conftest.py (shared across all tests)


# ── Fixtures ──────────────────────────────────────────────────────────


# Deterministic routing vectors — no noise in fixture, so pattern
# discretization produces consistent keys across calls.
ROUTING_CONSISTENT = torch.tensor(
    [0.5, 0.3, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0]
)  # always TopK → (0, 1, 2)
ROUTING_LOW_ENTROPY = torch.tensor(
    [0.9, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
)  # always TopK → (0, 1)
ROUTING_HIGH_ENTROPY = torch.ones(8) / 8                     # max entropy
ROUTING_ALT_PATTERN = torch.tensor(
    [0.0, 0.0, 0.7, 0.3, 0.0, 0.0, 0.0, 0.0]
)  # always TopK → (2, 3)


# ── Initialization ────────────────────────────────────────────────────

class TestInit:
    def test_default_params(self):
        h = HardeningEngine()
        assert h.n_regimes == 10
        assert h.n_experts == 10
        assert h.threshold == 100
        assert h.min_confidence == 0.95
        assert h.entropy_multiplier == 2.0

    def test_starts_empty(self, hardening):
        assert len(hardening.cache) == 0
        assert hardening.total_decisions == 0
        assert hardening.n_fast_path == 0
        assert hardening.n_slow_path == 0

    def test_custom_params(self):
        h = HardeningEngine(
            n_regimes=4, n_experts=6,
            threshold=50, min_confidence=0.8, entropy_multiplier=3.0,
        )
        assert h.n_regimes == 4
        assert h.threshold == 50
        assert h.min_confidence == 0.8


# ── Fast/slow path routing ────────────────────────────────────────────

class TestRouting:
    def test_initial_decisions_are_slow(self, hardening):
        """Before threshold is reached, no cache entries exist → slow path."""
        for _ in range(10):
            result = hardening.should_use_fast_path(0, ROUTING_CONSISTENT)
            assert result is False
        # Verify the counter increased (slow path is actually being used)
        assert hardening.n_slow_path == 10
        assert hardening.n_fast_path == 0

    def test_fast_path_after_warmup(self):
        """After reaching threshold for a pattern, that pattern uses fast path."""
        h = HardeningEngine(threshold=5, min_confidence=0.01)
        # Warm up with the same deterministic pattern
        for _ in range(10):
            h.should_use_fast_path(0, ROUTING_CONSISTENT)
        stats = h.get_stats()
        # At least one cache entry should be created (threshold=5, 10 calls)
        assert stats["n_cached_patterns"] >= 1
        # Some fast-path decisions should have occurred after cache creation
        assert stats["n_fast_path"] > 0

    def test_fast_path_increments_counter(self):
        """After caching, get_cached_weights is actually called on fast path."""
        h = HardeningEngine(threshold=3, min_confidence=0.01)
        fast_calls = 0
        for _ in range(10):
            if h.should_use_fast_path(0, ROUTING_CONSISTENT):
                _ = h.get_cached_weights(0, ROUTING_CONSISTENT)
                fast_calls += 1
        assert fast_calls > 0, "No fast-path calls occurred — cache never created"
        stats = h.get_stats()
        assert stats["total_decisions"] == 10

    def test_different_regimes_separate_counters(self, hardening):
        for _ in range(10):
            hardening.should_use_fast_path(0, ROUTING_CONSISTENT)
            hardening.should_use_fast_path(1, ROUTING_ALT_PATTERN)
        # Two different (regime, pattern) keys
        assert len(hardening.pattern_counter) >= 2


# ── Cache ─────────────────────────────────────────────────────────────

class TestCache:
    def test_get_cached_weights_raises_for_missing_key(self, hardening):
        routing = torch.rand(8).softmax(dim=0)
        with pytest.raises(KeyError):
            hardening.get_cached_weights(0, routing)

    def test_cached_weights_are_detached(self):
        """Cached weights should be detached (no grad)."""
        h = HardeningEngine(threshold=3, min_confidence=0.01)
        for _ in range(5):
            h.should_use_fast_path(0, ROUTING_CONSISTENT)
        weights = h.get_cached_weights(0, ROUTING_CONSISTENT)
        assert not weights.requires_grad

    def test_cache_hit_rate_defined(self):
        """After caching, cache_hit_rate stat should be present."""
        h = HardeningEngine(threshold=3, min_confidence=0.01)
        for _ in range(8):
            if h.should_use_fast_path(0, ROUTING_CONSISTENT):
                _ = h.get_cached_weights(0, ROUTING_CONSISTENT)
        stats = h.get_stats()
        assert "cache_hit_rate" in stats


# ── Regime shift detection ────────────────────────────────────────────

class TestRegimeShift:
    def test_no_shift_with_few_samples(self, hardening):
        """< 20 samples — cannot detect regime shift."""
        for _ in range(10):
            hardening.should_use_fast_path(0, ROUTING_HIGH_ENTROPY)
        assert not hardening.detect_regime_shift()

    def test_shift_with_high_entropy(self, hardening):
        """Sustained high-entropy routing should trigger regime-shift detection."""
        # Establish baseline with low-entropy routing
        for _ in range(100):
            hardening.should_use_fast_path(0, ROUTING_LOW_ENTROPY)

        # Inject high-entropy (uniform) routing for 30 steps
        for _ in range(30):
            hardening.should_use_fast_path(0, ROUTING_HIGH_ENTROPY)

        shift = hardening.detect_regime_shift()
        assert shift is True, (
            f"Expected regime shift with high-entropy routing. "
            f"baseline_entropy={hardening.baseline_entropy:.3f}"
        )

    def test_no_shift_with_consistent_routing(self, hardening):
        """Consistent low-entropy routing → no shift detected."""
        for _ in range(200):
            hardening.should_use_fast_path(0, ROUTING_LOW_ENTROPY)
        assert not hardening.detect_regime_shift()


# ── Eviction ──────────────────────────────────────────────────────────

class TestEviction:
    def test_evict_empty_cache(self, hardening):
        n = hardening.evict_stale_entries(max_age=5)
        assert isinstance(n, int)
        assert n == 0

    def test_fresh_entries_not_evicted(self):
        """Fresh cache entries should survive eviction."""
        h = HardeningEngine(threshold=3, min_confidence=0.01)
        for _ in range(6):
            h.should_use_fast_path(0, ROUTING_CONSISTENT)
        n_before = len(h.cache)
        n = h.evict_stale_entries(max_age=100000)
        assert n == 0
        assert len(h.cache) == n_before

    def test_stale_entries_evicted(self):
        """Cache entries with low hit_count and old age should be evicted."""
        h = HardeningEngine(threshold=2, min_confidence=0.01)
        # Create a cache entry
        for _ in range(5):
            h.should_use_fast_path(0, ROUTING_CONSISTENT)
        assert len(h.cache) == 1  # confirms entry was created

        # Artificially advance total_decisions to make it "old"
        # and ensure hit_count stays low (we only hit it during creation)
        h.total_decisions += 50000

        n = h.evict_stale_entries(max_age=100)  # age > 100 + hit_count < 10
        assert n == 1, f"Expected 1 eviction, got {n}"
        assert len(h.cache) == 0

    def test_high_hit_entries_survive_eviction(self):
        """Entries with hit_count ≥ 10 survive even if old."""
        h = HardeningEngine(threshold=3, min_confidence=0.01)
        # Create cache entry
        for _ in range(6):
            h.should_use_fast_path(0, ROUTING_CONSISTENT)
        # Hit the cache many times
        for _ in range(20):
            if h.should_use_fast_path(0, ROUTING_CONSISTENT):
                h.get_cached_weights(0, ROUTING_CONSISTENT)
        # Make it old
        h.total_decisions += 50000
        n = h.evict_stale_entries(max_age=100)
        assert n == 0  # hit_count >= 10, survives
        assert len(h.cache) == 1


# ── Statistics ────────────────────────────────────────────────────────

class TestStatistics:
    def test_get_stats_keys(self, hardening):
        stats = hardening.get_stats()
        expected_keys = {
            "total_decisions", "n_cached_patterns", "n_fast_path",
            "n_slow_path", "n_degradations", "fast_path_ratio",
            "baseline_entropy", "cache_hit_rate",
        }
        assert set(stats.keys()) == expected_keys

    def test_stats_values_consistent(self, hardening):
        for _ in range(50):
            hardening.should_use_fast_path(0, ROUTING_CONSISTENT)
        stats = hardening.get_stats()
        assert stats["total_decisions"] == 50
        assert stats["n_fast_path"] + stats["n_slow_path"] == 50
        assert 0.0 <= stats["fast_path_ratio"] <= 1.0

    def test_n_degradations_present(self, hardening):
        for _ in range(100):
            hardening.should_use_fast_path(0, ROUTING_LOW_ENTROPY)
        for _ in range(30):
            hardening.should_use_fast_path(0, ROUTING_HIGH_ENTROPY)
        hardening.detect_regime_shift()
        stats = hardening.get_stats()
        assert "n_degradations" in stats


# ── Edge cases ────────────────────────────────────────────────────────

class TestEdgeCases:
    def test_single_decision(self, hardening):
        hardening.should_use_fast_path(0, ROUTING_CONSISTENT)
        stats = hardening.get_stats()
        assert stats["total_decisions"] == 1

    def test_deterministic_routing(self, hardening):
        """One-hot routing (zero entropy) should not cause issues."""
        r = torch.zeros(8)
        r[0] = 1.0
        for _ in range(10):
            hardening.should_use_fast_path(0, r)
        stats = hardening.get_stats()
        assert stats["total_decisions"] == 10

    def test_cached_weights_frozen(self):
        """Cached weights are immutable across subsequent decisions."""
        h = HardeningEngine(threshold=3, min_confidence=0.01)
        for _ in range(8):
            h.should_use_fast_path(0, ROUTING_CONSISTENT)
        w1 = h.get_cached_weights(0, ROUTING_CONSISTENT)
        # More decisions — cached weights should be unchanged
        for _ in range(5):
            h.should_use_fast_path(0, ROUTING_CONSISTENT)
        w2 = h.get_cached_weights(0, ROUTING_CONSISTENT)
        assert torch.equal(w1, w2)
