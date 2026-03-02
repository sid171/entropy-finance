"""Tests for entropy_calibration module — empirical normalization functions."""

from entropy_calibration import (
    normalize_shannon_entropy,
    normalize_transfer_entropy,
    normalize_changepoint_count,
    normalize_correlation_stress,
    SHANNON_ENTROPY_P5,
    SHANNON_ENTROPY_P95,
    TRANSFER_ENTROPY_P5,
    TRANSFER_ENTROPY_P95,
    CHANGEPOINT_P5,
    CHANGEPOINT_P95,
    CORRELATION_P5,
    CORRELATION_P95,
)


class TestShannonEntropyNormalization:
    def test_p5_maps_to_zero(self):
        assert normalize_shannon_entropy(SHANNON_ENTROPY_P5) == 0.0

    def test_p95_maps_to_hundred(self):
        assert normalize_shannon_entropy(SHANNON_ENTROPY_P95) == 100.0

    def test_midpoint_maps_near_fifty(self):
        mid = (SHANNON_ENTROPY_P5 + SHANNON_ENTROPY_P95) / 2
        score = normalize_shannon_entropy(mid)
        assert 45 <= score <= 55

    def test_below_p5_clamped_to_zero(self):
        assert normalize_shannon_entropy(0.0) == 0.0

    def test_above_p95_clamped_to_hundred(self):
        assert normalize_shannon_entropy(99.0) == 100.0

    def test_monotonically_increasing(self):
        values = [2.5, 2.8, 3.0, 3.2, 3.5, 3.8]
        scores = [normalize_shannon_entropy(v) for v in values]
        assert scores == sorted(scores)

    def test_returns_float(self):
        assert isinstance(normalize_shannon_entropy(3.0), float)


class TestTransferEntropyNormalization:
    def test_p5_maps_to_zero(self):
        assert normalize_transfer_entropy(TRANSFER_ENTROPY_P5) == 0.0

    def test_p95_maps_to_hundred(self):
        assert normalize_transfer_entropy(TRANSFER_ENTROPY_P95) == 100.0

    def test_zero_clamped_to_zero(self):
        assert normalize_transfer_entropy(0.0) == 0.0

    def test_large_value_clamped_to_hundred(self):
        assert normalize_transfer_entropy(1.0) == 100.0

    def test_monotonically_increasing(self):
        values = [0.0001, 0.001, 0.002, 0.004, 0.006, 0.008]
        scores = [normalize_transfer_entropy(v) for v in values]
        assert scores == sorted(scores)

    def test_returns_float(self):
        assert isinstance(normalize_transfer_entropy(0.003), float)


class TestChangepointNormalization:
    def test_p5_maps_to_zero(self):
        assert normalize_changepoint_count(CHANGEPOINT_P5) == 0.0

    def test_p95_capped_at_eighty(self):
        # Cap at 80 to leave room for recency bonus
        assert normalize_changepoint_count(CHANGEPOINT_P95) == 80.0

    def test_zero_clamped_to_zero(self):
        assert normalize_changepoint_count(0) == 0.0

    def test_large_count_capped_at_eighty(self):
        assert normalize_changepoint_count(100) == 80.0

    def test_monotonically_increasing(self):
        scores = [normalize_changepoint_count(n) for n in range(1, 12)]
        assert scores == sorted(scores)

    def test_returns_float(self):
        assert isinstance(normalize_changepoint_count(5), float)


class TestCorrelationStressNormalization:
    def test_high_correlation_low_stress(self):
        # p95 correlation → 0 stress
        stress = normalize_correlation_stress(CORRELATION_P95)
        assert stress == 0.0

    def test_low_correlation_high_stress(self):
        # p5 correlation → 100 stress
        stress = normalize_correlation_stress(CORRELATION_P5)
        assert stress == 100.0

    def test_midpoint_near_fifty(self):
        mid = (CORRELATION_P5 + CORRELATION_P95) / 2
        stress = normalize_correlation_stress(mid)
        assert 45 <= stress <= 55

    def test_above_p95_clamped_to_zero(self):
        assert normalize_correlation_stress(1.0) == 0.0

    def test_below_p5_clamped_to_hundred(self):
        assert normalize_correlation_stress(0.0) == 100.0

    def test_monotonically_decreasing(self):
        # Higher correlation → lower stress
        correlations = [0.2, 0.4, 0.6, 0.8]
        stresses = [normalize_correlation_stress(c) for c in correlations]
        assert stresses == sorted(stresses, reverse=True)

    def test_returns_float(self):
        assert isinstance(normalize_correlation_stress(0.5), float)
