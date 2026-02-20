"""Tests for entropy_tools module."""

import numpy as np
import pandas as pd
import pytest

from entropy_tools import (
    shannon_entropy,
    transfer_entropy,
    net_transfer_entropy,
    rolling_entropy,
    detect_regimes,
    correlation_stability,
)


# --- Fixtures ---

@pytest.fixture
def random_returns():
    """Generate random normal returns."""
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", periods=500, freq="B")
    return pd.Series(np.random.normal(0, 0.02, 500), index=dates, name="RANDOM")


@pytest.fixture
def trending_returns():
    """Generate returns with a clear regime shift midway."""
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", periods=500, freq="B")
    low_vol = np.random.normal(0.001, 0.005, 250)
    high_vol = np.random.normal(-0.002, 0.03, 250)
    return pd.Series(np.concatenate([low_vol, high_vol]), index=dates, name="TREND")


@pytest.fixture
def leader_follower():
    """Generate two series where A leads B with a lag."""
    np.random.seed(42)
    n = 500
    dates = pd.date_range("2023-01-01", periods=n, freq="B")
    a = np.random.normal(0, 0.02, n)
    # B follows A with 1-day lag + noise
    b = np.zeros(n)
    b[1:] = 0.7 * a[:-1] + 0.3 * np.random.normal(0, 0.02, n - 1)
    return (
        pd.Series(a, index=dates, name="LEADER"),
        pd.Series(b, index=dates, name="FOLLOWER"),
    )


# --- Shannon Entropy ---

class TestShannonEntropy:
    def test_returns_positive_float(self, random_returns):
        result = shannon_entropy(random_returns)
        assert isinstance(result, float)
        assert result > 0

    def test_uniform_has_high_entropy(self):
        """Uniform distribution should have higher entropy than concentrated."""
        np.random.seed(42)
        uniform = pd.Series(np.random.uniform(-0.05, 0.05, 1000))
        concentrated = pd.Series(np.random.normal(0, 0.001, 1000))
        assert shannon_entropy(uniform) > shannon_entropy(concentrated)

    def test_empty_series_handled(self):
        result = shannon_entropy(pd.Series(dtype=float), bins=10)
        assert result == 0.0 or np.isfinite(result)


# --- Transfer Entropy ---

class TestTransferEntropy:
    def test_returns_non_negative(self, random_returns):
        te = transfer_entropy(random_returns, random_returns)
        assert te >= 0

    def test_leader_has_higher_te(self, leader_follower):
        leader, follower = leader_follower
        te_lead_to_follow = transfer_entropy(leader, follower)
        te_follow_to_lead = transfer_entropy(follower, leader)
        # Leader should transfer more information to follower
        assert te_lead_to_follow > te_follow_to_lead

    def test_net_transfer_entropy_identifies_leader(self, leader_follower):
        leader, follower = leader_follower
        result = net_transfer_entropy(leader, follower)
        assert "LEADER leads FOLLOWER" in result["leader"]
        assert result["net_te"] > 0


# --- Rolling Entropy ---

class TestRollingEntropy:
    def test_returns_series(self, random_returns):
        result = rolling_entropy(random_returns, window=60)
        assert isinstance(result, pd.Series)
        assert len(result) > 0

    def test_regime_shift_shows_entropy_change(self, trending_returns):
        result = rolling_entropy(trending_returns, window=60)
        # Entropy should be higher in the volatile regime (second half)
        midpoint = len(result) // 2
        if midpoint > 0 and len(result) > midpoint:
            early = result.iloc[:midpoint].mean()
            late = result.iloc[midpoint:].mean()
            assert late > early


# --- Regime Detection ---

class TestRegimeDetection:
    def test_detects_changepoints(self, trending_returns):
        result = detect_regimes(trending_returns)
        assert result["n_changepoints"] > 0
        assert len(result["regimes"]) >= 2

    def test_returns_valid_structure(self, random_returns):
        result = detect_regimes(random_returns)
        assert "n_changepoints" in result
        assert "changepoint_dates" in result
        assert "regimes" in result
        for regime in result["regimes"]:
            assert "mean_return" in regime
            assert "volatility" in regime


# --- Correlation Stability ---

class TestCorrelationStability:
    def test_correlated_series_not_collapsed(self):
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", periods=300, freq="B")
        a = pd.Series(np.random.normal(0, 0.02, 300), index=dates, name="A")
        b = 0.8 * a + pd.Series(np.random.normal(0, 0.005, 300), index=dates)
        b.name = "B"
        result = correlation_stability(a, b, window=60)
        assert not result["is_collapsed"]

    def test_uncorrelated_series_collapsed(self):
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", periods=300, freq="B")
        a = pd.Series(np.random.normal(0, 0.02, 300), index=dates, name="A")
        b = pd.Series(np.random.normal(0, 0.02, 300), index=dates, name="B")
        result = correlation_stability(a, b, window=60)
        # Truly independent series should show near-zero correlation
        assert abs(result["mean_recent_correlation"]) < 0.3
