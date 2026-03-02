"""Tests for valuation module — entropy_adjusted_wacc and DCF logic."""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch

from valuation import entropy_adjusted_wacc, _estimate_wacc_std


class TestEntropyAdjustedWACC:
    """Tests for entropy_adjusted_wacc — pure logic, no network required."""

    def test_zero_entropy_no_premium(self):
        result = entropy_adjusted_wacc(0.10, 0.0)
        assert result["entropy_premium"] == 0.0
        assert result["adjusted_wacc"] == pytest.approx(result["base_wacc"], abs=0.01)

    def test_high_entropy_adds_premium(self):
        low = entropy_adjusted_wacc(0.10, 10.0)
        high = entropy_adjusted_wacc(0.10, 90.0)
        assert high["adjusted_wacc"] > low["adjusted_wacc"]

    def test_adjusted_wacc_never_below_base(self):
        for score in [0, 25, 50, 75, 100]:
            result = entropy_adjusted_wacc(0.10, score)
            assert result["adjusted_wacc"] >= result["base_wacc"]

    def test_max_premium_bounded_by_max_premium_param(self):
        result = entropy_adjusted_wacc(0.10, 100.0, max_premium=0.06)
        premium = result["entropy_premium"] / 100  # convert back to fraction
        assert premium <= 0.06 + 1e-9

    def test_returns_correct_keys(self):
        result = entropy_adjusted_wacc(0.10, 50.0)
        for key in ["base_wacc", "entropy_premium", "adjusted_wacc", "entropy_score", "rationale"]:
            assert key in result

    def test_rationale_high_entropy(self):
        result = entropy_adjusted_wacc(0.10, 80.0)
        assert "High entropy" in result["rationale"]

    def test_rationale_low_entropy(self):
        result = entropy_adjusted_wacc(0.10, 15.0)
        assert "Low entropy" in result["rationale"] or "Very low" in result["rationale"]

    def test_entropy_score_stored_in_result(self):
        result = entropy_adjusted_wacc(0.10, 42.5)
        assert result["entropy_score"] == pytest.approx(42.5, abs=0.1)

    def test_custom_max_premium(self):
        result_low = entropy_adjusted_wacc(0.10, 100.0, max_premium=0.02)
        result_high = entropy_adjusted_wacc(0.10, 100.0, max_premium=0.08)
        assert result_high["adjusted_wacc"] > result_low["adjusted_wacc"]

    def test_base_wacc_stored_as_percentage(self):
        result = entropy_adjusted_wacc(0.10, 50.0)
        assert result["base_wacc"] == pytest.approx(10.0, abs=0.01)


class TestEstimateWACCStd:
    """Tests for _estimate_wacc_std — mocked yfinance to avoid network calls."""

    def _make_price_series(self, n=504, seed=42, drift=0.0001, vol=0.015):
        """Generate synthetic daily price series."""
        np.random.seed(seed)
        returns = np.random.normal(drift, vol, n)
        prices = 100 * np.exp(np.cumsum(returns))
        dates = pd.date_range("2022-01-01", periods=n, freq="B")
        return pd.DataFrame({"Close": prices}, index=dates)

    @patch("valuation.yf.download")
    def test_returns_value_in_valid_range(self, mock_download):
        prices = self._make_price_series()
        mock_download.return_value = prices
        info = {"marketCap": 1e12, "totalDebt": 2e11}
        std = _estimate_wacc_std("AAPL", info, 0.10)
        assert 0.005 <= std <= 0.04

    @patch("valuation.yf.download")
    def test_higher_beta_vol_gives_higher_std(self, mock_download):
        # High-vol stock vs low-vol stock
        low_vol = self._make_price_series(vol=0.008)
        high_vol = self._make_price_series(vol=0.035)

        info = {"marketCap": 1e12, "totalDebt": 2e11}

        # First call: stock prices (low vol), second: SPY, third: TNX
        mock_download.side_effect = [low_vol, low_vol, low_vol]
        std_low = _estimate_wacc_std("AAPL", info, 0.10)

        mock_download.side_effect = [high_vol, low_vol, low_vol]
        std_high = _estimate_wacc_std("TSLA", info, 0.10)

        assert std_high >= std_low

    @patch("valuation.yf.download")
    def test_empty_data_falls_back_gracefully(self, mock_download):
        mock_download.return_value = pd.DataFrame()
        info = {"marketCap": 1e12, "totalDebt": 2e11}
        std = _estimate_wacc_std("FAKE", info, 0.10)
        assert 0.005 <= std <= 0.04

    @patch("valuation.yf.download")
    def test_missing_market_cap_uses_fallback_equity_weight(self, mock_download):
        prices = self._make_price_series()
        mock_download.return_value = prices
        info = {}  # no marketCap or totalDebt
        std = _estimate_wacc_std("AAPL", info, 0.10)
        assert 0.005 <= std <= 0.04

    @patch("valuation.yf.download")
    def test_returns_float(self, mock_download):
        prices = self._make_price_series()
        mock_download.return_value = prices
        info = {"marketCap": 5e11, "totalDebt": 1e11}
        std = _estimate_wacc_std("MSFT", info, 0.09)
        assert isinstance(std, float)
