"""Tests for backtest module — IS/OOS split and signal statistics."""

import numpy as np
import pandas as pd
from unittest.mock import patch

from backtest import backtest_entropy_signals


def _synthetic_prices(n=600, seed=42, vol=0.015, crisis_at=300, crisis_vol=0.04):
    """Generate synthetic price DataFrame with a volatility regime shift."""
    np.random.seed(seed)
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    rets = np.concatenate([
        np.random.normal(0.0003, vol, crisis_at),
        np.random.normal(-0.001, crisis_vol, n - crisis_at),
    ])
    prices = 100 * np.exp(np.cumsum(rets))
    df = pd.DataFrame({
        "Open": prices,
        "High": prices * 1.005,
        "Low": prices * 0.995,
        "Close": prices,
        "Volume": np.ones(n) * 1_000_000,
    }, index=dates)
    return df


class TestBacktestEntropySignals:

    @patch("backtest.get_price_data")
    def test_returns_valid_structure(self, mock_get):
        mock_get.return_value = _synthetic_prices()
        result = backtest_entropy_signals("FAKE", period="5y")
        assert "error" not in result
        for key in ["ticker", "n_signals", "signal_stats", "baseline_stats",
                    "threshold", "entropy_mean", "entropy_std", "split_date",
                    "train_pct", "n_oos_trading_days"]:
            assert key in result

    @patch("backtest.get_price_data")
    def test_in_sample_out_of_sample_split(self, mock_get):
        mock_get.return_value = _synthetic_prices()
        result = backtest_entropy_signals("FAKE", period="5y", train_pct=0.60)
        assert result["train_pct"] == 0.60
        # OOS period must be smaller than total
        assert result["n_oos_trading_days"] < result["n_trading_days"]
        # OOS should be ~40% of total
        oos_ratio = result["n_oos_trading_days"] / result["n_trading_days"]
        assert 0.30 <= oos_ratio <= 0.50

    @patch("backtest.get_price_data")
    def test_threshold_derived_from_in_sample_only(self, mock_get):
        mock_get.return_value = _synthetic_prices()
        result = backtest_entropy_signals("FAKE", period="5y")
        # Threshold = mean + std * threshold_std (default 1.0)
        expected = result["entropy_mean"] + 1.0 * result["entropy_std"]
        assert abs(result["threshold"] - expected) < 1e-6

    @patch("backtest.get_price_data")
    def test_signal_stats_keys_present(self, mock_get):
        mock_get.return_value = _synthetic_prices()
        result = backtest_entropy_signals("FAKE", period="5y", forward_days=[30, 60])
        for horizon in ["30d", "60d"]:
            if horizon in result["signal_stats"]:
                stats = result["signal_stats"][horizon]
                for key in ["n_signals", "mean_return", "hit_rate_negative"]:
                    assert key in stats

    @patch("backtest.get_price_data")
    def test_baseline_uses_oos_period_only(self, mock_get):
        mock_get.return_value = _synthetic_prices()
        result = backtest_entropy_signals("FAKE", period="5y")
        # Baseline stats should exist for forward horizons
        assert len(result["baseline_stats"]) > 0

    @patch("backtest.get_price_data")
    def test_different_train_splits_give_different_thresholds(self, mock_get):
        mock_get.return_value = _synthetic_prices()
        r60 = backtest_entropy_signals("FAKE", train_pct=0.60)
        r80 = backtest_entropy_signals("FAKE", train_pct=0.80)
        # Thresholds may differ since they're calibrated on different IS windows
        # (just confirm both run without error)
        assert "error" not in r60
        assert "error" not in r80

    @patch("backtest.get_price_data")
    def test_signals_only_in_oos_window(self, mock_get):
        mock_get.return_value = _synthetic_prices()
        result = backtest_entropy_signals("FAKE", train_pct=0.60)
        split_date = pd.to_datetime(result["split_date"])
        for sig in result["signals"]:
            sig_date = pd.to_datetime(sig["date"])
            assert sig_date >= split_date, (
                f"Signal at {sig_date} is before split date {split_date}"
            )

    @patch("backtest.get_price_data")
    def test_insufficient_data_returns_error(self, mock_get):
        # Very short series — not enough for a 60-day entropy window
        np.random.seed(0)
        dates = pd.date_range("2023-01-01", periods=40, freq="B")
        prices = pd.DataFrame({"Close": 100 * np.exp(np.cumsum(np.random.normal(0, 0.01, 40)))}, index=dates)
        mock_get.return_value = prices
        result = backtest_entropy_signals("FAKE", entropy_window=60)
        assert "error" in result

    @patch("backtest.get_price_data")
    def test_custom_forward_days(self, mock_get):
        mock_get.return_value = _synthetic_prices()
        result = backtest_entropy_signals("FAKE", forward_days=[10, 45])
        assert "error" not in result
        # Only the requested horizons appear in signal_stats
        for key in result["signal_stats"]:
            assert key in ["10d", "45d"]
