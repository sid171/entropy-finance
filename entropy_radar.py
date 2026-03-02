"""Entropy Radar — composite entropy profile for any stock.

Synthesizes entropy concepts from information theory into a single
actionable intelligence layer:
- Is the regime around this stock changing?
- What drives this stock, and what does it drive?
- Are the relationships this stock depends on still intact?
- Composite entropy score: how much is "the game" changing?
"""

import numpy as np
import pandas as pd
from market_data import get_returns, get_price_data
from entropy_tools import (
    shannon_entropy,
    rolling_entropy,
    detect_regimes,
    net_transfer_entropy,
    correlation_stability,
)
from entropy_calibration import (
    normalize_shannon_entropy,
    normalize_transfer_entropy,
    normalize_changepoint_count,
    normalize_correlation_stress,
)

# Sector ETF mapping for automatic peer selection
SECTOR_ETFS = {
    "Technology": "XLK",
    "Healthcare": "XLV",
    "Financial Services": "XLF",
    "Financials": "XLF",
    "Consumer Cyclical": "XLY",
    "Consumer Defensive": "XLP",
    "Communication Services": "XLC",
    "Industrials": "XLI",
    "Energy": "XLE",
    "Utilities": "XLU",
    "Real Estate": "XLRE",
    "Basic Materials": "XLB",
}


def get_comparison_tickers(sector: str) -> list[str]:
    """Get comparison tickers for entropy radar based on sector."""
    comparisons = ["SPY"]  # Always compare to market
    sector_etf = SECTOR_ETFS.get(sector)
    if sector_etf:
        comparisons.append(sector_etf)
    return comparisons


def compute_entropy_score(
    regime_score: float,
    information_flow_score: float,
    relationship_health_score: float,
    uncertainty_score: float,
) -> dict:
    """Compute composite entropy score (0-100).

    Higher = more entropy disruption = the game is changing.
    Lower = stable regime = business as usual.

    Components (each 0-100):
    - Regime instability: based on number and recency of changepoints
    - Information flow: how strongly this asset leads/follows others
    - Relationship stress: how unstable correlations are
    - Uncertainty level: how high Shannon entropy is vs baseline
    """
    weights = {
        "regime_instability": 0.30,
        "information_flow": 0.20,
        "relationship_stress": 0.25,
        "uncertainty": 0.25,
    }

    composite = (
        weights["regime_instability"] * regime_score
        + weights["information_flow"] * information_flow_score
        + weights["relationship_stress"] * relationship_health_score
        + weights["uncertainty"] * uncertainty_score
    )

    if composite > 70:
        interpretation = "HIGH ENTROPY — The rules around this asset are actively changing. Elevated risk of structural break."
    elif composite > 45:
        interpretation = "MODERATE ENTROPY — Some unusual dynamics detected. Monitor for regime transition."
    elif composite > 25:
        interpretation = "LOW ENTROPY — Relatively stable regime. Normal market dynamics."
    else:
        interpretation = "VERY LOW ENTROPY — Highly stable. Predictable behavior within current regime."

    return {
        "composite_score": round(composite, 1),
        "regime_instability": round(regime_score, 1),
        "information_flow": round(information_flow_score, 1),
        "relationship_stress": round(relationship_health_score, 1),
        "uncertainty": round(uncertainty_score, 1),
        "interpretation": interpretation,
    }


def run_entropy_radar(ticker: str, sector: str, period: str = "1y") -> dict:
    """Run full entropy radar analysis for a ticker.

    Returns a comprehensive entropy profile with:
    - Shannon entropy + rolling entropy
    - Regime detection
    - Transfer entropy against market & sector
    - Correlation stability
    - Composite entropy score
    """
    results = {"ticker": ticker, "period": period}

    # 1. Get returns
    returns = get_returns(ticker, period)
    results["n_days"] = len(returns)

    # 2. Shannon entropy
    ent = shannon_entropy(returns)
    results["shannon_entropy"] = round(ent, 4)

    # Uncertainty score: empirically calibrated from p5/p95 across 57 stocks.
    # Range [2.68, 3.72] nats maps to [0, 100]. See entropy_calibration.py.
    uncertainty_score = normalize_shannon_entropy(ent)

    # 3. Rolling entropy
    roll_ent = rolling_entropy(returns, window=60)
    results["rolling_entropy"] = roll_ent
    if len(roll_ent) > 0:
        results["current_rolling_entropy"] = round(float(roll_ent.iloc[-1]), 4)
        results["mean_rolling_entropy"] = round(float(roll_ent.mean()), 4)
        results["max_rolling_entropy"] = round(float(roll_ent.max()), 4)
        results["max_entropy_date"] = roll_ent.idxmax()

    # 4. Regime detection
    regimes = detect_regimes(returns)
    results["regimes"] = regimes

    # Regime score: empirically calibrated from p5/p95 across 57 stocks.
    # Observed range [1, 11] changepoints maps to [0, 80], leaving 20 points
    # for recency bonus. See entropy_calibration.py.
    n_cp = regimes["n_changepoints"]
    regime_score = normalize_changepoint_count(n_cp)
    # Recency bonus: recent changepoints signal active regime transitions
    if regimes["changepoint_dates"]:
        last_cp = pd.to_datetime(regimes["changepoint_dates"][-1])
        days_since = (returns.index[-1] - last_cp).days
        if days_since < 30:
            regime_score = min(100, regime_score + 20)
        elif days_since < 60:
            regime_score = min(100, regime_score + 10)

    # 5. Transfer entropy against comparisons
    comparisons = get_comparison_tickers(sector)
    te_results = []
    information_flow_score = 0

    for comp_ticker in comparisons:
        try:
            comp_returns = get_returns(comp_ticker, period)
            comp_returns.name = comp_ticker
            returns_named = returns.copy()
            returns_named.name = ticker

            te = net_transfer_entropy(returns_named, comp_returns, lag=1, bins=10)
            te["comparison"] = comp_ticker
            te_results.append(te)

            # Information flow score: empirically calibrated from p5/p95 of
            # |net_te| across 57 stocks vs SPY/sector ETF. See entropy_calibration.py.
            asymmetry = abs(te["net_te"])
            information_flow_score = max(
                information_flow_score,
                normalize_transfer_entropy(asymmetry),
            )
        except Exception:
            continue

    results["transfer_entropy"] = te_results

    # 6. Correlation stability against comparisons
    corr_results = []
    relationship_stress_score = 0

    for comp_ticker in comparisons:
        try:
            comp_returns = get_returns(comp_ticker, period)
            corr = correlation_stability(returns, comp_returns, window=60)
            rolling_corr_data = corr.pop("rolling_correlation")
            corr["comparison"] = comp_ticker
            corr["rolling_data"] = rolling_corr_data
            corr_results.append(corr)

            # Stress score: empirically calibrated from p5/p95 of abs rolling
            # correlation across 57 stocks. See entropy_calibration.py.
            if corr["is_collapsed"]:
                relationship_stress_score = max(relationship_stress_score, 90)
            elif corr["mean_recent_correlation"] is not None:
                abs_corr = abs(corr["mean_recent_correlation"])
                stress = normalize_correlation_stress(abs_corr)
                relationship_stress_score = max(relationship_stress_score, stress)
        except Exception:
            continue

    results["correlation_stability"] = corr_results

    # 7. Composite score
    results["entropy_score"] = compute_entropy_score(
        regime_score=regime_score,
        information_flow_score=information_flow_score,
        relationship_health_score=relationship_stress_score,
        uncertainty_score=uncertainty_score,
    )

    return results
