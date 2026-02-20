"""Accuracy Testing — cross-stock validation of the entropy framework.

Runs entropy analysis and backtests across multiple stocks to determine:
1. Does the entropy signal generalize beyond a single ticker?
2. How does entropy-adjusted DCF compare to analyst consensus?
3. Is the signal statistically significant?
"""

import numpy as np
import pandas as pd
import yfinance as yf
from entropy_radar import run_entropy_radar
from valuation import get_company_info, compute_dcf, entropy_adjusted_wacc
from backtest import backtest_entropy_signals


# Default test universe — diversified across sectors
DEFAULT_UNIVERSE = [
    "AAPL", "MSFT", "NVDA",   # Tech
    "JPM", "GS",              # Financials
    "JNJ", "UNH",             # Healthcare
    "XOM", "CVX",             # Energy
    "AMZN", "TSLA",           # Consumer
]


def run_cross_stock_backtest(tickers: list[str] | None = None,
                             period: str = "5y",
                             threshold_std: float = 1.0) -> dict:
    """Run entropy signal backtest across multiple stocks.

    Tests whether high-entropy signals predict worse forward returns
    across a diversified basket, not just one ticker.

    Returns:
        Dict with per-stock results, aggregate statistics, and
        a determination of whether the signal generalizes.
    """
    if tickers is None:
        tickers = DEFAULT_UNIVERSE

    results = []
    errors = []

    for tkr in tickers:
        try:
            bt = backtest_entropy_signals(tkr, period=period,
                                          threshold_std=threshold_std)
            if "error" in bt:
                errors.append({"ticker": tkr, "error": bt["error"]})
                continue

            row = {"ticker": tkr, "n_signals": bt["n_signals"]}

            for horizon in ["30d", "60d", "90d"]:
                sig = bt["signal_stats"].get(horizon, {})
                base = bt["baseline_stats"].get(horizon, {})
                if sig and base:
                    row[f"signal_return_{horizon}"] = sig["mean_return"]
                    row[f"baseline_return_{horizon}"] = base["mean_return"]
                    row[f"edge_{horizon}"] = sig["mean_return"] - base["mean_return"]
                    row[f"hit_rate_{horizon}"] = sig["hit_rate_negative"]

            results.append(row)
        except Exception as e:
            errors.append({"ticker": tkr, "error": str(e)})

    if not results:
        return {"error": "No stocks could be analyzed", "errors": errors}

    df = pd.DataFrame(results)

    # Aggregate statistics
    aggregate = {}
    for horizon in ["30d", "60d", "90d"]:
        edge_col = f"edge_{horizon}"
        if edge_col in df.columns:
            edges = df[edge_col].dropna()
            n_negative_edge = (edges < 0).sum()  # stocks where signal underperforms
            aggregate[horizon] = {
                "mean_edge": round(float(edges.mean()), 2),
                "median_edge": round(float(edges.median()), 2),
                "stocks_with_negative_edge": int(n_negative_edge),
                "total_stocks": len(edges),
                "pct_confirming": round(n_negative_edge / len(edges) * 100, 1) if len(edges) > 0 else 0,
                "std_edge": round(float(edges.std()), 2),
            }
            # Simple t-test: is the mean edge significantly < 0?
            if len(edges) > 1 and edges.std() > 0:
                t_stat = float(edges.mean() / (edges.std() / np.sqrt(len(edges))))
                aggregate[horizon]["t_statistic"] = round(t_stat, 2)

    return {
        "results": results,
        "results_df": df,
        "aggregate": aggregate,
        "n_stocks": len(results),
        "n_errors": len(errors),
        "errors": errors,
    }


def run_analyst_comparison(tickers: list[str] | None = None,
                           base_wacc: float = 0.10,
                           terminal_growth: float = 0.03) -> dict:
    """Compare standard DCF, entropy-adjusted DCF, and analyst targets.

    For each stock, computes:
    - Standard DCF intrinsic value
    - Entropy-adjusted DCF intrinsic value
    - Analyst consensus target price (from yfinance)
    - Current market price

    Returns which method is closer to analyst consensus.
    """
    if tickers is None:
        tickers = DEFAULT_UNIVERSE

    results = []

    for tkr in tickers:
        try:
            info = get_company_info(tkr)
            t = yf.Ticker(tkr)
            analyst_target = t.info.get("targetMeanPrice")

            dcf_std = compute_dcf(tkr, discount_rate=base_wacc,
                                  terminal_growth=terminal_growth)

            # Get entropy score for adjustment
            radar = run_entropy_radar(tkr, info.get("sector", ""), "1y")
            es_score = radar["entropy_score"]["composite_score"]
            wacc_adj = entropy_adjusted_wacc(base_wacc, es_score)
            dcf_adj = compute_dcf(tkr,
                                  discount_rate=wacc_adj["adjusted_wacc"] / 100,
                                  terminal_growth=terminal_growth)

            current_price = info.get("current_price")

            row = {
                "ticker": tkr,
                "current_price": current_price,
                "analyst_target": analyst_target,
                "entropy_score": es_score,
            }

            if "error" not in dcf_std:
                row["dcf_standard"] = dcf_std["intrinsic_value"]
            if "error" not in dcf_adj:
                row["dcf_entropy_adjusted"] = dcf_adj["intrinsic_value"]

            # Compute which is closer to analyst target
            if analyst_target and "dcf_standard" in row and "dcf_entropy_adjusted" in row:
                std_error = abs(row["dcf_standard"] - analyst_target)
                adj_error = abs(row["dcf_entropy_adjusted"] - analyst_target)
                row["closer_to_analyst"] = "Entropy-Adjusted" if adj_error < std_error else "Standard"
                row["std_error_vs_analyst"] = round(std_error, 2)
                row["adj_error_vs_analyst"] = round(adj_error, 2)

            results.append(row)
        except Exception:
            continue

    if not results:
        return {"error": "No stocks could be analyzed"}

    df = pd.DataFrame(results)

    # Summary: which method is closer more often?
    if "closer_to_analyst" in df.columns:
        counts = df["closer_to_analyst"].value_counts().to_dict()
    else:
        counts = {}

    return {
        "results": results,
        "results_df": df,
        "method_comparison": counts,
        "n_stocks": len(results),
    }
