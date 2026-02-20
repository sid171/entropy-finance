"""Sector Entropy Heatmap — real-time entropy scores across the stock universe."""

import pandas as pd
from entropy_radar import run_entropy_radar
from valuation import STOCK_UNIVERSE
import yfinance as yf


def compute_sector_heatmap(period: str = "1y", max_per_sector: int = 5) -> dict:
    """Compute entropy scores for stocks across all sectors.

    Args:
        period: Analysis period for entropy computation.
        max_per_sector: Max stocks to analyze per sector (for speed).

    Returns:
        Dict with per-stock scores, sector averages, and sorted rankings.
    """
    results = []
    errors = []

    for sector, tickers in STOCK_UNIVERSE.items():
        subset = tickers[:max_per_sector]
        for tkr in subset:
            try:
                info = yf.Ticker(tkr).info
                stock_sector = info.get("sector", sector)
                radar = run_entropy_radar(tkr, stock_sector, period)
                es = radar["entropy_score"]

                results.append({
                    "ticker": tkr,
                    "sector": sector,
                    "name": info.get("longName", tkr),
                    "composite_score": es["composite_score"],
                    "regime_instability": es["regime_instability"],
                    "information_flow": es["information_flow"],
                    "relationship_stress": es["relationship_stress"],
                    "uncertainty": es["uncertainty"],
                    "interpretation": es["interpretation"],
                })
            except Exception as e:
                errors.append({"ticker": tkr, "sector": sector, "error": str(e)})

    if not results:
        return {"error": "No stocks could be analyzed", "errors": errors}

    df = pd.DataFrame(results)

    # Sector averages
    sector_avg = df.groupby("sector")["composite_score"].mean().round(1).sort_values(ascending=False)

    # Top risk stocks
    top_risk = df.nlargest(10, "composite_score")

    # Most stable stocks
    most_stable = df.nsmallest(10, "composite_score")

    return {
        "results": results,
        "results_df": df,
        "sector_averages": sector_avg.to_dict(),
        "top_risk": top_risk.to_dict("records"),
        "most_stable": most_stable.to_dict("records"),
        "n_stocks": len(results),
        "n_errors": len(errors),
        "errors": errors,
    }
