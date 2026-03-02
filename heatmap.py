"""Sector Entropy Heatmap — real-time entropy scores across the stock universe."""

import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from entropy_radar import run_entropy_radar
from valuation import STOCK_UNIVERSE
import yfinance as yf


def _fetch_stock_entropy(tkr: str, sector: str, period: str) -> dict:
    """Fetch entropy radar for a single stock. Designed for thread-pool use."""
    info = yf.Ticker(tkr).info
    stock_sector = info.get("sector", sector)
    radar = run_entropy_radar(tkr, stock_sector, period)
    es = radar["entropy_score"]
    return {
        "ticker": tkr,
        "sector": sector,
        "name": info.get("longName", tkr),
        "composite_score": es["composite_score"],
        "regime_instability": es["regime_instability"],
        "information_flow": es["information_flow"],
        "relationship_stress": es["relationship_stress"],
        "uncertainty": es["uncertainty"],
        "interpretation": es["interpretation"],
    }


def compute_sector_heatmap(period: str = "1y", max_per_sector: int = 5,
                           max_workers: int = 8) -> dict:
    """Compute entropy scores for stocks across all sectors using parallel fetching.

    Args:
        period: Analysis period for entropy computation.
        max_per_sector: Max stocks to analyze per sector (for speed).
        max_workers: Thread-pool size for parallel yfinance fetches.

    Returns:
        Dict with per-stock scores, sector averages, and sorted rankings.
    """
    results = []
    errors = []

    # Build flat list of (ticker, sector) pairs to analyze
    tasks = []
    for sector, tickers in STOCK_UNIVERSE.items():
        for tkr in tickers[:max_per_sector]:
            tasks.append((tkr, sector))

    # Parallel fetch with ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_task = {
            executor.submit(_fetch_stock_entropy, tkr, sector, period): (tkr, sector)
            for tkr, sector in tasks
        }
        for future in as_completed(future_to_task):
            tkr, sector = future_to_task[future]
            try:
                results.append(future.result())
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
