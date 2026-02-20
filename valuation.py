"""Valuation analysis tools — DCF, key ratios, peer comparison, and historical trends."""

import yfinance as yf
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Stock universe for peer matching (~100 major stocks across sectors)
# ---------------------------------------------------------------------------
STOCK_UNIVERSE = {
    "Technology": [
        "AAPL", "MSFT", "NVDA", "GOOGL", "META", "AVGO", "ORCL", "CRM",
        "AMD", "ADBE", "INTC", "CSCO", "QCOM", "TXN", "NOW", "IBM",
    ],
    "Financial Services": [
        "JPM", "V", "MA", "BAC", "GS", "MS", "WFC", "BLK", "SCHW",
        "AXP", "C", "SPGI", "CME",
    ],
    "Healthcare": [
        "UNH", "JNJ", "LLY", "PFE", "ABBV", "MRK", "TMO", "ABT",
        "DHR", "BMY", "AMGN", "MDT",
    ],
    "Consumer Cyclical": [
        "AMZN", "TSLA", "HD", "MCD", "NKE", "SBUX", "LOW", "TJX",
        "BKNG", "CMG",
    ],
    "Consumer Defensive": [
        "WMT", "PG", "KO", "PEP", "COST", "PM", "CL", "MDLZ",
    ],
    "Energy": [
        "XOM", "CVX", "COP", "SLB", "EOG", "MPC", "PSX", "VLO",
    ],
    "Industrials": [
        "CAT", "GE", "RTX", "HON", "UNP", "BA", "DE", "LMT", "UPS",
    ],
    "Communication Services": [
        "GOOG", "DIS", "NFLX", "CMCSA", "T", "VZ", "TMUS",
    ],
    "Utilities": [
        "NEE", "DUK", "SO", "D", "AEP", "SRE",
    ],
    "Real Estate": [
        "AMT", "PLD", "CCI", "EQIX", "SPG", "O",
    ],
    "Basic Materials": [
        "LIN", "APD", "ECL", "SHW", "NEM", "FCX",
    ],
}


def get_company_info(ticker: str) -> dict:
    """Fetch key company info and ratios from yfinance."""
    t = yf.Ticker(ticker)
    info = t.info

    return {
        "name": info.get("longName", ticker),
        "sector": info.get("sector", "N/A"),
        "industry": info.get("industry", "N/A"),
        "market_cap": info.get("marketCap"),
        "enterprise_value": info.get("enterpriseValue"),
        "current_price": info.get("currentPrice") or info.get("regularMarketPrice"),
        "52w_high": info.get("fiftyTwoWeekHigh"),
        "52w_low": info.get("fiftyTwoWeekLow"),
        "pe_ratio": info.get("trailingPE"),
        "forward_pe": info.get("forwardPE"),
        "pb_ratio": info.get("priceToBook"),
        "ps_ratio": info.get("priceToSalesTrailing12Months"),
        "ev_ebitda": info.get("enterpriseToEbitda"),
        "ev_revenue": info.get("enterpriseToRevenue"),
        "profit_margin": info.get("profitMargins"),
        "operating_margin": info.get("operatingMargins"),
        "roe": info.get("returnOnEquity"),
        "roa": info.get("returnOnAssets"),
        "revenue_growth": info.get("revenueGrowth"),
        "earnings_growth": info.get("earningsGrowth"),
        "dividend_yield": info.get("dividendYield"),
        "beta": info.get("beta"),
        "shares_outstanding": info.get("sharesOutstanding"),
        "total_debt": info.get("totalDebt"),
        "total_cash": info.get("totalCash"),
        "free_cash_flow": info.get("freeCashflow"),
        "operating_cash_flow": info.get("operatingCashflow"),
        "revenue": info.get("totalRevenue"),
        "ebitda": info.get("ebitda"),
    }


def get_financial_statements(ticker: str) -> dict:
    """Fetch financial statements for DCF analysis."""
    t = yf.Ticker(ticker)

    cashflow = t.cashflow
    income = t.financials
    balance = t.balance_sheet

    return {
        "cashflow": cashflow,
        "income": income,
        "balance_sheet": balance,
    }


def entropy_adjusted_wacc(base_wacc: float, entropy_score: float,
                          max_premium: float = 0.06) -> dict:
    """Adjust WACC using the entropy score as a risk premium.

    The insight: traditional DCF uses a static discount rate, but entropy
    tells us how much the regime is changing. A stock in a high-entropy
    regime carries more uncertainty, so the discount rate should be higher.

    Formula: adjusted_wacc = base_wacc + (entropy_score / 100) * max_premium

    At entropy=0 (perfectly stable), no premium added.
    At entropy=100 (maximum disruption), full premium (e.g., +6%) added.

    Args:
        base_wacc: Base weighted average cost of capital.
        entropy_score: Composite entropy score (0-100).
        max_premium: Maximum additional risk premium at entropy=100.

    Returns:
        Dict with adjusted WACC, premium applied, and explanation.
    """
    # Non-linear scaling: use sqrt to make moderate entropy more impactful
    # than a pure linear mapping (most stocks cluster in 20-60 range)
    normalized = min(1.0, max(0.0, entropy_score / 100))
    premium = normalized ** 0.7 * max_premium  # concave curve

    adjusted = base_wacc + premium

    if entropy_score > 70:
        rationale = "High entropy regime — significant risk premium applied. The rules around this asset are actively changing."
    elif entropy_score > 45:
        rationale = "Moderate entropy — meaningful risk premium. Some unusual dynamics warrant caution."
    elif entropy_score > 25:
        rationale = "Low entropy — minor risk premium. Stable regime supports base discount rate."
    else:
        rationale = "Very low entropy — minimal premium. Highly predictable regime."

    return {
        "base_wacc": round(base_wacc * 100, 2),
        "entropy_premium": round(premium * 100, 2),
        "adjusted_wacc": round(adjusted * 100, 2),
        "entropy_score": round(entropy_score, 1),
        "rationale": rationale,
    }


def compute_dcf(ticker: str, discount_rate: float = 0.10,
                terminal_growth: float = 0.03,
                projection_years: int = 5) -> dict:
    """Compute a simple DCF valuation.

    Uses historical free cash flow to project future FCF,
    then discounts back to present value.

    Args:
        ticker: Stock ticker symbol.
        discount_rate: WACC / required return (default 10%).
        terminal_growth: Long-term growth rate (default 3%).
        projection_years: Years to project (default 5).

    Returns:
        Dict with DCF results including intrinsic value per share.
    """
    t = yf.Ticker(ticker)
    info = t.info
    cashflow = t.cashflow

    # Get free cash flow history
    fcf_row = None
    for label in ["Free Cash Flow", "FreeCashFlow"]:
        if label in cashflow.index:
            fcf_row = cashflow.loc[label]
            break

    if fcf_row is None:
        # Try computing: Operating Cash Flow - CapEx
        op_cf = None
        capex = None
        for label in ["Total Cash From Operating Activities", "Operating Cash Flow"]:
            if label in cashflow.index:
                op_cf = cashflow.loc[label]
                break
        for label in ["Capital Expenditures", "Capital Expenditure"]:
            if label in cashflow.index:
                capex = cashflow.loc[label]
                break

        if op_cf is not None and capex is not None:
            fcf_row = op_cf + capex  # capex is typically negative
        else:
            return {"error": "Could not find free cash flow data"}

    # Clean and sort
    fcf_values = fcf_row.dropna().sort_index()
    if len(fcf_values) < 2:
        return {"error": "Insufficient FCF history for projection"}

    latest_fcf = float(fcf_values.iloc[-1])
    if latest_fcf <= 0:
        return {
            "error": "Negative or zero FCF — DCF not applicable",
            "latest_fcf": latest_fcf,
            "note": "Company is not generating positive free cash flow. Consider other valuation methods.",
        }

    # Calculate blended growth rate using multiple signals
    # Pure FCF growth is too noisy (capex timing, one-time items)
    # Blend: revenue growth, earnings growth, and FCF growth
    growth_signals = []

    # 1. Historical FCF growth (lowest weight — most volatile)
    fcf_arr = fcf_values.values.astype(float)
    positive_fcf = fcf_arr[fcf_arr > 0]
    if len(positive_fcf) >= 2:
        fcf_growth_rates = np.diff(positive_fcf) / positive_fcf[:-1]
        fcf_growth = float(np.median(fcf_growth_rates))
        fcf_growth = max(min(fcf_growth, 0.30), -0.10)
        growth_signals.append(("fcf", fcf_growth, 0.25))

    # 2. Revenue growth from yfinance (forward-looking, analyst-informed)
    rev_growth = info.get("revenueGrowth")
    if rev_growth is not None:
        rev_growth = max(min(float(rev_growth), 0.35), -0.10)
        growth_signals.append(("revenue", rev_growth, 0.40))

    # 3. Earnings growth (captures profitability trajectory)
    earn_growth = info.get("earningsGrowth")
    if earn_growth is not None:
        earn_growth = max(min(float(earn_growth), 0.35), -0.10)
        growth_signals.append(("earnings", earn_growth, 0.35))

    if growth_signals:
        # Weighted average of available signals
        total_weight = sum(w for _, _, w in growth_signals)
        avg_growth = sum(g * w for _, g, w in growth_signals) / total_weight
        # Cap at reasonable bounds
        avg_growth = max(min(avg_growth, 0.25), -0.05)
    else:
        avg_growth = 0.05  # fallback

    # Two-stage DCF: high growth tapers to terminal growth
    # Years 1-3: full growth rate, Years 4-5: linearly taper toward terminal
    projected_fcf = []
    fcf = latest_fcf
    for year in range(1, projection_years + 1):
        if year <= 3:
            year_growth = avg_growth
        else:
            # Linear taper from avg_growth to terminal_growth over years 4-5
            fade = (year - 3) / (projection_years - 3 + 1)
            year_growth = avg_growth * (1 - fade) + terminal_growth * fade
        fcf = fcf * (1 + year_growth)
        projected_fcf.append({
            "year": year,
            "fcf": round(fcf, 0),
            "pv_fcf": round(fcf / (1 + discount_rate) ** year, 0),
        })

    # Terminal value
    terminal_fcf = projected_fcf[-1]["fcf"] * (1 + terminal_growth)
    terminal_value = terminal_fcf / (discount_rate - terminal_growth)
    pv_terminal = terminal_value / (1 + discount_rate) ** projection_years

    # Total equity value
    pv_fcfs = sum(p["pv_fcf"] for p in projected_fcf)
    total_equity_value = pv_fcfs + pv_terminal

    # Add cash and subtract debt
    total_cash = info.get("totalCash", 0) or 0
    total_debt = info.get("totalDebt", 0) or 0
    equity_value = total_equity_value + total_cash - total_debt

    # Per share
    shares = info.get("sharesOutstanding", 1)
    intrinsic_value = equity_value / shares if shares else 0
    current_price = info.get("currentPrice") or info.get("regularMarketPrice") or 0

    upside = ((intrinsic_value - current_price) / current_price * 100) if current_price else 0

    return {
        "ticker": ticker,
        "current_price": round(current_price, 2),
        "intrinsic_value": round(intrinsic_value, 2),
        "upside_pct": round(upside, 1),
        "verdict": "UNDERVALUED" if upside > 15 else "OVERVALUED" if upside < -15 else "FAIRLY VALUED",
        "latest_fcf": round(latest_fcf, 0),
        "fcf_growth_rate": round(avg_growth * 100, 1),
        "discount_rate": round(discount_rate * 100, 1),
        "terminal_growth": round(terminal_growth * 100, 1),
        "projected_fcf": projected_fcf,
        "pv_of_fcfs": round(pv_fcfs, 0),
        "terminal_value": round(terminal_value, 0),
        "pv_terminal_value": round(pv_terminal, 0),
        "total_cash": round(total_cash, 0),
        "total_debt": round(total_debt, 0),
        "equity_value": round(equity_value, 0),
        "shares_outstanding": shares,
    }


# ---------------------------------------------------------------------------
# Monte Carlo DCF
# ---------------------------------------------------------------------------

def monte_carlo_dcf(ticker: str, base_wacc: float = 0.10,
                    base_terminal_growth: float = 0.03,
                    n_simulations: int = 1500,
                    projection_years: int = 5) -> dict:
    """Run Monte Carlo simulation on DCF by randomizing key assumptions.

    Randomly samples growth rate and WACC from normal distributions
    centered on the base values to produce a distribution of intrinsic values.

    Args:
        ticker: Stock ticker.
        base_wacc: Central WACC estimate.
        base_terminal_growth: Central terminal growth estimate.
        n_simulations: Number of scenarios to simulate.
        projection_years: DCF projection horizon.

    Returns:
        Dict with simulation results, percentiles, and probability of undervaluation.
    """
    t = yf.Ticker(ticker)
    info = t.info
    cashflow = t.cashflow

    # Get latest FCF
    fcf_row = None
    for label in ["Free Cash Flow", "FreeCashFlow"]:
        if label in cashflow.index:
            fcf_row = cashflow.loc[label]
            break
    if fcf_row is None:
        op_cf = None
        capex = None
        for label in ["Total Cash From Operating Activities", "Operating Cash Flow"]:
            if label in cashflow.index:
                op_cf = cashflow.loc[label]
                break
        for label in ["Capital Expenditures", "Capital Expenditure"]:
            if label in cashflow.index:
                capex = cashflow.loc[label]
                break
        if op_cf is not None and capex is not None:
            fcf_row = op_cf + capex
        else:
            return {"error": "Could not find free cash flow data"}

    fcf_values = fcf_row.dropna().sort_index()
    latest_fcf = float(fcf_values.iloc[-1])
    if latest_fcf <= 0:
        return {"error": "Negative FCF — Monte Carlo not applicable"}

    # Estimate base growth rate (same blended logic as compute_dcf)
    growth_signals = []
    fcf_arr = fcf_values.values.astype(float)
    positive_fcf = fcf_arr[fcf_arr > 0]
    if len(positive_fcf) >= 2:
        fcf_growth = float(np.median(np.diff(positive_fcf) / positive_fcf[:-1]))
        fcf_growth = max(min(fcf_growth, 0.30), -0.10)
        growth_signals.append(fcf_growth * 0.25)
    rev_g = info.get("revenueGrowth")
    if rev_g is not None:
        growth_signals.append(max(min(float(rev_g), 0.35), -0.10) * 0.40)
    earn_g = info.get("earningsGrowth")
    if earn_g is not None:
        growth_signals.append(max(min(float(earn_g), 0.35), -0.10) * 0.35)
    base_growth = sum(growth_signals) / (sum([0.25, 0.40, 0.35][:len(growth_signals)])) if growth_signals else 0.05
    base_growth = max(min(base_growth, 0.25), -0.05)

    shares = info.get("sharesOutstanding", 1) or 1
    total_cash = info.get("totalCash", 0) or 0
    total_debt = info.get("totalDebt", 0) or 0
    current_price = info.get("currentPrice") or info.get("regularMarketPrice") or 0

    # Run simulations
    np.random.seed(42)
    # Sample WACC: normal distribution, std = 1.5% of WACC
    wacc_samples = np.random.normal(base_wacc, base_wacc * 0.15, n_simulations)
    wacc_samples = np.clip(wacc_samples, 0.04, 0.20)

    # Sample growth: normal distribution, std = 30% of growth rate (min 2pp)
    growth_std = max(abs(base_growth) * 0.30, 0.02)
    growth_samples = np.random.normal(base_growth, growth_std, n_simulations)
    growth_samples = np.clip(growth_samples, -0.05, 0.30)

    # Sample terminal growth: narrow distribution
    tg_samples = np.random.normal(base_terminal_growth, 0.005, n_simulations)
    tg_samples = np.clip(tg_samples, 0.01, 0.05)

    intrinsic_values = []
    for i in range(n_simulations):
        wacc = wacc_samples[i]
        growth = growth_samples[i]
        tg = tg_samples[i]

        if wacc <= tg:
            continue  # invalid scenario

        # Two-stage projection
        fcf = latest_fcf
        pv_fcfs = 0
        for year in range(1, projection_years + 1):
            if year <= 3:
                yr_growth = growth
            else:
                fade = (year - 3) / (projection_years - 3 + 1)
                yr_growth = growth * (1 - fade) + tg * fade
            fcf = fcf * (1 + yr_growth)
            pv_fcfs += fcf / (1 + wacc) ** year

        # Terminal value
        terminal_fcf = fcf * (1 + tg)
        terminal_value = terminal_fcf / (wacc - tg)
        pv_terminal = terminal_value / (1 + wacc) ** projection_years

        equity_value = pv_fcfs + pv_terminal + total_cash - total_debt
        iv = equity_value / shares
        if iv > 0:
            intrinsic_values.append(round(float(iv), 2))

    if not intrinsic_values:
        return {"error": "All simulations produced invalid results"}

    iv_array = np.array(intrinsic_values)
    pct_undervalued = float(np.mean(iv_array > current_price) * 100) if current_price > 0 else 0

    return {
        "ticker": ticker,
        "n_simulations": len(intrinsic_values),
        "current_price": round(current_price, 2),
        "intrinsic_values": intrinsic_values,
        "mean": round(float(np.mean(iv_array)), 2),
        "median": round(float(np.median(iv_array)), 2),
        "std": round(float(np.std(iv_array)), 2),
        "p10": round(float(np.percentile(iv_array, 10)), 2),
        "p25": round(float(np.percentile(iv_array, 25)), 2),
        "p50": round(float(np.percentile(iv_array, 50)), 2),
        "p75": round(float(np.percentile(iv_array, 75)), 2),
        "p90": round(float(np.percentile(iv_array, 90)), 2),
        "pct_undervalued": round(pct_undervalued, 1),
        "base_growth": round(base_growth * 100, 1),
        "base_wacc": round(base_wacc * 100, 1),
        "base_terminal_growth": round(base_terminal_growth * 100, 1),
    }


# ---------------------------------------------------------------------------
# Peer comparison
# ---------------------------------------------------------------------------

def _get_peer_metrics(ticker: str) -> dict | None:
    """Fetch the metrics used for peer similarity scoring."""
    try:
        info = yf.Ticker(ticker).info
        return {
            "ticker": ticker,
            "name": info.get("longName", ticker),
            "sector": info.get("sector", "N/A"),
            "current_price": info.get("currentPrice") or info.get("regularMarketPrice"),
            "market_cap": info.get("marketCap"),
            "pe_ratio": info.get("trailingPE"),
            "ev_ebitda": info.get("enterpriseToEbitda"),
            "operating_margin": info.get("operatingMargins"),
            "revenue_growth": info.get("revenueGrowth"),
            "roe": info.get("returnOnEquity"),
            "roic": _compute_roic_from_info(info),
        }
    except Exception:
        return None


def _compute_roic_from_info(info: dict) -> float | None:
    """Compute ROIC = EBIT × (1 - tax_rate) / Invested Capital."""
    ebitda = info.get("ebitda")
    total_debt = info.get("totalDebt", 0) or 0
    total_cash = info.get("totalCash", 0) or 0
    market_cap = info.get("marketCap")
    # Approximate EBIT as EBITDA * 0.85 (rough DA adjustment)
    if ebitda is None or market_cap is None:
        return None
    ebit = ebitda * 0.85
    tax_rate = 0.21  # US corporate rate
    nopat = ebit * (1 - tax_rate)
    # Invested capital ≈ total equity (market cap proxy) + debt - cash
    invested_capital = market_cap + total_debt - total_cash
    if invested_capital <= 0:
        return None
    return nopat / invested_capital


def _similarity_score(target: dict, candidate: dict) -> float:
    """Score how financially similar two companies are (lower = more similar)."""
    score = 0.0
    # Market cap similarity (log scale since caps vary by orders of magnitude)
    if target["market_cap"] and candidate["market_cap"]:
        if target["market_cap"] > 0 and candidate["market_cap"] > 0:
            score += abs(np.log10(target["market_cap"]) - np.log10(candidate["market_cap"]))
    else:
        score += 3  # penalty for missing data

    # Operating margin similarity
    if target["operating_margin"] is not None and candidate["operating_margin"] is not None:
        score += abs(target["operating_margin"] - candidate["operating_margin"]) * 5
    else:
        score += 1

    # Revenue growth similarity
    if target["revenue_growth"] is not None and candidate["revenue_growth"] is not None:
        score += abs(target["revenue_growth"] - candidate["revenue_growth"]) * 3
    else:
        score += 1

    # ROE similarity
    if target["roe"] is not None and candidate["roe"] is not None:
        score += abs(target["roe"] - candidate["roe"]) * 2
    else:
        score += 0.5

    return score


def find_peers(ticker: str, n_peers: int = 5) -> dict:
    """Find the most financially similar peers for a given stock.

    1. Identifies the stock's sector
    2. Pulls metrics for all universe stocks in that sector
    3. Ranks by financial similarity
    4. Returns top N peers with comparison metrics
    """
    target_metrics = _get_peer_metrics(ticker)
    if target_metrics is None:
        return {"error": f"Could not fetch data for {ticker}"}

    sector = target_metrics["sector"]

    # Get candidates from the same sector in our universe
    candidates = STOCK_UNIVERSE.get(sector, [])
    if not candidates:
        # Fall back: try all sectors, score will handle it
        for s, tickers in STOCK_UNIVERSE.items():
            candidates.extend(tickers)

    # Remove the target itself
    candidates = [c for c in candidates if c != ticker]

    # Fetch metrics for all candidates
    peer_data = []
    for c in candidates:
        metrics = _get_peer_metrics(c)
        if metrics is not None:
            metrics["similarity"] = _similarity_score(target_metrics, metrics)
            peer_data.append(metrics)

    # Sort by similarity (lower = more similar)
    peer_data.sort(key=lambda x: x["similarity"])
    top_peers = peer_data[:n_peers]

    # Compute median of peers for ranking
    peer_medians = {}
    all_data = top_peers + [target_metrics]
    for metric in ["pe_ratio", "ev_ebitda", "operating_margin", "revenue_growth", "roic"]:
        values = [d[metric] for d in all_data if d.get(metric) is not None]
        if values:
            peer_medians[metric] = float(np.median(values))

    return {
        "target": target_metrics,
        "peers": top_peers,
        "peer_medians": peer_medians,
        "sector": sector,
    }


# ---------------------------------------------------------------------------
# Historical trends (4-year annual data)
# ---------------------------------------------------------------------------

def get_historical_trends(ticker: str) -> dict:
    """Fetch 4-year historical trends for key valuation metrics.

    Returns annual data for: revenue growth, operating margin, ROIC, P/E,
    and price performance (1Y, 2Y, 3Y returns).
    """
    t = yf.Ticker(ticker)
    info = t.info
    income = t.financials        # annual income statement
    balance = t.balance_sheet    # annual balance sheet

    years = []  # list of dicts, one per year

    if income is not None and not income.empty:
        # Columns are dates, rows are line items — sorted newest first
        for col in sorted(income.columns):
            year_label = col.strftime("%Y") if hasattr(col, "strftime") else str(col)[:4]

            revenue = _safe_float(income, col, ["Total Revenue", "Revenue"])
            op_income = _safe_float(income, col, ["Operating Income", "EBIT"])
            net_income = _safe_float(income, col, ["Net Income", "Net Income Common Stockholders"])

            # Operating margin
            op_margin = (op_income / revenue) if (revenue and op_income and revenue != 0) else None

            # ROIC from financials
            roic = None
            if balance is not None and col in balance.columns and op_income:
                total_equity = _safe_float(balance, col,
                                           ["Total Stockholders Equity", "Stockholders Equity",
                                            "Total Equity Gross Minority Interest"])
                total_debt_bs = _safe_float(balance, col,
                                            ["Total Debt", "Long Term Debt", "Long Term Debt And Capital Lease Obligation"])
                cash = _safe_float(balance, col, ["Cash And Cash Equivalents", "Cash Cash Equivalents And Short Term Investments"])
                if total_equity and total_debt_bs:
                    invested = (total_equity + total_debt_bs - (cash or 0))
                    if invested > 0:
                        nopat = op_income * (1 - 0.21)
                        roic = nopat / invested

            years.append({
                "year": year_label,
                "revenue": revenue,
                "operating_income": op_income,
                "net_income": net_income,
                "operating_margin": round(op_margin * 100, 1) if op_margin is not None else None,
                "roic": round(roic * 100, 1) if roic is not None else None,
            })

    # Compute revenue growth (YoY)
    for i in range(1, len(years)):
        prev_rev = years[i - 1].get("revenue")
        curr_rev = years[i].get("revenue")
        if prev_rev and curr_rev and prev_rev != 0:
            years[i]["revenue_growth"] = round((curr_rev - prev_rev) / prev_rev * 100, 1)
        else:
            years[i]["revenue_growth"] = None
    if years:
        years[0]["revenue_growth"] = None  # no prior year for first entry

    # Historical P/E — use annual earnings + year-end price
    shares = info.get("sharesOutstanding", 0) or 0
    if shares > 0:
        prices = t.history(period="5y")
        if not prices.empty:
            for yr in years:
                try:
                    year_int = int(yr["year"])
                    # Get last trading day of that year
                    year_prices = prices[prices.index.year == year_int]
                    if not year_prices.empty and yr.get("net_income"):
                        year_end_price = float(year_prices["Close"].iloc[-1])
                        eps = yr["net_income"] / shares
                        if eps > 0:
                            yr["pe_ratio"] = round(year_end_price / eps, 1)
                        else:
                            yr["pe_ratio"] = None
                    else:
                        yr["pe_ratio"] = None
                except (ValueError, IndexError):
                    yr["pe_ratio"] = None
        else:
            for yr in years:
                yr["pe_ratio"] = None
    else:
        for yr in years:
            yr["pe_ratio"] = None

    # Price performance
    price_performance = _get_price_performance(ticker)

    return {
        "ticker": ticker,
        "annual_data": years,
        "price_performance": price_performance,
    }


def _safe_float(df: pd.DataFrame, col, labels: list) -> float | None:
    """Safely extract a float from a DataFrame trying multiple row labels."""
    for label in labels:
        if label in df.index:
            val = df.loc[label, col]
            if pd.notna(val):
                return float(val)
    return None


def _get_price_performance(ticker: str) -> dict:
    """Compute 1Y, 2Y, 3Y price returns."""
    try:
        prices = yf.Ticker(ticker).history(period="5y")
        if prices.empty:
            return {}
        current = float(prices["Close"].iloc[-1])
        result = {}
        for years, label in [(1, "1Y"), (2, "2Y"), (3, "3Y")]:
            target_idx = len(prices) - (252 * years)
            if target_idx >= 0:
                past_price = float(prices["Close"].iloc[target_idx])
                result[label] = round((current - past_price) / past_price * 100, 1)
        return result
    except Exception:
        return {}
