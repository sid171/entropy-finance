"""Valuation analysis tools — DCF, key ratios, and financial summary."""

import yfinance as yf
import numpy as np
import pandas as pd


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

    # Calculate historical FCF growth rate
    fcf_arr = fcf_values.values.astype(float)
    positive_fcf = fcf_arr[fcf_arr > 0]
    if len(positive_fcf) >= 2:
        growth_rates = np.diff(positive_fcf) / positive_fcf[:-1]
        avg_growth = float(np.median(growth_rates))
        # Cap growth rate at reasonable bounds
        avg_growth = max(min(avg_growth, 0.30), -0.10)
    else:
        avg_growth = 0.05  # default 5%

    # Project future FCF
    projected_fcf = []
    fcf = latest_fcf
    for year in range(1, projection_years + 1):
        fcf = fcf * (1 + avg_growth)
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
