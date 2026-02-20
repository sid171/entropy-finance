"""Fintropy — Valuation Terminal with Entropy Radar.

Combines traditional valuation (DCF, ratios) with an entropy intelligence
layer that detects when the rules around a stock are changing.
"""

import json
import os
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import yfinance as yf

from config import SYSTEM_PROMPT, TOOL_DEFINITIONS
from market_data import get_returns, get_price_data
from entropy_tools import shannon_entropy, net_transfer_entropy, rolling_entropy, detect_regimes, correlation_stability
from valuation import get_company_info, compute_dcf, entropy_adjusted_wacc, find_peers, get_historical_trends, monte_carlo_dcf
from heatmap import compute_sector_heatmap
from entropy_radar import run_entropy_radar
from backtest import backtest_entropy_signals
from accuracy import run_cross_stock_backtest, run_analyst_comparison

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
load_dotenv()
st.set_page_config(page_title="Fintropy", page_icon="📊", layout="wide")

# ---------------------------------------------------------------------------
# Custom CSS — clean professional finance aesthetic
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    /* Global font */
    .stApp {
        font-family: 'Inter', 'SF Pro Display', -apple-system, sans-serif;
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
        background-color: #F6F8FA;
        border-radius: 8px;
        padding: 4px;
        border: 1px solid #E5E7EB;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 6px;
        padding: 8px 16px;
        color: #6B7280;
        font-weight: 500;
        font-size: 0.85rem;
    }
    .stTabs [aria-selected="true"] {
        background-color: #FFFFFF;
        color: #1B6B4A;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
    }

    /* Metric cards */
    [data-testid="stMetric"] {
        background-color: #F9FAFB;
        border: 1px solid #E5E7EB;
        border-radius: 10px;
        padding: 14px 18px;
        box-shadow: 0 1px 2px rgba(0,0,0,0.04);
    }
    [data-testid="stMetricLabel"] {
        color: #6B7280;
        font-size: 0.72rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        font-weight: 500;
    }
    [data-testid="stMetricValue"] {
        color: #111827;
        font-weight: 700;
        font-size: 1.15rem;
    }

    /* Dataframe */
    [data-testid="stDataFrame"] {
        border: 1px solid #E5E7EB;
        border-radius: 10px;
        overflow: hidden;
    }

    /* Dividers */
    hr {
        border-color: #E5E7EB;
        opacity: 0.6;
    }

    /* Info/warning boxes */
    .stAlert {
        border-radius: 10px;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #F9FAFB;
        border-right: 1px solid #E5E7EB;
    }

    /* Subheader accent */
    h3 {
        color: #1B6B4A;
        font-weight: 700;
        letter-spacing: -0.02em;
        border-bottom: 2px solid #D1FAE5;
        padding-bottom: 6px;
    }

    /* Main title */
    h1 {
        color: #111827;
        font-weight: 800;
        letter-spacing: -0.03em;
    }

    /* Caption text */
    .stCaption {
        color: #9CA3AF;
    }

    /* Button styling */
    .stButton > button {
        background-color: #1B6B4A;
        color: #FFFFFF;
        font-weight: 600;
        border: none;
        border-radius: 8px;
        padding: 8px 24px;
        transition: all 0.2s;
        box-shadow: 0 1px 3px rgba(27,107,74,0.2);
    }
    .stButton > button:hover {
        background-color: #15573D;
        color: #FFFFFF;
        box-shadow: 0 2px 6px rgba(27,107,74,0.3);
    }

    /* Number input */
    [data-testid="stNumberInput"] input {
        border: 1px solid #D1D5DB;
        border-radius: 8px;
        color: #FFFFFF;
    }

    /* Select box */
    [data-testid="stSelectbox"] > div > div {
        border: 1px solid #D1D5DB;
        border-radius: 8px;
    }

    /* Chat messages */
    [data-testid="stChatMessage"] {
        background-color: #F9FAFB;
        border: 1px solid #E5E7EB;
        border-radius: 10px;
        color: #1F2937;
    }
    [data-testid="stChatMessage"] p,
    [data-testid="stChatMessage"] li,
    [data-testid="stChatMessage"] span {
        color: #1F2937 !important;
    }

    /* Expander */
    .streamlit-expanderHeader {
        background-color: #F6F8FA;
        border-radius: 8px;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)

api_key = os.environ.get("OPENAI_API_KEY", "")
client = OpenAI(api_key=api_key) if api_key else None

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def fmt_num(n, prefix="$"):
    if n is None: return "N/A"
    if abs(n) >= 1e12: return f"{prefix}{n/1e12:.2f}T"
    if abs(n) >= 1e9: return f"{prefix}{n/1e9:.2f}B"
    if abs(n) >= 1e6: return f"{prefix}{n/1e6:.1f}M"
    return f"{prefix}{n:,.0f}"

def fmt_pct(n):
    if n is None: return "N/A"
    return f"{n*100:.1f}%" if abs(n) < 1 else f"{n:.1f}%"

def fmt_ratio(n):
    if n is None: return "N/A"
    return f"{n:.2f}"


@st.cache_data(ttl=300, show_spinner=False)
def get_dcf_recommendations(ticker: str) -> dict:
    """Compute suggested WACC and terminal growth for a ticker using CAPM."""
    try:
        info = yf.Ticker(ticker).info
        beta = info.get("beta", 1.0) or 1.0
        revenue_growth = info.get("revenueGrowth")
        sector = info.get("sector", "")

        # CAPM-based cost of equity
        risk_free = 0.043       # ~10Y US Treasury
        market_premium = 0.055  # long-run equity risk premium
        cost_of_equity = risk_free + beta * market_premium

        # Rough WACC: blend equity and debt cost
        total_debt = info.get("totalDebt", 0) or 0
        market_cap = info.get("marketCap", 1) or 1
        debt_ratio = total_debt / (total_debt + market_cap)
        cost_of_debt = 0.05 * (1 - 0.21)  # after-tax ~4%
        wacc = cost_of_equity * (1 - debt_ratio) + cost_of_debt * debt_ratio
        wacc_pct = round(wacc * 100, 1)
        wacc_pct = max(5.0, min(20.0, wacc_pct))  # clamp to reasonable range

        # Terminal growth suggestion based on growth profile
        if revenue_growth is not None and revenue_growth > 0.15:
            tg = 3.5
            tg_reason = "above-average growth company"
        elif revenue_growth is not None and revenue_growth > 0.05:
            tg = 3.0
            tg_reason = "moderate growth profile"
        elif sector in ("Utilities", "Consumer Defensive", "Real Estate"):
            tg = 2.0
            tg_reason = "stable/defensive sector"
        else:
            tg = 2.5
            tg_reason = "mature company"

        wacc_reason = f"Beta={beta:.2f}, Rf=4.3%, ERP=5.5%"

        return {
            "wacc": wacc_pct,
            "wacc_reason": wacc_reason,
            "terminal_growth": tg,
            "tg_reason": tg_reason,
        }
    except Exception:
        return {}

def score_color(score):
    if score > 70: return "🔴"
    if score > 45: return "🟡"
    if score > 25: return "🟢"
    return "⚪"

def execute_chat_tool(name, args):
    """Execute entropy tools from chat context."""
    try:
        if name == "compute_shannon_entropy":
            r = get_returns(args["ticker"], args.get("period", "1y"))
            return json.dumps({"ticker": args["ticker"], "shannon_entropy": round(shannon_entropy(r), 4)})
        elif name == "compute_transfer_entropy":
            s = get_returns(args["source_ticker"], args.get("period", "1y"))
            t = get_returns(args["target_ticker"], args.get("period", "1y"))
            s.name, t.name = args["source_ticker"], args["target_ticker"]
            return json.dumps(net_transfer_entropy(s, t, lag=args.get("lag", 1)))
        elif name == "compute_rolling_entropy":
            r = get_returns(args["ticker"], args.get("period", "2y"))
            result = rolling_entropy(r, window=args.get("window", 60))
            return json.dumps({"ticker": args["ticker"], "current": round(float(result.iloc[-1]), 4), "mean": round(float(result.mean()), 4)})
        elif name == "detect_regime_changes":
            r = get_returns(args["ticker"], args.get("period", "2y"))
            return json.dumps(detect_regimes(r))
        elif name == "analyze_correlation_stability":
            ra = get_returns(args["ticker_a"], args.get("period", "2y"))
            rb = get_returns(args["ticker_b"], args.get("period", "2y"))
            result = correlation_stability(ra, rb, window=args.get("window", 60))
            result.pop("rolling_correlation", None)
            return json.dumps(result)
        return json.dumps({"error": f"Unknown tool: {name}"})
    except Exception as e:
        return json.dumps({"error": str(e)})

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 16px 0 8px 0;">
        <div style="font-size: 2.2rem; font-weight: 800; color: #1B6B4A; letter-spacing: -0.03em; line-height: 1.1;">
            Fintropy
        </div>
        <div style="font-size: 0.78rem; color: #6B7280; margin-top: 4px; letter-spacing: 0.08em; text-transform: uppercase;">
            Valuation + Entropy Intelligence
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown('<p style="font-size:0.75rem; color:#6B7280; text-transform:uppercase; letter-spacing:0.08em; font-weight:600; margin-bottom:4px;">Stock Ticker</p>', unsafe_allow_html=True)
    ticker = st.text_input("Ticker", value="AAPL", placeholder="e.g. AAPL, NVDA, TSLA", label_visibility="collapsed")

    st.markdown('<p style="font-size:0.75rem; color:#6B7280; text-transform:uppercase; letter-spacing:0.08em; font-weight:600; margin-bottom:4px;">Analysis Period</p>', unsafe_allow_html=True)
    period = st.selectbox("Period", ["6mo", "1y", "2y", "5y"], index=1, label_visibility="collapsed")

    st.markdown("")
    analyze_btn = st.button("Analyze", type="primary", use_container_width=True)

    st.markdown("---")

    st.markdown('<p style="font-size:0.75rem; color:#6B7280; text-transform:uppercase; letter-spacing:0.08em; font-weight:600; margin-bottom:2px;">DCF Parameters</p>', unsafe_allow_html=True)
    st.markdown('<p style="font-size:0.72rem; color:#374151; margin-bottom:2px;">WACC (%)</p>', unsafe_allow_html=True)
    discount_rate = st.number_input("WACC", min_value=1.0, max_value=30.0, value=10.0, step=0.5, format="%.1f", label_visibility="collapsed") / 100
    st.markdown('<p style="font-size:0.72rem; color:#374151; margin-bottom:2px;">Terminal Growth (%)</p>', unsafe_allow_html=True)
    terminal_growth = st.number_input("Terminal Growth", min_value=0.0, max_value=10.0, value=3.0, step=0.5, format="%.1f", label_visibility="collapsed") / 100

    # Ticker-specific recommendations
    recs = get_dcf_recommendations(ticker)
    if recs:
        st.markdown(f"""
        <div style="padding: 10px; background-color: #FFF7ED; border-radius: 8px; border: 1px solid #FED7AA; margin-top: 8px;">
            <p style="font-size: 0.7rem; color: #9A3412; margin: 0; font-weight: 600;">Suggested for {ticker}</p>
            <p style="font-size: 0.68rem; color: #374151; margin: 4px 0 0 0; line-height: 1.5;">
                <b>WACC: {recs['wacc']}%</b> — {recs['wacc_reason']}<br>
                <b>Terminal Growth: {recs['terminal_growth']}%</b> — {recs['tg_reason']}
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("""
    <div style="padding: 12px; background-color: #ECFDF5; border-radius: 8px; border: 1px solid #D1FAE5;">
        <p style="font-size: 0.72rem; color: #1B6B4A; margin: 0; font-weight: 600;">How it works</p>
        <p style="font-size: 0.7rem; color: #374151; margin: 4px 0 0 0; line-height: 1.4;">
            Enter a ticker and hit Analyze. This tool combines traditional DCF valuation with entropy-based regime detection to surface hidden risks.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("")
    st.markdown('<p style="font-size: 0.65rem; color: #9CA3AF; text-align: center;">MGMT 69000 &middot; Purdue MSF &middot; DRIVER Framework</p>', unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------------------------------------------------------------------------
# Run analysis (compute and cache in session_state)
# ---------------------------------------------------------------------------
if analyze_btn or "cached_analysis" not in st.session_state:
    tkr = ticker
    try:
        with st.spinner(f"Analyzing {tkr}..."):
            info = get_company_info(tkr)
            dcf = compute_dcf(tkr, discount_rate=discount_rate, terminal_growth=terminal_growth)
            prices = get_price_data(tkr, period)
            radar = run_entropy_radar(tkr, info.get("sector", ""), period)

        # Entropy-adjusted DCF
        es_score = radar["entropy_score"]["composite_score"]
        wacc_adj = entropy_adjusted_wacc(discount_rate, es_score)
        dcf_adjusted = compute_dcf(tkr,
                                    discount_rate=wacc_adj["adjusted_wacc"] / 100,
                                    terminal_growth=terminal_growth)

        # Peer comparison and historical trends
        with st.spinner("Finding comparable peers..."):
            peers = find_peers(tkr)
        with st.spinner("Loading historical trends..."):
            historical = get_historical_trends(tkr)

        # Cache everything for re-rendering
        st.session_state["cached_analysis"] = {
            "info": info,
            "dcf": dcf,
            "dcf_adjusted": dcf_adjusted,
            "wacc_adj": wacc_adj,
            "prices": prices,
            "radar": radar,
            "peers": peers,
            "historical": historical,
            "ticker": tkr,
        }

        # Also store lightweight version for chat context
        st.session_state["analysis_data"] = {
            "info": info, "dcf": dcf, "radar": {
                k: v for k, v in radar.items()
                if k not in ("rolling_entropy", "correlation_stability", "transfer_entropy")
            }, "ticker": tkr,
        }

    except Exception as e:
        st.error(f"Error analyzing {tkr}: {e}")
        import traceback
        st.code(traceback.format_exc())

# ---------------------------------------------------------------------------
# Render analysis from cache
# ---------------------------------------------------------------------------
if "cached_analysis" in st.session_state:
    cached = st.session_state["cached_analysis"]
    info = cached["info"]
    dcf = cached["dcf"]
    dcf_adjusted = cached["dcf_adjusted"]
    wacc_adj = cached["wacc_adj"]
    prices = cached["prices"]
    radar = cached["radar"]
    peers = cached.get("peers", {})
    historical = cached.get("historical", {})
    tkr = cached["ticker"]

    # ================================================================
    # HEADER + ENTROPY SCORE
    # ================================================================
    header_col, score_col = st.columns([3, 1])
    with header_col:
        st.title(f"{info['name']} ({tkr})")
        st.caption(f"{info.get('sector', 'N/A')} · {info.get('industry', 'N/A')}")
    with score_col:
        es = radar["entropy_score"]
        st.metric(
            f"{score_color(es['composite_score'])} Entropy Score",
            f"{es['composite_score']}/100",
        )
        st.caption(es["interpretation"])

    # ================================================================
    # TABS
    # ================================================================
    tab_val, tab_radar, tab_flow, tab_heatmap, tab_backtest, tab_accuracy, tab_chat, tab_method, tab_detail = st.tabs([
        "📈 Valuation", "🎯 Entropy Radar", "🔀 Information Flow", "🗺 Sector Heatmap",
        "🔬 Backtest", "🎯 Accuracy", "💬 AI Chat", "📖 Methodology", "📋 Details",
    ])

    # ---- VALUATION TAB ----
    with tab_val:
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("Price", f"${info['current_price']:.2f}" if info['current_price'] else "N/A")
        c2.metric("Market Cap", fmt_num(info['market_cap']))
        c3.metric("P/E", fmt_ratio(info['pe_ratio']))
        c4.metric("P/B", fmt_ratio(info['pb_ratio']))
        c5.metric("EV/EBITDA", fmt_ratio(info['ev_ebitda']))
        c6.metric("Beta", fmt_ratio(info['beta']))

        c7, c8, c9, c10, c11, c12 = st.columns(6)
        c7.metric("Revenue", fmt_num(info['revenue']))
        c8.metric("EBITDA", fmt_num(info['ebitda']))
        c9.metric("FCF", fmt_num(info['free_cash_flow']))
        c10.metric("Profit Margin", fmt_pct(info['profit_margin']))
        c11.metric("ROE", fmt_pct(info['roe']))
        c12.metric("Div Yield", fmt_pct(info['dividend_yield']))

        # ---- METRICS WRITE-UP ----
        st.divider()
        st.subheader("What These Metrics Tell Us")

        writeup_parts = []

        # Valuation assessment
        pe = info.get("pe_ratio")
        pb = info.get("pb_ratio")
        ev_ebitda = info.get("ev_ebitda")
        if pe is not None:
            if pe > 35:
                writeup_parts.append(f"The **P/E ratio of {pe:.1f}x** indicates the market is pricing in strong future earnings growth — investors are paying a premium for each dollar of current earnings. This typically reflects high growth expectations but also means the stock is vulnerable to disappointment if growth slows.")
            elif pe > 20:
                writeup_parts.append(f"The **P/E ratio of {pe:.1f}x** suggests a moderately valued stock. The market expects solid earnings growth, but the premium is not extreme. This is consistent with a mature growth company.")
            elif pe > 0:
                writeup_parts.append(f"The **P/E ratio of {pe:.1f}x** suggests the stock is valued conservatively. This could mean the market sees limited growth ahead, or it could represent a value opportunity if earnings improve.")

        if ev_ebitda is not None:
            if ev_ebitda > 25:
                writeup_parts.append(f"An **EV/EBITDA of {ev_ebitda:.1f}x** is elevated, meaning the enterprise is valued at a significant multiple of its operating cash generation. This is common in high-growth or asset-light businesses but implies that a large portion of the valuation depends on future performance.")
            elif ev_ebitda > 12:
                writeup_parts.append(f"An **EV/EBITDA of {ev_ebitda:.1f}x** is in the moderate range — the market values the business reasonably relative to its cash generation capacity.")
            elif ev_ebitda > 0:
                writeup_parts.append(f"An **EV/EBITDA of {ev_ebitda:.1f}x** is relatively low, suggesting the business generates strong cash flow relative to its enterprise value. This can signal a value opportunity or reflect limited growth expectations.")

        # Profitability
        op_margin = info.get("operating_margin")
        profit_margin = info.get("profit_margin")
        roe = info.get("roe")
        if op_margin is not None and profit_margin is not None:
            if op_margin > 0.25:
                writeup_parts.append(f"**Operating margin of {op_margin*100:.1f}%** and **net margin of {profit_margin*100:.1f}%** indicate a highly profitable business with strong pricing power and cost discipline. These margins suggest a durable competitive advantage.")
            elif op_margin > 0.12:
                writeup_parts.append(f"**Operating margin of {op_margin*100:.1f}%** and **net margin of {profit_margin*100:.1f}%** show solid profitability. The business converts revenue to profit at a healthy rate, though there may be room for operational improvement.")
            elif op_margin > 0:
                writeup_parts.append(f"**Operating margin of {op_margin*100:.1f}%** and **net margin of {profit_margin*100:.1f}%** are relatively thin, which could indicate a competitive industry, high cost structure, or a business still scaling toward profitability.")

        if roe is not None:
            if roe > 0.25:
                writeup_parts.append(f"**ROE of {roe*100:.1f}%** demonstrates the company is generating strong returns on shareholder equity — it is effectively converting invested capital into profit.")
            elif roe > 0.10:
                writeup_parts.append(f"**ROE of {roe*100:.1f}%** shows adequate returns on equity, consistent with a stable business that generates reasonable profit from its capital base.")
            elif roe > 0:
                writeup_parts.append(f"**ROE of {roe*100:.1f}%** is below average, suggesting the company may not be deploying shareholder capital as efficiently as peers.")

        # Cash flow
        fcf = info.get("free_cash_flow")
        revenue = info.get("revenue")
        if fcf is not None and revenue is not None and revenue > 0:
            fcf_margin = fcf / revenue
            if fcf_margin > 0.15:
                writeup_parts.append(f"**Free cash flow of {fmt_num(fcf)}** ({fcf_margin*100:.1f}% of revenue) is strong — the business generates significant cash after capital expenditures, providing flexibility for dividends, buybacks, debt reduction, or reinvestment.")
            elif fcf_margin > 0.05:
                writeup_parts.append(f"**Free cash flow of {fmt_num(fcf)}** ({fcf_margin*100:.1f}% of revenue) is positive and healthy, indicating the business can fund operations and some growth internally.")
            elif fcf > 0:
                writeup_parts.append(f"**Free cash flow of {fmt_num(fcf)}** ({fcf_margin*100:.1f}% of revenue) is positive but modest relative to revenue, which may limit the company's financial flexibility.")

        # Beta / risk
        beta = info.get("beta")
        if beta is not None:
            if beta > 1.3:
                writeup_parts.append(f"A **beta of {beta:.2f}** means this stock is significantly more volatile than the market — it tends to amplify market moves, making it a higher-risk holding.")
            elif beta > 0.8:
                writeup_parts.append(f"A **beta of {beta:.2f}** indicates the stock moves roughly in line with the broader market, suggesting moderate systematic risk.")
            else:
                writeup_parts.append(f"A **beta of {beta:.2f}** suggests the stock is less volatile than the market — it may offer defensive characteristics during downturns.")

        if writeup_parts:
            st.markdown("\n\n".join(writeup_parts))
        else:
            st.info("Insufficient data to generate a metrics summary.")

        st.divider()

        # DCF — Standard vs Entropy-Adjusted side by side
        st.subheader("DCF Valuation")
        if "error" in dcf:
            st.warning(f"DCF: {dcf['error']}")
            if "note" in dcf:
                st.info(dcf["note"])
        else:
            st.markdown("**Standard DCF vs Entropy-Adjusted DCF**")
            st.caption("The entropy-adjusted DCF adds a risk premium to the discount rate based on how unstable the regime is.")

            col_std, col_adj = st.columns(2)

            with col_std:
                st.markdown("##### Standard DCF")
                delta_color = "normal" if dcf["upside_pct"] > 0 else "inverse"
                st.metric("Intrinsic Value", f"${dcf['intrinsic_value']:.2f}",
                          delta=f"{dcf['upside_pct']:+.1f}%", delta_color=delta_color)
                st.metric("WACC", f"{dcf['discount_rate']}%")
                st.metric("Verdict", dcf["verdict"])

            with col_adj:
                st.markdown("##### Entropy-Adjusted DCF")
                if "error" in dcf_adjusted:
                    st.warning(dcf_adjusted["error"])
                else:
                    adj_delta_color = "normal" if dcf_adjusted["upside_pct"] > 0 else "inverse"
                    st.metric("Intrinsic Value", f"${dcf_adjusted['intrinsic_value']:.2f}",
                              delta=f"{dcf_adjusted['upside_pct']:+.1f}%", delta_color=adj_delta_color)
                    st.metric("Adjusted WACC",
                              f"{wacc_adj['adjusted_wacc']}%",
                              delta=f"+{wacc_adj['entropy_premium']}% entropy premium")
                    st.metric("Verdict", dcf_adjusted["verdict"])

            st.info(f"**Entropy Risk Premium:** {wacc_adj['rationale']} "
                    f"(Score: {wacc_adj['entropy_score']}/100 → +{wacc_adj['entropy_premium']}% added to WACC)")

            with st.expander("DCF Assumptions & Details"):
                dc1, dc2, dc3 = st.columns(3)
                dc1.markdown(f"**Inputs:**\n- Base WACC: {dcf['discount_rate']}%\n- Entropy Premium: +{wacc_adj['entropy_premium']}%\n- Adjusted WACC: {wacc_adj['adjusted_wacc']}%\n- Terminal Growth: {dcf['terminal_growth']}%")
                dc2.markdown(f"**Standard DCF:**\n- PV of FCFs: {fmt_num(dcf['pv_of_fcfs'])}\n- PV Terminal: {fmt_num(dcf['pv_terminal_value'])}\n- Equity Value: {fmt_num(dcf['equity_value'])}")
                if "error" not in dcf_adjusted:
                    dc3.markdown(f"**Entropy-Adjusted DCF:**\n- PV of FCFs: {fmt_num(dcf_adjusted['pv_of_fcfs'])}\n- PV Terminal: {fmt_num(dcf_adjusted['pv_terminal_value'])}\n- Equity Value: {fmt_num(dcf_adjusted['equity_value'])}")
                else:
                    dc3.markdown(f"**Entropy-Adjusted DCF:**\n- {dcf_adjusted['error']}")
                fcf_df = pd.DataFrame(dcf["projected_fcf"])
                fcf_df.columns = ["Year", "Projected FCF", "PV of FCF"]
                fcf_df["Projected FCF"] = fcf_df["Projected FCF"].apply(lambda x: fmt_num(x))
                fcf_df["PV of FCF"] = fcf_df["PV of FCF"].apply(lambda x: fmt_num(x))
                st.dataframe(fcf_df, use_container_width=True, hide_index=True)

        # ---- PEER COMPARISON ----
        st.divider()
        st.subheader("Peer Comparison")

        if "error" in peers:
            st.warning(peers["error"])
        elif peers.get("peers"):
            peer_list = peers["peers"]
            target = peers["target"]
            medians = peers.get("peer_medians", {})

            st.caption(f"Top {len(peer_list)} most financially similar companies in **{peers.get('sector', 'N/A')}**")

            # Build comparison table
            rows = []
            for p in [target] + peer_list:
                row = {
                    "Ticker": p["ticker"],
                    "Company": p.get("name", p["ticker"]),
                    "Market Cap": fmt_num(p.get("market_cap")),
                    "P/E": fmt_ratio(p.get("pe_ratio")),
                    "EV/EBITDA": fmt_ratio(p.get("ev_ebitda")),
                    "Op. Margin": fmt_pct(p.get("operating_margin")),
                    "Rev. Growth": fmt_pct(p.get("revenue_growth")),
                    "ROIC": f"{p['roic']*100:.1f}%" if p.get("roic") is not None else "N/A",
                }
                rows.append(row)

            comp_df = pd.DataFrame(rows)
            st.dataframe(comp_df, use_container_width=True, hide_index=True)

            # Ranking summary
            rank_metrics = {
                "P/E": ("pe_ratio", "lower"),
                "Operating Margin": ("operating_margin", "higher"),
                "Revenue Growth": ("revenue_growth", "higher"),
                "ROIC": ("roic", "higher"),
            }
            rank_items = []
            all_peers_with_target = [target] + peer_list
            for label, (key, direction) in rank_metrics.items():
                vals = [(p["ticker"], p.get(key)) for p in all_peers_with_target if p.get(key) is not None]
                if not vals:
                    continue
                vals.sort(key=lambda x: x[1], reverse=(direction == "higher"))
                rank = next((i + 1 for i, (t_name, _) in enumerate(vals) if t_name == tkr), None)
                if rank is not None:
                    rank_items.append(f"**{label}:** #{rank} of {len(vals)}")

            if rank_items:
                st.markdown("**Peer Ranking:** " + " · ".join(rank_items))
        else:
            st.info("No peer data available for this stock.")

        # ---- HISTORICAL TRENDS ----
        st.divider()
        st.subheader("Historical Trends")

        annual = historical.get("annual_data", [])
        price_perf = historical.get("price_performance", {})

        if annual and len(annual) >= 2:
            # Price performance summary
            if price_perf:
                perf_cols = st.columns(len(price_perf))
                for i, (label, ret) in enumerate(price_perf.items()):
                    delta_color = "normal" if ret > 0 else "inverse"
                    perf_cols[i].metric(f"{label} Return", f"{ret:+.1f}%", delta_color=delta_color)

            # Build charts for the 4 annual metrics
            years = [d["year"] for d in annual]

            chart_configs = [
                ("Revenue Growth (%)", "revenue_growth", "#636EFA"),
                ("Operating Margin (%)", "operating_margin", "#EF553B"),
                ("ROIC (%)", "roic", "#00CC96"),
                ("P/E Ratio", "pe_ratio", "#AB63FA"),
            ]

            col_left, col_right = st.columns(2)

            for idx, (title, key, color) in enumerate(chart_configs):
                values = [d.get(key) for d in annual]
                # Only plot if we have at least 2 non-None values
                valid = [(y, v) for y, v in zip(years, values) if v is not None]
                if len(valid) < 2:
                    continue

                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=[v[0] for v in valid],
                    y=[v[1] for v in valid],
                    mode="lines+markers",
                    line=dict(color=color, width=2),
                    marker=dict(size=8),
                    name=title,
                ))
                fig.update_layout(
                    title=title,
                    height=280,
                    template="plotly_white",
                    margin=dict(l=40, r=20, t=40, b=30),
                    yaxis_title=title,
                    xaxis=dict(tickmode="array", tickvals=[v[0] for v in valid]),
                    showlegend=False,
                )

                target_col = col_left if idx % 2 == 0 else col_right
                target_col.plotly_chart(fig, use_container_width=True)

                # One-liner interpretation under each chart
                first_val = valid[0][1]
                last_val = valid[-1][1]
                change = last_val - first_val
                if key == "revenue_growth":
                    if change > 5:
                        caption = "Revenue growth is accelerating — the top line is expanding faster over time."
                    elif change < -5:
                        caption = "Revenue growth is decelerating — top-line expansion is slowing."
                    else:
                        caption = "Revenue growth has been relatively stable over this period."
                elif key == "operating_margin":
                    if change > 3:
                        caption = "Operating margins are expanding — the business is becoming more efficient at converting revenue to profit."
                    elif change < -3:
                        caption = "Operating margins are compressing — costs are growing faster than revenue."
                    else:
                        caption = "Operating margins have held steady, indicating consistent operational efficiency."
                elif key == "roic":
                    if change > 3:
                        caption = "ROIC is improving — each dollar of invested capital is generating more return."
                    elif change < -3:
                        caption = "ROIC is declining — capital efficiency is deteriorating."
                    else:
                        caption = "ROIC has remained stable, reflecting consistent capital deployment effectiveness."
                elif key == "pe_ratio":
                    if change > 5:
                        caption = "P/E is expanding — the market is willing to pay more per dollar of earnings, signaling rising expectations."
                    elif change < -5:
                        caption = "P/E is compressing — the market is discounting future earnings more heavily."
                    else:
                        caption = "P/E has been relatively flat, suggesting stable market sentiment toward this stock."
                else:
                    caption = ""
                if caption:
                    target_col.caption(caption)
        else:
            st.info("Insufficient historical data available.")

        # ---- MONTE CARLO DCF ----
        st.divider()
        st.subheader("Monte Carlo DCF Simulation")
        st.caption("Stress-tests the DCF by running 1,500 scenarios with randomized growth, WACC, and terminal growth assumptions.")

        if st.button("Run Monte Carlo Simulation", key="mc_btn"):
            with st.spinner("Running 1,500 simulations..."):
                mc = monte_carlo_dcf(tkr, base_wacc=discount_rate,
                                     base_terminal_growth=terminal_growth)
            if "error" in mc:
                st.warning(mc["error"])
            else:
                st.session_state["mc_results"] = mc

        mc = st.session_state.get("mc_results")
        if mc and mc.get("ticker") == tkr:
            mc_col1, mc_col2, mc_col3, mc_col4 = st.columns(4)
            mc_col1.metric("Median Intrinsic Value", f"${mc['median']:.2f}")
            mc_col2.metric("Mean Intrinsic Value", f"${mc['mean']:.2f}")
            mc_col3.metric("Std Deviation", f"${mc['std']:.2f}")
            pct = mc['pct_undervalued']
            mc_col4.metric("Prob. Undervalued", f"{pct:.1f}%",
                           delta="Buy signal" if pct > 60 else "Caution" if pct < 40 else "Neutral",
                           delta_color="normal" if pct > 60 else "inverse" if pct < 40 else "off")

            mc_p1, mc_p2, mc_p3, mc_p4, mc_p5 = st.columns(5)
            mc_p1.metric("10th Pctl", f"${mc['p10']:.2f}")
            mc_p2.metric("25th Pctl", f"${mc['p25']:.2f}")
            mc_p3.metric("50th Pctl", f"${mc['p50']:.2f}")
            mc_p4.metric("75th Pctl", f"${mc['p75']:.2f}")
            mc_p5.metric("90th Pctl", f"${mc['p90']:.2f}")

            # Distribution chart
            fig_mc = go.Figure()
            fig_mc.add_trace(go.Histogram(
                x=mc["intrinsic_values"],
                nbinsx=60,
                marker_color="#1B6B4A",
                opacity=0.75,
                name="Simulated Values",
            ))
            # Current price line
            fig_mc.add_vline(x=mc["current_price"], line_dash="dash",
                             line_color="#DC2626", line_width=2,
                             annotation_text=f"Current: ${mc['current_price']:.2f}",
                             annotation_position="top right")
            # Median line
            fig_mc.add_vline(x=mc["median"], line_dash="dot",
                             line_color="#2563EB", line_width=2,
                             annotation_text=f"Median: ${mc['median']:.2f}",
                             annotation_position="top left")
            fig_mc.update_layout(
                title="Distribution of Intrinsic Values",
                xaxis_title="Intrinsic Value ($)",
                yaxis_title="Frequency",
                template="plotly_white",
                height=400,
                showlegend=False,
                margin=dict(l=50, r=20, t=50, b=40),
            )
            st.plotly_chart(fig_mc, use_container_width=True)

            # Interpretation
            if mc["pct_undervalued"] > 70:
                st.success(f"**{mc['pct_undervalued']:.1f}%** of scenarios suggest the stock is undervalued. The market price is below the median simulated intrinsic value, indicating potential upside across a wide range of assumptions.")
            elif mc["pct_undervalued"] > 40:
                st.info(f"**{mc['pct_undervalued']:.1f}%** of scenarios suggest undervaluation. The stock is near fair value — the current price falls within the core of the simulated distribution.")
            else:
                st.warning(f"Only **{mc['pct_undervalued']:.1f}%** of scenarios suggest undervaluation. The majority of simulated intrinsic values fall below the current price, indicating the stock may be overvalued on fundamentals.")

        # ---- RECOMMENDATION / CONCLUSION ----
        st.divider()
        st.subheader("Recommendation")

        # Gather all available signals
        signals = []
        signal_score = 0  # -100 to +100 scale

        # 1. DCF verdict
        if "error" not in dcf:
            if dcf["upside_pct"] > 15:
                signals.append(("Standard DCF suggests **undervalued** with {:.1f}% upside.".format(dcf["upside_pct"]), 1))
                signal_score += 20
            elif dcf["upside_pct"] < -15:
                signals.append(("Standard DCF suggests **overvalued** with {:.1f}% downside.".format(dcf["upside_pct"]), -1))
                signal_score -= 20
            else:
                signals.append(("Standard DCF suggests **fairly valued** ({:+.1f}%).".format(dcf["upside_pct"]), 0))

        # 2. Entropy-adjusted DCF
        if "error" not in dcf_adjusted:
            if dcf_adjusted["upside_pct"] > 15:
                signals.append(("Entropy-adjusted DCF (accounting for regime risk) also shows upside of {:.1f}%.".format(dcf_adjusted["upside_pct"]), 1))
                signal_score += 15
            elif dcf_adjusted["upside_pct"] < -15:
                signals.append(("Entropy-adjusted DCF shows {:.1f}% downside after adding regime risk premium.".format(dcf_adjusted["upside_pct"]), -1))
                signal_score -= 15
            else:
                signals.append(("Entropy-adjusted DCF shows fair value ({:+.1f}%) after risk premium.".format(dcf_adjusted["upside_pct"]), 0))

        # 3. Entropy score
        es = radar["entropy_score"]
        if es["composite_score"] > 70:
            signals.append(("Entropy score of **{}/100 (HIGH)** — the regime around this stock is actively shifting. High uncertainty warrants caution regardless of valuation.".format(es["composite_score"]), -1))
            signal_score -= 20
        elif es["composite_score"] > 45:
            signals.append(("Entropy score of **{}/100 (MODERATE)** — some unusual dynamics detected. Monitor for regime transition.".format(es["composite_score"]), 0))
            signal_score -= 5
        elif es["composite_score"] > 25:
            signals.append(("Entropy score of **{}/100 (LOW)** — stable regime. Valuation assumptions are more likely to hold.".format(es["composite_score"]), 1))
            signal_score += 10
        else:
            signals.append(("Entropy score of **{}/100 (VERY LOW)** — highly predictable regime. Strong foundation for valuation-based decisions.".format(es["composite_score"]), 1))
            signal_score += 15

        # 4. Monte Carlo if available
        mc = st.session_state.get("mc_results")
        if mc and mc.get("ticker") == tkr:
            pct = mc["pct_undervalued"]
            if pct > 60:
                signals.append(("Monte Carlo simulation: **{:.1f}%** of 1,500 scenarios suggest undervaluation — strong probabilistic support for upside.".format(pct), 1))
                signal_score += 20
            elif pct > 40:
                signals.append(("Monte Carlo simulation: **{:.1f}%** of scenarios suggest undervaluation — the stock is near fair value probabilistically.".format(pct), 0))
            else:
                signals.append(("Monte Carlo simulation: only **{:.1f}%** of scenarios suggest undervaluation — the majority of assumptions point to overvaluation.".format(pct), -1))
                signal_score -= 15

        # 5. Peer ranking
        if peers.get("peers") and peers.get("target"):
            target = peers["target"]
            peer_list = peers["peers"]
            all_pe = [p.get("pe_ratio") for p in [target] + peer_list if p.get("pe_ratio") is not None]
            if all_pe and target.get("pe_ratio") is not None:
                pe_rank = sorted(all_pe).index(target["pe_ratio"]) + 1
                if pe_rank <= 2:
                    signals.append(("Trades at a **lower P/E** than most peers — potentially cheaper relative to comparable companies.", 1))
                    signal_score += 5
                elif pe_rank >= len(all_pe) - 1:
                    signals.append(("Trades at a **higher P/E** than most peers — premium valuation relative to comparables.", -1))
                    signal_score -= 5

        # Build recommendation
        for text, _ in signals:
            st.markdown(f"- {text}")

        st.markdown("---")

        # Overall verdict
        if signal_score > 20:
            verdict_text = "BUY / UNDERVALUED"
            verdict_detail = "Multiple signals converge on undervaluation. The DCF models suggest intrinsic value above current price, the entropy regime is supportive, and the risk-reward profile is favorable."
            st.success(f"**Overall: {verdict_text}**\n\n{verdict_detail}")
        elif signal_score > 0:
            verdict_text = "HOLD / FAIRLY VALUED"
            verdict_detail = "Signals are mixed but lean slightly positive. The stock appears near fair value with some potential upside, but not enough margin of safety for a strong conviction buy."
            st.info(f"**Overall: {verdict_text}**\n\n{verdict_detail}")
        elif signal_score > -20:
            verdict_text = "HOLD / MONITOR"
            verdict_detail = "Signals are mixed with a slight bearish lean. Valuation appears stretched or the entropy regime introduces uncertainty that tempers confidence. Monitor for improvement in fundamentals or regime stabilization."
            st.info(f"**Overall: {verdict_text}**\n\n{verdict_detail}")
        else:
            verdict_text = "CAUTION / OVERVALUED"
            verdict_detail = "Multiple signals point to overvaluation or elevated regime risk. The current price appears above what fundamentals support across most scenarios. Consider waiting for a better entry point or reduced entropy."
            st.warning(f"**Overall: {verdict_text}**\n\n{verdict_detail}")

        st.caption("This recommendation is generated algorithmically based on DCF, entropy analysis, Monte Carlo simulation, and peer comparison. It is not financial advice.")

    # ---- ENTROPY RADAR TAB ----
    with tab_radar:
        st.subheader("Entropy Radar")
        st.caption("Is the regime around this stock changing?")

        es = radar["entropy_score"]
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Regime Instability", f"{es['regime_instability']}/100")
        s2.metric("Information Flow", f"{es['information_flow']}/100")
        s3.metric("Relationship Stress", f"{es['relationship_stress']}/100")
        s4.metric("Uncertainty", f"{es['uncertainty']}/100")

        st.divider()

        # Price + entropy chart
        fig = make_subplots(
            rows=2, cols=1, shared_xaxes=True,
            subplot_titles=(f"{tkr} Price (regime changes marked)", "Rolling Entropy (60-day)"),
            vertical_spacing=0.08, row_heights=[0.6, 0.4],
        )

        close = prices["Close"]
        fig.add_trace(go.Scatter(
            x=close.index, y=close.values, mode="lines",
            name="Price", line=dict(color="#1f77b4"),
        ), row=1, col=1)

        regimes = radar["regimes"]
        for cp_date in regimes.get("changepoint_dates", []):
            fig.add_vline(x=cp_date, line_dash="dash", line_color="red", opacity=0.6, row=1, col=1)

        roll_ent = radar.get("rolling_entropy")
        if roll_ent is not None and len(roll_ent) > 0:
            fig.add_trace(go.Scatter(
                x=roll_ent.index, y=roll_ent.values, mode="lines",
                name="Rolling Entropy", line=dict(color="#ff7f0e"),
            ), row=2, col=1)

        fig.update_layout(height=500, template="plotly_white", showlegend=False,
                          margin=dict(l=50, r=20, t=40, b=20))
        fig.update_yaxes(title_text="Price ($)", row=1, col=1)
        fig.update_yaxes(title_text="Entropy (nats)", row=2, col=1)
        st.plotly_chart(fig, use_container_width=True)

        if regimes["regimes"]:
            with st.expander(f"Regime Details ({regimes['n_changepoints']} changepoints)"):
                regime_df = pd.DataFrame(regimes["regimes"])
                regime_df.columns = ["Regime", "Start", "End", "Mean Return", "Volatility", "Days"]
                st.dataframe(regime_df, use_container_width=True, hide_index=True)

    # ---- INFORMATION FLOW TAB ----
    with tab_flow:
        st.subheader("Information Flow")
        st.caption("Who leads whom? Transfer entropy reveals directional causation.")

        te_results = radar.get("transfer_entropy", [])
        corr_results = radar.get("correlation_stability", [])

        if te_results:
            for te in te_results:
                comp = te["comparison"]
                col1, col2, col3 = st.columns([2, 2, 3])

                with col1:
                    st.metric(f"TE({tkr}→{comp})", f"{te['te_a_to_b']:.5f}")
                with col2:
                    st.metric(f"TE({comp}→{tkr})", f"{te['te_b_to_a']:.5f}")
                with col3:
                    leader = te["leader"]
                    if "neither" in leader:
                        st.info("Roughly symmetric information flow")
                    elif " leads " in leader and tkr in leader.split(" leads ")[0]:
                        st.success(f"**{tkr} leads {comp}** (net TE: {te['net_te']:.5f})")
                    else:
                        st.warning(f"**{comp} leads {tkr}** (net TE: {te['net_te']:.5f})")

                st.divider()

        if corr_results:
            st.subheader("Correlation Health")
            st.caption("Are the relationships this stock depends on still intact?")

            for corr in corr_results:
                comp = corr["comparison"]
                rolling_data = corr.get("rolling_data")

                cc1, cc2 = st.columns([1, 2])
                with cc1:
                    status = "🔴 COLLAPSED" if corr["is_collapsed"] else "🟢 INTACT"
                    st.metric(f"{tkr} vs {comp}", status)
                    curr = corr['current_correlation']
                    recent = corr['mean_recent_correlation']
                    st.caption(f"Current: {curr}, Recent avg: {recent}")

                with cc2:
                    if rolling_data is not None and len(rolling_data) > 0:
                        fig_corr = go.Figure()
                        fig_corr.add_trace(go.Scatter(
                            x=rolling_data.index, y=rolling_data.values,
                            mode="lines", line=dict(color="#ff7f0e"),
                        ))
                        fig_corr.add_hline(y=0, line_dash="dash", line_color="gray")
                        fig_corr.update_layout(
                            height=200, template="plotly_white",
                            margin=dict(l=40, r=20, t=10, b=20),
                            yaxis_title="Correlation",
                        )
                        st.plotly_chart(fig_corr, use_container_width=True)

                st.divider()

    # ---- SECTOR HEATMAP TAB ----
    with tab_heatmap:
        st.subheader("Sector Entropy Heatmap")
        st.caption("Real-time entropy scores across the stock universe — see which sectors and stocks carry the most regime risk right now.")

        if st.button("Scan All Sectors", key="heatmap_btn", type="primary"):
            with st.spinner("Scanning stocks across all sectors (this may take a minute)..."):
                hm = compute_sector_heatmap(period=period, max_per_sector=4)
            if "error" in hm:
                st.warning(hm["error"])
            else:
                st.session_state["heatmap_results"] = hm

        hm = st.session_state.get("heatmap_results")
        if hm and "error" not in hm:
            st.markdown(f"**{hm['n_stocks']} stocks analyzed** across {len(hm['sector_averages'])} sectors")

            # Sector averages bar chart
            sector_avg = hm["sector_averages"]
            sectors_sorted = sorted(sector_avg.items(), key=lambda x: x[1], reverse=True)

            fig_sectors = go.Figure()
            s_names = [s[0] for s in sectors_sorted]
            s_scores = [s[1] for s in sectors_sorted]
            colors = ["#DC2626" if s > 55 else "#F59E0B" if s > 40 else "#16A34A" for s in s_scores]
            fig_sectors.add_trace(go.Bar(
                x=s_scores, y=s_names,
                orientation="h",
                marker_color=colors,
                text=[f"{s:.1f}" for s in s_scores],
                textposition="outside",
            ))
            fig_sectors.update_layout(
                title="Average Entropy Score by Sector",
                xaxis_title="Entropy Score (0-100)",
                template="plotly_white",
                height=max(300, len(s_names) * 35),
                margin=dict(l=180, r=40, t=40, b=30),
                yaxis=dict(autorange="reversed"),
            )
            st.plotly_chart(fig_sectors, use_container_width=True)

            # Full heatmap table
            st.markdown("---")
            df = hm["results_df"].copy()
            display_cols = ["ticker", "sector", "composite_score", "regime_instability",
                           "relationship_stress", "uncertainty", "information_flow", "interpretation"]
            df_display = df[display_cols].sort_values("composite_score", ascending=False)
            df_display.columns = ["Ticker", "Sector", "Score", "Regime", "Stress", "Uncertainty", "Flow", "Reading"]
            st.dataframe(df_display, use_container_width=True, hide_index=True)

            # Top risk and most stable side by side
            col_risk, col_stable = st.columns(2)
            with col_risk:
                st.markdown("**Highest Entropy (Most Risk)**")
                for stock in hm["top_risk"][:5]:
                    score = stock["composite_score"]
                    color = "#DC2626" if score > 55 else "#F59E0B"
                    st.markdown(f'<span style="color:{color}; font-weight:700;">{score}</span> — **{stock["ticker"]}** ({stock["sector"]})', unsafe_allow_html=True)

            with col_stable:
                st.markdown("**Lowest Entropy (Most Stable)**")
                for stock in hm["most_stable"][:5]:
                    score = stock["composite_score"]
                    st.markdown(f'<span style="color:#16A34A; font-weight:700;">{score}</span> — **{stock["ticker"]}** ({stock["sector"]})', unsafe_allow_html=True)
        elif not hm:
            st.info("Click **Scan All Sectors** to compute entropy scores across the stock universe.")

    # ---- BACKTEST TAB ----
    with tab_backtest:
        st.subheader("Entropy Signal Backtest")
        st.caption("When entropy spikes, what happens to forward returns? Testing if entropy predicts drawdowns.")

        bt_col1, bt_col2 = st.columns([1, 3])
        with bt_col1:
            bt_period = st.selectbox("Backtest Period", ["2y", "5y", "10y"], index=1, key="bt_period")
            bt_threshold = st.slider("Signal Threshold (std devs)", 0.5, 2.5, 1.0, 0.25, key="bt_thresh")
            run_bt = st.button("Run Backtest", type="primary", use_container_width=True)

        if run_bt or "backtest_data" in st.session_state:
            if run_bt:
                with st.spinner("Running backtest..."):
                    bt_data = backtest_entropy_signals(
                        tkr, period=bt_period,
                        threshold_std=bt_threshold,
                    )
                    st.session_state["backtest_data"] = bt_data
                    st.session_state["bt_ticker"] = tkr

            bt = st.session_state.get("backtest_data", {})
            if "error" in bt:
                st.warning(bt["error"])
            elif bt.get("n_signals", 0) == 0:
                st.info("No high-entropy signals detected in this period. Try lowering the threshold.")
            else:
                # Summary metrics
                with bt_col2:
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Signals Found", bt["n_signals"])
                    m2.metric("Trading Days", bt["n_trading_days"])
                    m3.metric("Entropy Threshold", f"{bt['threshold']:.3f}")
                    m4.metric("Signal Rate", f"{bt['n_signals'] / (bt['n_trading_days'] / 252):.1f}/year")

                # Signal vs Baseline comparison table
                st.markdown("#### Signal Returns vs Baseline")
                st.caption("Does high entropy predict worse forward returns than the unconditional average?")

                comparison_rows = []
                for horizon in ["30d", "60d", "90d"]:
                    sig = bt["signal_stats"].get(horizon, {})
                    base = bt["baseline_stats"].get(horizon, {})
                    if sig and base:
                        edge = sig["mean_return"] - base["mean_return"]
                        comparison_rows.append({
                            "Horizon": horizon,
                            "Signal Avg Return": f"{sig['mean_return']:+.2f}%",
                            "Baseline Avg Return": f"{base['mean_return']:+.2f}%",
                            "Edge": f"{edge:+.2f}%",
                            "Signal Hit Rate (neg)": f"{sig['hit_rate_negative']:.0f}%",
                            "Baseline Hit Rate (neg)": f"{base['hit_rate_negative']:.0f}%",
                            "Worst Signal": f"{sig['worst']:+.2f}%",
                            "Best Signal": f"{sig['best']:+.2f}%",
                            "N": sig["n_signals"],
                        })

                if comparison_rows:
                    st.dataframe(pd.DataFrame(comparison_rows), use_container_width=True, hide_index=True)

                # Chart: price with signal markers + entropy
                fig_bt = make_subplots(
                    rows=2, cols=1, shared_xaxes=True,
                    subplot_titles=(f"{tkr} Price with High-Entropy Signals", "Rolling Entropy"),
                    vertical_spacing=0.08, row_heights=[0.6, 0.4],
                )

                bt_close = bt["close_prices"]
                bt_roll = bt["rolling_entropy"]

                fig_bt.add_trace(go.Scatter(
                    x=bt_close.index, y=bt_close.values, mode="lines",
                    name="Price", line=dict(color="#1f77b4"),
                ), row=1, col=1)

                # Mark signal dates on price chart
                signal_dates = bt["signal_dates"]
                signal_prices = [float(bt_close.loc[d]) for d in signal_dates if d in bt_close.index]
                signal_x = [d for d in signal_dates if d in bt_close.index]
                fig_bt.add_trace(go.Scatter(
                    x=signal_x, y=signal_prices, mode="markers",
                    name="High Entropy Signal",
                    marker=dict(color="red", size=10, symbol="triangle-down"),
                ), row=1, col=1)

                # Entropy with threshold line
                fig_bt.add_trace(go.Scatter(
                    x=bt_roll.index, y=bt_roll.values, mode="lines",
                    name="Rolling Entropy", line=dict(color="#ff7f0e"),
                ), row=2, col=1)
                fig_bt.add_hline(y=bt["threshold"], line_dash="dash",
                                 line_color="red", opacity=0.7, row=2, col=1,
                                 annotation_text="Signal Threshold")

                fig_bt.update_layout(height=550, template="plotly_white",
                                     margin=dict(l=50, r=20, t=40, b=20))
                fig_bt.update_yaxes(title_text="Price ($)", row=1, col=1)
                fig_bt.update_yaxes(title_text="Entropy (nats)", row=2, col=1)
                st.plotly_chart(fig_bt, use_container_width=True)

                # Individual signal table
                with st.expander(f"All Signals ({bt['n_signals']})"):
                    sig_df = pd.DataFrame(bt["signals"])
                    display_cols = ["date", "entropy", "price_at_signal"]
                    for d in [30, 60, 90]:
                        col_name = f"return_{d}d"
                        if col_name in sig_df.columns:
                            display_cols.append(col_name)
                    st.dataframe(sig_df[display_cols], use_container_width=True, hide_index=True)
        else:
            st.markdown("""
            **How this works:**
            1. Computes rolling Shannon entropy over the full history
            2. Flags dates where entropy exceeds the threshold (mean + N standard deviations)
            3. Measures what happens to the stock price 30, 60, and 90 days after each signal
            4. Compares signal returns against the unconditional baseline

            **If high entropy is predictive**, signal returns should be worse than baseline returns.

            Click **Run Backtest** to test the thesis.
            """)

    # ---- ACCURACY TAB ----
    with tab_accuracy:
        st.subheader("Accuracy Testing")
        st.caption("Validate the entropy framework across multiple stocks and against analyst consensus.")

        acc_test = st.selectbox("Select Test", [
            "Cross-Stock Signal Validation",
            "Analyst Target Comparison",
        ], key="acc_test")

        acc_col1, acc_col2 = st.columns([1, 3])

        if acc_test == "Cross-Stock Signal Validation":
            with acc_col1:
                custom_tickers = st.text_area(
                    "Test Universe (one per line)",
                    value="AAPL\nMSFT\nNVDA\nJPM\nGS\nJNJ\nXOM\nAMZN\nTSLA\nMETA",
                    height=200, key="acc_tickers"
                )
                acc_period = st.selectbox("Period", ["2y", "5y"], index=1, key="acc_period")
                acc_thresh = st.number_input("Threshold (std devs)", min_value=0.5, max_value=3.0, value=1.0, step=0.25, key="acc_thresh")
                run_acc = st.button("Run Validation", type="primary", use_container_width=True, key="run_acc")

            if run_acc or "cross_stock_results" in st.session_state:
                if run_acc:
                    ticker_list = [t.strip().upper() for t in custom_tickers.strip().split("\n") if t.strip()]
                    with st.spinner(f"Backtesting {len(ticker_list)} stocks... (this may take a minute)"):
                        cross_results = run_cross_stock_backtest(
                            tickers=ticker_list, period=acc_period,
                            threshold_std=acc_thresh,
                        )
                        st.session_state["cross_stock_results"] = cross_results

                cr = st.session_state.get("cross_stock_results", {})
                if "error" in cr:
                    st.warning(cr["error"])
                else:
                    with acc_col2:
                        st.markdown("#### Aggregate Results")
                        st.caption(f"Tested {cr['n_stocks']} stocks | {cr['n_errors']} errors")

                    # Aggregate stats per horizon
                    agg_rows = []
                    for horizon in ["30d", "60d", "90d"]:
                        agg = cr["aggregate"].get(horizon, {})
                        if agg:
                            agg_rows.append({
                                "Horizon": horizon,
                                "Mean Edge": f"{agg['mean_edge']:+.2f}%",
                                "Median Edge": f"{agg['median_edge']:+.2f}%",
                                "Stocks Confirming": f"{agg['stocks_with_negative_edge']}/{agg['total_stocks']} ({agg['pct_confirming']:.0f}%)",
                                "t-statistic": agg.get("t_statistic", "N/A"),
                                "Significant (t<-1.65)": "Yes" if isinstance(agg.get("t_statistic"), (int, float)) and agg["t_statistic"] < -1.65 else "No",
                            })
                    if agg_rows:
                        st.dataframe(pd.DataFrame(agg_rows), use_container_width=True, hide_index=True)

                    st.markdown("**Interpretation:**")
                    best_horizon = None
                    best_pct = 0
                    for horizon in ["30d", "60d", "90d"]:
                        agg = cr["aggregate"].get(horizon, {})
                        if agg.get("pct_confirming", 0) > best_pct:
                            best_pct = agg["pct_confirming"]
                            best_horizon = horizon

                    if best_pct >= 70:
                        st.success(f"Strong validation: {best_pct:.0f}% of stocks show worse returns after high-entropy signals at the {best_horizon} horizon. The entropy signal generalizes well.")
                    elif best_pct >= 50:
                        st.info(f"Moderate validation: {best_pct:.0f}% of stocks confirm the signal at {best_horizon}. The framework has some predictive value but is not universal.")
                    else:
                        st.warning(f"Weak validation: only {best_pct:.0f}% confirm at {best_horizon}. The entropy signal may not generalize to this universe, or the threshold needs tuning.")

                    # Per-stock breakdown
                    st.markdown("#### Per-Stock Results")
                    per_stock_rows = []
                    for r in cr["results"]:
                        row = {"Ticker": r["ticker"], "Signals": r["n_signals"]}
                        for h in ["30d", "60d", "90d"]:
                            edge = r.get(f"edge_{h}")
                            if edge is not None:
                                row[f"Edge {h}"] = f"{edge:+.2f}%"
                                row[f"Hit Rate {h}"] = f"{r.get(f'hit_rate_{h}', 0):.0f}%"
                        per_stock_rows.append(row)
                    st.dataframe(pd.DataFrame(per_stock_rows), use_container_width=True, hide_index=True)

                    if cr["errors"]:
                        with st.expander(f"Errors ({cr['n_errors']})"):
                            st.dataframe(pd.DataFrame(cr["errors"]), use_container_width=True, hide_index=True)

        else:  # Analyst Target Comparison
            with acc_col1:
                custom_tickers_analyst = st.text_area(
                    "Test Universe (one per line)",
                    value="AAPL\nMSFT\nNVDA\nJPM\nGS\nJNJ\nXOM\nAMZN",
                    height=200, key="acc_tickers_analyst"
                )
                base_wacc_acc = st.number_input("Base WACC (%)", min_value=1.0, max_value=30.0, value=10.0, step=0.5, key="acc_wacc") / 100
                run_analyst = st.button("Run Comparison", type="primary", use_container_width=True, key="run_analyst")

            if run_analyst or "analyst_results" in st.session_state:
                if run_analyst:
                    ticker_list = [t.strip().upper() for t in custom_tickers_analyst.strip().split("\n") if t.strip()]
                    with st.spinner(f"Analyzing {len(ticker_list)} stocks vs analyst targets..."):
                        analyst_results = run_analyst_comparison(
                            tickers=ticker_list, base_wacc=base_wacc_acc,
                        )
                        st.session_state["analyst_results"] = analyst_results

                ar = st.session_state.get("analyst_results", {})
                if "error" in ar:
                    st.warning(ar["error"])
                else:
                    with acc_col2:
                        st.markdown("#### DCF vs Analyst Consensus")
                        st.caption(f"Comparing standard and entropy-adjusted DCF against analyst mean price targets for {ar['n_stocks']} stocks.")

                    comp = ar.get("method_comparison", {})
                    if comp:
                        m1, m2 = st.columns(2)
                        m1.metric("Entropy-Adjusted Closer", f"{comp.get('Entropy-Adjusted', 0)} stocks")
                        m2.metric("Standard DCF Closer", f"{comp.get('Standard', 0)} stocks")

                    # Detailed table
                    display_rows = []
                    for r in ar["results"]:
                        row = {
                            "Ticker": r["ticker"],
                            "Price": f"${r.get('current_price', 0):.2f}" if r.get("current_price") else "N/A",
                            "Analyst Target": f"${r['analyst_target']:.2f}" if r.get("analyst_target") else "N/A",
                            "Standard DCF": f"${r['dcf_standard']:.2f}" if r.get("dcf_standard") else "N/A",
                            "Entropy DCF": f"${r['dcf_entropy_adjusted']:.2f}" if r.get("dcf_entropy_adjusted") else "N/A",
                            "Entropy Score": r.get("entropy_score", "N/A"),
                            "Closer": r.get("closer_to_analyst", "N/A"),
                        }
                        display_rows.append(row)
                    st.dataframe(pd.DataFrame(display_rows), use_container_width=True, hide_index=True)

                    st.markdown("**Note:** Analyst targets reflect 12-month forward expectations and incorporate qualitative factors. "
                                "Our DCF is a pure quantitative model. The comparison shows whether entropy adjustment "
                                "moves valuations in a direction consistent with market consensus.")

    # ---- AI CHAT TAB ----
    with tab_chat:
        st.subheader("AI Entropy Analyst")
        st.caption("Ask questions about the current stock, run custom entropy analyses, or explore cross-asset relationships.")

        def render_chat_content(text):
            """Render chat text, handling LaTeX blocks and loose dollar signs."""
            import re
            # Extract LaTeX blocks ($$...$$) and inline math ($...$) first
            parts = re.split(r'(\$\$[\s\S]*?\$\$|\$[^$\n]+?\$)', text)
            for part in parts:
                if part.startswith("$$") and part.endswith("$$"):
                    # Display block LaTeX
                    st.latex(part.strip("$").strip())
                elif part.startswith("$") and part.endswith("$") and len(part) > 2:
                    # Inline LaTeX — render as block since st.markdown can mangle it
                    st.latex(part.strip("$").strip())
                else:
                    # Regular text — escape any stray dollar signs
                    safe = part.replace("$", "\\$")
                    if safe.strip():
                        st.markdown(safe)

        # Display chat history
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                render_chat_content(msg["content"])

        # Chat input
        chat_input = st.chat_input("e.g. 'Who leads whom between NVDA and AMD?' or 'What drives this stock's entropy score?'", key="chat")

        if chat_input:
            if not client:
                st.warning("OpenAI API key not configured. Add it to your .env file.")
            else:
                st.session_state.messages.append({"role": "user", "content": chat_input})
                with st.chat_message("user"):
                    st.markdown(chat_input)

                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        context = ""
                        if "analysis_data" in st.session_state:
                            ad = st.session_state["analysis_data"]
                            context = "\nCurrently analyzing: {}. Info: {}\nDCF: {}\nEntropy Radar: {}".format(
                                ad['ticker'],
                                json.dumps({k: v for k, v in ad['info'].items() if v is not None}, default=str),
                                json.dumps({k: v for k, v in ad['dcf'].items() if k != 'projected_fcf'}, default=str),
                                json.dumps(ad.get('radar', {}), default=str),
                            )

                        api_messages = [{"role": "system", "content": SYSTEM_PROMPT + context}]
                        for m in st.session_state.messages:
                            api_messages.append({"role": m["role"], "content": m["content"]})

                        response = client.chat.completions.create(
                            model="gpt-4o-mini", messages=api_messages,
                            tools=TOOL_DEFINITIONS, tool_choice="auto",
                        )
                        message = response.choices[0].message

                        # Handle tool calls
                        while message.tool_calls:
                            api_messages.append({
                                "role": "assistant", "content": message.content,
                                "tool_calls": [{"id": tc.id, "type": "function", "function": {"name": tc.function.name, "arguments": tc.function.arguments}} for tc in message.tool_calls]
                            })
                            for tc in message.tool_calls:
                                result = execute_chat_tool(tc.function.name, json.loads(tc.function.arguments))
                                api_messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})
                            response = client.chat.completions.create(
                                model="gpt-4o-mini", messages=api_messages,
                                tools=TOOL_DEFINITIONS, tool_choice="auto",
                            )
                            message = response.choices[0].message

                        reply = message.content or "Analysis complete."
                        st.session_state.messages.append({"role": "assistant", "content": reply})
                        render_chat_content(reply)

    # ---- METHODOLOGY TAB ----
    with tab_method:
        st.subheader("Methodology")
        st.caption("How this tool works and why it works this way.")

        st.markdown("""### Overview

This tool combines **traditional valuation** (DCF, financial ratios) with an **entropy intelligence layer** built on information theory. The core thesis: traditional metrics tell you what a stock is worth *today*, but entropy tells you whether the *regime* around that stock is changing — and therefore whether today's valuation assumptions will hold tomorrow.

---

### 1. Shannon Entropy (Uncertainty Measurement)

**What:** Measures the disorder/uncertainty in a stock's return distribution.

**How:** Returns are discretized into histogram bins, then Shannon entropy is computed:""")

        st.latex(r"H(X) = -\sum_{x} p(x) \cdot \ln(p(x))")

        st.markdown("""Higher entropy means returns are spread across many bins (unpredictable). Lower entropy means returns are concentrated (predictable).

**Why it matters:** A stock with high Shannon entropy has a wider range of possible outcomes. This should be reflected in the discount rate.

**Reference:** Shannon, C.E. (1948). "A Mathematical Theory of Communication."

---

### 2. Transfer Entropy (Directional Information Flow)

**What:** Measures how much knowing Asset X's past reduces uncertainty about Asset Y's future — beyond what Y's own past tells you.

**How:**""")

        st.latex(r"TE(X \rightarrow Y) = H(Y_t \mid Y_{t-1}) - H(Y_t \mid Y_{t-1}, X_{t-\text{lag}})")

        st.markdown("""Computed by discretizing returns, building joint probability distributions, and measuring conditional entropy reduction.

**Why it matters:** Correlation tells you assets move together. Transfer entropy tells you *who leads and who follows*. If SPY leads a stock (high TE from SPY to stock), the stock is a market follower. If the stock leads SPY, it may be a market driver.

**Reference:** Schreiber, T. (2000). "Measuring Information Transfer."

---

### 3. Regime Detection (Changepoint Analysis)

**What:** Identifies points in time where the statistical properties of returns fundamentally change.

**How:** Uses the PELT (Pruned Exact Linear Time) algorithm from the `ruptures` library. PELT finds changepoints by minimizing a cost function that penalizes both poor segment fits and too many segments.

**Why it matters:** A stock that has undergone recent regime changes is in a period of structural uncertainty. Valuations calibrated to the old regime may not hold.

**Reference:** Killick, R., Fearnhead, P., Eckley, I.A. (2012). "Optimal Detection of Changepoints with a Linear Computational Cost."

---

### 4. Correlation Stability (Entropy Collapse Detection)

**What:** Monitors whether the rolling correlation between a stock and its benchmarks (SPY, sector ETF) is stable or has collapsed.

**How:** Computes 60-day rolling Pearson correlation. If the mean recent correlation drops below |0.1|, the relationship is flagged as "collapsed" — a permanent structural break.

**Why it matters:** Diversification relies on stable correlations. When a stock's correlation to its sector or the market collapses, it signals the stock is decoupling from normal market dynamics — a risk that standard models miss.

**Inspired by:** Case 4 (Europe Energy Crisis) — EU-Russia gas correlation collapsed permanently after 2022.

---

### 5. Composite Entropy Score (0-100)

**What:** Synthesizes the four entropy dimensions into a single actionable score.

**How:** Weighted average of four component scores:""")

        st.latex(r"S_{\text{composite}} = 0.30 \cdot S_{\text{regime}} + 0.25 \cdot S_{\text{stress}} + 0.25 \cdot S_{\text{uncertainty}} + 0.20 \cdot S_{\text{flow}}")

        st.markdown("Each component is scored 0-100:")

        st.latex(r"S_{\text{regime}} = \min\!\bigl(100,\; 20 \times n_{\text{changepoints}} + \text{recency bonus}\bigr)")
        st.latex(r"S_{\text{stress}} = \begin{cases} 80 & \text{if correlation collapsed} \\ (1 - |\bar{\rho}_{\text{recent}}|) \times 60 & \text{otherwise} \end{cases}")
        st.latex(r"S_{\text{uncertainty}} = \min\!\left(100,\; \frac{H - 2.0}{2.0} \times 100\right)")
        st.latex(r"S_{\text{flow}} = \min\!\bigl(100,\; |TE_{\text{net}}| \times 5000\bigr)")

        st.markdown("""
**Interpretation:**
- **>70:** HIGH — Rules are actively changing. Structural break risk.
- **45-70:** MODERATE — Unusual dynamics detected. Monitor closely.
- **25-45:** LOW — Stable regime. Normal market behavior.
- **<25:** VERY LOW — Highly predictable within current regime.

---

### 6. Entropy-Adjusted DCF

**What:** Modifies the traditional DCF discount rate based on the entropy score.

**How:**""")

        st.latex(r"\text{Adjusted WACC} = \text{Base WACC} + \left(\frac{\text{Entropy Score}}{100}\right)^{0.7} \times \text{Max Premium}")

        st.markdown("""The exponent 0.7 creates a concave curve — moderate entropy (30-60) has meaningful impact, while the marginal effect diminishes at extreme scores. Default max premium is 6%.

**Why it matters:** A stock in a high-entropy regime faces more uncertainty about future cash flows. The discount rate should reflect this. A stable regime (low entropy) means the base WACC is appropriate. A disrupted regime (high entropy) demands a higher discount rate.

**Novel contribution:** No existing tool dynamically adjusts DCF discount rates based on information-theoretic regime analysis.

---

### 7. Signal Backtesting

**What:** Tests whether high-entropy signals predict negative forward returns.

**How:**
1. Compute rolling Shannon entropy (60-day window) over 2-5 years of history
2. Flag dates where entropy exceeds mean + N standard deviations
3. Cluster nearby signals (>10 days apart = separate signal)
4. Measure forward returns at 30, 60, and 90 days after each signal
5. Compare signal returns against unconditional baseline returns

**What to look for:**
- **Negative edge:** Signal returns should be worse than baseline
- **High hit rate:** >50% of signals should precede negative returns
- **Statistical significance:** t-statistic < -1.65 (90% confidence one-tailed)

---

### Data Sources and Limitations

| Source | What | Limitations |
|--------|------|-------------|
| yfinance | Prices, fundamentals, analyst targets | Free but may have gaps; not institutional grade |
| scipy.stats | Shannon entropy computation | Well-established, no known issues |
| ruptures | Changepoint detection (PELT) | Sensitivity depends on penalty parameter |
| OpenAI | AI-generated analysis summary | May hallucinate; always verify against numbers |

**Key limitations:**
- **DCF assumes positive FCF** — Companies with negative FCF cannot be valued via DCF
- **Transfer entropy needs sufficient data** — Short periods (<6 months) produce noisy estimates
- **Entropy score weights are heuristic** — The 30/25/25/20 weighting is a judgment call, not empirically optimized
- **Backtest is in-sample** — Threshold is computed on the same data used for testing (future work: walk-forward validation)
- **No transaction costs** — Backtest returns don't account for trading friction

---

### References

1. Shannon, C.E. (1948). "A Mathematical Theory of Communication." *Bell System Technical Journal*, 27(3), 379-423.
2. Schreiber, T. (2000). "Measuring Information Transfer." *Physical Review Letters*, 85(2), 461.
3. Killick, R., Fearnhead, P., Eckley, I.A. (2012). "Optimal Detection of Changepoints with a Linear Computational Cost." *JASA*, 107(500), 1590-1598.
4. MGMT 69000 Case Studies: Tariff Shock (textual entropy), Europe Energy (structural collapse), Japan Carry Trade (transfer entropy).
        """)

    # ---- DETAILS TAB ----
    with tab_detail:
        st.subheader("Raw Metrics")

        d1, d2 = st.columns(2)
        with d1:
            st.markdown("**Entropy Metrics**")
            st.write(f"- Shannon Entropy: {radar['shannon_entropy']} nats")
            st.write(f"- Current Rolling Entropy: {radar.get('current_rolling_entropy', 'N/A')}")
            st.write(f"- Mean Rolling Entropy: {radar.get('mean_rolling_entropy', 'N/A')}")
            st.write(f"- Peak Entropy Date: {radar.get('max_entropy_date', 'N/A')}")
            st.write(f"- Regime Changes: {regimes['n_changepoints']}")
            st.write(f"- Trading Days Analyzed: {radar['n_days']}")

        with d2:
            st.markdown("**Entropy Score Breakdown**")
            es = radar["entropy_score"]
            st.write(f"- **Composite: {es['composite_score']}/100**")
            st.write(f"- Regime Instability: {es['regime_instability']}/100 (30% weight)")
            st.write(f"- Information Flow: {es['information_flow']}/100 (20% weight)")
            st.write(f"- Relationship Stress: {es['relationship_stress']}/100 (25% weight)")
            st.write(f"- Uncertainty: {es['uncertainty']}/100 (25% weight)")

        # AI Summary
        if client:
            st.divider()
            st.subheader("AI Analysis")

            # Cache AI analysis to avoid regenerating on every rerun
            if "ai_analysis" not in st.session_state or st.session_state.get("ai_ticker") != tkr:
                with st.spinner("Generating analysis..."):
                    te_summary = "; ".join([
                        "{}: {}".format(te['comparison'], te['leader'])
                        for te in radar.get("transfer_entropy", [])
                    ])
                    corr_summary = "; ".join([
                        "{}: {}".format(
                            c['comparison'],
                            'COLLAPSED' if c['is_collapsed'] else 'intact ({})'.format(c['mean_recent_correlation'])
                        )
                        for c in radar.get("correlation_stability", [])
                    ])

                    prompt = f"""Analyze {tkr} ({info['name']}) using this data:

Valuation: P/E={info['pe_ratio']}, P/B={info['pb_ratio']}, EV/EBITDA={info['ev_ebitda']}, ROE={fmt_pct(info['roe'])}
DCF: {json.dumps({k: dcf[k] for k in ['intrinsic_value', 'current_price', 'upside_pct', 'verdict', 'fcf_growth_rate'] if k in dcf}, default=str)}

Entropy Radar Score: {radar['entropy_score']['composite_score']}/100
- Regime Instability: {radar['entropy_score']['regime_instability']}/100 ({regimes['n_changepoints']} changepoints)
- Uncertainty: {radar['entropy_score']['uncertainty']}/100 (Shannon entropy: {radar['shannon_entropy']})
- Information Flow: {radar['entropy_score']['information_flow']}/100 ({te_summary})
- Relationship Stress: {radar['entropy_score']['relationship_stress']}/100 ({corr_summary})

Write a concise 3-4 paragraph analysis:
1. What the entropy radar reveals — is this stock in a stable regime or are the rules changing?
2. Valuation context — is the current price justified given the entropy profile?
3. What to watch — specific signals or thresholds that would change the thesis."""

                    response = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[
                            {"role": "system", "content": "You are a financial analyst who specializes in entropy-based market analysis. Be concise, reference specific numbers, and provide actionable insights. Avoid generic statements."},
                            {"role": "user", "content": prompt},
                        ],
                    )
                    st.session_state["ai_analysis"] = response.choices[0].message.content
                    st.session_state["ai_ticker"] = tkr

            # Escape $ signs so Streamlit doesn't render them as LaTeX
            safe_text = st.session_state["ai_analysis"].replace("$", "\\$")
            st.markdown(safe_text)

else:
    st.title("📊 Fintropy")
    st.markdown("""
    **Enter a ticker and click Analyze** to get:
    - Traditional valuation (DCF, key ratios)
    - Entropy Radar score — is the game changing around this stock?
    - Information flow — who leads whom?
    - Correlation health — are relationships breaking?
    """)
