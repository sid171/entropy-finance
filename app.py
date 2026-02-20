"""Entropy Finance — Valuation Terminal with Entropy Radar.

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

from config import SYSTEM_PROMPT, TOOL_DEFINITIONS
from market_data import get_returns, get_price_data
from entropy_tools import shannon_entropy, net_transfer_entropy, rolling_entropy, detect_regimes, correlation_stability
from valuation import get_company_info, compute_dcf, entropy_adjusted_wacc
from entropy_radar import run_entropy_radar
from backtest import backtest_entropy_signals

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
load_dotenv()
st.set_page_config(page_title="Entropy Finance", page_icon="📊", layout="wide")

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
    st.title("📊 Entropy Finance")
    st.caption("Valuation + Entropy Radar")

    ticker = st.text_input("Ticker", value="AAPL", placeholder="AAPL, NVDA, TSLA...")
    period = st.selectbox("Period", ["6mo", "1y", "2y", "5y"], index=1)
    analyze_btn = st.button("🔍 Analyze", type="primary", use_container_width=True)

    with st.expander("DCF Parameters"):
        discount_rate = st.slider("Discount Rate (WACC)", 0.06, 0.15, 0.10, 0.01)
        terminal_growth = st.slider("Terminal Growth", 0.01, 0.05, 0.03, 0.005)

    st.divider()
    st.subheader("AI Assistant")
    if "messages" not in st.session_state:
        st.session_state.messages = []
    chat_container = st.container(height=250)
    with chat_container:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
    chat_input = st.chat_input("Ask a question...", key="chat")

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

        # Cache everything for re-rendering
        st.session_state["cached_analysis"] = {
            "info": info,
            "dcf": dcf,
            "dcf_adjusted": dcf_adjusted,
            "wacc_adj": wacc_adj,
            "prices": prices,
            "radar": radar,
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
    tab_val, tab_radar, tab_flow, tab_backtest, tab_detail = st.tabs([
        "📈 Valuation", "🎯 Entropy Radar", "🔀 Information Flow", "🔬 Backtest", "📋 Details"
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
                st.dataframe(fcf_df, use_container_width=True, hide_index=True)

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
    st.title("📊 Entropy Finance")
    st.markdown("""
    **Enter a ticker and click Analyze** to get:
    - Traditional valuation (DCF, key ratios)
    - Entropy Radar score — is the game changing around this stock?
    - Information flow — who leads whom?
    - Correlation health — are relationships breaking?
    """)

# ---------------------------------------------------------------------------
# Chat handler
# ---------------------------------------------------------------------------
if chat_input and client:
    st.session_state.messages.append({"role": "user", "content": chat_input})

    context = ""
    if "analysis_data" in st.session_state:
        ad = st.session_state["analysis_data"]
        context = f"\nCurrently analyzing: {ad['ticker']}. Info: {json.dumps({k: v for k, v in ad['info'].items() if v is not None}, default=str)}\nDCF: {json.dumps({k: v for k, v in ad['dcf'].items() if k != 'projected_fcf'}, default=str)}\nEntropy Radar: {json.dumps(ad.get('radar', {}), default=str)}"

    api_messages = [{"role": "system", "content": SYSTEM_PROMPT + context}]
    for msg in st.session_state.messages:
        api_messages.append({"role": msg["role"], "content": msg["content"]})

    response = client.chat.completions.create(
        model="gpt-4o-mini", messages=api_messages,
        tools=TOOL_DEFINITIONS, tool_choice="auto",
    )
    message = response.choices[0].message

    while message.tool_calls:
        api_messages.append({
            "role": "assistant", "content": message.content,
            "tool_calls": [{"id": tc.id, "type": "function", "function": {"name": tc.function.name, "arguments": tc.function.arguments}} for tc in message.tool_calls]
        })
        for tc in message.tool_calls:
            result = execute_chat_tool(tc.function.name, json.loads(tc.function.arguments))
            api_messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})
        response = client.chat.completions.create(model="gpt-4o-mini", messages=api_messages, tools=TOOL_DEFINITIONS, tool_choice="auto")
        message = response.choices[0].message

    content = message.content or "Analysis complete."
    st.session_state.messages.append({"role": "assistant", "content": content})
    st.rerun()
