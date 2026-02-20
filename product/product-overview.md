# Entropy Finance — Valuation Terminal with Entropy Radar

## The Problem
Traditional valuation tools (P/E, DCF) tell you what a stock is worth *today*, but they don't tell you whether the regime around that stock is changing. Information theory provides tools to detect regime shifts, directional causation, and structural breaks — but these are trapped in academic papers and custom scripts. There's no tool that synthesizes entropy concepts into an actionable intelligence layer alongside traditional valuation.

## Success Looks Like
Enter a ticker, get a comprehensive analysis in seconds:
- Traditional valuation (DCF, key ratios) for fundamental assessment
- **Entropy Radar Score (0-100)** answering: "How much is the game changing around this stock?"
- Information flow analysis showing who leads whom (stock vs market, stock vs sector)
- Correlation health monitoring to detect if key relationships are breaking
- AI-generated synthesis combining valuation + entropy insights

## Building On (Existing Foundations)
- **yfinance** — Free market data (prices, fundamentals, cash flows)
- **scipy.stats.entropy** — Shannon entropy computation
- **ruptures** — Changepoint detection (PELT algorithm) for regime analysis
- **Streamlit** — Tabbed UI with sidebar chat
- **OpenAI function calling** — LLM selects and executes entropy tools based on natural language
- **Plotly** — Interactive charts with regime overlays

## The Unique Part
**The Entropy Radar** — a composite entropy score that synthesizes five information-theoretic frameworks into a single actionable metric, layered on top of traditional valuation:

1. **Regime Instability** (30%) — Changepoint frequency and recency via PELT algorithm
2. **Relationship Stress** (25%) — Rolling correlation stability vs market and sector ETF
3. **Uncertainty** (25%) — Shannon entropy of return distribution vs baseline
4. **Information Flow** (20%) — Transfer entropy asymmetry (who leads whom)

This is inspired by (not a copy of) the MGMT 69000 case studies:
- Case 1 (Tariff Shock): Shannon entropy as uncertainty measurement
- Case 4 (Europe Energy): Correlation collapse as irreversible structural break
- Case 5 (Japan Carry): Transfer entropy for directional causation chains

The novel contribution is **synthesizing these into a composite score** with automatic sector-aware peer comparison.

## Tech Stack
- **UI:** Streamlit (4-tab layout + sidebar AI chat)
- **LLM:** OpenAI GPT-4o-mini (function calling for chat, analysis generation)
- **Data:** yfinance (prices, fundamentals, financials)
- **Entropy:** scipy.stats, numpy, ruptures
- **Valuation:** DCF model with adjustable WACC/terminal growth
- **Charts:** Plotly (interactive)
- **CI/CD:** GitHub Actions (pytest on Python 3.11/3.12)

## Architecture
```
entropy-finance-app/
├── app.py              # Streamlit UI — 4 tabs + sidebar chat
├── entropy_radar.py    # Composite entropy scoring engine
├── entropy_tools.py    # 5 entropy computation functions
├── market_data.py      # yfinance data layer
├── valuation.py        # DCF + company fundamentals
├── config.py           # System prompts + tool definitions
├── tests/
│   └── test_entropy.py # 12 pytest tests
└── product/            # DRIVER methodology docs
```
