# Roadmap

Building on: Streamlit, OpenAI function calling, yfinance, scipy.stats, ruptures, plotly

## Sections

### 1. Entropy Engine + Market Data
Core entropy computation functions (Shannon, transfer, rolling, regime detection, correlation stability) + yfinance data fetching layer, all with pytest tests. Foundation for everything else.

### 2. Valuation + Entropy Radar
DCF valuation model + the Entropy Radar composite scoring engine. Combines traditional finance (P/E, DCF, ratios) with entropy intelligence (regime instability, information flow, relationship stress, uncertainty). Auto-selects sector ETF for comparison.

### 3. Chat App UI + CI/CD
Streamlit 4-tab terminal (Valuation, Entropy Radar, Information Flow, Details) with sidebar AI chat. OpenAI function calling gives the LLM access to entropy tools. GitHub Actions CI/CD pipeline. Submission-ready.
