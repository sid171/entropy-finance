Fintropy


A financial analysis terminal that combines traditional valuation (DCF, Monte Carlo, ratios) with an entropy intelligence layer — detecting when the rules around a stock are changing using information theory.

What It Does

Enter a ticker and get instant analysis across four dimensions:


Valuation — DCF intrinsic value, a 1,500-scenario Monte Carlo DCF, an entropy-adjusted WACC, and key ratios (P/E, P/B, EV/EBITDA, ROE, margins) with peer comparison
Entropy Radar — Composite entropy score (0-100) showing regime stability, with price chart overlaid with detected changepoints and rolling entropy
Information Flow — Transfer entropy reveals who leads whom (stock vs market, stock vs sector ETF), plus correlation health monitoring
AI Analysis — GPT-powered synthesis of valuation + entropy data with actionable insights


The Entropy Radar Score

The composite score (0-100) answers: "How much is the game changing around this stock?"

ComponentWeightWhat It MeasuresRegime Instability30%Changepoint frequency and recencyRelationship Stress25%Correlation stability vs market/sectorUncertainty25%Shannon entropy of return distributionInformation Flow20%Transfer entropy asymmetry

ScoreInterpretation>70HIGH — Rules are actively changing. Structural break risk.45-70MODERATE — Unusual dynamics. Monitor for regime transition.25-45LOW — Stable regime. Normal market dynamics.<25VERY LOW — Highly predictable within current regime.

Entropy Frameworks

Five frameworks from information theory applied to financial markets:

FrameworkWhat It MeasuresInspired ByShannon EntropyUncertainty/disorder in returnsCase 1: Tariff ShockTransfer EntropyDirectional information flow (who leads whom)Case 5: Japan Carry TradeRolling EntropyTime-varying uncertaintyCase 4: Europe Energy CrisisRegime DetectionStructural breaks (changepoint detection)Case 4: Correlation CollapseEntropy CollapsePermanent correlation breakdownCase 4: EU-Russia structural break

Valuation Engine

The valuation layer does not just report a DCF — it stress-tests and adjusts it:


DCF — intrinsic value from projected free cash flows discounted at a CAPM-based WACC.
Monte Carlo DCF — 1,500 simulations randomizing WACC, growth, and terminal growth. The WACC uncertainty is derived per ticker via CAPM error propagation rather than a universal fixed assumption:


  σ(WACC) ≈ (E/V) × √[ (ERP × σ_β)² + σ_Rf² ]


Entropy-Adjusted WACC — layers a regime-risk premium onto the base discount rate as a concave function of the composite entropy score, up to +600 bps at maximum instability. At entropy = 0, no premium is applied. This embeds regime uncertainty directly into the valuation rather than leaving it as a qualitative caveat.
Peer comparison — ranks financially similar companies via a multi-factor similarity score.


Setup

Prerequisites


Python 3.11+
OpenAI API key


Installation

git clone https://github.com/sid171/entropy-finance.git
cd entropy-finance
pip install -r requirements.txt

Configuration

cp .env.example .env
# Edit .env and add your OpenAI API key

Run

streamlit run app.py

Architecture

entropy-finance/
├── app.py                  # Streamlit UI — 4-tab terminal with sidebar chat
├── entropy_radar.py        # Composite Entropy Radar scoring engine
├── entropy_tools.py        # 5 entropy functions (Shannon, transfer, rolling, regime, correlation)
├── entropy_calibration.py  # Empirical p5/p95 normalization across 57 stocks / 11 GICS sectors
├── valuation.py            # DCF, Monte Carlo DCF, entropy-adjusted WACC, peer comparison
├── backtest.py             # In-sample / out-of-sample entropy signal backtester
├── heatmap.py              # Parallel sector entropy heatmap (ThreadPoolExecutor)
├── accuracy.py             # Cross-stock validation of the entropy framework
├── market_data.py          # yfinance data layer
├── config.py               # System prompts + OpenAI tool definitions
├── requirements.txt        # Dependencies
├── tests/
│   ├── test_entropy.py      # 12 tests — entropy computations + boundary conditions
│   ├── test_calibration.py  # 26 tests — empirical p5/p95 normalization
│   ├── test_valuation.py    # 15 tests — DCF, Monte Carlo, entropy-adjusted WACC
│   └── test_backtest.py     #  9 tests — in-sample / out-of-sample signal backtest
├── product/                # DRIVER methodology artifacts
│   ├── product-overview.md
│   └── product-roadmap.md
└── .github/
    └── workflows/
        └── ci.yml          # CI/CD — ruff lint + pytest on Python 3.11 & 3.12

How It Works

User enters ticker → Analyze button
    │
    ├── Valuation: yfinance → company info → DCF
    │      ├── Monte Carlo DCF (1,500 scenarios, CAPM-propagated σ(WACC))
    │      └── Entropy-adjusted WACC (regime-risk premium up to +600 bps)
    │
    └── Entropy Radar: yfinance → returns
         ├── Shannon entropy (uncertainty level)
         ├── Rolling entropy (60-day window)
         ├── Regime detection (PELT changepoint algorithm)
         ├── Transfer entropy vs SPY + sector ETF
         ├── Correlation stability vs SPY + sector ETF
         └── Composite score (weighted blend)

Sidebar chat: OpenAI function calling → entropy tools
    LLM decides which tool to call → computes on real data → explains results

Testing

62 tests across four suites, run on every push via GitHub Actions on Python 3.11 and 3.12:

SuiteTestsFocustest_entropy.py12Entropy computations + boundary conditionstest_calibration.py26Empirical p5/p95 normalizationtest_valuation.py15DCF, Monte Carlo, entropy-adjusted WACCtest_backtest.py9In-sample / out-of-sample signal backtest

The CI pipeline gates a ruff lint stage before the test stage, and enforces a 90% coverage threshold on the pure-logic modules (entropy_tools, entropy_calibration, backtest) that run without live network calls.

Run locally:

python -m pytest tests/ -v

Tech Stack


UI: Streamlit (tabbed layout + sidebar chat)
LLM: OpenAI GPT-4o-mini (function calling for chat)
Data: yfinance (prices, fundamentals, cash flows)
Entropy: scipy.stats, numpy, ruptures (PELT)
Charts: Plotly (interactive price + entropy charts)
CI/CD: GitHub Actions (ruff lint + pytest on Python 3.11/3.12)


DRIVER Methodology

Built following the DRIVER framework (MGMT 69000):


Define — Researched entropy/info-theory libraries (ruptures, scipy, infomeasure) and financial app patterns
Represent — 3-section roadmap: Entropy Engine, Chat + UI, CI/CD + Polish
Implement — Iterative build: entropy tools first (with tests), then valuation, then entropy radar composite, then UI
Validate — 62 passing tests + live demo with real market data
Evolve — Public GitHub repo with CI/CD
Reflect — Key learning: entropy concepts from case studies become more powerful when synthesized into a composite score


Attribution

Built following the DRIVER framework using the DRIVER plugin by Cinder Zhang.

Open-source libraries used: Streamlit, OpenAI Python SDK, yfinance, pandas, NumPy, SciPy, ruptures, Plotly, pytest, ruff.

References


Shannon, C.E. (1948). "A Mathematical Theory of Communication"
Schreiber, T. (2000). "Measuring Information Transfer" — Transfer entropy formulation
Killick, R., Fearnhead, P., Eckley, I.A. (2012). "Optimal Detection of Changepoints" — PELT algorithm
MGMT 69000 Case Studies: Tariff Shock (textual entropy), Europe Energy (structural collapse), Japan Carry Trade (transfer entropy)



MGMT 69000: Mastering AI for Finance | Purdue MSF | DRIVER Framework
