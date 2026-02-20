# Fintropy

A financial analysis terminal that combines traditional valuation (DCF, ratios) with an **entropy intelligence layer** — detecting when the rules around a stock are changing using information theory.

## What It Does

Enter a ticker and get instant analysis across four dimensions:

1. **Valuation** — DCF intrinsic value, key ratios (P/E, P/B, EV/EBITDA, ROE, margins)
2. **Entropy Radar** — Composite entropy score (0-100) showing regime stability, with price chart overlaid with detected changepoints and rolling entropy
3. **Information Flow** — Transfer entropy reveals who leads whom (stock vs market, stock vs sector ETF), plus correlation health monitoring
4. **AI Analysis** — GPT-powered synthesis of valuation + entropy data with actionable insights

### The Entropy Radar Score

The composite score (0-100) answers: **"How much is the game changing around this stock?"**

| Component | Weight | What It Measures |
|-----------|--------|-----------------|
| Regime Instability | 30% | Changepoint frequency and recency |
| Relationship Stress | 25% | Correlation stability vs market/sector |
| Uncertainty | 25% | Shannon entropy of return distribution |
| Information Flow | 20% | Transfer entropy asymmetry |

| Score | Interpretation |
|-------|---------------|
| >70 | HIGH — Rules are actively changing. Structural break risk. |
| 45-70 | MODERATE — Unusual dynamics. Monitor for regime transition. |
| 25-45 | LOW — Stable regime. Normal market dynamics. |
| <25 | VERY LOW — Highly predictable within current regime. |

## Entropy Frameworks

Five frameworks from information theory applied to financial markets:

| Framework | What It Measures | Inspired By |
|-----------|-----------------|-------------|
| **Shannon Entropy** | Uncertainty/disorder in returns | Case 1: Tariff Shock |
| **Transfer Entropy** | Directional information flow (who leads whom) | Case 5: Japan Carry Trade |
| **Rolling Entropy** | Time-varying uncertainty | Case 4: Europe Energy Crisis |
| **Regime Detection** | Structural breaks (changepoint detection) | Case 4: Correlation Collapse |
| **Entropy Collapse** | Permanent correlation breakdown | Case 4: EU-Russia structural break |

## Setup

### Prerequisites
- Python 3.11+
- OpenAI API key

### Installation

```bash
git clone https://github.com/sid171/entropy-finance.git
cd entropy-finance
pip install -r requirements.txt
```

### Configuration

```bash
cp .env.example .env
# Edit .env and add your OpenAI API key
```

### Run

```bash
streamlit run app.py
```

## Architecture

```
entropy-finance-app/
├── app.py              # Streamlit UI — 4-tab terminal with sidebar chat
├── entropy_radar.py    # Composite entropy scoring engine
├── entropy_tools.py    # 5 entropy computation functions
├── market_data.py      # yfinance data layer
├── valuation.py        # DCF model + company fundamentals
├── config.py           # System prompts + OpenAI tool definitions
├── requirements.txt    # Dependencies
├── tests/
│   └── test_entropy.py # Pytest suite (12 tests)
├── product/            # DRIVER methodology artifacts
│   ├── product-overview.md
│   └── product-roadmap.md
└── .github/
    └── workflows/
        └── ci.yml      # CI/CD — tests on Python 3.11 + 3.12
```

### How It Works

```
User enters ticker → Analyze button
    │
    ├── Valuation: yfinance → company info + DCF model
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
```

## Testing

```bash
python -m pytest tests/ -v
```

## Tech Stack

- **UI:** Streamlit (tabbed layout + sidebar chat)
- **LLM:** OpenAI GPT-4o-mini (function calling for chat)
- **Data:** yfinance (prices, fundamentals, cash flows)
- **Entropy:** scipy.stats, numpy, ruptures (PELT)
- **Charts:** Plotly (interactive price + entropy charts)
- **CI/CD:** GitHub Actions (pytest on Python 3.11/3.12)

## DRIVER Methodology

Built following the DRIVER framework (MGMT 69000):

1. **Define** — Researched entropy/info-theory libraries (ruptures, scipy, infomeasure) and financial app patterns
2. **Represent** — 3-section roadmap: Entropy Engine, Chat + UI, CI/CD + Polish
3. **Implement** — Iterative build: entropy tools first (with tests), then valuation, then entropy radar composite, then UI
4. **Validate** — 12 passing tests + live demo with real market data
5. **Evolve** — Public GitHub repo with CI/CD
6. **Reflect** — Key learning: entropy concepts from case studies become more powerful when synthesized into a composite score

## References

- Shannon, C.E. (1948). "A Mathematical Theory of Communication"
- Schreiber, T. (2000). "Measuring Information Transfer" — Transfer entropy formulation
- Killick, R., Fearnhead, P., Eckley, I.A. (2012). "Optimal Detection of Changepoints" — PELT algorithm
- MGMT 69000 Case Studies: Tariff Shock (textual entropy), Europe Energy (structural collapse), Japan Carry Trade (transfer entropy)

---

*MGMT 69000: Mastering AI for Finance | Purdue MSF | DRIVER Framework*
