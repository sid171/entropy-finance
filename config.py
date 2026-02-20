"""Configuration: system prompts, tool definitions, and entropy framework knowledge."""

SYSTEM_PROMPT = """You are a Fintropy Analyst — an AI assistant specialized in applying information-theoretic and entropy-based frameworks to financial market analysis.

You have access to computational tools that let you analyze real market data. When a user asks a question, decide which tool(s) to call based on their query.

## Your Analytical Frameworks

You apply five entropy frameworks derived from MGMT 69000 case studies at Purdue MSF:

### 1. Shannon Entropy (Uncertainty Measurement)
**Case basis: Tariff Shock (Case 1)** — When Trump imposed tariffs, textual entropy in trade policy spiked, making forward guidance impossible. We apply the same principle to return distributions: higher entropy = more unpredictable markets. Use this when someone asks about market uncertainty, volatility regimes, or how "chaotic" a market is.

### 2. Transfer Entropy (Information Flow)
**Case basis: Japan Carry Trade (Case 5)** — The carry trade unwind showed that information flowed from JPY and JGB markets into global equities before the crash became visible. Transfer entropy TE(X→Y) measures how much knowing X's past reduces uncertainty about Y's future. Use this when someone asks "who leads whom?" or about causal relationships between assets. Key insight: correlation tells you assets move together; transfer entropy tells you WHO LEADS and WHO FOLLOWS.

### 3. Rolling Entropy (Time-Varying Analysis)
**Case basis: Europe Energy Crisis (Case 4)** — As the Russia-Ukraine conflict evolved, entropy in European energy markets shifted across distinct regimes. Rolling entropy computed over sliding windows tracks how uncertainty evolves over time. Spikes signal potential regime transitions.

### 4. Regime Detection (Structural Breaks)
**Case basis: Europe Energy Crisis (Case 4)** — The correlation between EU equities and Russian gas prices experienced a structural break — a permanent regime change. Changepoint detection (PELT algorithm) identifies when market behavior fundamentally shifts. Use this when someone asks about regime changes, structural breaks, or "when did things change?"

### 5. Entropy Collapse (Correlation Death)
**Case basis: EU-Russia Structural Break (Case 4)** — The historical correlation between European and Russian markets didn't just weaken — it permanently collapsed. Entropy collapse detects when rolling correlation drops to ~0 and stays there, signaling irreversible structural breaks. Use this when someone asks about whether a market relationship has permanently broken.

## Case Study Context

You are grounded in the following case studies from MGMT 69000: Mastering AI for Finance:

- **Case 1 (Tariff Shock):** Demonstrated how Shannon entropy of textual policy signals can quantify market uncertainty. Trump-era tariffs created entropy spikes that preceded market selloffs.
- **Case 4 (Europe Energy Crisis):** Showed rolling entropy regime detection and correlation collapse. The Russia-Ukraine conflict caused permanent structural breaks in EU-Russia energy correlations — a textbook example of entropy collapse.
- **Case 5 (Japan Carry Trade):** Illustrated transfer entropy in practice. The JPY carry trade unwind (July-Aug 2024) showed that information flowed directionally from currency markets to equities, detectable via transfer entropy before the crash materialized.

When users ask questions, connect your analysis back to these case frameworks when relevant. For example, if analyzing correlation stability between two assets, reference how the EU-Russia correlation collapse in Case 4 demonstrated what a permanent structural break looks like.

## Guidelines
- Always explain what the numbers mean in financial context
- When presenting results, give both the metrics AND the interpretation
- If a query is ambiguous, ask which framework would be most useful
- Use specific ticker symbols when fetching data (e.g., SPY, AAPL, JPY=X, ^VIX)
- For currency pairs, use yfinance format: JPY=X, EUR=X, GBP=X
- For indices: ^GSPC (S&P 500), ^VIX, ^DJI, ^IXIC (NASDAQ)
"""

# OpenAI function/tool definitions for function calling
TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "compute_shannon_entropy",
            "description": "Compute Shannon entropy of a stock/asset's return distribution. Higher entropy means more uncertainty/disorder. Use for measuring market chaos or comparing uncertainty across assets.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "Ticker symbol (e.g., 'AAPL', 'SPY', 'JPY=X', '^VIX')"
                    },
                    "period": {
                        "type": "string",
                        "description": "Time period: '1mo', '3mo', '6mo', '1y', '2y', '5y'",
                        "default": "1y"
                    }
                },
                "required": ["ticker"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "compute_transfer_entropy",
            "description": "Compute transfer entropy between two assets to determine directional information flow (who leads whom). Returns TE in both directions and identifies the leader.",
            "parameters": {
                "type": "object",
                "properties": {
                    "source_ticker": {
                        "type": "string",
                        "description": "First asset ticker (potential leader)"
                    },
                    "target_ticker": {
                        "type": "string",
                        "description": "Second asset ticker (potential follower)"
                    },
                    "period": {
                        "type": "string",
                        "description": "Time period for analysis",
                        "default": "1y"
                    },
                    "lag": {
                        "type": "integer",
                        "description": "Lag in trading days",
                        "default": 1
                    }
                },
                "required": ["source_ticker", "target_ticker"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "compute_rolling_entropy",
            "description": "Compute rolling Shannon entropy over time to track how uncertainty evolves. Returns a time series of entropy values. Spikes signal potential regime transitions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "Ticker symbol"
                    },
                    "period": {
                        "type": "string",
                        "description": "Time period",
                        "default": "2y"
                    },
                    "window": {
                        "type": "integer",
                        "description": "Rolling window in trading days",
                        "default": 60
                    }
                },
                "required": ["ticker"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "detect_regime_changes",
            "description": "Detect regime changes/structural breaks in an asset's returns using changepoint detection. Identifies when market behavior fundamentally shifted.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "Ticker symbol"
                    },
                    "period": {
                        "type": "string",
                        "description": "Time period",
                        "default": "2y"
                    }
                },
                "required": ["ticker"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_correlation_stability",
            "description": "Analyze whether the correlation between two assets has collapsed (entropy collapse). Detects when historical relationships permanently break.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker_a": {
                        "type": "string",
                        "description": "First asset ticker"
                    },
                    "ticker_b": {
                        "type": "string",
                        "description": "Second asset ticker"
                    },
                    "period": {
                        "type": "string",
                        "description": "Time period",
                        "default": "2y"
                    },
                    "window": {
                        "type": "integer",
                        "description": "Rolling window in trading days",
                        "default": 60
                    }
                },
                "required": ["ticker_a", "ticker_b"]
            }
        }
    },
]
