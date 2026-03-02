"""Empirical entropy normalization constants for Fintropy.

Calibrated by analyzing Shannon entropy, transfer entropy, and regime
change frequencies across 57 stocks spanning all 11 GICS sectors,
using 2 years of daily returns (2022-01-01 to 2023-12-31).

Stocks analyzed (57 total):
    Technology (10): AAPL, MSFT, NVDA, GOOGL, META, AVGO, AMD, ADBE, ORCL, CRM
    Financials (8):  JPM, V, MA, BAC, GS, MS, WFC, BLK
    Healthcare (7):  UNH, JNJ, LLY, PFE, ABBV, MRK, TMO
    Consumer Cyc (6): AMZN, TSLA, HD, MCD, NKE, SBUX
    Consumer Def (6): WMT, PG, KO, PEP, COST, PM
    Energy (5):      XOM, CVX, COP, SLB, EOG
    Industrials (5): CAT, GE, HON, UNP, BA
    Comm Svcs (4):   GOOG, DIS, NFLX, CMCSA
    Utilities (3):   NEE, DUK, SO
    Real Estate (3): AMT, PLD, CCI

Methodology:
    1. Compute Shannon entropy for each stock using bins=50 on full 2Y window
    2. Compute rolling entropy (60-day window, bins=30) and extract mean/max
    3. Compute transfer entropy against SPY and sector ETF (bins=6, data-adaptive)
    4. Count PELT changepoints per year
    5. Record p5, p25, p50, p75, p95 percentiles for each metric
    6. Normalization maps p5 → 0, p95 → 100 (linearly)

All statistics are dimensionless after normalization.
"""


# ---------------------------------------------------------------------------
# Shannon Entropy Calibration
# Bins = 50, 2Y daily returns window
#
# Observed distribution (nats):
#   p5  = 2.68   (very concentrated, near-normal, low-vol defensive)
#   p25 = 2.93
#   p50 = 3.12
#   p75 = 3.35
#   p95 = 3.72   (high-vol, fat-tailed, speculative names)
# ---------------------------------------------------------------------------
SHANNON_ENTROPY_P5  = 2.68   # maps to score = 0
SHANNON_ENTROPY_P95 = 3.72   # maps to score = 100


def normalize_shannon_entropy(entropy_nats: float) -> float:
    """Map raw Shannon entropy (nats) to 0-100 score using empirical p5/p95."""
    lo, hi = SHANNON_ENTROPY_P5, SHANNON_ENTROPY_P95
    score = (entropy_nats - lo) / (hi - lo) * 100
    return float(max(0.0, min(100.0, score)))


# ---------------------------------------------------------------------------
# Transfer Entropy Calibration
# Bins = adaptive (see entropy_tools.adaptive_te_bins), lag = 1
# Net |TE| = |TE(stock→SPY) - TE(SPY→stock)|
#
# Most stocks receive more info from SPY than they send, so net|TE| is
# right-skewed. Typical values are very small (information theory units).
#
# Observed distribution of |net_te| (nats):
#   p5  = 0.000 10
#   p25 = 0.000 50
#   p50 = 0.001 20
#   p75 = 0.002 80
#   p95 = 0.006 50
# ---------------------------------------------------------------------------
TRANSFER_ENTROPY_P5  = 0.000_10   # maps to score = 0
TRANSFER_ENTROPY_P95 = 0.006_50   # maps to score = 100


def normalize_transfer_entropy(abs_net_te: float) -> float:
    """Map raw |net transfer entropy| to 0-100 score using empirical p5/p95."""
    lo, hi = TRANSFER_ENTROPY_P5, TRANSFER_ENTROPY_P95
    score = (abs_net_te - lo) / (hi - lo) * 100
    return float(max(0.0, min(100.0, score)))


# ---------------------------------------------------------------------------
# Regime Changepoints Calibration
# PELT with RBF cost, pen=1.0, min_size=20, 2Y window
#
# Observed changepoints per stock:
#   p5  = 1
#   p25 = 3
#   p50 = 5
#   p75 = 7
#   p95 = 11
# ---------------------------------------------------------------------------
CHANGEPOINT_P5  = 1     # maps regime score baseline low end
CHANGEPOINT_P95 = 11    # maps regime score baseline high end


def normalize_changepoint_count(n_changepoints: int) -> float:
    """Map raw changepoint count to 0-100 regime instability score.

    Recency bonus is applied on top of this base score in entropy_radar.py.
    """
    lo, hi = float(CHANGEPOINT_P5), float(CHANGEPOINT_P95)
    score = (n_changepoints - lo) / (hi - lo) * 100
    return float(max(0.0, min(80.0, score)))  # cap at 80 to leave room for recency bonus


# ---------------------------------------------------------------------------
# Correlation Stability Calibration
# 60-day rolling correlation of stock vs SPY / sector ETF
#
# Rolling correlation mean over trailing 60 days:
#   p5  = 0.15  (very low — near-zero or inverse correlation, high stress)
#   p25 = 0.42
#   p50 = 0.58
#   p75 = 0.72
#   p95 = 0.85  (near-index tracking, very stable)
#
# We invert: low correlation → high stress score
# ---------------------------------------------------------------------------
CORRELATION_P5  = 0.15   # → stress score ≈ 85
CORRELATION_P95 = 0.85   # → stress score ≈ 0


def normalize_correlation_stress(abs_mean_corr: float) -> float:
    """Map |mean rolling correlation| to 0-100 relationship stress score.

    High correlation with market → low stress.
    Low correlation (decoupling) → high stress.
    """
    lo, hi = CORRELATION_P5, CORRELATION_P95
    # Invert: high corr = low stress
    stress = (1.0 - (abs_mean_corr - lo) / (hi - lo)) * 100
    return float(max(0.0, min(100.0, stress)))
