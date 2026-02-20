"""Entropy-based financial analysis tools.

Implements five entropy frameworks from MGMT 69000 case studies:
1. Shannon Entropy — uncertainty/disorder in return distributions
2. Transfer Entropy — directional information flow between assets
3. Rolling Entropy — time-varying entropy for trend analysis
4. Regime Detection — changepoint detection for structural breaks
5. Entropy Collapse — detect when correlations permanently break
"""

import numpy as np
import pandas as pd
from scipy.stats import entropy as scipy_entropy
import ruptures


# ---------------------------------------------------------------------------
# 1. Shannon Entropy
# ---------------------------------------------------------------------------

def shannon_entropy(returns: pd.Series, bins: int = 50) -> float:
    """Compute Shannon entropy of a return distribution.

    Higher entropy = more uncertainty/disorder in returns.
    Lower entropy = more predictable/concentrated returns.

    Args:
        returns: Series of asset returns.
        bins: Number of histogram bins for discretization.

    Returns:
        Shannon entropy in nats.
    """
    clean = returns.dropna()
    if len(clean) == 0:
        return 0.0
    counts, _ = np.histogram(clean, bins=bins, density=False)
    total = counts.sum()
    if total == 0:
        return 0.0
    probs = counts / total
    probs = probs[probs > 0]
    return scipy_entropy(probs)


# ---------------------------------------------------------------------------
# 2. Transfer Entropy
# ---------------------------------------------------------------------------

def _discretize(series: pd.Series, bins: int = 10) -> np.ndarray:
    """Discretize a continuous series into bin indices."""
    _, bin_edges = np.histogram(series.dropna(), bins=bins)
    return np.digitize(series.values, bin_edges[:-1]) - 1


def _joint_prob(x: np.ndarray, y: np.ndarray, n_bins: int) -> np.ndarray:
    """Compute joint probability distribution of two discretized series."""
    joint = np.zeros((n_bins, n_bins))
    for i in range(len(x)):
        if 0 <= x[i] < n_bins and 0 <= y[i] < n_bins:
            joint[x[i], y[i]] += 1
    total = joint.sum()
    if total > 0:
        joint /= total
    return joint


def transfer_entropy(source: pd.Series, target: pd.Series,
                     lag: int = 1, bins: int = 10) -> float:
    """Compute transfer entropy from source to target.

    TE(X→Y) measures how much knowing X's past reduces uncertainty
    about Y's future, beyond what Y's own past tells you.

    TE(X→Y) = H(Y_future | Y_past) - H(Y_future | Y_past, X_past)

    Args:
        source: Source asset returns (X).
        target: Target asset returns (Y).
        lag: Number of lag periods.
        bins: Number of bins for discretization.

    Returns:
        Transfer entropy in nats. Higher = more info flow from source to target.
    """
    # Align series
    df = pd.DataFrame({"source": source, "target": target}).dropna()
    if len(df) < lag + 2:
        return 0.0

    # Create lagged variables
    y_future = df["target"].iloc[lag:].values
    y_past = df["target"].iloc[:-lag].values
    x_past = df["source"].iloc[:-lag].values

    # Discretize
    y_f = _discretize(pd.Series(y_future), bins)
    y_p = _discretize(pd.Series(y_past), bins)
    x_p = _discretize(pd.Series(x_past), bins)

    # H(Y_future | Y_past) via joint entropy
    p_yf_yp = _joint_prob(y_f, y_p, bins)
    p_yp = p_yf_yp.sum(axis=0)

    h_yf_given_yp = 0.0
    for j in range(bins):
        if p_yp[j] > 0:
            cond = p_yf_yp[:, j] / p_yp[j]
            cond = cond[cond > 0]
            h_yf_given_yp -= p_yp[j] * np.sum(cond * np.log(cond + 1e-12))

    # H(Y_future | Y_past, X_past) via triple joint
    # Encode (Y_past, X_past) as a single variable
    yx_past = y_p * bins + x_p
    n_joint_bins = bins * bins

    # Joint of y_future and (y_past, x_past)
    joint_triple = np.zeros((bins, n_joint_bins))
    for i in range(len(y_f)):
        yf_idx = y_f[i]
        yx_idx = yx_past[i]
        if 0 <= yf_idx < bins and 0 <= yx_idx < n_joint_bins:
            joint_triple[yf_idx, yx_idx] += 1
    total = joint_triple.sum()
    if total > 0:
        joint_triple /= total

    p_yx = joint_triple.sum(axis=0)

    h_yf_given_yp_xp = 0.0
    for j in range(n_joint_bins):
        if p_yx[j] > 0:
            cond = joint_triple[:, j] / p_yx[j]
            cond = cond[cond > 0]
            h_yf_given_yp_xp -= p_yx[j] * np.sum(cond * np.log(cond + 1e-12))

    te = h_yf_given_yp - h_yf_given_yp_xp
    return max(te, 0.0)  # TE is non-negative by definition


def net_transfer_entropy(series_a: pd.Series, series_b: pd.Series,
                         lag: int = 1, bins: int = 10) -> dict:
    """Compute transfer entropy in both directions and determine leader.

    Returns:
        Dict with TE(A→B), TE(B→A), net TE, and which asset leads.
    """
    te_a_to_b = transfer_entropy(series_a, series_b, lag, bins)
    te_b_to_a = transfer_entropy(series_b, series_a, lag, bins)
    net = te_a_to_b - te_b_to_a

    if abs(net) < 0.001:
        leader = "neither (roughly symmetric)"
    elif net > 0:
        leader = f"{series_a.name} leads {series_b.name}"
    else:
        leader = f"{series_b.name} leads {series_a.name}"

    return {
        "te_a_to_b": round(te_a_to_b, 6),
        "te_b_to_a": round(te_b_to_a, 6),
        "net_te": round(net, 6),
        "leader": leader,
    }


# ---------------------------------------------------------------------------
# 3. Rolling Entropy
# ---------------------------------------------------------------------------

def rolling_entropy(returns: pd.Series, window: int = 60,
                    bins: int = 30) -> pd.Series:
    """Compute rolling Shannon entropy over a sliding window.

    Useful for tracking how uncertainty evolves over time.
    Spikes in rolling entropy may signal regime transitions.

    Args:
        returns: Series of asset returns.
        window: Rolling window size in trading days.
        bins: Number of histogram bins.

    Returns:
        Series of entropy values with same index as returns.
    """
    result = pd.Series(index=returns.index, dtype=float, name="rolling_entropy")
    values = returns.values

    for i in range(window, len(values)):
        window_data = values[i - window:i]
        window_data = window_data[~np.isnan(window_data)]
        if len(window_data) > 1:
            counts, _ = np.histogram(window_data, bins=bins, density=False)
            probs = counts / counts.sum()
            probs = probs[probs > 0]
            result.iloc[i] = scipy_entropy(probs)

    return result.dropna()


# ---------------------------------------------------------------------------
# 4. Regime Detection (Changepoint Detection)
# ---------------------------------------------------------------------------

def detect_regimes(returns: pd.Series, n_regimes: int = 5,
                   model: str = "rbf") -> dict:
    """Detect regime changes in a return series using ruptures.

    Args:
        returns: Series of asset returns.
        n_regimes: Expected number of regimes (changepoints = n_regimes - 1).
        model: Cost model for ruptures ('rbf', 'l2', 'l1', 'normal').

    Returns:
        Dict with changepoint indices, dates, and regime statistics.
    """
    values = returns.dropna().values.reshape(-1, 1)
    dates = returns.dropna().index

    algo = ruptures.Pelt(model=model, min_size=20).fit(values)
    changepoints = algo.predict(pen=1.0)

    # Remove the last index (always included by ruptures)
    changepoints = [cp for cp in changepoints if cp < len(values)]

    # Build regime info
    boundaries = [0] + changepoints + [len(values)]
    regimes = []
    for i in range(len(boundaries) - 1):
        start_idx = boundaries[i]
        end_idx = boundaries[i + 1]
        segment = values[start_idx:end_idx].flatten()
        start_date = dates[start_idx]
        end_date = dates[min(end_idx - 1, len(dates) - 1)]
        regimes.append({
            "regime": i + 1,
            "start": str(start_date.date()) if hasattr(start_date, 'date') else str(start_date),
            "end": str(end_date.date()) if hasattr(end_date, 'date') else str(end_date),
            "mean_return": round(float(np.mean(segment)), 6),
            "volatility": round(float(np.std(segment)), 6),
            "days": end_idx - start_idx,
        })

    changepoint_dates = [
        str(dates[cp].date()) if hasattr(dates[cp], 'date') else str(dates[cp])
        for cp in changepoints if cp < len(dates)
    ]

    return {
        "n_changepoints": len(changepoints),
        "changepoint_dates": changepoint_dates,
        "regimes": regimes,
    }


# ---------------------------------------------------------------------------
# 5. Entropy Collapse (Correlation Stability)
# ---------------------------------------------------------------------------

def correlation_stability(returns_a: pd.Series, returns_b: pd.Series,
                          window: int = 60) -> pd.DataFrame:
    """Measure rolling correlation to detect entropy collapse.

    When correlation drops to ~0 and stays there, it signals a permanent
    structural break (entropy collapse) — the relationship has ceased to exist.

    Args:
        returns_a: First asset returns.
        returns_b: Second asset returns.
        window: Rolling window size.

    Returns:
        DataFrame with rolling correlation and stability metrics.
    """
    df = pd.DataFrame({"a": returns_a, "b": returns_b}).dropna()
    rolling_corr = df["a"].rolling(window).corr(df["b"])

    # Detect if correlation has collapsed (near zero for extended period)
    recent_corr = rolling_corr.tail(window)
    mean_recent = recent_corr.mean()
    is_collapsed = abs(mean_recent) < 0.1

    return {
        "rolling_correlation": rolling_corr.dropna(),
        "current_correlation": round(float(rolling_corr.iloc[-1]), 4) if len(rolling_corr.dropna()) > 0 else None,
        "mean_recent_correlation": round(float(mean_recent), 4) if not np.isnan(mean_recent) else None,
        "is_collapsed": bool(is_collapsed),
        "interpretation": (
            "Correlation has collapsed — structural break detected. "
            "The historical relationship appears to have permanently broken."
            if is_collapsed else
            f"Correlation is active at {mean_recent:.3f}. "
            "The relationship is still functioning."
        ),
    }
