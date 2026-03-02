"""Entropy Signal Backtester — does high entropy predict drawdowns?

Tests the core thesis: when Shannon entropy spikes, does it predict
negative forward returns? This validates whether the entropy framework
is predictive, not just descriptive.

Uses a strict in-sample / out-of-sample split: the entropy threshold is
calibrated on the first `train_pct` of the history (in-sample), and
signals are generated and evaluated only on the remaining data
(out-of-sample). This prevents look-ahead bias in threshold selection.
"""

import numpy as np
from market_data import get_price_data
from entropy_tools import rolling_entropy


def backtest_entropy_signals(ticker: str, period: str = "5y",
                             entropy_window: int = 60,
                             threshold_std: float = 1.0,
                             forward_days: list[int] | None = None,
                             train_pct: float = 0.60) -> dict:
    """Backtest: when entropy spikes, what happens to forward returns?

    Method:
    1. Compute rolling Shannon entropy over the full history
    2. Split entropy series at train_pct: in-sample for calibration,
       out-of-sample for signal generation and evaluation
    3. Compute threshold from in-sample mean/std only (no look-ahead)
    4. Identify "high entropy" signals in out-of-sample only:
       entropy > in-sample mean + threshold_std * in-sample std
    5. For each signal date, measure forward returns at 30/60/90 days
    6. Compare signal returns vs out-of-sample unconditional baseline

    Args:
        ticker: Stock ticker symbol.
        period: Historical period to analyze.
        entropy_window: Rolling window for entropy computation.
        threshold_std: Number of std deviations above mean for signal.
        forward_days: Forward return horizons (default [30, 60, 90]).
        train_pct: Fraction of history used for threshold calibration.

    Returns:
        Dict with signal dates, forward returns, hit rates, and statistics.
    """
    if forward_days is None:
        forward_days = [30, 60, 90]

    # Get price data and compute returns
    prices = get_price_data(ticker, period)
    close = prices["Close"]
    log_returns = np.log(close / close.shift(1)).dropna()

    # Compute rolling entropy over the full history
    roll_ent = rolling_entropy(log_returns, window=entropy_window)
    if len(roll_ent) < 60:
        return {"error": "Insufficient data for backtesting"}

    # ── In-sample / Out-of-sample split ──────────────────────────────────────
    split_idx = int(len(roll_ent) * train_pct)
    if split_idx < 30:
        return {"error": "Insufficient in-sample data for threshold calibration"}

    in_sample_ent = roll_ent.iloc[:split_idx]
    out_sample_ent = roll_ent.iloc[split_idx:]
    split_date = roll_ent.index[split_idx]

    # Calibrate threshold on in-sample data only — no look-ahead
    ent_mean = float(in_sample_ent.mean())
    ent_std = float(in_sample_ent.std())
    threshold = ent_mean + threshold_std * ent_std

    # ── Signal generation on out-of-sample data only ─────────────────────────
    signal_mask = out_sample_ent > threshold
    signal_dates = out_sample_ent[signal_mask].index

    # Cluster nearby signals (don't count consecutive days as separate signals)
    clustered_signals = []
    if len(signal_dates) > 0:
        clustered_signals.append(signal_dates[0])
        for date in signal_dates[1:]:
            if (date - clustered_signals[-1]).days > 10:
                clustered_signals.append(date)

    # Compute forward returns for each signal (forward window may extend past
    # the OOS period end — that is fine, we're just measuring what happened)
    signal_results = []
    for signal_date in clustered_signals:
        signal_data = {
            "date": str(signal_date.date()) if hasattr(signal_date, 'date') else str(signal_date),
            "entropy": round(float(roll_ent.loc[signal_date]), 4),
            "price_at_signal": round(float(close.loc[signal_date]), 2),
        }

        for days in forward_days:
            future_idx = close.index.searchsorted(signal_date) + days
            if future_idx < len(close):
                future_price = float(close.iloc[future_idx])
                signal_price = float(close.loc[signal_date])
                fwd_return = (future_price - signal_price) / signal_price * 100
                signal_data[f"return_{days}d"] = round(fwd_return, 2)
                signal_data[f"price_{days}d"] = round(future_price, 2)
            else:
                signal_data[f"return_{days}d"] = None
                signal_data[f"price_{days}d"] = None

        signal_results.append(signal_data)

    # Signal statistics
    stats = {}
    for days in forward_days:
        returns = [s[f"return_{days}d"] for s in signal_results if s[f"return_{days}d"] is not None]
        if returns:
            stats[f"{days}d"] = {
                "n_signals": len(returns),
                "mean_return": round(np.mean(returns), 2),
                "median_return": round(np.median(returns), 2),
                "hit_rate_negative": round(sum(1 for r in returns if r < 0) / len(returns) * 100, 1),
                "worst": round(min(returns), 2),
                "best": round(max(returns), 2),
            }

    # ── Baseline: unconditional forward returns in the out-of-sample period ──
    # Restricting to OOS ensures baseline and signals share the same data universe.
    oos_start_idx = close.index.searchsorted(split_date)
    oos_close = close.iloc[oos_start_idx:]

    baseline_stats = {}
    for days in forward_days:
        all_fwd = []
        for i in range(len(oos_close) - days):
            fwd = (float(oos_close.iloc[i + days]) - float(oos_close.iloc[i])) / float(oos_close.iloc[i]) * 100
            all_fwd.append(fwd)
        if all_fwd:
            baseline_stats[f"{days}d"] = {
                "mean_return": round(np.mean(all_fwd), 2),
                "median_return": round(np.median(all_fwd), 2),
                "hit_rate_negative": round(sum(1 for r in all_fwd if r < 0) / len(all_fwd) * 100, 1),
            }

    return {
        "ticker": ticker,
        "period": period,
        "entropy_window": entropy_window,
        "threshold": round(threshold, 4),
        "entropy_mean": round(ent_mean, 4),
        "entropy_std": round(ent_std, 4),
        "n_signals": len(clustered_signals),
        "n_trading_days": len(log_returns),
        "n_oos_trading_days": len(out_sample_ent),
        "split_date": str(split_date.date()) if hasattr(split_date, 'date') else str(split_date),
        "train_pct": train_pct,
        "signals": signal_results,
        "signal_stats": stats,
        "baseline_stats": baseline_stats,
        "rolling_entropy": roll_ent,
        "close_prices": close,
        "signal_dates": clustered_signals,
    }
