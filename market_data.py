"""Market data fetching via yfinance."""

import yfinance as yf
import pandas as pd
import numpy as np


def get_price_data(ticker: str, period: str = "1y") -> pd.DataFrame:
    """Fetch historical price data for a ticker.

    Args:
        ticker: Stock/asset ticker symbol (e.g., 'AAPL', 'JPY=X')
        period: yfinance period string ('1mo', '3mo', '6mo', '1y', '2y', '5y')

    Returns:
        DataFrame with Date index and OHLCV columns.
    """
    data = yf.download(ticker, period=period, progress=False)
    if data.empty:
        raise ValueError(f"No data found for ticker '{ticker}'")
    # Flatten multi-level columns if present
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    return data


def get_returns(ticker: str, period: str = "1y") -> pd.Series:
    """Fetch log returns for a ticker.

    Args:
        ticker: Stock/asset ticker symbol
        period: yfinance period string

    Returns:
        Series of log returns with Date index.
    """
    data = get_price_data(ticker, period)
    close = data["Close"]
    returns = np.log(close / close.shift(1)).dropna()
    returns.name = ticker
    return returns


def get_multiple_returns(tickers: list[str], period: str = "1y") -> pd.DataFrame:
    """Fetch log returns for multiple tickers, aligned by date.

    Args:
        tickers: List of ticker symbols
        period: yfinance period string

    Returns:
        DataFrame with Date index and one column per ticker.
    """
    data = yf.download(tickers, period=period, progress=False)
    if data.empty:
        raise ValueError(f"No data found for tickers {tickers}")
    close = data["Close"]
    if isinstance(close, pd.Series):
        close = close.to_frame(name=tickers[0])
    returns = np.log(close / close.shift(1)).dropna()
    return returns
