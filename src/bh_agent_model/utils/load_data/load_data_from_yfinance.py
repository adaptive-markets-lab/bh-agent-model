from datetime import datetime

import numpy as np
import yfinance as yf

from bh_agent_model.utils.base.models import AssetTimeSeries


def load_data_from_yfinance(
    ticker: str,
    start_date: str,
    end_date: str,
    return_type: str = "log",
) -> AssetTimeSeries:
    """
    Download historical market data from Yahoo Finance and compute returns.

    This function retrieves price data for a given financial instrument (e.g.,
    equity, index, ETF, or cryptocurrency) using the Yahoo Finance API via
    `yfinance`. It then computes returns and basic statistics required for
    agent-based financial models such as the Brock–Hommes framework.

    The returned data is structured as a `AssetTimeSeries` object, which includes
    aligned price and return series as well as the variance of returns.

    Args:
        ticker (str): Ticker symbol recognized by Yahoo Finance (e.g., "AAPL",
            "^GSPC", "BTC-USD").
        start_date (str): Start date in ISO format "YYYY-MM-DD".
        end_date (str): End date in ISO format "YYYY-MM-DD".
        return_type (str, optional): Type of return to compute. Must be one of:
            - "log": Logarithmic returns (default)
            - "simple": Simple percentage returns

    Returns:
        AssetTimeSeries: A data object containing:
            - ticker: The ticker symbol
            - dates: Array of dates aligned with returns
            - prices: Array of adjusted closing prices
            - returns: Array of computed returns
            - sigma2: Variance of returns

    """
    # type checks
    assert isinstance(ticker, str), "ticker must be a string"
    assert isinstance(start_date, str), "start_date must be a string in format YYYY-MM-DD"
    assert isinstance(end_date, str), "end_date must be a string in format YYYY-MM-DD"

    # data format validation
    try:
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    except ValueError as error:
        raise ValueError("start_date must be in format YYYY-MM-DD") from error
    try:
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    except ValueError as error:
        raise AssertionError("end_date must be in format YYYY-MM-DD") from error

    # logical check
    assert start_dt <= end_dt, "start_date must be before end_date"

    # download data
    df = yf.download(tickers=ticker, start=start_dt, end=end_dt, auto_adjust=True, progress=False)

    if df.empty:
        raise AssertionError(f"No data found for ticker {ticker}")

    # get closing price
    prices = np.asarray(df["Close"], dtype=float).reshape(-1)

    # get dates
    dates = np.asarray(df.index)

    # compute returns
    if return_type == "log":
        returns = np.diff(np.log(prices))
    elif return_type == "simple":
        returns = np.diff(prices) / prices[:-1]
    else:
        raise ValueError("return_type must be 'log' or 'simple'")

    # compute variance of returns
    sigma2 = float(np.var(returns))

    return AssetTimeSeries(
        ticker=ticker,
        dates=dates[1:],
        prices=prices[1:],
        returns=returns,
        sigma2=sigma2,
    )
