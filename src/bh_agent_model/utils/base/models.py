from dataclasses import dataclass

import numpy as np


@dataclass
class AssetTimeSeries:
    """
    Container for financial time series data of a single asset.

    Attributes:
        ticker (str): Ticker symbol (e.g., "AAPL", "^GSPC").
        dates (np.ndarray): Dates aligned with the return series.
        prices (np.ndarray): Adjusted closing prices.
        returns (np.ndarray): Computed returns (log or simple).
        sigma2 (float): Variance of returns.

    """

    ticker: str
    dates: np.ndarray
    prices: np.ndarray
    returns: np.ndarray
    sigma2: float
