from dataclasses import dataclass

import numpy as np
import pandas as pd


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


@dataclass
class BaselineParams:
    """
    Baseline parameter set for the Brock–Hommes simulation.

    Used only to initialise sigma2 from empirical data; Sobol samples
    replace all other values during the analysis.

    Args:
        beta: Intensity of choice (evolutionary switching speed).
        r: Gross risk-free return.
        sigma2: Perceived return variance.
        risk_aversion: Risk aversion coefficient.
        g_chartist: Trend parameter for the chartist.
        g_contrarian: Trend parameter for the contrarian.
        b_optimist: Bias term for the optimist.
        cost_fundamentalist: Information cost for the fundamentalist.
        cost_optimist: Strategy cost for the optimist.

    """

    beta: float = 0.5
    r: float = 1.01
    sigma2: float = 1e-4
    risk_aversion: float = 5.0
    g_chartist: float = 1.1
    g_contrarian: float = -0.8
    b_optimist: float = 0.0005
    cost_fundamentalist: float = 0.0002
    cost_optimist: float = 0.0001


@dataclass
class SobolResult:
    """
    Sobol indices for a single output metric.

    Args:
        output_name: Name of the output metric (e.g. 'mean_weight_chartist').
        param_names: Ordered list of parameter names.
        s1: First-order indices, shape (n_params,).
        s1_conf: 95% confidence intervals for S1, shape (n_params,).
        st: Total-order indices, shape (n_params,).
        st_conf: 95% confidence intervals for ST, shape (n_params,).

    """

    output_name: str
    param_names: list[str]
    s1: np.ndarray
    s1_conf: np.ndarray
    st: np.ndarray
    st_conf: np.ndarray


@dataclass
class SimulationConfig:
    """
    Configuration for a Brock-Hommes simulation run.

    Args:
        beta: Intensity of choice parameter.
        periods: Number of simulated periods.
        r: Gross risk-free return.
        sigma2: Perceived variance.
        risk_aversion: Risk aversion coefficient.
        noise_std: Standard deviation of additive noise.
        x0: Initial price deviation from fundamentals.
        seed: Random seed for reproducibility.

    """

    beta: float
    periods: int
    r: float = 1.01
    sigma2: float = 1.0
    risk_aversion: float = 1.0
    noise_std: float = 0.0
    x0: float = 0.10
    seed: int = 42


@dataclass
class SimulationResult:
    """Container for a single simulation output."""

    series: pd.DataFrame
    traders: list[str]
    config: SimulationConfig
