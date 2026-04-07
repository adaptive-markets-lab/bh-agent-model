from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from bh_agent_model.utils.base.agents import Trader


@dataclass(slots=True)
class Market:
    """
    Represent the market environment in the Brock–Hommes model.

    The market aggregates the behavior of different trader types and determines
    the evolution of prices over time.

    Price evolve endogenously through heterogeneous beliefs (forecast rules)
    and evolutionary competition (fitness-based switching)

    Args:
            traders: Trader types available in the market.
            beta: Intensity of choice parameter for strategy switching.
            r: Gross risk-free return.
            sigma2: Variance of returns.
            risk_aversion: Risk aversion parameter.
            noise_std: Standard deviation of the market noise shock.

    """

    traders: Sequence[Trader]
    beta: float
    r: float
    sigma2: float
    risk_aversion: float
    noise_std: float

    n_types: int = field(init=False)
    weights: np.ndarray = field(init=False)
    x: float = field(default=0.0, init=False)

    def __post_init__(self) -> None:
        """Check data properties and preprocess data."""
        if len(self.traders) == 0:
            raise ValueError("traders must contain at least one strategy")
        if self.beta < 0:
            raise ValueError("beta must be non-negative")
        if self.r <= 0:
            raise ValueError("r must be positive")
        if self.sigma2 <= 0:
            raise ValueError("sigma2 must be positive")
        if self.risk_aversion <= 0:
            raise ValueError("risk_aversion must be positive")
        if self.noise_std < 0:
            raise ValueError("noise_std must be non-negative")

        self.n_types = len(self.traders)
        self.weights = np.full(self.n_types, 1.0 / self.n_types, dtype=float)

        for trader in self.traders:
            trader.reset()

    def softmax(self, fitnesses: np.ndarray) -> np.ndarray:
        """
        Convert fitness values into population weights using a softmax rule.

        This implements the evolutionary selection mechanism where
        better-performing strategies attract more followers.

        Intuition:
            - Higher fitness -> larger population share
            - Lower fitness -> smaller share
            - Beta controls sensitivity:
                * Low beta -> weak response to performance
                * High beta -> strong winner-takes-all behavior

        Note:
            Uses a numerically stable version (log-sum-exp trick) to avoid
            overflow when fitness values are large.

        Args:
            fitnesses: Array of fitness values for each trader type.

        Returns:
            Normalized weights that sum to 1.

        """
        max_f = np.max(fitnesses)
        exp_vals = np.exp(self.beta * (fitnesses - max_f))
        return exp_vals / np.sum(exp_vals)

    def step(self) -> tuple[float, np.ndarray]:
        """
        Execute one time step of the market simulation.

        This function implements the full dynamic loop of the Brock–Hommes model.

        Process:
            1. Forecasting:
                Each trader forms expectations about the next price.
            2. Price formation:
                The market aggregates expectations into a new price.
            3. Realized returns:
                The actual price change is observed.
            4. Fitness update:
                Traders evaluate how well their strategy performed.
            5. Strategy switching:
                Population shares are updated based on fitness.

        Intuition:
            The market evolves through a feedback loop:
                beliefs -> prices -> profits -> strategy switching -> new beliefs

        Returns:
            A tuple containing:
                - Updated price deviation (`x_new`)
                - Updated strategy weights

        """
        # previous price deviation (x_{t-1})
        x_prev = self.x

        # belief formation
        # each trader produces a forecast of next-period price deviation
        forecasts = np.array([trader.forecast(x_prev) for trader in self.traders])

        # price formation
        # weighted average of forecasts + noise, discounted by risk-free return
        noise = np.random.normal(0, self.noise_std)
        x_new = (np.dot(self.weights, forecasts) + noise) / self.r

        # realized return
        realized_return = x_new - x_prev

        # traders evaluate performance based on realized return
        for trader in self.traders:
            trader.update_fitness(realized_return)

        # compute new population shares based on fitness
        fitnesses = np.array([trader.fitness for trader in self.traders])
        self.weights = self.softmax(fitnesses)

        # update state
        self.x = x_new

        return x_new, self.weights
