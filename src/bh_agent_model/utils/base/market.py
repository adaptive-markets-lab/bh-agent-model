"""
Core insight of the Brock–Hommes model.

Market dynamics emerge endogenously from a feedback loop:

    Beliefs -> Price -> Profits -> Strategy Switching -> New Beliefs

This implementation follows that loop explicitly at each time step.
"""

from typing import Sequence

import numpy as np

from bh_agent_model.utils.base.agents import Trader


class Market:
    """
    Represent the market environment in the Brock–Hommes model.

    The market aggregates the behavior of different trader types and determines
    the evolution of prices over time.

    Prices are not set externally but emerge from the interaction of traders'
    expectations, demands, and strategy switching.
    """

    def __init__(
        self,
        traders: Sequence[Trader],
        beta: float,
        r: float,
        sigma2: float,
        a: float,
        noise_std: float = 0.001,
    ) -> None:
        """
        Initialize the market.

        Args:
            traders: Trader types available in the market.
            beta: Intensity of choice parameter for strategy switching.
            r: Gross risk-free return.
            sigma2: Variance of returns.
            a: Risk aversion parameter.
            noise_std: Standard deviation of the market noise shock.

        """
        self.traders = traders
        self.beta = beta
        self.r = r
        self.sigma2 = sigma2
        self.a = a
        self.noise_std = noise_std

        self.n_types = len(traders)
        self.weights = np.ones(self.n_types) / self.n_types

        self.x = 0.0  # price deviation

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
        x_prev = self.x

        forecasts = np.array([t.forecast(x_prev) for t in self.traders])

        noise = np.random.normal(0, self.noise_std)
        x_new = (np.dot(self.weights, forecasts) + noise) / self.r

        realized_return = x_new - x_prev

        for t in self.traders:
            t.update_fitness(realized_return)

        fitnesses = np.array([t.fitness for t in self.traders])
        self.weights = self.softmax(fitnesses)

        self.x = x_new

        return x_new, self.weights
