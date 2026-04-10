from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class Trader:
    """
    Represents an individual agent type within the Brock-Hommes Asset Pricing Model.

    The model simulates heterogeneous expectations where traders choose strategies
    based on past performance (fitness). This class encapsulates the forecasting
    logic, demand calculation, and evolutionary fitness tracking for a specific rule.

    Args:
        g (float): Trend-following parameter (slope).
        b (float): Constant bias parameter (intercept).
        cost (float): Periodic information or transaction cost for this strategy.
        name (str): Identifier for the strategy type (e.g., 'Chartist').
        fitness (float): Accumulation of past profits used for strategy switching.
        last_demand (float): Quantity of the risky asset held in the previous period.

    """

    g: float
    b: float
    cost: float
    name: str
    fitness: float = field(default=0.0, init=False)
    last_demand: float = field(default=0.0, init=False)

    def forecast(self, x_prev: float) -> float:
        """
        Predict the next period's price deviation from the fundamental value.

        Formula: f_h(t) = g * x(t-1) + b.
        """
        return self.g * x_prev + self.b

    def demand(self, x_prev: float, r: float, sigma2: float, risk_aversion: float) -> float:
        """
        Calculate optimal asset demand based on mean-variance utility maximization.

        Args:
            x_prev: Price deviation at t-1.
            r: Gross return of the risk-free asset.
            sigma2: Constant conditional variance of the risky asset.
            risk_aversion: The 'a' parameter in the demand equation.

        """
        f = self.forecast(x_prev)
        # z_h = (Expected_Price - Risk_Free_Return) / (Risk_Aversion * Variance)
        z = (f - r * x_prev) / (risk_aversion * sigma2)
        self.last_demand = z
        return z

    def update_fitness(self, realized_return: float) -> None:
        """
        Update the strategy's performance metric using Exponentially Weighted Averaging (EWA).

        Args:
            realized_return: The actual excess return (x_t - R * x_{t-1}).

        """
        profit = realized_return * self.last_demand
        # Memory parameter: 0.5 balances immediate performance with historical stability
        eta = 0.5
        self.fitness = eta * self.fitness + (1.0 - eta) * profit - self.cost

    def reset(self) -> None:
        """Reset agent state for new simulation runs."""
        self.fitness = 0.0
        self.last_demand = 0.0


# --- Strategy Factory Functions ---


def fundamentalist(cost: float = 0.001) -> Trader:
    """
    Rational-lite agent.

    Forecasts that the price will return to fundamental value (0).
    Usually incurs a cost to reflect the effort of analyzing fundamentals.

    Args:
        cost: Information or analysis cost deducted from the agent's fitness score.

    Returns:
        A Trader configured as a fundamentalist with zero trend and bias coefficients.

    """
    return Trader(g=0.0, b=0.0, cost=cost, name="Fundamentalist")


def chartist(g: float = 1.2) -> Trader:
    """
    Trend-follower. Extrapolates recent price movements.

    Values of g > R lead to explosive price bubbles and instability.

    Args:
        g: Trend-extrapolation coefficient. Values above the risk-free rate
            destabilise the market and can generate persistent bubbles.

    Returns:
        A Trader configured as a chartist with zero bias and no information cost.

    """
    return Trader(g=g, b=0.0, cost=0.0, name="Chartist")


def contrarian(g: float = -0.6) -> Trader:
    """
    Mean-reversion agent. Bets against the current trend.

    Provides liquidity and stabilizes the market during high volatility.

    Args:
        g: Trend-extrapolation coefficient. Negative values cause the agent
            to forecast a reversal, dampening momentum and stabilising prices.

    Returns:
        A Trader configured as a contrarian with zero bias and no information cost.

    """
    return Trader(g=g, b=0.0, cost=0.0, name="Contrarian")


def optimist(b: float = 0.002, cost: float = 0.001) -> Trader:
    """
    Biased agent.

    Maintains a constant positive outlook (bullish) regardless of price.
    Useful for simulating 'noise' or persistent market sentiment.

    Args:
        b: Constant bullish bias added to the price forecast each period.
        cost: Information or analysis cost deducted from the agent's fitness score.

    Returns:
        A Trader configured as an optimist with zero trend coefficient.

    """
    return Trader(g=0.0, b=b, cost=cost, name="Optimist")
