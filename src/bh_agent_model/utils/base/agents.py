class Trader:
    """
    Represents a trader type in the Brock–Hommes model.

    Each Trader instance corresponds to a strategy (not an individual agent).
    The model tracks how well each strategy performs over time and how popular
    it becomes in the market.

    Traders form expectations about future prices using a linear rule, take
    positions based on those expectations, and earn profits depending on
    whether they were correct.

    """

    def __init__(self, g: float, b: float, cost: float, name: str):
        """
        Initialize a trader strategy.

        Args:
            g: Trend parameter (strategy type).
            b: Bias term.
            cost: Strategy cost.
            name: Name of the strategy.

        """
        self.g = g
        self.b = b
        self.cost = cost
        self.name = name
        self.fitness = 0.0
        self.last_demand = 0.0

    def forecast(self, x_prev: float) -> float:
        """
        Compute the trader's expectation of next period's price deviation.

        The forecasting rule is:
            f_h(x_{t-1}) = g * x_{t-1} + b

        Args:
            x_prev (float): Previous price deviation from fundamental value.

        Returns:
            float: Expected next-period price deviation.

        """
        return self.g * x_prev + self.b

    def demand(self, x_prev: float, r: float, sigma2: float, a: float) -> float:
        """
        Compute the trader's demand for the risky asset.

        The demand function is:
            z = (f - R * x_prev) / (a * sigma^2)

        Args:
            x_prev (float): Previous price deviation.
            r (float): Gross risk-free return (1 + interest rate).
            sigma2 (float): Variance of returns (perceived risk).
            a (float): Risk aversion parameter.

        Returns:
            float: Quantity of the risky asset demanded.

        """
        f = self.forecast(x_prev)
        z = (f - r * x_prev) / (a * sigma2)

        self.last_demand = z
        return z

    def update_fitness(self, realized_return: float):
        """
        Update the trader's fitness based on realized profits.

        Fitness = realized_return * last_demand - cost

        Args:
            realized_return (float): Actual price change between periods.

        """
        profit = realized_return * self.last_demand
        self.fitness = profit - self.cost


def fundamentalist(cost: float = 1.0):
    """
    Create a fundamentalist trader.

    Fundamentalists believe prices revert to their fundamental value.

    Args:
        cost (float, optional): Cost of fundamental analysis. Defaults to 1.0.

    Returns:
        Trader: Fundamentalist trader instance.

    """
    return Trader(g=0.0, b=0.0, cost=cost, name="Fundamentalist")


def chartist(g: float = 1.2):
    """
    Create a chartist (trend-following) trader.

    Args:
        g (float, optional): Trend strength parameter (g > 0). Defaults to 1.2.

    Returns:
        Trader: Chartist trader instance.

    """
    return Trader(g=g, b=0.0, cost=0.0, name="Chartist")


def contrarian(g: float = -1.0):
    """
    Create a contrarian trader.

    Args:
        g (float, optional): Trend parameter (g < 0 for reversal). Defaults to -1.0.

    Returns:
        Trader: Contrarian trader instance.

    """
    return Trader(g=g, b=0.0, cost=0.0, name="Contrarian")


def optimist(b: float = 0.5, cost: float = 0.5):
    """
    Create an optimistic trader.

    Args:
        b (float, optional): Positive bias term. Defaults to 0.5.
        cost (float, optional): Strategy cost. Defaults to 0.5.

    Returns:
        Trader: Optimistic trader instance.

    """
    return Trader(g=0.0, b=b, cost=cost, name="Optimist")
