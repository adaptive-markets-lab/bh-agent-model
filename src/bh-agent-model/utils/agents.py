#code creates different types of traders for the BH model

class Trader:
    """
    Represents a trader type in the Brock–Hommes model.

    Each Trader instance corresponds to a strategy (not an individual person).
    The model tracks how well each strategy performs over time and how popular
    it becomes in the market.

    Key idea:
    Traders form expectations about future prices using a simple linear rule,
    take positions based on those expectations, and earn profits depending on
    whether they were correct.

    Attributes
    ----------
    g : float
        Trend parameter. Determines how strongly the trader reacts to past price
        deviations (trend-following if g > 0, contrarian if g < 0).
    b : float
        Bias term. A constant belief independent of market conditions (e.g. optimism).
    cost : float
        Information or strategy cost. Penalizes more complex strategies.
    fitness : float
        Current performance (profit minus cost). Used to determine how popular
        this strategy becomes.
    last_demand : float
        Position taken in the previous period (used to compute profits).
    name : str
        Human-readable name of the strategy.
    """

    def __init__(self, g: float, b: float, cost: float, name: str):
        self.g = g              # trend parameter shows how much agent believes in trends (if > 0, trend-following; if < 0, trend-reversing, if = 0, fundamentalist, if < 0, contrarian)
        self.b = b              # bias (belief regardsless of data, e.g. optimists have b > 0, pessimists have b < 0)
        self.cost = cost        # information cost of how much to use strategy (e.g. fundamentalists have cost > 0, chartists and contrarians have cost = 0)
        self.name = name

        self.fitness = 0.0      # current fitness of the strategy, updated based on profits
        self.last_demand = 0.00  # last demand, how much trader bought/sold last time

    def forecast(self, x_prev: float) -> float:
        """
        Compute the trader's expectation of next period's price deviation.

        The forecasting rule is linear:
            f_h(x_{t-1}) = g * x_{t-1} + b

        Intuition
        ---------
        The trader looks at how far the price was from its fundamental value in the
        previous period and forms a belief about where it will go next.

        - If g > 0: extrapolates trends ("price will keep moving in same direction")
        - If g < 0: expects reversal ("price will go back toward fundamental")
        - If b != 0: adds a constant bias (e.g. persistent optimism)

        Parameters
        ----------
        x_prev : float
            Previous price deviation from fundamental value:
                x = price - fundamental value
            x > 0 → price above fundamental (overvalued)
            x < 0 → price below fundamental (undervalued)

        Returns
        -------
        float
        Expected next-period price deviation.
        """
        return self.g * x_prev + self.b

    def demand(self, x_prev: float, R: float, sigma2: float, a: float) -> float:
        """
        Compute the trader's demand for the risky asset.

        The trader compares expected returns to a benchmark (risk-free return)
        and adjusts for risk. This comes from a mean-variance optimization problem.

        Formula
        -------
        z = (forecast - R * x_prev) / (a * sigma^2)

        Intuition
        ---------
        - If expected return is high → buy more (positive demand)
        - If expected return is low → sell (negative demand)
        - Higher risk (sigma^2) or risk aversion (a) → smaller positions

        Parameters
        ----------
        x_prev : float
            Previous price deviation.
        R : float
            Gross risk-free return (1 + interest rate).
        sigma2 : float
            Variance of returns (perceived risk).
        a : float
            Risk aversion parameter.

        Returns
        -------
        float
            Quantity of the risky asset demanded by this trader.
        """
        f = self.forecast(x_prev)
        z = (f - R * x_prev) / (a * sigma2)

        self.last_demand = z
        return z

    def update_fitness(self, realized_return: float):
        """
        Update the trader's fitness (performance) based on realized profits.

        Fitness determines how attractive this strategy is to the population
        and drives the evolutionary switching mechanism in the model.

        Intuition
        ---------
        - If the trader took a position in the correct direction → profit
        - If wrong → loss
        - Strategy cost is subtracted

        This reflects:
            "Did this strategy make money, net of its cost?"

        Parameters
        ----------
        realized_return : float
            Actual price change between periods:
                x_t - x_{t-1}

        Updates
        -------
        self.fitness : float
            Profit minus cost for the current period.
        """
        profit = realized_return * self.last_demand
        self.fitness = profit - self.cost


def fundamentalist(cost=1.0):
    """
    Create a fundamentalist trader.

    Fundamentalists believe prices revert to their fundamental value.
    They do not respond to trends and have no bias.

    Characteristics
    ---------------
    - g = 0 → ignores past price movements
    - b = 0 → no systematic bias
    - cost > 0 → represents effort of fundamental analysis

    Returns
    -------
    Trader
        A fundamentalist trader instance.
    """
    return Trader(g=0.0, b=0.0, cost=cost, name="Fundamentalist")

def chartist(g=1.2):
    """
    Create a chartist (trend-following) trader.

    Trend chasers extrapolate recent price movements and amplify trends,
    which can destabilize the market.

    Characteristics
    ---------------
    - g > 0 → follows trends
    - b = 0 → no bias
    - cost = 0 → simple, cheap strategy

    Returns
    -------
    Trader
        A trend-following trader instance.
    """
    return Trader(g=g, b=0.0, cost=0.0, name="Chartist")

def contrarian(g=-1.0):
    """
    Create a contrarian trader.

    Contrarians bet against recent price movements, expecting mean reversion.

    Characteristics
    ---------------
    - g < 0 → opposes trends
    - b = 0 → no bias
    - cost = 0 → simple strategy

    Returns
    -------
    Trader
        A contrarian trader instance.
    """
    return Trader(g=g, b=0.0, cost=0.0, name="Contrarian")

def optimist(b=0.5, cost=0.5):
    """
    Create an optimistic trader.

    Optimists maintain a persistent positive bias regardless of market conditions.

    Characteristics
    ---------------
    - g = 0 → ignores trends
    - b > 0 → always expects upward movement
    - cost >= 0 → may reflect informational or behavioral cost

    Returns
    -------
    Trader
        An optimistic trader instance.
    """
    return Trader(g=0.0, b=b, cost=cost, name="Optimist")




