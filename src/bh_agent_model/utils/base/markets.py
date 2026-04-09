from dataclasses import dataclass, field
from typing import Sequence
import numpy as np
from bh_agent_model.utils.base.agents import Trader

@dataclass(slots=True)
class Market:
    """
    The Market Engine for the Brock-Hommes (1998) Asset Pricing Model.
    
    This class manages the lifecycle of a simulation step:
    1. Collecting forecasts and demands from all trader types.
    2. Clearing the market to find the new price deviation (x_t).
    3. Updating strategy fitness based on realized returns.
    4. Re-allocating market weights via evolutionary switching.

    Note on Stability: 
    Uses sigma2=0.25 to maintain demand scales that prevent numerical overflow 
    or 'fitness lock-in' where strategies become indistinguishable.
    """
    traders: Sequence[Trader]  # List of unique Trader strategy objects
    beta: float                # Intensity of choice (higher = faster switching)
    r: float                   # Gross risk-free return (e.g., 1.05 for 5%)
    sigma2: float              # Belief about price volatility
    risk_aversion: float       # Global risk aversion parameter ('a')
    noise_std: float           # Standard deviation of exogenous market noise
    
    # Internal state variables
    n_types: int = field(init=False)      # Number of strategies in the market
    weights: np.ndarray = field(init=False) # Current market share of each strategy
    x: float = field(default=0.0, init=False) # Current price deviation from fundamental

    def __post_init__(self) -> None:
        """Validates model parameters and initializes market shares uniformly."""
        if len(self.traders) == 0:
            raise ValueError("traders must contain at least one strategy")
        if self.beta < 0:
            raise ValueError("beta must be non-negative")
        if self.r <= 0:
            raise ValueError("r must be positive")
        if self.sigma2 <= 0:
            raise ValueError("sigma2 must be positive")
        
        self.n_types = len(self.traders)
        # Start with an equal distribution across all strategies
        self.weights = np.full(self.n_types, 1.0 / self.n_types, dtype=float)
        
        # Ensure all traders start with a clean slate
        for trader in self.traders:
            trader.reset()

    def softmax(self, fitnesses: np.ndarray) -> np.ndarray:
        """
        Calculates the Adaptive Belief System (ABS) switching rule.
        
        Converts strategy fitness into market weights. Uses 'Max Subtraction' 
        and clipping to prevent floating-point overflow during np.exp().
        """
        # Subtract max for numerical stability (prevents exp(large_number))
        scaled = self.beta * (fitnesses - np.max(fitnesses))
        scaled = np.clip(scaled, -500, 0) 
        
        exp_vals = np.exp(scaled)
        total = np.sum(exp_vals)
        
        # Fallback to uniform distribution if math breaks
        if total == 0 or not np.isfinite(total):
            return np.full(self.n_types, 1.0 / self.n_types)
            
        return exp_vals / total

    def step(self) -> tuple[float, np.ndarray]:
        """
        Executes one full iteration of the market simulation.
        
        Returns:
            x_new (float): The new price deviation from fundamental value.
            weights (np.ndarray): The updated market shares for the next period.
        """
        x_prev = self.x

        # 1. Collect demands from all agent types based on t-1 price
        demands = np.array([
            trader.demand(
                x_prev=x_prev, 
                r=self.r, 
                sigma2=self.sigma2, 
                risk_aversion=self.risk_aversion
            ) for trader in self.traders
        ])

        # 2. Price Formation: Aggregate demand + noise / risk-free rate
        noise = np.random.normal(0.0, self.noise_std)
        x_new = (np.dot(self.weights, demands) + noise) / self.r

        # 3. Calculate Realized Return: Actual performance of the risky asset
        realized_return = x_new - x_prev

        # 4. Fitness Update: Agents evaluate how much profit they just made
        for trader in self.traders:
            trader.update_fitness(realized_return)

        # 5. Evolutionary Switching: Re-calculate market weights for period t+1
        fitnesses = np.array([trader.fitness for trader in self.traders])
        self.weights = self.softmax(fitnesses)
        
        self.x = x_new
        return x_new, self.weights.copy()