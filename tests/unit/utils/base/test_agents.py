import numpy as np
import pytest
from bh_agent_model.utils.base.agents import Trader, fundamentalist, chartist, contrarian, optimist

def test_forecast():
    t = Trader(g=2.0, b=1.0, cost=0.0, name="test")
    result = t.forecast(3.0)

    assert result == 2.0 * 3.0 + 1.0

def test_demand():
    t = Trader(g=1.0, b=0.0, cost=0.0, name="test")

    z = t.demand(x_prev=1.0, r=1.0, sigma2=1.0, risk_aversion=1.0)

    expected = (1.0 - 1.0 * 1.0) / (1.0 * 1.0)

    assert np.isclose(z, expected)
    assert t.last_demand == z

def test_update_fitness():
    t = Trader(g=0.0, b=0.0, cost=0.1, name="test")
    
    t.last_demand = 2.0
    t.fitness = 1.0

    t.update_fitness(realized_return=1.0)

    # profit = 1 * 2 = 2
    # fitness = 0.5*1 + 0.5*2 - 0.1 = 1.4
    assert np.isclose(t.fitness, 1.4)

def test_reset():
    t = Trader(g=0.0, b=0.0, cost=0.0, name="test")
    t.fitness = 10
    t.last_demand = 5

    t.reset()

    assert t.fitness == 0.0
    assert t.last_demand == 0.0

def test_fundamentalist():
    t = fundamentalist()
    assert t.g == 0.0
    assert t.b == 0.0


def test_chartist():
    t = chartist(g=1.5)
    assert t.g == 1.5


def test_contrarian():
    t = contrarian()
    assert t.g < 0


def test_optimist():
    t = optimist()
    assert t.b > 0