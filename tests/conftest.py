import pytest

from bh_agent_model.utils.base.agents import Trader
from bh_agent_model.utils.base.markets import Market


@pytest.fixture
def traders():
    return [
        Trader(g=0.0, b=0.0, cost=0.0, name="F"),
        Trader(g=1.0, b=0.0, cost=0.0, name="C"),
    ]


@pytest.fixture
def market(traders):
    return Market(
        traders=traders,
        beta=1.0,
        r=1.01,
        sigma2=1.0,
        risk_aversion=1.0,
        noise_std=0.0,
    )
