import numpy as np
import pytest

from bh_agent_model.utils.base.markets import Market


def test_market_initialization(traders):
    traders[0].fitness = 99
    traders[0].last_demand = 88

    m = Market(
        traders=traders,
        beta=1.0,
        r=1.01,
        sigma2=1.0,
        risk_aversion=1.0,
        noise_std=0.0,
    )

    assert m.n_types == 2
    assert np.allclose(m.weights, [0.5, 0.5])
    assert traders[0].fitness == 0
    assert traders[0].last_demand == 0


@pytest.mark.parametrize(
    "values",
    [
        np.array([1.0, 2.0]),
        np.array([10.0, 10.0]),
    ],
)
def test_softmax(market, values):
    weights = market.softmax(values)

    assert np.isclose(weights.sum(), 1.0)


def test_step_without_noise(market, monkeypatch):
    monkeypatch.setattr(np.random, "normal", lambda loc, scale: 0.0)

    x_new, weights = market.step()

    assert np.isclose(weights.sum(), 1.0)
    assert isinstance(x_new, float)
