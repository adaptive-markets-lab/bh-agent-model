import matplotlib.pyplot as plt
import numpy as np

from bh_agent_model.utils.base.agents import chartist, contrarian, fundamentalist, optimist
from bh_agent_model.utils.base.market import Market
from bh_agent_model.utils.load_data.load_data_from_yfinance import load_data_from_yfinance

if __name__ == "__main__":
    # set up params to get asset time series data
    ticker = "^GSPC"
    start_date = "2016-01-01"
    end_date = "2025-12-31"

    # get asset times series data
    data = load_data_from_yfinance(ticker=ticker, start_date=start_date, end_date=end_date)
    returns = np.asarray(data.returns).reshape(-1)

    # set up traders
    traders = [
        fundamentalist(cost=0.0002),
        chartist(g=1.1),
        contrarian(g=-0.8),
        optimist(b=0.0005, cost=0.0001),
    ]

    # set up market
    beta = 0.5
    r = 1.01
    sigma2 = max(data.sigma2, 1e-4)
    risk_aversion = 5.0
    noise_std = 0.01
    market = Market(
        traders=traders,
        beta=beta,
        r=r,
        sigma2=sigma2,
        risk_aversion=risk_aversion,
        noise_std=noise_std,
    )

    weights_history = []

    for trader in traders:
        trader.reset()

    weights = np.ones(len(traders)) / len(traders)

    for t in range(1, len(returns)):
        x_prev = float(returns[t - 1])
        realized_return = float(returns[t])

        for trader in traders:
            trader.demand(x_prev=x_prev, r=r, sigma2=sigma2, risk_aversion=risk_aversion)
            trader.update_fitness(realized_return=realized_return)

        fitnesses = np.array([trader.fitness for trader in traders], dtype=float)

        # softmax switching
        max_f = np.max(fitnesses)
        weights = np.exp(beta * (fitnesses - max_f))
        weights /= np.sum(weights)

        weights_history.append(weights.copy())

    weights_history = np.array(weights_history)

    dates = data.dates
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    # top: price
    axes[0].plot(data.dates, data.prices)
    axes[0].set_title("S&P 500 Price")
    axes[0].set_ylabel("Price")

    # bottom: strategy weights
    for i, trader in enumerate(traders):
        axes[1].plot(dates[1:], weights_history[:, i], label=trader.name)

    axes[1].set_title("Strategy Dominance")
    axes[1].set_ylabel("Weight")
    axes[1].set_xlabel("Date")
    axes[1].legend()

    # --- Fix spacing ---
    plt.tight_layout()

    plt.show()
