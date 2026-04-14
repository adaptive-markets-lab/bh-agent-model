"""
Run Brock-Hommes simulations using the bh-agent-model repository API.

This script is written against the current repo layout:
- bh_agent_model.utils.base.agents
- bh_agent_model.utils.base.market.Market

It implements four experiments:
1. Two-type baseline: fundamentalist vs. trend chaser at beta=3.0 over T=1000.
2. Beta regime comparison: beta in {0, 2, 5, 8}.
3. Bifurcation sweep: beta in [0, 8] with 250 grid points, 500 burn-in,
   then 200 retained observations per beta.
4. Four-type extension: add contrarian and optimist at beta=3.0.

Why the custom simulation loop instead of Market.step()?
--------------------------------------------------------
The current repo's Trader fitness depends on ``last_demand``. The bundled
analysis example updates trader demand before fitness switching, while
``Market.step()`` updates fitness without first refreshing demand. To stay
consistent with the existing example and with Brock-Hommes style dynamics,
this script performs the demand -> price update -> fitness -> softmax loop
explicitly while still using the repo's Trader constructors and Market
softmax/validation.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from bh_agent_model.utils.base.agents import (
    Trader,
    chartist,
    contrarian,
    fundamentalist,
    optimist,
)
from bh_agent_model.utils.base.markets import Market
from bh_agent_model.utils.helper.setup_logging import setup_logging


@dataclass
class SimulationConfig:
    """
    Configuration for a Brock-Hommes simulation run.

    Args:
        beta: Intensity of choice parameter.
        periods: Number of simulated periods.
        r: Gross risk-free return.
        sigma2: Perceived variance.
        risk_aversion: Risk aversion coefficient.
        noise_std: Standard deviation of additive noise.
        x0: Initial price deviation from fundamentals.
        seed: Random seed for reproducibility.

    """

    beta: float
    periods: int
    r: float = 1.01
    sigma2: float = 1.0
    risk_aversion: float = 1.0
    noise_std: float = 0.0
    x0: float = 0.10
    seed: int = 42


@dataclass
class SimulationResult:
    """Container for a single simulation output."""

    series: pd.DataFrame
    traders: list[str]
    config: SimulationConfig


def softmax_stable(beta: float, fitnesses: np.ndarray) -> np.ndarray:
    """
    Compute numerically stable softmax weights.

    Args:
        beta: Intensity of choice parameter.
        fitnesses: Fitness values.

    Returns:
        Normalized strategy weights.

    """
    max_f = np.max(fitnesses)
    exp_vals = np.exp(beta * (fitnesses - max_f))
    return exp_vals / exp_vals.sum()


def run_simulation(traders: list[Trader], config: SimulationConfig) -> SimulationResult:
    """
    Run one Brock-Hommes simulation.

    The simulation uses repo-compatible trader objects and a Market instance
    for parameter validation and initialization. The state update itself is
    executed explicitly so that demand is refreshed before fitness updates.

    Args:
        traders: Strategy objects from the repository.
        config: Simulation configuration.

    Returns:
        A SimulationResult with period-by-period prices, fitnesses, and weights.

    """
    rng = np.random.default_rng(config.seed)

    market = Market(
        traders=traders,
        beta=config.beta,
        r=config.r,
        sigma2=config.sigma2,
        risk_aversion=config.risk_aversion,
        noise_std=config.noise_std,
    )
    market.x = config.x0

    weights = market.weights.copy()
    trader_names = [trader.name for trader in traders]
    records: list[dict[str, float]] = []

    for t in range(config.periods):
        x_prev = float(market.x)

        for trader in traders:
            trader.demand(
                x_prev=x_prev,
                r=config.r,
                sigma2=config.sigma2,
                risk_aversion=config.risk_aversion,
            )

        forecasts = np.array([trader.forecast(x_prev) for trader in traders], dtype=float)
        noise = float(rng.normal(0.0, config.noise_std))
        x_new = float((weights @ forecasts + noise) / config.r)
        realized_return = x_new - x_prev

        for trader in traders:
            trader.update_fitness(realized_return=realized_return)

        fitnesses = np.array([trader.fitness for trader in traders], dtype=float)
        weights = softmax_stable(config.beta, fitnesses)

        row: dict[str, float] = {
            "t": t + 1,
            "x": x_new,
            "x_prev": x_prev,
            "realized_return": realized_return,
            "noise": noise,
            "beta": config.beta,
        }
        for i, name in enumerate(trader_names):
            slug = slugify(name)
            row[f"weight_{slug}"] = weights[i]
            row[f"fitness_{slug}"] = fitnesses[i]
            row[f"forecast_{slug}"] = forecasts[i]
        records.append(row)

        market.x = x_new

    return SimulationResult(series=pd.DataFrame(records), traders=trader_names, config=config)


def slugify(name: str) -> str:
    """Create a lowercase column-safe trader name."""
    return name.strip().lower().replace(" ", "_")


def make_two_type_traders(trend_g: float, fundamentalist_cost: float) -> list[Trader]:
    """Create baseline fundamentalist and trend-chaser traders."""
    return [
        fundamentalist(cost=fundamentalist_cost),
        chartist(g=trend_g),
    ]


def make_four_type_traders(
    trend_g: float,
    contrarian_g: float,
    optimist_bias: float,
    fundamentalist_cost: float,
    optimist_cost: float,
) -> list[Trader]:
    """Create four-strategy extension traders."""
    return [
        fundamentalist(cost=fundamentalist_cost),
        chartist(g=trend_g),
        contrarian(g=contrarian_g),
        optimist(b=optimist_bias, cost=optimist_cost),
    ]


def plot_price_and_weights(
    result: SimulationResult,
    title: str,
    output_path: Path,
) -> None:
    """Plot price deviation and strategy weights for one simulation."""
    df = result.series

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    axes[0].plot(df["t"], df["x"], linewidth=1.3)
    axes[0].set_ylabel("Price deviation x_t")
    axes[0].set_title(title)
    axes[0].axhline(0.0, linestyle="--", linewidth=0.8)

    for trader_name in result.traders:
        slug = slugify(trader_name)
        axes[1].plot(df["t"], df[f"weight_{slug}"], label=trader_name, linewidth=1.2)
    axes[1].set_xlabel("Period")
    axes[1].set_ylabel("Population weight")
    axes[1].legend(loc="best")

    plt.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_beta_regimes(results: dict[float, SimulationResult], output_path: Path) -> None:
    """Plot price paths for several beta regimes."""
    fig, axes = plt.subplots(len(results), 1, figsize=(12, 10), sharex=True)

    for ax, (beta, result) in zip(axes, results.items()):
        ax.plot(result.series["t"], result.series["x"], linewidth=1.0)
        ax.axhline(0.0, linestyle="--", linewidth=0.7)
        ax.set_ylabel(f"beta={beta}")
    axes[0].set_title("Two-type model across beta regimes")
    axes[-1].set_xlabel("Period")

    plt.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def run_bifurcation_sweep(
    betas: Iterable[float],
    burn_in: int,
    keep: int,
    trend_g: float,
    fundamentalist_cost: float,
    r: float,
    sigma2: float,
    risk_aversion: float,
    noise_std: float,
    x0: float,
    seed: int,
) -> pd.DataFrame:
    """
    Run a bifurcation sweep over beta.

    Args:
        betas: Beta grid.
        burn_in: Number of transient observations to discard.
        keep: Number of observations retained after burn-in.
        trend_g: Trend parameter for chartists.
        fundamentalist_cost: Cost for fundamentalists.
        r: Gross risk-free return.
        sigma2: Perceived variance.
        risk_aversion: Risk aversion coefficient.
        noise_std: Standard deviation of market noise.
        x0: Initial state.
        seed: Base random seed.

    Returns:
        Long-form DataFrame with retained observations for each beta.

    """
    frames: list[pd.DataFrame] = []

    for i, beta in enumerate(betas):
        config = SimulationConfig(
            beta=float(beta),
            periods=burn_in + keep,
            r=r,
            sigma2=sigma2,
            risk_aversion=risk_aversion,
            noise_std=noise_std,
            x0=x0,
            seed=seed + i,
        )
        result = run_simulation(
            traders=make_two_type_traders(
                trend_g=trend_g,
                fundamentalist_cost=fundamentalist_cost,
            ),
            config=config,
        )
        retained = result.series.iloc[burn_in:].copy()
        retained["beta"] = beta
        frames.append(retained[["beta", "t", "x"]])

    return pd.concat(frames, ignore_index=True)


def plot_bifurcation(df: pd.DataFrame, output_path: Path) -> None:
    """Plot bifurcation diagram."""
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.plot(df["beta"], df["x"], ".", markersize=1.2, alpha=0.7)
    ax.set_title("Bifurcation sweep")
    ax.set_xlabel("beta")
    ax.set_ylabel("Retained x_t")
    ax.axhline(0.0, linestyle="--", linewidth=0.8)
    plt.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_results_csv(df: pd.DataFrame, output_path: Path) -> None:
    """Persist a DataFrame to CSV."""
    df.to_csv(output_path, index=False)


def main() -> None:
    """Run the full simulation suite."""
    parser = argparse.ArgumentParser(description="Run BH repo simulations.")
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trend-g", type=float, default=1.2)
    parser.add_argument("--contrarian-g", type=float, default=-1.0)
    parser.add_argument("--optimist-bias", type=float, default=0.5)
    parser.add_argument("--fundamentalist-cost", type=float, default=0.0)
    parser.add_argument("--optimist-cost", type=float, default=0.0)
    parser.add_argument("--r", type=float, default=1.01)
    parser.add_argument("--sigma2", type=float, default=1.0)
    parser.add_argument("--risk-aversion", type=float, default=1.0)
    parser.add_argument("--noise-std", type=float, default=0.0)
    parser.add_argument("--x0", type=float, default=0.10)
    args = parser.parse_args()

    outdir = args.output_dir
    outdir.mkdir(parents=True, exist_ok=True)

    # (i) Two-type baseline.
    baseline_config = SimulationConfig(
        beta=3.0,
        periods=1000,
        r=args.r,
        sigma2=args.sigma2,
        risk_aversion=args.risk_aversion,
        noise_std=args.noise_std,
        x0=args.x0,
        seed=args.seed,
    )
    baseline_result = run_simulation(
        traders=make_two_type_traders(
            trend_g=args.trend_g,
            fundamentalist_cost=args.fundamentalist_cost,
        ),
        config=baseline_config,
    )
    save_results_csv(baseline_result.series, outdir / "two_type_baseline.csv")
    plot_price_and_weights(
        baseline_result,
        title="Two-type baseline (Fundamentalist vs Trend Chaser, beta=3.0)",
        output_path=outdir / "two_type_baseline.png",
    )

    # (ii) Beta regime comparison.
    regime_results: dict[float, SimulationResult] = {}
    regime_frames: list[pd.DataFrame] = []
    for beta in [0.0, 2.0, 5.0, 8.0]:
        config = SimulationConfig(
            beta=beta,
            periods=1000,
            r=args.r,
            sigma2=args.sigma2,
            risk_aversion=args.risk_aversion,
            noise_std=args.noise_std,
            x0=args.x0,
            seed=args.seed + int(beta * 100),
        )
        result = run_simulation(
            traders=make_two_type_traders(
                trend_g=args.trend_g,
                fundamentalist_cost=args.fundamentalist_cost,
            ),
            config=config,
        )
        regime_results[beta] = result
        temp = result.series.copy()
        temp["beta_regime"] = beta
        regime_frames.append(temp)
    regime_df = pd.concat(regime_frames, ignore_index=True)
    save_results_csv(regime_df, outdir / "beta_regime_comparison.csv")
    plot_beta_regimes(regime_results, outdir / "beta_regime_comparison.png")

    # (iii) Bifurcation sweep.
    beta_grid = np.linspace(0.0, 8.0, 250)
    bifurcation_df = run_bifurcation_sweep(
        betas=beta_grid,
        burn_in=500,
        keep=200,
        trend_g=args.trend_g,
        fundamentalist_cost=args.fundamentalist_cost,
        r=args.r,
        sigma2=args.sigma2,
        risk_aversion=args.risk_aversion,
        noise_std=args.noise_std,
        x0=args.x0,
        seed=args.seed,
    )
    save_results_csv(bifurcation_df, outdir / "bifurcation_sweep.csv")
    plot_bifurcation(bifurcation_df, outdir / "bifurcation_sweep.png")

    # (iv) Four-type extension.
    four_type_config = SimulationConfig(
        beta=3.0,
        periods=1000,
        r=args.r,
        sigma2=args.sigma2,
        risk_aversion=args.risk_aversion,
        noise_std=args.noise_std,
        x0=args.x0,
        seed=args.seed,
    )
    four_type_result = run_simulation(
        traders=make_four_type_traders(
            trend_g=args.trend_g,
            contrarian_g=args.contrarian_g,
            optimist_bias=args.optimist_bias,
            fundamentalist_cost=args.fundamentalist_cost,
            optimist_cost=args.optimist_cost,
        ),
        config=four_type_config,
    )
    save_results_csv(four_type_result.series, outdir / "four_type_extension.csv")
    plot_price_and_weights(
        four_type_result,
        title="Four-type extension (beta=3.0)",
        output_path=outdir / "four_type_extension.png",
    )

    summary = pd.DataFrame(
        [
            {
                "experiment": "two_type_baseline",
                "output_csv": "two_type_baseline.csv",
                "output_plot": "two_type_baseline.png",
            },
            {
                "experiment": "beta_regime_comparison",
                "output_csv": "beta_regime_comparison.csv",
                "output_plot": "beta_regime_comparison.png",
            },
            {
                "experiment": "bifurcation_sweep",
                "output_csv": "bifurcation_sweep.csv",
                "output_plot": "bifurcation_sweep.png",
            },
            {
                "experiment": "four_type_extension",
                "output_csv": "four_type_extension.csv",
                "output_plot": "four_type_extension.png",
            },
        ]
    )
    save_results_csv(summary, outdir / "summary_outputs.csv")

    logging.info(f"Saved outputs to: {outdir.resolve()}")


if __name__ == "__main__":
    # set up logging
    setup_logging()

    main()
