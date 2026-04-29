import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

from bh_agent_model.utils.base.agents import chartist, contrarian, fundamentalist, optimist
from bh_agent_model.utils.base.markets import Market

if __name__ == "__main__":
    """
    Synthetic Simulation — Brock-Hommes ABM
    ========================================

    NOTE: This script runs exclusively on synthetically generated data.
    It is NOT designed for empirical/real-world return series.

    The market generates its own price path endogenously each period via:
        x_t = (Σ n_h · z_h(x_{t-1}) + ε) / R

    where ε ~ N(0, noise_std) is the exogenous shock injected at each step.
    This means the price deviation x_t is a direct output of the model's
    own internal dynamics — agent demands, fitness updates, and strategy
    switching — rather than being driven by observed data.

    Purpose:
        Isolate and study the theoretical mechanics of the Adaptive Belief
        System (ABS): how strategy switching (controlled by beta) interacts
        with trend-following (chartist g), mean-reversion (fundamentalist),
        and sentiment bias (optimist b) to produce emergent price dynamics.
    """
    # ---------------------------------------------------------------------------
    # Configuration: Model Hyperparameters
    # ---------------------------------------------------------------------------
    SEED = 42
    T = 1000  # Total simulation time steps
    np.random.seed(SEED)

    # Set output directory for plots
    base_path = Path(__file__).resolve().parent
    OUTPUT_DIR = str(base_path / "output" / "plots")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Define the composition of the agent population
    traders = [
        fundamentalist(cost=0.001),  # Keeps price near 0
        chartist(g=1.2),  # Source of instability/bubbles
        contrarian(g=-0.2),  # Stabilising — exploits overreaction / reverses trends
        optimist(b=0.1, cost=0.001),  # Constant upward pressure
    ]

    # Initialize the environment
    market = Market(
        traders=traders,
        beta=200.0,  # Intensity of choice: high values create rapid regime shifts
        r=1.01,  # Risk-free rate (1%)
        sigma2=0.25,  # Calibrated variance for numerical stability
        risk_aversion=5.0,  # Penalty for price uncertainty
        noise_std=0.1,  # Exogenous shocks that trigger strategy switching
    )

    # ---------------------------------------------------------------------------
    # Execution: Main Simulation Loop
    # ---------------------------------------------------------------------------
    x_history = []  # Stores price deviation over time
    weights_history = []  # Stores market share of each strategy over time

    for _ in range(T):
        x, w = market.step()
        x_history.append(x)
        weights_history.append(w)

    # Convert lists to arrays for vectorized plotting/analysis
    x_history = np.array(x_history)
    weights_history = np.array(weights_history)
    steps = np.arange(T)
    n_traders = len(traders)
    equal_share = 1.0 / n_traders
    chartist_idx = 1
    COLORS = ["blue", "orange", "green", "red"]
    WINDOW = 20  # Window size for rolling averages

    # ---------------------------------------------------------------------------
    # Visualization 1: Standard Time Series
    # Overview of price movements and strategy evolution.
    # ---------------------------------------------------------------------------
    fig1, axes1 = plt.subplots(2, 1, figsize=(13, 7), sharex=True)

    axes1[0].plot(steps, x_history, lw=0.8, color="steelblue")
    axes1[0].axhline(0, color="black", lw=0.5, ls="--", alpha=0.5)
    axes1[0].set_ylabel("Price deviation (x_t)")
    axes1[0].set_title("Simulated Price Deviation — Brock-Hommes ABM")

    for i, trader in enumerate(traders):
        axes1[1].plot(steps, weights_history[:, i], label=trader.name, color=COLORS[i], lw=0.9, alpha=0.85)

    axes1[1].axhline(equal_share, color="black", lw=0.5, ls="--", alpha=0.5, label=f"Equal share ({equal_share:.2f})")
    axes1[1].set_ylabel("Population weight")
    axes1[1].set_xlabel("Time step")
    axes1[1].set_title("Strategy Dominance")
    axes1[1].legend(loc="upper right")
    axes1[1].set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "fig1_main_simulation.png"), dpi=150, bbox_inches="tight")
    plt.show()

    # ---------------------------------------------------------------------------
    # Visualization 2: Trend Comparison (Smoothed)
    # Focuses on the struggle between rational agents and trend-followers.
    # ---------------------------------------------------------------------------
    fig2, axes2 = plt.subplots(2, 1, figsize=(13, 7), sharex=True)

    axes2[0].plot(steps, x_history, lw=0.8, color="steelblue")
    axes2[0].axhline(0, color="black", lw=0.5, ls="--", alpha=0.5)
    axes2[0].set_ylabel("Price deviation (x_t)")
    axes2[0].set_title("Price Deviation")

    for i, trader in enumerate(traders):
        raw = weights_history[:, i]
        ma = pd.Series(raw).rolling(WINDOW, min_periods=1).mean().values

        axes2[1].plot(steps, raw, color=COLORS[i], alpha=0.2, lw=0.8)
        axes2[1].plot(steps, ma, color=COLORS[i], lw=2.0, label=f"{trader.name} ({WINDOW}-step MA)")

    axes2[1].axhline(equal_share, color="black", lw=0.5, ls="--", alpha=0.5)
    axes2[1].set_ylabel("Population weight")
    axes2[1].set_xlabel("Time step")
    axes2[1].set_title("Strategy Dominance (smoothed)")
    axes2[1].legend(loc="upper right")
    axes2[1].set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "fig2_rolling_dominance.png"), dpi=150, bbox_inches="tight")
    plt.show()

    # ---------------------------------------------------------------------------
    # Visualization 3: Phase Analysis
    # Examines the correlation between price levels and future strategy popularity.
    # ---------------------------------------------------------------------------
    fig3, ax3 = plt.subplots(figsize=(7, 7))

    for i, trader in enumerate(traders):
        ax3.scatter(
            x_history[:-1],
            weights_history[1:, i],
            alpha=0.25,
            s=6,
            color=COLORS[i],
            label=trader.name,
        )

    ax3.axvline(0, color="black", lw=0.5, ls="--", alpha=0.6)
    ax3.axhline(equal_share, color="black", lw=0.5, ls="--", alpha=0.6)

    ax3.set_xlabel("Price deviation x_t")
    ax3.set_ylabel("Strategy weight (t+1)")
    ax3.set_title("Phase plot: Price vs Next-Period Strategy Dominance")

    ax3.legend(markerscale=2, fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "fig3_phase_plot.png"), dpi=150, bbox_inches="tight")
    plt.show()

    # ---------------------------------------------------------------------------
    # Visualization 4: Regime Segmentation
    # Shading the background based on which strategy is currently "winning".
    # ---------------------------------------------------------------------------
    dominant = np.argmax(weights_history, axis=1)

    regime_colors = {
        0: "cornflowerblue",
        1: "orange",
        2: "lightgreen",
        3: "lightcoral",
    }

    regime_labels = {
        0: "Fundamentalist dominant",
        1: "Chartist dominant",
        2: "Contrarian dominant",
        3: "Optimist dominant",
    }
    fig4, axes4 = plt.subplots(2, 1, figsize=(13, 7), sharex=True)

    prev, start = dominant[0], 0
    for t in range(1, T):
        if dominant[t] != prev or t == T - 1:
            axes4[0].axvspan(start, t, color=regime_colors.get(prev, "gray"), alpha=0.25)
            axes4[1].axvspan(start, t, color=regime_colors.get(prev, "gray"), alpha=0.15)
            start, prev = t, dominant[t]

    axes4[0].plot(steps, x_history, lw=0.8, color="black", zorder=3)
    axes4[0].axhline(0, color="grey", lw=0.5, ls="--")
    axes4[0].set_ylabel("Price deviation (x_t)")
    axes4[0].set_title("Price Deviation — shaded by dominant strategy")
    axes4[0].legend(
        handles=[Patch(facecolor=regime_colors[k], alpha=0.5, label=regime_labels[k]) for k in regime_colors],
        loc="upper right",
        fontsize=8,
    )

    for i, trader in enumerate(traders):
        axes4[1].plot(steps, weights_history[:, i], label=trader.name, color=COLORS[i], lw=1.2, alpha=0.9)

    axes4[1].axhline(equal_share, color="black", lw=0.5, ls="--", alpha=0.5)
    axes4[1].set_ylabel("Population weight")
    axes4[1].set_xlabel("Time step")
    axes4[1].set_title("Strategy Weights")
    axes4[1].legend(loc="upper right")
    axes4[1].set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "fig4_regime_highlight.png"), dpi=150, bbox_inches="tight")
    plt.show()
