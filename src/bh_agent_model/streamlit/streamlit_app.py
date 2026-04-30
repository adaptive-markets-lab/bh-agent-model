"""
Brock-Hommes ABM — Streamlit Presentation App.

Self-contained: agents, market, and all plots live in this single file.

No package imports needed — just run:

    streamlit run streamlit_app.py

Navigation (left sidebar):

    1. Motivation        — why this model matters
    2. Model Design      — how the ABM works (ODD summary)
    3. Live Demo         — interactive parameter sliders + instant plots
    4. Parameter Sweep   — how beta changes market dynamics
    5. Real data Analysis
    6. Sensitivity analysis
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.figure import Figure
from matplotlib.patches import Patch
from SALib.analyze import sobol
from SALib.sample import sobol as sobol_sample

from bh_agent_model.utils.base.agents import chartist, contrarian, fundamentalist, optimist
from bh_agent_model.utils.base.math_ops import softmax_stable
from bh_agent_model.utils.base.models import MarketDataLike, SobolResult
from bh_agent_model.utils.load_data.load_data_from_yfinance import load_data_from_yfinance

# ---------------------------------------------------------------------------
# Page config — must be the very first Streamlit call
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Brock-Hommes ABM",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


def run_simulation(
    beta: float,
    g_chartist: float,
    g_contrarian: float,
    b_optimist: float,
    noise_std: float,
    periods: int = 1000,
    seed: int = 42,
    sigma2: float = 0.25,
    risk_aversion: float = 5.0,
    r: float = 1.01,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Run BH simulation, return (x_history, weights_history, trader_names)."""
    np.random.seed(seed)

    traders = [
        fundamentalist(cost=0.001),
        chartist(g=g_chartist),
        contrarian(g=g_contrarian),
        optimist(b=b_optimist, cost=0.001),
    ]

    names = [t.name for t in traders]
    n = len(traders)

    weights = np.full(n, 1.0 / n)
    x = 0.0

    x_hist, w_hist = [], []

    for _ in range(periods):
        x_prev = x
        demands = np.array([t.demand(x_prev, r, sigma2, risk_aversion) for t in traders])
        noise = np.random.normal(0.0, noise_std)

        x_new = (np.dot(weights, demands) + noise) / r
        realized = x_new - x_prev

        for t in traders:
            t.update_fitness(realized)

        fitnesses = np.array([t.fitness for t in traders])
        weights = softmax_stable(beta, fitnesses)

        x = x_new
        x_hist.append(x)
        w_hist.append(weights.copy())

    return np.array(x_hist), np.array(w_hist), names


@st.cache_data(show_spinner=False)
def run_real_data_strategy_simulation(
    ticker: str = "^GSPC",
    start_date: str = "2016-01-01",
    end_date: str = "2025-12-31",
    beta: float = 0.5,
    r: float = 1.01,
    risk_aversion: float = 5.0,
) -> tuple[MarketDataLike, np.ndarray, list[str]]:
    """Run BH strategy-weight simulation on real asset returns."""
    data = load_data_from_yfinance(
        ticker=ticker,
        start_date=start_date,
        end_date=end_date,
    )

    returns = np.asarray(data.returns).reshape(-1)

    traders = [
        fundamentalist(cost=0.0002),
        chartist(g=1.1),
        contrarian(g=-0.8),
        optimist(b=0.0005, cost=0.0001),
    ]

    sigma2 = max(data.sigma2, 1e-4)

    for trader in traders:
        trader.reset()

    weights_history = []

    for t in range(1, len(returns)):
        x_prev = float(returns[t - 1])
        realized_return = float(returns[t])

        for trader in traders:
            trader.demand(
                x_prev=x_prev,
                r=r,
                sigma2=sigma2,
                risk_aversion=risk_aversion,
            )
            trader.update_fitness(realized_return=realized_return)

        fitnesses = np.array([trader.fitness for trader in traders], dtype=float)

        max_f = np.max(fitnesses)
        weights = np.exp(beta * (fitnesses - max_f))
        weights /= np.sum(weights)

        weights_history.append(weights.copy())

    return data, np.array(weights_history), [trader.name for trader in traders]


# ---------------------------------------------------------------------------
# ── Shared plot helpers ──────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

COLORS = ["#2196F3", "#FF9800", "#4CAF50", "#F44336"]  # blue, orange, green, red

SOBOL_PROBLEM = {
    "num_vars": 9,
    "names": [
        "beta",
        "r",
        "sigma2",
        "risk_aversion",
        "g_chartist",
        "g_contrarian",
        "b_optimist",
        "cost_fundamentalist",
        "cost_optimist",
    ],
    "bounds": [
        [0.0, 200.0],
        [1.001, 1.05],
        [1e-5, 1e-2],
        [1.0, 20.0],
        [0.5, 2.0],
        [-2.0, -0.1],
        [0.0, 0.005],
        [0.0, 0.002],
        [0.0, 0.002],
    ],
}

OUTPUT_METRICS = [
    ("mean_weight_fundamentalist", 0, "mean"),
    ("mean_weight_chartist", 1, "mean"),
    ("mean_weight_contrarian", 2, "mean"),
    ("mean_weight_optimist", 3, "mean"),
    ("dominance_fundamentalist", 0, "dominance"),
    ("dominance_chartist", 1, "dominance"),
    ("dominance_contrarian", 2, "dominance"),
    ("dominance_optimist", 3, "dominance"),
    ("switching_volatility", -1, "std_market"),
]


def run_sobol_simulation_from_array(returns: np.ndarray, param_row: np.ndarray) -> np.ndarray:
    """Run BH strategy-weight sobol simulation on real asset returns."""
    (
        beta,
        r,
        sigma2,
        risk_aversion,
        g_chartist,
        g_contrarian,
        b_optimist,
        cost_fundamentalist,
        cost_optimist,
    ) = param_row

    sigma2 = max(float(sigma2), 1e-8)

    traders = [
        fundamentalist(cost=float(cost_fundamentalist)),
        chartist(g=float(g_chartist)),
        contrarian(g=float(g_contrarian)),
        optimist(b=float(b_optimist), cost=float(cost_optimist)),
    ]

    for trader in traders:
        trader.reset()

    n_periods = len(returns) - 1
    weights_history = np.empty((n_periods, len(traders)), dtype=float)

    for t in range(1, len(returns)):
        x_prev = float(returns[t - 1])
        realized_return = float(returns[t])

        for trader in traders:
            trader.demand(
                x_prev=x_prev,
                r=float(r),
                sigma2=sigma2,
                risk_aversion=float(risk_aversion),
            )
            trader.update_fitness(realized_return=realized_return)

        fitnesses = np.array([trader.fitness for trader in traders], dtype=float)
        weights = softmax_stable(float(beta), fitnesses)

        weights_history[t - 1] = weights

    return weights_history


def extract_sobol_outputs(weights_history: np.ndarray) -> np.ndarray:
    """Extract sobol outputs from history and return them in a numpy array."""
    mean_w = weights_history.mean(axis=0)
    std_w = weights_history.std(axis=0)

    dominance = np.array(
        [np.mean(weights_history[:, h] > 0.30) for h in range(4)],
        dtype=float,
    )
    switching_vol = float(std_w.mean())

    outputs = np.empty(len(OUTPUT_METRICS), dtype=float)

    for i, (_, trader_idx, stat) in enumerate(OUTPUT_METRICS):
        if stat == "mean":
            outputs[i] = mean_w[trader_idx]
        elif stat == "dominance":
            outputs[i] = dominance[trader_idx]
        elif stat == "std_market":
            outputs[i] = switching_vol

    return outputs


@st.cache_data(show_spinner=False)
def run_sobol_cached(
    ticker: str,
    start_date: str,
    end_date: str,
    n_base: int,
) -> tuple[dict[str, SobolResult], pd.DataFrame]:
    """Run sobol with cached."""
    data = load_data_from_yfinance(
        ticker=ticker,
        start_date=start_date,
        end_date=end_date,
    )
    returns = np.asarray(data.returns).reshape(-1)

    problem = SOBOL_PROBLEM
    param_matrix = sobol_sample.sample(problem, n_base, calc_second_order=True)

    output_matrix = np.empty((param_matrix.shape[0], len(OUTPUT_METRICS)), dtype=float)

    for i, param_row in enumerate(param_matrix):
        weights_history = run_sobol_simulation_from_array(returns, param_row)
        output_matrix[i] = extract_sobol_outputs(weights_history)

    results = {}

    for col_idx, (metric_name, _, _) in enumerate(OUTPUT_METRICS):
        y = output_matrix[:, col_idx]

        if np.std(y) < 1e-12:
            results[metric_name] = SobolResult(
                output_name=metric_name,
                param_names=problem["names"],
                s1=np.zeros(len(problem["names"])),
                s1_conf=np.zeros(len(problem["names"])),
                st=np.zeros(len(problem["names"])),
                st_conf=np.zeros(len(problem["names"])),
            )
            continue

        si = sobol.analyze(
            problem,
            y,
            calc_second_order=True,
            print_to_console=False,
        )

        results[metric_name] = SobolResult(
            output_name=metric_name,
            param_names=problem["names"],
            s1=np.array(si["S1"]),
            s1_conf=np.array(si["S1_conf"]),
            st=np.array(si["ST"]),
            st_conf=np.array(si["ST_conf"]),
        )

    rows = []
    for metric_name, result in results.items():
        for i, param in enumerate(result.param_names):
            rows.append(
                {
                    "Output": metric_name,
                    "Parameter": param,
                    "S1": result.s1[i],
                    "S1 conf": result.s1_conf[i],
                    "ST": result.st[i],
                    "ST conf": result.st_conf[i],
                    "ST - S1": result.st[i] - result.s1[i],
                }
            )

    return results, pd.DataFrame(rows)


def plot_sobol_results(results: dict[str, SobolResult]) -> Figure:
    """Create plot for sobol results."""
    n_outputs = len(results)
    n_cols = 3
    n_rows = -(-n_outputs // n_cols)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(6 * n_cols, 4 * n_rows),
        squeeze=False,
    )

    ax_flat = axes.flatten()

    short_labels = {
        "beta": r"$\beta$",
        "r": r"$r$",
        "sigma2": r"$\sigma^2$",
        "risk_aversion": r"$a$",
        "g_chartist": r"$g_c$",
        "g_contrarian": r"$g_{ct}$",
        "b_optimist": r"$b_{opt}$",
        "cost_fundamentalist": r"$c_f$",
        "cost_optimist": r"$c_o$",
    }

    for ax_idx, (metric_name, result) in enumerate(results.items()):
        ax = ax_flat[ax_idx]
        x = np.arange(len(result.param_names))
        width = 0.35

        ax.bar(
            x - width / 2,
            result.s1,
            width,
            yerr=result.s1_conf,
            capsize=3,
            label="S1 direct",
            color="#2196F3",
            alpha=0.85,
        )
        ax.bar(
            x + width / 2,
            result.st,
            width,
            yerr=result.st_conf,
            capsize=3,
            label="ST total",
            color="#F44336",
            alpha=0.85,
        )

        ax.set_title(metric_name.replace("_", " "), fontsize=9, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(
            [short_labels.get(n, n) for n in result.param_names],
            fontsize=8,
        )
        ax.set_ylabel("Sobol index", fontsize=8)
        ax.set_ylim(0, min(1.05, max(result.st.max(), result.s1.max()) * 1.3 + 0.05))
        ax.axhline(0, color="black", linewidth=0.5)
        ax.tick_params(labelsize=8)

        if ax_idx == 0:
            ax.legend(fontsize=7)

    for ax_idx in range(len(results), len(ax_flat)):
        ax_flat[ax_idx].set_visible(False)

    fig.suptitle(
        "Sobol Sensitivity Analysis — Brock-Hommes Model\n" "S1 = direct effect | ST = total effect | Gap = interaction",
        fontsize=11,
        fontweight="bold",
    )

    plt.tight_layout()
    return fig


def plot_main(x_hist: np.ndarray, w_hist: np.ndarray, names: list, title="Simulated Price Deviation — BH ABM") -> Figure:
    """Plot price deviation and population weights over time."""
    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    steps = np.arange(len(x_hist))

    ax0.plot(steps, x_hist, lw=0.8, color="steelblue")
    ax0.axhline(0, color="black", lw=0.5, ls="--", alpha=0.5)
    ax0.set_ylabel("Price deviation (x_t)")
    ax0.set_title(title)

    equal = 1.0 / len(names)
    for i, name in enumerate(names):
        ax1.plot(steps, w_hist[:, i], label=name, color=COLORS[i], lw=0.9)
    ax1.axhline(equal, color="black", lw=0.5, ls="--", alpha=0.5, label=f"Equal share ({equal:.2f})")
    ax1.set_ylim(0, 1)
    ax1.set_ylabel("Population weight")
    ax1.set_xlabel("Time step")
    ax1.set_title("Strategy Dominance")
    ax1.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    return fig


def plot_rolling(x_hist: np.ndarray, w_hist: np.ndarray, names, window=20) -> Figure:
    """Plot raw and rolling-average strategy weights for all strategies."""
    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    steps = np.arange(len(x_hist))

    ax0.plot(steps, x_hist, lw=0.8, color="steelblue")
    ax0.axhline(0, color="black", lw=0.5, ls="--", alpha=0.5)
    ax0.set_ylabel("Price deviation (x_t)")
    ax0.set_title("Price Deviation")

    equal = 1.0 / w_hist.shape[1]

    for i, name in enumerate(names):
        raw = w_hist[:, i]
        ma = pd.Series(raw).rolling(window, min_periods=1).mean().values

        ax1.plot(steps, raw, color=COLORS[i], alpha=0.15, lw=0.7)
        ax1.plot(steps, ma, color=COLORS[i], lw=2.0, label=f"{name} ({window}-step MA)")

    ax1.axhline(equal, color="black", lw=0.5, ls="--", alpha=0.5, label=f"Equal share ({equal:.2f})")
    ax1.set_ylim(0, 1)
    ax1.set_ylabel("Population weight")
    ax1.set_xlabel("Time step")
    ax1.set_title("Strategy Dominance — All Strategies (smoothed)")
    ax1.legend(loc="upper right", fontsize=8)

    plt.tight_layout()
    return fig


def plot_phase(x_hist: np.ndarray, w_hist: np.ndarray, names: list) -> Figure:
    """Plot price deviation against next-period weight for all strategies."""
    equal = 1.0 / w_hist.shape[1]

    fig, ax = plt.subplots(figsize=(7, 7))

    for i, name in enumerate(names):
        ax.scatter(
            x_hist[:-1],
            w_hist[1:, i],
            alpha=0.25,
            s=6,
            color=COLORS[i],
            label=name,
        )

    ax.axvline(0, color="black", lw=0.5, ls="--", alpha=0.6)
    ax.axhline(equal, color="black", lw=0.5, ls="--", alpha=0.6)

    ax.set_xlabel("Price deviation x_t")
    ax.set_ylabel("Strategy weight (t+1)")
    ax.set_title("Phase plot: Price vs Next-Period Strategy Dominance")
    ax.legend(markerscale=2, fontsize=8)

    plt.tight_layout()
    return fig


def plot_regime(x_hist: np.ndarray, w_hist: np.ndarray, names: list) -> Figure:
    """Plot price deviation with shaded dominant-strategy regimes."""
    dominant = np.argmax(w_hist, axis=1)

    regime_colors = {
        0: "cornflowerblue",
        1: "orange",
        2: "lightgreen",
        3: "lightcoral",
    }

    periods = len(x_hist)
    steps = np.arange(periods)

    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    prev, start = dominant[0], 0
    for t in range(1, periods):
        if dominant[t] != prev or t == periods - 1:
            ax0.axvspan(start, t, color=regime_colors.get(prev, "gray"), alpha=0.25)
            ax1.axvspan(start, t, color=regime_colors.get(prev, "gray"), alpha=0.15)
            start, prev = t, dominant[t]

    ax0.plot(steps, x_hist, lw=0.8, color="black", zorder=3)
    ax0.axhline(0, color="grey", lw=0.5, ls="--")
    ax0.set_ylabel("Price deviation (x_t)")
    ax0.set_title("Price Deviation — shaded by dominant strategy")

    ax0.legend(
        handles=[
            Patch(
                facecolor=regime_colors[i],
                alpha=0.5,
                label=f"{names[i]} dominant",
            )
            for i in range(len(names))
        ],
        loc="upper right",
        fontsize=8,
    )

    equal = 1.0 / len(names)

    for i, name in enumerate(names):
        ax1.plot(steps, w_hist[:, i], label=name, color=COLORS[i], lw=0.9)

    ax1.axhline(equal, color="black", lw=0.5, ls="--", alpha=0.5)
    ax1.set_ylim(0, 1)
    ax1.set_ylabel("Population weight")
    ax1.set_xlabel("Time step")
    ax1.set_title("Strategy Weights")
    ax1.legend(loc="upper right", fontsize=8)

    plt.tight_layout()
    return fig


def plot_real_data_strategy_weights(data, weights_history, trader_names) -> Figure:
    """Plot real asset price and inferred strategy weights."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    axes[0].plot(data.dates, data.prices, lw=1.0, color="black")
    axes[0].set_title("S&P 500 Price")
    axes[0].set_ylabel("Price")

    colors = ["#2196F3", "#FF9800", "#9C27B0", "#4CAF50"]

    for i, name in enumerate(trader_names):
        axes[1].plot(
            data.dates[1:],
            weights_history[:, i],
            label=name,
            lw=0.9,
            color=colors[i % len(colors)],
        )

    axes[1].set_title("Strategy Dominance on Real S&P 500 Returns")
    axes[1].set_ylabel("Population weight")
    axes[1].set_xlabel("Date")
    axes[1].set_ylim(0, 1)
    axes[1].legend(loc="upper right", fontsize=8)

    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# ── Sidebar navigation ───────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

st.sidebar.title("📈 BH Agent-Based Model")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Navigate",
    [
        "1 · Motivation",
        "2 · Model Design",
        "3 · Live Demo",
        "4 · Parameter Sweep",
        "5 · Real Data Analysis",
        "6 · Sobol Sensitivity",
    ],
)
st.sidebar.markdown("---")
st.sidebar.caption("Brock & Hommes (1998) — Heterogeneous Beliefs and Routes to Chaos")


# ===========================================================================
# PAGE 1 — MOTIVATION
# ===========================================================================
if page == "1 · Motivation":
    st.title("📈 Brock-Hommes Agent-Based Model")
    st.subheader("Why do financial markets become unstable — even when traders follow simple, reasonable rules?")

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### The Problem")
        st.markdown("""
        Standard economic models assume all traders are identical and fully rational.
        Under those assumptions, prices should always reflect true fundamental value.

        But real markets show:
        - 📉 **Bubbles** — prices rising far above fundamentals
        - 💥 **Crashes** — sudden violent reversals
        - 🔄 **Volatility clustering** — calm periods followed by turbulent ones
        - 📊 **Fat-tailed returns** — extreme events far more common than theory predicts

        *Something is missing from the standard story.*
        """)

    with col2:
        st.markdown("### Our Answer")
        st.markdown("""
        Brock & Hommes (1998) showed that these patterns **emerge naturally** when you
        allow traders to hold *different beliefs* and *switch strategies* based on past performance.

        No irrationality needed. No external shocks needed.
        Complex market dynamics arise from a simple feedback loop:

        > **beliefs → prices → profits → switching → new beliefs**

        This is **emergence** — macro-level chaos from micro-level rules.
        """)

    st.markdown("---")
    st.markdown("### Policy Relevance")
    col3, col4, col5 = st.columns(3)
    with col3:
        st.info("**Regulation**\nUnderstanding what tips markets into instability helps design circuit-breakers and position limits.")
    with col4:
        st.info("**Risk Management**\nFitness-based switching explains why herding happens, informing stress-testing.")
    with col5:
        st.info("**Forecasting**\nStrategy composition predicts volatility regimes better than price history alone.")

    st.markdown("---")
    st.markdown("### Research Question")
    st.success(
        "**How does heterogeneous belief formation and evolutionary strategy switching "
        "generate emergent financial market dynamics — including bubbles, crashes, and "
        "volatility clustering — in the Brock-Hommes Adaptive Belief System?**"
    )


# ===========================================================================
# PAGE 2 — MODEL DESIGN
# ===========================================================================
elif page == "2 · Model Design":
    st.title("🧩 Model Design — ODD Summary")
    st.markdown("*Following the Overview, Design concepts, Details (ODD) protocol (Grimm et al., 2006)*")
    st.markdown("---")

    tab1, tab2, tab3 = st.tabs(["Overview", "Design Concepts", "Details"])

    with tab1:
        st.markdown("### Purpose")
        st.markdown("""
        Replicate and extend the Brock-Hommes (1998) Adaptive Belief System to demonstrate
        how heterogeneous trader expectations and evolutionary strategy switching produce
        emergent market dynamics: bubbles, crashes, and volatility clustering.
        """)

        st.markdown("### Agents")
        st.markdown(r"""
        | Strategy        | Belief Rule                          | Role |
        |-----------------|--------------------------------------|------|
        | Fundamentalist  | $f_h = 0$                            | Stabilising — anchors price |
        | Chartist        | $f_h = g \, x_{t-1}$                 | Destabilising — amplifies trends |
        | Contrarian      | $f_h = -g \, x_{t-1}$                | Stabilising — counteracts trends |
        | Optimist        | $f_h = b$                            | Sentiment — upward pressure |
        """)

        st.markdown("### Environment")
        st.markdown("""
        - Single risky asset with price deviation $x_t$ from fundamental value
        - Risk-free asset with gross return $R = 1.01$
        - Market clears each period via excess demand
        """)

        st.markdown("### Time")
        st.markdown("Discrete time steps. Each step = one trading period.")

    with tab2:
        st.markdown("### Key Design Concepts")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**🔄 Emergence**")
            st.markdown("Bubble-crash cycles arise from the strategy-switching feedback loop — not programmed in, but arising from agent interactions.")

            st.markdown("**📊 Fitness & Selection**")
            st.markdown("""
            Each strategy earns fitness (profit minus cost) each period:

            $$
            U_{h,t} = \\text{realized\\_return} \\times \\text{last\\_demand} - \\text{cost}
            $$

            Exponentially Weighted Average (EWA) with memory parameter η=0.5.
            """)

        with col2:
            st.markdown("**🧬 Evolutionary Switching**")
            st.markdown("""
            Population shares update via the discrete-choice softmax rule:

            $$
            n_{h,t} = \\frac{\\exp(\\beta \\cdot U_h)}{\\sum_{k} \\exp(\\beta \\cdot U_k)}
            $$

            $\\beta$ (intensity of switching) controls switching speed:
            - Low $\\beta$ → weak response to fitness differences
            - High $\\beta$ → winner-takes-all dynamics
            """)

            st.markdown("**💰 Demand**")

            st.markdown("""
            Mean-variance optimal demand:

            $$
            z_h = \\frac{f_h - R \\cdot x_{t-1}}{a \\cdot \\sigma^2}
            $$
            """)

        st.markdown("**🔁 The Feedback Loop**")
        st.code(
            """
beliefs (forecast rules)
    → demands (mean-variance optimisation)
        → price (weighted average demand / R + noise)
            → realised return
                → fitness update (EWA)
                    → strategy switching (softmax)
                        → new beliefs  [repeat]
        """,
            language="text",
        )

    with tab3:
        st.markdown("### Initialisation")
        st.markdown("""
        - Equal population weights: $n_{h,0} = 1/4$ for all strategies
        - Starting price deviation: $x_0 = 0$
        - All fitness and demand histories: 0
        - Random seed fixed for reproducibility
        """)

        st.markdown("### Price Formation Equation")
        st.latex(r"x_t = \frac{1}{R} \left( \sum_h n_{h,t} \cdot z_{h,t} + \varepsilon_t \right)")
        st.markdown("where $ε_t$ ~ $N(0, σ_{noise})$")

        st.markdown("### Stability Condition")
        st.markdown("""
        The fundamental equilibrium ($x=0$) is stable when $g < R$ for the chartist.
        With $g=1.2 > R=1.01$, the system is **above the instability threshold**,
        generating endogenous boom-bust cycles.
        """)

        st.markdown("### Calibrated Parameters")
        st.markdown(r"""
        ### Calibrated Parameters

        | Parameter | Value | Why |
        |----------|------|-----|
        | $\beta$ (intensity) | 200 | Sharp switching — verified numerically stable |
        | $\sigma^2$ (variance) | 0.25 | Calibrated so demands stay in ~0.1–0.5 range |
        | $a$ (risk aversion) | 5.0 | Moderate risk aversion — prevents demand explosion |
        | $\sigma_{\text{noise}}$ (noise std) | 0.1 | Sufficient to kick system between regimes |
        | $g$ (chartist trend) | 1.2 | Above stability threshold ($g > R = 1.01$) |
        | $\eta$ (EWA memory) | 0.5 | 2-period weighted average of past profits |
        """)


# ===========================================================================
# PAGE 3 — LIVE DEMO
# ===========================================================================
elif page == "3 · Live Demo":
    st.title("🎮 Live Demo — Interact with the Model")
    st.markdown("Adjust the sliders and watch the market dynamics change in real time.")
    st.markdown("---")

    # ── Sidebar sliders ──────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### 🎛 Model Parameters")

        beta = st.slider(
            "β — Intensity of switching", min_value=0.0, max_value=500.0, value=200.0, step=10.0, help="How aggressively traders switch to better-performing strategies. Higher = faster switching."
        )
        g_chartist = st.slider("g — Chartist trend strength", min_value=0.5, max_value=2.0, value=1.2, step=0.05, help="How strongly chartists extrapolate past trends. g > 1.01 is destabilising.")
        g_contrarian = st.slider(
            "g_c — Contrarian strength",
            min_value=0.0,
            max_value=2.0,
            value=0.8,
            step=0.05,
            help="How strongly contrarians bet against price deviations. Larger values create stronger reversal pressure.",
        )
        b_optimist = st.slider("b — Optimist bias", min_value=0.0, max_value=0.1, value=0.05, step=0.01, help="Constant upward price bias held by the optimist strategy.")
        noise_std = st.slider("σ_noise — Market noise", min_value=0.0, max_value=0.3, value=0.1, step=0.01, help="Standard deviation of random exogenous shocks each period.")
        periods = st.slider(
            "T — Simulation length",
            min_value=100,
            max_value=2000,
            value=1000,
            step=100,
        )
        seed = st.number_input("Random seed", value=42, step=1)

    # ── Run simulation ───────────────────────────────────────────────────────
    with st.spinner("Running simulation..."):
        x_hist, w_hist, names = run_simulation(
            beta=beta,
            g_chartist=g_chartist,
            g_contrarian=g_contrarian,
            b_optimist=b_optimist,
            noise_std=noise_std,
            periods=periods,
            seed=int(seed),
        )

    # ── Summary metrics ──────────────────────────────────────────────────────
    dominant = np.argmax(w_hist, axis=1)
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Price std", f"{x_hist.std():.3f}")
    col2.metric("Chartist dominant", f"{np.mean(dominant==1)*100:.1f}%")
    col3.metric("Chartist weight range", f"{w_hist[:,1].min():.2f} – {w_hist[:,1].max():.2f}")
    col4.metric("Max |price deviation|", f"{np.max(np.abs(x_hist)):.3f}")

    st.markdown("---")

    # ── Tabs for each plot ───────────────────────────────────────────────────
    t1, t2, t3 = st.tabs(["📊 Main Simulation", "📈 Rolling Dominance", "🎨 Regime View"])

    with t1:
        st.markdown("""
        **What to look for:** Weight lines that cross and diverge over time.
        Chartist (orange) should spike during trending episodes then collapse.
        Fundamentalist (blue) should rise during reversals.
        """)
        st.pyplot(plot_main(x_hist, w_hist, names))

    with t2:
        st.markdown("""
        **What to look for:** Smoothed dominance across all strategies.
        Chartists rise during trends, contrarians rise during reversals,
        and fundamentalists stabilise price around the fundamental value.
        """)
        st.pyplot(plot_rolling(x_hist, w_hist, names))

    with t3:
        st.markdown("""
        **What to look for:** Sustained coloured blocks, not per-step flickering.
        Blue blocks = fundamentalists stabilising the market.
        Orange blocks = chartist-driven trending/bubble episodes.
        """)
        st.pyplot(plot_regime(x_hist, w_hist, names))

    # ── Interpretation helper ────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🔍 Auto-interpretation")
    chart_std = w_hist[:, 1].std()
    price_std = x_hist.std()
    pct_chart = np.mean(dominant == 1) * 100

    if chart_std < 0.02:
        st.warning("⚠️ **No switching detected.** Weights are nearly frozen. " "Try increasing β or noise_std.")
    elif pct_chart > 40:
        st.error("🔴 **Chartist lock-in.** Chartists dominate too much — " "try reducing g or β.")
    elif price_std < 0.02:
        st.info("📉 **Stable fundamental equilibrium.** Prices barely move. " "Increase noise_std or g to push the system toward instability.")
    else:
        st.success(f"✅ **Emergence detected.** Chartist weight std = {chart_std:.3f} " f"(above 0.05 threshold). Market switches regimes. " f"Chartists dominant {pct_chart:.1f}% of steps.")


# ===========================================================================
# PAGE 4 — PARAMETER SWEEP
# ===========================================================================
elif page == "4 · Parameter Sweep":
    st.title("🔬 Parameter Sweep — How β Changes Everything")
    st.markdown("""
    The intensity-of-choice parameter **β** controls how sharply traders respond
    to fitness differences. This is one of the most important parameters in the BH model.

    Below we run the simulation across a range of β values and show how market
    dynamics change — from stable equilibrium to complex switching behaviour.
    """)
    st.markdown("---")

    col1, col2 = st.columns([1, 2])
    with col1:
        sweep_betas = st.multiselect(
            "Select β values to compare",
            options=[0.0, 1.0, 5.0, 20.0, 50.0, 100.0, 200.0, 500.0],
            default=[0.0, 5.0, 50.0, 200.0],
        )
        sweep_periods = st.slider("Steps per run", 200, 1000, 500, 100)
        sweep_g = st.slider("Chartist g", 0.5, 2.0, 1.2, 0.05)
        run_sweep = st.button("▶ Run sweep", type="primary")

    with col2:
        st.markdown("""
        **What each regime looks like:**

        | β | Behaviour |
        |---|-----------|
        | 0 | Equal weights always — no switching |
        | 1–10 | Slow drift toward better strategy |
        | 50–100 | Visible regime switching |
        | 200+ | Sharp switching — emergent boom-bust |
        | 500+ | Risk of winner-takes-all lock-in |
        """)

    if run_sweep and len(sweep_betas) > 0:
        st.markdown("---")
        st.markdown("### Strategy weight dynamics across β values")
        st.markdown(
            "Each panel shows how the chartist population share evolves over time at "
            "that β. This is where β's effect is clearest — price paths look similar "
            "because noise masks the signal, but weight dynamics change dramatically."
        )

        fig, axes = plt.subplots(len(sweep_betas), 1, figsize=(12, 3 * len(sweep_betas)), sharex=True)
        if len(sweep_betas) == 1:
            axes = [axes]

        summary_rows = []
        for ax, beta_val in zip(axes, sweep_betas):
            xh, wh, nm = run_simulation(
                beta=beta_val,
                g_chartist=sweep_g,
                g_contrarian=0.8,
                b_optimist=0.01,
                noise_std=0.1,
                periods=sweep_periods,
            )
            steps = np.arange(len(xh))
            equal = 1.0 / wh.shape[1]

            # Plot all four strategy weights
            for i, name in enumerate(nm):
                ax.plot(steps, wh[:, i], lw=0.9, color=COLORS[i], label=name if beta_val == sweep_betas[0] else "")
            ax.axhline(equal, color="black", lw=0.8, ls="--", alpha=0.6, label=f"Equal share ({equal:.2f})" if beta_val == sweep_betas[0] else "")
            ax.set_ylim(0, 1)
            ax.set_ylabel(f"β = {beta_val}", fontsize=10)
            ax.set_title(
                f"β = {beta_val}  |  Chartist weight std = {wh[:,1].std():.3f}  |  Price std = {xh.std():.3f}",
                fontsize=9,
            )

            dom = np.argmax(wh, axis=1)
            summary_rows.append(
                {
                    "β": beta_val,
                    "Price std": round(xh.std(), 4),
                    "Chartist weight std": round(wh[:, 1].std(), 4),
                    "Chartist dominant (%)": round(np.mean(dom == 1) * 100, 1),
                    "Max |x|": round(np.max(np.abs(xh)), 3),
                }
            )

        # Single shared legend at the top
        handles = [plt.Line2D([0], [0], color=COLORS[i], lw=1.5, label=nm[i]) for i in range(len(nm))]
        handles.append(plt.Line2D([0], [0], color="black", lw=0.8, ls="--", label=f"Equal share ({equal:.2f})"))
        axes[0].legend(handles=handles, loc="upper right", fontsize=8)
        axes[-1].set_xlabel("Time step")
        plt.tight_layout()
        st.pyplot(fig)

        st.markdown("---")
        st.markdown("### Summary statistics")
        st.dataframe(pd.DataFrame(summary_rows), use_container_width=True)

        # ── Weight std vs beta chart ─────────────────────────────────────────
        st.markdown("---")
        st.markdown("### Chartist weight variability vs β")
        st.markdown("Higher variability = more switching = more emergence.")

        fig2, ax2 = plt.subplots(figsize=(8, 4))
        betas_plot = [r["β"] for r in summary_rows]
        stds_plot = [r["Chartist weight std"] for r in summary_rows]
        ax2.bar(range(len(betas_plot)), stds_plot, color="#FF9800", alpha=0.8)
        ax2.set_xticks(range(len(betas_plot)))
        ax2.set_xticklabels([f"β={b}" for b in betas_plot])
        ax2.set_ylabel("Chartist weight std")
        ax2.set_title("Strategy switching intensity vs β")
        ax2.axhline(0.05, color="red", ls="--", lw=1, label="Emergence threshold (0.05)")
        ax2.legend()
        plt.tight_layout()
        st.pyplot(fig2)

    elif run_sweep:
        st.warning("Please select at least one β value.")


# ===========================================================================
# PAGE 5 — REAL DATA
# ===========================================================================
elif page == "5 · Real Data Analysis":
    st.title("📊 Real Data — Strategy Dominance on S&P 500")
    st.markdown("""
    This page connects the Brock-Hommes model to real S&P 500 data by asking whether
    observed market returns can be interpreted as shifts in dominance among heterogeneous
    trading strategies.

    Instead of generating artificial price deviations, the model uses observed returns
    to update strategy fitness and population weights. During turbulent or crash-like
    periods, the key question is whether the model shows visible shifts in strategy
    dominance rather than constant representative-agent behavior.
    """)

    st.markdown("---")

    col1, col2 = st.columns([1, 2])

    with col1:
        ticker = st.text_input("Ticker", value="^GSPC")
        start_date = st.date_input("Start date", value=pd.to_datetime("2016-01-01"))
        end_date = st.date_input("End date", value=pd.to_datetime("2025-12-31"))

        beta_real = st.slider(
            "β — Intensity of Switching",
            min_value=0.0,
            max_value=5.0,
            value=0.5,
            step=0.1,
            help="Controls how strongly strategies switch toward higher realized fitness.",
        )

        r_real = st.slider(
            "R — Gross risk-free return",
            min_value=1.0,
            max_value=1.05,
            value=1.01,
            step=0.001,
            format="%.3f",
        )

        risk_aversion_real = st.slider(
            "a — Risk aversion",
            min_value=1.0,
            max_value=20.0,
            value=5.0,
            step=0.5,
        )

        run_real = st.button("▶ Run real-data simulation", type="primary")

    with col2:
        st.markdown(r"""
        **Strategies used**

        | Strategy | Belief rule |
        |---|---|
        | Fundamentalist | $f_h = 0$ |
        | Chartist | $f_h = g x_{t-1}$ with $g = 1.1$ |
        | Contrarian | $f_h = g x_{t-1}$ with $g = -0.8$ |
        | Optimist | $f_h = b$ with $b = 0.1$ |

        Fitness is updated using realized market returns, then strategy shares are
        updated through the softmax rule.
        """)

    if run_real:
        with st.spinner("Loading market data and running strategy simulation..."):
            data, weights_history, trader_names = run_real_data_strategy_simulation(
                ticker=ticker,
                start_date=str(start_date),
                end_date=str(end_date),
                beta=beta_real,
                r=r_real,
                risk_aversion=risk_aversion_real,
            )

        st.markdown("---")
        st.pyplot(plot_real_data_strategy_weights(data=data, weights_history=weights_history, trader_names=trader_names))

        dominant = np.argmax(weights_history, axis=1)

        st.markdown("### Summary statistics")
        c1, c2, c3, c4 = st.columns(4)

        c1.metric("Observations", f"{len(weights_history):,}")
        c2.metric("Return variance", f"{data.sigma2:.6f}")
        c3.metric("Max strategy weight", f"{weights_history.max():.2f}")
        c4.metric("Most dominant", trader_names[np.bincount(dominant).argmax()])

        summary = pd.DataFrame(
            {
                "Strategy": trader_names,
                "Mean weight": weights_history.mean(axis=0),
                "Min weight": weights_history.min(axis=0),
                "Max weight": weights_history.max(axis=0),
                "Dominant periods (%)": [np.mean(dominant == i) * 100 for i in range(len(trader_names))],
            }
        )

        st.dataframe(
            summary.style.format(
                {
                    "Mean weight": "{:.3f}",
                    "Min weight": "{:.3f}",
                    "Max weight": "{:.3f}",
                    "Dominant periods (%)": "{:.1f}",
                }
            ),
            use_container_width=True,
        )


elif page == "6 · Sobol Sensitivity":
    st.title("🧪 Sobol Sensitivity Analysis")
    st.markdown("""
    This page runs a global Sobol sensitivity analysis for the Brock-Hommes model.

    **S1** measures the direct effect of each parameter.
    **ST** measures the total effect, including interactions.
    **ST - S1** captures interaction contribution.
    """)

    st.markdown("---")

    col1, col2 = st.columns([1, 2])

    with col1:
        ticker_sobol = st.text_input("Ticker", value="^GSPC")
        start_sobol = st.date_input("Sobol start date", value=pd.to_datetime("2016-01-01"))
        end_sobol = st.date_input("Sobol end date", value=pd.to_datetime("2025-12-31"))

        n_base = st.select_slider(
            "Base sample size N",
            options=[64, 128, 256, 512, 1024],
            value=256,
            help="Total runs = N × (2k + 2). With k=9, total runs = N × 20.",
        )

        total_runs = n_base * (2 * SOBOL_PROBLEM["num_vars"] + 2)
        st.metric("Total simulations", f"{total_runs:,}")

        run_sobol_button = st.button("▶ Run Sobol analysis", type="primary")

    with col2:
        st.markdown("""
        **Recommended use**

        | N | Runs | Use case |
        |---:|---:|---|
        | 64 | 1,280 | Fast demo |
        | 128 | 2,560 | Classroom run |
        | 256 | 5,120 | Reasonable app default |
        | 512 | 10,240 | More stable |
        | 1024 | 20,480 | Slow but stronger |
        """)

    if st.button("Clear Sobol cache"):
        run_sobol_cached.clear()
        st.success("Sobol cache cleared.")

    if run_sobol_button:
        run_sobol_cached.clear()  # forces fresh run every button click

        with st.spinner(f"Running {total_runs:,} simulations..."):
            results, sobol_table = run_sobol_cached(
                ticker=ticker_sobol,
                start_date=str(start_sobol),
                end_date=str(end_sobol),
                n_base=int(n_base),
            )
            sobol_table = sobol_table.fillna(0.0)

        st.success("Sobol analysis complete.")

        st.markdown("### Sobol index table")
        st.dataframe(
            sobol_table.style.format(
                {
                    "S1": "{:.3f}",
                    "S1 conf": "{:.3f}",
                    "ST": "{:.3f}",
                    "ST conf": "{:.3f}",
                    "ST - S1": "{:.3f}",
                }
            ),
            use_container_width=True,
        )

        csv = sobol_table.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download Sobol table as CSV",
            data=csv,
            file_name="sobol_sensitivity_results.csv",
            mime="text/csv",
        )

        st.markdown("### S1 vs ST by output metric")
        st.pyplot(plot_sobol_results(results))

        st.markdown("### Interpretation guide")
        st.info("""
        - Large **S1**: parameter has a strong direct effect.
        - Large **ST** but small **S1**: parameter matters mostly through interactions.
        - Large **ST - S1**: interaction-heavy parameter.
        - Near-zero **S1** and **ST**: parameter has little influence on that output.
        """)
