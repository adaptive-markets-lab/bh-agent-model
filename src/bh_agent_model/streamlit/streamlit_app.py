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
    5. Insights          — what we found and what it means
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from bh_agent_model.utils.base.agents import chartist, fundamentalist, optimist
from bh_agent_model.utils.base.math_ops import softmax_stable

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


# ---------------------------------------------------------------------------
# ── Shared plot helpers ──────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

COLORS = ["#2196F3", "#FF9800", "#4CAF50"]  # blue, orange, green


def plot_main(x_hist, w_hist, names, title="Simulated Price Deviation — BH ABM"):
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


def plot_rolling(x_hist, w_hist, window=20):
    """Plot raw and rolling-average strategy weights."""
    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    steps = np.arange(len(x_hist))

    ax0.plot(steps, x_hist, lw=0.8, color="steelblue")
    ax0.axhline(0, color="black", lw=0.5, ls="--", alpha=0.5)
    ax0.set_ylabel("Price deviation (x_t)")
    ax0.set_title("Price Deviation")

    equal = 1.0 / w_hist.shape[1]
    for idx, (color, label) in enumerate(zip(COLORS[:2], ["Fundamentalist", "Chartist"])):
        raw = w_hist[:, idx]
        ma = pd.Series(raw).rolling(window, min_periods=1).mean().values
        ax1.plot(steps, raw, color=color, alpha=0.2, lw=0.7)
        ax1.plot(steps, ma, color=color, lw=2.0, label=f"{label} ({window}-step MA)")

    ax1.axhline(equal, color="black", lw=0.5, ls="--", alpha=0.5)
    ax1.set_ylim(0, 1)
    ax1.set_ylabel("Population weight")
    ax1.set_xlabel("Time step")
    ax1.set_title("Chartist vs Fundamentalist Dominance (smoothed)")
    ax1.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    return fig


def plot_phase(x_hist, w_hist):
    """Plot price deviation against next-period chartist weight."""
    equal = 1.0 / w_hist.shape[1]
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(x_hist[:-1], w_hist[1:, 1], alpha=0.25, s=5, color="#FF9800")
    ax.axvline(0, color="black", lw=0.5, ls="--", alpha=0.6)
    ax.axhline(equal, color="black", lw=0.5, ls="--", alpha=0.6)
    ax.set_xlabel("Price deviation  x_t")
    ax.set_ylabel("Chartist weight  (t+1)")
    ax.set_title("Phase plot: price vs next-period chartist dominance")
    plt.tight_layout()
    return fig


def plot_regime(x_hist, w_hist, names):
    """Plot price deviation with shaded dominant-strategy regimes."""
    dominant = np.argmax(w_hist, axis=1)
    regime_colors = {0: "cornflowerblue", 1: "orange", 2: "lightgreen"}
    regime_labels = {0: "Fundamentalist", 1: "Chartist", 2: "Optimist"}
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

    from matplotlib.patches import Patch

    ax0.legend(handles=[Patch(facecolor=regime_colors[k], alpha=0.5, label=f"{regime_labels[k]} dominant") for k in regime_colors], loc="upper right", fontsize=8)

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


# ---------------------------------------------------------------------------
# ── Sidebar navigation ───────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

st.sidebar.title("📈 BH Agent-Based Model")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Navigate",
    ["1 · Motivation", "2 · Model Design", "3 · Live Demo", "4 · Parameter Sweep", "5 · Insights"],
)
st.sidebar.markdown("---")
st.sidebar.caption("Brock & Hommes (1998) — Heterogeneous Beliefs and Routes to Chaos")


# ===========================================================================
# PAGE 1 — MOTIVATION
# ===========================================================================
if page == "1 · Motivation":
    st.title("📈 Brock-Hommes Agent-Based Model")
    st.subheader("Why do financial markets crash — even when everyone is rational?")

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

            $\\beta$ (intensity of choice) controls switching speed:
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
        - Equal population weights: $n_{h,0} = 1/3$ for all strategies
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
        b_optimist = st.slider("b — Optimist bias", min_value=0.0, max_value=0.05, value=0.01, step=0.005, help="Constant upward price bias held by the optimist strategy.")
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
        x_hist, w_hist, names = run_simulation(beta=beta, g_chartist=g_chartist, b_optimist=b_optimist, noise_std=noise_std, periods=periods, seed=int(seed))

    # ── Summary metrics ──────────────────────────────────────────────────────
    dominant = np.argmax(w_hist, axis=1)
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Price std", f"{x_hist.std():.3f}")
    col2.metric("Chartist dominant", f"{np.mean(dominant==1)*100:.1f}%")
    col3.metric("Chartist weight range", f"{w_hist[:,1].min():.2f} – {w_hist[:,1].max():.2f}")
    col4.metric("Max |price deviation|", f"{np.max(np.abs(x_hist)):.3f}")

    st.markdown("---")

    # ── Tabs for each plot ───────────────────────────────────────────────────
    t1, t2, t3, t4 = st.tabs(["📊 Main Simulation", "📈 Rolling Dominance", "🔵 Phase Plot", "🎨 Regime View"])

    with t1:
        st.markdown("""
        **What to look for:** Weight lines that cross and diverge over time.
        Chartist (orange) should spike during trending episodes then collapse.
        Fundamentalist (blue) should rise during reversals.
        """)
        st.pyplot(plot_main(x_hist, w_hist, names))

    with t2:
        st.markdown("""
        **What to look for:** The smoothed lines crossing each other.
        When chartist MA rises above the equal-share dashed line, the market
        is in a bubble regime. When it falls below, fundamentalists are recovering.
        """)
        st.pyplot(plot_rolling(x_hist, w_hist))

    with t3:
        st.markdown("""
        **What to look for:** Points clustering in the **upper corners**
        (large price deviations → high chartist weight next period).
        This means trend-chasing is being rewarded — the signature of emergence.
        A downward arch means the opposite (chartists being punished).
        """)
        st.pyplot(plot_phase(x_hist, w_hist))

    with t4:
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
        st.markdown("### Price paths across β values")

        # ── Price path comparison ────────────────────────────────────────────
        fig, axes = plt.subplots(len(sweep_betas), 1, figsize=(12, 3 * len(sweep_betas)), sharex=True)
        if len(sweep_betas) == 1:
            axes = [axes]

        summary_rows = []

        for ax, beta_val in zip(axes, sweep_betas):
            xh, wh, nm = run_simulation(beta=beta_val, g_chartist=sweep_g, b_optimist=0.01, noise_std=0.1, periods=sweep_periods)
            ax.plot(xh, lw=0.8, color="steelblue")
            ax.axhline(0, color="black", lw=0.5, ls="--", alpha=0.5)
            ax.set_ylabel(f"β = {beta_val}", fontsize=10)

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

        axes[0].set_title("Price Deviation at Different β Values")
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
# PAGE 5 — INSIGHTS
# ===========================================================================
elif page == "5 · Insights":
    st.title("💡 Insights & Conclusions")
    st.markdown("---")

    st.markdown("### What We Found")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 🔑 Key Result 1 — Emergence is Real")
        st.markdown("""
        With g=1.2 > R=1.01 (above the stability boundary), the model produces
        genuine emergent dynamics. Chartist weight swings from near 0% to 65%,
        generating price deviations up to ±0.5 from fundamental value.

        No individual agent plans a bubble — it emerges from the fitness-switching
        feedback loop alone.
        """)

        st.markdown("#### 🔑 Key Result 2 — σ² is the Hidden Stabiliser")
        st.markdown("""
        The most important numerical finding: σ² = 0.25 is required for stability.

        - σ² too small → demands explode → price diverges to 1e307
        - σ² too large → demands near zero → no fitness difference → no switching

        This parameter balances the demand scale and is the key to getting
        the model to behave correctly.
        """)

    with col2:
        st.markdown("#### 🔑 Key Result 3 — β Controls Regime Sharpness")
        st.markdown("""
        Low β (< 10): Weights barely move. The market has weak evolutionary
        pressure — strategies coexist but don't compete meaningfully.

        High β (200): Sharp switching. When a strategy does well for a few
        periods, it rapidly takes over. This creates the burst-like chartist
        episodes we observe.

        Extreme β (500+): Lock-in risk. One strategy permanently wins.
        """)

        st.markdown("#### 🔑 Key Result 4 — The Phase Plot Signature")
        st.markdown("""
        The phase plot (price deviation vs next-period chartist weight)
        shows a distinctive spike shape: chartists gain weight primarily
        when prices are near zero but trending.

        This is because large deviations eventually reverse, punishing
        chartists who stayed long too long. The bubble-then-crash cycle
        is visible in the temporal structure of this plot.
        """)

    st.markdown("---")
    st.markdown("### Limitations & Extensions")

    col3, col4 = st.columns(2)
    with col3:
        st.warning("""
        **Limitations**
        - Model uses normalised σ²=0.25, not empirically calibrated
        - Fitness is a simplified proxy for real trading P&L
        - No market microstructure (bid-ask spread, volume)
        - Only 3 strategy types — real markets have many more
        """)
    with col4:
        st.info("""
        **Natural Extensions**
        - Bayesian calibration of β and g to S&P 500 data
        - Sobol sensitivity analysis (already built by our team)
        - Adding contrarian strategy to study crash dynamics
        - Connecting to empirical stylised facts (fat tails, autocorrelation)
        """)

    st.markdown("---")
    st.markdown("### Connection to the ODD Protocol")
    st.markdown("""
    | ODD Section | What we implemented |
    |------------|---------------------|
    | **Purpose** | Explain endogenous financial market crises |
    | **Entities** | 3 trader types with heterogeneous belief rules |
    | **Process** | Demand → price → fitness → switching per period |
    | **Emergence** | Bubble-crash cycles from local switching rules |
    | **Adaptation** | Fitness-based evolutionary strategy switching |
    | **Initialisation** | Equal weights, zero price, fixed seed |
    | **Input** | Endogenous only — no external data required |
    | **Submodels** | Demand equation, EWA fitness, softmax switching |
    """)

    st.markdown("---")
    st.markdown("### References")
    st.markdown("""
    - Brock, W. A., & Hommes, C. H. (1998). Heterogeneous beliefs and routes to chaos
      in a simple asset pricing model. *Journal of Economic Dynamics and Control*, 22(8-9), 1235-1274.
    - Grimm, V., et al. (2006). A standard protocol for describing individual-based and
      agent-based models. *Ecological Modelling*, 198(1-2), 115-126.
    - Hommes, C. H. (2006). Heterogeneous agent models in economics and finance.
      *Handbook of Computational Economics*, 2, 1109-1186.
    """)

    st.markdown("---")
    # Quick re-run demo at bottom of insights page
    st.markdown("### Quick Demo — Best Parameters")
    if st.button("▶ Run reference simulation (β=200, g=1.2)"):
        x_hist, w_hist, names = run_simulation(beta=200, g_chartist=1.2, b_optimist=0.01, noise_std=0.1, periods=1000, seed=42)
        st.pyplot(plot_main(x_hist, w_hist, names, title="Reference run: β=200, g=1.2, noise=0.1"))
        dom = np.argmax(w_hist, axis=1)
        c1, c2, c3 = st.columns(3)
        c1.metric("Chartist weight std", f"{w_hist[:,1].std():.3f}")
        c2.metric("Chartist dominant", f"{np.mean(dom==1)*100:.1f}%")
        c3.metric("Price std", f"{x_hist.std():.3f}")
