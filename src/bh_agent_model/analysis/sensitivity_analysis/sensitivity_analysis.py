"""
Sobol sensitivity analysis for the Brock–Hommes agent model.

Uses the Saltelli sampling scheme (via SALib) to generate N*(2k+2) parameter
combinations, runs the BH simulation at each, and computes first-order (S1)
and total-order (ST) Sobol indices for each parameter × output combination.

Outputs:
    - printed index table per output metric
    - bar chart of S1 vs ST per output, saved to disk
    - live progress printed to terminal as batches complete

S1 : direct effect of a parameter on output variance (ignoring interactions)
ST : total effect including all interactions with other parameters
ST - S1 : interaction contribution — how much the parameter matters
           only in combination with others
"""

from __future__ import annotations

import logging
import time

import matplotlib.pyplot as plt
import numpy as np
from SALib.analyze import sobol
from SALib.sample import sobol as sobol_sample

from bh_agent_model.utils.base.agents import Trader, chartist, contrarian, fundamentalist, optimist
from bh_agent_model.utils.base.models import SobolResult
from bh_agent_model.utils.helper.setup_logging import setup_logging
from bh_agent_model.utils.load_data.load_data_from_yfinance import load_data_from_yfinance

# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Sobol problem definition
# ---------------------------------------------------------------------------

# SALib requires a problem dict that defines the parameter space to sample from.
# Bounds match the OAT sweep ranges so results are directly comparable.
SOBOL_PROBLEM: dict = {
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
        [0.0, 5.0],  # beta
        [1.001, 1.05],  # r
        [1e-5, 1e-2],  # sigma2
        [1.0, 20.0],  # risk_aversion
        [0.5, 2.0],  # g_chartist
        [-2.0, -0.1],  # g_contrarian
        [0.0, 0.005],  # b_optimist
        [0.0, 0.002],  # cost_fundamentalist
        [0.0, 0.002],  # cost_optimist
    ],
}

# Output metrics computed from each simulation run.
# Each entry is (label, trader_index, statistic) where statistic is one of
# 'mean', 'dominance', 'std_market'. Trader order: 0=Fund, 1=Chart, 2=Contr, 3=Opt.
OUTPUT_METRICS: list[tuple[str, int, str]] = [
    ("mean_weight_fundamentalist", 0, "mean"),
    ("mean_weight_chartist", 1, "mean"),
    ("mean_weight_contrarian", 2, "mean"),
    ("mean_weight_optimist", 3, "mean"),
    ("dominance_fundamentalist", 0, "dominance"),
    ("dominance_chartist", 1, "dominance"),
    ("dominance_contrarian", 2, "dominance"),
    ("dominance_optimist", 3, "dominance"),
    ("switching_volatility", -1, "std_market"),  # market-wide weight std
]


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------


def _build_traders(
    beta: float,
    r: float,
    sigma2: float,
    risk_aversion: float,
    g_chartist: float,
    g_contrarian: float,
    b_optimist: float,
    cost_fundamentalist: float,
    cost_optimist: float,
) -> tuple[list[Trader], float, float, float, float]:
    """
    Build traders and return them alongside the scalar market params.

    Returns:
        Tuple of (traders, beta, r, sigma2, risk_aversion).

    """
    traders = [
        fundamentalist(cost=cost_fundamentalist),
        chartist(g=g_chartist),
        contrarian(g=g_contrarian),
        optimist(b=b_optimist, cost=cost_optimist),
    ]
    return traders, beta, r, sigma2, risk_aversion


def run_simulation_from_array(
    returns: np.ndarray,
    param_row: np.ndarray,
) -> np.ndarray:
    """
    Run one BH simulation from a flat parameter array (Sobol sample row).

    Parameter order must match SOBOL_PROBLEM['names']:
        [beta, r, sigma2, risk_aversion, g_chartist, g_contrarian,
         b_optimist, cost_fundamentalist, cost_optimist]

    Args:
        returns: 1-D array of empirical log returns.
        param_row: 1-D array of 9 parameter values from Saltelli sampler.

    Returns:
        weights_history: Array of shape (n_periods, 4) with per-period weights.

    """
    (beta, r, sigma2, risk_aversion, g_chartist, g_contrarian, b_optimist, cost_fundamentalist, cost_optimist) = param_row

    # guard: sigma2 must be strictly positive for demand calculation
    sigma2 = max(float(sigma2), 1e-8)

    traders, beta, r, sigma2, risk_aversion = _build_traders(
        beta=float(beta),
        r=float(r),
        sigma2=sigma2,
        risk_aversion=float(risk_aversion),
        g_chartist=float(g_chartist),
        g_contrarian=float(g_contrarian),
        b_optimist=float(b_optimist),
        cost_fundamentalist=float(cost_fundamentalist),
        cost_optimist=float(cost_optimist),
    )

    n_periods = len(returns) - 1
    n_traders = len(traders)
    weights_history = np.empty((n_periods, n_traders), dtype=float)

    for trader in traders:
        trader.reset()

    for t in range(1, len(returns)):
        x_prev = float(returns[t - 1])
        realized_return = float(returns[t - 1])

        for trader in traders:
            trader.demand(
                x_prev=x_prev,
                r=r,
                sigma2=sigma2,
                risk_aversion=risk_aversion,
            )
            trader.update_fitness(realized_return=realized_return)

        fitnesses = np.array([trader.fitness for trader in traders], dtype=float)

        # numerically stable softmax (log-sum-exp trick)
        max_f = np.max(fitnesses)
        weights = np.exp(beta * (fitnesses - max_f))
        weights /= np.sum(weights)

        weights_history[t - 1] = weights

    return weights_history


def _extract_outputs(weights_history: np.ndarray) -> np.ndarray:
    """
    Extract all output metrics from a weight history array.

    Output order matches OUTPUT_METRICS list.

    Args:
        weights_history: Array of shape (n_periods, n_traders).

    Returns:
        1-D array of scalar output values, length = len(OUTPUT_METRICS).

    """
    mean_w = weights_history.mean(axis=0)
    std_w = weights_history.std(axis=0)
    dominant = np.argmax(weights_history, axis=1)
    dominance = np.array(
        [np.mean(dominant == h) for h in range(4)],
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


# ---------------------------------------------------------------------------
# Sobol runner
# ---------------------------------------------------------------------------


def run_sobol(
    returns: np.ndarray,
    n_base: int = 512,
    problem: dict | None = None,
    save_path: str = "sobol_sensitivity_results.png",
) -> dict[str, SobolResult]:
    """
    Run the full Sobol sensitivity analysis.

    Generates N*(2k+2) samples via Saltelli scheme, evaluates the simulation
    at each, and computes S1 and ST indices for every output metric.

    Args:
        returns: 1-D array of empirical log returns.
        n_base: Base sample size N. Total runs = N*(2k+2) = N*20 for k=9.
                512 gives ~10,000 runs (recommended minimum for stable estimates).
                256 gives ~5,000 runs (faster, less stable).
        problem: Optional override for SOBOL_PROBLEM definition.
        save_path: File path for the saved PNG figure.

    Returns:
        Dictionary mapping output metric name to its SobolResult.

    """
    if problem is None:
        problem = SOBOL_PROBLEM

    n_params = problem["num_vars"]
    n_outputs = len(OUTPUT_METRICS)

    # generate sample matrix first — let SALib decide the row count so that
    # calc_second_order is consistent between sampling and analysis
    logging.info("Generating Sobol sample: N=%d, k=%d", n_base, n_params)
    param_matrix = sobol_sample.sample(problem, n_base, calc_second_order=True)
    total_runs = param_matrix.shape[0]
    logging.info("Total simulation runs: %d", total_runs)

    # model output matrix — shape (total_runs, n_outputs)
    output_matrix = np.empty((total_runs, n_outputs), dtype=float)

    t_start = time.time()
    report_every = max(1, total_runs // 20)  # log progress ~20 times

    for i, param_row in enumerate(param_matrix):
        weights_history = run_simulation_from_array(returns, param_row)
        output_matrix[i] = _extract_outputs(weights_history)

        if (i + 1) % report_every == 0 or (i + 1) == total_runs:
            elapsed = time.time() - t_start
            pct = 100 * (i + 1) / total_runs
            rate = (i + 1) / elapsed
            eta = (total_runs - i - 1) / rate
            logging.info(
                "%d/%d (%.1f%%)  elapsed: %.1fs  ETA: %.1fs",
                i + 1,
                total_runs,
                pct,
                elapsed,
                eta,
            )

    total_time = time.time() - t_start
    logging.info("All %d runs complete in %.1fs.", total_runs, total_time)

    # compute Sobol indices for each output metric separately
    results: dict[str, SobolResult] = {}

    logging.info("Computing Sobol indices...")
    for col_idx, (metric_name, _, _) in enumerate(OUTPUT_METRICS):
        si = sobol.analyze(
            problem,
            output_matrix[:, col_idx],
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

    logging.info("Done.")
    _log_index_table(results)
    _plot_sobol_results(results, save_path=save_path)

    return results


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _log_index_table(results: dict[str, SobolResult]) -> None:
    """
    Log a formatted table of S1 and ST indices for every output metric.

    Args:
        results: Dictionary of SobolResult objects from run_sobol().

    """
    for metric_name, result in results.items():
        lines = [
            "─" * 62,
            f"  Output: {metric_name}",
            "─" * 62,
            f"  {'Parameter':<25}  {'S1':>6}  {'S1_conf':>7}  {'ST':>6}  {'ST_conf':>7}  {'ST-S1':>6}",
            f"  {'-' * 25}  {'-' * 6}  {'-' * 7}  {'-' * 6}  {'-' * 7}  {'-' * 6}",
        ]
        for i, name in enumerate(result.param_names):
            interaction = result.st[i] - result.s1[i]
            lines.append(f"  {name:<25}  {result.s1[i]:>6.3f}  {result.s1_conf[i]:>7.3f}  {result.st[i]:>6.3f}  {result.st_conf[i]:>7.3f}  {interaction:>6.3f}")
        logging.info("\n".join(lines))


def _plot_sobol_results(
    results: dict[str, SobolResult],
    save_path: str = "sobol_sensitivity_results.png",
) -> None:
    """
    Plot S1 and ST indices as grouped bar charts, one panel per output metric.

    Each panel shows all 9 parameters on the x-axis. Blue bars are S1 (direct
    effect), red bars are ST (total effect). The gap between them is the
    interaction contribution.

    Args:
        results: Dictionary of SobolResult objects from run_sobol().
        save_path: File path for the saved PNG figure.

    """
    n_outputs = len(results)
    n_cols = 3
    n_rows = -(-n_outputs // n_cols)  # ceiling division

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
        n_params = len(result.param_names)
        x = np.arange(n_params)
        width = 0.35

        ax.bar(
            x - width / 2,
            result.s1,
            width,
            yerr=result.s1_conf,
            capsize=3,
            label="S1 (direct)",
            color="#2196F3",
            alpha=0.85,
        )
        ax.bar(
            x + width / 2,
            result.st,
            width,
            yerr=result.st_conf,
            capsize=3,
            label="ST (total)",
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

    # hide unused axes if n_outputs is not a multiple of n_cols
    for ax_idx in range(len(results), len(ax_flat)):
        ax_flat[ax_idx].set_visible(False)

    fig.suptitle(
        "Sobol Sensitivity Analysis — Brock–Hommes Model\nBlue = direct effect (S1)   |   Red = total effect (ST)   |   Gap = interaction",
        fontsize=11,
        fontweight="bold",
    )
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    logging.info("Figure saved to: %s", save_path)
    plt.show()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # set up logging environment
    setup_logging()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    ticker = "^GSPC"
    start_date = "2016-01-01"
    end_date = "2025-12-31"

    logging.info("Loading %s data from %s to %s...", ticker, start_date, end_date)
    data = load_data_from_yfinance(ticker=ticker, start_date=start_date, end_date=end_date)
    returns = np.asarray(data.returns).reshape(-1)
    logging.info("Loaded %d daily returns.", len(returns))
    run_sobol(
        returns=returns,
        n_base=16384,
        save_path="sobol_sensitivity_results.png",
    )
