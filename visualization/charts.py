"""
Module 4: Visualizations
UHNW Portfolio Optimizer
------------------------
Produces four institutional-quality charts:
1. Efficient Frontier Comparison
2. Monte Carlo Fan Chart
3. Asset Allocation Comparison
4. Drawdown Comparison
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.market_data import (
    fetch_public_returns,
    compute_public_stats,
    build_alts_series,
    build_combined_universe,
)
from optimizer.efficient_frontier import (
    build_public_frontier,
    build_uhnw_frontier,
    find_key_portfolios,
)
from simulation.monte_carlo import (
    run_simulation,
    compute_max_drawdown,
    extract_optimal_weights as mc_extract,
    get_6040_weights,
    build_stress_cov,
)


# ─────────────────────────────────────────
# STYLE CONFIGURATION
# Institutional, clean, print-ready
# ─────────────────────────────────────────

COLORS = {
    "public":     "#2C3E50",   # dark navy
    "uhnw":       "#1A6B4A",   # deep green
    "stress":     "#C0392B",   # deep red
    "baseline":   "#7F8C8D",   # grey
    "accent":     "#D4AF37",   # gold — UHNW feel
    "background": "#FAFAFA",
    "grid":       "#E8E8E8",
}

plt.rcParams.update({
    "font.family":        "serif",
    "font.size":          11,
    "axes.titlesize":     13,
    "axes.titleweight":   "bold",
    "axes.labelsize":     11,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.grid":          True,
    "grid.color":         COLORS["grid"],
    "grid.linewidth":     0.6,
    "figure.facecolor":   COLORS["background"],
    "axes.facecolor":     COLORS["background"],
    "savefig.dpi":        150,
    "savefig.bbox":       "tight",
    "savefig.facecolor":  COLORS["background"],
})

os.makedirs("outputs/charts", exist_ok=True)


# ─────────────────────────────────────────
# CHART 1: EFFICIENT FRONTIER COMPARISON
# ─────────────────────────────────────────

def plot_efficient_frontier(
    public_frontier: pd.DataFrame,
    uhnw_frontier:   pd.DataFrame,
    public_keys:     dict,
    uhnw_keys:       dict,
):
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot frontiers
    ax.plot(
        public_frontier["Volatility"] * 100,
        public_frontier["Expected Return"] * 100,
        color=COLORS["public"], linewidth=2.5,
        label="Public Assets Only", zorder=3,
    )
    ax.plot(
        uhnw_frontier["Volatility"] * 100,
        uhnw_frontier["Expected Return"] * 100,
        color=COLORS["uhnw"], linewidth=2.5,
        label="UHNW + Alternatives", zorder=3,
    )

    # Fill between frontiers
    pub_interp = np.interp(
        uhnw_frontier["Volatility"],
        public_frontier["Volatility"].sort_values(),
        public_frontier.set_index("Volatility")
            .reindex(public_frontier["Volatility"].sort_values())["Expected Return"]
    )
    ax.fill_betweenx(
        uhnw_frontier["Expected Return"] * 100,
        uhnw_frontier["Volatility"] * 100,
        np.interp(
            uhnw_frontier["Expected Return"],
            public_frontier["Expected Return"],
            public_frontier["Volatility"],
        ) * 100,
        alpha=0.08, color=COLORS["uhnw"],
        label="Diversification Benefit",
    )

    # Key portfolio markers
    marker_styles = {
        "Max Sharpe":     ("★", 180, "Max Sharpe"),
        "Min Volatility": ("●", 100, "Min Volatility"),
        "Target (8%)":    ("◆", 100, "8% Target"),
    }

    for key_label, (marker, size, display) in marker_styles.items():
        if key_label in public_keys:
            p = public_keys[key_label]
            ax.scatter(
                p["Volatility"] * 100, p["Expected Return"] * 100,
                color=COLORS["public"], s=size, zorder=5,
                marker="*" if marker == "★" else
                "D" if marker == "◆" else "o",
            )
        if key_label in uhnw_keys:
            u = uhnw_keys[key_label]
            ax.scatter(
                u["Volatility"] * 100, u["Expected Return"] * 100,
                color=COLORS["uhnw"], s=size, zorder=5,
                marker="*" if marker == "★" else
                "D" if marker == "◆" else "o",
            )

    # Annotations for max sharpe points
    pub_ms  = public_keys["Max Sharpe"]
    uhnw_ms = uhnw_keys["Max Sharpe"]

    ax.annotate(
        f"Sharpe: {pub_ms['Sharpe']:.2f}",
        xy=(pub_ms["Volatility"]*100, pub_ms["Expected Return"]*100),
        xytext=(15, -20), textcoords="offset points",
        fontsize=9, color=COLORS["public"],
        arrowprops=dict(arrowstyle="-", color=COLORS["public"], lw=0.8),
    )
    ax.annotate(
        f"Sharpe: {uhnw_ms['Sharpe']:.2f}",
        xy=(uhnw_ms["Volatility"]*100, uhnw_ms["Expected Return"]*100),
        xytext=(15, 10), textcoords="offset points",
        fontsize=9, color=COLORS["uhnw"],
        arrowprops=dict(arrowstyle="-", color=COLORS["uhnw"], lw=0.8),
    )

    ax.set_xlabel("Portfolio Volatility (%)")
    ax.set_ylabel("Expected Annual Return (%)")
    ax.set_title(
        "Efficient Frontier: Public Assets vs. UHNW Multi-Asset Portfolio\n"
        "Alternatives access improves risk-adjusted returns at every point on the frontier",
        pad=15,
    )
    ax.legend(loc="lower right", framealpha=0.9, fontsize=10)

    # Footnote
    fig.text(
        0.12, 0.01,
        "Source: Historical data 2013–2024 (yfinance). "
        "Alternatives based on Cambridge Associates benchmarks. "
        "PE returns adjusted for mean reversion per institutional assumptions.",
        fontsize=7.5, color="#666666",
    )

    plt.tight_layout()
    plt.savefig("outputs/charts/chart1_efficient_frontier.png")
    plt.close()
    print("  Chart 1 saved: efficient frontier")


# ─────────────────────────────────────────
# CHART 2: MONTE CARLO FAN CHART
# ─────────────────────────────────────────

def plot_fan_chart(
    sim_6040:        pd.DataFrame,
    sim_uhnw:        pd.DataFrame,
    sim_stress:      pd.DataFrame,
    sim_6040_stress: pd.DataFrame,
):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    years = np.arange(sim_6040.shape[1])

    def draw_fan(ax, sim, color, title, show_stress=None):
        p10 = sim.quantile(0.10).values / 1e6
        p25 = sim.quantile(0.25).values / 1e6
        p50 = sim.quantile(0.50).values / 1e6
        p75 = sim.quantile(0.75).values / 1e6
        p90 = sim.quantile(0.90).values / 1e6

        ax.fill_between(years, p10, p90,
                        alpha=0.12, color=color, label="10th–90th %ile")
        ax.fill_between(years, p25, p75,
                        alpha=0.22, color=color, label="25th–75th %ile")
        ax.plot(years, p50, color=color,
                linewidth=2.5, label="Median", zorder=4)
        ax.plot(years, p10, color=color,
                linewidth=0.8, linestyle="--", alpha=0.6)
        ax.plot(years, p90, color=color,
                linewidth=0.8, linestyle="--", alpha=0.6)

        if show_stress is not None:
            s50 = show_stress.quantile(0.50).values / 1e6
            ax.plot(years, s50, color=COLORS["stress"],
                    linewidth=1.8, linestyle=":", label="Stress Median", zorder=4)

        # Reference lines
        ax.axhline(100, color="#999999", linewidth=0.8,
                   linestyle="--", alpha=0.5, label="Initial $100M")
        ax.axhline(500, color=COLORS["accent"], linewidth=0.8,
                   linestyle="--", alpha=0.6, label="$500M Target")

        ax.set_xlabel("Year")
        ax.set_ylabel("Portfolio Value ($M)")
        ax.set_title(title, pad=12)
        ax.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, _: f"${x:.0f}M")
        )
        ax.legend(fontsize=8.5, loc="upper left", framealpha=0.9)

    draw_fan(axes[0], sim_6040,
         COLORS["baseline"],
         "60/40 Baseline Portfolio",
         show_stress=sim_6040_stress)

    draw_fan(axes[1], sim_uhnw,
         COLORS["uhnw"],
         "UHNW + Alternatives Portfolio",
         show_stress=sim_stress)

    fig.suptitle(
        "Monte Carlo Wealth Simulation — $100M UHNW Portfolio, 25-Year Horizon\n"
        "10,000 simulations | 3% annual distributions | Annual rebalancing",
        fontsize=13, fontweight="bold", y=1.01,
    )

    fig.text(
        0.12, -0.02,
        "Shaded bands show 10th–90th and 25th–75th percentile wealth outcomes. "
        "Dotted red line shows stressed scenario median (elevated volatility, "
        "correlation spike, PE markdown). Not a guarantee of future performance.",
        fontsize=7.5, color="#666666",
    )

    plt.tight_layout()
    plt.savefig("outputs/charts/chart2_fan_chart.png", bbox_inches="tight")
    plt.close()
    print("  Chart 2 saved: Monte Carlo fan chart")


# ─────────────────────────────────────────
# CHART 3: ASSET ALLOCATION COMPARISON
# ─────────────────────────────────────────

def plot_asset_allocation(
    weights_dict: dict,
):
    """
    weights_dict: {"60/40": {asset: weight}, "UHNW": {asset: weight}}
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    tier_colors = {
        # Tier 1 — liquid
        "Cash":               "#A8D5BA",
        "US Agg Bonds":       "#76C893",
        "Long Duration UST":  "#52B788",
        "TIPS":               "#40916C",
        # Tier 2 — public equity
        "US Large Cap":       "#2C3E50",
        "US Small Cap":       "#34495E",
        "Intl Developed":     "#5D6D7E",
        "Emerging Markets":   "#85929E",
        "Real Estate (REIT)": "#AAB7B8",
        "Commodities":        "#BDC3C7",
        # Tier 3 — alternatives
        "Private Equity":     "#D4AF37",
        "Private Credit":     "#C49A00",
        "Hedge Funds":        "#A67C00",
        "Real Assets":        "#8B6914",
    }

    portfolio_labels = list(weights_dict.keys())

    for ax, label in zip(axes, portfolio_labels):
        weights = weights_dict[label]
        # Filter near-zero weights
        weights = {k: v for k, v in weights.items() if v > 0.005}
        assets  = list(weights.keys())
        vals    = [weights[a] * 100 for a in assets]
        colors  = [tier_colors.get(a, "#999999") for a in assets]

        bars = ax.barh(assets, vals, color=colors,
                       edgecolor="white", linewidth=0.5, height=0.7)

        # Value labels
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                f"{val:.1f}%", va="center", fontsize=9,
            )

        ax.set_xlabel("Allocation (%)")
        ax.set_title(f"{label}\nPortfolio Allocation", pad=12)
        ax.set_xlim(0, max(vals) * 1.25)
        ax.invert_yaxis()

    # Tier legend
    tier_patches = [
        mpatches.Patch(color="#52B788", label="Tier 1 — Liquid (0–2yr)"),
        mpatches.Patch(color="#2C3E50", label="Tier 2 — Public Markets"),
        mpatches.Patch(color="#D4AF37", label="Tier 3 — Illiquid Alternatives"),
    ]
    fig.legend(
        handles=tier_patches, loc="lower center",
        ncol=3, bbox_to_anchor=(0.5, -0.05),
        framealpha=0.9, fontsize=10,
    )

    fig.suptitle(
        "Portfolio Allocation: 60/40 Baseline vs. UHNW Multi-Asset\n"
        "Liquidity bucket framework reflects institutional UHNW practice",
        fontsize=13, fontweight="bold",
    )

    plt.tight_layout()
    plt.savefig("outputs/charts/chart3_allocation.png", bbox_inches="tight")
    plt.close()
    print("  Chart 3 saved: asset allocation")


# ─────────────────────────────────────────
# CHART 4: DRAWDOWN COMPARISON
# ─────────────────────────────────────────

def plot_drawdown_comparison(
    sim_6040:   pd.DataFrame,
    sim_uhnw:   pd.DataFrame,
    sim_stress: pd.DataFrame,
):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: Drawdown distribution
    ax = axes[0]

    def get_max_drawdowns(sim):
        dds = []
        for _, path in sim.iterrows():
            vals = path.values.astype(float)
            peak = np.maximum.accumulate(vals)
            dd   = (vals - peak) / peak
            dds.append(dd.min() * 100)
        return np.array(dds)

    dd_6040   = get_max_drawdowns(sim_6040)
    dd_uhnw   = get_max_drawdowns(sim_uhnw)
    dd_stress = get_max_drawdowns(sim_stress)

    bins = np.linspace(-60, 0, 40)
    ax.hist(dd_6040,   bins=bins, alpha=0.5,
            color=COLORS["baseline"], label="60/40 Baseline",    density=True)
    ax.hist(dd_uhnw,   bins=bins, alpha=0.5,
            color=COLORS["uhnw"],    label="UHNW + Alternatives", density=True)
    ax.hist(dd_stress, bins=bins, alpha=0.4,
            color=COLORS["stress"],  label="UHNW Stress Test",    density=True)

    ax.axvline(np.median(dd_6040),   color=COLORS["baseline"],
               linewidth=2, linestyle="--")
    ax.axvline(np.median(dd_uhnw),   color=COLORS["uhnw"],
               linewidth=2, linestyle="--")
    ax.axvline(np.median(dd_stress), color=COLORS["stress"],
               linewidth=2, linestyle="--")

    ax.set_xlabel("Maximum Drawdown (%)")
    ax.set_ylabel("Density")
    ax.set_title("Maximum Drawdown Distribution\nUHNW clients prioritize capital preservation", pad=12)
    ax.legend(fontsize=9)

    # Right: Median wealth path with drawdown shading
    ax2 = axes[1]
    years = np.arange(sim_6040.shape[1])

    med_6040   = sim_6040.median().values   / 1e6
    med_uhnw   = sim_uhnw.median().values   / 1e6
    med_stress = sim_stress.median().values / 1e6

    ax2.plot(years, med_6040,   color=COLORS["baseline"],
             linewidth=2, label="60/40 Median")
    ax2.plot(years, med_uhnw,   color=COLORS["uhnw"],
             linewidth=2, label="UHNW + Alts Median")
    ax2.plot(years, med_stress, color=COLORS["stress"],
             linewidth=2, linestyle=":", label="UHNW Stress Median")

    # Shade gap between UHNW and 60/40
    ax2.fill_between(years, med_6040, med_uhnw,
                     alpha=0.10, color=COLORS["uhnw"],
                     label="Wealth Advantage")

    ax2.axhline(100, color="#999999", linewidth=0.8,
                linestyle="--", alpha=0.5)
    ax2.set_xlabel("Year")
    ax2.set_ylabel("Median Portfolio Value ($M)")
    ax2.set_title("Median Wealth Path Comparison\nStress scenario shows resilience of alternatives sleeve", pad=12)
    ax2.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"${x:.0f}M")
    )
    ax2.legend(fontsize=9)

    fig.suptitle(
        "Capital Preservation Analysis — Drawdown & Wealth Path Comparison\n"
        "Alternatives reduce drawdown risk while improving long-run wealth outcomes",
        fontsize=13, fontweight="bold",
    )

    fig.text(
        0.12, -0.02,
        "Maximum drawdown computed across all 10,000 simulation paths. "
        "Stress scenario applies elevated volatility (1.5x), correlation spike (0.85), "
        "and PE markdown (−4%). Distributions include 3% annual withdrawal.",
        fontsize=7.5, color="#666666",
    )

    plt.tight_layout()
    plt.savefig("outputs/charts/chart4_drawdown.png", bbox_inches="tight")
    plt.close()
    print("  Chart 4 saved: drawdown comparison")


# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────

if __name__ == "__main__":
    print("Loading data and building frontiers...")
    monthly_returns                      = fetch_public_returns()
    public_ret, public_cov               = compute_public_stats(monthly_returns)
    alts_ret, alts_vol                   = build_alts_series()
    combined_ret, combined_cov, illiquid = build_combined_universe(
        public_ret, public_cov, alts_ret, alts_vol
    )

    public_frontier = build_public_frontier(public_ret, public_cov)
    uhnw_frontier   = build_uhnw_frontier(combined_ret, combined_cov)

    public_keys = find_key_portfolios(public_frontier, "Public Only")
    uhnw_keys   = find_key_portfolios(uhnw_frontier,   "UHNW + Alts")

    # Asset lists
    public_assets   = list(public_ret.index)
    combined_assets = list(combined_ret.index)

    # Weights
    w_6040  = get_6040_weights(combined_assets)
    w_uhnw  = mc_extract(uhnw_frontier, combined_assets, "max_sharpe")

    stress_cov, stress_ret = build_stress_cov(combined_cov, combined_ret)

    # Load or re-run simulations
    print("\nRunning simulations...")
    sim_6040        = run_simulation(w_6040,  combined_ret, combined_cov, "60/40 Baseline")
    sim_6040_stress = run_simulation(w_6040,  stress_ret,   stress_cov,   "60/40 Stress Test")
    sim_uhnw        = run_simulation(w_uhnw,  combined_ret, combined_cov, "UHNW + Alternatives")
    sim_stress      = run_simulation(w_uhnw,  stress_ret,   stress_cov,   "UHNW Stress Test")

    # Build weight dicts for allocation chart
    uhnw_weights_dict = {
        asset: w_uhnw[i] for i, asset in enumerate(combined_assets)
    }
    baseline_weights_dict = {
        asset: w_6040[i] for i, asset in enumerate(combined_assets)
    }

    weights_for_chart = {
        "60/40 Baseline":      baseline_weights_dict,
        "UHNW + Alternatives": uhnw_weights_dict,
    }

    # Generate all charts
    print("\nGenerating charts...")
    plot_efficient_frontier(public_frontier, uhnw_frontier,
                            public_keys, uhnw_keys)
    plot_fan_chart(sim_6040, sim_uhnw, sim_stress, sim_6040_stress)
    plot_asset_allocation(weights_for_chart)
    plot_drawdown_comparison(sim_6040, sim_uhnw, sim_stress)

    print("\nAll charts saved to outputs/charts/")
    print("Module 4 complete.")