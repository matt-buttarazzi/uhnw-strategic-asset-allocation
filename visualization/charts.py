'''
Author: Matthew Buttarazzi
Date: May 2026
Project: UHNW Strategic Asset Allocation with Alternatives
Description: Visualization module for the UHNW portfolio optimizer. Produces four institutional-quality
charts: an efficient frontier comparison showing the diversification benefit of alternatives, a Monte
Carlo fan chart showing 25-year wealth distributions for both portfolios under base and stress scenarios,
an asset allocation comparison showing the three-tier liquidity bucket framework, and a drawdown
analysis showing capital preservation outcomes. All charts are saved to outputs/charts/.
'''

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
    extract_optimal_weights,
    get_6040_weights,
)
from simulation.monte_carlo import (
    run_simulation,
    compute_max_drawdown,
    build_stress_cov,
)

# Color palette: navy for public, green for UHNW, gold for accent, grey for baseline
# The gold specifically signals the UHNW/wealth management context
COLORS = {
    "public":     "#2C3E50",
    "uhnw":       "#1A6B4A",
    "stress":     "#C0392B",
    "baseline":   "#7F8C8D",
    "accent":     "#D4AF37",
    "background": "#FAFAFA",
    "grid":       "#E8E8E8",
}

# Tier colors for the allocation chart, greens for liquid, navy for public equity, gold for alternatives
TIER_COLORS = {
    "Cash":               "#A8D5BA",
    "US Agg Bonds":       "#76C893",
    "Long Duration UST":  "#52B788",
    "TIPS":               "#40916C",
    "US Large Cap":       "#2C3E50",
    "US Small Cap":       "#34495E",
    "Intl Developed":     "#5D6D7E",
    "Emerging Markets":   "#85929E",
    "Real Estate (REIT)": "#AAB7B8",
    "Commodities":        "#BDC3C7",
    "Private Equity":     "#D4AF37",
    "Private Credit":     "#C49A00",
    "Hedge Funds":        "#A67C00",
    "Real Assets":        "#8B6914",
}

plt.rcParams.update({
    "font.family":       "serif",
    "font.size":         11,
    "axes.titlesize":    13,
    "axes.titleweight":  "bold",
    "axes.labelsize":    11,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.color":        COLORS["grid"],
    "grid.linewidth":    0.6,
    "figure.facecolor":  COLORS["background"],
    "axes.facecolor":    COLORS["background"],
    "savefig.dpi":       150,
    "savefig.bbox":      "tight",
    "savefig.facecolor": COLORS["background"],
})

os.makedirs("outputs/charts", exist_ok=True)


def plot_efficient_frontier(public_frontier, uhnw_frontier, public_keys, uhnw_keys):
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(
        public_frontier["Volatility"] * 100,
        public_frontier["Expected Return"] * 100,
        color=COLORS["public"], linewidth=2.5, label="Public Assets Only", zorder=3,
    )
    ax.plot(
        uhnw_frontier["Volatility"] * 100,
        uhnw_frontier["Expected Return"] * 100,
        color=COLORS["uhnw"], linewidth=2.5, label="UHNW + Alternatives", zorder=3,
    )

    # Shade the region between the two frontiers to show the diversification benefit visually
    ax.fill_betweenx(
        uhnw_frontier["Expected Return"] * 100,
        uhnw_frontier["Volatility"] * 100,
        np.interp(
            uhnw_frontier["Expected Return"],
            public_frontier["Expected Return"],
            public_frontier["Volatility"],
        ) * 100,
        alpha=0.08, color=COLORS["uhnw"], label="Diversification Benefit",
    )

    # Plot key portfolio markers on each frontier, star for max Sharpe, diamond for target, circle for min vol
    marker_map = {
        "Max Sharpe":    ("*", 180),
        "Min Volatility": ("o", 100),
        "Target (8%)":   ("D", 100),
    }
    for key_label, (marker, size) in marker_map.items():
        if key_label in public_keys:
            p = public_keys[key_label]
            ax.scatter(p["Volatility"] * 100, p["Expected Return"] * 100,
                       color=COLORS["public"], s=size, zorder=5, marker=marker)
        if key_label in uhnw_keys:
            u = uhnw_keys[key_label]
            ax.scatter(u["Volatility"] * 100, u["Expected Return"] * 100,
                       color=COLORS["uhnw"], s=size, zorder=5, marker=marker)

    pub_ms  = public_keys["Max Sharpe"]
    uhnw_ms = uhnw_keys["Max Sharpe"]

    ax.annotate(
        f"Sharpe: {pub_ms['Sharpe']:.2f}",
        xy=(pub_ms["Volatility"] * 100, pub_ms["Expected Return"] * 100),
        xytext=(15, -20), textcoords="offset points",
        fontsize=9, color=COLORS["public"],
        arrowprops=dict(arrowstyle="-", color=COLORS["public"], lw=0.8),
    )
    ax.annotate(
        f"Sharpe: {uhnw_ms['Sharpe']:.2f}",
        xy=(uhnw_ms["Volatility"] * 100, uhnw_ms["Expected Return"] * 100),
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
    fig.text(
        0.12, 0.01,
        "Source: Historical data 2013-2024 (yfinance). "
        "Alternatives based on Cambridge Associates benchmarks. "
        "PE returns adjusted for mean reversion per institutional assumptions.",
        fontsize=7.5, color="#666666",
    )

    plt.tight_layout()
    plt.savefig("outputs/charts/chart1_efficient_frontier.png")
    plt.close()
    print("  Chart 1 saved: efficient frontier")


def plot_fan_chart(sim_6040, sim_uhnw, sim_stress, sim_6040_stress):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    years     = np.arange(sim_6040.shape[1])

    # Inner function draws a single fan panel, used for both the 60/40 and UHNW panels
    # The shaded bands show the 10th-90th and 25th-75th percentile wealth distributions
    def draw_fan(ax, sim, color, title, show_stress=None):
        p10 = sim.quantile(0.10).values / 1e6
        p25 = sim.quantile(0.25).values / 1e6
        p50 = sim.quantile(0.50).values / 1e6
        p75 = sim.quantile(0.75).values / 1e6
        p90 = sim.quantile(0.90).values / 1e6

        ax.fill_between(years, p10, p90, alpha=0.12, color=color, label="10th-90th %ile")
        ax.fill_between(years, p25, p75, alpha=0.22, color=color, label="25th-75th %ile")
        ax.plot(years, p50, color=color, linewidth=2.5, label="Median", zorder=4)
        ax.plot(years, p10, color=color, linewidth=0.8, linestyle="--", alpha=0.6)
        ax.plot(years, p90, color=color, linewidth=0.8, linestyle="--", alpha=0.6)

        if show_stress is not None:
            s50 = show_stress.quantile(0.50).values / 1e6
            ax.plot(years, s50, color=COLORS["stress"],
                    linewidth=1.8, linestyle=":", label="Stress Median", zorder=4)

        ax.axhline(100, color="#999999", linewidth=0.8, linestyle="--", alpha=0.5, label="Initial $100M")
        ax.axhline(500, color=COLORS["accent"], linewidth=0.8, linestyle="--", alpha=0.6, label="$500M Target")
        ax.set_xlabel("Year")
        ax.set_ylabel("Portfolio Value ($M)")
        ax.set_title(title, pad=12)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:.0f}M"))
        ax.legend(fontsize=8.5, loc="upper left", framealpha=0.9)

    draw_fan(axes[0], sim_6040, COLORS["baseline"], "60/40 Baseline Portfolio",      show_stress=sim_6040_stress)
    draw_fan(axes[1], sim_uhnw, COLORS["uhnw"],     "UHNW + Alternatives Portfolio", show_stress=sim_stress)

    fig.suptitle(
        "Monte Carlo Wealth Simulation - $100M UHNW Portfolio, 25-Year Horizon\n"
        "10,000 simulations | 3% annual distributions | Annual rebalancing",
        fontsize=13, fontweight="bold", y=1.01,
    )
    fig.text(
        0.12, -0.02,
        "Shaded bands show 10th-90th and 25th-75th percentile wealth outcomes. "
        "Dotted red line shows stressed scenario median (elevated volatility, "
        "correlation spike, PE markdown). Not a guarantee of future performance.",
        fontsize=7.5, color="#666666",
    )

    plt.tight_layout()
    plt.savefig("outputs/charts/chart2_fan_chart.png", bbox_inches="tight")
    plt.close()
    print("  Chart 2 saved: Monte Carlo fan chart")


def plot_asset_allocation(weights_dict):
    fig, axes          = plt.subplots(1, 2, figsize=(14, 6))
    portfolio_labels   = list(weights_dict.keys())

    for ax, label in zip(axes, portfolio_labels):
        weights = {k: v for k, v in weights_dict[label].items() if v > 0.005}
        assets  = list(weights.keys())
        vals    = [weights[a] * 100 for a in assets]
        colors  = [TIER_COLORS.get(a, "#999999") for a in assets]

        bars = ax.barh(assets, vals, color=colors, edgecolor="white", linewidth=0.5, height=0.7)

        # Label each bar with its percentage value
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%", va="center", fontsize=9,
            )

        ax.set_xlabel("Allocation (%)")
        ax.set_title(f"{label}\nPortfolio Allocation", pad=12)
        ax.set_xlim(0, max(vals) * 1.25)
        ax.invert_yaxis()

    tier_patches = [
        mpatches.Patch(color="#52B788", label="Tier 1 - Liquid (0-2yr)"),
        mpatches.Patch(color="#2C3E50", label="Tier 2 - Public Markets"),
        mpatches.Patch(color="#D4AF37", label="Tier 3 - Illiquid Alternatives"),
    ]
    fig.legend(handles=tier_patches, loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, -0.05), framealpha=0.9, fontsize=10)
    fig.suptitle(
        "Portfolio Allocation: 60/40 Baseline vs. UHNW Multi-Asset\n"
        "Liquidity bucket framework reflects institutional UHNW practice",
        fontsize=13, fontweight="bold",
    )

    plt.tight_layout()
    plt.savefig("outputs/charts/chart3_allocation.png", bbox_inches="tight")
    plt.close()
    print("  Chart 3 saved: asset allocation")


def plot_drawdown_comparison(sim_6040, sim_uhnw, sim_stress):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Compute the maximum drawdown for every simulation path in a given sim DataFrame
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

    # Left panel, overlapping histograms show the full drawdown distribution for each portfolio
    ax  = axes[0]
    bins = np.linspace(-60, 0, 40)
    ax.hist(dd_6040,   bins=bins, alpha=0.5, color=COLORS["baseline"], label="60/40 Baseline",     density=True)
    ax.hist(dd_uhnw,   bins=bins, alpha=0.5, color=COLORS["uhnw"],     label="UHNW + Alternatives", density=True)
    ax.hist(dd_stress, bins=bins, alpha=0.4, color=COLORS["stress"],   label="UHNW Stress Test",    density=True)

    ax.axvline(np.median(dd_6040),   color=COLORS["baseline"], linewidth=2, linestyle="--")
    ax.axvline(np.median(dd_uhnw),   color=COLORS["uhnw"],     linewidth=2, linestyle="--")
    ax.axvline(np.median(dd_stress), color=COLORS["stress"],   linewidth=2, linestyle="--")

    ax.set_xlabel("Maximum Drawdown (%)")
    ax.set_ylabel("Density")
    ax.set_title("Maximum Drawdown Distribution\nUHNW clients prioritize capital preservation", pad=12)
    ax.legend(fontsize=9)

    # Right panel, median wealth paths with the wealth advantage region shaded green
    ax2   = axes[1]
    years = np.arange(sim_6040.shape[1])

    med_6040   = sim_6040.median().values   / 1e6
    med_uhnw   = sim_uhnw.median().values   / 1e6
    med_stress = sim_stress.median().values / 1e6

    ax2.plot(years, med_6040,   color=COLORS["baseline"], linewidth=2, label="60/40 Median")
    ax2.plot(years, med_uhnw,   color=COLORS["uhnw"],     linewidth=2, label="UHNW + Alts Median")
    ax2.plot(years, med_stress, color=COLORS["stress"],   linewidth=2, linestyle=":", label="UHNW Stress Median")
    ax2.fill_between(years, med_6040, med_uhnw, alpha=0.10, color=COLORS["uhnw"], label="Wealth Advantage")
    ax2.axhline(100, color="#999999", linewidth=0.8, linestyle="--", alpha=0.5)

    ax2.set_xlabel("Year")
    ax2.set_ylabel("Median Portfolio Value ($M)")
    ax2.set_title("Median Wealth Path Comparison\nStress scenario shows resilience of alternatives sleeve", pad=12)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:.0f}M"))
    ax2.legend(fontsize=9)

    fig.suptitle(
        "Capital Preservation Analysis - Drawdown & Wealth Path Comparison\n"
        "Alternatives reduce drawdown risk while improving long-run wealth outcomes",
        fontsize=13, fontweight="bold",
    )
    fig.text(
        0.12, -0.02,
        "Maximum drawdown computed across all 10,000 simulation paths. "
        "Stress scenario applies elevated volatility (1.5x), correlation spike (0.85), "
        "and PE markdown (-4%). Distributions include 3% annual withdrawal.",
        fontsize=7.5, color="#666666",
    )

    plt.tight_layout()
    plt.savefig("outputs/charts/chart4_drawdown.png", bbox_inches="tight")
    plt.close()
    print("  Chart 4 saved: drawdown comparison")


def main():
    print("Loading data and building frontiers...")
    monthly_returns              = fetch_public_returns()
    public_ret, public_cov       = compute_public_stats(monthly_returns)
    alts_ret, alts_vol           = build_alts_series()
    combined_ret, combined_cov, illiquid = build_combined_universe(
        public_ret, public_cov, alts_ret, alts_vol
    )

    public_frontier = build_public_frontier(public_ret, public_cov)
    uhnw_frontier   = build_uhnw_frontier(combined_ret, combined_cov)

    public_keys = find_key_portfolios(public_frontier, "Public Only")
    uhnw_keys   = find_key_portfolios(uhnw_frontier,   "UHNW + Alts")

    combined_assets = list(combined_ret.index)

    w_6040 = get_6040_weights(combined_assets)
    w_uhnw = extract_optimal_weights(uhnw_frontier, combined_assets, "max_sharpe")

    stress_cov, stress_ret = build_stress_cov(combined_cov, combined_ret)

    print("\nRunning simulations...")
    sim_6040        = run_simulation(w_6040, combined_ret, combined_cov, "60/40 Baseline")
    sim_6040_stress = run_simulation(w_6040, stress_ret,   stress_cov,   "60/40 Stress Test")
    sim_uhnw        = run_simulation(w_uhnw, combined_ret, combined_cov, "UHNW + Alternatives")
    sim_stress      = run_simulation(w_uhnw, stress_ret,   stress_cov,   "UHNW Stress Test")

    weights_for_chart = {
        "60/40 Baseline":      {asset: w_6040[i] for i, asset in enumerate(combined_assets)},
        "UHNW + Alternatives": {asset: w_uhnw[i] for i, asset in enumerate(combined_assets)},
    }

    print("\nGenerating charts...")
    plot_efficient_frontier(public_frontier, uhnw_frontier, public_keys, uhnw_keys)
    plot_fan_chart(sim_6040, sim_uhnw, sim_stress, sim_6040_stress)
    plot_asset_allocation(weights_for_chart)
    plot_drawdown_comparison(sim_6040, sim_uhnw, sim_stress)

    print("\nAll charts saved to outputs/charts/")
    print("Module 4 complete.")

if __name__ == "__main__":
    main()