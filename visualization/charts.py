'''
Author: Matthew R. Buttarazzi
Date: May 2026
Project: UHNW Strategic Asset Allocation with Alternatives
Description: Visualization module for the UHNW portfolio optimizer. All four plot functions
return matplotlib Figure objects so they can be used both by the Streamlit dashboard (via
st.pyplot) and by the standalone script which saves them to disk. Produces four charts:
efficient frontier comparison, Monte Carlo fan chart, asset allocation comparison, and
drawdown analysis.
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
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

COLORS = {
    "public":     "#2C3E50",
    "uhnw":       "#1A6B4A",
    "stress":     "#C0392B",
    "baseline":   "#7F8C8D",
    "accent":     "#D4AF37",
    "background": "#FFFFFF",
    "grid":       "#F0F0EC",
}

TIER_COLORS = {
    "Cash":               "#A8D5BA",
    "US Agg Bonds":       "#76C893",
    "Long Duration UST":  "#52B788",
    "TIPS":               "#40916C",
    "US Large Cap":       "#1A3A5C",
    "US Small Cap":       "#2C4E72",
    "Intl Developed":     "#3D6480",
    "Emerging Markets":   "#5D7D94",
    "Real Estate (REIT)": "#8AA5B8",
    "Commodities":        "#B0C4D0",
    "Private Equity":     "#D4AF37",
    "Private Credit":     "#C49A00",
    "Hedge Funds":        "#A67C00",
    "Real Assets":        "#8B6914",
}

plt.rcParams.update({
    "font.family":       "serif",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.color":        COLORS["grid"],
    "grid.linewidth":    0.5,
    "figure.facecolor":  COLORS["background"],
    "axes.facecolor":    COLORS["background"],
    "axes.labelsize":    11,
    "axes.titlesize":    12,
    "xtick.labelsize":   10,
    "ytick.labelsize":   10,
})


# Returns a Figure so the caller decides whether to save it or pass it to st.pyplot
def plot_efficient_frontier(public_frontier, uhnw_frontier, public_keys, uhnw_keys, start_year=2013, end_year=2024):
    fig, ax = plt.subplots(figsize=(11, 6))

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
    ax.fill_betweenx(
        uhnw_frontier["Expected Return"] * 100,
        uhnw_frontier["Volatility"] * 100,
        np.interp(
            uhnw_frontier["Expected Return"],
            public_frontier["Expected Return"],
            public_frontier["Volatility"],
        ) * 100,
        alpha=0.10, color=COLORS["uhnw"], label="Diversification Benefit",
    )

    marker_map = {
        "Max Sharpe":    ("*", 200),
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
        f"Sharpe: {pub_ms['Sharpe']:.2f}\n{pub_ms['Expected Return']*100:.1f}% ret / {pub_ms['Volatility']*100:.1f}% vol",
        xy=(pub_ms["Volatility"]*100, pub_ms["Expected Return"]*100),
        xytext=(20, -30), textcoords="offset points", fontsize=9, color=COLORS["public"],
        arrowprops=dict(arrowstyle="-", color=COLORS["public"], lw=0.8),
    )
    ax.annotate(
        f"Sharpe: {uhnw_ms['Sharpe']:.2f}\n{uhnw_ms['Expected Return']*100:.1f}% ret / {uhnw_ms['Volatility']*100:.1f}% vol",
        xy=(uhnw_ms["Volatility"]*100, uhnw_ms["Expected Return"]*100),
        xytext=(20, 15), textcoords="offset points", fontsize=9, color=COLORS["uhnw"],
        arrowprops=dict(arrowstyle="-", color=COLORS["uhnw"], lw=0.8),
    )

    ax.set_xlabel("Portfolio Volatility (%)")
    ax.set_ylabel("Expected Annual Return (%)")
    ax.set_title(
        f"Efficient Frontier: Public Assets vs. UHNW Multi-Asset Portfolio  |  {start_year}–{end_year}",
        pad=14, fontweight="bold",
    )
    ax.legend(loc="lower right", framealpha=0.95, fontsize=10, frameon=True)
    fig.text(
        0.12, 0.01,
        f"Source: Historical data {start_year}–{end_year} (yfinance). "
        "Alternatives based on Cambridge Associates benchmarks. "
        "PE returns adjusted for mean reversion per practitioner assumptions.",
        fontsize=8, color="#999999",
    )
    plt.tight_layout()
    return fig


def plot_fan_chart(sim_6040, sim_uhnw, sim_6040_stress, sim_stress,
                   initial_wealth=100_000_000, withdrawal_rate=0.03,
                   horizon=25, n_sims=10_000, start_year=2013, end_year=2024):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    years     = np.arange(sim_6040.shape[1])

    def draw_fan(ax, sim, color, title, show_stress=None):
        p10 = sim.quantile(0.10).values / 1e6
        p25 = sim.quantile(0.25).values / 1e6
        p50 = sim.quantile(0.50).values / 1e6
        p75 = sim.quantile(0.75).values / 1e6
        p90 = sim.quantile(0.90).values / 1e6

        ax.fill_between(years, p10, p90, alpha=0.12, color=color, label="10th–90th %ile")
        ax.fill_between(years, p25, p75, alpha=0.22, color=color, label="25th–75th %ile")
        ax.plot(years, p50, color=color, linewidth=2.5, label="Median", zorder=4)
        ax.plot(years, p10, color=color, linewidth=0.8, linestyle="--", alpha=0.5)
        ax.plot(years, p90, color=color, linewidth=0.8, linestyle="--", alpha=0.5)

        if show_stress is not None:
            s50 = show_stress.quantile(0.50).values / 1e6
            ax.plot(years, s50, color=COLORS["stress"],
                    linewidth=1.8, linestyle=":", label="Stress Median", zorder=4)

        ax.axhline(initial_wealth / 1e6, color="#BBBBBB", linewidth=1.0,
                   linestyle="--", alpha=0.7, label=f"Initial ${initial_wealth/1e6:.0f}M")
        ax.axhline(500, color=COLORS["accent"], linewidth=1.0,
                   linestyle="--", alpha=0.7, label="$500M Target")
        ax.set_xlabel("Year")
        ax.set_ylabel("Portfolio Value ($M)")
        ax.set_title(title, pad=12, fontweight="bold")
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:.0f}M"))
        ax.legend(fontsize=9, loc="upper left", framealpha=0.95, frameon=True)

    draw_fan(axes[0], sim_6040, COLORS["baseline"], "60/40 Baseline Portfolio",      show_stress=sim_6040_stress)
    draw_fan(axes[1], sim_uhnw, COLORS["uhnw"],     "UHNW + Alternatives Portfolio", show_stress=sim_stress)

    fig.suptitle(
        f"Monte Carlo Wealth Simulation  |  ${initial_wealth/1e6:.0f}M Portfolio, {horizon}-Year Horizon  |  "
        f"{n_sims:,} Paths  |  {withdrawal_rate:.1%} Annual Distributions",
        fontsize=12, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    return fig


def plot_asset_allocation(weights_dict):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, (label, weights) in zip(axes, weights_dict.items()):
        filtered   = {k: v for k, v in weights.items() if v > 0.005}
        asset_list = list(filtered.keys())
        vals       = [filtered[a] * 100 for a in asset_list]
        colors     = [TIER_COLORS.get(a, "#999999") for a in asset_list]

        bars = ax.barh(asset_list, vals, color=colors, edgecolor="white", linewidth=0.8, height=0.65)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_width() + 0.4, bar.get_y() + bar.get_height() / 2,
                    f"{val:.1f}%", va="center", fontsize=10, color="#444444")

        ax.set_xlabel("Allocation (%)")
        ax.set_title(label, pad=12, fontweight="bold")
        ax.set_xlim(0, max(vals) * 1.28)
        ax.invert_yaxis()

    tier_patches = [
        mpatches.Patch(color="#52B788", label="Tier 1 — Liquid (0–2yr)"),
        mpatches.Patch(color="#1A3A5C", label="Tier 2 — Public Markets"),
        mpatches.Patch(color="#D4AF37", label="Tier 3 — Illiquid Alternatives"),
    ]
    fig.legend(handles=tier_patches, loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, -0.04), framealpha=0.95, fontsize=10)
    fig.suptitle("Portfolio Allocation: 60/40 Baseline vs. UHNW Multi-Asset",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    return fig


def plot_drawdown_comparison(sim_6040, sim_uhnw, sim_stress, initial_wealth=100_000_000):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    def get_max_drawdowns(sim):
        dds = []
        for _, path in sim.iterrows():
            vals = path.values.astype(float)
            peak = np.maximum.accumulate(vals)
            dd   = (vals - peak) / peak
            dds.append(dd.min() * 100)
        return np.array(dds)

    dd_6040 = get_max_drawdowns(sim_6040)
    dd_uhnw = get_max_drawdowns(sim_uhnw)
    dd_stress = get_max_drawdowns(sim_stress)

    bins = np.linspace(-60, 0, 40)
    axes[0].hist(dd_6040,   bins=bins, alpha=0.55, color=COLORS["baseline"],
                 label=f"60/40  (median {np.median(dd_6040):.1f}%)",    density=True)
    axes[0].hist(dd_uhnw,   bins=bins, alpha=0.55, color=COLORS["uhnw"],
                 label=f"UHNW + Alts  (median {np.median(dd_uhnw):.1f}%)", density=True)
    axes[0].hist(dd_stress, bins=bins, alpha=0.40, color=COLORS["stress"],
                 label=f"UHNW Stress  (median {np.median(dd_stress):.1f}%)", density=True)

    axes[0].axvline(np.median(dd_6040),   color=COLORS["baseline"], linewidth=2, linestyle="--")
    axes[0].axvline(np.median(dd_uhnw),   color=COLORS["uhnw"],     linewidth=2, linestyle="--")
    axes[0].axvline(np.median(dd_stress), color=COLORS["stress"],   linewidth=2, linestyle="--")
    axes[0].set_xlabel("Maximum Drawdown (%)")
    axes[0].set_ylabel("Density")
    axes[0].set_title("Maximum Drawdown Distribution", pad=12, fontweight="bold")
    axes[0].legend(fontsize=9, frameon=True, framealpha=0.95)

    years      = np.arange(sim_6040.shape[1])
    med_6040   = sim_6040.median().values   / 1e6
    med_uhnw   = sim_uhnw.median().values   / 1e6
    med_stress = sim_stress.median().values / 1e6

    axes[1].plot(years, med_6040,   color=COLORS["baseline"], linewidth=2, label="60/40 Median")
    axes[1].plot(years, med_uhnw,   color=COLORS["uhnw"],     linewidth=2, label="UHNW + Alts Median")
    axes[1].plot(years, med_stress, color=COLORS["stress"],   linewidth=2, linestyle=":", label="UHNW Stress Median")
    axes[1].fill_between(years, med_6040, med_uhnw, alpha=0.12, color=COLORS["uhnw"], label="Wealth Advantage")
    axes[1].axhline(initial_wealth / 1e6, color="#BBBBBB", linewidth=1.0, linestyle="--", alpha=0.7)
    axes[1].set_xlabel("Year")
    axes[1].set_ylabel("Median Portfolio Value ($M)")
    axes[1].set_title("Median Wealth Path Comparison", pad=12, fontweight="bold")
    axes[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:.0f}M"))
    axes[1].legend(fontsize=9, frameon=True, framealpha=0.95)

    fig.suptitle(
        "Capital Preservation Analysis — Drawdown and Wealth Path Comparison",
        fontsize=13, fontweight="bold",
    )
    fig.text(
        0.12, -0.02,
        "Maximum drawdown computed across all simulation paths. "
        "Stress scenario applies elevated volatility (1.5x), correlation spike (0.85), "
        "and PE markdown (-4%). Distributions include annual withdrawal rate.",
        fontsize=7.5, color="#999999",
    )
    plt.tight_layout()
    return fig


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
    os.makedirs("outputs/charts", exist_ok=True)

    fig1 = plot_efficient_frontier(public_frontier, uhnw_frontier, public_keys, uhnw_keys)
    fig1.savefig("outputs/charts/chart1_efficient_frontier.png", dpi=150, bbox_inches="tight")
    plt.close(fig1)
    print("  Chart 1 saved")

    fig2 = plot_fan_chart(sim_6040, sim_uhnw, sim_6040_stress, sim_stress)
    fig2.savefig("outputs/charts/chart2_fan_chart.png", dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print("  Chart 2 saved")

    fig3 = plot_asset_allocation(weights_for_chart)
    fig3.savefig("outputs/charts/chart3_allocation.png", dpi=150, bbox_inches="tight")
    plt.close(fig3)
    print("  Chart 3 saved")

    fig4 = plot_drawdown_comparison(sim_6040, sim_uhnw, sim_stress)
    fig4.savefig("outputs/charts/chart4_drawdown.png", dpi=150, bbox_inches="tight")
    plt.close(fig4)
    print("  Chart 4 saved")

    print("\nAll charts saved to outputs/charts/")
    print("Module 4 complete.")

if __name__ == "__main__":
    main()