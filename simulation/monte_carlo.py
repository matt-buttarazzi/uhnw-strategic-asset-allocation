"""
Module 3: Monte Carlo Simulation
UHNW Portfolio Optimizer
------------------------
Simulates long-term wealth outcomes for a $100M UHNW portfolio.
Compares:
1. Traditional 60/40 baseline
2. Optimized public-only portfolio
3. UHNW portfolio with alternatives
4. Stress test / crisis regime scenario
"""

import numpy as np
import pandas as pd
import sys
import os
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


# ─────────────────────────────────────────
# 3.1 SIMULATION PARAMETERS
# ─────────────────────────────────────────

INITIAL_WEALTH      = 100_000_000   # $100M starting portfolio
TIME_HORIZON        = 25            # years — generational wealth horizon
N_SIMULATIONS       = 10_000        # Monte Carlo paths
ANNUAL_WITHDRAWAL   = 0.03          # 3% annual distribution rate
REBALANCE_ANNUAL    = True          # rebalance back to target weights

# Wealth outcome thresholds
WEALTH_TARGET       = 500_000_000   # $500M — meaningful growth
WEALTH_FLOOR        = 50_000_000    # $50M — ruin threshold (50% drawdown)

# Stress test parameters (crisis regime)
STRESS_EQUITY_VOL_MULT      = 1.5   # equity vol spikes 50%
STRESS_EQUITY_CORR          = 0.85  # correlations spike
STRESS_PE_RETURN_REDUCTION  = 0.04  # PE marks down 4% in crisis
STRESS_BOND_EQUITY_CORR     = 0.20  # bonds lose diversification benefit


# ─────────────────────────────────────────
# 3.2 PORTFOLIO WEIGHT DEFINITIONS
# ─────────────────────────────────────────

def get_6040_weights(asset_names: list) -> np.ndarray:
    """Traditional 60/40 portfolio — public assets only."""
    weights = np.zeros(len(asset_names))
    allocations = {
        "US Large Cap":       0.40,
        "Intl Developed":     0.10,
        "Emerging Markets":   0.05,
        "US Small Cap":       0.05,
        "US Agg Bonds":       0.30,
        "Cash":               0.05,
        "Long Duration UST":  0.05,
    }
    for asset, w in allocations.items():
        if asset in asset_names:
            weights[asset_names.index(asset)] = w
    return weights


def extract_optimal_weights(
    frontier: pd.DataFrame,
    asset_names: list,
    portfolio_type: str = "max_sharpe",
) -> np.ndarray:
    """Extracts weight array from frontier key portfolio."""
    if portfolio_type == "max_sharpe":
        row = frontier.loc[frontier["Sharpe"].idxmax()]
    elif portfolio_type == "min_vol":
        row = frontier.loc[frontier["Volatility"].idxmin()]
    else:
        row = frontier.loc[
            (frontier["Expected Return"] - 0.08).abs().idxmin()
        ]

    weights = np.array([
        row["Weights"].get(asset, 0.0) for asset in asset_names
    ])
    return weights / weights.sum()


# ─────────────────────────────────────────
# 3.3 STRESS TEST COVARIANCE MATRIX
# ─────────────────────────────────────────

def build_stress_cov(
    combined_cov: pd.DataFrame,
    combined_ret: pd.Series,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Constructs stressed covariance matrix for crisis regime.
    - Equity volatilities spike by 1.5x
    - Equity correlations converge to 0.85
    - PE expected return reduced by 4%
    - Bond-equity correlation increases
    """
    assets      = list(combined_cov.index)
    stress_cov  = combined_cov.copy().values.astype(float)
    stress_ret  = combined_ret.copy()

    equity_assets = [
        "US Large Cap", "US Small Cap", "Intl Developed",
        "Emerging Markets", "Real Estate (REIT)", "Private Equity"
    ]
    bond_assets = ["US Agg Bonds", "Long Duration UST", "TIPS"]

    eq_idx   = [assets.index(a) for a in equity_assets if a in assets]
    bond_idx = [assets.index(a) for a in bond_assets   if a in assets]

    # Spike equity volatilities
    for i in eq_idx:
        stress_cov[i, i] *= STRESS_EQUITY_VOL_MULT ** 2

    # Spike equity-equity correlations
    for i in eq_idx:
        for j in eq_idx:
            if i != j:
                vol_i = np.sqrt(stress_cov[i, i])
                vol_j = np.sqrt(stress_cov[j, j])
                stress_cov[i, j] = STRESS_EQUITY_CORR * vol_i * vol_j
                stress_cov[j, i] = stress_cov[i, j]

    # Increase bond-equity correlation
    for i in eq_idx:
        for j in bond_idx:
            vol_i = np.sqrt(stress_cov[i, i])
            vol_j = np.sqrt(stress_cov[j, j])
            stress_cov[i, j] = STRESS_BOND_EQUITY_CORR * vol_i * vol_j
            stress_cov[j, i] = stress_cov[i, j]

    # Reduce PE return
    if "Private Equity" in stress_ret.index:
        stress_ret["Private Equity"] -= STRESS_PE_RETURN_REDUCTION

    # Ensure positive semi-definite
    eigvals = np.linalg.eigvals(stress_cov)
    if np.any(eigvals < 0):
        stress_cov += np.eye(len(assets)) * abs(eigvals.min()) * 1.01

    return pd.DataFrame(stress_cov, index=assets, columns=assets), stress_ret


# ─────────────────────────────────────────
# 3.4 CORE SIMULATION ENGINE
# ─────────────────────────────────────────

def run_simulation(
    weights: np.ndarray,
    returns: pd.Series,
    cov: pd.DataFrame,
    label: str,
    initial_wealth: float = INITIAL_WEALTH,
    n_sims: int = N_SIMULATIONS,
    horizon: int = TIME_HORIZON,
    withdrawal_rate: float = ANNUAL_WITHDRAWAL,
) -> pd.DataFrame:
    """
    Runs Monte Carlo simulation for a given portfolio.
    Uses correlated multivariate normal annual returns.
    Returns DataFrame of wealth paths [n_sims x horizon+1].
    """
    print(f"  Running {n_sims:,} simulations for: {label}...")

    asset_names = list(returns.index)
    mu          = returns.values
    sigma       = cov.values

    # Align weights to return vector
    w = np.array([
        weights[i] if i < len(weights) else 0.0
        for i in range(len(asset_names))
    ])
    w = w / w.sum()

    # Portfolio-level parameters
    port_return = float(np.dot(w, mu))
    port_vol    = float(np.sqrt(w @ sigma @ w))

    # Simulate correlated annual returns
    np.random.seed(42)
    annual_returns = np.random.multivariate_normal(
        mean=mu,
        cov=sigma,
        size=(n_sims, horizon)
    )  # shape: (n_sims, horizon, n_assets)

    # Portfolio returns per year per simulation
    port_annual_rets = annual_returns @ w  # shape: (n_sims, horizon)

    # Wealth paths
    wealth = np.zeros((n_sims, horizon + 1))
    wealth[:, 0] = initial_wealth

    for t in range(horizon):
        # Growth
        wealth[:, t+1] = wealth[:, t] * (1 + port_annual_rets[:, t])
        # Annual withdrawal
        withdrawal = wealth[:, t+1] * withdrawal_rate
        wealth[:, t+1] -= withdrawal
        # Floor at zero
        wealth[:, t+1] = np.maximum(wealth[:, t+1], 0)

    wealth_df = pd.DataFrame(
        wealth,
        columns=[f"Year {i}" for i in range(horizon + 1)]
    )
    wealth_df.name = label
    return wealth_df


# ─────────────────────────────────────────
# 3.5 OUTCOME METRICS
# ─────────────────────────────────────────

def compute_outcomes(
    wealth_df: pd.DataFrame,
    label: str,
) -> dict:
    """
    Computes key wealth outcome statistics at end of horizon.
    """
    terminal = wealth_df.iloc[:, -1]

    outcomes = {
        "Label":                label,
        "Median Wealth":        terminal.median(),
        "10th Percentile":      terminal.quantile(0.10),
        "25th Percentile":      terminal.quantile(0.25),
        "75th Percentile":      terminal.quantile(0.75),
        "90th Percentile":      terminal.quantile(0.90),
        "Prob > $500M":         (terminal > WEALTH_TARGET).mean(),
        "Prob < $50M (Ruin)":   (terminal < WEALTH_FLOOR).mean(),
        "Mean Wealth":          terminal.mean(),
        "Std Deviation":        terminal.std(),
    }

    print(f"\n  === {label} — Year {TIME_HORIZON} Outcomes ===")
    print(f"  Median Wealth:      ${outcomes['Median Wealth']/1e6:.1f}M")
    print(f"  10th Percentile:    ${outcomes['10th Percentile']/1e6:.1f}M")
    print(f"  90th Percentile:    ${outcomes['90th Percentile']/1e6:.1f}M")
    print(f"  Prob > $500M:       {outcomes['Prob > $500M']:.1%}")
    print(f"  Prob < $50M (Ruin): {outcomes['Prob < $50M (Ruin)']:.1%}")

    return outcomes


# ─────────────────────────────────────────
# 3.6 DRAWDOWN ANALYSIS
# ─────────────────────────────────────────

def compute_max_drawdown(wealth_paths: pd.DataFrame) -> dict:
    """
    Computes median maximum drawdown across all simulation paths.
    UHNW clients care deeply about capital preservation.
    """
    drawdowns = []
    for _, path in wealth_paths.iterrows():
        path_vals   = path.values
        peak        = np.maximum.accumulate(path_vals)
        dd          = (path_vals - peak) / peak
        drawdowns.append(dd.min())

    return {
        "Median Max Drawdown": np.median(drawdowns),
        "Worst 10% Drawdown":  np.percentile(drawdowns, 10),
    }


# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────

if __name__ == "__main__":
    # Load data
    print("Loading market data...")
    monthly_returns                      = fetch_public_returns()
    public_ret, public_cov               = compute_public_stats(monthly_returns)
    alts_ret, alts_vol                   = build_alts_series()
    combined_ret, combined_cov, illiquid = build_combined_universe(
        public_ret, public_cov, alts_ret, alts_vol
    )

    # Build frontiers
    print("\nBuilding frontiers...")
    public_frontier = build_public_frontier(public_ret, public_cov)
    uhnw_frontier   = build_uhnw_frontier(combined_ret, combined_cov)

    # Asset name lists
    public_assets   = list(public_ret.index)
    combined_assets = list(combined_ret.index)

    # Portfolio weights
    w_6040        = get_6040_weights(combined_assets)
    w_public_opt  = extract_optimal_weights(
        public_frontier, public_assets, "max_sharpe"
    )
    # Pad public weights to combined asset length
    w_public_padded = np.zeros(len(combined_assets))
    for i, asset in enumerate(public_assets):
        if asset in combined_assets:
            w_public_padded[combined_assets.index(asset)] = w_public_opt[i]

    w_uhnw = extract_optimal_weights(
        uhnw_frontier, combined_assets, "max_sharpe"
    )

    # Build stress covariance
    stress_cov, stress_ret = build_stress_cov(combined_cov, combined_ret)

    # ── Run simulations ──
    print("\nRunning Monte Carlo simulations...")
    sim_6040       = run_simulation(w_6040,          combined_ret, combined_cov, "60/40 Baseline")
    sim_public_opt = run_simulation(w_public_padded, combined_ret, combined_cov, "Public Optimized")
    sim_uhnw       = run_simulation(w_uhnw,          combined_ret, combined_cov, "UHNW + Alternatives")
    sim_stress     = run_simulation(w_uhnw,          stress_ret,   stress_cov,   "UHNW Stress Test")

    # ── Outcome metrics ──
    print("\nComputing outcomes...")
    outcomes = [
        compute_outcomes(sim_6040,       "60/40 Baseline"),
        compute_outcomes(sim_public_opt, "Public Optimized"),
        compute_outcomes(sim_uhnw,       "UHNW + Alternatives"),
        compute_outcomes(sim_stress,     "UHNW Stress Test"),
    ]

    # ── Drawdown analysis ──
    print("\nDrawdown Analysis:")
    for sim, label in [
        (sim_6040,   "60/40 Baseline"),
        (sim_uhnw,   "UHNW + Alternatives"),
        (sim_stress, "UHNW Stress Test"),
    ]:
        dd = compute_max_drawdown(sim)
        print(f"  {label}:")
        print(f"    Median Max Drawdown: {dd['Median Max Drawdown']:.1%}")
        print(f"    Worst 10% Drawdown:  {dd['Worst 10% Drawdown']:.1%}")

    # ── Summary table ──
    print("\n=== SUMMARY TABLE ===")
    summary = pd.DataFrame(outcomes).set_index("Label")
    summary["Median Wealth"]   = summary["Median Wealth"].apply(
        lambda x: f"${x/1e6:.1f}M"
    )
    summary["10th Percentile"] = summary["10th Percentile"].apply(
        lambda x: f"${x/1e6:.1f}M"
    )
    summary["90th Percentile"] = summary["90th Percentile"].apply(
        lambda x: f"${x/1e6:.1f}M"
    )
    summary["Prob > $500M"]       = summary["Prob > $500M"].apply(
        lambda x: f"{x:.1%}"
    )
    summary["Prob < $50M (Ruin)"] = summary["Prob < $50M (Ruin)"].apply(
        lambda x: f"{x:.1%}"
    )
    print(summary[[
        "Median Wealth", "10th Percentile",
        "90th Percentile", "Prob > $500M", "Prob < $50M (Ruin)"
    ]].to_string())

    print("\nModule 3 complete.")

    # Save simulation results for visualization
    os.makedirs("outputs", exist_ok=True)
    sim_6040.to_csv("outputs/sim_6040.csv", index=False)
    sim_public_opt.to_csv("outputs/sim_public_opt.csv", index=False)
    sim_uhnw.to_csv("outputs/sim_uhnw.csv", index=False)
    sim_stress.to_csv("outputs/sim_stress.csv", index=False)
    print("Simulation results saved to outputs/")
