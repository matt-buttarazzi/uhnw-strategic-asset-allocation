'''
Author: Matthew Buttarazzi
Date: May 2026
Project: UHNW Strategic Asset Allocation with Alternatives
Description: Monte Carlo simulation engine for the UHNW portfolio optimizer. Runs 10,000 correlated
annual return paths over a 25-year horizon for four portfolios: 60/40 baseline, optimized public-only,
UHNW with alternatives, and a stress test regime. The stress test applies elevated equity volatility,
correlation spikes, and a PE markdown to simulate a severe market dislocation. Outputs wealth
distribution metrics and drawdown analysis for each portfolio.
'''

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
    extract_optimal_weights,
    get_6040_weights,
)

INITIAL_WEALTH    = 100_000_000  # $100M starting portfolio
TIME_HORIZON      = 25           # 25 year generational wealth horizon
N_SIMULATIONS     = 10_000       # number of Monte Carlo paths
ANNUAL_WITHDRAWAL = 0.03         # 3% annual distribution rate reflects realistic UHNW spending needs

# Wealth thresholds for outcome probability calculations
WEALTH_TARGET = 500_000_000  # $500M: meaningful long-run growth
WEALTH_FLOOR  = 50_000_000   # $50M: ruin threshold, 50% real drawdown from starting value

# Stress test parameters, designed to simulate a severe dislocation similar to 2008
# Equity vol spikes, correlations converge, bonds lose their diversification benefit, PE marks down
STRESS_EQUITY_VOL_MULT     = 1.5   # scale equity volatility up by 50%
STRESS_EQUITY_CORR         = 0.85  # equity-equity correlations converge toward 1
STRESS_PE_RETURN_REDUCTION = 0.04  # PE marks down 4% due to lagged appraisal effect
STRESS_BOND_EQUITY_CORR    = 0.20  # bonds lose most of their negative equity correlation


# Constructs a stressed covariance matrix and return vector for the crisis regime.
# Modify the base covariance matrix in place rather than rebuilding from scratch,
# equity variances scale up, cross-correlations spike, and PE return gets a markdown.
# After modifying we check for positive semi-definiteness since the manual edits can
# introduce small negative eigenvalues from floating point arithmetic.
def build_stress_cov(combined_cov, combined_ret):
    assets     = list(combined_cov.index)
    stress_cov = combined_cov.copy().values.astype(float)
    stress_ret = combined_ret.copy()

    equity_assets = [
        "US Large Cap", "US Small Cap", "Intl Developed",
        "Emerging Markets", "Real Estate (REIT)", "Private Equity"
    ]
    bond_assets = ["US Agg Bonds", "Long Duration UST", "TIPS"]

    eq_idx   = [assets.index(a) for a in equity_assets if a in assets]
    bond_idx = [assets.index(a) for a in bond_assets if a in assets]

    # Scale up equity variances first so the correlation spike uses the stressed vols
    for i in eq_idx:
        stress_cov[i, i] *= STRESS_EQUITY_VOL_MULT ** 2

    # Drive equity-equity correlations up toward STRESS_EQUITY_CORR
    for i in eq_idx:
        for j in eq_idx:
            if i != j:
                vol_i = np.sqrt(stress_cov[i, i])
                vol_j = np.sqrt(stress_cov[j, j])
                stress_cov[i, j] = STRESS_EQUITY_CORR * vol_i * vol_j
                stress_cov[j, i] = stress_cov[i, j]

    # Increase bond-equity correlation, bonds become less useful as a hedge during stress
    for i in eq_idx:
        for j in bond_idx:
            vol_i = np.sqrt(stress_cov[i, i])
            vol_j = np.sqrt(stress_cov[j, j])
            stress_cov[i, j] = STRESS_BOND_EQUITY_CORR * vol_i * vol_j
            stress_cov[j, i] = stress_cov[i, j]

    # Apply PE markdown, lagged appraisal-based reporting means PE marks down after public markets
    if "Private Equity" in stress_ret.index:
        stress_ret["Private Equity"] -= STRESS_PE_RETURN_REDUCTION

    # Fix any negative eigenvalues introduced by the manual correlation edits
    eigvals = np.linalg.eigvals(stress_cov)
    if np.any(eigvals < 0):
        stress_cov += np.eye(len(assets)) * abs(eigvals.min()) * 1.01

    return pd.DataFrame(stress_cov, index=assets, columns=assets), stress_ret

# Runs n_sims correlated annual return paths over the full horizon and compounds wealth
# with annual withdrawals and rebalancing. Uses multivariate normal draws from the full
# asset covariance matrix so correlations between assets are preserved each year.
def run_simulation(weights, returns, cov, label, initial_wealth=INITIAL_WEALTH,
                   n_sims=N_SIMULATIONS, horizon=TIME_HORIZON, withdrawal_rate=ANNUAL_WITHDRAWAL):
    print(f"  Running {n_sims:,} simulations for: {label}...")

    asset_names = list(returns.index)
    mu          = returns.values
    sigma       = cov.values

    # Align and normalize weights in case the input vector has a different length
    w = np.array([weights[i] if i < len(weights) else 0.0 for i in range(len(asset_names))])
    w = w / w.sum()

    # Draw correlated annual returns for all assets across all sims and years at once
    # shape is (n_sims, horizon, n_assets), then dot with weights to get portfolio returns
    np.random.seed(42)
    annual_returns   = np.random.multivariate_normal(mean=mu, cov=sigma, size=(n_sims, horizon))
    port_annual_rets = annual_returns @ w

    wealth         = np.zeros((n_sims, horizon + 1))
    wealth[:, 0]   = initial_wealth

    for t in range(horizon):
        # Grow, then withdraw, order matters since withdrawal is taken after growth
        wealth[:, t + 1] = wealth[:, t] * (1 + port_annual_rets[:, t])
        wealth[:, t + 1] -= wealth[:, t + 1] * withdrawal_rate
        wealth[:, t + 1]  = np.maximum(wealth[:, t + 1], 0)

    wealth_df = pd.DataFrame(wealth, columns=[f"Year {i}" for i in range(horizon + 1)])
    wealth_df.name = label
    return wealth_df

def compute_outcomes(wealth_df, label):
    terminal = wealth_df.iloc[:, -1]

    outcomes = {
        "Label":              label,
        "Median Wealth":      terminal.median(),
        "10th Percentile":    terminal.quantile(0.10),
        "25th Percentile":    terminal.quantile(0.25),
        "75th Percentile":    terminal.quantile(0.75),
        "90th Percentile":    terminal.quantile(0.90),
        "Prob > $500M":       (terminal > WEALTH_TARGET).mean(),
        "Prob < $50M (Ruin)": (terminal < WEALTH_FLOOR).mean(),
        "Mean Wealth":        terminal.mean(),
        "Std Deviation":      terminal.std(),
    }

    print(f"\n  === {label} — Year {TIME_HORIZON} Outcomes ===")
    print(f"  Median Wealth:      ${outcomes['Median Wealth']/1e6:.1f}M")
    print(f"  10th Percentile:    ${outcomes['10th Percentile']/1e6:.1f}M")
    print(f"  90th Percentile:    ${outcomes['90th Percentile']/1e6:.1f}M")
    print(f"  Prob > $500M:       {outcomes['Prob > $500M']:.1%}")
    print(f"  Prob < $50M (Ruin): {outcomes['Prob < $50M (Ruin)']:.1%}")

    return outcomes


# Tracks the running peak across each simulation path and measures how far each path
# drops from its peak. UHNW clients care more about avoiding catastrophic drawdowns
# than maximizing Sharpe ratios, so this is a key output alongside the wealth percentiles.
def compute_max_drawdown(wealth_paths):
    drawdowns = []
    for _, path in wealth_paths.iterrows():
        path_vals = path.values
        peak      = np.maximum.accumulate(path_vals)
        dd        = (path_vals - peak) / peak
        drawdowns.append(dd.min())

    return {
        "Median Max Drawdown": np.median(drawdowns),
        "Worst 10% Drawdown":  np.percentile(drawdowns, 10),
    }

def main():
    print("Loading market data...")
    monthly_returns              = fetch_public_returns()
    public_ret, public_cov       = compute_public_stats(monthly_returns)
    alts_ret, alts_vol           = build_alts_series()
    combined_ret, combined_cov, illiquid = build_combined_universe(
        public_ret, public_cov, alts_ret, alts_vol
    )

    print("\nBuilding frontiers...")
    public_frontier = build_public_frontier(public_ret, public_cov)
    uhnw_frontier   = build_uhnw_frontier(combined_ret, combined_cov)

    public_assets   = list(public_ret.index)
    combined_assets = list(combined_ret.index)

    w_6040       = get_6040_weights(combined_assets)
    w_public_opt = extract_optimal_weights(public_frontier, public_assets, "max_sharpe")

    # Public optimizer only covers public assets so we pad the weight vector with zeros
    # for the alternatives slots before passing it to the combined universe simulation
    w_public_padded = np.zeros(len(combined_assets))
    for i, asset in enumerate(public_assets):
        if asset in combined_assets:
            w_public_padded[combined_assets.index(asset)] = w_public_opt[i]

    w_uhnw = extract_optimal_weights(uhnw_frontier, combined_assets, "max_sharpe")

    stress_cov, stress_ret = build_stress_cov(combined_cov, combined_ret)

    print("\nRunning Monte Carlo simulations...")
    sim_6040        = run_simulation(w_6040,          combined_ret, combined_cov, "60/40 Baseline")
    sim_6040_stress = run_simulation(w_6040,          stress_ret,   stress_cov,   "60/40 Stress Test")
    sim_public_opt  = run_simulation(w_public_padded, combined_ret, combined_cov, "Public Optimized")
    sim_uhnw        = run_simulation(w_uhnw,          combined_ret, combined_cov, "UHNW + Alternatives")
    sim_stress      = run_simulation(w_uhnw,          stress_ret,   stress_cov,   "UHNW Stress Test")

    print("\nComputing outcomes...")
    outcomes = [
        compute_outcomes(sim_6040,        "60/40 Baseline"),
        compute_outcomes(sim_public_opt,  "Public Optimized"),
        compute_outcomes(sim_uhnw,        "UHNW + Alternatives"),
        compute_outcomes(sim_stress,      "UHNW Stress Test"),
    ]

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

    print("\n=== SUMMARY TABLE ===")
    summary = pd.DataFrame(outcomes).set_index("Label")
    summary["Median Wealth"]      = summary["Median Wealth"].apply(lambda x: f"${x/1e6:.1f}M")
    summary["10th Percentile"]    = summary["10th Percentile"].apply(lambda x: f"${x/1e6:.1f}M")
    summary["90th Percentile"]    = summary["90th Percentile"].apply(lambda x: f"${x/1e6:.1f}M")
    summary["Prob > $500M"]       = summary["Prob > $500M"].apply(lambda x: f"{x:.1%}")
    summary["Prob < $50M (Ruin)"] = summary["Prob < $50M (Ruin)"].apply(lambda x: f"{x:.1%}")
    print(summary[[
        "Median Wealth", "10th Percentile",
        "90th Percentile", "Prob > $500M", "Prob < $50M (Ruin)"
    ]].to_string())

    print("\nModule 3 complete.")

    os.makedirs("outputs", exist_ok=True)
    sim_6040.to_csv("outputs/sim_6040.csv", index=False)
    sim_6040_stress.to_csv("outputs/sim_6040_stress.csv", index=False)
    sim_public_opt.to_csv("outputs/sim_public_opt.csv", index=False)
    sim_uhnw.to_csv("outputs/sim_uhnw.csv", index=False)
    sim_stress.to_csv("outputs/sim_stress.csv", index=False)
    print("Simulation results saved to outputs/")

if __name__ == "__main__":
    main()