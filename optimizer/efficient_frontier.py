'''
Author: Matthew R. Buttarazzi
Date: May 2026
Project: UHNW Strategic Asset Allocation with Alternatives
Description: Portfolio optimizer for the UHNW asset allocation framework. Builds mean-variance
efficient frontiers for two universes: public assets only (baseline) and the full UHNW universe
with alternatives. All constraint parameters and the risk-free rate are accepted as function
arguments so the dashboard can pass in user-selected values without touching module constants.
'''

import numpy as np
import pandas as pd
from scipy.optimize import minimize
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.market_data import (
    fetch_public_returns,
    compute_public_stats,
    build_alts_series,
    build_combined_universe,
    ALTS_ASSUMPTIONS,
    PE_LONG_RUN_MEAN,
    PE_MEAN_REVERSION,
)

# Tier definitions used by both the optimizer and the dashboard
TIER1_ASSETS = ["Cash", "Long Duration UST", "TIPS", "US Agg Bonds"]
TIER2_ASSETS = ["US Large Cap", "US Small Cap", "Intl Developed",
                "Emerging Markets", "Real Estate (REIT)", "Commodities", "Hedge Funds"]
TIER3_ASSETS = ["Private Equity", "Private Credit", "Real Assets"]

# Default constraint values — used by standalone script, overridden by dashboard
DEFAULT_TIER1_MIN = 0.08
DEFAULT_TIER3_MIN = 0.15
DEFAULT_TIER3_MAX = 0.40
DEFAULT_PE_MAX    = 0.20
DEFAULT_PC_MAX    = 0.15
DEFAULT_HF_MAX    = 0.15
DEFAULT_RA_MAX    = 0.10
DEFAULT_RF        = 0.04


# PE returns tend to revert toward their long-run average after periods of outperformance. This pulls the raw 
# Cambridge Associates figure of 15.5% down toward the 13% long-run mean at a 30% reversion speed, giving 
# a more defensible forward estimate.
def adjust_pe_return(current_pe_return, pe_long_run=PE_LONG_RUN_MEAN, pe_reversion=PE_MEAN_REVERSION):
    adjusted = current_pe_return - pe_reversion * (current_pe_return - pe_long_run)
    print(f"  PE Return: raw={current_pe_return:.1%} -> mean-reverted={adjusted:.1%} "
          f"(pull toward {pe_long_run:.1%} long-run mean)")
    return adjusted


def portfolio_return(weights, returns):
    return float(np.dot(weights, returns))


def portfolio_volatility(weights, cov):
    return float(np.sqrt(weights @ cov @ weights))


def sharpe_ratio(weights, returns, cov, rf=DEFAULT_RF):
    ret = portfolio_return(weights, returns)
    vol = portfolio_volatility(weights, cov)
    if vol > 0:
        return (ret - rf) / vol
    return 0.0


# Minimizes portfolio volatility for a given target return using SLSQP.
# Returns the optimal weight array, or None if the optimizer couldn't find a feasible solution.
def optimize_portfolio(target_return, returns, cov, constraints, bounds, n_assets):
    constraints_full = constraints + [{
        "type": "eq",
        "fun": lambda w, r=target_return: portfolio_return(w, returns) - r
    }]

    result = minimize(
        fun=lambda w: portfolio_volatility(w, cov),
        x0=np.ones(n_assets) / n_assets,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints_full,
        options={"ftol": 1e-12, "maxiter": 1000},
    )

    if result.success:
        return result.x
    return None


def build_public_frontier(public_returns, public_cov, n_points=50, rf=DEFAULT_RF):
    assets  = list(public_returns.index)
    n       = len(assets)
    ret_arr = public_returns.values
    cov_arr = public_cov.values

    # Cap any single public asset at 30%
    bounds      = [(0.0, 0.30)] * n
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]

    # Sweep target returns from just above min to just below max achievable return
    ret_min        = ret_arr.min() + 0.001
    ret_max        = ret_arr.max() - 0.001
    target_returns = np.linspace(ret_min, ret_max, n_points)

    frontier = []
    for tr in target_returns:
        w = optimize_portfolio(tr, ret_arr, cov_arr, constraints, bounds, n)
        if w is not None:
            frontier.append({
                "Expected Return": portfolio_return(w, ret_arr),
                "Volatility":      portfolio_volatility(w, cov_arr),
                "Sharpe":          sharpe_ratio(w, ret_arr, cov_arr, rf),
                "Weights":         dict(zip(assets, w)),
            })

    df = pd.DataFrame(frontier).sort_values("Volatility").reset_index(drop=True)

    # Drop the inefficient lower portion — only keep points where return increases with volatility
    clean = [df.iloc[0]]
    for i in range(1, len(df)):
        if df.iloc[i]["Expected Return"] > clean[-1]["Expected Return"]:
            clean.append(df.iloc[i])

    result_df = pd.DataFrame(clean).reset_index(drop=True)
    print(f"  Public frontier: {len(result_df)} feasible points")
    return result_df


def build_uhnw_frontier(
    combined_returns, combined_cov, n_points=50,
    rf=DEFAULT_RF,
    tier3_min=DEFAULT_TIER3_MIN, tier3_max=DEFAULT_TIER3_MAX,
    pe_max=DEFAULT_PE_MAX, pc_max=DEFAULT_PC_MAX,
    hf_max=DEFAULT_HF_MAX, ra_max=DEFAULT_RA_MAX,
    pe_long_run=PE_LONG_RUN_MEAN, pe_reversion=PE_MEAN_REVERSION,
):
    assets = list(combined_returns.index)
    n      = len(assets)

    # Apply PE mean reversion before optimizing
    adj_returns = combined_returns.copy()
    if "Private Equity" in adj_returns.index:
        adj_returns["Private Equity"] = adjust_pe_return(
            adj_returns["Private Equity"], pe_long_run, pe_reversion
        )

    ret_arr = adj_returns.values
    cov_arr = combined_cov.values

    # Build an index map so the constraint lambdas can look up asset positions by name
    idx = {asset: i for i, asset in enumerate(assets)}

    bounds = []
    for asset in assets:
        if asset == "Private Equity":
            bounds.append((0.0, pe_max))
        elif asset == "Private Credit":
            bounds.append((0.0, pc_max))
        elif asset == "Hedge Funds":
            bounds.append((0.0, hf_max))
        elif asset == "Real Assets":
            bounds.append((0.0, ra_max))
        else:
            bounds.append((0.0, 0.30))

    constraints = [
        # Weights must sum to 1
        {"type": "eq", "fun": lambda w: np.sum(w) - 1},

        # Tier 1 floor — keep enough liquid assets to fund near-term needs
        {"type": "ineq", "fun": lambda w: sum(w[idx[a]] for a in TIER1_ASSETS if a in idx) - DEFAULT_TIER1_MIN},

        # Tier 3 floor — always hold some alternatives to capture illiquidity premium
        {"type": "ineq", "fun": lambda w: sum(w[idx[a]] for a in TIER3_ASSETS if a in idx) - tier3_min},

        # Tier 3 ceiling — cap illiquid exposure to stay within the liquidity budget
        {"type": "ineq", "fun": lambda w: tier3_max - sum(w[idx[a]] for a in TIER3_ASSETS if a in idx)},
    ]

    ret_min        = ret_arr.min() + 0.001
    ret_max        = ret_arr.max() - 0.001
    target_returns = np.linspace(ret_min, ret_max, n_points)

    frontier = []
    for tr in target_returns:
        w = optimize_portfolio(tr, ret_arr, cov_arr, constraints, bounds, n)
        if w is not None:
            weights_dict = dict(zip(assets, w))
            tier3_alloc  = sum(weights_dict.get(a, 0) for a in TIER3_ASSETS)
            frontier.append({
                "Expected Return":  portfolio_return(w, ret_arr),
                "Volatility":       portfolio_volatility(w, cov_arr),
                "Sharpe":           sharpe_ratio(w, ret_arr, cov_arr, rf),
                "Tier3 Allocation": tier3_alloc,
                "Weights":          weights_dict,
            })

    df = pd.DataFrame(frontier).sort_values("Volatility").reset_index(drop=True)

    # Drop the inefficient lower portion — only keep points where return increases with volatility
    clean = [df.iloc[0]]
    for i in range(1, len(df)):
        if df.iloc[i]["Expected Return"] > clean[-1]["Expected Return"]:
            clean.append(df.iloc[i])

    result_df = pd.DataFrame(clean).reset_index(drop=True)
    print(f"  UHNW frontier: {len(result_df)} feasible points")
    return result_df


# Identifies the max Sharpe, min volatility, and target return portfolios on a given frontier.
# The 8% target return is a typical UHNW real return objective for long-horizon portfolios.
def find_key_portfolios(frontier, label, target_return=0.08):
    max_sharpe_idx = frontier["Sharpe"].idxmax()
    min_vol_idx    = frontier["Volatility"].idxmin()
    target_idx     = (frontier["Expected Return"] - target_return).abs().idxmin()

    key = {
        "Max Sharpe":                    frontier.loc[max_sharpe_idx],
        "Min Volatility":                frontier.loc[min_vol_idx],
        f"Target ({target_return:.0%})": frontier.loc[target_idx],
    }

    print(f"\n  === {label} Key Portfolios ===")
    for name, port in key.items():
        print(f"  {name}: Return={port['Expected Return']:.1%}, "
              f"Vol={port['Volatility']:.1%}, "
              f"Sharpe={port['Sharpe']:.2f}")

    return key


def extract_optimal_weights(frontier, asset_names, portfolio_type="max_sharpe"):
    if portfolio_type == "max_sharpe":
        row = frontier.loc[frontier["Sharpe"].idxmax()]
    elif portfolio_type == "min_vol":
        row = frontier.loc[frontier["Volatility"].idxmin()]
    else:
        row = frontier.loc[(frontier["Expected Return"] - 0.08).abs().idxmin()]

    weights = np.array([row["Weights"].get(asset, 0.0) for asset in asset_names])
    return weights / weights.sum()


def get_6040_weights(asset_names):
    weights = np.zeros(len(asset_names))
    allocations = {
        "US Large Cap":      0.40,
        "Intl Developed":    0.10,
        "Emerging Markets":  0.05,
        "US Small Cap":      0.05,
        "US Agg Bonds":      0.30,
        "Cash":              0.05,
        "Long Duration UST": 0.05,
    }
    for asset, w in allocations.items():
        if asset in asset_names:
            weights[asset_names.index(asset)] = w
    return weights


def main():
    print("Loading market data...")
    monthly_returns              = fetch_public_returns()
    public_ret, public_cov       = compute_public_stats(monthly_returns)
    alts_ret, alts_vol           = build_alts_series()
    combined_ret, combined_cov, illiquid = build_combined_universe(
        public_ret, public_cov, alts_ret, alts_vol
    )

    print("\nBuilding public-only frontier...")
    public_frontier = build_public_frontier(public_ret, public_cov)

    print("\nBuilding UHNW frontier with alternatives...")
    uhnw_frontier = build_uhnw_frontier(combined_ret, combined_cov)

    print("\nIdentifying key portfolios...")
    public_keys = find_key_portfolios(public_frontier, "Public Only")
    uhnw_keys   = find_key_portfolios(uhnw_frontier,   "UHNW + Alts")

    print("\nModule 2 complete.")
    print(f"Public frontier points: {len(public_frontier)}")
    print(f"UHNW frontier points:   {len(uhnw_frontier)}")

if __name__ == "__main__":
    main()