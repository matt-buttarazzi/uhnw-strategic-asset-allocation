"""
Module 2: Portfolio Optimizer
UHNW Portfolio Optimizer
------------------------
Builds mean-variance efficient frontiers for:
1. Public assets only (baseline)
2. Full UHNW universe with alternatives + illiquidity constraints
3. Liquidity bucket framework
4. PE mean reversion adjustment
"""

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


# ─────────────────────────────────────────
# 2.1 PE MEAN REVERSION ADJUSTMENT
# ─────────────────────────────────────────

def adjust_pe_return(current_pe_return: float) -> float:
    """
    Applies mean reversion dampening to PE forward return assumption.
    Per Roth/Barth (Capital Allocators): PE returns tend to revert
    toward long-run mean after periods of outperformance.

    Formula: adjusted = current - reversion_speed * (current - long_run_mean)
    """
    adjusted = current_pe_return - PE_MEAN_REVERSION * (
        current_pe_return - PE_LONG_RUN_MEAN
    )
    print(f"  PE Return: raw={current_pe_return:.1%} → "
          f"mean-reverted={adjusted:.1%} "
          f"(pull toward {PE_LONG_RUN_MEAN:.1%} long-run mean)")
    return adjusted


# ─────────────────────────────────────────
# 2.2 LIQUIDITY BUCKET FRAMEWORK
# ─────────────────────────────────────────

# Tier 1: 0-2yr spending needs — highly liquid
TIER1_ASSETS = ["Cash", "Long Duration UST", "TIPS", "US Agg Bonds"]
TIER1_MIN    = 0.08   # minimum 8% in Tier 1 at all times

# Tier 2: Intermediate liquidity — public markets
TIER2_ASSETS = ["US Large Cap", "US Small Cap", "Intl Developed",
                "Emerging Markets", "Real Estate (REIT)", "Commodities",
                "Hedge Funds"]

# Tier 3: Illiquid growth capital — alternatives
TIER3_ASSETS = ["Private Equity", "Private Credit", "Real Assets"]
TIER3_MAX    = 0.40   # maximum 40% illiquid at UHNW level
TIER3_MIN    = 0.15   # minimum 15% to capture illiquidity premium

# Individual alternatives caps
PE_MAX      = 0.20
PC_MAX      = 0.15
HF_MAX      = 0.15
RA_MAX      = 0.10


# ─────────────────────────────────────────
# 2.3 CORE OPTIMIZER
# ─────────────────────────────────────────

def portfolio_return(weights: np.ndarray, returns: np.ndarray) -> float:
    return float(np.dot(weights, returns))


def portfolio_volatility(weights: np.ndarray, cov: np.ndarray) -> float:
    return float(np.sqrt(weights @ cov @ weights))


def sharpe_ratio(weights: np.ndarray, returns: np.ndarray,
                 cov: np.ndarray, rf: float = 0.04) -> float:
    ret = portfolio_return(weights, returns)
    vol = portfolio_volatility(weights, cov)
    return (ret - rf) / vol if vol > 0 else 0.0


def optimize_portfolio(
    target_return: float,
    returns: np.ndarray,
    cov: np.ndarray,
    constraints: list,
    bounds: list,
    n_assets: int,
) -> np.ndarray | None:
    """
    Minimizes portfolio variance for a given target return.
    Returns optimal weights or None if infeasible.
    """
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

    return result.x if result.success else None


# ─────────────────────────────────────────
# 2.4 PUBLIC-ONLY FRONTIER
# ─────────────────────────────────────────

def build_public_frontier(
    public_returns: pd.Series,
    public_cov: pd.DataFrame,
    n_points: int = 50,
) -> pd.DataFrame:
    """
    Builds efficient frontier using public assets only.
    Baseline 60/40-style universe — no alternatives.
    """
    assets  = list(public_returns.index)
    n       = len(assets)
    ret_arr = public_returns.values
    cov_arr = public_cov.values

    bounds = [(0.0, 0.30)] * n   # max 30% single asset

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]

    # Return range: min to max achievable
    ret_min = ret_arr.min() + 0.001
    ret_max = ret_arr.max() - 0.001
    target_returns = np.linspace(ret_min, ret_max, n_points)

    frontier = []
    for tr in target_returns:
        w = optimize_portfolio(tr, ret_arr, cov_arr, constraints, bounds, n)
        if w is not None:
            frontier.append({
                "Expected Return": portfolio_return(w, ret_arr),
                "Volatility":      portfolio_volatility(w, cov_arr),
                "Sharpe":          sharpe_ratio(w, ret_arr, cov_arr),
                "Weights":         dict(zip(assets, w)),
            })

    df = pd.DataFrame(frontier)
    print(f"  Public frontier: {len(df)} feasible points")
    # Remove inefficient lower portion — keep only upward-sloping frontier
    df = pd.DataFrame(frontier)
    df = df.sort_values("Volatility").reset_index(drop=True)

    # Keep only points where return increases with volatility
    clean = [df.iloc[0]]
    for i in range(1, len(df)):
        if df.iloc[i]["Expected Return"] > clean[-1]["Expected Return"]:
            clean.append(df.iloc[i])

    return pd.DataFrame(clean).reset_index(drop=True)


# ─────────────────────────────────────────
# 2.5 UHNW FRONTIER WITH ALTERNATIVES
# ─────────────────────────────────────────

def build_uhnw_frontier(
    combined_returns: pd.Series,
    combined_cov: pd.DataFrame,
    n_points: int = 50,
) -> pd.DataFrame:
    """
    Builds efficient frontier for full UHNW universe.
    Incorporates:
    - Liquidity bucket constraints (Tier 1/2/3)
    - Individual alternatives caps
    - PE mean reversion on return assumption
    """
    assets  = list(combined_returns.index)
    n       = len(assets)

    # Apply PE mean reversion
    adj_returns = combined_returns.copy()
    if "Private Equity" in adj_returns.index:
        adj_returns["Private Equity"] = adjust_pe_return(
            adj_returns["Private Equity"]
        )

    ret_arr = adj_returns.values
    cov_arr = combined_cov.values

    # Asset index lookup
    idx = {asset: i for i, asset in enumerate(assets)}

    # Bounds — per asset max
    bounds = []
    for asset in assets:
        if asset == "Private Equity":
            bounds.append((0.0, PE_MAX))
        elif asset == "Private Credit":
            bounds.append((0.0, PC_MAX))
        elif asset == "Hedge Funds":
            bounds.append((0.0, HF_MAX))
        elif asset == "Real Assets":
            bounds.append((0.0, RA_MAX))
        else:
            bounds.append((0.0, 0.30))

    # Constraints
    constraints = [
        # Weights sum to 1
        {"type": "eq",  "fun": lambda w: np.sum(w) - 1},

        # Tier 1 minimum — liquidity for distributions + capital calls
        {"type": "ineq", "fun": lambda w: sum(
            w[idx[a]] for a in TIER1_ASSETS if a in idx
        ) - TIER1_MIN},

        # Tier 3 minimum — capture illiquidity premium
        {"type": "ineq", "fun": lambda w: sum(
            w[idx[a]] for a in TIER3_ASSETS if a in idx
        ) - TIER3_MIN},

        # Tier 3 maximum — illiquidity budget
        {"type": "ineq", "fun": lambda w: TIER3_MAX - sum(
            w[idx[a]] for a in TIER3_ASSETS if a in idx
        )},
    ]

    ret_min = ret_arr.min() + 0.001
    ret_max = ret_arr.max() - 0.001
    target_returns = np.linspace(ret_min, ret_max, n_points)

    frontier = []
    for tr in target_returns:
        w = optimize_portfolio(tr, ret_arr, cov_arr, constraints, bounds, n)
        if w is not None:
            weights_dict = dict(zip(assets, w))
            tier3_alloc  = sum(
                weights_dict.get(a, 0) for a in TIER3_ASSETS
            )
            frontier.append({
                "Expected Return": portfolio_return(w, ret_arr),
                "Volatility":      portfolio_volatility(w, cov_arr),
                "Sharpe":          sharpe_ratio(w, ret_arr, cov_arr),
                "Tier3 Allocation": tier3_alloc,
                "Weights":         weights_dict,
            })

    df = pd.DataFrame(frontier)
    print(f"  UHNW frontier: {len(df)} feasible points")

    # Remove inefficient lower portion — keep only upward-sloping frontier
    df = pd.DataFrame(frontier)
    df = df.sort_values("Volatility").reset_index(drop=True)

    # Keep only points where return increases with volatility
    clean = [df.iloc[0]]
    for i in range(1, len(df)):
        if df.iloc[i]["Expected Return"] > clean[-1]["Expected Return"]:
            clean.append(df.iloc[i])

    return pd.DataFrame(clean).reset_index(drop=True)


# ─────────────────────────────────────────
# 2.6 KEY PORTFOLIO IDENTIFICATION
# ─────────────────────────────────────────

def find_key_portfolios(
    frontier: pd.DataFrame,
    label: str,
    target_return: float = 0.08,
) -> dict:
    """
    Identifies three key portfolios on a given frontier:
    1. Maximum Sharpe ratio
    2. Minimum volatility
    3. Target return (default 8% — typical UHNW real return target)
    """
    max_sharpe_idx = frontier["Sharpe"].idxmax()
    min_vol_idx    = frontier["Volatility"].idxmin()
    target_idx     = (
        frontier["Expected Return"] - target_return
    ).abs().idxmin()

    key = {
        "Max Sharpe":      frontier.loc[max_sharpe_idx],
        "Min Volatility":  frontier.loc[min_vol_idx],
        f"Target ({target_return:.0%})": frontier.loc[target_idx],
    }

    print(f"\n  === {label} Key Portfolios ===")
    for name, port in key.items():
        print(f"  {name}: Return={port['Expected Return']:.1%}, "
              f"Vol={port['Volatility']:.1%}, "
              f"Sharpe={port['Sharpe']:.2f}")

    return key


# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────

if __name__ == "__main__":
    print("Loading market data...")
    monthly_returns             = fetch_public_returns()
    public_ret, public_cov      = compute_public_stats(monthly_returns)
    alts_ret, alts_vol          = build_alts_series()
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