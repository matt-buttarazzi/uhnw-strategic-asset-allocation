"""
Module 1: Data Layer
UHNW Portfolio Optimizer
------------------------
Pulls public market data via yfinance and defines
institutional alternative investment assumptions.
"""

import yfinance as yf
import pandas as pd
import numpy as np

# ─────────────────────────────────────────
# 1.1 PUBLIC ASSET CLASS PROXIES
# ─────────────────────────────────────────

TICKERS = {
    "US Large Cap":        "SPY",
    "US Small Cap":        "IWM",
    "Intl Developed":      "EFA",
    "Emerging Markets":    "EEM",
    "US Agg Bonds":        "AGG",
    "Long Duration UST":   "TLT",
    "TIPS":                "TIP",
    "Real Estate (REIT)":  "VNQ",
    "Commodities":         "GSG",
    "Cash":                "SHY",
}

START_DATE = "2013-01-01"
END_DATE   = "2024-12-31"  # 10 full years of data


def fetch_public_returns() -> pd.DataFrame:
    """
    Downloads monthly adjusted close prices for all public
    asset class proxies and returns a DataFrame of monthly returns.
    """
    ticker_list = list(TICKERS.values())
    
    raw = yf.download(
        ticker_list,
        start=START_DATE,
        end=END_DATE,
        interval="1mo",
        auto_adjust=True,
        progress=False,
    )
    # Handle MultiIndex columns from newer yfinance versions
    if isinstance(raw.columns, pd.MultiIndex):
        raw = raw["Close"]
    else:
        raw = raw["Close"]

    # Drop any tickers that failed to download
    raw = raw.dropna(axis=1, how="all")

    # Rename columns from ticker to asset class name
    ticker_to_name = {v: k for k, v in TICKERS.items()}
    raw.rename(columns=ticker_to_name, inplace=True)

    # Monthly returns
    monthly_returns = raw.pct_change().dropna()
    
    return monthly_returns


def compute_public_stats(monthly_returns: pd.DataFrame):
    """
    Computes annualized mean returns and covariance matrix
    for public asset classes.
    """
    ann_returns = monthly_returns.mean() * 12
    ann_cov     = monthly_returns.cov() * 12
    return ann_returns, ann_cov


# ─────────────────────────────────────────
# 1.2 ALTERNATIVE ASSET CLASS ASSUMPTIONS
# ─────────────────────────────────────────
# Source: Cambridge Associates benchmarks + institutional consensus
# Note: Alternatives use point estimates due to lack of daily pricing.
# Smoothed reporting introduces downward bias in measured volatility —
# true economic volatility is likely higher (acknowledged in memo).

ALTS_ASSUMPTIONS = {
    #                          exp_return  volatility  illiquid  liquidity_years
    "Private Equity":         (0.155,      0.230,      True,     10),
    "Private Credit":         (0.100,      0.090,      True,     4),
    "Hedge Funds":            (0.080,      0.100,      False,    1),
    "Real Assets":            (0.090,      0.110,      True,     7),
}

# PE mean reversion parameter (per Roth/Barth, Capital Allocators)
# PE returns tend to mean-revert toward long-run avg ~12-13% after
# periods of outperformance. Modeled as a dampening factor applied
# to forward-looking PE return assumption.
PE_LONG_RUN_MEAN   = 0.130
PE_MEAN_REVERSION  = 0.30   # 30% pull toward long-run mean per period

# Correlation of alternatives to US Large Cap equity
ALTS_EQUITY_CORR = {
    "Private Equity":  0.65,
    "Private Credit":  0.35,
    "Hedge Funds":     0.45,
    "Real Assets":     0.30,
}

# Correlation between alternatives themselves
ALTS_CROSS_CORR = {
    ("Private Equity",  "Private Credit"): 0.40,
    ("Private Equity",  "Hedge Funds"):    0.45,
    ("Private Equity",  "Real Assets"):    0.35,
    ("Private Credit",  "Hedge Funds"):    0.30,
    ("Private Credit",  "Real Assets"):    0.25,
    ("Hedge Funds",     "Real Assets"):    0.20,
}


def build_alts_series() -> tuple[pd.Series, pd.Series]:
    """
    Returns annualized expected returns and volatilities
    for alternative asset classes as pandas Series.
    """
    names   = list(ALTS_ASSUMPTIONS.keys())
    returns = pd.Series(
        {k: v[0] for k, v in ALTS_ASSUMPTIONS.items()}, name="Expected Return"
    )
    vols    = pd.Series(
        {k: v[1] for k, v in ALTS_ASSUMPTIONS.items()}, name="Volatility"
    )
    return returns, vols


# ─────────────────────────────────────────
# 1.3 COMBINED ASSET UNIVERSE
# ─────────────────────────────────────────

def build_combined_universe(
    public_returns: pd.Series,
    public_cov: pd.DataFrame,
    alts_returns: pd.Series,
    alts_vols: pd.Series,
) -> tuple[pd.Series, pd.DataFrame, list]:
    """
    Merges public and alternative assets into a single
    return vector and covariance matrix for optimization.

    Returns:
        combined_returns  : pd.Series of annualized expected returns
        combined_cov      : pd.DataFrame covariance matrix
        illiquid_assets   : list of illiquid asset names
    """
    # Combined return vector
    combined_returns = pd.concat([public_returns, alts_returns])

    # Start with public covariance matrix
    all_assets  = list(public_returns.index) + list(alts_returns.index)
    n_public    = len(public_returns)
    n_alts      = len(alts_returns)
    n_total     = n_public + n_alts

    full_cov = pd.DataFrame(
        np.zeros((n_total, n_total)),
        index=all_assets,
        columns=all_assets
    )

    # Fill public-public block
    full_cov.loc[public_returns.index, public_returns.index] = public_cov

    # Fill alts-alts and alts-public blocks
    us_lc_vol = np.sqrt(public_cov.loc["US Large Cap", "US Large Cap"])

    for alt, (exp_ret, vol, illiq, _) in ALTS_ASSUMPTIONS.items():
        # Alt variance
        full_cov.loc[alt, alt] = vol ** 2

        # Alt-to-public covariance via equity correlation
        eq_corr = ALTS_EQUITY_CORR[alt]
        for pub in public_returns.index:
            pub_vol  = np.sqrt(public_cov.loc[pub, pub])
            eq_beta  = public_cov.loc[pub, "US Large Cap"] / (us_lc_vol ** 2)
            cov_val  = eq_corr * vol * pub_vol * eq_beta
            full_cov.loc[alt, pub] = cov_val
            full_cov.loc[pub, alt] = cov_val

    # Alt-to-alt covariances
    for (a1, a2), corr in ALTS_CROSS_CORR.items():
        vol1 = ALTS_ASSUMPTIONS[a1][1]
        vol2 = ALTS_ASSUMPTIONS[a2][1]
        cov_val = corr * vol1 * vol2
        full_cov.loc[a1, a2] = cov_val
        full_cov.loc[a2, a1] = cov_val

    # Illiquid asset list
    illiquid_assets = [
        k for k, v in ALTS_ASSUMPTIONS.items() if v[2]
    ]

    return combined_returns, full_cov, illiquid_assets


# ─────────────────────────────────────────
# 1.4 CAPITAL MARKET ASSUMPTIONS TABLE
# ─────────────────────────────────────────

def print_cma_table(
    public_returns: pd.Series,
    public_cov: pd.DataFrame,
    alts_returns: pd.Series,
    alts_vols: pd.Series,
):
    """
    Prints a formatted Capital Market Assumptions table
    suitable for the institutional memo.
    """
    rows = []

    for asset, ret in public_returns.items():
        vol      = np.sqrt(public_cov.loc[asset, asset])
        illiq    = "Daily"
        corr     = 1.0 if asset == "US Large Cap" else \
                   public_cov.loc[asset, "US Large Cap"] / (
                       np.sqrt(public_cov.loc[asset, asset]) *
                       np.sqrt(public_cov.loc["US Large Cap", "US Large Cap"])
                   )
        rows.append({
            "Asset Class":        asset,
            "Exp. Return":        f"{ret:.1%}",
            "Volatility":         f"{vol:.1%}",
            "Liquidity":          illiq,
            "Corr. to US Equity": f"{corr:.2f}",
        })

    for asset, ret in alts_returns.items():
        vol   = alts_vols[asset]
        corr  = ALTS_EQUITY_CORR[asset]
        liq   = f"{ALTS_ASSUMPTIONS[asset][3]}yr lock"
        rows.append({
            "Asset Class":        asset,
            "Exp. Return":        f"{ret:.1%}",
            "Volatility":         f"{vol:.1%}",
            "Liquidity":          liq,
            "Corr. to US Equity": f"{corr:.2f}",
        })

    df = pd.DataFrame(rows).set_index("Asset Class")
    print("\n=== CAPITAL MARKET ASSUMPTIONS ===\n")
    print(df.to_string())
    print()


# ─────────────────────────────────────────
# MAIN — run module standalone to verify
# ─────────────────────────────────────────

if __name__ == "__main__":
    print("Fetching public market data...")
    monthly_returns = fetch_public_returns()
    public_ret, public_cov = compute_public_stats(monthly_returns)

    print("Building alternatives assumptions...")
    alts_ret, alts_vol = build_alts_series()

    print("Building combined universe...")
    combined_ret, combined_cov, illiquid = build_combined_universe(
        public_ret, public_cov, alts_ret, alts_vol
    )

    print_cma_table(public_ret, public_cov, alts_ret, alts_vol)

    print(f"Total assets in universe: {len(combined_ret)}")
    print(f"Illiquid assets: {illiquid}")
    print(f"\nCombined return vector:\n{combined_ret.round(4)}")
    print("\nModule 1 complete.")
