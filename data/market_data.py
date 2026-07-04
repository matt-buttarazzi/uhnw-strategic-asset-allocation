'''
Author: Matthew R. Buttarazzi
Date: May 2026
Project: UHNW Strategic Asset Allocation with Alternatives
Description: Data layer for the UHNW portfolio optimizer. Pulls monthly public market
return data via yfinance and defines institutional alternative investment assumptions
sourced from Cambridge Associates benchmarks. Builds the combined 14-asset return vector
and covariance matrix used by the optimizer. All functions accept date and parameter
arguments so they can be called from both the standalone script and the Streamlit dashboard.
'''

import yfinance as yf
import pandas as pd
import numpy as np

TICKERS = {
    "US Large Cap":       "SPY",
    "US Small Cap":       "IWM",
    "Intl Developed":     "EFA",
    "Emerging Markets":   "EEM",
    "US Agg Bonds":       "AGG",
    "Long Duration UST":  "TLT",
    "TIPS":               "TIP",
    "Real Estate (REIT)": "VNQ",
    "Commodities":        "GSG",
    "Cash":               "SHY",
}

# Default date range used by the standalone script
DEFAULT_START = "2013-01-01"
DEFAULT_END   = "2024-12-31"

# CITE: Cambridge Associates Benchmarks + institutional consensus
# DESC: Alternatives don't have daily prices so we use point estimates. Each tuple is
# (expected_return, volatility, is_illiquid, liquidity_years). Smoothed reporting in
# alternatives data introduces a downward bias in measured volatility — true economic
# volatility is likely higher than what these figures suggest.
ALTS_ASSUMPTIONS = {
    #                         exp_return  volatility  illiquid  liquidity_years
    "Private Equity":        (0.155,      0.230,      True,     10),
    "Private Credit":        (0.100,      0.090,      True,     4),
    "Hedge Funds":           (0.080,      0.100,      False,    1),
    "Real Assets":           (0.090,      0.110,      True,     7),
}

# CITE: Roth and Barth, Capital Allocators Podcast
# DESC: PE returns tend to mean-revert toward their long-run average after periods of
# outperformance. We model this as a dampening factor that pulls the forward-looking
# PE return assumption toward PE_LONG_RUN_MEAN at a 30% reversion speed per period.
PE_LONG_RUN_MEAN  = 0.130
PE_MEAN_REVERSION = 0.30

# Correlation of each alternative asset class to US Large Cap equity
ALTS_EQUITY_CORR = {
    "Private Equity": 0.65,
    "Private Credit": 0.35,
    "Hedge Funds":    0.45,
    "Real Assets":    0.30,
}

# Pairwise correlations between alternative asset classes
ALTS_CROSS_CORR = {
    ("Private Equity", "Private Credit"): 0.40,
    ("Private Equity", "Hedge Funds"):    0.45,
    ("Private Equity", "Real Assets"):    0.35,
    ("Private Credit", "Hedge Funds"):    0.30,
    ("Private Credit", "Real Assets"):    0.25,
    ("Hedge Funds",    "Real Assets"):    0.20,
}


def fetch_public_returns(start_date=DEFAULT_START, end_date=DEFAULT_END):
    # Download monthly adjusted close prices for all tickers
    raw = yf.download(
        list(TICKERS.values()),
        start=start_date,
        end=end_date,
        interval="1mo",
        auto_adjust=True,
        progress=False,
    )

    # Newer yfinance versions return a MultiIndex, older versions return flat columns
    if isinstance(raw.columns, pd.MultiIndex):
        raw = raw["Close"]
    else:
        raw = raw["Close"]

    # Drop any tickers that failed to download entirely
    raw = raw.dropna(axis=1, how="all")

    ticker_to_name = {v: k for k, v in TICKERS.items()}
    raw.rename(columns=ticker_to_name, inplace=True)

    monthly_returns = raw.pct_change().dropna()
    return monthly_returns


def compute_public_stats(monthly_returns):
    # Annualize by multiplying monthly mean and covariance by 12
    ann_returns = monthly_returns.mean() * 12
    ann_cov     = monthly_returns.cov() * 12
    return ann_returns, ann_cov


def build_alts_series():
    returns = pd.Series(
        {k: v[0] for k, v in ALTS_ASSUMPTIONS.items()}, name="Expected Return"
    )
    vols = pd.Series(
        {k: v[1] for k, v in ALTS_ASSUMPTIONS.items()}, name="Volatility"
    )
    return returns, vols


def build_combined_universe(public_returns, public_cov, alts_returns, alts_vols):
    # Stack public and alternative return vectors into one combined series
    combined_returns = pd.concat([public_returns, alts_returns])

    all_assets = list(public_returns.index) + list(alts_returns.index)
    n_total    = len(all_assets)

    # Start with a zero matrix and fill in each block
    full_cov = pd.DataFrame(
        np.zeros((n_total, n_total)),
        index=all_assets,
        columns=all_assets
    )

    # Public-public block comes directly from realized covariance
    full_cov.loc[public_returns.index, public_returns.index] = public_cov

    # Use equity beta scaling to estimate alt-to-public covariances
    us_lc_vol = np.sqrt(public_cov.loc["US Large Cap", "US Large Cap"])
    for alt, (exp_ret, vol, illiq, _) in ALTS_ASSUMPTIONS.items():
        full_cov.loc[alt, alt] = vol ** 2

        eq_corr = ALTS_EQUITY_CORR[alt]
        for pub in public_returns.index:
            pub_vol = np.sqrt(public_cov.loc[pub, pub])
            eq_beta = public_cov.loc[pub, "US Large Cap"] / (us_lc_vol ** 2)
            cov_val = eq_corr * vol * pub_vol * eq_beta
            full_cov.loc[alt, pub] = cov_val
            full_cov.loc[pub, alt] = cov_val

    # Fill alt-to-alt covariances from our pairwise correlation assumptions
    for (a1, a2), corr in ALTS_CROSS_CORR.items():
        vol1    = ALTS_ASSUMPTIONS[a1][1]
        vol2    = ALTS_ASSUMPTIONS[a2][1]
        cov_val = corr * vol1 * vol2
        full_cov.loc[a1, a2] = cov_val
        full_cov.loc[a2, a1] = cov_val

    illiquid_assets = [k for k, v in ALTS_ASSUMPTIONS.items() if v[2]]
    return combined_returns, full_cov, illiquid_assets


def print_cma_table(public_returns, public_cov, alts_returns, alts_vols):
    rows = []

    for asset, ret in public_returns.items():
        vol = np.sqrt(public_cov.loc[asset, asset])
        if asset == "US Large Cap":
            corr = 1.0
        else:
            corr = public_cov.loc[asset, "US Large Cap"] / (
                np.sqrt(public_cov.loc[asset, asset]) *
                np.sqrt(public_cov.loc["US Large Cap", "US Large Cap"])
            )
        rows.append({
            "Asset Class":        asset,
            "Exp. Return":        f"{ret:.1%}",
            "Volatility":         f"{vol:.1%}",
            "Liquidity":          "Daily",
            "Corr. to US Equity": f"{corr:.2f}",
        })

    for asset, ret in alts_returns.items():
        vol  = alts_vols[asset]
        corr = ALTS_EQUITY_CORR[asset]
        liq  = f"{ALTS_ASSUMPTIONS[asset][3]}yr lock"
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


def main():
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

if __name__ == "__main__":
    main()