'''
Author: Matthew R. Buttarazzi
Date: May 2026
Project: UHNW Strategic Asset Allocation with Alternatives
Description: Streamlit dashboard for the UHNW portfolio optimizer. Imports all computation
from the four module files — market_data, efficient_frontier, monte_carlo, and charts.
This file contains only UI code: sidebar parameters, layout, metrics, and tab rendering.
All analysis is delegated to the underlying modules.
'''

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data.market_data import (
    fetch_public_returns,
    compute_public_stats,
    build_alts_series,
    build_combined_universe,
    ALTS_ASSUMPTIONS,
    ALTS_EQUITY_CORR,
    ALTS_CROSS_CORR,
    PE_LONG_RUN_MEAN,
    PE_MEAN_REVERSION,
)
from optimizer.efficient_frontier import (
    build_public_frontier,
    build_uhnw_frontier,
    find_key_portfolios,
    extract_optimal_weights,
    get_6040_weights,
    adjust_pe_return,
    TIER1_ASSETS,
    TIER3_ASSETS,
)
from simulation.monte_carlo import (
    run_simulation,
    compute_outcomes,
    compute_max_drawdown,
    build_stress_cov,
    WEALTH_TARGET,
    WEALTH_FLOOR,
)
from visualization.charts import (
    plot_efficient_frontier,
    plot_fan_chart,
    plot_asset_allocation,
    plot_drawdown_comparison,
    COLORS,
    TIER_COLORS,
)

st.set_page_config(
    page_title="UHNW Portfolio Optimizer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .stApp { background-color: #F4F4F0; }

    [data-testid="stSidebar"] {
        background-color: #1A3A5C;
        padding-top: 0;
    }
    [data-testid="stSidebar"] .stMarkdown p,
    [data-testid="stSidebar"] .stMarkdown h1,
    [data-testid="stSidebar"] .stMarkdown h2,
    [data-testid="stSidebar"] .stMarkdown h3,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] .stSlider span,
    [data-testid="stSidebar"] .stRadio label,
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stNumberInput label {
        color: #FFFFFF !important;
    }

    [data-testid="stSidebar"] input[type="number"] {
        background-color: #2A4A6C !important;
        color: #FFFFFF !important;
        border: 1px solid #4A7AAC !important;
        border-radius: 6px !important;
    }
    [data-testid="stSidebar"] input[type="number"]:focus {
        border-color: #D4AF37 !important;
        outline: none !important;
    }

    [data-testid="stSidebar"] .stSlider > div > div > div > div {
        background-color: #D4AF37 !important;
    }

    [data-testid="stSidebar"] .stRadio > div { gap: 6px; }
    [data-testid="stSidebar"] .stRadio label {
        background-color: #2A4A6C;
        border-radius: 6px;
        padding: 6px 12px;
        cursor: pointer;
    }
    [data-testid="stSidebar"] .stRadio label:hover {
        background-color: #3A5A7C;
    }

    [data-testid="stSidebar"] [data-testid="stExpander"] summary,
    [data-testid="stSidebar"] [data-testid="stExpander"] summary:hover,
    [data-testid="stSidebar"] details summary,
    [data-testid="stSidebar"] .streamlit-expanderHeader,
    [data-testid="stSidebar"] .streamlit-expanderHeader:hover {
        background-color: #FFFFFF !important;
        border-radius: 6px !important;
        color: #1A3A5C !important;
        font-weight: 700 !important;
        padding: 10px 14px !important;
    }
    [data-testid="stSidebar"] [data-testid="stExpander"] summary p,
    [data-testid="stSidebar"] [data-testid="stExpander"] summary span,
    [data-testid="stSidebar"] details summary p,
    [data-testid="stSidebar"] details summary span {
        color: #1A3A5C !important;
    }
    [data-testid="stSidebar"] [data-testid="stExpander"] summary svg,
    [data-testid="stSidebar"] details summary svg {
        fill: #1A3A5C !important;
        stroke: #1A3A5C !important;
    }
    [data-testid="stSidebar"] .streamlit-expanderContent {
        background-color: #1A3A5C !important;
        border: 1px solid #2A4A6C !important;
        border-radius: 0 0 6px 6px !important;
    }

    .scroll-hint {
        color: #8AADCC;
        font-size: 11px;
        text-align: center;
        padding: 4px 0 8px 0;
        letter-spacing: 0.05em;
    }

    .stButton > button {
        background-color: #D4AF37 !important;
        color: #1A3A5C !important;
        font-weight: 700 !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 18px 32px !important;
        width: 85% !important;
        display: block !important;
        margin: 16px auto 8px auto !important;
        font-size: 1.05rem !important;
        letter-spacing: 0.08em !important;
        text-transform: uppercase !important;
    }
    .stButton > button:hover {
        background-color: #C49A00 !important;
        transform: translateY(-1px);
    }

    [data-testid="stMetric"] {
        background-color: #FFFFFF;
        border-radius: 10px;
        border: 1px solid #E0E0D8;
        padding: 18px 20px 14px 20px;
        box-shadow: 0 1px 4px rgba(26,58,92,0.06);
    }
    [data-testid="stMetricLabel"] {
        color: #888880 !important;
        font-size: 11px !important;
        font-weight: 600 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.08em !important;
    }
    [data-testid="stMetricValue"] {
        color: #1A3A5C !important;
        font-size: 1.65rem !important;
        font-weight: 700 !important;
        font-family: Georgia, serif !important;
    }
    [data-testid="stMetricDelta"] { font-size: 12px !important; font-weight: 500 !important; }

    .stTabs [data-baseweb="tab-list"] {
        background-color: #FFFFFF;
        border-radius: 10px 10px 0 0;
        padding: 4px 4px 0 4px;
        border-bottom: 2px solid #E0E0D8;
        gap: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        color: #888880;
        font-weight: 500;
        font-size: 13px;
        padding: 10px 18px;
        border-radius: 8px 8px 0 0;
        border: none;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1A3A5C !important;
        color: #FFFFFF !important;
        font-weight: 600 !important;
    }

    .page-header {
        background: linear-gradient(135deg, #1A3A5C 0%, #0D2440 100%);
        border-radius: 12px;
        padding: 28px 36px;
        margin-bottom: 24px;
    }
    .page-header h1 {
        color: #FFFFFF;
        font-family: Georgia, serif;
        font-size: 1.8rem;
        font-weight: 700;
        margin: 0 0 6px 0;
    }
    .page-header p {
        color: #D4AF37;
        font-size: 13px;
        margin: 0;
        letter-spacing: 0.04em;
    }

    .section-header {
        font-size: 11px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: #D4AF37;
        margin: 16px 0 8px 0;
        padding-bottom: 4px;
        border-bottom: 1px solid #2A4A6C;
    }

    [data-testid="stDataFrame"] {
        border-radius: 8px;
        overflow: hidden;
        border: 1px solid #E0E0D8;
    }

    .caption {
        font-size: 11px;
        color: #888880;
        font-style: italic;
        margin-top: 8px;
    }

    hr { border-color: #2A4A6C; margin: 12px 0; }
</style>
""", unsafe_allow_html=True)

plt.rcParams.update({
    "font.family":       "serif",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.color":        "#F0F0EC",
    "grid.linewidth":    0.5,
    "figure.facecolor":  "#FFFFFF",
    "axes.facecolor":    "#FFFFFF",
    "axes.labelsize":    11,
    "axes.titlesize":    12,
    "xtick.labelsize":   10,
    "ytick.labelsize":   10,
})

ALTS_LIQ = {
    "Private Equity": "10yr lock",
    "Private Credit": "4yr lock",
    "Hedge Funds":    "1yr lock",
    "Real Assets":    "7yr lock",
}


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="background:#0D2440; padding:20px 16px 16px 16px; margin:-1rem -1rem 0 -1rem;">
        <div style="color:#D4AF37; font-size:10px; font-weight:700; letter-spacing:0.12em; text-transform:uppercase; margin-bottom:4px;">UHNW Portfolio Optimizer</div>
        <div style="color:#FFFFFF; font-size:16px; font-weight:700; font-family:Georgia,serif; line-height:1.3;">Parameter Controls</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="scroll-hint">↕ scroll for more parameters</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-header">Market Data</div>', unsafe_allow_html=True)
    col_s, col_e = st.columns(2)
    with col_s:
        start_year = st.number_input("Start", min_value=2005, max_value=2020, value=2013, step=1)
    with col_e:
        end_year = st.number_input("End", min_value=2015, max_value=2024, value=2024, step=1)
    start_date = f"{start_year}-01-01"
    end_date   = f"{end_year}-12-31"

    st.markdown('<div class="section-header">Portfolio</div>', unsafe_allow_html=True)
    initial_wealth_m = st.number_input(
        "Starting Portfolio ($M)",
        min_value=10, max_value=1000, value=100, step=10,
    )
    initial_wealth  = initial_wealth_m * 1_000_000
    withdrawal_pct  = st.slider("Annual Withdrawal (%)", 0.0, 8.0, 3.0, 0.5)
    withdrawal_rate = withdrawal_pct / 100
    horizon         = st.slider("Time Horizon (Years)", 10, 40, 25, 1)
    rf_pct          = st.slider("Risk-Free Rate (%)", 0.0, 6.0, 4.0, 0.25)
    rf_rate         = rf_pct / 100
    n_sims          = st.radio(
        "Monte Carlo Paths",
        options=[1_000, 5_000, 10_000],
        format_func=lambda x: f"{x:,}",
        index=2,
        horizontal=True,
    )

    with st.expander("Alternatives Constraints", expanded=False):
        tier3_min_pct = st.slider("Min Alts Allocation (%)", 0,  30, 15, 1)
        tier3_max_pct = st.slider("Max Alts Allocation (%)", 20, 60, 40, 1)
        pe_max_pct    = st.slider("Max Private Equity (%)",  0,  30, 20, 1)
        pc_max_pct    = st.slider("Max Private Credit (%)",  0,  25, 15, 1)
        hf_max_pct    = st.slider("Max Hedge Funds (%)",     0,  25, 15, 1)
        ra_max_pct    = st.slider("Max Real Assets (%)",     0,  20, 10, 1)

    tier3_min = tier3_min_pct / 100
    tier3_max = tier3_max_pct / 100
    pe_max    = pe_max_pct    / 100
    pc_max    = pc_max_pct    / 100
    hf_max    = hf_max_pct    / 100
    ra_max    = ra_max_pct    / 100

    with st.expander("PE Mean Reversion", expanded=False):
        st.caption("Per Roth & Barth (Capital Allocators): PE returns revert toward their long-run mean after periods of outperformance.")
        pe_long_run_pct = st.slider("Long-Run Mean (%)", 8.0, 16.0, 13.0, 0.5)
        pe_rev_spd_pct  = st.slider("Reversion Speed (%)", 0, 60, 30, 5)

    pe_long_run = pe_long_run_pct / 100
    pe_rev_spd  = pe_rev_spd_pct  / 100

    st.markdown("<br>", unsafe_allow_html=True)
    run = st.button("Run Analysis")


# ── Page header ────────────────────────────────────────────────────────────────
st.markdown("""
<div class="page-header">
    <h1>UHNW Strategic Asset Allocation</h1>
    <p>QUANTIFYING THE LONG-RUN WEALTH IMPACT OF ALTERNATIVES ACCESS FOR ULTRA-HIGH-NET-WORTH FAMILIES</p>
</div>
""", unsafe_allow_html=True)

if not run:
    st.markdown("""
    <div style="background:#FFFFFF; border-radius:10px; border:1px solid #E0E0D8; padding:40px; text-align:center; color:#888880;">
        <div style="font-size:2rem; margin-bottom:12px;">📊</div>
        <div style="font-size:16px; font-weight:600; color:#1A3A5C; margin-bottom:8px;">Configure parameters and run analysis</div>
        <div style="font-size:13px;">Adjust the controls in the sidebar — scroll down for alternatives constraints and PE assumptions — then click <strong>Run Analysis</strong> to generate results.</div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()


# ── Run pipeline ───────────────────────────────────────────────────────────────
with st.spinner("Fetching market data..."):
    monthly_returns = fetch_public_returns(start_date, end_date)

with st.spinner("Building asset universe and optimizing..."):
    public_ret, public_cov = compute_public_stats(monthly_returns)
    alts_ret, alts_vol     = build_alts_series()

    # Apply user-controlled PE mean reversion before building the combined universe
    raw_pe = alts_ret["Private Equity"]
    alts_ret["Private Equity"] = raw_pe - pe_rev_spd * (raw_pe - pe_long_run)

    combined_ret, combined_cov, illiquid = build_combined_universe(
        public_ret, public_cov, alts_ret, alts_vol
    )

    assets     = list(combined_ret.index)
    pub_assets = list(public_ret.index)

    public_frontier = build_public_frontier(public_ret, public_cov, rf=rf_rate)
    uhnw_frontier   = build_uhnw_frontier(
        combined_ret, combined_cov,
        rf=rf_rate,
        tier3_min=tier3_min, tier3_max=tier3_max,
        pe_max=pe_max, pc_max=pc_max,
        hf_max=hf_max, ra_max=ra_max,
        pe_long_run=pe_long_run, pe_reversion=pe_rev_spd,
    )

public_keys = find_key_portfolios(public_frontier, "Public Only")
uhnw_keys   = find_key_portfolios(uhnw_frontier,   "UHNW + Alts")

w_uhnw = extract_optimal_weights(uhnw_frontier, assets, "max_sharpe")
w_6040 = get_6040_weights(assets)

w_pub_opt    = extract_optimal_weights(public_frontier, pub_assets, "max_sharpe")
w_pub_padded = np.zeros(len(assets))
for i, asset in enumerate(pub_assets):
    if asset in assets:
        w_pub_padded[assets.index(asset)] = w_pub_opt[i]

stress_cov, stress_ret = build_stress_cov(combined_cov, combined_ret)

with st.spinner(f"Running {n_sims:,} Monte Carlo simulations..."):
    sim_6040        = run_simulation(w_6040,       combined_ret, combined_cov, "60/40 Baseline",
                                     initial_wealth=initial_wealth, n_sims=n_sims,
                                     horizon=horizon, withdrawal_rate=withdrawal_rate)
    sim_6040_stress = run_simulation(w_6040,       stress_ret,   stress_cov,   "60/40 Stress",
                                     initial_wealth=initial_wealth, n_sims=n_sims,
                                     horizon=horizon, withdrawal_rate=withdrawal_rate)
    sim_pub         = run_simulation(w_pub_padded, combined_ret, combined_cov, "Public Optimized",
                                     initial_wealth=initial_wealth, n_sims=n_sims,
                                     horizon=horizon, withdrawal_rate=withdrawal_rate)
    sim_uhnw        = run_simulation(w_uhnw,       combined_ret, combined_cov, "UHNW + Alternatives",
                                     initial_wealth=initial_wealth, n_sims=n_sims,
                                     horizon=horizon, withdrawal_rate=withdrawal_rate)
    sim_stress      = run_simulation(w_uhnw,       stress_ret,   stress_cov,   "UHNW Stress",
                                     initial_wealth=initial_wealth, n_sims=n_sims,
                                     horizon=horizon, withdrawal_rate=withdrawal_rate)


# ── Key metrics ────────────────────────────────────────────────────────────────
terminal_6040 = sim_6040.iloc[:, -1]
terminal_uhnw = sim_uhnw.iloc[:, -1]
median_6040   = terminal_6040.median() / 1e6
median_uhnw   = terminal_uhnw.median() / 1e6
prob_6040     = (terminal_6040 > WEALTH_TARGET).mean()
prob_uhnw     = (terminal_uhnw > WEALTH_TARGET).mean()
pub_sharpe    = public_frontier["Sharpe"].max()
uhnw_sharpe   = uhnw_frontier["Sharpe"].max()

c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("UHNW Median Wealth",  f"${median_uhnw:.0f}M",  f"+{((median_uhnw/median_6040)-1)*100:.0f}% vs 60/40")
c2.metric("60/40 Median Wealth", f"${median_6040:.0f}M")
c3.metric("Prob > $500M (UHNW)", f"{prob_uhnw:.1%}",      f"+{(prob_uhnw-prob_6040)*100:.1f}pp vs 60/40")
c4.metric("UHNW Max Sharpe",     f"{uhnw_sharpe:.2f}",    f"+{uhnw_sharpe-pub_sharpe:.2f} vs public")
c5.metric("Data Window",         f"{start_year}–{end_year}")
c6.metric("Simulations",         f"{n_sims:,}")

st.markdown("<br>", unsafe_allow_html=True)


# ── Tabs ───────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "  Efficient Frontier  ",
    "  Monte Carlo  ",
    "  Allocation  ",
    "  Capital Market Assumptions  ",
    "  Drawdown Analysis  ",
])

weights_for_chart = {
    "60/40 Baseline":      {asset: w_6040[i] for i, asset in enumerate(assets)},
    "UHNW + Alternatives": {asset: w_uhnw[i] for i, asset in enumerate(assets)},
}

with tab1:
    fig = plot_efficient_frontier(public_frontier, uhnw_frontier, public_keys, uhnw_keys,
                                  start_year=start_year, end_year=end_year)
    st.pyplot(fig)
    plt.close(fig)

with tab2:
    fig = plot_fan_chart(sim_6040, sim_uhnw, sim_6040_stress, sim_stress,
                         initial_wealth=initial_wealth, withdrawal_rate=withdrawal_rate,
                         horizon=horizon, n_sims=n_sims,
                         start_year=start_year, end_year=end_year)
    st.pyplot(fig)
    plt.close(fig)

    st.markdown("#### Wealth outcome summary")
    rows = []
    for label, sim in [("60/40 Baseline", sim_6040), ("Public Optimized", sim_pub), ("UHNW + Alternatives", sim_uhnw)]:
        t = sim.iloc[:, -1]
        rows.append({
            "Portfolio":          label,
            "Median Wealth":      f"${t.median()/1e6:.1f}M",
            "10th Percentile":    f"${t.quantile(0.10)/1e6:.1f}M",
            "90th Percentile":    f"${t.quantile(0.90)/1e6:.1f}M",
            "Prob > $500M":       f"{(t > WEALTH_TARGET).mean():.1%}",
            "Prob < $50M (Ruin)": f"{(t < WEALTH_FLOOR).mean():.2%}",
        })
    st.dataframe(pd.DataFrame(rows).set_index("Portfolio"), use_container_width=True)
    st.markdown(f'<div class="caption">Starting portfolio: ${initial_wealth_m}M | {withdrawal_pct:.1f}% annual distributions | {horizon}-year horizon | {n_sims:,} simulations</div>', unsafe_allow_html=True)

with tab3:
    fig = plot_asset_allocation(weights_for_chart)
    st.pyplot(fig)
    plt.close(fig)

with tab4:
    st.markdown(f"#### Capital market assumptions  |  {start_year}–{end_year}")
    st.markdown(f'<div class="caption">Public asset returns reflect realized monthly data {start_year}–{end_year}. Alternative asset assumptions sourced from Cambridge Associates institutional benchmarks. PE return adjusted for mean reversion toward {pe_long_run_pct:.1f}% long-run mean.</div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    rows = []
    for asset, ret in public_ret.items():
        vol = np.sqrt(public_cov.loc[asset, asset])
        if asset == "US Large Cap":
            corr = 1.0
        else:
            corr = public_cov.loc[asset, "US Large Cap"] / (
                np.sqrt(public_cov.loc[asset, asset]) *
                np.sqrt(public_cov.loc["US Large Cap", "US Large Cap"])
            )
        tier = "Tier 1" if asset in TIER1_ASSETS else "Tier 2"
        rows.append({
            "Asset Class":    asset,
            "Exp. Return":    f"{ret:.1%}",
            "Volatility":     f"{vol:.1%}",
            "Liquidity":      "Daily",
            "Corr to Equity": f"{corr:.2f}",
            "Tier":           tier,
        })

    for asset, (exp_ret, vol, illiq, liq_yrs) in ALTS_ASSUMPTIONS.items():
        ret  = combined_ret[asset]
        corr = ALTS_EQUITY_CORR[asset]
        rows.append({
            "Asset Class":    asset,
            "Exp. Return":    f"{ret:.1%}",
            "Volatility":     f"{vol:.1%}",
            "Liquidity":      ALTS_LIQ[asset],
            "Corr to Equity": f"{corr:.2f}",
            "Tier":           "Tier 3",
        })

    st.dataframe(pd.DataFrame(rows).set_index("Asset Class"), use_container_width=True)

with tab5:
    fig = plot_drawdown_comparison(sim_6040, sim_uhnw, sim_stress, initial_wealth=initial_wealth)
    st.pyplot(fig)
    plt.close(fig)

    st.markdown("#### Drawdown summary")
    dd_rows = []
    for label, sim in [("60/40 Baseline", sim_6040), ("UHNW + Alternatives", sim_uhnw), ("UHNW Stress Test", sim_stress)]:
        dd = compute_max_drawdown(sim)
        dd_rows.append({
            "Portfolio":           label,
            "Median Max Drawdown": f"{dd['Median Max Drawdown']:.1f}%",
            "Worst 10% Drawdown":  f"{dd['Worst 10% Drawdown']:.1f}%",
        })
    st.dataframe(pd.DataFrame(dd_rows).set_index("Portfolio"), use_container_width=True)
    st.markdown('<div class="caption">Maximum drawdown computed across all simulation paths. Distributions include annual withdrawal rate.</div>', unsafe_allow_html=True)