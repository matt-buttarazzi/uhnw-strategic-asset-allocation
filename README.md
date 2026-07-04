# UHNW Strategic Asset Allocation with Alternatives

![Python](https://img.shields.io/badge/python-3.11-blue)
![NumPy](https://img.shields.io/badge/numpy-1.26-blue)
![SciPy](https://img.shields.io/badge/scipy-1.12-blue)
![Streamlit](https://img.shields.io/badge/streamlit-1.x-red)
![License](https://img.shields.io/badge/license-MIT-brightgreen)

A quantitative portfolio construction framework analyzing how alternative investments impact long-term wealth outcomes for ultra-high-net-worth families. Using mean-variance optimization, institutional capital market assumptions, and Monte Carlo simulation, the analysis compares traditional public-market portfolios against diversified multi-asset allocations incorporating private equity, private credit, hedge funds, and real assets.

Designed to reflect portfolio construction principles commonly employed by multifamily offices serving ultra-high-net-worth families, incorporating liquidity management, alternatives allocation, and long-term capital market assumptions.

---

## Live Dashboard

An interactive Streamlit dashboard allows users to adjust date ranges, portfolio parameters, alternatives constraints, and PE mean reversion assumptions — then rerun the full optimization and Monte Carlo pipeline on demand.

```bash
pip install -r requirements.txt
streamlit run app.py
```

**Dashboard features:**
- Date range selector — compare results across different market regimes
- Starting portfolio size, withdrawal rate, time horizon, and risk-free rate inputs
- Alternatives allocation constraints (tier min/max, individual asset caps)
- PE mean reversion controls (long-run mean, reversion speed)
- Five output tabs: efficient frontier, Monte Carlo fan chart, portfolio allocation, capital market assumptions, drawdown analysis

---

## Key Findings

| Portfolio | Median 25yr Wealth | Prob > $500M | Median Max Drawdown |
|---|---|---|---|
| 60/40 Baseline | $267M | 7.5% | -18.2% |
| Public Optimized | $416M | 39.7% | — |
| UHNW + Alternatives | $486M | 47.3% | -11.1% |
| UHNW Stress Test | $397M | 34.1% | -21.2% |

Starting portfolio: $100M | 3% annual distributions | 25-year horizon | 10,000 simulations

The UHNW alternatives portfolio produces **82% higher median terminal wealth** than the 60/40 baseline, with **39% lower maximum drawdown**, and nearly **6x the probability** of exceeding $500M over a 25-year horizon.

---

## Methodology

### Asset Universe

**Public assets** (10 asset classes): Historical monthly returns pulled via yfinance, 2013–2024. Annualized means and covariance matrix computed from realized data using ETF proxies — SPY, IWM, EFA, EEM, AGG, TLT, TIP, VNQ, GSG, SHY.

**Alternative assets** (4 asset classes): Point estimates based on Cambridge Associates institutional benchmarks. Modeled with explicit correlation assumptions to public equity and cross-asset correlation structure.

| Asset Class | Exp. Return | Volatility | Liquidity | Corr to Equity |
|---|---|---|---|---|
| US Large Cap | 14.4% | 14.7% | Daily | 1.00 |
| US Agg Bonds | 1.5% | 5.0% | Daily | 0.36 |
| Private Equity | 14.8%* | 23.0% | 10yr lock | 0.65 |
| Private Credit | 10.0% | 9.0% | 4yr lock | 0.35 |
| Hedge Funds | 8.0% | 10.0% | 1yr lock | 0.45 |
| Real Assets | 9.0% | 11.0% | 7yr lock | 0.30 |

*PE return adjusted downward from raw 15.5% via mean reversion toward 13.0% long-run mean, per practitioner observations from Roth and Barth (Capital Allocators Podcast) that PE returns compress following periods of outperformance.

### Optimization

Mean-variance optimization via `scipy.optimize.minimize` (SLSQP). The UHNW portfolio incorporates a three-tier liquidity bucket framework reflecting institutional practice:

| Tier | Assets | Purpose | Constraint |
|---|---|---|---|
| Tier 1 — Liquid | Cash, Treasuries, TIPS, Agg Bonds | Fund 0–2yr distributions and capital calls | Minimum 8% |
| Tier 2 — Public | Equities, REITs, Hedge Funds | Intermediate liquidity | No explicit cap |
| Tier 3 — Illiquid | PE, Private Credit, Real Assets | Illiquidity premium capture | 15% – 40% |

### Monte Carlo Simulation

10,000 paths using correlated multivariate normal annual returns drawn from the full asset covariance matrix. Annual rebalancing to target weights. 3% annual withdrawal rate reflecting typical UHNW distribution needs.

**Stress test regime** applied to both portfolios:
- Equity volatility scaled 1.5x
- Equity-equity correlations spike to 0.85
- Bond-equity correlation increases to 0.20
- PE expected return reduced by 4% (lagged markdown effect)

### Known Limitations

- **Smoothing bias**: Alternative asset returns are reported on a lagged, appraised basis — measured volatility understates true economic volatility
- **Survivorship bias**: Cambridge Associates benchmarks reflect funds that reported returns; failed funds are underrepresented
- **Manager selection**: PE and private credit outcomes depend heavily on vintage year and manager quality — top-quartile access is a key assumption
- **Normal distribution assumption**: Annual returns modeled as multivariate normal; fat tails and skewness are not fully captured

---

## Repository Structure
uhnw-portfolio-optimizer/
├── app.py                          ← Streamlit dashboard (UI only, imports from modules)
├── data/
│   └── market_data.py              ← Data layer: yfinance pull, CMA table, universe construction
├── optimizer/
│   └── efficient_frontier.py       ← Mean-variance optimizer, frontier construction, liquidity constraints
├── simulation/
│   └── monte_carlo.py              ← Monte Carlo engine, stress test, drawdown analysis
├── visualization/
│   └── charts.py                   ← Four institutional-quality chart functions, return Figure objects
├── memo/
│   └── UHNW_Portfolio_Analysis.pdf ← Investment research memo
├── README.md
└── requirements.txt
---

## Setup

```bash
pip install -r requirements.txt
streamlit run app.py
```

Each module also runs standalone and prints results to the terminal:

```bash
python data/market_data.py
python optimizer/efficient_frontier.py
python simulation/monte_carlo.py
python visualization/charts.py
```

---

## Context

This project reflects the analytical framework used by multifamily offices and UHNW-focused RIAs when constructing portfolios for families with $50M–$500M+ in investable assets. The $50M+ account minimum and ~$180M average account size at firms like BBR Partners means advisors routinely navigate the liquidity, complexity, and manager-selection tradeoffs modeled here.

The analysis is intended as a quantitative foundation for understanding why UHNW families allocate meaningfully to alternatives — and what the long-term wealth implications of that access look like relative to traditional public-market portfolios.

---

## License

MIT. See LICENSE for details.