# Macro Carry Allocation

> A long-only multi-asset allocation strategy combining 12-month trend and cross-asset carry signals across nine ETFs, with monthly rebalancing, inverse-volatility position sizing, and portfolio-level volatility targeting at 10%.

![Python](https://img.shields.io/badge/python-3.13+-blue.svg)
![Status](https://img.shields.io/badge/status-v1%20complete-green.svg)

## Overview

This project extends Meb Faber's *A Quantitative Approach to Tactical Asset Allocation* ([SSRN 962461](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=962461)) by combining two signals with distinct economic motivations:

- **Trend** (behavioural premium): investors under-react to news, so prices that have risen tend to continue rising over multi-month horizons.
- **Carry** (risk premium): investors are paid to bear term, credit, devaluation, and storage risks. Assets currently offering higher carry are compensated for risks others want to offload.

The two signals are roughly uncorrelated within asset classes, so combining them diversifies risk-premium exposures rather than duplicating a single behavioural bet.

## Results

Backtest period: 2003-01-31 → 2026-05-20 (~22 years of out-of-sample-feeling history, given no parameter optimization on this window).

![Metrics table](results/metrics_table.png)

The strategy delivers comparable absolute return to a 60/40 SPY+IEF benchmark with **~9pp lower max drawdown** (-22.45% vs -31.41%). The Calmar ratio (return per unit of max drawdown) is meaningfully better (0.37 vs 0.28), confirming the strategy's value-add is concentrated in tail protection rather than mean return enhancement.

![Equity curve](results/equity_curve.png)

![Drawdown](results/drawdown.png)

The drawdown plot is where the strategy most visibly differentiates from 60/40. In 2008-2009, vol targeting and diversification limited the strategy's drawdown to ~19% while 60/40 hit -31%. In 2022 (the simultaneous bond/equity selloff), the strategy stayed in a single-digit drawdown while 60/40 had a sustained -20% drawdown.

![Rolling Sharpe](results/rolling_sharpe.png)

Rolling 1-year Sharpe ratios show the strategy and 60/40 co-moving with the overall equity environment. The strategy's relative outperformance is concentrated in EM-heavy periods (2010-2011) and the 2022-2023 stress period, while 60/40 wins in low-vol US equity bull-market periods (notably 2012-2015 QE).

![Weights history](results/weights_history.png)

The weights chart confirms the strategy is genuinely rotating: EM equity (EEM) dominates in 2009-2011; bonds (TLT/IEF, purple/brown) feature in 2015 and 2019; commodities (DBC, orange) take large positions during the 2021-2022 inflation shock; and the cash sleeve (BIL, grey) grows substantially in 2022 when vol-targeting scaled exposure down hard.

## Methodology

### Signal pipeline

For each of nine risk-asset ETFs (SPY, EFA, EEM, IEF, TLT, LQD, GLD, DBC, VNQ), compute two raw signals at each month-end:

- **Trend**: 12-month total return.
- **Carry**: asset yield minus 3-month T-bill yield, in decimal units.
  - *Equities*: trailing earnings yield (1 / index P/E) of the underlying index.
  - *Bonds and credit*: yield-to-worst of the underlying index.
  - *Gold*: zero asset yield, so carry equals minus the risk-free rate (you forgo T-bill yield to hold a non-yielding asset).
  - *Commodities*: weighted-average annualized log roll yield across the 14 commodities in the DBC index, computed from front-month and second-month generic futures prices.
  - *REITs*: trailing earnings yield from the RMZ index.

Then apply the three-stage transform pipeline on each rebalance date independently:

1. **Winsorize** each signal cross-sectionally using MAD-based ±3σ clipping. Robust to single outliers in a small cross-section (e.g. MXEA trailing P/E of 129 during the 2002-2003 earnings trough).
2. **Z-score** each signal cross-sectionally so trend and carry are on the same scale.
3. **Blend** with equal weights: composite = 0.5·z_trend + 0.5·z_carry. Assets with missing carry (RMZ pre-Oct 2005 for VNQ) get composite = z_trend (full weight on the available signal).

### Portfolio construction

A seven-step deterministic chain converts composite signals into actual weights:

1. Rank assets by composite z-score.
2. Select top 4.
3. Signal-proportional raw weights (negative composites floored at zero).
4. Inverse-volatility scaling using rolling 63-day annualized realized vol, so each held asset contributes equal risk to the portfolio.
5. Portfolio-level vol target at 10% annualized, using a constant-weights vol forecast that implicitly captures both asset vols and cross-asset correlations.
6. Gross leverage capped at 1.0 (no borrowing).
7. BIL cash sleeve absorbs any unused capital.

### Execution

Weights generated on month-end `t` are held from the open of `t+1` onwards — a one-day execution lag to avoid look-ahead bias. Transaction costs of 5bps per dollar of two-way turnover are applied on the execution day, conservative for liquid US-listed ETFs traded through a retail broker (Interactive Brokers or similar). Average monthly one-way turnover is ~20%; total cost drag over the full backtest is 0.28% per year.

## Project Structure

```
macro-carry-allocation/
├── README.md
├── requirements.txt
├── data/
│   ├── raw/                       # Bloomberg exports (gitignored, licensed data)
│   │   ├── macro_carry_data.xlsx
│   │   └── commodity_futures.xlsx
│   └── processed/                 # parquet cache (gitignored, reproducible)
├── src/
│   ├── data_loader.py             # Bloomberg workbook → DataFrames
│   ├── signals.py                 # trend, carry, winsorize, z-score, blend
│   ├── portfolio.py               # top-N, inverse-vol weights, vol target, cash sleeve
│   ├── backtest.py                # daily P&L with t+1 execution and costs
│   ├── metrics.py                 # CAGR, Sharpe, Sortino, drawdown analysis
│   └── make_charts.py             # produces results/ PNGs
└── results/
├── equity_curve.png
├── drawdown.png
├── rolling_sharpe.png
├── weights_history.png
└── metrics_table.png
```

Each module exposes a `__main__` smoke test that runs in isolation. The full pipeline is executed by `python src/make_charts.py`.

## Data Sources

All data sourced from a **Bloomberg Terminal** via manual BDH export. The raw `.xlsx` files are excluded from version control because Bloomberg data is licensed.

| Data type | Bloomberg field | Series | Sheet |
|---|---|---|---|
| ETF total return | `TOT_RETURN_INDEX_GROSS_DVDS` | SPY, EFA, EEM, IEF, TLT, LQD, GLD, DBC, VNQ, BIL | `prices` |
| Equity P/E | `PE_RATIO` | SPX, MXEA, MXEF, RMZ indices | `equity_pe` |
| Bond/credit yields | `INDEX_YIELD_TO_WORST` | LUATTRUU, LUTLTRUU, LUACTRUU indices | `bond_yields` |
| Risk-free | `PX_LAST` | USGG3M | `rf` |
| Commodity futures | `PX_LAST` | CL/CO/HO/XB/NG/GC/SI/LA/LX/LP/C/W/S/SB, front + second month | `futures` |

Date range: 2003-01-01 to present.

## v2 — Learning the signal blend (results)

We hypothesized that the hand-coded 50/50 blend of trend and carry signals in v1 might be improved by learning the blend weights from historical data. v2 implements a **walk-forward (expanding-window) OLS regression** of next-month asset returns on (z_trend, z_carry), refit each month using all prior data. The learned β coefficients replace v1's fixed 0.5/0.5 weighting at each rebalance date.

Two formulations were tested:
- **v2a**: regression on raw next-month asset returns.
- **v2b**: regression on cross-sectionally demeaned returns, to isolate the signal's ranking value from market-wide moves.

Both v2 formulations are evaluated on the same window as v1 from 2008-02-01 onwards (when v2's first valid composite is available, after the 60-month initial training period).

### Results

![v1 vs v2 metrics](results/compare_metrics.png)

| Metric | v1 (0.5/0.5) | v2 (learned) | Δ |
|---|---:|---:|---:|
| CAGR | 7.90% | 7.12% | −0.78% |
| Volatility | 10.84% | 10.67% | −0.17% |
| Sharpe ratio | 0.63 | 0.57 | −0.06 |
| Sortino ratio | 0.79 | 0.73 | −0.06 |
| Max drawdown | −22.45% | −26.32% | −3.87% |
| Calmar ratio | 0.35 | 0.27 | −0.08 |

![Equity curves](results/compare_equity.png)
![Drawdown comparison](results/compare_drawdown.png)
![Learned coefficients over time](results/compare_coefs.png)

**v2 underperformed v1 across every risk-adjusted metric.** The learned β coefficients show a 13x range across the walk-forward (β_trend from 0.0007 to 0.0084) but the variation was not predictive — month-to-month coefficient adjustments responded to estimation noise rather than persistent structural changes in the signal-return relationship.

### Interpretation

This is consistent with the well-documented "Markowitz problem" in portfolio optimization: estimation error in optimal weights can exceed the benefit of adapting weights over time. Real-world examples include AQR's findings that fixed-weight factor blends often outperform dynamically-optimized ones, and the Treynor-Black critique of mean-variance optimization more broadly.

Three concrete takeaways:

1. **v1's hand-coded 50/50 blend was already near-optimal for this signal set and universe.** The mean learned ratio β_carry / β_trend across the walk-forward was 0.91 (v2a) and 0.72 (v2b), both close to v1's implicit ratio of 1.0.

2. **Monthly-frequency macro factor signals carry insufficient information density to support stable weight learning.** With roughly 540 (asset, month) observations in the initial training window, β estimates have standard errors large enough that month-to-month adjustments inject more noise than signal.

3. **The cross-sectional demeaning fix (v2b) didn't help.** The hypothesis was that market-wide return co-movement was distorting the regression; in fact, removing it made performance slightly worse, suggesting the issue is fundamental to learning at this frequency rather than a fixable specification error.

### Why Model B (per-asset-class regressions) was not pursued

The original v2 plan included an extension to three per-asset-class regressions (equities / bonds / real assets) if Model A succeeded. Because Model A's failure was diagnosed as estimation noise rather than missing signal heterogeneity, the per-class extension would have made the problem worse (~150-200 observations per class versus ~540 for the global regression) and was abandoned.

### Suggested next direction

Improving this strategy further is more likely to come from **sensitivity testing the portfolio construction parameters** (TOP_N, vol target, vol lookback) than from further ML iteration on the signal blend. The structural decisions in `portfolio.py` (e.g., the choice of top-4 over top-3 or top-5; the 10% vol target; the 63-day vol lookback) were chosen heuristically and likely have more performance leverage than the trend/carry blend ratio.

## References

- Asness, Moskowitz, Pedersen (2013). *Value and Momentum Everywhere*. Journal of Finance.
- Koijen, Moskowitz, Pedersen, Vrugt (2018). *Carry*. Journal of Financial Economics.
- Faber (2007). *A Quantitative Approach to Tactical Asset Allocation*. SSRN.
- Moreira, Muir (2017). *Volatility-Managed Portfolios*. Journal of Finance.

## License

MIT. All performance figures are backtest-derived; none of this is investment advice.