"""
Signal generation for the macro carry allocation strategy.

Two signals are combined:

    - Trend: 12-month total return on rebalance date.
      Available for all assets with >= 12 months of price history.

    - Carry: asset yield minus 3-month T-bill yield (in decimal).
      Equity ETFs: 1/PE - rf.  Bond/credit ETFs: index YTW - rf.
      GLD: -rf (gold pays no yield, so its carry is the forgone T-bill).
      DBC: NaN (v1 skips DBC carry; futures roll-yield deferred to v2).
      VNQ: NaN before RMZ index inception (Oct 2005).

Winsorization uses MAD-based 3-sigma capping rather than std-based,
because the cross-section is small (9 assets) and a single extreme value
(e.g. MXEA trailing P/E of 129 in early 2003) would otherwise inflate
the std and compress every other asset's z-score toward zero.

Z-scoring is CROSS-SECTIONAL on each rebalance date — i.e., relative
to the other assets in the universe at that moment, not to the asset's
own time-series history. This frames both signals as risk-premium
captures ("hold the assets currently offering the most compensation"),
which is the standard formulation for multi-asset allocation.

Blending: composite = 0.5 * z_trend + 0.5 * z_carry, with NaN handling.
Assets with NaN carry get composite = z_trend (full weight on the
signal that's available). This is the v1 choice; v2 will learn the
blend weights from data.
"""

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR: int = 252
TREND_LOOKBACK_DAYS: int = 252
WINSORIZE_K: float = 3.0
MAD_NORMAL_SCALE: float = 1.4826
RISK_ASSETS: list[str] = [
    "SPY", "EFA", "EEM", "IEF", "TLT", "LQD", "GLD", "DBC", "VNQ",
]
DBC_WEIGHTS: dict[str, float] = {
    "CL": 0.12375, "CO": 0.12375, "HO": 0.12375, "XB": 0.12375,
    "NG": 0.05500,
    "GC": 0.08000, "SI": 0.02000,
    "LA": 0.04166, "LX": 0.04166, "LP": 0.04166,
    "C":  0.05625, "W":  0.05625, "S":  0.05625, "SB": 0.05625,
}
ROLL_ANNUALIZATION: float = 12.0
# ---------------------------------------------------------------
# Rebalance calendar
# ---------------------------------------------------------------

def month_end_rebalance_dates(prices: pd.DataFrame) -> pd.DatetimeIndex:
    return prices.groupby(prices.index.to_period("M")).tail(1).index


# ---------------------------------------------------------------
# Raw signals
# ---------------------------------------------------------------

def compute_trend(
    prices: pd.DataFrame,
    rebalance_dates: pd.DatetimeIndex,
    lookback: int = TREND_LOOKBACK_DAYS,
) -> pd.DataFrame:
    trend_daily = prices.pct_change(lookback)
    return trend_daily.reindex(rebalance_dates)

def compute_dbc_carry(
    futures: pd.DataFrame,
    rebalance_dates: pd.DatetimeIndex,
    weights: dict[str, float] = None,
) -> pd.Series:
    """
    DBC's carry as a weighted-average annualized log roll yield across
    the 14 underlying commodities.

    For each commodity c with front-month price M1 and second-month price M2:

        roll_yield_c = ln(M1_c / M2_c) * 12

    Backwardation (M1 > M2) gives a positive roll yield; contango (M1 < M2)
    gives a negative one. Annualized by multiplying by 12 since each
    generic contract spans ~1 month.

    DBC carry is then the weighted average across commodities using DBC's
    index weights. Commodities with missing data (notably XB before
    October 2005) are dropped and the remaining weights renormalized.

    Note: roll yield is already an excess-over-cash quantity.
    """
    if weights is None:
        weights = DBC_WEIGHTS

    # Snap futures to rebalance dates using last observed prices
    futures_rebal = futures.reindex(rebalance_dates, method="ffill")

    # Per-commodity annualized log roll yield (one column per commodity)
    roll_yields = pd.DataFrame(index=rebalance_dates, dtype=float)
    for root in weights:
        m1 = futures_rebal.get(f"{root}_M1")
        m2 = futures_rebal.get(f"{root}_M2")
        # log(m1/m2) returns NaN where either price is NaN or non-positive
        roll_yields[root] = np.log(m1 / m2) * ROLL_ANNUALIZATION

    # Weighted average, renormalizing across commodities available on each date
    weight_series = pd.Series(weights)
    available_mask = roll_yields.notna().astype(float)
    weighted_contrib = roll_yields.fillna(0.0) * weight_series
    available_weight = available_mask * weight_series

    numer = weighted_contrib.sum(axis=1)
    denom = available_weight.sum(axis=1)
    carry = numer / denom.where(denom > 0)
    carry.name = "DBC_carry"
    return carry

def compute_carry(
    equity_ey: pd.DataFrame,
    bond_yields: pd.DataFrame,
    rf_annual: pd.Series,
    futures: pd.DataFrame,
    rebalance_dates: pd.DatetimeIndex,
    risk_assets: list[str] = RISK_ASSETS,
) -> pd.DataFrame:
    """
    Raw carry signal: asset yield minus 3M T-bill yield, in DECIMAL.
    For DBC: weighted-average log roll yield across 14 underlying
    commodities (already an excess-over-cash quantity; no rf subtraction).
    """
    rf_on_rebal = rf_annual.reindex(rebalance_dates, method="ffill")
    equity_on_rebal = equity_ey.reindex(rebalance_dates, method="ffill")
    bonds_on_rebal = bond_yields.reindex(rebalance_dates, method="ffill")

    carry = pd.DataFrame(
        np.nan, index=rebalance_dates, columns=risk_assets, dtype=float,
    )

    for asset in ["SPY", "EFA", "EEM", "VNQ"]:
        if asset in equity_on_rebal.columns:
            carry[asset] = equity_on_rebal[asset] - rf_on_rebal

    for asset in ["IEF", "TLT", "LQD"]:
        if asset in bonds_on_rebal.columns:
            carry[asset] = bonds_on_rebal[asset] - rf_on_rebal

    carry["GLD"] = -rf_on_rebal
    carry["DBC"] = compute_dbc_carry(futures, rebalance_dates)

    return carry


# ---------------------------------------------------------------
# Winsorization
# ---------------------------------------------------------------

def winsorize_cross_section_mad(
    cross_section: pd.Series,
    k: float = WINSORIZE_K,
) -> pd.Series:
    """
    MAD-based winsorization on a single cross-sectional Series.

    Robust to single outliers in a small cross-section, where classical
    std-based winsorization would have its std inflated by the very
    outlier it's trying to cap.
    """
    valid = cross_section.dropna()
    if len(valid) < 3:
        # Not enough data to estimate a meaningful spread
        return cross_section

    med = valid.median()
    mad = MAD_NORMAL_SCALE * (valid - med).abs().median()

    if mad == 0:
        # All values identical or near-identical; nothing to winsorize
        return cross_section

    lower = med - k * mad
    upper = med + k * mad
    return cross_section.clip(lower=lower, upper=upper)


def winsorize_panel(raw_signal: pd.DataFrame, k: float = WINSORIZE_K) -> pd.DataFrame:
    return raw_signal.apply(lambda row: winsorize_cross_section_mad(row, k=k), axis=1)


# ---------------------------------------------------------------
# Cross-sectional z-scoring
# ---------------------------------------------------------------

def zscore_cross_section(cross_section: pd.Series) -> pd.Series:
    valid = cross_section.dropna()
    if len(valid) < 2:
        return cross_section * np.nan

    mu = valid.mean()
    sigma = valid.std(ddof=0)

    if sigma == 0:
        return cross_section * 0.0

    return (cross_section - mu) / sigma


def zscore_panel(panel: pd.DataFrame) -> pd.DataFrame:
    """Apply z-scoring row-by-row (per rebalance date)."""
    return panel.apply(zscore_cross_section, axis=1)


# ---------------------------------------------------------------
# Blending
# ---------------------------------------------------------------

def blend_signals(
    z_trend: pd.DataFrame,
    z_carry: pd.DataFrame,
    trend_weight: float = 0.5,
    carry_weight: float = 0.5,
) -> pd.DataFrame:
    
    both_present = z_trend.notna() & z_carry.notna()
    only_trend = z_trend.notna() & z_carry.isna()

    composite = pd.DataFrame(np.nan, index=z_trend.index, columns=z_trend.columns)
    composite = composite.mask(
        both_present, trend_weight * z_trend + carry_weight * z_carry,
    )
    composite = composite.mask(only_trend, z_trend)

    return composite


# ---------------------------------------------------------------
# End-to-end pipeline
# ---------------------------------------------------------------

def compute_composite_signal(
    prices: pd.DataFrame,
    equity_ey: pd.DataFrame,
    bond_yields: pd.DataFrame,
    rf_daily: pd.Series,
    futures: pd.DataFrame,
    risk_assets: list[str] = RISK_ASSETS,
    trend_lookback: int = TREND_LOOKBACK_DAYS,
    winsorize_k: float = WINSORIZE_K,
    trend_weight: float = 0.5,
    carry_weight: float = 0.5,
) -> dict:
    
    rebalance_dates = month_end_rebalance_dates(prices[risk_assets])
    rf_annual = rf_daily * TRADING_DAYS_PER_YEAR

    raw_trend = compute_trend(prices[risk_assets], rebalance_dates, trend_lookback)
    raw_carry = compute_carry(
        equity_ey, bond_yields, rf_annual, futures, rebalance_dates, risk_assets,
    )

    wins_trend = winsorize_panel(raw_trend, k=winsorize_k)
    wins_carry = winsorize_panel(raw_carry, k=winsorize_k)

    z_trend = zscore_panel(wins_trend)
    z_carry = zscore_panel(wins_carry)

    composite = blend_signals(z_trend, z_carry, trend_weight, carry_weight)

    return {
        "rebalance_dates": rebalance_dates,
        "raw_trend":  raw_trend,
        "raw_carry":  raw_carry,
        "wins_trend": wins_trend,
        "wins_carry": wins_carry,
        "z_trend":    z_trend,
        "z_carry":    z_carry,
        "composite":  composite,
    }

V2_TRAIN_MIN_MONTHS: int = 60  
V2_COEF_CAP: float = 2.0

def _fit_blend_coefficients(
    z_trend_history: pd.DataFrame,
    z_carry_history: pd.DataFrame,
    returns_history: pd.DataFrame,
) -> tuple[float, float]:
    z_t = z_trend_history.stack()
    z_c = z_carry_history.stack()
    y   = returns_history.stack()
    panel = pd.concat([z_t, z_c, y], axis=1, keys=["zt", "zc", "y"]).dropna()

    if len(panel) < 30:
        return 0.5, 0.5
    
    X = np.column_stack([np.ones(len(panel)), panel["zt"].values, panel["zc"].values])
    y_vec = panel["y"].values
    coefs, *_ = np.linalg.lstsq(X, y_vec, rcond=None)

    beta_trend = float(np.clip(coefs[1], -V2_COEF_CAP, V2_COEF_CAP))
    beta_carry = float(np.clip(coefs[2], -V2_COEF_CAP, V2_COEF_CAP))
    return beta_trend, beta_carry


def compute_composite_signal_v2(
    prices, equity_ey, bond_yields, rf_daily, futures,
    risk_assets=RISK_ASSETS,
    trend_lookback=TREND_LOOKBACK_DAYS,
    winsorize_k=WINSORIZE_K,
    train_min_months=V2_TRAIN_MIN_MONTHS,
) -> dict:
    rebalance_dates = month_end_rebalance_dates(prices[risk_assets])
    rf_annual = rf_daily * TRADING_DAYS_PER_YEAR

    raw_trend = compute_trend(prices[risk_assets], rebalance_dates, trend_lookback)
    raw_carry = compute_carry(
        equity_ey, bond_yields, rf_annual, futures, rebalance_dates, risk_assets,
    )
    wins_trend = winsorize_panel(raw_trend, k=winsorize_k)
    wins_carry = winsorize_panel(raw_carry, k=winsorize_k)
    z_trend = zscore_panel(wins_trend)
    z_carry = zscore_panel(wins_carry)

    # Next-month asset returns from month-end to next month-end,
    # cross-sectionally demeaned so the regression learns relative
    # outperformance rather than absolute return.
    monthly_prices = prices[risk_assets].reindex(rebalance_dates, method="ffill")
    next_month_returns_raw = monthly_prices.pct_change().shift(-1)
    next_month_returns = next_month_returns_raw.sub(
        next_month_returns_raw.mean(axis=1), axis=0,
    )

    composite = pd.DataFrame(np.nan, index=rebalance_dates, columns=risk_assets)
    learned_coefs = pd.DataFrame(np.nan, index=rebalance_dates,
                                 columns=["beta_trend", "beta_carry"])

    for i, t in enumerate(rebalance_dates):
        if i < train_min_months:
            continue
        train_dates = rebalance_dates[:i]
        beta_t, beta_c = _fit_blend_coefficients(
            z_trend.loc[train_dates],
            z_carry.loc[train_dates],
            next_month_returns.loc[train_dates],
            )
        learned_coefs.loc[t] = [beta_t, beta_c]

        zt_t = z_trend.loc[t]
        zc_t = z_carry.loc[t]
        both = zt_t.notna() & zc_t.notna()
        only_t = zt_t.notna() & zc_t.isna()
        row = pd.Series(np.nan, index=risk_assets)
        row[both] = beta_t * zt_t[both] + beta_c * zc_t[both]
        row[only_t] = beta_t * zt_t[only_t]   # only trend, weighted by its learned beta
        composite.loc[t] = row

    return {
        "rebalance_dates": rebalance_dates,
        "raw_trend": raw_trend, "raw_carry": raw_carry,
        "wins_trend": wins_trend, "wins_carry": wins_carry,
        "z_trend": z_trend, "z_carry": z_carry,
        "composite": composite,
        "learned_coefs": learned_coefs,
        "next_month_returns": next_month_returns,
    }


# ---- Smoke test ----

if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    from src.data_loader import load_all

    data = load_all()
    result = compute_composite_signal(
        prices=data["prices"],
        equity_ey=data["equity_ey"],
        bond_yields=data["bond_yields"],
        rf_daily=data["rf"],
        futures=data["futures"],
    )

    print(f"Rebalance dates: {len(result['rebalance_dates'])} "
          f"(first: {result['rebalance_dates'][0].date()}, "
          f"last: {result['rebalance_dates'][-1].date()})")
    print()

    
    sample_date = pd.Timestamp("2020-12-31")
   
    available = result["rebalance_dates"]
    nearest = available[available <= sample_date][-1]

    print(f"=== Snapshot on {nearest.date()} ===")
    snapshot = pd.DataFrame({
        "raw_trend":  result["raw_trend"].loc[nearest],
        "raw_carry":  result["raw_carry"].loc[nearest],
        "z_trend":    result["z_trend"].loc[nearest],
        "z_carry":    result["z_carry"].loc[nearest],
        "composite":  result["composite"].loc[nearest],
    })
    print(snapshot.to_string(float_format="{:+.3f}".format))
    print()

    # v2 smoke test
    print("\n=== v2: walk-forward learned blend ===")
    result_v2 = compute_composite_signal_v2(
        prices=data["prices"], equity_ey=data["equity_ey"],
        bond_yields=data["bond_yields"], rf_daily=data["rf"],
        futures=data["futures"],
    )

    coefs = result_v2["learned_coefs"].dropna()
    print(f"Learned coefs computed for {len(coefs)} rebalance dates "
          f"(from {coefs.index[0].date()} to {coefs.index[-1].date()})")
    print()
    print("First 5 learned coefficients:")
    print(coefs.head().to_string(float_format="{:+.3f}".format))
    print()
    print("Last 5 learned coefficients:")
    print(coefs.tail().to_string(float_format="{:+.3f}".format))
    print()
    print("Summary statistics:")
    print(coefs.describe().to_string(float_format="{:+.3f}".format))

    # Verify z-scores have mean ~0 and std ~1 cross-sectionally
    z_trend_snap = result["z_trend"].loc[nearest].dropna()
    z_carry_snap = result["z_carry"].loc[nearest].dropna()
    print("Cross-sectional sanity check on snapshot:")
    print(f"  z_trend  mean = {z_trend_snap.mean():+.4f}, std = {z_trend_snap.std(ddof=0):.4f}")
    print(f"  z_carry  mean = {z_carry_snap.mean():+.4f}, std = {z_carry_snap.std(ddof=0):.4f}")
    print()

    # Snapshot of carry NaN pattern over time
    carry = result["raw_carry"]
    print("Carry availability (% of rebalance dates where signal is present):")
    avail = (carry.notna().sum() / len(carry) * 100).round(1)
    print(avail.to_string(float_format="{:.1f}%".format))