"""
Portfolio construction for the macro carry allocation strategy.

Translates composite signals into actual position weights via a
deterministic seven-step chain:

    1. Composite signal     →  cross-sectional z-scores per asset
    2. Rank & select top 4  →  keep highest-scoring assets
    3. Signal-proportional  →  weights = z / sum(z), sums to 1
    4. Inverse-vol scaling  →  divide by asset vol, renormalize
    5. Portfolio vol target →  scale by target / forecast portfolio vol
    6. Cap at 100% gross    →  no leverage used
    7. Cash sleeve          →  BIL absorbs the unused capital

Step 4 ensures each held asset contributes roughly equal risk to the
portfolio. Step 5-6 ensure total portfolio vol stays near the target, 
or below when the target would require
leverage we don't take. Step 7 puts the unused capital into BIL.

All volatility estimates are backward-looking realized vol from a
rolling window of past daily returns. No look-ahead.
"""

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR: int = 252
TOP_N: int = 4
VOL_LOOKBACK_DAYS: int = 63
TARGET_PORTFOLIO_VOL: float = 0.10
GROSS_LEVERAGE_CAP: float = 1.0
CASH_ASSET: str = "BIL"

def select_top_n(
    composite: pd.DataFrame,
    n: int = TOP_N,
) -> pd.DataFrame:
    ranked = composite.rank(axis=1, ascending=False, method="first")
    keep_mask = ranked <= n
    return composite.where(keep_mask, np.nan)

def signal_proportional_weights(selected: pd.DataFrame) -> pd.DataFrame:
    bullish = selected.clip(lower=0.0)
    row_sums = bullish.sum(axis=1)
    weights = bullish.div(row_sums.where(row_sums > 0), axis=0)
    return weights.fillna(0.0)

def rolling_annualized_vol(
    prices: pd.DataFrame,
    lookback: int = VOL_LOOKBACK_DAYS,
) -> pd.DataFrame:
    log_returns = np.log(prices / prices.shift(1))
    return log_returns.rolling(window=lookback, min_periods=lookback).std() * np.sqrt(TRADING_DAYS_PER_YEAR)

def apply_inverse_vol_scaling(
    raw_weights: pd.DataFrame,
    asset_vols_on_rebal: pd.DataFrame,
    floor_vol: float = 0.02,
) -> pd.DataFrame:
    safe_vols = asset_vols_on_rebal.clip(lower=floor_vol)
    risk_weighted = raw_weights.div(safe_vols)
    row_sums = risk_weighted.sum(axis=1)
    return risk_weighted.div(row_sums.where(row_sums > 0), axis=0).fillna(0.0)

def forecast_portfolio_vol(
    weights: pd.DataFrame,
    returns: pd.DataFrame,
    rebalance_dates: pd.DatetimeIndex,
    lookback: int = VOL_LOOKBACK_DAYS,
) -> pd.Series:
    forecasts = pd.Series(np.nan, index=rebalance_dates, dtype=float)

    for rebal_date in rebalance_dates:
        w = weights.loc[rebal_date]
        if w.sum() == 0:
            forecasts.loc[rebal_date] = 0.0
            continue

        history = returns.loc[:rebal_date].tail(lookback)
        if len(history) < lookback:
            continue

        port_returns = (history[w.index] * w).sum(axis=1)
        forecasts.loc[rebal_date] = port_returns.std(ddof=0) * np.sqrt(TRADING_DAYS_PER_YEAR)

    return forecasts

def apply_vol_target(
    risk_weights: pd.DataFrame,
    portfolio_vol_forecast: pd.Series,
    target_vol: float = TARGET_PORTFOLIO_VOL,
    leverage_cap: float = GROSS_LEVERAGE_CAP,
) -> pd.DataFrame:
    scaling = target_vol / portfolio_vol_forecast.replace(0.0, np.nan)
    scaling = scaling.clip(upper=leverage_cap / risk_weights.sum(axis=1).replace(0.0, np.nan))
    scaling = scaling.fillna(0.0)
    return risk_weights.mul(scaling, axis=0)

def add_cash_sleeve(
    risk_weights: pd.DataFrame,
    cash_asset: str = CASH_ASSET,
) -> pd.DataFrame:
    if cash_asset in risk_weights.columns:
        raise ValueError(f"{cash_asset} is already in risk_weights")
    weights = risk_weights.copy()
    weights[cash_asset] = 1.0 - weights.sum(axis=1)
    return weights

def construct_weights(
    composite: pd.DataFrame,
    prices: pd.DataFrame,
    risk_assets: list[str] = None,
    n: int = TOP_N,
    vol_lookback: int = VOL_LOOKBACK_DAYS,
    target_vol: float = TARGET_PORTFOLIO_VOL,
    leverage_cap: float = GROSS_LEVERAGE_CAP,
) -> dict:
    if risk_assets is None:
        risk_assets = list(composite.columns)
    rebalance_dates = composite.index

    returns = prices[risk_assets].pct_change()

    selected = select_top_n(composite, n=n)

    raw_weights = signal_proportional_weights(selected)

    asset_vols_daily = rolling_annualized_vol(prices[risk_assets], lookback=vol_lookback)
    asset_vols_on_rebal = asset_vols_daily.reindex(rebalance_dates, method="ffill")
    risk_weights = apply_inverse_vol_scaling(raw_weights, asset_vols_on_rebal)

    vol_forecast = forecast_portfolio_vol(
        risk_weights, returns, rebalance_dates, lookback=vol_lookback,
    )
    scaled_risk_weights = apply_vol_target(
        risk_weights, vol_forecast, target_vol=target_vol, leverage_cap=leverage_cap,
    )

    final_weights = add_cash_sleeve(scaled_risk_weights, cash_asset=CASH_ASSET)

    return {
        "selected": selected,
        "raw_weights": raw_weights,
        "asset_vols": asset_vols_on_rebal,
        "risk_weights_pre_target": risk_weights,
        "vol_forecast": vol_forecast,
        "scaled_risk_weights": scaled_risk_weights,
        "final_weights": final_weights,
    }

#Test
if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    from src.data_loader import load_all
    from src.signals import compute_composite_signal, RISK_ASSETS

    data = load_all()
    signals = compute_composite_signal(
        prices=data["prices"],
        equity_ey=data["equity_ey"],
        bond_yields=data["bond_yields"],
        rf_daily=data["rf"],
        futures=data["futures"],
    )

    portfolio = construct_weights(
        composite=signals["composite"],
        prices=data["prices"],
        risk_assets=RISK_ASSETS,
    )
    
    final = portfolio["final_weights"]
    print(f"Final weights shape: {final.shape}")
    print(f"Columns: {list(final.columns)}")
    print(f"Date range: {final.index[0].date()} → {final.index[-1].date()}")

    sample = pd.Timestamp("2020-12-31")
    nearest = final.index[final.index <= sample][-1]
    print(f"=== Snapshot on {nearest.date()} ===")
    snapshot = pd.DataFrame({
        "composite":     signals["composite"].loc[nearest],
        "selected":      portfolio["selected"].loc[nearest],
        "raw_weight":    portfolio["raw_weights"].loc[nearest],
        "asset_vol":     portfolio["asset_vols"].loc[nearest],
        "risk_weight":   portfolio["risk_weights_pre_target"].loc[nearest],
        "scaled":        portfolio["scaled_risk_weights"].loc[nearest],
    })
    print(snapshot.to_string(float_format="{:+.4f}".format))
    
    final = portfolio["final_weights"]
    print(f"Final weights shape: {final.shape}")
    print(f"Columns: {list(final.columns)}")
    print(f"Date range: {final.index[0].date()} → {final.index[-1].date()}")

    final_row = final.loc[nearest]
    print(final_row[final_row != 0].to_string(float_format="{:.2%}".format))
    print(f"Sum: {final_row.sum():.4f}")

    print("=== Average behavior over full sample ===")
    risk_exposure = (final.drop(columns=CASH_ASSET).sum(axis=1))
    cash_exposure = final[CASH_ASSET]
    print(f"Mean risk asset exposure: {risk_exposure.mean():.1%}")
    print(f"Mean cash exposure: {cash_exposure.mean():.1%}")
    print(f"Mean portfolio vol forecast: {portfolio['vol_forecast'].mean():.2%}")
    print(f"Median portfolio vol forecast: {portfolio['vol_forecast'].median():.2%}")
    print(f"# rebalance dates with non-zero allocation: " f"{(risk_exposure > 0).sum()} / {len(risk_exposure)}")