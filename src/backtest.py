"""
Backtest engine for the macro carry allocation strategy.

Simulates daily P&L of a portfolio holding the monthly weights from
portfolio.construct_weights(). Weights produced on month-end t are
held from t+1 (next business day open) onward — one-day lag to avoid
look-ahead. Transaction cost applied on the execution day, equal to
turnover × commission (default 5bps for liquid US ETFs through IB).
"""

import numpy as np
import pandas as pd

DEFAULT_COMMISSION: float = 0.0005   # 5 bps round-trip

def run_backtest(
    prices: pd.DataFrame,
    monthly_weights: pd.DataFrame,
    commission: float = DEFAULT_COMMISSION,
    start_date: pd.Timestamp = None,
) -> dict:
    """
    Daily backtest of monthly-rebalanced weights.

    Returns a dict with daily gross/net returns, daily weights held,
    monthly turnover, daily costs, and cumulative cost drag.
    """
    assets = list(monthly_weights.columns)
    asset_returns = prices[assets].pct_change()

    if start_date is None:
        start_date = monthly_weights.dropna(how="all").index.min()
    daily_index = asset_returns.loc[start_date:].index

    # Forward-fill monthly weights onto daily calendar, then lag by 1
    # so new weights take effect the day after rebalance
    daily_weights = (
        monthly_weights
        .reindex(daily_index, method="ffill")
        .shift(1)
    )

    returns_gross = (daily_weights.fillna(0.0) * asset_returns).sum(axis=1)

    # Turnover and cost per rebalance
    rebal_dates = monthly_weights.index.intersection(daily_index)
    prev_weights = monthly_weights.shift(1).loc[rebal_dates].fillna(0.0)
    new_weights = monthly_weights.loc[rebal_dates].fillna(0.0)
    weight_changes = (new_weights - prev_weights).abs().sum(axis=1)
    turnover = weight_changes / 2.0   # one-way turnover

    # Charge cost on the execution day (business day after rebalance)
    costs = pd.Series(0.0, index=daily_index)
    for rebal_date in rebal_dates:
        next_days = daily_index[daily_index > rebal_date]
        if len(next_days) == 0:
            continue
        costs.loc[next_days[0]] = weight_changes.loc[rebal_date] * commission

    returns_net = returns_gross - costs

    return {
        "returns_gross": returns_gross,
        "returns_net":   returns_net,
        "daily_weights": daily_weights,
        "turnover":      turnover,
        "costs":         costs,
        "cum_costs":     costs.cumsum(),
    }
# ---- Smoke test ----

if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    from src.data_loader import load_all
    from src.signals import compute_composite_signal, RISK_ASSETS
    from src.portfolio import construct_weights

    data = load_all()
    signals = compute_composite_signal(
        prices=data["prices"], equity_ey=data["equity_ey"],
        bond_yields=data["bond_yields"], rf_daily=data["rf"],
        futures=data["futures"],
    )
    portfolio = construct_weights(
        composite=signals["composite"], prices=data["prices"],
        risk_assets=RISK_ASSETS,
    )
    result = run_backtest(
        prices=data["prices"], monthly_weights=portfolio["final_weights"],
    )

    gross, net, turn = result["returns_gross"], result["returns_net"], result["turnover"]
    n_days = (gross.dropna() != 0).sum()
    n_years = n_days / 252

    cagr_gross = (1 + gross.fillna(0)).prod() ** (1 / n_years) - 1
    cagr_net   = (1 + net.fillna(0)).prod()   ** (1 / n_years) - 1
    realized_vol = net.dropna().std() * np.sqrt(252)
    sharpe = cagr_net / realized_vol
    max_dd = ((1 + net.fillna(0)).cumprod() /
              (1 + net.fillna(0)).cumprod().cummax() - 1).min()

    print(f"Backtest period: {gross.index[0].date()} → {gross.index[-1].date()}")
    print(f"Active days: {n_days} ({n_years:.1f} years)\n")

    print(f"CAGR (gross): {cagr_gross:.2%}")
    print(f"CAGR (net):   {cagr_net:.2%}")
    print(f"Cost drag:    {cagr_gross - cagr_net:.2%}")
    print(f"Realized vol: {realized_vol:.2%}  (target: 10.00%)")
    print(f"Sharpe:       {sharpe:.2f}")
    print(f"Max drawdown: {max_dd:.2%}\n")

    print(f"Avg monthly turnover:    {turn.mean():.2%}")
    print(f"Median monthly turnover: {turn.median():.2%}")
    print(f"Total cost drag:         {result['cum_costs'].iloc[-1]:.2%} of NAV")