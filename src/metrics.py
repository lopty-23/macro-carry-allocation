"""
Performance metrics for backtest evaluation.

Pure functions that operate on daily return series. No dependencies on
other project modules — these are reusable across any backtest. 
This file is copied from the SMA-crossover backtest with
three additions specific to this project's diagnostic needs:
sortino_ratio, rolling_sharpe, and drawdown_table.

Typical usage:
    >>> from src.metrics import compute_all_metrics
    >>> stats = compute_all_metrics(strategy_returns, rf=rf_returns)
    >>> print(stats)

Conventions:
- Returns are expressed as decimals (0.01 = 1%, not 1).
- Annualization assumes 252 trading days per year.
- Risk-free rate (rf) is a daily returns Series, aligned (or alignable)
  to the strategy returns' date index. The project's standard source
  is Bloomberg's USGG3M Index, loaded via src.data_loader.load_risk_free_rate.
"""

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR: int = 252

def annualized_return(returns: pd.Series) -> float:
    total_return = (1 + returns.fillna(0)).prod()
    n_periods = len(returns.dropna())
    if n_periods == 0:
        return float("nan")
    return total_return ** (TRADING_DAYS_PER_YEAR / n_periods) - 1

def annualized_vol(returns: pd.Series) -> float:
    return returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)

def sharpe_ratio(returns: pd.Series, rf: pd.Series) -> float:
    aligned_rf = rf.reindex(returns.index).fillna(0)
    excess = returns - aligned_rf
    sigma = excess.std()
    if sigma == 0 or np.isnan(sigma):
        return float("nan")
    return excess.mean() / sigma * np.sqrt(TRADING_DAYS_PER_YEAR)

def drawdown_series(returns: pd.Series) -> pd.Series:
    equity = (1 + returns.fillna(0)).cumprod()
    running_max = equity.cummax()
    return (equity - running_max) / running_max

def max_drawdown(returns: pd.Series) -> float:
    return drawdown_series(returns).min()

def avg_drawdown_depth(returns: pd.Series) -> float:
    dd = drawdown_series(returns)
    return dd[dd < 0].mean()

def time_underwater(returns: pd.Series) -> float:
    dd = drawdown_series(returns)
    return (dd < 0).mean()

def calmar_ratio(returns: pd.Series) -> float:
    cagr = annualized_return(returns)
    mdd = max_drawdown(returns)
    if mdd == 0 or np.isnan(mdd):
        return float("nan")
    return cagr / abs(mdd)

def drawdown_details(returns: pd.Series) -> dict:
    equity = (1 + returns.fillna(0)).cumprod()
    runing_max = equity.cummax()
    dd = (equity - runing_max) / runing_max

    worst_idx = dd.idxmin()
    worst_value = dd.min()

    peak_idx = equity[:worst_idx].idxmax()

    after_trough = equity.loc[worst_idx:]
    recovery_mask = after_trough >= equity[peak_idx]
    recovery_idx = after_trough[recovery_mask].index[0] if recovery_mask.any() else pd.NaT

    return {
        "max_drawdown":       worst_value,
        "peak_date":          peak_idx,
        "trough_date":        worst_idx,
        "recovery_date":      recovery_idx,
        "time_underwater":    time_underwater(returns),
        "avg_drawdown_depth": avg_drawdown_depth(returns),
    }

def compute_all_metrics(returns: pd.Series, rf: pd.Series) -> pd.Series:
    return pd.Series({
        "CAGR": annualized_return(returns),
        "Vol_Annualized": annualized_vol(returns),
        "Sharpe Ratio": sharpe_ratio(returns, rf),
        "Sortino Ratio": sortino_ratio(returns, rf),
        "Max DD": max_drawdown(returns),
        "Avg DD Depth":    avg_drawdown_depth(returns),
        "Calmar Ratio": calmar_ratio(returns),
        "Time Underwater": time_underwater(returns),
    })

def sortino_ratio(returns: pd.Series, rf: pd.Series, mar: float = 0.0) -> float:
    """
    Sortino ratio = excess return / downside deviation. 
    Like Sharpe but penalizes only downside vol. 
    """
    aligned_rf = rf.reindex(returns.index).fillna(0)
    excess = returns - aligned_rf
    downside = excess[excess < mar]
    if len(downside) == 0:
        return float("nan")
    downside_vol = downside.std()
    if downside_vol == 0 or np.isnan(downside_vol):
        return float("nan")
    return excess.mean() / downside_vol * np.sqrt(TRADING_DAYS_PER_YEAR)


def rolling_sharpe(
    returns: pd.Series,
    rf: pd.Series,
    window: int = TRADING_DAYS_PER_YEAR,
) -> pd.Series:
    aligned_rf = rf.reindex(returns.index).fillna(0)
    excess = returns - aligned_rf
    rolling_mean = excess.rolling(window).mean()
    rolling_std = excess.rolling(window).std()
    return rolling_mean / rolling_std * np.sqrt(TRADING_DAYS_PER_YEAR)


def drawdown_table(returns: pd.Series, threshold: float = -0.05) -> pd.DataFrame:
    """
    All drawdowns deeper than `threshold` (default -5%), sorted by depth.

    Each row gives the peak date, trough date, recovery date (NaT if
    not yet recovered), depth, and duration in trading days from peak
    to recovery.
    """
    dd = drawdown_series(returns)
    in_dd = dd < 0
    starts, ends = [], []
    in_episode = False
    for date, underwater in in_dd.items():
        if underwater and not in_episode:
            starts.append(date)
            in_episode = True
        elif not underwater and in_episode:
            ends.append(date)
            in_episode = False
    if in_episode:
        ends.append(pd.NaT)

    rows = []
    for start, end in zip(starts, ends):
        episode = dd.loc[start:end] if pd.notna(end) else dd.loc[start:]
        trough_date = episode.idxmin()
        depth = episode.min()
        if depth > threshold:
            continue
        equity = (1 + returns.fillna(0)).cumprod()
        peak_date = equity.loc[:start].idxmax()
        duration = ((end - peak_date).days
                    if pd.notna(end) else (returns.index[-1] - peak_date).days)
        rows.append({
            "peak_date":     peak_date,
            "trough_date":   trough_date,
            "recovery_date": end,
            "depth":         depth,
            "duration_days": duration,
        })
    df = pd.DataFrame(rows)
    if len(df) == 0:
        return df
    return df.sort_values("depth").reset_index(drop=True)
    
# ---- Smoke test ----

if __name__ == "__main__":
    np.random.seed(42)
    n_days = TRADING_DAYS_PER_YEAR * 3
    dates = pd.date_range("2020-01-01", periods=n_days, freq="B")

    fake_returns = pd.Series(
        np.random.normal(0.0005, 0.01, n_days),
        index=dates,
    )
    fake_rf = pd.Series(
        np.full(n_days, 0.02 / TRADING_DAYS_PER_YEAR),
        index=dates,
    )

    print("=== Summary metrics ===")
    print(compute_all_metrics(fake_returns, fake_rf).to_string(float_format="{:.4f}".format))
    print()

    print("=== Rolling 1-year Sharpe (last 5 values) ===")
    rs = rolling_sharpe(fake_returns, fake_rf)
    print(rs.dropna().tail().to_string(float_format="{:.4f}".format))
    print()

    print("=== Drawdown table (worse than -2%) ===")
    dd_table = drawdown_table(fake_returns, threshold=-0.02)
    print(dd_table.to_string(float_format="{:.4f}".format))

