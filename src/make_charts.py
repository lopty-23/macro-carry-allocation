"""
Run the full strategy end-to-end, compute metrics, and generate charts
for the README.

Outputs (to results/):
    equity_curve.png       — strategy vs 60/40, log scale
    drawdown.png           — underwater plot
    rolling_sharpe.png     — 1-year rolling Sharpe
    weights_history.png    — stacked area of weights over time
    metrics_table.png      — summary metrics image

Also prints the summary metrics to stdout.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_loader import load_all
from src.signals import compute_composite_signal, RISK_ASSETS
from src.portfolio import construct_weights, CASH_ASSET
from src.backtest import run_backtest
from src.metrics import (
    compute_all_metrics, drawdown_series, drawdown_table,
    rolling_sharpe, TRADING_DAYS_PER_YEAR,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS = REPO_ROOT / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

# Visual style
plt.rcParams.update({
    "figure.figsize":  (11, 5.5),
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid":       True,
    "grid.alpha":      0.3,
    "font.size":       11,
})
STRAT_COLOR = "#1F4E78"
BENCH_COLOR = "#A0A0A0"


# ---------------------------------------------------------------
# Benchmark: 60/40 SPY+IEF, monthly rebalanced
# ---------------------------------------------------------------

def build_6040_benchmark(prices: pd.DataFrame, rebalance_dates: pd.DatetimeIndex) -> pd.Series:
    """60% SPY, 40% IEF, rebalanced monthly. Returns a daily return Series."""
    weights = pd.DataFrame(0.0, index=rebalance_dates, columns=["SPY", "IEF"])
    weights["SPY"] = 0.6
    weights["IEF"] = 0.4
    result = run_backtest(prices=prices, monthly_weights=weights, commission=0.0)
    return result["returns_net"]


# ---------------------------------------------------------------
# Charts
# ---------------------------------------------------------------

def plot_equity_curve(strategy: pd.Series, benchmark: pd.Series, out_path: Path) -> None:
    fig, ax = plt.subplots()
    strat_eq = (1 + strategy.fillna(0)).cumprod()
    bench_eq = (1 + benchmark.fillna(0)).cumprod()
    ax.plot(strat_eq.index, strat_eq.values, color=STRAT_COLOR, lw=1.8, label="Strategy (net)")
    ax.plot(bench_eq.index, bench_eq.values, color=BENCH_COLOR, lw=1.5, label="60/40 SPY+IEF")
    ax.set_yscale("log")
    ax.set_title("Cumulative growth of $1 (log scale)")
    ax.set_ylabel("Equity")
    ax.legend(loc="upper left", frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_drawdown(strategy: pd.Series, benchmark: pd.Series, out_path: Path) -> None:
    fig, ax = plt.subplots()
    dd_strat = drawdown_series(strategy) * 100
    dd_bench = drawdown_series(benchmark) * 100
    ax.fill_between(dd_strat.index, dd_strat.values, 0,
                    color=STRAT_COLOR, alpha=0.4, label="Strategy")
    ax.plot(dd_bench.index, dd_bench.values,
            color=BENCH_COLOR, lw=1.2, label="60/40")
    ax.set_title("Drawdown from prior peak")
    ax.set_ylabel("Drawdown (%)")
    ax.legend(loc="lower left", frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_rolling_sharpe(
    strategy: pd.Series,
    benchmark: pd.Series,
    rf: pd.Series,
    daily_weights: pd.DataFrame,
    out_path: Path,
    window: int = TRADING_DAYS_PER_YEAR,
) -> None:
    # Trim to the first date the strategy is actually deployed
    # (any non-cash asset has a non-zero weight). Avoids the
    # warmup-period artifact where rolling Sharpe explodes.
    risk_only = daily_weights.drop(columns=["BIL"], errors="ignore")
    active = (risk_only.fillna(0).abs() > 0).any(axis=1)
    if active.any():
        first_active = active.idxmax()
        strategy = strategy.loc[first_active:]
        benchmark = benchmark.loc[first_active:]
        rf = rf.loc[first_active:]

    fig, ax = plt.subplots()
    rs_strat = rolling_sharpe(strategy, rf, window)
    rs_bench = rolling_sharpe(benchmark, rf, window)
    ax.plot(rs_strat.index, rs_strat.values, color=STRAT_COLOR, lw=1.6, label="Strategy")
    ax.plot(rs_bench.index, rs_bench.values, color=BENCH_COLOR, lw=1.2, label="60/40")
    ax.axhline(0, color="black", lw=0.8, ls="--", alpha=0.5)
    ax.set_title(f"Rolling {window}-day Sharpe ratio")
    ax.set_ylabel("Sharpe")
    ax.legend(loc="lower right", frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_weights_history(daily_weights: pd.DataFrame, out_path: Path) -> None:
    """Stacked area chart of weight allocation over time."""
    # Use rebalance-date weights for cleaner visual (less noise from ffill)
    weights = daily_weights.resample("ME").last().fillna(0)
    asset_order = ["SPY", "EFA", "EEM", "IEF", "TLT", "LQD",
                   "GLD", "DBC", "VNQ", "BIL"]
    weights = weights[asset_order]

    palette = {
        "SPY": "#1F77B4", "EFA": "#2CA02C", "EEM": "#17BECF",
        "IEF": "#9467BD", "TLT": "#8C564B", "LQD": "#E377C2",
        "GLD": "#BCBD22", "DBC": "#FF7F0E", "VNQ": "#7F7F7F",
        "BIL": "#D3D3D3",
    }
    colors = [palette[c] for c in weights.columns]

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.stackplot(
        weights.index,
        weights.T.values * 100,
        labels=weights.columns,
        colors=colors, alpha=0.85,
    )
    ax.set_title("Portfolio weights over time")
    ax.set_ylabel("Weight (%)")
    ax.set_ylim(0, 100)
    ax.legend(
        loc="upper center", bbox_to_anchor=(0.5, -0.10),
        ncol=10, frameon=False, fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_metrics_table(
    strat_metrics: pd.Series,
    bench_metrics: pd.Series,
    out_path: Path,
) -> None:
    rows = [
        ("CAGR",            f"{strat_metrics['CAGR']:.2%}",            f"{bench_metrics['CAGR']:.2%}"),
        ("Volatility",      f"{strat_metrics['Vol_Annualized']:.2%}",  f"{bench_metrics['Vol_Annualized']:.2%}"),
        ("Sharpe ratio",    f"{strat_metrics['Sharpe Ratio']:.2f}",    f"{bench_metrics['Sharpe Ratio']:.2f}"),
        ("Sortino ratio",   f"{strat_metrics['Sortino Ratio']:.2f}",   f"{bench_metrics['Sortino Ratio']:.2f}"),
        ("Max drawdown",    f"{strat_metrics['Max DD']:.2%}",          f"{bench_metrics['Max DD']:.2%}"),
        ("Avg DD depth",    f"{strat_metrics['Avg DD Depth']:.2%}",    f"{bench_metrics['Avg DD Depth']:.2%}"),
        ("Calmar ratio",    f"{strat_metrics['Calmar Ratio']:.2f}",    f"{bench_metrics['Calmar Ratio']:.2f}"),
        ("Time underwater", f"{strat_metrics['Time Underwater']:.1%}", f"{bench_metrics['Time Underwater']:.1%}"),
    ]
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.axis("off")
    table = ax.table(
        cellText=rows,
        colLabels=["Metric", "Strategy", "60/40"],
        loc="center",
        cellLoc="center",
        colColours=["#1F4E78"] * 3,
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.0, 1.6)
    for i in range(3):
        cell = table[(0, i)]
        cell.set_text_props(color="white", weight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------

def main():
    print("Loading data and running strategy...")
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
    bt = run_backtest(
        prices=data["prices"],
        monthly_weights=portfolio["final_weights"],
    )

    strategy_returns = bt["returns_net"]
    rf_returns = data["rf"].reindex(strategy_returns.index).fillna(0)

    print("Building benchmark...")
    bench_rebal = signals["rebalance_dates"]
    benchmark_returns = build_6040_benchmark(data["prices"], bench_rebal)
    # Align benchmark to strategy date range
    benchmark_returns = benchmark_returns.reindex(strategy_returns.index).fillna(0)

    # ---- Metrics ----
    print("\nComputing metrics...")
    strat_metrics = compute_all_metrics(strategy_returns, rf_returns)
    bench_metrics = compute_all_metrics(benchmark_returns, rf_returns)

    summary = pd.DataFrame({"Strategy": strat_metrics, "60/40": bench_metrics})
    print("\n=== Summary metrics ===")
    print(summary.to_string(float_format="{:.4f}".format))

    print("\n=== Worst 5 strategy drawdowns (deeper than -5%) ===")
    dd_table = drawdown_table(strategy_returns, threshold=-0.05)
    print(dd_table.head().to_string(float_format="{:.4f}".format))

    # ---- Charts ----
    print("\nGenerating charts...")
    plot_equity_curve(strategy_returns, benchmark_returns,
                      RESULTS / "equity_curve.png")
    plot_drawdown(strategy_returns, benchmark_returns,
                  RESULTS / "drawdown.png")
    plot_rolling_sharpe(strategy_returns, benchmark_returns, rf_returns,
                    bt["daily_weights"], RESULTS / "rolling_sharpe.png")
    plot_weights_history(bt["daily_weights"],
                         RESULTS / "weights_history.png")
    plot_metrics_table(strat_metrics, bench_metrics,
                       RESULTS / "metrics_table.png")

    print(f"\nCharts saved to {RESULTS}/")


if __name__ == "__main__":
    main()