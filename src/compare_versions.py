"""
Compare v1 (hand-coded 0.5/0.5 blend) vs v2 (walk-forward learned
blend) on the same evaluation window: 2008-01-31 onwards, when v2's
first valid composite appears.

Outputs to results/:
    compare_metrics.png   — side-by-side metrics table
    compare_equity.png    — overlaid equity curves
    compare_drawdown.png  — overlaid drawdowns
    compare_coefs.png     — v2 learned coefficients over time

Also prints metrics and learned-coefficient summary to stdout.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_loader import load_all
from src.signals import (
    compute_composite_signal,
    compute_composite_signal_v2,
    RISK_ASSETS,
)
from src.portfolio import construct_weights
from src.backtest import run_backtest
from src.metrics import compute_all_metrics

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS = REPO_ROOT / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "figure.figsize":  (11, 5.5),
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid":       True,
    "grid.alpha":      0.3,
    "font.size":       11,
})
V1_COLOR = "#A0A0A0"
V2_COLOR = "#1F4E78"


def run_strategy(composite_dict: dict, prices: pd.DataFrame) -> dict:
    portfolio = construct_weights(
        composite=composite_dict["composite"],
        prices=prices,
        risk_assets=RISK_ASSETS,
    )
    return run_backtest(prices=prices, monthly_weights=portfolio["final_weights"])


def restrict_to_window(returns: pd.Series, start: pd.Timestamp) -> pd.Series:
    return returns.loc[start:].copy()


def plot_metrics_compare(v1_metrics, v2_metrics, out_path):
    rows = [
        ("CAGR",            f"{v1_metrics['CAGR']:.2%}",            f"{v2_metrics['CAGR']:.2%}"),
        ("Volatility",      f"{v1_metrics['Vol_Annualized']:.2%}",  f"{v2_metrics['Vol_Annualized']:.2%}"),
        ("Sharpe ratio",    f"{v1_metrics['Sharpe Ratio']:.2f}",    f"{v2_metrics['Sharpe Ratio']:.2f}"),
        ("Sortino ratio",   f"{v1_metrics['Sortino Ratio']:.2f}",   f"{v2_metrics['Sortino Ratio']:.2f}"),
        ("Max drawdown",    f"{v1_metrics['Max DD']:.2%}",          f"{v2_metrics['Max DD']:.2%}"),
        ("Avg DD depth",    f"{v1_metrics['Avg DD Depth']:.2%}",    f"{v2_metrics['Avg DD Depth']:.2%}"),
        ("Calmar ratio",    f"{v1_metrics['Calmar Ratio']:.2f}",    f"{v2_metrics['Calmar Ratio']:.2f}"),
        ("Time underwater", f"{v1_metrics['Time Underwater']:.1%}", f"{v2_metrics['Time Underwater']:.1%}"),
    ]
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.axis("off")
    table = ax.table(
        cellText=rows,
        colLabels=["Metric", "v1 (0.5/0.5)", "v2 (learned)"],
        loc="center", cellLoc="center",
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


def plot_equity_compare(v1_returns, v2_returns, out_path):
    fig, ax = plt.subplots()
    v1_eq = (1 + v1_returns.fillna(0)).cumprod()
    v2_eq = (1 + v2_returns.fillna(0)).cumprod()
    ax.plot(v1_eq.index, v1_eq.values, color=V1_COLOR, lw=1.5, label="v1 (0.5/0.5)")
    ax.plot(v2_eq.index, v2_eq.values, color=V2_COLOR, lw=1.8, label="v2 (learned)")
    ax.set_yscale("log")
    ax.set_title("Cumulative growth of $1 (log scale) — v1 vs v2")
    ax.set_ylabel("Equity")
    ax.legend(loc="upper left", frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_drawdown_compare(v1_returns, v2_returns, out_path):
    from src.metrics import drawdown_series
    fig, ax = plt.subplots()
    dd_v1 = drawdown_series(v1_returns) * 100
    dd_v2 = drawdown_series(v2_returns) * 100
    ax.plot(dd_v1.index, dd_v1.values, color=V1_COLOR, lw=1.5, label="v1")
    ax.fill_between(dd_v2.index, dd_v2.values, 0,
                    color=V2_COLOR, alpha=0.4, label="v2")
    ax.set_title("Drawdown from prior peak — v1 vs v2")
    ax.set_ylabel("Drawdown (%)")
    ax.legend(loc="lower left", frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_learned_coefs(coefs_df, out_path):
    fig, ax = plt.subplots()
    ax.plot(coefs_df.index, coefs_df["beta_trend"],
            color="#1F4E78", lw=1.5, label="β_trend")
    ax.plot(coefs_df.index, coefs_df["beta_carry"],
            color="#A0522D", lw=1.5, label="β_carry")
    ax.axhline(0, color="black", lw=0.8, ls="--", alpha=0.5)
    ax.set_title("v2 learned coefficients over time (walk-forward OLS)")
    ax.set_ylabel("Coefficient")
    ax.legend(loc="upper right", frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    print("Loading data...")
    data = load_all()

    print("Computing v1 composite...")
    v1_composite = compute_composite_signal(
        prices=data["prices"], equity_ey=data["equity_ey"],
        bond_yields=data["bond_yields"], rf_daily=data["rf"],
        futures=data["futures"],
    )

    print("Computing v2 composite (walk-forward, ~30 sec)...")
    v2_composite = compute_composite_signal_v2(
        prices=data["prices"], equity_ey=data["equity_ey"],
        bond_yields=data["bond_yields"], rf_daily=data["rf"],
        futures=data["futures"],
    )

    print("Running backtests...")
    v1_bt = run_strategy(v1_composite, data["prices"])
    v2_bt = run_strategy(v2_composite, data["prices"])

    # Define the common evaluation window: v2's first valid composite + 1 month
    eval_start = v2_composite["learned_coefs"].dropna().index[0] + pd.offsets.MonthBegin(1)
    print(f"\nEvaluation window starts {eval_start.date()} (v2 first deployment)")

    v1_returns = restrict_to_window(v1_bt["returns_net"], eval_start)
    v2_returns = restrict_to_window(v2_bt["returns_net"], eval_start)
    rf_returns = data["rf"].reindex(v1_returns.index).fillna(0)

    v1_metrics = compute_all_metrics(v1_returns, rf_returns)
    v2_metrics = compute_all_metrics(v2_returns, rf_returns)

    summary = pd.DataFrame({"v1 (0.5/0.5)": v1_metrics, "v2 (learned)": v2_metrics})
    summary["Δ (v2 - v1)"] = summary["v2 (learned)"] - summary["v1 (0.5/0.5)"]
    print("\n=== v1 vs v2 metrics (same window) ===")
    print(summary.to_string(float_format="{:.4f}".format))

    # Learned-coefficient summary
    coefs = v2_composite["learned_coefs"].dropna()
    print("\n=== v2 learned coefficient summary ===")
    print(coefs.describe().to_string(float_format="{:+.4f}".format))
    print(f"\nMean ratio β_carry / β_trend: {(coefs['beta_carry'] / coefs['beta_trend']).mean():.3f}")
    print(f"(v1 fixed ratio: 1.0)")

    print("\nGenerating charts...")
    plot_metrics_compare(v1_metrics, v2_metrics, RESULTS / "compare_metrics.png")
    plot_equity_compare(v1_returns, v2_returns, RESULTS / "compare_equity.png")
    plot_drawdown_compare(v1_returns, v2_returns, RESULTS / "compare_drawdown.png")
    plot_learned_coefs(coefs, RESULTS / "compare_coefs.png")
    print(f"Charts saved to {RESULTS}/")


if __name__ == "__main__":
    main()