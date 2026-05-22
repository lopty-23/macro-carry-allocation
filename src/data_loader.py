"""
Data loader for the macro carry allocation strategy.

Parses the Bloomberg Excel workbook at data/raw/macro_carry_data.xlsx
and returns four cleaned DataFrames/Series for downstream use:

    - prices:        wide DataFrame of total-return index levels per ETF
    - equity_ey:     wide DataFrame of equity earnings yields in DECIMAL
                     (e.g. 0.045 = 4.5%), derived from index P/E ratios
    - bond_yields:   wide DataFrame of bond/credit index yields in DECIMAL
    - rf:            Series of daily risk-free returns derived from USGG3M

Bloomberg workbook format
-------------------------
Each sheet (prices, equity_pe, bond_yields, rf) is in raw BDH output
format with one (Date, Value) column-pair per ticker, side by side.
Tickers within a sheet have non-aligned date ranges due to differing
inception dates and trading calendars. This loader reconciles them.

Data conventions after loading
------------------------------
- All DataFrames are wide: DatetimeIndex with one column per asset.
- Yields and earnings yields are expressed in DECIMAL (e.g. 0.04 = 4%),
  so carry computations work without unit conversions.
- Slow-moving series (yields, P/E) are forward-filled on gaps.
- Prices are NEVER forward-filled (a missing price = market closed,
  filling would inject look-ahead).

The raw Bloomberg file is excluded from version control; see README.
"""

from importlib.resources import path
from pathlib import Path
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RAW_PATH = REPO_ROOT / "data" / "raw" / "macro_carry_data.xlsx"
DEFAULT_CACHE_DIR = REPO_ROOT / "data" / "processed"

PRICE_TICKERS: list[str] = [
    "SPY", "EFA", "EEM", "IEF", "TLT", "LQD", "GLD", "DBC", "VNQ", "BIL",
]
PE_TICKERS: list[str] = ["SPX", "MXEA", "MXEF", "RMZ"]
YIELD_TICKERS: list[str] = ["LUATTRUU", "LUTLTRUU", "LUACTRUU"]

PE_TO_ETF: dict[str, str] = {
    "SPX": "SPY", "MXEA": "EFA", "MXEF": "EEM", "RMZ": "VNQ",
}

YIELD_TO_ETF: dict[str, str] = {
    "LUATTRUU": "IEF", "LUTLTRUU": "TLT", "LUACTRUU": "LQD",
}

DEFAULT_FUTURES_PATH = REPO_ROOT / "data" / "raw" / "commodity_futures.xlsx"

COMMODITY_ROOTS: list[str] = [
    "CL", "CO", "HO", "XB", "NG", "GC", "SI",
    "LA", "LX", "LP", "C", "W", "S", "SB",
]

FUTURES_COLUMNS: list[str] = [
    f"{root}_{m}" for root in COMMODITY_ROOTS for m in ("M1", "M2")
]

TRADING_DAYS_PER_YEAR: int = 252

def _parse_multi_date_sheet(
    path: Path,
    sheet_name: str,
    tickers: list[str],
) -> pd.DataFrame:
    
    raw = pd.read_excel(path, sheet_name=sheet_name, header=1)

    series_dict: dict[str, pd.Series] = {}
    for i, ticker in enumerate(tickers):
        date_col = raw.columns[i * 2]
        val_col = raw.columns[i * 2 + 1]
        dates = pd.to_datetime(raw[date_col], errors="coerce")
        values = pd.to_numeric(raw[val_col], errors="coerce")
        s = pd.Series(values.values, index=dates).dropna()
        s = s[~s.index.duplicated(keep="first")]
        series_dict[ticker] = s.sort_index()

    return pd.concat(series_dict, axis=1).sort_index()

def load_prices(path: Path = DEFAULT_RAW_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Bloomberg workbook not found at {path}.\n"
            f"This project requires manual export from a Bloomberg terminal. "
            f"See README for the field list and instructions."
        )
    return _parse_multi_date_sheet(path, "prices", PRICE_TICKERS)

def load_equity_earnings_yields(path: Path = DEFAULT_RAW_PATH) -> pd.DataFrame:
    pe = _parse_multi_date_sheet(path, "equity_pe", PE_TICKERS)
    ey = 1.0 / pe
    ey = ey.rename(columns=PE_TO_ETF)
    return ey.ffill()

def load_bond_yields(path: Path = DEFAULT_RAW_PATH) -> pd.DataFrame:
    raw = _parse_multi_date_sheet(path, "bond_yields", YIELD_TICKERS)
    decimal = raw / 100.0
    decimal = decimal.rename(columns=YIELD_TO_ETF)
    return decimal.ffill()

def load_rf(path: Path = DEFAULT_RAW_PATH) -> pd.Series:
    raw = _parse_multi_date_sheet(path, "rf", ["USGG3M"])
    annual_decimal = raw["USGG3M"].ffill() / 100.0
    daily_rf = annual_decimal / TRADING_DAYS_PER_YEAR
    daily_rf.name = "risk_free_return"
    return daily_rf

def load_commodity_futures(path: Path = DEFAULT_FUTURES_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Commodity futures workbook not found at {path}.\n"
            f"See README for the Bloomberg field list."
        )
    futures = _parse_multi_date_sheet(path, "futures", FUTURES_COLUMNS)
    return futures.ffill()

def load_all(
    path: Path = DEFAULT_RAW_PATH,
    futures_path: Path = DEFAULT_FUTURES_PATH,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    force_refresh: bool = False,
) -> dict:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_paths = {
        "prices":      cache_dir / "prices.parquet",
        "equity_ey":   cache_dir / "equity_ey.parquet",
        "bond_yields": cache_dir / "bond_yields.parquet",
        "rf":          cache_dir / "rf.parquet",
        "futures":     cache_dir / "futures.parquet",
    }

    all_cached = all(p.exists() for p in cache_paths.values())
    if all_cached and not force_refresh:
        return {
            "prices":      pd.read_parquet(cache_paths["prices"]),
            "equity_ey":   pd.read_parquet(cache_paths["equity_ey"]),
            "bond_yields": pd.read_parquet(cache_paths["bond_yields"]),
            "rf":          pd.read_parquet(cache_paths["rf"])["risk_free_return"],
            "futures":     pd.read_parquet(cache_paths["futures"]),
        }

    prices = load_prices(path)
    equity_ey = load_equity_earnings_yields(path)
    bond_yields = load_bond_yields(path)
    rf = load_rf(path)
    futures = load_commodity_futures(futures_path)

    prices.to_parquet(cache_paths["prices"])
    equity_ey.to_parquet(cache_paths["equity_ey"])
    bond_yields.to_parquet(cache_paths["bond_yields"])
    rf.to_frame().to_parquet(cache_paths["rf"])
    futures.to_parquet(cache_paths["futures"])

    return {
        "prices": prices,
        "equity_ey": equity_ey,
        "bond_yields": bond_yields,
        "rf": rf,
        "futures": futures,
    }

# ---- Test ----

if __name__ == "__main__":
    print(f"Loading from {DEFAULT_RAW_PATH}")
    print()

    data = load_all(force_refresh=True)

    prices = data["prices"]
    print("=== Prices (total-return index) ===")
    print(f"Shape: {prices.shape}")
    print(f"Date range: {prices.index.min().date()} → {prices.index.max().date()}")
    print(f"Tickers: {list(prices.columns)}")
    print("First-valid date per ticker (ETF inception):")
    print(prices.apply(lambda c: c.first_valid_index().date()).to_string())
    print()

    ey = data["equity_ey"]
    print("=== Equity earnings yields (decimal) ===")
    print(f"Shape: {ey.shape}, Tickers: {list(ey.columns)}")
    print("Latest values:")
    print(ey.iloc[-1].to_string(float_format="{:.4f}".format))
    print()

    by = data["bond_yields"]
    print("=== Bond/credit yields (decimal) ===")
    print(f"Shape: {by.shape}, Tickers: {list(by.columns)}")
    print("Latest values:")
    print(by.iloc[-1].to_string(float_format="{:.4f}".format))
    print()

    rf = data["rf"]
    print("=== Risk-free rate (daily returns, decimal) ===")
    print(f"Shape: {rf.shape}")
    print(f"Latest daily return: {rf.iloc[-1]:.6f} "
          f"(annualized: {rf.iloc[-1] * TRADING_DAYS_PER_YEAR:.2%})")
    
    futures = data["futures"]
    print("=== Commodity futures (front + second month) ===")
    print(f"Shape: {futures.shape}")
    print(f"Commodities: {sorted(set(c.split('_')[0] for c in futures.columns))}")
    print("First-valid date per series:")
    print(futures.apply(lambda c: c.first_valid_index().date()).to_string())
    print()
    print("Latest prices:")
    print(futures.iloc[-1].to_string(float_format="{:.2f}".format))