"""S&P 500 price download and caching via yfinance."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from astro_market.config import PROJECT_ROOT, load_config


class DataFetchError(RuntimeError):
    """Raised when market data cannot be fetched and no cache exists."""


def _cache_path(cfg: dict | None = None) -> Path:
    cfg = cfg or load_config()
    return Path(cfg["asset"]["cache_path"])


def fetch_and_cache_prices(
    start: str = "1970-01-01",
    end: str | None = None,
    force: bool = False,
    cfg: dict | None = None,
) -> Path:
    """
    Download ^GSPC adjusted close via yfinance and cache to parquet.

    Note: adj close is a convenient proxy; a true total-return series
    (reinvested dividends) would be preferable for long-horizon BH
    comparisons.
    """
    cfg = cfg or load_config()
    path = _cache_path(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists() and not force:
        return path

    try:
        import yfinance as yf
    except ImportError as e:
        raise DataFetchError(
            "yfinance is required to fetch prices. Install with: pip install yfinance"
        ) from e

    ticker = cfg["asset"]["ticker"]
    try:
        raw = yf.download(
            ticker,
            start=start,
            end=end,
            auto_adjust=False,
            progress=False,
            threads=False,
        )
    except Exception as e:  # network / API failures
        raise DataFetchError(
            f"Failed to download {ticker} from yfinance: {e}. "
            "Check network access or place a cached parquet at "
            f"{path}."
        ) from e

    if raw is None or raw.empty:
        raise DataFetchError(
            f"yfinance returned empty data for {ticker}. "
            f"Place a cached parquet at {path} to proceed offline."
        )

    # Handle MultiIndex columns from newer yfinance
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    price_col = cfg["asset"]["price_col"]
    if price_col not in raw.columns:
        # Fallback: Close if Adj Close missing
        if "Close" in raw.columns:
            price_col = "Close"
        else:
            raise DataFetchError(
                f"Neither '{cfg['asset']['price_col']}' nor 'Close' in download columns: "
                f"{list(raw.columns)}"
            )

    series = raw[price_col].astype(float).dropna()
    series.index = pd.to_datetime(series.index).tz_localize(None)
    series.name = "adj_close"
    out = series.to_frame()
    out.to_parquet(path)
    return path


def load_prices(cfg: dict | None = None, auto_fetch: bool = True) -> pd.Series:
    """
    Load cached adj-close series. Optionally fetch if cache missing.
    Returns Series named 'adj_close' with DatetimeIndex (trading days).
    """
    cfg = cfg or load_config()
    path = _cache_path(cfg)

    if not path.exists():
        if not auto_fetch:
            raise DataFetchError(
                f"No price cache at {path}. Run: python -m astro_market fetch-data"
            )
        try:
            fetch_and_cache_prices(cfg=cfg)
        except DataFetchError:
            raise
        except Exception as e:
            raise DataFetchError(
                f"Could not fetch prices and no cache at {path}: {e}"
            ) from e

    df = pd.read_parquet(path)
    if "adj_close" in df.columns:
        s = df["adj_close"]
    else:
        s = df.iloc[:, 0]
    s = s.astype(float).dropna()
    s.index = pd.to_datetime(s.index).tz_localize(None)
    s.name = "adj_close"
    return s.sort_index()


def daily_returns(prices: pd.Series) -> pd.Series:
    """Simple daily percentage returns from price series."""
    return prices.pct_change().dropna()


def split_mask(
    index: pd.DatetimeIndex,
    start: str,
    end: str | None,
) -> pd.Series:
    """Boolean mask for dates in [start, end] inclusive (end=None => open)."""
    start_ts = pd.Timestamp(start)
    if end is None or (isinstance(end, float) and pd.isna(end)):
        return (index >= start_ts)
    end_ts = pd.Timestamp(end)
    return (index >= start_ts) & (index <= end_ts)


def get_split_series(
    series: pd.Series,
    split_name: str,
    cfg: dict | None = None,
) -> pd.Series:
    """Slice a Series to a named protocol split (train/validation/holdout)."""
    cfg = cfg or load_config()
    start, end = cfg["splits"][split_name]
    mask = split_mask(series.index, start, end)
    return series.loc[mask]


# ---------------------------------------------------------------------------
# Nasdaq (secondary index) — sweep v3 cross-asset filter
# ---------------------------------------------------------------------------

DEFAULT_NASDAQ_CACHE = "data/ixic_adj_close.parquet"
DEFAULT_NASDAQ_TICKER = "^IXIC"


def _nasdaq_cache_path(cfg: dict | None = None) -> Path:
    cfg = cfg or load_config()
    raw = (
        cfg.get("nasdaq", {}).get("cache_path")
        or cfg.get("asset_nasdaq", {}).get("cache_path")
        or DEFAULT_NASDAQ_CACHE
    )
    p = Path(raw)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    return p


def fetch_and_cache_nasdaq(
    start: str = "1970-01-01",
    end: str | None = None,
    force: bool = False,
    cfg: dict | None = None,
) -> Path:
    """Download ^IXIC adjusted close via yfinance and cache to parquet."""
    cfg = cfg or load_config()
    path = _nasdaq_cache_path(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists() and not force:
        return path

    try:
        import yfinance as yf
    except ImportError as e:
        raise DataFetchError(
            "yfinance is required to fetch Nasdaq. Install with: pip install yfinance"
        ) from e

    ticker = (
        cfg.get("nasdaq", {}).get("ticker")
        or cfg.get("asset_nasdaq", {}).get("ticker")
        or DEFAULT_NASDAQ_TICKER
    )
    try:
        raw = yf.download(
            ticker,
            start=start,
            end=end,
            auto_adjust=False,
            progress=False,
            threads=False,
        )
    except Exception as e:
        raise DataFetchError(
            f"Failed to download {ticker} from yfinance: {e}. "
            f"Check network or place a cached parquet at {path}."
        ) from e

    if raw is None or raw.empty:
        raise DataFetchError(
            f"yfinance returned empty data for {ticker}. "
            f"Place a cached parquet at {path} to proceed offline."
        )

    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    price_col = (
        cfg.get("nasdaq", {}).get("price_col")
        or cfg.get("asset", {}).get("price_col")
        or "Adj Close"
    )
    if price_col not in raw.columns:
        if "Close" in raw.columns:
            price_col = "Close"
        else:
            raise DataFetchError(
                f"Neither '{price_col}' nor 'Close' in Nasdaq download columns: "
                f"{list(raw.columns)}"
            )

    series = raw[price_col].astype(float).dropna()
    series.index = pd.to_datetime(series.index).tz_localize(None)
    series.name = "adj_close"
    out = series.to_frame()
    out.to_parquet(path)
    return path


def load_nasdaq_prices(cfg: dict | None = None, auto_fetch: bool = True) -> pd.Series:
    """Load cached Nasdaq adj-close; optionally fetch if missing."""
    cfg = cfg or load_config()
    path = _nasdaq_cache_path(cfg)

    if not path.exists():
        if not auto_fetch:
            raise DataFetchError(
                f"No Nasdaq cache at {path}. Run fetch or enable auto_fetch."
            )
        try:
            fetch_and_cache_nasdaq(cfg=cfg)
        except DataFetchError:
            raise
        except Exception as e:
            raise DataFetchError(
                f"Could not fetch Nasdaq and no cache at {path}: {e}"
            ) from e

    df = pd.read_parquet(path)
    if "adj_close" in df.columns:
        s = df["adj_close"]
    else:
        s = df.iloc[:, 0]
    s = s.astype(float).dropna()
    s.index = pd.to_datetime(s.index).tz_localize(None)
    s.name = "adj_close"
    return s.sort_index()


def align_returns_to_calendar(
    primary_returns: pd.Series,
    secondary_prices: pd.Series,
) -> pd.Series:
    """
    Align secondary asset returns to the primary trading calendar (inner join).

    Computes pct_change on secondary prices, then intersects with primary_returns.index.
    Result shares the primary calendar dates that exist in both.
    """
    sec_rets = daily_returns(secondary_prices)
    common = primary_returns.index.intersection(sec_rets.index)
    return sec_rets.loc[common].sort_index()


def align_multi_asset(
    spx_returns: pd.Series,
    nasdaq_prices: pd.Series,
    atoms: pd.DataFrame,
) -> tuple[pd.Series, pd.Series, pd.DataFrame]:
    """
    Intersect SPX returns, Nasdaq returns, and atoms on a shared calendar.

    Returns (spx_rets_aligned, nasdaq_rets_aligned, atoms_aligned).
    """
    ndq = align_returns_to_calendar(spx_returns, nasdaq_prices)
    common = spx_returns.index.intersection(ndq.index).intersection(atoms.index)
    common = common.sort_values()
    return (
        spx_returns.loc[common],
        ndq.loc[common],
        atoms.loc[common],
    )


def is_cross_asset_survivor(
    spx_val_excess_sharpe: float,
    nasdaq_val_excess_sharpe: float,
) -> bool:
    """True iff validation excess Sharpe > 0 on BOTH SPX and Nasdaq."""
    return (
        np.isfinite(spx_val_excess_sharpe)
        and np.isfinite(nasdaq_val_excess_sharpe)
        and spx_val_excess_sharpe > 0
        and nasdaq_val_excess_sharpe > 0
    )


