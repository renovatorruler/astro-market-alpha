"""FX daily OHLC download and caching via yfinance."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from astro_market.config import PROJECT_ROOT, load_config, resolve_path

# Majors (daily). yfinance tickers append =X.
FX_PAIRS: tuple[str, ...] = (
    "EURUSD",
    "GBPUSD",
    "USDJPY",
    "AUDUSD",
    "USDCAD",
    "USDCHF",
    "NZDUSD",
)

YF_SUFFIX = "=X"


class FXDataError(RuntimeError):
    """Raised when FX data cannot be fetched and no cache exists."""


def load_fx_config(path: Path | str | None = None) -> dict:
    """Load configs/fx.yaml (or override path)."""
    cfg_path = Path(path) if path else (PROJECT_ROOT / "configs" / "fx.yaml")
    return load_config(cfg_path)


def _cache_dir(cfg: dict | None = None) -> Path:
    cfg = cfg or load_fx_config()
    raw = cfg.get("fx", {}).get("cache_dir") or cfg.get("cache_dir") or "data/fx"
    return resolve_path(raw)


def pair_to_yf(pair: str) -> str:
    p = pair.upper().replace("=X", "").replace("/", "")
    return f"{p}{YF_SUFFIX}"


def cache_path_for(pair: str, cfg: dict | None = None) -> Path:
    p = pair.upper().replace("=X", "").replace("/", "")
    return _cache_dir(cfg) / f"{p}.parquet"


def fetch_and_cache_fx(
    pair: str,
    start: str = "2000-01-01",
    end: str | None = None,
    force: bool = False,
    cfg: dict | None = None,
) -> Path:
    """
    Download daily OHLC for an FX pair via yfinance and cache to parquet.

    Columns: open, high, low, close (lowercase). Index: DatetimeIndex (UTC-naive).
    """
    cfg = cfg or load_fx_config()
    path = cache_path_for(pair, cfg)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists() and not force:
        return path

    try:
        import yfinance as yf
    except ImportError as e:
        raise FXDataError(
            "yfinance is required to fetch FX. Install with: pip install yfinance"
        ) from e

    ticker = pair_to_yf(pair)
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
        raise FXDataError(
            f"Failed to download {ticker} from yfinance: {e}. "
            f"Check network or place a cached parquet at {path}."
        ) from e

    if raw is None or raw.empty:
        raise FXDataError(
            f"yfinance returned empty data for {ticker}. "
            f"Place a cached parquet at {path} to proceed offline."
        )

    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    cols = {}
    for src, dst in (("Open", "open"), ("High", "high"), ("Low", "low"), ("Close", "close")):
        if src not in raw.columns:
            raise FXDataError(
                f"Missing column {src!r} in {ticker} download: {list(raw.columns)}"
            )
        cols[dst] = raw[src].astype(float)

    out = pd.DataFrame(cols)
    out.index = pd.to_datetime(out.index).tz_localize(None)
    out = out.dropna(how="any").sort_index()
    out.to_parquet(path)
    return path


def fetch_all_fx(
    pairs: Iterable[str] | None = None,
    start: str = "2000-01-01",
    end: str | None = None,
    force: bool = False,
    cfg: dict | None = None,
) -> dict[str, Path]:
    """Fetch & cache all configured FX pairs. Returns {pair: path}."""
    cfg = cfg or load_fx_config()
    pairs = list(pairs) if pairs is not None else list(
        cfg.get("fx", {}).get("pairs") or FX_PAIRS
    )
    return {
        p.upper().replace("=X", ""): fetch_and_cache_fx(
            p, start=start, end=end, force=force, cfg=cfg
        )
        for p in pairs
    }


def load_fx_ohlc(
    pair: str,
    cfg: dict | None = None,
    auto_fetch: bool = True,
    start: str = "2000-01-01",
) -> pd.DataFrame:
    """
    Load cached OHLC for a pair. Optionally fetch if missing.

    Returns DataFrame with columns open/high/low/close, DatetimeIndex.
    """
    cfg = cfg or load_fx_config()
    path = cache_path_for(pair, cfg)

    if not path.exists():
        if not auto_fetch:
            raise FXDataError(
                f"No FX cache at {path}. Run: python -m astro_market fetch-fx"
            )
        try:
            fetch_and_cache_fx(pair, start=start, cfg=cfg)
        except FXDataError:
            raise
        except Exception as e:
            raise FXDataError(
                f"Could not fetch FX {pair} and no cache at {path}: {e}"
            ) from e

    df = pd.read_parquet(path)
    # normalize column names
    df.columns = [c.lower() for c in df.columns]
    needed = ["open", "high", "low", "close"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise FXDataError(f"FX cache {path} missing columns {missing}")
    df = df[needed].astype(float).dropna(how="any")
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df.sort_index()


def load_all_fx(
    pairs: Iterable[str] | None = None,
    cfg: dict | None = None,
    auto_fetch: bool = True,
) -> dict[str, pd.DataFrame]:
    """Load OHLC for all pairs. Keys are normalized pair codes (EURUSD, …)."""
    cfg = cfg or load_fx_config()
    pairs = list(pairs) if pairs is not None else list(
        cfg.get("fx", {}).get("pairs") or FX_PAIRS
    )
    out: dict[str, pd.DataFrame] = {}
    for p in pairs:
        key = p.upper().replace("=X", "").replace("/", "")
        out[key] = load_fx_ohlc(key, cfg=cfg, auto_fetch=auto_fetch)
    return out


def atr(ohlc: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Wilder-style ATR (EMA of true range) over `period` bars.

    Uses only past/current bar OHLC — no lookahead. First `period` values NaN
    until enough history (we use ewm with adjust=False / Wilder alpha).
    """
    high = ohlc["high"].astype(float)
    low = ohlc["low"].astype(float)
    close = ohlc["close"].astype(float)
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    # Wilder ATR: RMA = EWM with alpha = 1/period
    alpha = 1.0 / float(period)
    out = tr.ewm(alpha=alpha, adjust=False, min_periods=period).mean()
    out.name = f"atr_{period}"
    return out


def data_start_end(ohlc: pd.DataFrame) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Actual first/last dates present in an OHLC frame."""
    return pd.Timestamp(ohlc.index.min()), pd.Timestamp(ohlc.index.max())
