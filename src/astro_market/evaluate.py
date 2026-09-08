"""Walk-forward style rule evaluation vs buy-and-hold."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

from astro_market.config import load_config
from astro_market.rules import evaluate_rule_signal, parse_rule, rule_complexity


@dataclass
class EvalMetrics:
    total_return: float
    cagr: float
    sharpe: float
    excess_sharpe: float  # sharpe_strategy - sharpe_bh
    max_dd: float
    turnover: float
    n_trades: int
    complexity: int
    sharpe_bh: float
    total_return_bh: float
    cagr_bh: float
    n_days: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _align(
    returns: pd.Series,
    signal: pd.Series,
) -> tuple[pd.Series, pd.Series]:
    """Align returns and boolean signal on common index; drop NaNs in returns."""
    common = returns.dropna().index.intersection(signal.index)
    r = returns.loc[common].astype(float)
    s = signal.loc[common].astype(bool)
    return r, s


def positions_from_signal(
    signal: pd.Series,
    long: float = 1.0,
    flat: float = 0.0,
) -> pd.Series:
    """Map boolean signal to positions: True->long, False->flat."""
    return signal.map({True: long, False: flat}).astype(float)


def apply_costs(
    positions: pd.Series,
    returns: pd.Series,
    bps_per_side: float = 10.0,
) -> pd.Series:
    """
    Strategy returns = position_{t-1} * r_t - cost on position changes.

    Cost charged when position changes between t-1 and t, applied at t
    (bps_per_side / 10000 * |delta_position|). For long/flat, |delta| in {0,1}.
    """
    pos = positions.astype(float)
    # Use yesterday's position for today's return (no look-ahead)
    pos_lag = pos.shift(1)
    # First day: assume starting flat -> entering costs if first pos is long
    pos_lag = pos_lag.fillna(0.0)

    gross = pos_lag * returns

    delta = pos_lag.diff().abs().fillna(pos_lag.abs())  # first day: cost of entering from 0
    cost_frac = (bps_per_side / 10_000.0) * delta
    net = gross - cost_frac
    return net


def _total_return(r: pd.Series) -> float:
    return float((1.0 + r).prod() - 1.0)


def _cagr(r: pd.Series, periods_per_year: float = 252.0) -> float:
    n = len(r)
    if n == 0:
        return float("nan")
    total = (1.0 + r).prod()
    years = n / periods_per_year
    if years <= 0 or total <= 0:
        return float("nan")
    return float(total ** (1.0 / years) - 1.0)


def _sharpe(r: pd.Series, rf: float = 0.0, periods_per_year: float = 252.0) -> float:
    if len(r) < 2:
        return float("nan")
    excess = r - rf / periods_per_year
    mu = excess.mean()
    sigma = excess.std(ddof=1)
    if sigma == 0 or np.isnan(sigma):
        return float("nan")
    return float(np.sqrt(periods_per_year) * mu / sigma)


def _max_drawdown(r: pd.Series) -> float:
    if len(r) == 0:
        return float("nan")
    wealth = (1.0 + r).cumprod()
    peak = wealth.cummax()
    dd = wealth / peak - 1.0
    return float(dd.min())


def _turnover_and_trades(positions: pd.Series) -> tuple[float, int]:
    """
    Turnover = mean(|delta position|) per day (lagged positions used for trading).
    n_trades = count of position changes (from lagged series).
    """
    pos = positions.astype(float)
    pos_lag = pos.shift(1).fillna(0.0)
    delta = pos_lag.diff().abs()
    # First observation after fill: count change from 0 if non-zero
    delta.iloc[0] = abs(pos_lag.iloc[0])
    turnover = float(delta.mean()) if len(delta) else float("nan")
    n_trades = int((delta > 1e-12).sum())
    return turnover, n_trades


def evaluate_rule(
    rule: str,
    returns: pd.Series,
    atoms: pd.DataFrame,
    *,
    bps_per_side: float = 10.0,
    rf: float = 0.0,
    long: float = 1.0,
    flat: float = 0.0,
) -> EvalMetrics:
    """
    Evaluate a rule on a returns series given atom frame.

    Position: long when rule True, flat when False.
    BH: always long.
    Costs: bps_per_side on each side of position changes.
    """
    signal = evaluate_rule_signal(rule, atoms)
    r, sig = _align(returns, signal)
    if len(r) == 0:
        raise ValueError("No overlapping dates between returns and signal")

    pos = positions_from_signal(sig, long=long, flat=flat)
    strat_r = apply_costs(pos, r, bps_per_side=bps_per_side)

    # Buy-and-hold: always long, no turnover after initial entry
    bh_pos = pd.Series(long, index=r.index, dtype=float)
    bh_r = apply_costs(bh_pos, r, bps_per_side=bps_per_side)

    sharpe_s = _sharpe(strat_r, rf=rf)
    sharpe_bh = _sharpe(bh_r, rf=rf)
    turnover, n_trades = _turnover_and_trades(pos)
    complexity = rule_complexity(rule)

    return EvalMetrics(
        total_return=_total_return(strat_r),
        cagr=_cagr(strat_r),
        sharpe=sharpe_s,
        excess_sharpe=(sharpe_s - sharpe_bh) if np.isfinite(sharpe_s) and np.isfinite(sharpe_bh) else float("nan"),
        max_dd=_max_drawdown(strat_r),
        turnover=turnover,
        n_trades=n_trades,
        complexity=complexity,
        sharpe_bh=sharpe_bh,
        total_return_bh=_total_return(bh_r),
        cagr_bh=_cagr(bh_r),
        n_days=len(r),
    )


def fitness(
    metrics: EvalMetrics,
    lambda_complexity: float = 0.01,
) -> float:
    """Fitness = sharpe_s - sharpe_bh - lambda * complexity."""
    if not np.isfinite(metrics.excess_sharpe):
        return float("-inf")
    return float(metrics.excess_sharpe - lambda_complexity * metrics.complexity)


def embargo_split(
    index: pd.DatetimeIndex,
    train_end: pd.Timestamp,
    embargo_days: int = 5,
) -> tuple[pd.DatetimeIndex, pd.DatetimeIndex]:
    """
    Internal walk-forward helper: after train_end, skip `embargo_days`
    trading days before validation begins.
    Returns (train_index, val_index) within `index`.
    """
    idx = pd.DatetimeIndex(index).sort_values()
    train_mask = idx <= train_end
    train_idx = idx[train_mask]
    remaining = idx[~train_mask]
    if embargo_days > 0 and len(remaining) > 0:
        val_idx = remaining[embargo_days:]
    else:
        val_idx = remaining
    return train_idx, val_idx


def evaluate_on_split(
    rule: str,
    returns: pd.Series,
    atoms: pd.DataFrame,
    split_name: str,
    cfg: dict | None = None,
) -> EvalMetrics:
    """Evaluate rule on a named protocol split."""
    cfg = cfg or load_config()
    start, end = cfg["splits"][split_name]
    start_ts = pd.Timestamp(start)
    if end is None or (isinstance(end, float) and pd.isna(end)):
        mask = returns.index >= start_ts
    else:
        end_ts = pd.Timestamp(end)
        mask = (returns.index >= start_ts) & (returns.index <= end_ts)

    r = returns.loc[mask]
    a = atoms.reindex(r.index)
    # Drop rows where atoms missing
    valid = a.notna().all(axis=1) if len(a.columns) else pd.Series(True, index=r.index)
    # atoms are bool; after reindex may be NaN
    a = a.loc[valid].astype(bool)
    r = r.loc[a.index]

    costs = float(cfg["costs"]["bps_per_side"])
    rf = float(cfg["evaluate"]["risk_free"])
    long = float(cfg["position"]["long"])
    flat = float(cfg["position"]["flat"])
    return evaluate_rule(
        rule, r, a, bps_per_side=costs, rf=rf, long=long, flat=flat
    )
