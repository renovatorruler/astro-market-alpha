"""Metrics and split-wise evaluation for the FX risk framework."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

from astro_market.fx.broker import FxBrokerSim, SimResult, Trade
from astro_market.fx.data import load_fx_config
from astro_market.fx.strategy import FxStrategy


@dataclass
class FxMetrics:
    total_return: float
    cagr: float
    sharpe: float
    max_dd: float
    n_trades: int
    win_rate: float
    avg_R: float
    profit_factor: float
    expectancy_R: float
    n_days: int
    final_equity: float
    initial_equity: float
    pct_sl: float
    pct_tp: float
    pct_time: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _returns_from_equity(equity: pd.Series) -> pd.Series:
    if len(equity) < 2:
        return pd.Series(dtype=float)
    return equity.pct_change().dropna()


def _cagr(equity: pd.Series, periods_per_year: float = 252.0) -> float:
    if len(equity) < 2:
        return float("nan")
    total = float(equity.iloc[-1] / equity.iloc[0])
    years = (len(equity) - 1) / periods_per_year
    if years <= 0 or total <= 0:
        return float("nan")
    return float(total ** (1.0 / years) - 1.0)


def _sharpe(r: pd.Series, periods_per_year: float = 252.0) -> float:
    if len(r) < 2:
        return float("nan")
    mu = r.mean()
    sigma = r.std(ddof=1)
    if sigma == 0 or np.isnan(sigma):
        return float("nan")
    return float(np.sqrt(periods_per_year) * mu / sigma)


def _max_dd(equity: pd.Series) -> float:
    if len(equity) == 0:
        return float("nan")
    peak = equity.cummax()
    dd = equity / peak - 1.0
    return float(dd.min())


def _trade_R(tr: Trade) -> float:
    if tr.risk_usd <= 0:
        return float("nan")
    return tr.pnl / tr.risk_usd


def metrics_from_sim(sim: SimResult) -> FxMetrics:
    eq = sim.equity_curve.astype(float)
    initial = float(sim.meta.get("initial_equity", eq.iloc[0] if len(eq) else 1.0))
    final = float(eq.iloc[-1]) if len(eq) else initial
    rets = _returns_from_equity(eq)
    trades = sim.trades
    n = len(trades)
    wins = [t for t in trades if t.pnl > 0]
    losses = [t for t in trades if t.pnl < 0]
    Rs = [ _trade_R(t) for t in trades if t.risk_usd > 0 ]
    Rs = [r for r in Rs if np.isfinite(r)]
    gross_win = sum(t.pnl for t in wins) if wins else 0.0
    gross_loss = abs(sum(t.pnl for t in losses)) if losses else 0.0
    pf = (gross_win / gross_loss) if gross_loss > 0 else (float("inf") if gross_win > 0 else float("nan"))
    reasons = [t.exit_reason for t in trades]
    def _pct(tag: str) -> float:
        if n == 0:
            return float("nan")
        return sum(1 for r in reasons if r == tag) / n

    return FxMetrics(
        total_return=(final / initial - 1.0) if initial else float("nan"),
        cagr=_cagr(eq),
        sharpe=_sharpe(rets),
        max_dd=_max_dd(eq),
        n_trades=n,
        win_rate=(len(wins) / n) if n else float("nan"),
        avg_R=float(np.mean(Rs)) if Rs else float("nan"),
        profit_factor=float(pf),
        expectancy_R=float(np.mean(Rs)) if Rs else float("nan"),
        n_days=len(eq),
        final_equity=final,
        initial_equity=initial,
        pct_sl=_pct("sl"),
        pct_tp=_pct("tp"),
        pct_time=_pct("time"),
    )


def slice_ohlc(
    ohlc_by_pair: dict[str, pd.DataFrame],
    start: str | pd.Timestamp,
    end: str | pd.Timestamp | None,
) -> dict[str, pd.DataFrame]:
    start_ts = pd.Timestamp(start)
    out: dict[str, pd.DataFrame] = {}
    for p, df in ohlc_by_pair.items():
        if end is None or (isinstance(end, float) and pd.isna(end)):
            out[p] = df.loc[df.index >= start_ts]
        else:
            end_ts = pd.Timestamp(end)
            out[p] = df.loc[(df.index >= start_ts) & (df.index <= end_ts)]
    return out


def evaluate_fx_strategy(
    strategy: FxStrategy,
    ohlc_by_pair: dict[str, pd.DataFrame],
    atoms: pd.DataFrame,
    *,
    broker: FxBrokerSim | None = None,
    pairs: list[str] | None = None,
    cfg: dict | None = None,
) -> tuple[FxMetrics, SimResult]:
    """Run full-sample simulation and return metrics + raw sim result."""
    cfg = cfg or load_fx_config()
    fx = cfg.get("fx", cfg)
    risk = fx.get("risk", {})
    costs = fx.get("costs", {})
    broker = broker or FxBrokerSim(
        initial_equity=float(fx.get("initial_equity", 100_000.0)),
        spread_pips=float(costs.get("spread_pips", 1.0)),
        max_concurrent=int(risk.get("max_concurrent", 5)),
        max_total_risk_pct=float(risk.get("max_total_risk_pct", 0.05)),
    )
    sim = broker.run(strategy, ohlc_by_pair, atoms, pairs=pairs)
    return metrics_from_sim(sim), sim


def evaluate_on_fx_split(
    strategy: FxStrategy,
    ohlc_by_pair: dict[str, pd.DataFrame],
    atoms: pd.DataFrame,
    split: str,
    *,
    cfg: dict | None = None,
    broker: FxBrokerSim | None = None,
    pairs: list[str] | None = None,
) -> FxMetrics:
    """
    Evaluate on a named protocol split (train / validation / holdout).

    NEVER tune on holdout — this helper only reports metrics.
    """
    cfg = cfg or load_fx_config()
    splits = cfg.get("splits") or cfg.get("fx", {}).get("splits")
    if not splits or split not in splits:
        raise KeyError(f"Unknown FX split {split!r}; available={list((splits or {}).keys())}")
    start, end = splits[split]
    sliced = slice_ohlc(ohlc_by_pair, start, end)
    # Warm-up ATR: include lookback bars before split start from original data
    atr_period = strategy.atr_period
    warmed: dict[str, pd.DataFrame] = {}
    start_ts = pd.Timestamp(start)
    for p, full in ohlc_by_pair.items():
        if p not in sliced or sliced[p].empty:
            warmed[p] = sliced.get(p, full.iloc[0:0])
            continue
        # take up to atr_period*3 prior bars for ATR seed
        prior = full.loc[full.index < start_ts].tail(atr_period * 3 + 5)
        warmed[p] = pd.concat([prior, sliced[p]]).sort_index()
        warmed[p] = warmed[p][~warmed[p].index.duplicated(keep="last")]

    m, sim = evaluate_fx_strategy(
        strategy, warmed, atoms, broker=broker, pairs=pairs, cfg=cfg
    )
    # Restrict equity/metrics reporting to in-split dates
    if len(sim.equity_curve):
        end_ts = None if end is None or (isinstance(end, float) and pd.isna(end)) else pd.Timestamp(end)
        mask = sim.equity_curve.index >= start_ts
        if end_ts is not None:
            mask = mask & (sim.equity_curve.index <= end_ts)
        eq = sim.equity_curve.loc[mask]
        # Rebuild a slim SimResult for metrics on the split window
        # Shift equity so first in-split point is initial (drop prior warm PnL)
        if len(eq):
            # trades that exited inside the window
            trades = [
                t
                for t in sim.trades
                if t.exit_time is not None
                and t.exit_time >= start_ts
                and (end_ts is None or t.exit_time <= end_ts)
            ]
            # Reconstruct equity from initial + in-window trade PnL chronologically
            fx = cfg.get("fx", cfg)
            initial = float(fx.get("initial_equity", 100_000.0))
            # Use daily equity pct from sim but rebase
            base = float(eq.iloc[0])
            rebased = eq / base * initial
            slim = SimResult(
                equity_curve=rebased,
                trades=trades,
                daily_pnl=sim.daily_pnl.loc[mask] if len(sim.daily_pnl) else sim.daily_pnl,
                meta={**sim.meta, "initial_equity": initial, "split": split},
            )
            return metrics_from_sim(slim)
    return m
