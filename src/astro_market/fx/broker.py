"""Event-driven FX broker simulator: fixed risk, SL/TP, spreads, concurrency caps.

No lookahead:
- Entry signals evaluated on bar ``t`` (rising edge) enter at bar ``t+1`` open.
- ATR / SL distance for an entry uses ATR known at signal bar ``t`` (shifted onto
  the entry bar), never future bars.
- Intrabar SL/TP: if both hit same bar, conservative assumption = SL first.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd

from astro_market.fx.data import atr as compute_atr
from astro_market.fx.sizing import (
    normalize_pair,
    pip_size,
    pips_to_price,
    pnl_usd,
    price_to_pips,
    risk_usd_for_units,
    units_for_risk,
)
from astro_market.fx.strategy import FxStrategy, rising_edge
from astro_market.rules import evaluate_rule_signal

ExitReason = Literal["sl", "tp", "time", "eod", "force"]


@dataclass
class Trade:
    pair: str
    side: int  # +1 long, -1 short
    entry_time: pd.Timestamp
    entry_price: float
    units: float
    sl: float
    tp: float
    risk_usd: float
    signal_time: pd.Timestamp
    bars_held: int = 0
    exit_time: pd.Timestamp | None = None
    exit_price: float | None = None
    exit_reason: ExitReason | None = None
    pnl: float = 0.0
    spread_cost: float = 0.0

    @property
    def open(self) -> bool:
        return self.exit_time is None


@dataclass
class SimResult:
    equity_curve: pd.Series
    trades: list[Trade]
    daily_pnl: pd.Series
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def final_equity(self) -> float:
        if len(self.equity_curve) == 0:
            return float("nan")
        return float(self.equity_curve.iloc[-1])


@dataclass
class FxBrokerSim:
    """
    Multi-pair FX simulator with risk caps.

    Parameters mirror ``configs/fx.yaml`` risk / cost section.
    """

    initial_equity: float = 100_000.0
    spread_pips: float = 1.0
    max_concurrent: int = 5
    max_total_risk_pct: float = 0.05
    # If True, refuse new entries that would push open risk above cap.
    enforce_risk_cap: bool = True

    def run(
        self,
        strategy: FxStrategy,
        ohlc_by_pair: dict[str, pd.DataFrame],
        atoms: pd.DataFrame,
        pairs: list[str] | None = None,
    ) -> SimResult:
        """
        Simulate ``strategy`` across pairs.

        ``atoms`` must cover the trading calendar (aligned internally).
        Entry on rising edge of rule (long) and/or rising edge of ~rule (short).
        """
        pairs = [normalize_pair(p) for p in (pairs or list(ohlc_by_pair.keys()))]
        # Build per-pair frames with ATR
        frames: dict[str, pd.DataFrame] = {}
        for p in pairs:
            if p not in ohlc_by_pair:
                raise KeyError(f"OHLC missing for pair {p}")
            df = ohlc_by_pair[p][["open", "high", "low", "close"]].astype(float).copy()
            df["atr"] = compute_atr(df, period=strategy.atr_period)
            frames[p] = df

        # Union calendar (sorted unique dates present in any pair & atoms)
        all_idx = None
        for df in frames.values():
            all_idx = df.index if all_idx is None else all_idx.union(df.index)
        assert all_idx is not None
        calendar = all_idx.intersection(atoms.index).sort_values()
        if len(calendar) == 0:
            return SimResult(
                equity_curve=pd.Series(dtype=float),
                trades=[],
                daily_pnl=pd.Series(dtype=float),
                meta={"error": "empty calendar after atom alignment"},
            )

        signal = evaluate_rule_signal(strategy.rule, atoms.loc[calendar])
        signal = signal.reindex(calendar).fillna(False).astype(bool)
        long_edge = rising_edge(signal)
        short_edge = rising_edge(~signal)

        # Pending entries queued on signal bar → fill next open
        # Each item: (pair, side, signal_time, sl_distance)
        pending: list[tuple[str, int, pd.Timestamp, float]] = []

        open_trades: list[Trade] = []
        closed: list[Trade] = []
        equity = float(self.initial_equity)
        equity_points: list[tuple[pd.Timestamp, float]] = []
        daily_pnl_map: dict[pd.Timestamp, float] = {}

        # Precompute signal-bar ATR / SL distance (known at t, used for entry t+1)
        sl_dist_at_signal: dict[str, pd.Series] = {}
        for p, df in frames.items():
            atr_s = df["atr"]
            fb = pips_to_price(p, strategy.sl_pips_fallback)
            dist = (strategy.sl_atr * atr_s).where(atr_s.notna() & (atr_s > 0), fb)
            sl_dist_at_signal[p] = dist.reindex(calendar)

        for i, ts in enumerate(calendar):
            day_pnl = 0.0

            # --- 1) Fill pending entries at today's open ---
            still_pending: list[tuple[str, int, pd.Timestamp, float]] = []
            for pair, side, sig_t, sl_dist in pending:
                df = frames[pair]
                if ts not in df.index:
                    # Pair holiday — keep pending one more bar if possible
                    still_pending.append((pair, side, sig_t, sl_dist))
                    continue
                if not self._can_open(open_trades, equity, strategy.risk_pct):
                    continue  # drop signal — risk/concurrent cap
                raw_open = float(df.loc[ts, "open"])
                # Pay half-spread on entry (adverse)
                spread_price = pips_to_price(pair, self.spread_pips)
                fill = raw_open + side * (spread_price / 2.0)
                risk_usd = equity * strategy.risk_pct
                units = units_for_risk(pair, fill, sl_dist, risk_usd)
                if units <= 0 or sl_dist <= 0:
                    continue
                # Risk-cap check with actual sized risk
                actual_risk = risk_usd_for_units(pair, fill, sl_dist, units)
                if self.enforce_risk_cap:
                    open_risk = sum(t.risk_usd for t in open_trades if t.open)
                    if open_risk + actual_risk > equity * self.max_total_risk_pct + 1e-9:
                        continue
                sl = fill - side * sl_dist
                tp = fill + side * sl_dist * strategy.tp_R
                spread_cost = abs(
                    pnl_usd(pair, raw_open, fill, units, side)
                )  # approx cost of adverse fill
                # Better: cost = units * pip_value * spread_pips / 2 already in fill;
                # track explicit round-trip later on exit too.
                trade = Trade(
                    pair=pair,
                    side=side,
                    entry_time=ts,
                    entry_price=fill,
                    units=units,
                    sl=sl,
                    tp=tp,
                    risk_usd=actual_risk,
                    signal_time=sig_t,
                    spread_cost=usd_spread_cost(pair, fill, units, self.spread_pips),
                )
                open_trades.append(trade)
            pending = still_pending

            # --- 2) Manage open trades on today's bar (SL/TP/time) ---
            still_open: list[Trade] = []
            for tr in open_trades:
                df = frames[tr.pair]
                if ts not in df.index:
                    still_open.append(tr)
                    continue
                # Skip same-bar entry management? Allow SL/TP from entry bar high/low
                # (intrabar after open fill). Time-stop counts this bar.
                bar = df.loc[ts]
                high = float(bar["high"])
                low = float(bar["low"])
                close = float(bar["close"])
                tr.bars_held += 1

                hit_sl = False
                hit_tp = False
                if tr.side > 0:
                    hit_sl = low <= tr.sl
                    hit_tp = high >= tr.tp
                else:
                    hit_sl = high >= tr.sl
                    hit_tp = low <= tr.tp

                exit_px: float | None = None
                reason: ExitReason | None = None
                if hit_sl and hit_tp:
                    # Conservative: stop first
                    exit_px = tr.sl
                    reason = "sl"
                elif hit_sl:
                    exit_px = tr.sl
                    reason = "sl"
                elif hit_tp:
                    exit_px = tr.tp
                    reason = "tp"
                elif (
                    strategy.time_stop is not None
                    and tr.bars_held >= int(strategy.time_stop)
                ):
                    # Exit at close; pay half-spread adverse
                    spread_price = pips_to_price(tr.pair, self.spread_pips)
                    exit_px = close - tr.side * (spread_price / 2.0)
                    reason = "time"

                if exit_px is not None and reason is not None:
                    tr.exit_time = ts
                    tr.exit_price = float(exit_px)
                    tr.exit_reason = reason
                    tr.pnl = pnl_usd(
                        tr.pair, tr.entry_price, tr.exit_price, tr.units, tr.side
                    )
                    # Round-trip spread already partially in fills; for SL/TP
                    # exact level exits, charge remaining half-spread once.
                    if reason in ("sl", "tp"):
                        tr.pnl -= 0.5 * tr.spread_cost
                    equity += tr.pnl
                    day_pnl += tr.pnl
                    closed.append(tr)
                else:
                    still_open.append(tr)
            open_trades = still_open

            # --- 3) Queue new signals from *today* for next bar open ---
            # Use ATR/sl distance known today (no lookahead into future ATR).
            if i + 1 < len(calendar):  # else cannot enter next bar
                for pair in pairs:
                    if pair not in frames or ts not in frames[pair].index:
                        continue
                    dist = sl_dist_at_signal[pair].loc[ts]
                    if dist is None or (isinstance(dist, float) and not np.isfinite(dist)):
                        continue
                    dist_f = float(dist)
                    if dist_f <= 0:
                        continue
                    if strategy.longs_enabled() and bool(long_edge.loc[ts]):
                        pending.append((pair, +1, ts, dist_f))
                    if strategy.shorts_enabled() and bool(short_edge.loc[ts]):
                        pending.append((pair, -1, ts, dist_f))

            equity_points.append((ts, equity))
            daily_pnl_map[ts] = day_pnl

        # Force-close remaining at last available close
        if open_trades and len(calendar):
            last = calendar[-1]
            for tr in open_trades:
                df = frames[tr.pair]
                # last available close on or before last
                sub = df.loc[:last]
                if sub.empty:
                    continue
                close = float(sub.iloc[-1]["close"])
                spread_price = pips_to_price(tr.pair, self.spread_pips)
                exit_px = close - tr.side * (spread_price / 2.0)
                tr.exit_time = sub.index[-1]
                tr.exit_price = exit_px
                tr.exit_reason = "force"
                tr.pnl = pnl_usd(
                    tr.pair, tr.entry_price, tr.exit_price, tr.units, tr.side
                )
                equity += tr.pnl
                daily_pnl_map[tr.exit_time] = (
                    daily_pnl_map.get(tr.exit_time, 0.0) + tr.pnl
                )
                closed.append(tr)
            if equity_points:
                equity_points[-1] = (equity_points[-1][0], equity)
            open_trades = []

        eq = pd.Series(
            {t: e for t, e in equity_points},
            dtype=float,
            name="equity",
        ).sort_index()
        dp = pd.Series(daily_pnl_map, dtype=float, name="daily_pnl").sort_index()
        return SimResult(
            equity_curve=eq,
            trades=closed,
            daily_pnl=dp,
            meta={
                "initial_equity": self.initial_equity,
                "final_equity": float(eq.iloc[-1]) if len(eq) else self.initial_equity,
                "n_trades": len(closed),
                "pairs": pairs,
                "rule": strategy.rule,
            },
        )

    def _can_open(
        self,
        open_trades: list[Trade],
        equity: float,
        risk_pct: float,
    ) -> bool:
        n_open = sum(1 for t in open_trades if t.open)
        if n_open >= self.max_concurrent:
            return False
        if self.enforce_risk_cap:
            open_risk = sum(t.risk_usd for t in open_trades if t.open)
            # tentative: another risk_pct of equity
            if open_risk + equity * risk_pct > equity * self.max_total_risk_pct + 1e-9:
                return False
        return True


def usd_spread_cost(pair: str, price: float, units: float, spread_pips: float) -> float:
    """Full round-trip spread cost in USD (entry+exit halves)."""
    from astro_market.fx.sizing import pip_value_usd

    return abs(pip_value_usd(pair, price, units=units) * float(spread_pips))
