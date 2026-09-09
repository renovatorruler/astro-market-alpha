"""Offline broker tests: TP/SL, no lookahead, concurrent / risk caps."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from astro_market.fx.broker import FxBrokerSim
from astro_market.fx.strategy import FxStrategy


def _ohlc_from_close(closes: list[float], start: str = "2020-01-01") -> pd.DataFrame:
    """Synthetic OHLC where H/L bracket close with small range; open=prev close."""
    idx = pd.date_range(start, periods=len(closes), freq="B")
    close = pd.Series(closes, index=idx, dtype=float)
    open_ = close.shift(1).fillna(close.iloc[0])
    # Wide enough range for SL/TP tests to control via explicit H/L later
    high = pd.concat([open_, close], axis=1).max(axis=1) + 0.0002
    low = pd.concat([open_, close], axis=1).min(axis=1) - 0.0002
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close})


def _atoms_constant(index: pd.DatetimeIndex, name: str = "sig", values=None) -> pd.DataFrame:
    if values is None:
        values = [False] * len(index)
    return pd.DataFrame({name: pd.Series(values, index=index, dtype=bool)})


def test_take_profit_hit():
    """Long entry then price runs to TP before SL."""
    # Build controlled path: signal on day0, enter day1 open, TP on day3
    n = 20
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    # Flat around 1.1000 then rally
    closes = [1.1000] * n
    ohlc = _ohlc_from_close(closes)
    # Force ATR-ish stability: set explicit bars
    # Entry at bar 2 open ≈ 1.10; with fixed pip SL fallback
    sig = [False] * n
    sig[0] = True  # rising edge day0 → enter day1
    atoms = pd.DataFrame({"sig": pd.Series(sig, index=idx, dtype=bool)})

    # Widen bar after entry so TP is reachable.
    # Strategy: sl_atr large but ATR NaN early → fallback 50 pips = 0.005
    # TP = 2R = 0.010. Set high on a post-entry bar above entry+0.010
    entry_i = 1
    ohlc.iloc[entry_i, ohlc.columns.get_loc("open")] = 1.1000
    ohlc.iloc[entry_i, ohlc.columns.get_loc("high")] = 1.1005
    ohlc.iloc[entry_i, ohlc.columns.get_loc("low")] = 1.0995
    tp_i = 3
    ohlc.iloc[tp_i, ohlc.columns.get_loc("high")] = 1.1200  # way above TP
    ohlc.iloc[tp_i, ohlc.columns.get_loc("low")] = 1.1000

    strat = FxStrategy(
        rule="sig",
        side="long",
        risk_pct=0.01,
        sl_atr=1.5,
        tp_R=2.0,
        atr_period=14,
        sl_pips_fallback=50.0,
        time_stop=None,
    )
    # Disable risk caps interference for single pair
    sim = FxBrokerSim(
        initial_equity=100_000,
        spread_pips=0.0,
        max_concurrent=10,
        max_total_risk_pct=1.0,
    )
    result = sim.run(strat, {"EURUSD": ohlc}, atoms, pairs=["EURUSD"])
    assert result.trades, "expected at least one trade"
    # First trade should hit TP
    t0 = result.trades[0]
    assert t0.exit_reason == "tp"
    assert t0.pnl > 0


def test_stop_loss_hit():
    n = 20
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    closes = [1.1000] * n
    ohlc = _ohlc_from_close(closes)
    sig = [False] * n
    sig[0] = True
    atoms = pd.DataFrame({"sig": pd.Series(sig, index=idx, dtype=bool)})
    entry_i = 1
    ohlc.iloc[entry_i, ohlc.columns.get_loc("open")] = 1.1000
    # Next bar crashes through SL (50 pips = 0.005 → SL=1.095)
    sl_i = 2
    ohlc.iloc[sl_i, ohlc.columns.get_loc("low")] = 1.0900
    ohlc.iloc[sl_i, ohlc.columns.get_loc("high")] = 1.1000

    strat = FxStrategy(
        rule="sig",
        side="long",
        risk_pct=0.01,
        sl_pips_fallback=50.0,
        time_stop=None,
    )
    sim = FxBrokerSim(
        initial_equity=100_000,
        spread_pips=0.0,
        max_concurrent=10,
        max_total_risk_pct=1.0,
    )
    result = sim.run(strat, {"EURUSD": ohlc}, atoms, pairs=["EURUSD"])
    assert result.trades
    assert result.trades[0].exit_reason == "sl"
    assert result.trades[0].pnl < 0


def test_no_lookahead_entry_next_open():
    """Signal on bar t must fill at bar t+1 open, not t close/open."""
    n = 15
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    closes = np.linspace(1.10, 1.12, n).tolist()
    ohlc = _ohlc_from_close(closes)
    # Distinct open on day after signal
    sig = [False] * n
    sig[2] = True
    atoms = pd.DataFrame({"sig": pd.Series(sig, index=idx, dtype=bool)})
    ohlc.iloc[2, ohlc.columns.get_loc("open")] = 1.1111
    ohlc.iloc[2, ohlc.columns.get_loc("close")] = 1.2222
    ohlc.iloc[3, ohlc.columns.get_loc("open")] = 1.3333
    ohlc.iloc[3, ohlc.columns.get_loc("high")] = 1.3340
    ohlc.iloc[3, ohlc.columns.get_loc("low")] = 1.3320
    ohlc.iloc[3, ohlc.columns.get_loc("close")] = 1.3335

    strat = FxStrategy(rule="sig", side="long", time_stop=5, sl_pips_fallback=50.0)
    sim = FxBrokerSim(
        initial_equity=100_000,
        spread_pips=0.0,
        max_concurrent=10,
        max_total_risk_pct=1.0,
    )
    result = sim.run(strat, {"EURUSD": ohlc}, atoms, pairs=["EURUSD"])
    assert result.trades
    tr = result.trades[0]
    assert tr.signal_time == idx[2]
    assert tr.entry_time == idx[3]
    assert tr.entry_price == pytest.approx(1.3333)


def test_concurrent_cap():
    """max_concurrent=1 allows only one open trade; extra signals dropped."""
    n = 30
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    closes = [1.1000] * n
    ohlc = _ohlc_from_close(closes)
    # Many rising edges while first trade still open (time_stop large, no SL/TP hit)
    sig = [False] * n
    for i in (0, 2, 4, 6):
        sig[i] = True
    atoms = pd.DataFrame({"sig": pd.Series(sig, index=idx, dtype=bool)})
    # Keep range tight so SL/TP never hit; time_stop=20
    strat = FxStrategy(
        rule="sig",
        side="long",
        risk_pct=0.005,
        sl_pips_fallback=500.0,  # very wide
        tp_R=50.0,
        time_stop=20,
    )
    sim = FxBrokerSim(
        initial_equity=100_000,
        spread_pips=0.0,
        max_concurrent=1,
        max_total_risk_pct=1.0,
    )
    result = sim.run(strat, {"EURUSD": ohlc}, atoms, pairs=["EURUSD"])
    # Overlapping window: at most 1 trade should have been open; with one pair
    # and concurrent=1, number of entries while first is open is blocked.
    # We should see fewer trades than rising edges (4 edges).
    n_edges = sum(sig)
    assert n_edges == 4
    assert len(result.trades) < n_edges


def test_max_total_risk_cap():
    """max_total_risk_pct blocks stacking beyond ~5% when risk_pct=0.5% * many pairs."""
    n = 25
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    pairs = ["EURUSD", "GBPUSD", "AUDUSD", "NZDUSD", "USDCHF", "USDCAD"]
    ohlc_by = {}
    for p, base in zip(pairs, [1.10, 1.30, 0.70, 0.65, 0.90, 1.30]):
        ohlc_by[p] = _ohlc_from_close([base] * n)
        if p.endswith("JPY"):
            pass
    # USDJPY separately with jpy scale
    ohlc_by["USDJPY"] = _ohlc_from_close([150.0] * n)
    pairs = list(ohlc_by.keys())

    sig = [False] * n
    sig[0] = True  # one edge → tries to enter ALL pairs same next bar
    atoms = pd.DataFrame({"sig": pd.Series(sig, index=idx, dtype=bool)})

    strat = FxStrategy(
        rule="sig",
        side="long",
        risk_pct=0.01,  # 1% each
        sl_pips_fallback=50.0,
        tp_R=100.0,
        time_stop=15,
    )
    sim = FxBrokerSim(
        initial_equity=100_000,
        spread_pips=0.0,
        max_concurrent=20,
        max_total_risk_pct=0.05,  # 5% → at most ~5 trades at 1% each
    )
    result = sim.run(strat, ohlc_by, atoms, pairs=pairs)
    # All entries same bar; risk cap should limit to <=5
    # (some may be rejected mid-loop)
    assert len(result.trades) <= 5
    assert len(result.trades) >= 1


def test_both_sl_tp_same_bar_prefers_sl():
    n = 15
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    ohlc = _ohlc_from_close([1.10] * n)
    sig = [False] * n
    sig[0] = True
    atoms = pd.DataFrame({"sig": pd.Series(sig, index=idx, dtype=bool)})
    # Entry bar 1
    ohlc.iloc[1, ohlc.columns.get_loc("open")] = 1.1000
    # Bar 2 spans both SL and TP
    ohlc.iloc[2, ohlc.columns.get_loc("high")] = 1.2000
    ohlc.iloc[2, ohlc.columns.get_loc("low")] = 1.0000
    strat = FxStrategy(rule="sig", side="long", sl_pips_fallback=50.0, time_stop=None)
    sim = FxBrokerSim(
        initial_equity=100_000,
        spread_pips=0.0,
        max_concurrent=5,
        max_total_risk_pct=1.0,
    )
    result = sim.run(strat, {"EURUSD": ohlc}, atoms, pairs=["EURUSD"])
    assert result.trades
    assert result.trades[0].exit_reason == "sl"
