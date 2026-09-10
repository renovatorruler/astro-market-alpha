"""Tests for FX search fitness / fast trade helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from astro_market.fx.search import (
    PairTape,
    SplitTape,
    circular_shift_tapes,
    fx_fitness,
    rising_edge_indices,
    simulate_trade_Rs,
    trade_count_penalty,
    protocol_pass_fx,
    FxSearchResult,
)


def test_trade_count_penalty_hard_and_soft():
    assert trade_count_penalty(10, min_trades=30) >= 10.0
    assert trade_count_penalty(30, min_trades=30) == 0.0 or trade_count_penalty(30, min_trades=30) < 0.05
    assert trade_count_penalty(100, min_trades=30) == 0.0
    soft = trade_count_penalty(35, min_trades=30)
    hard = trade_count_penalty(20, min_trades=30)
    assert hard > soft


def test_fx_fitness_penalties():
    # Positive avg R with enough trades beats same with low trades
    good = fx_fitness(0.10, n_trades=50, complexity=1, lam=0.02, min_trades=30)
    low_n = fx_fitness(0.10, n_trades=5, complexity=1, lam=0.02, min_trades=30)
    complex_ = fx_fitness(0.10, n_trades=50, complexity=5, lam=0.02, min_trades=30)
    assert good > low_n
    assert good > complex_
    assert not np.isfinite(fx_fitness(float("nan"), 50, 1)) or fx_fitness(float("nan"), 50, 1) == float("-inf")


def test_rising_edge_indices():
    s = np.array([0, 1, 1, 0, 1, 0], dtype=bool)
    idx = rising_edge_indices(s)
    assert list(idx) == [1, 4]
    assert rising_edge_indices(np.array([], dtype=bool)).size == 0
    assert list(rising_edge_indices(np.array([1, 1, 0], dtype=bool))) == [0]


def _synthetic_tape(n: int = 80) -> list[PairTape]:
    rng = np.random.default_rng(0)
    # Random-walk-ish OHLC
    close = 1.0 + np.cumsum(rng.normal(0, 0.001, size=n))
    open_ = np.roll(close, 1)
    open_[0] = close[0]
    high = np.maximum(open_, close) + 0.002
    low = np.minimum(open_, close) - 0.002
    atr = np.full(n, 0.005)
    return [
        PairTape(
            pair="EURUSD",
            open=open_,
            high=high,
            low=low,
            close=close,
            atr=atr,
            spread=0.0001,
            fallback=0.005,
        )
    ]


def test_simulate_trade_Rs_produces_finite():
    tapes = _synthetic_tape(100)
    # Edges every 10 bars
    edges = np.arange(0, 90, 10, dtype=np.int64)
    Rs = simulate_trade_Rs(edges, tapes, side=+1, sl_atr=1.5, tp_R=2.0, time_stop=10, max_concurrent=5)
    assert Rs.size > 0
    assert np.all(np.isfinite(Rs))
    # Short side also runs
    Rs_s = simulate_trade_Rs(edges, tapes, side=-1, sl_atr=1.5, tp_R=2.0, time_stop=10, max_concurrent=5)
    assert Rs_s.size > 0


def test_circular_shift_preserves_shape():
    n = 50
    cal = pd.date_range("2015-01-01", periods=n, freq="B")
    tapes = _synthetic_tape(n)
    atoms = pd.DataFrame({"moon_phase_new": np.zeros(n, dtype=bool)}, index=cal)
    st = SplitTape(
        name="validation",
        calendar=cal,
        pairs=tapes,
        atoms=atoms,
        start=cal[0],
        end=cal[-1],
    )
    shifted = circular_shift_tapes(st, 7)
    assert len(shifted) == 1
    assert shifted[0].open.shape == tapes[0].open.shape
    assert np.allclose(shifted[0].open, np.roll(tapes[0].open, 7), equal_nan=True)


def test_protocol_pass_fx_requires_all_bars():
    rr = FxSearchResult(
        rule="moon_phase_new",
        side="long",
        length=1,
        complexity=1,
        val_avg_R=0.1,
        val_total_R=5.0,
        val_n_trades=40,
        val_fitness=0.08,
        stable=True,
        holdout_avg_R=0.05,
        null_pvalue=0.01,
        null_n=2000,
    )
    assert protocol_pass_fx(rr)
    rr2 = FxSearchResult(
        rule="moon_phase_new",
        side="long",
        length=1,
        complexity=1,
        val_avg_R=0.1,
        val_total_R=5.0,
        val_n_trades=40,
        val_fitness=0.08,
        stable=True,
        holdout_avg_R=-0.01,
        null_pvalue=0.01,
        null_n=2000,
    )
    assert not protocol_pass_fx(rr2)
