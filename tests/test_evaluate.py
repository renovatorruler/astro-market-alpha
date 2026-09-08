"""Offline evaluation tests on synthetic returns/atoms."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from astro_market.evaluate import (
    apply_costs,
    embargo_split,
    evaluate_rule,
    fitness,
    positions_from_signal,
)


@pytest.fixture
def synthetic():
    n = 252 * 3
    idx = pd.date_range("2000-01-01", periods=n, freq="B")
    rng = np.random.default_rng(42)
    # Mild positive drift
    rets = pd.Series(rng.normal(0.0003, 0.01, size=n), index=idx, name="ret")
    atoms = pd.DataFrame(
        {
            "moon_phase_new": rng.random(n) > 0.75,
            "always_true": np.ones(n, dtype=bool),
            "always_false": np.zeros(n, dtype=bool),
            "mercury_retro": rng.random(n) > 0.5,
        },
        index=idx,
    )
    return rets, atoms


def test_always_long_matches_bh(synthetic):
    rets, atoms = synthetic
    m = evaluate_rule("always_true", rets, atoms, bps_per_side=10.0)
    # Strategy always long: same as BH after initial entry cost
    assert abs(m.sharpe - m.sharpe_bh) < 1e-9
    assert abs(m.excess_sharpe) < 1e-9
    assert m.complexity == 1


def test_always_flat_near_zero_return(synthetic):
    rets, atoms = synthetic
    m = evaluate_rule("always_false", rets, atoms, bps_per_side=0.0)
    assert abs(m.total_return) < 1e-12
    assert m.n_trades == 0


def test_costs_reduce_return(synthetic):
    rets, atoms = synthetic
    m0 = evaluate_rule("mercury_retro", rets, atoms, bps_per_side=0.0)
    m1 = evaluate_rule("mercury_retro", rets, atoms, bps_per_side=10.0)
    # With costs, total return should be lower (or equal if no trades)
    assert m1.total_return <= m0.total_return + 1e-12
    assert m1.n_trades >= 1


def test_fitness_penalizes_complexity(synthetic):
    rets, atoms = synthetic
    m = evaluate_rule("mercury_retro", rets, atoms)
    f0 = fitness(m, lambda_complexity=0.0)
    f1 = fitness(m, lambda_complexity=0.5)
    assert f1 < f0


def test_embargo_split():
    idx = pd.date_range("2000-01-01", periods=20, freq="B")
    train_end = idx[9]
    train, val = embargo_split(idx, train_end, embargo_days=5)
    assert len(train) == 10
    assert train[-1] == train_end
    # 5 days embargo after train
    assert val[0] == idx[15]


def test_apply_costs_unit():
    idx = pd.date_range("2020-01-01", periods=5, freq="B")
    pos = pd.Series([1.0, 1.0, 0.0, 0.0, 1.0], index=idx)
    rets = pd.Series([0.01, 0.01, 0.01, 0.01, 0.01], index=idx)
    net = apply_costs(pos, rets, bps_per_side=10.0)
    # Day0: enter from 0->1 via lag fill 0, then lag becomes... 
    # pos_lag = [0, 1, 1, 0, 0]; gross = pos_lag * r
    # delta on pos_lag: first=|0|, then 1, 0, 1, 0
    assert len(net) == 5
    assert net.notna().all()
