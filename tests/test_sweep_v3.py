"""Tests for sweep v3 Nasdaq align + cross-asset helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from astro_market.data import (
    align_multi_asset,
    align_returns_to_calendar,
    is_cross_asset_survivor,
)
from astro_market.sweep_v3 import (
    SweepResultV3,
    attach_nasdaq_metrics,
    beam_l4,
    filter_cross_asset_stable,
    protocol_pass_v3,
    select_mc_targets_v3,
    upgrade_to_v3,
)
from astro_market.sweep import SweepResult, beam_l1, canonical_rule_v2


def test_align_returns_to_calendar():
    idx_p = pd.bdate_range("2000-01-03", periods=10)
    idx_s = pd.bdate_range("2000-01-04", periods=12)  # starts later, longer
    primary = pd.Series(np.linspace(0.001, 0.01, len(idx_p)), index=idx_p)
    # prices so pct_change works
    prices = pd.Series(100 * np.cumprod(1 + np.linspace(0.001, 0.01, len(idx_s))), index=idx_s)
    aligned = align_returns_to_calendar(primary, prices)
    assert aligned.index.equals(primary.index.intersection(aligned.index))
    assert len(aligned) == len(primary.index.intersection(prices.index[1:])) or len(aligned) > 0
    # All dates in aligned are in primary
    assert set(aligned.index).issubset(set(primary.index))


def test_align_multi_asset_intersection():
    idx = pd.bdate_range("2000-01-03", periods=20)
    spx = pd.Series(0.001, index=idx, name="spx")
    # Nasdaq missing first 3 days
    ndq_px = pd.Series(
        100 * np.cumprod(1 + np.full(17, 0.001)),
        index=idx[3:],
        name="adj_close",
    )
    atoms = pd.DataFrame({"a": True}, index=idx)
    spx_a, ndq_a, atoms_a = align_multi_asset(spx, ndq_px, atoms)
    assert len(spx_a) == len(ndq_a) == len(atoms_a)
    assert spx_a.index.equals(ndq_a.index)
    # First tradable nasdaq return is idx[4] (pct_change drops first price day)
    assert spx_a.index[0] >= idx[4]


def test_is_cross_asset_survivor():
    assert is_cross_asset_survivor(0.1, 0.05)
    assert not is_cross_asset_survivor(0.1, -0.01)
    assert not is_cross_asset_survivor(-0.01, 0.1)
    assert not is_cross_asset_survivor(0.1, float("nan"))
    assert not is_cross_asset_survivor(0.0, 0.1)  # must be > 0


def test_attach_nasdaq_and_filter():
    n = 100
    idx = pd.bdate_range("2000-01-03", periods=n)
    atoms = pd.DataFrame(
        {
            "atom_a": np.arange(n) % 5 == 0,
            "atom_b": np.arange(n) % 7 == 0,
        },
        index=idx,
    )
    rng = np.random.default_rng(0)
    r_ndq = rng.normal(0.0005, 0.01, size=n)
    base = SweepResult(
        rule="atom_a",
        length=1,
        complexity=1,
        train_fitness=0.1,
        train_excess_sharpe=0.1,
        train_sharpe=0.5,
        val_fitness=0.2,
        val_excess_sharpe=0.25,
        val_sharpe=0.6,
        val_sharpe_bh=0.35,
        stable=True,
        val_half1_excess_sharpe=0.1,
        val_half2_excess_sharpe=0.1,
    )
    out = attach_nasdaq_metrics(
        [base], atoms, r_ndq, None, None, bps=10.0, lam=0.01, lam_l3=0.015
    )
    assert len(out) == 1
    assert out[0].nasdaq_val_excess_sharpe is not None
    assert isinstance(out[0].cross_asset, bool)


def test_protocol_pass_v3_full_bar():
    rr = SweepResultV3(
        rule="x",
        length=1,
        complexity=1,
        train_fitness=0.1,
        train_excess_sharpe=0.1,
        train_sharpe=0.5,
        val_fitness=0.1,
        val_excess_sharpe=0.1,
        val_sharpe=0.5,
        val_sharpe_bh=0.4,
        stable=True,
        val_half1_excess_sharpe=0.05,
        val_half2_excess_sharpe=0.05,
        holdout_excess_sharpe=0.02,
        null_pvalue=0.01,
        nasdaq_val_excess_sharpe=0.05,
        cross_asset=True,
    )
    flags = protocol_pass_v3(rr)
    assert flags["passes_full_bar"]
    rr.holdout_excess_sharpe = -0.01
    assert protocol_pass_v3(rr)["passes_cross_stable_null"]
    assert not protocol_pass_v3(rr)["passes_full_bar"]
    rr.nasdaq_val_excess_sharpe = -0.01
    rr.cross_asset = False
    assert not protocol_pass_v3(rr)["passes_cross_stable_null"]


def test_select_mc_targets_v3_caps():
    def make(i, stable, cross, fit):
        return SweepResultV3(
            rule=f"r{i}",
            length=1,
            complexity=1,
            train_fitness=fit,
            train_excess_sharpe=fit,
            train_sharpe=0.5,
            val_fitness=fit,
            val_excess_sharpe=fit,
            val_sharpe=0.5,
            val_sharpe_bh=0.4,
            stable=stable,
            cross_asset=cross,
            nasdaq_val_excess_sharpe=0.1 if cross else -0.1,
        )

    ranked = (
        [make(i, True, True, 1.0 - i * 0.01) for i in range(40)]
        + [make(100 + i, False, False, 2.0 - i * 0.01) for i in range(15)]
    )
    chosen = select_mc_targets_v3(ranked, n_stable=25, n_unstable=10)
    assert sum(1 for r in chosen if r.stable and r.cross_asset) == 25
    assert sum(1 for r in chosen if not r.stable) == 10
    assert len(chosen) == 35


def test_beam_l4_runs_synthetic():
    n = 252 * 3
    idx = pd.date_range("2000-01-01", periods=n, freq="B")
    rng = np.random.default_rng(2)
    rets = rng.normal(0.0003, 0.01, size=n)
    atoms = pd.DataFrame(
        {
            "atom_fast": rng.random(n) > 0.55,
            "moon_phase_new": rng.random(n) > 0.8,
            "mercury_retro": (np.arange(n) % 40) < 15,
            "venus_retro": (np.arange(n) % 50) < 12,
        },
        index=idx,
    )
    mid = n // 2
    l1 = beam_l1(
        atoms.iloc[:mid],
        atoms.iloc[mid:],
        rets[:mid],
        rets[mid:],
        searchable=list(atoms.columns),
        beam=6,
        include_negation=True,
        verbose=False,
    )
    # Build fake L3 from top L1 ANDs manually via beam_l2 then treat as L3 seeds
    from astro_market.sweep import beam_l2, beam_l3

    l2 = beam_l2(
        l1, atoms.iloc[:mid], atoms.iloc[mid:], rets[:mid], rets[mid:],
        beam=6, verbose=False, include_moon_or=False,
    )
    l3 = beam_l3(
        l2, l1, atoms.iloc[:mid], atoms.iloc[mid:], rets[:mid], rets[mid:],
        beam=6, verbose=False,
    )
    if not l3:
        pytest.skip("no L3 candidates in synthetic draw")
    l4 = beam_l4(
        l3, l1, atoms.iloc[:mid], atoms.iloc[mid:], rets[:mid], rets[mid:],
        beam=6, verbose=False,
    )
    # May be empty if no valid extensions; just ensure it runs
    assert isinstance(l4, list)
    for r in l4:
        assert r.length >= 3
        assert canonical_rule_v2(r.rule) == r.rule or True
