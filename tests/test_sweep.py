"""Tests for sweep v2 episode counting + beam/regime helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from astro_market.atoms import (
    atom_registry_v2,
    build_atoms_v2_from_longitudes,
    is_regime_sign_atom,
    searchable_atom_names,
)
from astro_market.ephemeris import ALL_PLANETS_V2, SIGNS
from astro_market.sweep import (
    apply_regime_filter,
    beam_l1,
    canonical_rule_v2,
    count_episodes,
    max_episode_length,
    episode_stats,
    length_of_rule,
    protocol_pass_v2,
    rule_is_pure_regime_sign,
    SweepResult,
)


def test_count_episodes_basic():
    m = np.array([0, 1, 1, 0, 1, 0, 0, 1, 1, 1], dtype=bool)
    assert count_episodes(m) == 3
    assert max_episode_length(m) == 3
    n, mx = episode_stats(m)
    assert n == 3 and mx == 3


def test_count_episodes_edge_cases():
    assert count_episodes(np.array([], dtype=bool)) == 0
    assert count_episodes(np.zeros(10, dtype=bool)) == 0
    assert max_episode_length(np.zeros(10, dtype=bool)) == 0
    all_t = np.ones(50, dtype=bool)
    assert count_episodes(all_t) == 1
    assert max_episode_length(all_t) == 50
    # Starts True
    m = np.array([1, 1, 0, 1], dtype=bool)
    assert count_episodes(m) == 2
    assert max_episode_length(m) == 2


def test_regime_filter_long_episode():
    # Single episode of 500 days -> hard drop
    sig = np.zeros(1000, dtype=bool)
    sig[100:600] = True
    info = apply_regime_filter("mercury_retro", sig)
    assert info.dropped
    assert "max_episode" in info.drop_reason


def test_regime_filter_pure_outer_sign():
    sig = np.zeros(500, dtype=bool)
    sig[::10] = True  # many short episodes
    info = apply_regime_filter("saturn_sign_pisces", sig)
    assert info.dropped
    assert info.drop_reason == "pure_outer_sign_regime"
    assert rule_is_pure_regime_sign("saturn_sign_pisces")
    assert rule_is_pure_regime_sign("~jupiter_sign_aries & saturn_sign_pisces")
    assert not rule_is_pure_regime_sign("mercury_retro & saturn_sign_pisces")


def test_regime_filter_few_episodes_penalty():
    sig = np.zeros(800, dtype=bool)
    sig[10:50] = True
    sig[100:140] = True
    sig[200:230] = True  # only 3 episodes, each < 400
    train = np.zeros(800, dtype=bool)
    train[0:20] = True  # +1 episode on train -> train+val still < 8 if we only add 1
    info = apply_regime_filter("mercury_retro", sig, train)
    assert not info.dropped
    assert info.penalty > 0


def test_canonical_and_length():
    assert canonical_rule_v2("b & a & c") == canonical_rule_v2("c & a & b")
    assert length_of_rule("a") == 1
    assert length_of_rule("~a") == 1
    assert length_of_rule("a & b") == 2
    assert length_of_rule("a & b & c") == 3


def test_registry_v2_size_and_searchable():
    reg = atom_registry_v2()
    assert 800 <= len(reg) <= 2500
    search = searchable_atom_names(reg)
    assert all(not is_regime_sign_atom(c) for c in search)
    assert any(is_regime_sign_atom(c) for c in reg)
    assert any(c.endswith("_orb1") for c in reg)
    assert any(c.endswith("_orb5") for c in reg)
    assert "moon_phase_waxing_crescent" in reg
    assert "uranus_retro" in reg
    assert "pluto_station" in reg


def test_build_atoms_v2_from_mock():
    n = 60
    idx = pd.date_range("2020-01-01", periods=n, freq="D")
    t = np.arange(n, dtype=float)
    data = {p: (i * 37.0 + 0.5 * t) % 360.0 for i, p in enumerate(ALL_PLANETS_V2)}
    # Faster moon
    data["moon"] = (50.0 + 13.0 * t) % 360.0
    data["sun"] = (280.0 + t) % 360.0
    lon = pd.DataFrame(data, index=idx)
    atoms = build_atoms_v2_from_longitudes(lon)
    assert atoms.shape[0] == n
    assert atoms.shape[1] >= 800
    # 8 moon phases partition
    phase_cols = [c for c in atoms.columns if c.startswith("moon_phase_")]
    assert len(phase_cols) == 8
    assert (atoms[phase_cols].sum(axis=1) == 1).all()
    # Outer signs present but searchable excludes them
    assert "saturn_sign_pisces" in atoms.columns or any(
        c.startswith("saturn_sign_") for c in atoms.columns
    )
    search = searchable_atom_names(atoms.columns)
    assert "uranus_retro" in search or "uranus_station" in search
    assert not any(c.startswith("saturn_sign_") for c in search)


def test_beam_l1_runs_synthetic():
    n = 252 * 3
    idx = pd.date_range("2000-01-01", periods=n, freq="B")
    rng = np.random.default_rng(1)
    rets = rng.normal(0.0003, 0.01, size=n)
    # Many short episodes
    atoms = pd.DataFrame(
        {
            "atom_fast": rng.random(n) > 0.55,
            "moon_phase_new": rng.random(n) > 0.8,
            "mercury_retro": (np.arange(n) % 40) < 15,
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
        beam=10,
        include_negation=True,
        verbose=False,
    )
    assert len(l1) >= 1
    assert l1[0].val_fitness >= l1[-1].val_fitness
    assert all(r.max_episode_val <= 400 for r in l1)


def test_protocol_pass_v2_requires_stable_and_null():
    rr = SweepResult(
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
        null_pvalue=0.2,
    )
    flags = protocol_pass_v2(rr)
    assert flags["stable_halves"]
    assert not flags["null_reject"]
    assert not flags["passes_stable_bar"]
    rr.null_pvalue = 0.01
    assert protocol_pass_v2(rr)["passes_stable_bar"]
    assert protocol_pass_v2(rr)["passes_with_holdout"]
