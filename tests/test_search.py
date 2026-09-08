"""Unit tests for lattice search helpers (offline, synthetic data)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from astro_market.search import (
    RuleResult,
    canonical_rule,
    length1_search,
    length2_and_search,
    metrics_from_signal,
    monte_carlo_null_for_rule,
    mutually_exclusive,
    protocol_pass,
    select_mc_survivors,
    signal_for_expr,
    strategy_and_bh_returns,
)


@pytest.fixture
def synthetic():
    n = 252 * 4
    idx = pd.date_range("2000-01-01", periods=n, freq="B")
    rng = np.random.default_rng(0)
    rets = rng.normal(0.0004, 0.01, size=n)
    a = rng.random(n) > 0.6
    b = rng.random(n) > 0.6
    excl_x = np.zeros(n, dtype=bool)
    excl_y = np.zeros(n, dtype=bool)
    excl_x[::2] = True
    excl_y[1::2] = True
    atoms = pd.DataFrame(
        {
            "atom_a": a,
            "atom_b": b,
            "excl_x": excl_x,
            "excl_y": excl_y,
            "moon_phase_new": rng.random(n) > 0.75,
            "moon_phase_full": rng.random(n) > 0.75,
        },
        index=idx,
    )
    return rets, atoms


def test_canonical_rule_commutative():
    assert canonical_rule("b & a") == canonical_rule("a & b")
    assert canonical_rule("moon_phase_full | moon_phase_new") == canonical_rule(
        "moon_phase_new | moon_phase_full"
    )
    assert canonical_rule("~mercury_retro") == "~mercury_retro"


def test_mutually_exclusive(synthetic):
    _, atoms = synthetic
    assert mutually_exclusive(
        atoms["excl_x"].to_numpy(), atoms["excl_y"].to_numpy()
    )
    assert not mutually_exclusive(
        atoms["atom_a"].to_numpy(), atoms["atom_b"].to_numpy()
    )


def test_signal_for_expr(synthetic):
    _, atoms = synthetic
    s = signal_for_expr("atom_a & atom_b", atoms)
    assert (s == (atoms["atom_a"] & atoms["atom_b"]).to_numpy()).all()
    s2 = signal_for_expr("~atom_a", atoms)
    assert (s2 == (~atoms["atom_a"]).to_numpy()).all()
    s3 = signal_for_expr("moon_phase_new | moon_phase_full", atoms)
    assert (
        s3 == (atoms["moon_phase_new"] | atoms["moon_phase_full"]).to_numpy()
    ).all()


def test_metrics_match_sign_conventions(synthetic):
    rets, atoms = synthetic
    sig = atoms["atom_a"].to_numpy()
    m = metrics_from_signal(
        sig, rets, complexity=1, bps_per_side=10.0, lambda_complexity=0.01
    )
    assert "fitness" in m and "excess_sharpe" in m
    assert m["n_days"] == len(rets)
    always = np.ones(len(rets), dtype=bool)
    m2 = metrics_from_signal(always, rets, complexity=1, bps_per_side=10.0)
    assert abs(m2["excess_sharpe"]) < 1e-9


def test_strategy_costs_bite(synthetic):
    rets, _ = synthetic
    sig = np.zeros(len(rets), dtype=bool)
    sig[::2] = True
    strat0, _ = strategy_and_bh_returns(sig, rets, bps_per_side=0.0)
    strat1, _ = strategy_and_bh_returns(sig, rets, bps_per_side=10.0)
    assert strat1.sum() < strat0.sum()


def test_length1_and_length2_search(synthetic):
    rets, atoms = synthetic
    mid = len(rets) // 2
    r_tr, r_va = rets[:mid], rets[mid:]
    a_tr, a_va = atoms.iloc[:mid], atoms.iloc[mid:]
    l1 = length1_search(
        a_tr, a_va, r_tr, r_va, include_negation=True, bps=10.0, lam=0.01
    )
    assert len(l1) == 2 * len(atoms.columns)
    assert l1[0].val_fitness >= l1[-1].val_fitness
    l2 = length2_and_search(
        l1, a_tr, a_va, r_tr, r_va, top_k=4, bps=10.0, lam=0.01, skip_mutex=True
    )
    for rr in l2:
        assert rr.length == 2
        assert "&" in rr.rule


def test_select_mc_survivors_includes_positive_xs():
    def mk(rule, fit, xs):
        return RuleResult(
            rule=rule,
            length=1,
            complexity=1,
            train_fitness=fit,
            train_excess_sharpe=xs,
            train_sharpe=0.0,
            val_fitness=fit,
            val_excess_sharpe=xs,
            val_sharpe=0.5,
            val_sharpe_bh=0.4,
        )

    ranked = [
        mk("a", 0.05, 0.06),
        mk("b", 0.04, 0.05),
        mk("c", -0.01, 0.02),
        mk("d", -0.5, -0.4),
    ]
    surv = select_mc_survivors(ranked, n_cap=10, hard_cap=20)
    names = {r.rule for r in surv}
    assert "a" in names and "b" in names and "c" in names


def test_protocol_pass_requires_null():
    rr = RuleResult(
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
        holdout_excess_sharpe=0.05,
        null_pvalue=0.2,
    )
    flags = protocol_pass(rr)
    assert flags["val_interesting"]
    assert not flags["null_reject"]
    assert not flags["passes_protocol_bar"]
    rr.null_pvalue = 0.01
    assert protocol_pass(rr)["passes_protocol_bar"]


def test_monte_carlo_null_runs(synthetic):
    rets, atoms = synthetic
    mid = len(rets) // 2
    r_va = rets[mid:]
    a_va = atoms.iloc[mid:]
    p = monte_carlo_null_for_rule(
        "atom_a", a_va, r_va, n=50, seed=1, bps=10.0, lam=0.01
    )
    assert 0.0 < p <= 1.0
