"""Unit tests for rule DSL parsing and evaluation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from astro_market.rules import (
    RuleParseError,
    evaluate_ast,
    evaluate_rule_signal,
    parse_rule,
    rule_complexity,
)


def _atoms(n: int = 10) -> pd.DataFrame:
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "moon_phase_new": rng.random(n) > 0.7,
            "moon_phase_full": rng.random(n) > 0.7,
            "mercury_retro": rng.random(n) > 0.5,
        },
        index=idx,
    )


def test_parse_simple_atom():
    p = parse_rule("moon_phase_new")
    assert p.atoms == ["moon_phase_new"]
    assert "moon_phase_new" in p.py_expr


def test_parse_and_or_not():
    p = parse_rule("moon_phase_new | moon_phase_full")
    assert set(p.atoms) == {"moon_phase_new", "moon_phase_full"}
    p2 = parse_rule("~mercury_retro")
    assert p2.atoms == ["mercury_retro"]
    p3 = parse_rule("(moon_phase_new & mercury_retro) | ~moon_phase_full")
    assert "moon_phase_new" in p3.atoms
    assert "not" in p3.py_expr


def test_parse_rejects_bad_syntax():
    with pytest.raises(RuleParseError):
        parse_rule("")
    with pytest.raises(RuleParseError):
        parse_rule("moon_phase_new ^ mercury_retro")
    with pytest.raises(RuleParseError):
        parse_rule("1 + 2")
    with pytest.raises(RuleParseError):
        parse_rule("__import__('os')")


def test_evaluate_signal():
    atoms = _atoms(20)
    s = evaluate_rule_signal("moon_phase_new", atoms)
    assert s.dtype == bool
    assert s.equals(atoms["moon_phase_new"])

    s2 = evaluate_rule_signal("~mercury_retro", atoms)
    assert (s2 == ~atoms["mercury_retro"]).all()

    s3 = evaluate_rule_signal("moon_phase_new | moon_phase_full", atoms)
    assert (s3 == (atoms["moon_phase_new"] | atoms["moon_phase_full"])).all()

    s4 = evaluate_rule_signal("moon_phase_new & mercury_retro", atoms)
    assert (s4 == (atoms["moon_phase_new"] & atoms["mercury_retro"])).all()


def test_missing_atom_raises():
    atoms = _atoms()
    with pytest.raises(KeyError):
        evaluate_rule_signal("venus_retro", atoms)


def test_complexity():
    assert rule_complexity("moon_phase_new") == 1
    assert rule_complexity("~mercury_retro") == 2  # 1 atom + 1 not
    c = rule_complexity("moon_phase_new | moon_phase_full")
    assert c == 3  # 2 atoms + 1 or
