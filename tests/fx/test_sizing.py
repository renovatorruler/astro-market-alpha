"""Offline tests: pip geometry, JPY pip value, risk sizing."""

from __future__ import annotations

import math

import pytest

from astro_market.fx.sizing import (
    STANDARD_LOT,
    is_jpy_pair,
    pip_size,
    pip_value_usd,
    pnl_usd,
    units_for_risk,
    risk_usd_for_units,
)


def test_pip_size_jpy_vs_others():
    assert pip_size("USDJPY") == 0.01
    assert pip_size("EURUSD") == 0.0001
    assert pip_size("GBPUSD") == 0.0001
    assert is_jpy_pair("USDJPY")
    assert not is_jpy_pair("EURUSD")


def test_pip_value_eurusd_standard_lot():
    # 100k * 0.0001 = $10 / pip
    assert pip_value_usd("EURUSD", price=1.10, units=STANDARD_LOT) == pytest.approx(10.0)


def test_pip_value_usdjpy():
    # 100k * 0.01 / 150 = 1000/150 ≈ 6.666...
    pv = pip_value_usd("USDJPY", price=150.0, units=STANDARD_LOT)
    assert pv == pytest.approx(1000.0 / 150.0)


def test_units_for_risk_matches_risk_usd():
    price = 1.1000
    sl = 0.0015  # 15 pips
    risk = 500.0  # $500
    units = units_for_risk("EURUSD", price, sl, risk)
    # EURUSD: units * sl = risk → units = risk/sl
    assert units == pytest.approx(risk / sl)
    assert risk_usd_for_units("EURUSD", price, sl, units) == pytest.approx(risk)


def test_units_for_risk_jpy():
    price = 150.0
    sl = 0.75  # 75 pips
    risk = 500.0
    units = units_for_risk("USDJPY", price, sl, risk)
    # usd per unit at sl = sl / price
    assert units == pytest.approx(risk / (sl / price))
    assert risk_usd_for_units("USDJPY", price, sl, units) == pytest.approx(risk)


def test_pnl_usd_quote_and_base():
    # Long EURUSD +100 pips on 10k units: 10000 * 0.01 = 100
    assert pnl_usd("EURUSD", 1.10, 1.11, 10_000, +1) == pytest.approx(100.0)
    # Long USDJPY: units * delta / exit
    # 100000 * (151-150) / 151
    assert pnl_usd("USDJPY", 150.0, 151.0, 100_000, +1) == pytest.approx(
        100_000 * 1.0 / 151.0
    )


def test_units_nonpositive_returns_zero():
    assert units_for_risk("EURUSD", 1.1, 0.0, 100) == 0.0
    assert units_for_risk("EURUSD", 1.1, 0.01, 0.0) == 0.0
    assert units_for_risk("EURUSD", -1.0, 0.01, 100) == 0.0
