"""Rising-edge entry helper."""

from __future__ import annotations

import pandas as pd

from astro_market.fx.strategy import rising_edge


def test_rising_edge_only_transitions():
    idx = pd.date_range("2020-01-01", periods=8, freq="D")
    s = pd.Series([0, 1, 1, 0, 0, 1, 0, 1], index=idx, dtype=bool)
    e = rising_edge(s)
    assert list(e.astype(int)) == [0, 1, 0, 0, 0, 1, 0, 1]


def test_rising_edge_first_true_counts():
    idx = pd.date_range("2020-01-01", periods=3, freq="D")
    s = pd.Series([True, True, False], index=idx)
    e = rising_edge(s)
    assert list(e.astype(int)) == [1, 0, 0]
