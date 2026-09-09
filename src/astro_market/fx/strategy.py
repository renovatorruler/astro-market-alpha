"""FX strategy definition and rising-edge signal helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd

SideMode = Literal["long", "short", "both"]


def rising_edge(signal: pd.Series) -> pd.Series:
    """
    Rising-edge mask: True only on bars where rule becomes True (False→True).

    Not every True bar — only the first bar of each True run.
    First bar is an edge only if it is True (treated as rising from False).
    """
    s = signal.astype(bool)
    prev = s.shift(1).fillna(False).astype(bool)
    out = s & ~prev
    out.name = getattr(signal, "name", None) or "rising_edge"
    return out.astype(bool)


@dataclass(frozen=True)
class FxStrategy:
    """
    Strategy = (rule, side, risk_pct, sl_atr, tp_R).

    Parameters
    ----------
    rule:
        Boolean DSL expression over atoms (reuses ``astro_market.rules``).
    side:
        ``long``, ``short``, or ``both`` (long on rising edge; short on falling
        edge of the same rule — i.e. rising edge of ``~rule``).
    risk_pct:
        Fraction of equity risked per trade (default 0.005 = 0.5%).
    sl_atr:
        Stop distance = ``sl_atr × ATR(atr_period)`` in price units.
        If ATR is NaN, fall back to ``sl_pips_fallback`` pips.
    tp_R:
        Take-profit distance = ``tp_R ×`` stop distance (R-multiple).
    atr_period:
        ATR lookback (default 14).
    sl_pips_fallback:
        Fixed pip stop when ATR unavailable.
    time_stop:
        Optional max holding bars (None = disabled). Default 10.
    allow_long / allow_short:
        Fine-grained overrides; if set, take precedence over ``side``.
    """

    rule: str
    side: SideMode = "long"
    risk_pct: float = 0.005
    sl_atr: float = 1.5
    tp_R: float = 2.0
    atr_period: int = 14
    sl_pips_fallback: float = 50.0
    time_stop: int | None = 10
    allow_long: bool | None = None
    allow_short: bool | None = None

    def longs_enabled(self) -> bool:
        if self.allow_long is not None:
            return bool(self.allow_long)
        return self.side in ("long", "both")

    def shorts_enabled(self) -> bool:
        if self.allow_short is not None:
            return bool(self.allow_short)
        return self.side in ("short", "both")
