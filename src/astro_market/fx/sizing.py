"""Pip geometry and position sizing for a USD-account FX book.

Lot / pip-value math (USD account)
----------------------------------
A *unit* is one unit of the **base** currency (first in the pair).
A *standard lot* is 100_000 units. We size in fractional units (not forced
to round lots) so risk matches ``risk_pct × equity`` exactly.

**Pip size**
- JPY pairs (``*JPY``): 0.01
- All other majors here: 0.0001

**USD value of one pip for ``units`` of base**

1. Quote currency is USD (EURUSD, GBPUSD, AUDUSD, NZDUSD)::

       pip_value_usd = units * pip_size
       # e.g. 100_000 * 0.0001 = $10 / pip / standard lot

2. Base currency is USD (USDJPY, USDCAD, USDCHF)::

       pip_value_usd = units * pip_size / price
       # price is the pair quote (USDJPY ≈ 150 → ~$6.67/pip/lot)

P&L in USD for a price move ``Δ`` (in price units)::

    quote=USD:  pnl = units * Δ
    base=USD:   pnl = units * Δ / price   (mark at exit mid)

Position sizing so that a stop at ``sl_distance`` (price units) loses
approximately ``risk_usd = risk_pct × equity``::

    usd_per_unit_at_sl = sl_distance          # quote=USD
    usd_per_unit_at_sl = sl_distance / price  # base=USD
    units = risk_usd / usd_per_unit_at_sl
"""

from __future__ import annotations

import math

STANDARD_LOT = 100_000.0


def normalize_pair(pair: str) -> str:
    return pair.upper().replace("=X", "").replace("/", "").strip()


def is_jpy_pair(pair: str) -> bool:
    return normalize_pair(pair).endswith("JPY")


def pip_size(pair: str) -> float:
    """Pip size in price units: 0.01 for JPY pairs, else 0.0001."""
    return 0.01 if is_jpy_pair(pair) else 0.0001


def quote_is_usd(pair: str) -> bool:
    """True when quote (second) currency is USD — P&L is directly in USD."""
    p = normalize_pair(pair)
    return p.endswith("USD") and not p.startswith("USD")


def base_is_usd(pair: str) -> bool:
    p = normalize_pair(pair)
    return p.startswith("USD")


def price_to_pips(pair: str, price_distance: float) -> float:
    """Convert a price distance to pips."""
    ps = pip_size(pair)
    if ps <= 0:
        raise ValueError("pip size must be positive")
    return float(price_distance) / ps


def pips_to_price(pair: str, pips: float) -> float:
    """Convert pips to a price distance."""
    return float(pips) * pip_size(pair)


def usd_per_price_unit(pair: str, price: float, units: float = 1.0) -> float:
    """
    USD P&L for a +1.0 move in the pair quote, holding ``units`` of base.

    For quote=USD pairs this is simply ``units``.
    For base=USD pairs this is ``units / price``.
    """
    if price <= 0:
        raise ValueError(f"price must be positive, got {price}")
    u = float(units)
    if quote_is_usd(pair):
        return u
    if base_is_usd(pair):
        return u / float(price)
    # Non-USD crosses: approximate via quote/price conversion.
    return u / float(price)


def pip_value_usd(pair: str, price: float, units: float = STANDARD_LOT) -> float:
    """
    USD value of a one-pip move for ``units`` of base currency.

    Documented formulas:
    - EURUSD etc.: ``units * 0.0001``
    - USDJPY: ``units * 0.01 / price``
    """
    return usd_per_price_unit(pair, price, units=units) * pip_size(pair)


def units_for_risk(
    pair: str,
    price: float,
    sl_distance: float,
    risk_usd: float,
) -> float:
    """
    Size position (base units) so that a move of ``sl_distance`` (price)
    loses approximately ``risk_usd``.

    ``units = risk_usd / usd_pnl_per_unit_at_sl``.
    Returns 0 if inputs are non-positive / non-finite.
    """
    if (
        risk_usd <= 0
        or sl_distance <= 0
        or price <= 0
        or not math.isfinite(risk_usd)
        or not math.isfinite(sl_distance)
        or not math.isfinite(price)
    ):
        return 0.0
    per_unit = usd_per_price_unit(pair, price, units=1.0) * sl_distance
    if per_unit <= 0:
        return 0.0
    return float(risk_usd) / per_unit


def risk_usd_for_units(
    pair: str,
    price: float,
    sl_distance: float,
    units: float,
) -> float:
    """USD risk if stop is hit for a given size."""
    return abs(float(units)) * usd_per_price_unit(pair, price, units=1.0) * abs(
        float(sl_distance)
    )


def pnl_usd(
    pair: str,
    entry: float,
    exit_price: float,
    units: float,
    side: int,
) -> float:
    """
    Mark-to-market P&L in USD.

    ``side``: +1 long base, -1 short base.
    Uses exit (or mark) price for base=USD conversion.
    """
    delta = (float(exit_price) - float(entry)) * int(side)
    mark = float(exit_price) if exit_price > 0 else float(entry)
    return usd_per_price_unit(pair, mark, units=float(units)) * delta
