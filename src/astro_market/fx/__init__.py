"""FX risk-management backtest: fixed fractional risk, SL/TP, rising-edge entries.

Unlike the equity index long/flat model, each FX signal risks a fixed fraction
of equity with fixed stop-loss and take-profit. See results/fx_framework.md.
"""

from astro_market.fx.data import (
    FX_PAIRS,
    FXDataError,
    fetch_and_cache_fx,
    load_fx_ohlc,
    load_all_fx,
    atr,
)
from astro_market.fx.sizing import (
    pip_size,
    is_jpy_pair,
    pip_value_usd,
    units_for_risk,
    risk_usd_for_units,
)
from astro_market.fx.strategy import FxStrategy, rising_edge
from astro_market.fx.broker import FxBrokerSim, Trade, SimResult
from astro_market.fx.evaluate import FxMetrics, evaluate_fx_strategy, evaluate_on_fx_split

__all__ = [
    "FX_PAIRS",
    "FXDataError",
    "fetch_and_cache_fx",
    "load_fx_ohlc",
    "load_all_fx",
    "atr",
    "pip_size",
    "is_jpy_pair",
    "pip_value_usd",
    "units_for_risk",
    "risk_usd_for_units",
    "FxStrategy",
    "rising_edge",
    "FxBrokerSim",
    "Trade",
    "SimResult",
    "FxMetrics",
    "evaluate_fx_strategy",
    "evaluate_on_fx_split",
]
