# FX risk-management backtest framework

## Why FX ≠ index long/flat

The equity lattice/sweep stack treats a rule as a **binary position**: long when True, flat when False, with bps costs on flips. FX trading here is modelled as a **risk-managed trade engine**:

- Each **rising-edge** of a rule risks a **fixed fraction of equity** (`risk_pct`, default **0.5%**).
- Every trade has a **fixed stop-loss** and **take-profit** (and optional **time-stop**).
- Position size is chosen so that a stop hit loses ≈ `risk_pct × equity`.
- Spreads are charged in **pips**; **max concurrent trades** and **max total risk-on** (~5%) cap stacking.

This matches how discretionary/systematic FX books actually size and exit, and avoids the “always-in / all-in” bias of the index model.

## Pairs & data

| Pair | yfinance | Pip size |
|------|----------|----------|
| EURUSD | `EURUSD=X` | 0.0001 |
| GBPUSD | `GBPUSD=X` | 0.0001 |
| USDJPY | `USDJPY=X` | **0.01** |
| AUDUSD | `AUDUSD=X` | 0.0001 |
| USDCAD | `USDCAD=X` | 0.0001 |
| USDCHF | `USDCHF=X` | 0.0001 |
| NZDUSD | `NZDUSD=X` | 0.0001 |

Daily OHLC cached under `data/fx/{PAIR}.parquet`. Fetch:

```bash
python -m astro_market fetch-fx
```

**Actual data start** depends on yfinance history (often ~2003–2005 per pair). Folklore output prints the observed span. **Observed cache span (this box, after `fetch-fx`):** USDJPY from **2000-01-03**; most pairs from **2003-09/12**; AUDUSD from **2006-05-16**; through **2026-09-09**.

Protocol splits:

| Split | Dates | Role |
|-------|-------|------|
| Train | 2000-01-01 → 2014-12-31 | Research / fitting (truncated by data start) |
| Validation | 2015-01-01 → 2019-12-31 | Selection |
| Holdout | 2020-01-01 → present | **Report only — NEVER tune** |

## Lot / pip-value math (USD account)

A **unit** = 1 unit of base currency. A **standard lot** = 100,000 units. The engine sizes in fractional units so risk matches exactly (no lot rounding).

**Pip value (USD) for `units` of base**

1. **Quote = USD** (EURUSD, GBPUSD, AUDUSD, NZDUSD):

   `pip_value = units × pip_size`  
   e.g. 100,000 × 0.0001 = **$10 / pip / lot**

2. **Base = USD** (USDJPY, USDCAD, USDCHF):

   `pip_value = units × pip_size / price`  
   e.g. USDJPY @ 150: 100,000 × 0.01 / 150 ≈ **$6.67 / pip / lot**

**Sizing**

```text
risk_usd     = equity × risk_pct
sl_distance  = sl_atr × ATR(14)     # else sl_pips_fallback in price
units        = risk_usd / usd_loss_per_unit_at_sl
```

where `usd_loss_per_unit_at_sl = sl_distance` (quote=USD) or `sl_distance / price` (base=USD).

## Trade lifecycle (no lookahead)

1. Evaluate boolean rule on atoms (reuse `astro_market.rules` + atoms cache).
2. **Rising edge** only: `signal & ~signal.shift(1)` (not every True bar).
3. On signal bar `t`, record SL distance from **ATR known at `t`**.
4. Enter at bar **`t+1` open** (adverse half-spread).
5. Manage with bar high/low: SL, TP (= `tp_R ×` SL distance), or time-stop (default 10 bars @ close).
6. If SL and TP both touched same bar → **SL wins** (conservative).
7. Caps: `max_concurrent`, `max_total_risk_pct` (~5%).

Long and/or short are configurable (`side: long|short|both`). Shorts use the rising edge of `~rule`.

## Strategy object

```text
FxStrategy(rule, side, risk_pct, sl_atr, tp_R, atr_period, sl_pips_fallback, time_stop)
```

Defaults from `configs/fx.yaml`: risk 0.5%, SL 1.5×ATR(14), TP 2R, time-stop 10.

## Layout

```text
src/astro_market/fx/
  data.py       # yfinance OHLC + ATR + cache
  sizing.py     # pip / lot math, units_for_risk
  strategy.py   # FxStrategy + rising_edge
  broker.py     # FxBrokerSim event loop
  evaluate.py   # FxMetrics, split evaluation
  folklore.py   # fx-folklore / fx-sim CLIs
configs/fx.yaml
tests/fx/       # offline sizing, JPY, edge, TP/SL, lookahead, caps
```

## Commands

```bash
python -m astro_market fetch-fx
python -m astro_market fx-folklore
python -m astro_market fx-sim --rule moon_phase_new --side long
pytest tests/fx -q
```

Equity lattice / sweep code is untouched; this package sits alongside it.
