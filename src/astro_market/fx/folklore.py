"""FX folklore smoke: rising-edge entries with default risk / TP / SL."""

from __future__ import annotations

import sys
from typing import Any

import click
import pandas as pd

from astro_market.atoms import load_atoms
from astro_market.fx.broker import FxBrokerSim
from astro_market.fx.data import (
    FXDataError,
    FX_PAIRS,
    data_start_end,
    load_all_fx,
    load_fx_config,
)
from astro_market.fx.evaluate import FxMetrics, evaluate_on_fx_split
from astro_market.fx.strategy import FxStrategy


FOLKLORE_RULES = [
    "moon_phase_new",
    "moon_phase_full",
    "mercury_retro",
    "~mercury_sign_aquarius",
]


def _fmt(x: float, pct: bool = False) -> str:
    if x is None or (isinstance(x, float) and (pd.isna(x) or not abs(x) < 1e100)):
        return "nan"
    if pct:
        return f"{100 * x:8.2f}%"
    return f"{x:8.3f}"


def _broker_from_cfg(cfg: dict) -> FxBrokerSim:
    fx = cfg.get("fx", cfg)
    risk = fx.get("risk", {})
    costs = fx.get("costs", {})
    return FxBrokerSim(
        initial_equity=float(fx.get("initial_equity", 100_000.0)),
        spread_pips=float(costs.get("spread_pips", 1.0)),
        max_concurrent=int(risk.get("max_concurrent", 5)),
        max_total_risk_pct=float(risk.get("max_total_risk_pct", 0.05)),
    )


def _default_strategy(rule: str, cfg: dict) -> FxStrategy:
    fx = cfg.get("fx", cfg)
    risk = fx.get("risk", {})
    return FxStrategy(
        rule=rule,
        side=str(fx.get("default_side", "long")),
        risk_pct=float(risk.get("risk_pct", 0.005)),
        sl_atr=float(risk.get("sl_atr", 1.5)),
        tp_R=float(risk.get("tp_R", 2.0)),
        atr_period=int(risk.get("atr_period", 14)),
        sl_pips_fallback=float(risk.get("sl_pips_fallback", 50.0)),
        time_stop=risk.get("time_stop", 10),
    )


def run_fx_folklore(
    rules: list[str] | None = None,
    cfg: dict | None = None,
    pairs: list[str] | None = None,
    auto_fetch: bool = True,
) -> pd.DataFrame:
    """
    Evaluate folklore rules on FX train / validation / holdout.

    Rising-edge entries, default risk/TP/SL. Never tunes on holdout.
    """
    cfg = cfg or load_fx_config()
    fx = cfg.get("fx", cfg)
    rules = rules or list(fx.get("folklore_rules") or FOLKLORE_RULES)
    pairs = pairs or list(fx.get("pairs") or FX_PAIRS)

    ohlc = load_all_fx(pairs=pairs, cfg=cfg, auto_fetch=auto_fetch)
    # Document actual data start across pairs
    starts = []
    ends = []
    for p, df in ohlc.items():
        if len(df):
            s, e = data_start_end(df)
            starts.append(s)
            ends.append(e)
    data_start = min(starts) if starts else None
    data_end = max(ends) if ends else None

    # Atoms on union of FX calendars (reuse equity atoms cache; align by date)
    all_idx = None
    for df in ohlc.values():
        all_idx = df.index if all_idx is None else all_idx.union(df.index)
    atoms = load_atoms(dates=all_idx, auto_build=True)

    broker = _broker_from_cfg(cfg)
    rows: list[dict[str, Any]] = []
    for rule in rules:
        strat = _default_strategy(rule, cfg)
        for split in ("train", "validation", "holdout"):
            m = evaluate_on_fx_split(
                strat, ohlc, atoms, split, cfg=cfg, broker=broker, pairs=list(pairs)
            )
            rows.append(metrics_row(rule, split, m, data_start, data_end))
    return pd.DataFrame(rows)


def metrics_row(
    rule: str,
    split: str,
    m: FxMetrics,
    data_start: pd.Timestamp | None = None,
    data_end: pd.Timestamp | None = None,
) -> dict[str, Any]:
    return {
        "rule": rule,
        "split": split,
        "total_ret": m.total_return,
        "cagr": m.cagr,
        "sharpe": m.sharpe,
        "max_dd": m.max_dd,
        "n_trades": m.n_trades,
        "win_rate": m.win_rate,
        "avg_R": m.avg_R,
        "profit_factor": m.profit_factor,
        "pct_sl": m.pct_sl,
        "pct_tp": m.pct_tp,
        "pct_time": m.pct_time,
        "n_days": m.n_days,
        "final_equity": m.final_equity,
        "data_start": str(data_start.date()) if data_start is not None else None,
        "data_end": str(data_end.date()) if data_end is not None else None,
    }


def print_table(df: pd.DataFrame) -> None:
    if df.empty:
        print("No FX folklore results.")
        return
    header = (
        f"{'rule':<28} {'split':<12} {'CAGR':>10} {'Sharpe':>8} {'MaxDD':>10} "
        f"{'Trades':>7} {'Win%':>8} {'avgR':>8} {'PF':>8}"
    )
    print(header)
    print("-" * len(header))
    for _, row in df.iterrows():
        print(
            f"{str(row['rule']):<28} {str(row['split']):<12} "
            f"{_fmt(row['cagr'], pct=True):>10} {_fmt(row['sharpe']):>8} "
            f"{_fmt(row['max_dd'], pct=True):>10} {int(row['n_trades']):>7d} "
            f"{_fmt(row['win_rate'], pct=True):>8} {_fmt(row['avg_R']):>8} "
            f"{_fmt(row['profit_factor']):>8}"
        )
    ds = df["data_start"].iloc[0] if "data_start" in df.columns else "?"
    de = df["data_end"].iloc[0] if "data_end" in df.columns else "?"
    print()
    print(
        f"FX protocol: train ~2000–2014 | validation 2015–2019 | holdout 2020–present. "
        f"Actual cached FX span: {ds} → {de}. NEVER tune on holdout."
    )
    print(
        "Model: rising-edge entries @ next open; risk=0.5% equity/trade; "
        "SL=1.5×ATR(14); TP=2R; time-stop=10; spreads in pips; concurrent/risk caps."
    )


@click.command()
@click.option("--config", "config_path", default=None, help="Path to configs/fx.yaml")
@click.option("--rule", "extra_rules", multiple=True, help="Rule expression (repeatable)")
@click.option("--pair", "pairs", multiple=True, help="FX pair code (repeatable)")
@click.option("--no-fetch", is_flag=True, help="Do not auto-fetch; require parquet cache")
def main(
    config_path: str | None,
    extra_rules: tuple[str, ...],
    pairs: tuple[str, ...],
    no_fetch: bool,
) -> None:
    """FX folklore smoke test (rising-edge, default risk/TP/SL)."""
    try:
        cfg = load_fx_config(config_path)
        rules = list(extra_rules) if extra_rules else None
        pair_list = list(pairs) if pairs else None
        df = run_fx_folklore(
            rules=rules, cfg=cfg, pairs=pair_list, auto_fetch=not no_fetch
        )
        print_table(df)
    except FXDataError as e:
        print(f"ERROR: Could not load FX data.\n{e}", file=sys.stderr)
        print(
            "Hint: ensure network for yfinance, or place caches under data/fx/*.parquet "
            "(EURUSD.parquet, …). Run: python -m astro_market fetch-fx",
            file=sys.stderr,
        )
        sys.exit(2)
    except Exception as e:
        print(f"ERROR: {type(e).__name__}: {e}", file=sys.stderr)
        sys.exit(1)


# ---------------------------------------------------------------------------
# Single-rule CLI: fx-sim
# ---------------------------------------------------------------------------

@click.command()
@click.option("--rule", required=True, help="Boolean DSL rule expression")
@click.option("--side", type=click.Choice(["long", "short", "both"]), default="long")
@click.option("--risk-pct", type=float, default=None, help="Override risk fraction")
@click.option("--sl-atr", type=float, default=None)
@click.option("--tp-r", "tp_R", type=float, default=None)
@click.option("--pair", "pairs", multiple=True, help="Limit to these pairs")
@click.option("--config", "config_path", default=None)
@click.option("--split", type=click.Choice(["train", "validation", "holdout", "all"]), default="all")
@click.option("--no-fetch", is_flag=True)
def sim_main(
    rule: str,
    side: str,
    risk_pct: float | None,
    sl_atr: float | None,
    tp_R: float | None,
    pairs: tuple[str, ...],
    config_path: str | None,
    split: str,
    no_fetch: bool,
) -> None:
    """Simulate one FX rule with the risk-management engine."""
    try:
        from astro_market.fx.evaluate import evaluate_fx_strategy

        cfg = load_fx_config(config_path)
        fx = cfg.get("fx", cfg)
        risk = fx.get("risk", {})
        strat = FxStrategy(
            rule=rule,
            side=side,  # type: ignore[arg-type]
            risk_pct=float(risk_pct if risk_pct is not None else risk.get("risk_pct", 0.005)),
            sl_atr=float(sl_atr if sl_atr is not None else risk.get("sl_atr", 1.5)),
            tp_R=float(tp_R if tp_R is not None else risk.get("tp_R", 2.0)),
            atr_period=int(risk.get("atr_period", 14)),
            sl_pips_fallback=float(risk.get("sl_pips_fallback", 50.0)),
            time_stop=risk.get("time_stop", 10),
        )
        pair_list = list(pairs) if pairs else list(fx.get("pairs") or FX_PAIRS)
        ohlc = load_all_fx(pairs=pair_list, cfg=cfg, auto_fetch=not no_fetch)
        starts, ends = [], []
        all_idx = None
        for df in ohlc.values():
            all_idx = df.index if all_idx is None else all_idx.union(df.index)
            if len(df):
                s, e = data_start_end(df)
                starts.append(s)
                ends.append(e)
        data_start = min(starts) if starts else None
        data_end = max(ends) if ends else None
        atoms = load_atoms(dates=all_idx, auto_build=True)
        broker = _broker_from_cfg(cfg)

        if split == "all":
            m, sim = evaluate_fx_strategy(
                strat, ohlc, atoms, broker=broker, pairs=pair_list, cfg=cfg
            )
            rows = [metrics_row(rule, "all", m, data_start, data_end)]
        else:
            m = evaluate_on_fx_split(
                strat, ohlc, atoms, split, cfg=cfg, broker=broker, pairs=pair_list
            )
            rows = [metrics_row(rule, split, m, data_start, data_end)]
        print_table(pd.DataFrame(rows))
    except FXDataError as e:
        print(f"ERROR: Could not load FX data.\n{e}", file=sys.stderr)
        sys.exit(2)
    except Exception as e:
        print(f"ERROR: {type(e).__name__}: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
