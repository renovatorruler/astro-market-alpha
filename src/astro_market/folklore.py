"""CLI: evaluate pre-registered folklore rules on train/val/holdout."""

from __future__ import annotations

import sys
from typing import Any

import click
import pandas as pd

from astro_market.config import load_config
from astro_market.data import DataFetchError, daily_returns, load_prices
from astro_market.atoms import load_atoms
from astro_market.ephemeris import EphemerisError
from astro_market.evaluate import EvalMetrics, evaluate_on_split, fitness


FOLKLORE_DEFAULTS = [
    "moon_phase_new",
    "moon_phase_full",
    "mercury_retro",
    "~mercury_retro",
    "moon_phase_new | moon_phase_full",
]


def _fmt(x: float, pct: bool = False) -> str:
    if x is None or (isinstance(x, float) and (pd.isna(x) or not abs(x) < 1e100)):
        return "nan"
    if pct:
        return f"{100 * x:8.2f}%"
    return f"{x:8.3f}"


def metrics_row(rule: str, split: str, m: EvalMetrics, lam: float) -> dict[str, Any]:
    return {
        "rule": rule,
        "split": split,
        "total_ret": m.total_return,
        "cagr": m.cagr,
        "sharpe": m.sharpe,
        "sharpe_bh": m.sharpe_bh,
        "excess_sharpe": m.excess_sharpe,
        "max_dd": m.max_dd,
        "turnover": m.turnover,
        "n_trades": m.n_trades,
        "complexity": m.complexity,
        "fitness": fitness(m, lambda_complexity=lam),
        "n_days": m.n_days,
    }


def run_folklore(
    rules: list[str] | None = None,
    cfg: dict | None = None,
) -> pd.DataFrame:
    """Evaluate folklore rules on train / validation / holdout. Never tunes on holdout."""
    cfg = cfg or load_config()
    rules = rules or list(cfg.get("folklore_rules") or FOLKLORE_DEFAULTS)
    lam = float(cfg["evaluate"]["complexity_lambda"])

    prices = load_prices(cfg=cfg, auto_fetch=True)
    rets = daily_returns(prices)
    atoms = load_atoms(cfg=cfg, dates=prices.index, auto_build=True)

    # Align
    common = rets.index.intersection(atoms.index)
    rets = rets.loc[common]
    atoms = atoms.loc[common]

    rows: list[dict[str, Any]] = []
    for rule in rules:
        for split in ("train", "validation", "holdout"):
            m = evaluate_on_split(rule, rets, atoms, split, cfg=cfg)
            rows.append(metrics_row(rule, split, m, lam))

    return pd.DataFrame(rows)


def print_table(df: pd.DataFrame) -> None:
    """Pretty-print metrics table to stdout."""
    cols = [
        "rule",
        "split",
        "cagr",
        "sharpe",
        "sharpe_bh",
        "excess_sharpe",
        "max_dd",
        "turnover",
        "n_trades",
        "fitness",
    ]
    show = df[cols].copy()
    # Format numerics for display
    lines = []
    header = (
        f"{'rule':<40} {'split':<12} {'CAGR':>10} {'Sharpe':>8} {'SH_BH':>8} "
        f"{'XS':>8} {'MaxDD':>10} {'Turn':>8} {'Trades':>7} {'Fit':>8}"
    )
    lines.append(header)
    lines.append("-" * len(header))
    for _, row in show.iterrows():
        lines.append(
            f"{str(row['rule']):<40} {str(row['split']):<12} "
            f"{_fmt(row['cagr'], pct=True):>10} {_fmt(row['sharpe']):>8} "
            f"{_fmt(row['sharpe_bh']):>8} {_fmt(row['excess_sharpe']):>8} "
            f"{_fmt(row['max_dd'], pct=True):>10} {_fmt(row['turnover']):>8} "
            f"{int(row['n_trades']):>7d} {_fmt(row['fitness']):>8}"
        )
    print("\n".join(lines))
    print()
    print(
        "Protocol: train 1970–1994 | validation 1995–2009 | holdout 2010–present. "
        "NEVER tune on holdout. Costs=10 bps/side. Position=long/flat."
    )
    print(
        "Pass/fail (illustrative): a rule 'beats BH' on validation if excess_sharpe > 0 "
        "AND fitness > 0; holdout is for final confirmation only."
    )


@click.command()
@click.option("--config", "config_path", default=None, help="Path to YAML config")
@click.option(
    "--rule",
    "extra_rules",
    multiple=True,
    help="Additional rule expression (repeatable)",
)
def main(config_path: str | None, extra_rules: tuple[str, ...]) -> None:
    """Evaluate pre-registered folklore astrology rules vs S&P 500 BH."""
    try:
        cfg = load_config(config_path)
        rules = list(cfg.get("folklore_rules") or FOLKLORE_DEFAULTS)
        if extra_rules:
            rules = list(extra_rules)
        df = run_folklore(rules=rules, cfg=cfg)
        print_table(df)
    except DataFetchError as e:
        print(f"ERROR: Could not load market data.\n{e}", file=sys.stderr)
        print(
            "Hint: ensure network access for yfinance, or place a parquet cache at data/gspc_adj_close.parquet",
            file=sys.stderr,
        )
        sys.exit(2)
    except EphemerisError as e:
        print(f"ERROR: Ephemeris unavailable.\n{e}", file=sys.stderr)
        print(
            "Hint: run `python -m astro_market ensure-ephemeris` with network, "
            "or place de421.bsp under data/.",
            file=sys.stderr,
        )
        sys.exit(2)
    except Exception as e:
        print(f"ERROR: {type(e).__name__}: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
