"""Monte Carlo null: circular-shift returns, keep atoms fixed."""

from __future__ import annotations

import sys
from typing import Callable

import click
import numpy as np
import pandas as pd

from astro_market.config import load_config
from astro_market.data import DataFetchError, daily_returns, load_prices, split_mask
from astro_market.atoms import atom_registry, load_atoms
from astro_market.ephemeris import EphemerisError
from astro_market.evaluate import evaluate_rule, fitness
from astro_market.rules import evaluate_rule_signal


def circular_shift(series: pd.Series, shift: int) -> pd.Series:
    """Circular (wrap-around) shift of values, preserving the index."""
    values = np.asarray(series.to_numpy())
    shifted = np.roll(values, shift)
    return pd.Series(shifted, index=series.index, name=series.name)


def validation_slice(
    returns: pd.Series,
    atoms: pd.DataFrame,
    cfg: dict,
) -> tuple[pd.Series, pd.DataFrame]:
    start, end = cfg["splits"]["validation"]
    mask = split_mask(returns.index, start, end)
    r = returns.loc[mask]
    a = atoms.reindex(r.index).dropna(how="any")
    # bool cast after dropna
    a = a.astype(bool)
    r = r.loc[a.index]
    return r, a


def null_distribution(
    rule: str,
    returns: pd.Series,
    atoms: pd.DataFrame,
    *,
    n: int = 1000,
    seed: int = 42,
    bps_per_side: float = 10.0,
    lambda_complexity: float = 0.01,
    score_fn: Callable | None = None,
) -> np.ndarray:
    """
    Circular-shift daily returns (atoms fixed), recompute validation score.

    Default score = fitness (excess Sharpe - lambda * complexity).
    Returns array of shape (n,).
    """
    rng = np.random.default_rng(seed)
    n_days = len(returns)
    if n_days < 2:
        raise ValueError("Need at least 2 days for circular shift null")

    scores = np.empty(n, dtype=float)

    def default_score(m):
        return fitness(m, lambda_complexity=lambda_complexity)

    scorer = score_fn or default_score

    # Observed (shift=0) not included unless drawn; pure null shifts in 1..n_days-1
    for i in range(n):
        k = int(rng.integers(1, n_days))
        r_shift = circular_shift(returns, k)
        m = evaluate_rule(rule, r_shift, atoms, bps_per_side=bps_per_side)
        scores[i] = scorer(m)

    return scores


def observed_score(
    rule: str,
    returns: pd.Series,
    atoms: pd.DataFrame,
    *,
    bps_per_side: float = 10.0,
    lambda_complexity: float = 0.01,
) -> float:
    m = evaluate_rule(rule, returns, atoms, bps_per_side=bps_per_side)
    return fitness(m, lambda_complexity=lambda_complexity)


def best_length1_atom_score(
    returns: pd.Series,
    atoms: pd.DataFrame,
    *,
    bps_per_side: float = 10.0,
    lambda_complexity: float = 0.01,
    atom_names: list[str] | None = None,
) -> tuple[str, float]:
    """
    Scan all length-1 atoms (single atom rules) and return (best_atom, best_fitness).
    Hook for null tests of best length-1 scan score.
    """
    names = atom_names or [c for c in atoms.columns]
    best_name = ""
    best_fit = float("-inf")
    for name in names:
        if name not in atoms.columns:
            continue
        try:
            m = evaluate_rule(name, returns, atoms, bps_per_side=bps_per_side)
            f = fitness(m, lambda_complexity=lambda_complexity)
        except Exception:
            continue
        if f > best_fit:
            best_fit = f
            best_name = name
    return best_name, best_fit


def null_best_length1(
    returns: pd.Series,
    atoms: pd.DataFrame,
    *,
    n: int = 1000,
    seed: int = 42,
    bps_per_side: float = 10.0,
    lambda_complexity: float = 0.01,
) -> np.ndarray:
    """
    Under circular-shift null, distribution of the *best* length-1 atom fitness.
    """
    rng = np.random.default_rng(seed)
    n_days = len(returns)
    scores = np.empty(n, dtype=float)
    names = list(atoms.columns)
    for i in range(n):
        k = int(rng.integers(1, n_days))
        r_shift = circular_shift(returns, k)
        _, best_f = best_length1_atom_score(
            r_shift,
            atoms,
            bps_per_side=bps_per_side,
            lambda_complexity=lambda_complexity,
            atom_names=names,
        )
        scores[i] = best_f
    return scores


def empirical_pvalue(observed: float, null_scores: np.ndarray, alternative: str = "greater") -> float:
    """One-sided empirical p-value with +1 continuity correction."""
    null_scores = np.asarray(null_scores, dtype=float)
    n = len(null_scores)
    if alternative == "greater":
        count = np.sum(null_scores >= observed)
    else:
        count = np.sum(null_scores <= observed)
    return float((count + 1) / (n + 1))


@click.command()
@click.option("--rule", default="moon_phase_new", show_default=True, help="Rule expression")
@click.option("--n", "n_sims", default=1000, show_default=True, help="Number of null simulations")
@click.option("--seed", default=42, show_default=True, help="RNG seed")
@click.option("--config", "config_path", default=None, help="Path to YAML config")
@click.option(
    "--scan-length1",
    is_flag=True,
    default=False,
    help="Also run null for best length-1 atom scan score",
)
def main(
    rule: str,
    n_sims: int,
    seed: int,
    config_path: str | None,
    scan_length1: bool,
) -> None:
    """Monte Carlo null: circular-shift returns, atoms fixed, score on validation."""
    try:
        cfg = load_config(config_path)
        prices = load_prices(cfg=cfg, auto_fetch=True)
        rets = daily_returns(prices)
        atoms = load_atoms(cfg=cfg, dates=prices.index, auto_build=True)
        common = rets.index.intersection(atoms.index)
        rets = rets.loc[common]
        atoms = atoms.loc[common]

        r_val, a_val = validation_slice(rets, atoms, cfg)
        bps = float(cfg["costs"]["bps_per_side"])
        lam = float(cfg["evaluate"]["complexity_lambda"])

        obs = observed_score(rule, r_val, a_val, bps_per_side=bps, lambda_complexity=lam)
        null_scores = null_distribution(
            rule,
            r_val,
            a_val,
            n=n_sims,
            seed=seed,
            bps_per_side=bps,
            lambda_complexity=lam,
        )
        p = empirical_pvalue(obs, null_scores)

        print(f"Rule: {rule}")
        print(f"Validation window: {cfg['splits']['validation']}")
        print(f"N days: {len(r_val)} | N sims: {n_sims} | seed: {seed}")
        print(f"Observed fitness: {obs:.4f}")
        print(
            f"Null fitness: mean={null_scores.mean():.4f} "
            f"std={null_scores.std():.4f} "
            f"p5={np.percentile(null_scores, 5):.4f} "
            f"p95={np.percentile(null_scores, 95):.4f}"
        )
        print(f"Empirical p-value (greater): {p:.4f}")
        print(
            "Interpretation: small p suggests the rule's validation fitness is "
            "unusual under a return-circular-shift null (atoms fixed)."
        )

        if scan_length1:
            print()
            print("--- Best length-1 atom scan ---")
            best_name, best_obs = best_length1_atom_score(
                r_val, a_val, bps_per_side=bps, lambda_complexity=lam
            )
            # Fewer sims by default for expensive scan — use same n but warn
            print(f"Observed best atom: {best_name}  fitness={best_obs:.4f}")
            print(f"Running null for best length-1 (n={n_sims}) — may take a while...")
            null_best = null_best_length1(
                r_val,
                a_val,
                n=n_sims,
                seed=seed,
                bps_per_side=bps,
                lambda_complexity=lam,
            )
            p_best = empirical_pvalue(best_obs, null_best)
            print(
                f"Null best-L1 fitness: mean={null_best.mean():.4f} "
                f"p95={np.percentile(null_best, 95):.4f}"
            )
            print(f"Empirical p-value (greater): {p_best:.4f}")

    except DataFetchError as e:
        print(f"ERROR: Could not load market data.\n{e}", file=sys.stderr)
        sys.exit(2)
    except EphemerisError as e:
        print(f"ERROR: Ephemeris unavailable.\n{e}", file=sys.stderr)
        sys.exit(2)
    except Exception as e:
        print(f"ERROR: {type(e).__name__}: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
