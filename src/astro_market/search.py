"""Boolean lattice search: length-1 / length-2 rules over the atom set.

Protocol (frozen — do not change splits):
  Train 1970–1994, Validation 1995–2009, Holdout 2010–present.
  Selection / ranking uses VALIDATION fitness only; holdout is reported once.
  Fitness = excess_sharpe - λ * complexity (λ from config).

Defaults (documented):
  --max-length 2
  --top-k 50  : carry the top 50 length-1 rules by validation fitness into
                length-2 AND pairing (and moon-phase style OR pairs).
  --include-negation : also evaluate ~atom for every atom (complexity=2).
  Monte Carlo: top survivors by val fitness — min(10, all with fitness>0),
               always including any with val excess_sharpe>0 (capped at 20);
               n=1000 circular-shift nulls on validation (seed from config).
"""

from __future__ import annotations

import sys
import time
from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Sequence

import click
import numpy as np
import pandas as pd

from astro_market.config import PROJECT_ROOT, load_config
from astro_market.data import DataFetchError, daily_returns, load_prices, split_mask
from astro_market.atoms import load_atoms
from astro_market.ephemeris import EphemerisError
from astro_market.null import empirical_pvalue
from astro_market.rules import rule_complexity


# ---------------------------------------------------------------------------
# Fast numpy metrics (matches evaluate.apply_costs / _sharpe semantics)
# ---------------------------------------------------------------------------


def _sharpe_np(r: np.ndarray, periods_per_year: float = 252.0) -> float:
    if len(r) < 2:
        return float("nan")
    mu = float(np.mean(r))
    sigma = float(np.std(r, ddof=1))
    # Always-flat (all zeros) => Sharpe 0 by convention (undefined otherwise).
    if sigma == 0.0 or not np.isfinite(sigma):
        return 0.0 if mu == 0.0 else float("nan")
    return float(np.sqrt(periods_per_year) * mu / sigma)


def strategy_and_bh_returns(
    signal: np.ndarray,
    returns: np.ndarray,
    bps_per_side: float = 10.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Net strategy returns and BH returns given a boolean (or 0/1) signal.

    Mirrors evaluate.apply_costs: position lag, fill leading 0, cost on |delta|.
    """
    n = len(returns)
    if n == 0:
        empty = np.asarray([], dtype=float)
        return empty, empty

    pos = np.asarray(signal, dtype=float)
    if pos.shape[0] != n:
        raise ValueError("signal and returns length mismatch")

    pos_lag = np.empty(n, dtype=float)
    pos_lag[0] = 0.0
    if n > 1:
        pos_lag[1:] = pos[:-1]

    cost_unit = bps_per_side / 10_000.0
    delta = np.empty(n, dtype=float)
    delta[0] = abs(pos_lag[0])
    if n > 1:
        delta[1:] = np.abs(np.diff(pos_lag))
    strat = pos_lag * returns - cost_unit * delta

    # BH always long
    bh_pos = np.ones(n, dtype=float)
    bh_lag = np.empty(n, dtype=float)
    bh_lag[0] = 0.0
    if n > 1:
        bh_lag[1:] = bh_pos[:-1]
    bh_delta = np.empty(n, dtype=float)
    bh_delta[0] = abs(bh_lag[0])
    if n > 1:
        bh_delta[1:] = np.abs(np.diff(bh_lag))
    bh = bh_lag * returns - cost_unit * bh_delta
    return strat, bh


def metrics_from_signal(
    signal: np.ndarray,
    returns: np.ndarray,
    *,
    complexity: int,
    bps_per_side: float = 10.0,
    lambda_complexity: float = 0.01,
) -> dict[str, float]:
    """Compute key metrics for a boolean signal vs BH."""
    strat, bh = strategy_and_bh_returns(signal, returns, bps_per_side=bps_per_side)
    if len(strat) == 0:
        return {
            "sharpe": float("nan"),
            "sharpe_bh": float("nan"),
            "excess_sharpe": float("nan"),
            "fitness": float("-inf"),
            "total_return": float("nan"),
            "cagr": float("nan"),
            "max_dd": float("nan"),
            "n_days": 0,
            "complexity": complexity,
        }

    sharpe_s = _sharpe_np(strat)
    sharpe_bh = _sharpe_np(bh)
    excess = (
        (sharpe_s - sharpe_bh)
        if np.isfinite(sharpe_s) and np.isfinite(sharpe_bh)
        else float("nan")
    )
    fit = (
        float(excess - lambda_complexity * complexity)
        if np.isfinite(excess)
        else float("-inf")
    )
    wealth = np.cumprod(1.0 + strat)
    peak = np.maximum.accumulate(wealth)
    dd = wealth / peak - 1.0
    total = float(wealth[-1] - 1.0) if len(wealth) else float("nan")
    years = len(strat) / 252.0
    if years > 0 and wealth[-1] > 0:
        cagr = float(wealth[-1] ** (1.0 / years) - 1.0)
    else:
        cagr = float("nan")

    return {
        "sharpe": sharpe_s,
        "sharpe_bh": sharpe_bh,
        "excess_sharpe": excess,
        "fitness": fit,
        "total_return": total,
        "cagr": cagr,
        "max_dd": float(dd.min()) if len(dd) else float("nan"),
        "n_days": int(len(strat)),
        "complexity": complexity,
    }


# ---------------------------------------------------------------------------
# Rule helpers
# ---------------------------------------------------------------------------


def canonical_rule(expr: str) -> str:
    """
    Canonicalize simple length-1 / length-2 rules for deduplication.

    Handles: atom, ~atom, a & b, a | b (commutative reorder by atom name).
    Falls back to stripped source for anything richer.
    """
    s = expr.strip()
    if not s:
        return s

    # ~atom
    if s.startswith("~") and "&" not in s and "|" not in s and "(" not in s:
        return f"~{s[1:].strip()}"

    # binary & or |
    for op, sym in (("&", " & "), ("|", " | ")):
        if op in s and "(" not in s:
            parts = [p.strip() for p in s.split(op)]
            if len(parts) == 2 and all(parts):
                # Keep negation attached to atom; sort by underlying atom name
                def sort_key(p: str) -> str:
                    return p[1:] if p.startswith("~") else p

                a, b = sorted(parts, key=sort_key)
                return f"{a}{sym}{b}"
    return s


def complexity_for_rule(expr: str) -> int:
    """Complexity via existing DSL counter (atoms + operators)."""
    return rule_complexity(expr)


def mutually_exclusive(a: np.ndarray, b: np.ndarray) -> bool:
    """True if a AND b is never True (easy skip for length-2 AND)."""
    return not bool(np.any(a & b))


def split_returns_atoms(
    returns: pd.Series,
    atoms: pd.DataFrame,
    cfg: dict,
    split_name: str,
) -> tuple[np.ndarray, pd.DataFrame, pd.DatetimeIndex]:
    """Slice returns/atoms to a protocol split; return numpy returns + bool atoms."""
    start, end = cfg["splits"][split_name]
    mask = split_mask(returns.index, start, end)
    r = returns.loc[mask]
    a = atoms.reindex(r.index)
    valid = a.notna().all(axis=1)
    a = a.loc[valid].astype(bool)
    r = r.loc[a.index]
    return r.to_numpy(dtype=float), a, a.index


# ---------------------------------------------------------------------------
# Candidate records
# ---------------------------------------------------------------------------


@dataclass
class RuleResult:
    rule: str
    length: int
    complexity: int
    train_fitness: float
    train_excess_sharpe: float
    train_sharpe: float
    val_fitness: float
    val_excess_sharpe: float
    val_sharpe: float
    val_sharpe_bh: float
    holdout_fitness: float | None = None
    holdout_excess_sharpe: float | None = None
    holdout_sharpe: float | None = None
    holdout_sharpe_bh: float | None = None
    null_pvalue: float | None = None
    null_n: int | None = None
    selected_on: str = "validation"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------


def evaluate_rule_on_splits(
    rule: str,
    signal_train: np.ndarray,
    signal_val: np.ndarray,
    r_train: np.ndarray,
    r_val: np.ndarray,
    *,
    bps: float,
    lam: float,
) -> RuleResult:
    """Score a rule on train + validation from precomputed signals."""
    c = complexity_for_rule(rule)
    m_tr = metrics_from_signal(
        signal_train, r_train, complexity=c, bps_per_side=bps, lambda_complexity=lam
    )
    m_va = metrics_from_signal(
        signal_val, r_val, complexity=c, bps_per_side=bps, lambda_complexity=lam
    )
    # Lattice length: binary &/| => 2, else 1 (negation still length-1)
    core = rule.replace("~", "").replace(" ", "")
    length = 2 if ("&" in core or "|" in core) else 1
    return RuleResult(
        rule=rule,
        length=length,
        complexity=c,
        train_fitness=m_tr["fitness"],
        train_excess_sharpe=m_tr["excess_sharpe"],
        train_sharpe=m_tr["sharpe"],
        val_fitness=m_va["fitness"],
        val_excess_sharpe=m_va["excess_sharpe"],
        val_sharpe=m_va["sharpe"],
        val_sharpe_bh=m_va["sharpe_bh"],
    )


def signal_for_expr(expr: str, atoms: pd.DataFrame) -> np.ndarray:
    """
    Fast signal for lattice exprs: atom | ~atom | a&b | a|b
    (no nested parens beyond what we generate).
    """
    s = expr.strip()
    if "&" in s and "|" not in s and "(" not in s:
        left, right = [p.strip() for p in s.split("&", 1)]
        return signal_for_expr(left, atoms) & signal_for_expr(right, atoms)
    if "|" in s and "&" not in s and "(" not in s:
        left, right = [p.strip() for p in s.split("|", 1)]
        return signal_for_expr(left, atoms) | signal_for_expr(right, atoms)
    if s.startswith("~"):
        name = s[1:].strip()
        return ~atoms[name].to_numpy(dtype=bool)
    return atoms[s].to_numpy(dtype=bool)


def length1_search(
    atoms_train: pd.DataFrame,
    atoms_val: pd.DataFrame,
    r_train: np.ndarray,
    r_val: np.ndarray,
    *,
    include_negation: bool = True,
    bps: float = 10.0,
    lam: float = 0.01,
) -> list[RuleResult]:
    """Evaluate every atom (and optional ~atom) on train+validation."""
    results: list[RuleResult] = []
    seen: set[str] = set()
    for col in atoms_val.columns:
        candidates = [col]
        if include_negation:
            candidates.append(f"~{col}")
        for rule in candidates:
            key = canonical_rule(rule)
            if key in seen:
                continue
            seen.add(key)
            sig_tr = signal_for_expr(rule, atoms_train)
            sig_va = signal_for_expr(rule, atoms_val)
            results.append(
                evaluate_rule_on_splits(
                    rule, sig_tr, sig_va, r_train, r_val, bps=bps, lam=lam
                )
            )
    results.sort(key=lambda x: x.val_fitness, reverse=True)
    return results


def length2_and_search(
    length1: Sequence[RuleResult],
    atoms_train: pd.DataFrame,
    atoms_val: pd.DataFrame,
    r_train: np.ndarray,
    r_val: np.ndarray,
    *,
    top_k: int = 50,
    bps: float = 10.0,
    lam: float = 0.01,
    skip_mutex: bool = True,
) -> list[RuleResult]:
    """AND combinations of the top-k length-1 rules by validation fitness."""
    seeds = list(length1[: max(1, top_k)])
    results: list[RuleResult] = []
    seen: set[str] = set()

    # Precompute signals for seeds on train/val
    sig_tr: dict[str, np.ndarray] = {}
    sig_va: dict[str, np.ndarray] = {}
    for rr in seeds:
        sig_tr[rr.rule] = signal_for_expr(rr.rule, atoms_train)
        sig_va[rr.rule] = signal_for_expr(rr.rule, atoms_val)

    for ra, rb in combinations(seeds, 2):
        # Skip pairing a rule with its own negation when both present
        a, b = ra.rule, rb.rule
        if a.lstrip("~") == b.lstrip("~") and (a.startswith("~") != b.startswith("~")):
            continue

        rule = canonical_rule(f"{a} & {b}")
        if rule in seen:
            continue
        seen.add(rule)

        sva = sig_va[a] & sig_va[b]
        if skip_mutex and mutually_exclusive(sig_va[a], sig_va[b]):
            continue
        # Also skip if AND is identical to one side (redundant)
        if np.array_equal(sva, sig_va[a]) or np.array_equal(sva, sig_va[b]):
            continue
        # Skip empty or always-true (degenerate)
        if not np.any(sva) or np.all(sva):
            continue

        str_ = sig_tr[a] & sig_tr[b]
        results.append(
            evaluate_rule_on_splits(
                rule, str_, sva, r_train, r_val, bps=bps, lam=lam
            )
        )

    results.sort(key=lambda x: x.val_fitness, reverse=True)
    return results


def length2_or_moon_pairs(
    atoms_train: pd.DataFrame,
    atoms_val: pd.DataFrame,
    r_train: np.ndarray,
    r_val: np.ndarray,
    *,
    bps: float = 10.0,
    lam: float = 0.01,
) -> list[RuleResult]:
    """Optional OR pairs among moon_phase_* atoms (folklore-style)."""
    phases = [c for c in atoms_val.columns if c.startswith("moon_phase_")]
    results: list[RuleResult] = []
    seen: set[str] = set()
    for a, b in combinations(phases, 2):
        rule = canonical_rule(f"{a} | {b}")
        if rule in seen:
            continue
        seen.add(rule)
        sig_tr = signal_for_expr(a, atoms_train) | signal_for_expr(b, atoms_train)
        sig_va = signal_for_expr(a, atoms_val) | signal_for_expr(b, atoms_val)
        if np.all(sig_va) or not np.any(sig_va):
            continue
        results.append(
            evaluate_rule_on_splits(
                rule, sig_tr, sig_va, r_train, r_val, bps=bps, lam=lam
            )
        )
    results.sort(key=lambda x: x.val_fitness, reverse=True)
    return results


def attach_holdout(
    rules: Sequence[RuleResult],
    atoms_hold: pd.DataFrame,
    r_hold: np.ndarray,
    *,
    bps: float,
    lam: float,
) -> list[RuleResult]:
    """Evaluate frozen rules once on holdout (reporting only — not for selection)."""
    out: list[RuleResult] = []
    for rr in rules:
        sig = signal_for_expr(rr.rule, atoms_hold)
        m = metrics_from_signal(
            sig, r_hold, complexity=rr.complexity, bps_per_side=bps, lambda_complexity=lam
        )
        rr.holdout_fitness = m["fitness"]
        rr.holdout_excess_sharpe = m["excess_sharpe"]
        rr.holdout_sharpe = m["sharpe"]
        rr.holdout_sharpe_bh = m["sharpe_bh"]
        out.append(rr)
    return out


def select_mc_survivors(
    ranked: Sequence[RuleResult],
    *,
    n_cap: int = 10,
    hard_cap: int = 20,
) -> list[RuleResult]:
    """
    Top N by val fitness: N = min(n_cap, count with fitness>0), but always
    include any with val excess_sharpe > 0 (up to hard_cap).
    """
    positive_fit = [r for r in ranked if r.val_fitness > 0]
    positive_xs = [r for r in ranked if r.val_excess_sharpe > 0]

    # Start with top min(n_cap, |fitness>0|) by val fitness among all ranked
    if positive_fit:
        n = min(n_cap, len(positive_fit))
        chosen = list(ranked[:n])
    else:
        # No positive fitness — still report top few for honesty
        chosen = list(ranked[: min(5, len(ranked))])

    # Ensure all excess_sharpe>0 are included (up to hard_cap)
    have = {canonical_rule(r.rule) for r in chosen}
    for r in positive_xs:
        key = canonical_rule(r.rule)
        if key not in have:
            chosen.append(r)
            have.add(key)
        if len(chosen) >= hard_cap:
            break

    # Dedup preserve order
    out: list[RuleResult] = []
    seen: set[str] = set()
    for r in chosen:
        k = canonical_rule(r.rule)
        if k not in seen:
            seen.add(k)
            out.append(r)
    return out[:hard_cap]


def monte_carlo_null_for_rule(
    rule: str,
    atoms_val: pd.DataFrame,
    r_val: np.ndarray,
    *,
    n: int = 1000,
    seed: int = 42,
    bps: float = 10.0,
    lam: float = 0.01,
) -> float:
    """
    Circular-shift null on validation; return empirical p-value for fitness.
    Signal fixed; only returns roll. Uses vectorized scoring.
    """
    signal = signal_for_expr(rule, atoms_val)
    c = complexity_for_rule(rule)
    observed = metrics_from_signal(
        signal, r_val, complexity=c, bps_per_side=bps, lambda_complexity=lam
    )["fitness"]

    rng = np.random.default_rng(seed)
    n_days = len(r_val)
    if n_days < 2:
        return float("nan")

    scores = np.empty(n, dtype=float)
    for i in range(n):
        k = int(rng.integers(1, n_days))
        r_shift = np.roll(r_val, k)
        scores[i] = metrics_from_signal(
            signal, r_shift, complexity=c, bps_per_side=bps, lambda_complexity=lam
        )["fitness"]

    return empirical_pvalue(observed, scores, alternative="greater")


# ---------------------------------------------------------------------------
# Orchestration + reporting
# ---------------------------------------------------------------------------


DEFAULT_TOP_K = 50
DEFAULT_MC_N = 1000


def run_lattice_search(
    *,
    max_length: int = 2,
    top_k: int = DEFAULT_TOP_K,
    include_negation: bool = True,
    include_or_moon: bool = True,
    mc_n: int = DEFAULT_MC_N,
    cfg: dict | None = None,
    verbose: bool = True,
) -> tuple[pd.DataFrame, list[RuleResult], dict[str, Any]]:
    """
    Full lattice search end-to-end.

    Returns (ranked_dataframe, mc_survivors, meta).
    """
    t0 = time.perf_counter()
    cfg = cfg or load_config()
    bps = float(cfg["costs"]["bps_per_side"])
    lam = float(cfg["evaluate"]["complexity_lambda"])
    seed = int(cfg.get("null", {}).get("seed", 42))

    if verbose:
        print("Loading prices + atoms from cache...")
    prices = load_prices(cfg=cfg, auto_fetch=False)
    rets = daily_returns(prices)
    atoms = load_atoms(cfg=cfg, dates=prices.index, auto_build=False)
    common = rets.index.intersection(atoms.index)
    rets = rets.loc[common]
    atoms = atoms.loc[common]

    r_train, a_train, _ = split_returns_atoms(rets, atoms, cfg, "train")
    r_val, a_val, _ = split_returns_atoms(rets, atoms, cfg, "validation")
    r_hold, a_hold, _ = split_returns_atoms(rets, atoms, cfg, "holdout")

    if verbose:
        print(
            f"Splits: train={len(r_train)}d  val={len(r_val)}d  holdout={len(r_hold)}d  "
            f"atoms={a_val.shape[1]}  λ={lam}  costs={bps}bps"
        )
        print(f"Length-1 search (negation={include_negation})...")

    t1 = time.perf_counter()
    l1 = length1_search(
        a_train,
        a_val,
        r_train,
        r_val,
        include_negation=include_negation,
        bps=bps,
        lam=lam,
    )
    if verbose:
        print(f"  Evaluated {len(l1)} length-1 rules in {time.perf_counter() - t1:.1f}s")
        print(f"  Top-5 by val fitness:")
        for rr in l1[:5]:
            print(
                f"    {rr.rule:<40} val_fit={rr.val_fitness:+.4f} "
                f"xs={rr.val_excess_sharpe:+.4f}"
            )

    all_rules: list[RuleResult] = list(l1)

    if max_length >= 2:
        if verbose:
            print(f"Length-2 AND search (top_k={top_k})...")
        t2 = time.perf_counter()
        l2 = length2_and_search(
            l1,
            a_train,
            a_val,
            r_train,
            r_val,
            top_k=top_k,
            bps=bps,
            lam=lam,
            skip_mutex=True,
        )
        if verbose:
            print(f"  Evaluated {len(l2)} AND rules in {time.perf_counter() - t2:.1f}s")
        all_rules.extend(l2)

        if include_or_moon:
            if verbose:
                print("Length-2 moon-phase OR pairs...")
            l2or = length2_or_moon_pairs(
                a_train, a_val, r_train, r_val, bps=bps, lam=lam
            )
            if verbose:
                print(f"  Evaluated {len(l2or)} OR rules")
            all_rules.extend(l2or)

    # Deduplicate by canonical rule, keep best val fitness
    best: dict[str, RuleResult] = {}
    for rr in all_rules:
        key = canonical_rule(rr.rule)
        rr.rule = key  # store canonical form
        prev = best.get(key)
        if prev is None or rr.val_fitness > prev.val_fitness:
            best[key] = rr
    ranked = sorted(best.values(), key=lambda x: x.val_fitness, reverse=True)

    if verbose:
        print(f"Unique rules after dedup: {len(ranked)}")
        print("Attaching holdout metrics (reporting only; not used for selection)...")

    # Attach holdout to all ranked (needed for CSV); MC subset is smaller
    ranked = attach_holdout(ranked, a_hold, r_hold, bps=bps, lam=lam)

    survivors = select_mc_survivors(ranked)
    if verbose:
        print(
            f"Monte Carlo null on {len(survivors)} survivors "
            f"(n={mc_n}, circular-shift validation)..."
        )
    t3 = time.perf_counter()
    for i, rr in enumerate(survivors):
        p = monte_carlo_null_for_rule(
            rr.rule, a_val, r_val, n=mc_n, seed=seed, bps=bps, lam=lam
        )
        rr.null_pvalue = p
        rr.null_n = mc_n
        # Mirror p into ranked list
        for r2 in ranked:
            if canonical_rule(r2.rule) == canonical_rule(rr.rule):
                r2.null_pvalue = p
                r2.null_n = mc_n
                break
        if verbose:
            print(
                f"  [{i + 1}/{len(survivors)}] {rr.rule:<40} "
                f"val_fit={rr.val_fitness:+.4f} p={p:.4f}"
            )
    if verbose:
        print(f"  MC done in {time.perf_counter() - t3:.1f}s")

    df = pd.DataFrame([r.to_dict() for r in ranked])
    meta = {
        "runtime_sec": time.perf_counter() - t0,
        "n_length1": len(l1),
        "n_ranked": len(ranked),
        "n_mc": len(survivors),
        "mc_n_sims": mc_n,
        "top_k": top_k,
        "max_length": max_length,
        "include_negation": include_negation,
        "lambda": lam,
        "bps": bps,
        "seed": seed,
        "n_train": len(r_train),
        "n_val": len(r_val),
        "n_holdout": len(r_hold),
        "n_atoms": int(a_val.shape[1]),
    }
    return df, survivors, meta


def protocol_pass(rr: RuleResult, p_threshold: float = 0.05) -> dict[str, bool]:
    """
    Illustrative protocol bar (from README):
      1) val excess_sharpe > 0 and val fitness > 0
      2) null p < threshold
      3) holdout reported (not a selection criterion); note if holdout xs > 0
    """
    val_ok = (
        rr.val_excess_sharpe is not None
        and rr.val_fitness is not None
        and rr.val_excess_sharpe > 0
        and rr.val_fitness > 0
    )
    null_ok = rr.null_pvalue is not None and rr.null_pvalue < p_threshold
    holdout_positive = (
        rr.holdout_excess_sharpe is not None and rr.holdout_excess_sharpe > 0
    )
    return {
        "val_interesting": bool(val_ok),
        "null_reject": bool(null_ok),
        "holdout_xs_positive": bool(holdout_positive),
        "passes_protocol_bar": bool(val_ok and null_ok),
    }


def write_results(
    df: pd.DataFrame,
    survivors: Sequence[RuleResult],
    meta: dict[str, Any],
    *,
    out_dir: Path | None = None,
) -> tuple[Path, Path]:
    """Write lattice_v1.csv and lattice_v1.md under results/."""
    out_dir = out_dir or (PROJECT_ROOT / "results")
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "lattice_v1.csv"
    md_path = out_dir / "lattice_v1.md"

    df.to_csv(csv_path, index=False)

    lines: list[str] = []
    lines.append("# Lattice search v1 — Boolean length-1 / length-2")
    lines.append("")
    lines.append("## Method")
    lines.append("")
    lines.append(
        "Exhaustive length-1 evaluation of every atom"
        + (" and its negation (`~atom`)" if meta.get("include_negation") else "")
        + ", ranked by **validation** fitness. "
        f"Length-2 = AND of the top `{meta.get('top_k')}` length-1 rules by validation "
        "fitness, plus folklore-style OR pairs among `moon_phase_*` atoms. "
        "Isomorphic rules (`a & b` ≡ `b & a`) are deduplicated; mutually exclusive "
        "AND pairs (never co-True on validation) are skipped."
    )
    lines.append("")
    lines.append("**Frozen protocol (unchanged):**")
    lines.append("")
    lines.append("| Item | Value |")
    lines.append("|------|-------|")
    lines.append("| Train | 1970-01-01 → 1994-12-31 |")
    lines.append("| Validation | 1995-01-01 → 2009-12-31 |")
    lines.append("| Holdout | 2010-01-01 → present (**never used for selection**) |")
    lines.append("| Position | Long when rule True; flat when False; BH = always long |")
    lines.append(f"| Costs | {meta.get('bps')} bps / side |")
    lines.append(
        f"| Fitness | `excess_sharpe − λ · complexity` with λ={meta.get('lambda')} |"
    )
    lines.append(
        f"| Monte Carlo | Circular-shift returns (atoms fixed), n={meta.get('mc_n_sims')}, "
        f"seed={meta.get('seed')}, on validation |"
    )
    lines.append("")
    lines.append(
        f"Runtime: **{meta.get('runtime_sec', float('nan')):.1f}s** · "
        f"atoms={meta.get('n_atoms')} · "
        f"L1 evaluated={meta.get('n_length1')} · "
        f"unique ranked={meta.get('n_ranked')} · "
        f"MC survivors={meta.get('n_mc')} · "
        f"days train/val/hold={meta.get('n_train')}/{meta.get('n_val')}/{meta.get('n_holdout')}"
    )
    lines.append("")
    lines.append("## Pass / fail bar (illustrative)")
    lines.append("")
    lines.append("A rule **passes the protocol bar** if:")
    lines.append("")
    lines.append("1. Validation `excess_sharpe > 0` **and** `fitness > 0`")
    lines.append("2. Circular-shift null empirical p-value `< 0.05` on validation fitness")
    lines.append(
        "3. Holdout metrics are reported for honesty but **must not** drive selection"
    )
    lines.append("")

    # Top table
    lines.append("## Top rules by validation fitness")
    lines.append("")
    lines.append(
        "| Rank | Rule | L | Cx | Train XS | Val XS | Val Fit | Hold XS | Hold Fit | Null p | Pass? |"
    )
    lines.append(
        "|-----:|------|--:|---:|---------:|-------:|--------:|--------:|---------:|-------:|:------:|"
    )

    top_n = min(25, len(df))
    any_pass = False
    for i, row in df.head(top_n).iterrows():
        # rebuild a RuleResult-like check
        rr = RuleResult(
            rule=str(row["rule"]),
            length=int(row["length"]),
            complexity=int(row["complexity"]),
            train_fitness=float(row["train_fitness"]),
            train_excess_sharpe=float(row["train_excess_sharpe"]),
            train_sharpe=float(row["train_sharpe"]),
            val_fitness=float(row["val_fitness"]),
            val_excess_sharpe=float(row["val_excess_sharpe"]),
            val_sharpe=float(row["val_sharpe"]),
            val_sharpe_bh=float(row["val_sharpe_bh"]),
            holdout_fitness=_maybe_float(row.get("holdout_fitness")),
            holdout_excess_sharpe=_maybe_float(row.get("holdout_excess_sharpe")),
            holdout_sharpe=_maybe_float(row.get("holdout_sharpe")),
            holdout_sharpe_bh=_maybe_float(row.get("holdout_sharpe_bh")),
            null_pvalue=_maybe_float(row.get("null_pvalue")),
            null_n=int(row["null_n"]) if pd.notna(row.get("null_n")) else None,
        )
        flags = protocol_pass(rr)
        if flags["passes_protocol_bar"]:
            any_pass = True
        pass_str = "YES" if flags["passes_protocol_bar"] else "no"
        if rr.null_pvalue is None:
            pass_str = "—"  # not MC'd
            p_str = "—"
        else:
            p_str = f"{rr.null_pvalue:.4f}"
        hold_xs = (
            f"{rr.holdout_excess_sharpe:+.3f}"
            if rr.holdout_excess_sharpe is not None and np.isfinite(rr.holdout_excess_sharpe)
            else "n/a"
        )
        hold_fit = (
            f"{rr.holdout_fitness:+.3f}"
            if rr.holdout_fitness is not None and np.isfinite(rr.holdout_fitness)
            else "n/a"
        )
        rank = list(df.index).index(i) + 1
        lines.append(
            f"| {rank} | `{rr.rule}` | {rr.length} | {rr.complexity} | "
            f"{rr.train_excess_sharpe:+.3f} | {rr.val_excess_sharpe:+.3f} | "
            f"{rr.val_fitness:+.3f} | {hold_xs} | {hold_fit} | {p_str} | {pass_str} |"
        )

    lines.append("")
    lines.append("## Monte Carlo survivors (detail)")
    lines.append("")
    if not survivors:
        lines.append("_No survivors selected for MC._")
    else:
        lines.append(
            "| Rule | Val Fit | Val XS | Hold XS | Null p | Val interesting | Null reject | Protocol pass |"
        )
        lines.append(
            "|------|--------:|-------:|--------:|-------:|:---------------:|:-----------:|:-------------:|"
        )
        for rr in survivors:
            flags = protocol_pass(rr)
            if flags["passes_protocol_bar"]:
                any_pass = True
            hxs = rr.holdout_excess_sharpe
            hxs_s = f"{hxs:+.4f}" if hxs is not None and np.isfinite(hxs) else "n/a"
            p_s = f"{rr.null_pvalue:.4f}" if rr.null_pvalue is not None else "n/a"
            lines.append(
                f"| `{rr.rule}` | {rr.val_fitness:+.4f} | {rr.val_excess_sharpe:+.4f} | "
                f"{hxs_s} | {p_s} | "
                f"{'Y' if flags['val_interesting'] else 'N'} | "
                f"{'Y' if flags['null_reject'] else 'N'} | "
                f"{'YES' if flags['passes_protocol_bar'] else 'no'} |"
            )

    lines.append("")
    lines.append("## Verdict")
    lines.append("")
    if any_pass:
        # Check whether any protocol-passer also has holdout xs > 0
        holdout_ok = False
        for rr in survivors:
            flags = protocol_pass(rr)
            if flags["passes_protocol_bar"] and flags["holdout_xs_positive"]:
                holdout_ok = True
                break
        lines.append(
            "**At least one rule passes the illustrative protocol bar** "
            "(val fitness & excess Sharpe > 0, and null p < 0.05). "
            "Holdout confirmation is reported separately above — do not re-select on it."
        )
        if not holdout_ok:
            lines.append("")
            lines.append(
                "However, **none of the protocol-passers show positive holdout excess Sharpe** "
                "in this run. Treat the validation/null clears as *interesting under the bar*, "
                "not as confirmed out-of-sample alpha — especially for multi-year sign regimes."
            )
    else:
        lines.append(
            "**No rule passes the illustrative protocol bar.** "
            "Either validation fitness/excess Sharpe were non-positive after costs, "
            "or the circular-shift null failed to reject (p ≥ 0.05) for the MC'd survivors. "
            "This is an honest negative / inconclusive result under the frozen protocol — "
            "not evidence that astrology 'works' or that the search was incomplete."
        )
    lines.append("")
    lines.append("## Honest interpretation")
    lines.append("")
    lines.append(
        "- Selection used **validation only**; holdout numbers are a one-shot report."
    )
    lines.append(
        "- Multiple testing is severe: hundreds of length-1 and thousands of length-2 "
        "candidates. A single-rule null p-value does **not** correct for the search. "
        "Even a p < 0.05 survivor should be treated cautiously."
    )
    lines.append(
        "- **Outer-planet sign regimes** (e.g. Saturn in Pisces / Aries) span years. "
        "High validation fitness for such atoms often reflects being long through a "
        "favorable multi-year market regime rather than high-frequency timing skill. "
        "Circular-shift nulls keep the contiguous True/False blocks intact, so they "
        "do **not** fully stress-test regime artifacts."
    )
    lines.append(
        "- Many top length-2 ANDs are near-aliases of a single strong length-1 atom "
        "(e.g. `~rare_aspect & saturn_sign_pisces` ≈ `saturn_sign_pisces`). "
        "Prefer the simpler length-1 form when holdout/val metrics are similar."
    )
    lines.append(
        "- Holdout excess Sharpe for the leading Saturn-Pisces family is typically "
        "**negative** (always-flat in holdout reads as −Sharpe_BH). "
        "That fails confirmation even when the validation+null bar is cleared."
    )
    lines.append(
        "- Costs are 10 bps/side; zero-cost wins are not counted. Flat earns 0 (no cash rf)."
    )
    lines.append(
        "- Adj Close is a proxy (dividends not fully total-return). Ephemeris is tropical geocentric."
    )
    lines.append(
        "- Prefer simpler rules (lower complexity); fitness already applies a λ penalty."
    )
    lines.append("")
    lines.append(f"Artifacts: `{csv_path.name}`, `{md_path.name}`.")
    lines.append("")

    md_path.write_text("\n".join(lines), encoding="utf-8")
    return csv_path, md_path


def _maybe_float(x: Any) -> float | None:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    try:
        if pd.isna(x):
            return None
    except (TypeError, ValueError):
        pass
    return float(x)


def print_console_summary(
    df: pd.DataFrame,
    survivors: Sequence[RuleResult],
    meta: dict[str, Any],
    csv_path: Path,
    md_path: Path,
) -> None:
    """Final console summary after search."""
    print()
    print("=" * 72)
    print("LATTICE SEARCH SUMMARY")
    print("=" * 72)
    print(
        f"Runtime {meta['runtime_sec']:.1f}s | ranked {meta['n_ranked']} | "
        f"MC n={meta['mc_n_sims']} on {meta['n_mc']} survivors"
    )
    print(f"Results: {md_path}")
    print(f"CSV:     {csv_path}")
    print()
    print(f"{'rank':>4}  {'rule':<42} {'val_fit':>8} {'hold_xs':>8} {'p':>8}  pass")
    print("-" * 84)
    any_pass = False
    for i, rr in enumerate(survivors[:10]):
        flags = protocol_pass(rr)
        if flags["passes_protocol_bar"]:
            any_pass = True
        hold_xs = (
            f"{rr.holdout_excess_sharpe:+.3f}"
            if rr.holdout_excess_sharpe is not None
            else "nan"
        )
        p = f"{rr.null_pvalue:.4f}" if rr.null_pvalue is not None else "n/a"
        print(
            f"{i + 1:4d}  {rr.rule:<42} {rr.val_fitness:+8.4f} {hold_xs:>8} {p:>8}  "
            f"{'YES' if flags['passes_protocol_bar'] else 'no'}"
        )
    print()
    if any_pass:
        print("Verdict: ≥1 rule PASSES protocol bar (val+ null). Holdout is report-only.")
    else:
        print("Verdict: NO rule passes protocol bar under frozen costs/splits/null.")
    print("=" * 72)


@click.command()
@click.option("--max-length", type=click.Choice(["1", "2"]), default="2", show_default=True)
@click.option(
    "--top-k",
    type=int,
    default=DEFAULT_TOP_K,
    show_default=True,
    help="Top length-1 by val fitness to carry into length-2 AND pairing",
)
@click.option(
    "--include-negation/--no-include-negation",
    default=True,
    show_default=True,
    help="Also evaluate ~atom for every length-1 atom",
)
@click.option(
    "--include-or-moon/--no-include-or-moon",
    default=True,
    show_default=True,
    help="Also evaluate moon_phase_* OR pairs",
)
@click.option(
    "--mc-n",
    type=int,
    default=DEFAULT_MC_N,
    show_default=True,
    help="Circular-shift null simulations per survivor",
)
@click.option("--config", "config_path", default=None, help="Path to YAML config")
@click.option(
    "--out-dir",
    type=click.Path(),
    default=None,
    help="Output directory (default: results/)",
)
def main(
    max_length: str,
    top_k: int,
    include_negation: bool,
    include_or_moon: bool,
    mc_n: int,
    config_path: str | None,
    out_dir: str | None,
) -> None:
    """Boolean lattice search (length-1 / length-2) under the frozen protocol."""
    try:
        cfg = load_config(config_path)
        df, survivors, meta = run_lattice_search(
            max_length=int(max_length),
            top_k=top_k,
            include_negation=include_negation,
            include_or_moon=include_or_moon,
            mc_n=mc_n,
            cfg=cfg,
            verbose=True,
        )
        out = Path(out_dir) if out_dir else (PROJECT_ROOT / "results")
        csv_path, md_path = write_results(df, survivors, meta, out_dir=out)
        print_console_summary(df, survivors, meta, csv_path, md_path)
    except FileNotFoundError as e:
        print(f"ERROR: missing cache.\n{e}", file=sys.stderr)
        sys.exit(2)
    except DataFetchError as e:
        print(f"ERROR: Could not load market data.\n{e}", file=sys.stderr)
        sys.exit(2)
    except EphemerisError as e:
        print(f"ERROR: Ephemeris unavailable.\n{e}", file=sys.stderr)
        sys.exit(2)
    except Exception as e:
        print(f"ERROR: {type(e).__name__}: {e}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
