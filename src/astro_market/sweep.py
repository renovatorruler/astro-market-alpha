"""Sweep v2: beam search L1→L2→L3 with regime/episode filters.

Protocol (frozen — do not change splits):
  Train 1970–1994, Validation 1995–2009, Holdout 2010–present.
  Selection uses VALIDATION fitness only (+ regime filters + split-half stability).
  Holdout is report-only.
  Fitness = excess_sharpe − λ·complexity (λ slightly higher for length-3).
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import asdict, dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Any, Sequence

import click
import numpy as np
import pandas as pd

from astro_market.atoms import (
    is_fast_planet_atom,
    is_regime_sign_atom,
    load_atoms_v2,
    searchable_atom_names,
)
from astro_market.config import PROJECT_ROOT, load_config
from astro_market.data import DataFetchError, daily_returns, load_prices
from astro_market.ephemeris import EphemerisError
from astro_market.null import empirical_pvalue
from astro_market.search import (
    complexity_for_rule,
    metrics_from_signal,
    signal_for_expr,
    split_returns_atoms,
    strategy_and_bh_returns,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_BEAM = 200
DEFAULT_MC_N = 5000
DEFAULT_LAMBDA = 0.01
DEFAULT_LAMBDA_L3 = 0.015
MAX_EPISODE_DAYS = 400
MIN_EPISODES_VAL = 4
MIN_EPISODES_TRAIN_VAL = 8
PARTIAL_EVERY = 500  # write partial CSV every N scored candidates at L2/L3


# ---------------------------------------------------------------------------
# Episode helpers
# ---------------------------------------------------------------------------


def count_episodes(mask: np.ndarray) -> int:
    """Count contiguous True runs (episodes) in a boolean mask."""
    m = np.asarray(mask, dtype=bool)
    if m.size == 0:
        return 0
    # Rising edges: False→True (treat start-of-series True as an edge)
    prev = np.concatenate([[False], m[:-1]])
    return int(np.sum(~prev & m))


def max_episode_length(mask: np.ndarray) -> int:
    """Length of the longest contiguous True run."""
    m = np.asarray(mask, dtype=bool)
    if m.size == 0 or not m.any():
        return 0
    # Run-length via cumulative sum reset
    # Pad with False so runs are bounded
    padded = np.concatenate([[False], m, [False]])
    diffs = np.diff(padded.astype(np.int8))
    starts = np.where(diffs == 1)[0]
    ends = np.where(diffs == -1)[0]
    if len(starts) == 0:
        return 0
    return int(np.max(ends - starts))


def episode_stats(mask: np.ndarray) -> tuple[int, int]:
    """Return (n_episodes, max_episode_length)."""
    return count_episodes(mask), max_episode_length(mask)


def rule_has_fast_atom(rule: str) -> bool:
    """True if any literal in the rule involves a fast planet."""
    # Split on & | ~ ( ) whitespace
    parts = (
        rule.replace("(", " ")
        .replace(")", " ")
        .replace("&", " ")
        .replace("|", " ")
        .replace("~", " ")
        .split()
    )
    return any(is_fast_planet_atom(p) for p in parts)


def rule_is_pure_regime_sign(rule: str) -> bool:
    """True if every atom literal is an outer-planet sign (or its negation)."""
    parts = (
        rule.replace("(", " ")
        .replace(")", " ")
        .replace("&", " ")
        .replace("|", " ")
        .replace("~", " ")
        .split()
    )
    if not parts:
        return False
    return all(is_regime_sign_atom(p) for p in parts)


# ---------------------------------------------------------------------------
# Canonical / complexity
# ---------------------------------------------------------------------------


def canonical_rule_v2(expr: str) -> str:
    """Canonicalize length-1/2/3 AND/OR rules (commutative sort of literals)."""
    s = expr.strip()
    if not s:
        return s
    if s.startswith("~") and "&" not in s and "|" not in s and "(" not in s:
        return f"~{s[1:].strip()}"

    for op, sym in (("&", " & "), ("|", " | ")):
        if op in s and "(" not in s:
            # Only pure AND or pure OR (no mixed)
            other = "|" if op == "&" else "&"
            if other in s:
                return s
            parts = [p.strip() for p in s.split(op) if p.strip()]
            if len(parts) >= 2:

                def sort_key(p: str) -> str:
                    return p[1:] if p.startswith("~") else p

                return sym.join(sorted(parts, key=sort_key))
    return s


def lambda_for_length(length: int, lam: float = DEFAULT_LAMBDA, lam_l3: float = DEFAULT_LAMBDA_L3) -> float:
    return lam_l3 if length >= 3 else lam


def length_of_rule(rule: str) -> int:
    core = rule.replace("~", "").replace(" ", "")
    if "&" not in core and "|" not in core:
        return 1
    # Count literals
    parts = (
        rule.replace("(", " ")
        .replace(")", " ")
        .replace("&", " ")
        .replace("|", " ")
        .replace("~", " ")
        .split()
    )
    return max(1, len(parts))


# ---------------------------------------------------------------------------
# Regime filter
# ---------------------------------------------------------------------------


@dataclass
class RegimeInfo:
    n_episodes_val: int = 0
    max_episode_val: int = 0
    n_episodes_train_val: int = 0
    dropped: bool = False
    drop_reason: str = ""
    penalty: float = 0.0


def apply_regime_filter(
    rule: str,
    sig_val: np.ndarray,
    sig_train: np.ndarray | None = None,
    *,
    max_episode: int = MAX_EPISODE_DAYS,
    min_ep_val: int = MIN_EPISODES_VAL,
    min_ep_tv: int = MIN_EPISODES_TRAIN_VAL,
) -> RegimeInfo:
    """
    Hard-drop / penalize long single-episode regime rules.

    - Hard drop if max True episode on validation > max_episode
      (chopping by a fast atom is evidenced by a shorter final max episode).
    - Hard drop pure outer-planet sign rules.
    - Soft penalty if too few distinct episodes.
    """
    info = RegimeInfo()
    n_ep, max_ep = episode_stats(sig_val)
    info.n_episodes_val = n_ep
    info.max_episode_val = max_ep

    if sig_train is not None:
        tv = np.concatenate([sig_train, sig_val])
        info.n_episodes_train_val = count_episodes(tv)
    else:
        info.n_episodes_train_val = n_ep

    if rule_is_pure_regime_sign(rule):
        info.dropped = True
        info.drop_reason = "pure_outer_sign_regime"
        return info

    if max_ep > max_episode:
        # Unless a fast planet chopped it — but then max_ep would be ≤ threshold.
        # Still allow if rule has fast atom AND max_ep somehow... no: hard drop.
        info.dropped = True
        info.drop_reason = f"max_episode_{max_ep}>{max_episode}"
        return info

    # Prefer ≥4 val episodes OR ≥8 train+val; else large fitness penalty
    if n_ep < min_ep_val and info.n_episodes_train_val < min_ep_tv:
        info.penalty = 0.50  # huge soft penalty
        info.drop_reason = "few_episodes_penalty"
    return info


# ---------------------------------------------------------------------------
# Candidate record
# ---------------------------------------------------------------------------


@dataclass
class SweepResult:
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
    val_fitness_raw: float = 0.0  # before regime penalty
    n_episodes_val: int = 0
    max_episode_val: int = 0
    n_episodes_train_val: int = 0
    val_half1_excess_sharpe: float | None = None
    val_half2_excess_sharpe: float | None = None
    stable: bool = False
    holdout_fitness: float | None = None
    holdout_excess_sharpe: float | None = None
    holdout_sharpe: float | None = None
    holdout_sharpe_bh: float | None = None
    null_pvalue: float | None = None
    null_n: int | None = None
    selected_on: str = "validation"
    drop_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def score_rule(
    rule: str,
    sig_train: np.ndarray,
    sig_val: np.ndarray,
    r_train: np.ndarray,
    r_val: np.ndarray,
    *,
    bps: float,
    lam: float,
    lam_l3: float,
    apply_filter: bool = True,
) -> SweepResult | None:
    """Score one rule; return None if hard-dropped by regime filter."""
    length = length_of_rule(rule)
    c = complexity_for_rule(rule)
    lam_use = lambda_for_length(length, lam, lam_l3)

    if apply_filter:
        info = apply_regime_filter(rule, sig_val, sig_train)
        if info.dropped:
            return None
    else:
        info = RegimeInfo(*episode_stats(sig_val), n_episodes_train_val=0)

    m_tr = metrics_from_signal(
        sig_train, r_train, complexity=c, bps_per_side=bps, lambda_complexity=lam_use
    )
    m_va = metrics_from_signal(
        sig_val, r_val, complexity=c, bps_per_side=bps, lambda_complexity=lam_use
    )
    raw_fit = m_va["fitness"]
    adj_fit = (
        float(raw_fit - info.penalty)
        if np.isfinite(raw_fit)
        else float("-inf")
    )
    return SweepResult(
        rule=canonical_rule_v2(rule),
        length=length,
        complexity=c,
        train_fitness=m_tr["fitness"],
        train_excess_sharpe=m_tr["excess_sharpe"],
        train_sharpe=m_tr["sharpe"],
        val_fitness=adj_fit,
        val_excess_sharpe=m_va["excess_sharpe"],
        val_sharpe=m_va["sharpe"],
        val_sharpe_bh=m_va["sharpe_bh"],
        val_fitness_raw=float(raw_fit) if np.isfinite(raw_fit) else float("-inf"),
        n_episodes_val=info.n_episodes_val,
        max_episode_val=info.max_episode_val,
        n_episodes_train_val=info.n_episodes_train_val,
        drop_reason=info.drop_reason,
    )


def split_val_halves(
    r_val: np.ndarray,
    sig_val: np.ndarray,
    *,
    complexity: int,
    bps: float,
    lam: float,
) -> tuple[float, float]:
    """Excess Sharpe on first / second temporal half of validation."""
    n = len(r_val)
    mid = n // 2
    m1 = metrics_from_signal(
        sig_val[:mid], r_val[:mid], complexity=complexity, bps_per_side=bps, lambda_complexity=lam
    )
    m2 = metrics_from_signal(
        sig_val[mid:], r_val[mid:], complexity=complexity, bps_per_side=bps, lambda_complexity=lam
    )
    return float(m1["excess_sharpe"]), float(m2["excess_sharpe"])


def attach_stability(
    results: Sequence[SweepResult],
    atoms_val: pd.DataFrame,
    r_val: np.ndarray,
    *,
    bps: float,
    lam: float,
    lam_l3: float,
) -> list[SweepResult]:
    """Mark stable survivors: positive excess Sharpe in BOTH val halves."""
    out: list[SweepResult] = []
    for rr in results:
        lam_use = lambda_for_length(rr.length, lam, lam_l3)
        sig = signal_for_expr(rr.rule, atoms_val)
        h1, h2 = split_val_halves(
            r_val, sig, complexity=rr.complexity, bps=bps, lam=lam_use
        )
        rr.val_half1_excess_sharpe = h1
        rr.val_half2_excess_sharpe = h2
        rr.stable = bool(
            np.isfinite(h1) and np.isfinite(h2) and h1 > 0 and h2 > 0
        )
        out.append(rr)
    return out


# ---------------------------------------------------------------------------
# Beam stages
# ---------------------------------------------------------------------------


def beam_l1(
    atoms_train: pd.DataFrame,
    atoms_val: pd.DataFrame,
    r_train: np.ndarray,
    r_val: np.ndarray,
    *,
    searchable: Sequence[str],
    beam: int = DEFAULT_BEAM,
    include_negation: bool = True,
    bps: float = 10.0,
    lam: float = DEFAULT_LAMBDA,
    lam_l3: float = DEFAULT_LAMBDA_L3,
    verbose: bool = True,
) -> list[SweepResult]:
    """Evaluate all searchable atoms (+negations); keep top `beam` by val fitness."""
    results: list[SweepResult] = []
    seen: set[str] = set()
    n_drop = 0
    for col in searchable:
        cands = [col]
        if include_negation:
            cands.append(f"~{col}")
        for rule in cands:
            key = canonical_rule_v2(rule)
            if key in seen:
                continue
            seen.add(key)
            sig_tr = signal_for_expr(rule, atoms_train)
            sig_va = signal_for_expr(rule, atoms_val)
            # Degenerate skip
            if not np.any(sig_va) or np.all(sig_va):
                continue
            rr = score_rule(
                rule, sig_tr, sig_va, r_train, r_val,
                bps=bps, lam=lam, lam_l3=lam_l3, apply_filter=True,
            )
            if rr is None:
                n_drop += 1
                continue
            results.append(rr)
    results.sort(key=lambda x: x.val_fitness, reverse=True)
    if verbose:
        print(
            f"  L1: scored={len(results)} dropped_regime={n_drop} "
            f"beam_keep={min(beam, len(results))}"
        )
    return results[:beam]


def beam_l2(
    l1_beam: Sequence[SweepResult],
    atoms_train: pd.DataFrame,
    atoms_val: pd.DataFrame,
    r_train: np.ndarray,
    r_val: np.ndarray,
    *,
    beam: int = DEFAULT_BEAM,
    bps: float = 10.0,
    lam: float = DEFAULT_LAMBDA,
    lam_l3: float = DEFAULT_LAMBDA_L3,
    include_moon_or: bool = True,
    partial_path: Path | None = None,
    verbose: bool = True,
) -> list[SweepResult]:
    """AND pairs among L1 beam (+ limited moon ORs); keep top `beam`."""
    seeds = list(l1_beam)
    sig_tr = {rr.rule: signal_for_expr(rr.rule, atoms_train) for rr in seeds}
    sig_va = {rr.rule: signal_for_expr(rr.rule, atoms_val) for rr in seeds}

    results: list[SweepResult] = []
    seen: set[str] = set()
    n_scored = 0
    n_drop = 0

    for ra, rb in combinations(seeds, 2):
        a, b = ra.rule, rb.rule
        # Skip pairing literal with its negation
        if a.lstrip("~") == b.lstrip("~") and (a.startswith("~") != b.startswith("~")):
            continue
        rule = canonical_rule_v2(f"{a} & {b}")
        if rule in seen:
            continue
        seen.add(rule)
        sva = sig_va[a] & sig_va[b]
        if not np.any(sva) or np.all(sva):
            continue
        if np.array_equal(sva, sig_va[a]) or np.array_equal(sva, sig_va[b]):
            continue
        str_ = sig_tr[a] & sig_tr[b]
        rr = score_rule(
            rule, str_, sva, r_train, r_val,
            bps=bps, lam=lam, lam_l3=lam_l3, apply_filter=True,
        )
        n_scored += 1
        if rr is None:
            n_drop += 1
            continue
        results.append(rr)
        if partial_path and n_scored % PARTIAL_EVERY == 0:
            _write_partial(results, partial_path, stage="L2")

    # Moon OR pairs (limited)
    if include_moon_or:
        phases = [c for c in atoms_val.columns if c.startswith("moon_phase_")]
        for a, b in combinations(phases, 2):
            rule = canonical_rule_v2(f"{a} | {b}")
            if rule in seen:
                continue
            seen.add(rule)
            sva = signal_for_expr(a, atoms_val) | signal_for_expr(b, atoms_val)
            if not np.any(sva) or np.all(sva):
                continue
            str_ = signal_for_expr(a, atoms_train) | signal_for_expr(b, atoms_train)
            rr = score_rule(
                rule, str_, sva, r_train, r_val,
                bps=bps, lam=lam, lam_l3=lam_l3, apply_filter=True,
            )
            n_scored += 1
            if rr is None:
                n_drop += 1
                continue
            results.append(rr)

    results.sort(key=lambda x: x.val_fitness, reverse=True)
    if verbose:
        print(
            f"  L2: scored={n_scored} kept={len(results)} dropped_regime={n_drop} "
            f"beam_keep={min(beam, len(results))}"
        )
    if partial_path:
        _write_partial(results[:beam], partial_path, stage="L2_beam")
    return results[:beam]


def beam_l3(
    l2_beam: Sequence[SweepResult],
    l1_beam: Sequence[SweepResult],
    atoms_train: pd.DataFrame,
    atoms_val: pd.DataFrame,
    r_train: np.ndarray,
    r_val: np.ndarray,
    *,
    beam: int = DEFAULT_BEAM,
    bps: float = 10.0,
    lam: float = DEFAULT_LAMBDA,
    lam_l3: float = DEFAULT_LAMBDA_L3,
    partial_path: Path | None = None,
    verbose: bool = True,
) -> list[SweepResult]:
    """Extend each top L2 with one more atom from L1 beam (AND)."""
    sig_l1_tr = {rr.rule: signal_for_expr(rr.rule, atoms_train) for rr in l1_beam}
    sig_l1_va = {rr.rule: signal_for_expr(rr.rule, atoms_val) for rr in l1_beam}
    sig_l2_tr = {rr.rule: signal_for_expr(rr.rule, atoms_train) for rr in l2_beam}
    sig_l2_va = {rr.rule: signal_for_expr(rr.rule, atoms_val) for rr in l2_beam}

    results: list[SweepResult] = []
    seen: set[str] = set()
    n_scored = 0
    n_drop = 0

    l1_lits = list(l1_beam)
    for r2 in l2_beam:
        base = r2.rule
        # Literals already in base
        base_atoms = set(
            base.replace("~", " ").replace("&", " ").replace("|", " ").split()
        )
        for r1 in l1_lits:
            lit = r1.rule
            lit_core = lit.lstrip("~")
            if lit_core in base_atoms or lit in base_atoms:
                continue
            # Skip adding negation of something already present
            if lit.startswith("~") and lit_core in base_atoms:
                continue
            if (not lit.startswith("~") and f"~{lit}" in base.replace(" ", "").split("&")):
                pass  # handled by literal set roughly
            rule = canonical_rule_v2(f"{base} & {lit}")
            if rule in seen:
                continue
            seen.add(rule)
            sva = sig_l2_va[base] & sig_l1_va[lit]
            if not np.any(sva) or np.all(sva):
                continue
            if np.array_equal(sva, sig_l2_va[base]) or np.array_equal(sva, sig_l1_va[lit]):
                continue
            str_ = sig_l2_tr[base] & sig_l1_tr[lit]
            rr = score_rule(
                rule, str_, sva, r_train, r_val,
                bps=bps, lam=lam, lam_l3=lam_l3, apply_filter=True,
            )
            n_scored += 1
            if rr is None:
                n_drop += 1
                continue
            results.append(rr)
            if partial_path and n_scored % PARTIAL_EVERY == 0:
                _write_partial(results, partial_path, stage="L3")

    results.sort(key=lambda x: x.val_fitness, reverse=True)
    if verbose:
        print(
            f"  L3: scored={n_scored} kept={len(results)} dropped_regime={n_drop} "
            f"beam_keep={min(beam, len(results))}"
        )
    if partial_path:
        _write_partial(results[:beam], partial_path, stage="L3_beam")
    return results[:beam]


def _write_partial(results: Sequence[SweepResult], path: Path, stage: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [r.to_dict() for r in sorted(results, key=lambda x: x.val_fitness, reverse=True)]
    df = pd.DataFrame(rows)
    df["partial_stage"] = stage
    df.to_csv(path, index=False)


# ---------------------------------------------------------------------------
# Holdout + MC
# ---------------------------------------------------------------------------


def attach_holdout(
    rules: Sequence[SweepResult],
    atoms_hold: pd.DataFrame,
    r_hold: np.ndarray,
    *,
    bps: float,
    lam: float,
    lam_l3: float,
) -> list[SweepResult]:
    out: list[SweepResult] = []
    for rr in rules:
        lam_use = lambda_for_length(rr.length, lam, lam_l3)
        sig = signal_for_expr(rr.rule, atoms_hold)
        m = metrics_from_signal(
            sig, r_hold, complexity=rr.complexity, bps_per_side=bps, lambda_complexity=lam_use
        )
        rr.holdout_fitness = m["fitness"]
        rr.holdout_excess_sharpe = m["excess_sharpe"]
        rr.holdout_sharpe = m["sharpe"]
        rr.holdout_sharpe_bh = m["sharpe_bh"]
        out.append(rr)
    return out


def monte_carlo_fitness_pvalue(
    rule: str,
    atoms_val: pd.DataFrame,
    r_val: np.ndarray,
    *,
    n: int = DEFAULT_MC_N,
    seed: int = 42,
    bps: float = 10.0,
    lam: float = DEFAULT_LAMBDA,
    lam_l3: float = DEFAULT_LAMBDA_L3,
) -> float:
    """
    Circular-shift null on validation fitness (vectorized cost precompute).
    """
    signal = signal_for_expr(rule, atoms_val)
    length = length_of_rule(rule)
    c = complexity_for_rule(rule)
    lam_use = lambda_for_length(length, lam, lam_l3)
    observed = metrics_from_signal(
        signal, r_val, complexity=c, bps_per_side=bps, lambda_complexity=lam_use
    )["fitness"]

    n_days = len(r_val)
    if n_days < 2:
        return float("nan")

    # Precompute position lag + costs (independent of return shifts)
    pos = signal.astype(float)
    pos_lag = np.empty(n_days, dtype=float)
    pos_lag[0] = 0.0
    if n_days > 1:
        pos_lag[1:] = pos[:-1]
    cost_unit = bps / 10_000.0
    delta = np.empty(n_days, dtype=float)
    delta[0] = abs(pos_lag[0])
    if n_days > 1:
        delta[1:] = np.abs(np.diff(pos_lag))
    costs = cost_unit * delta

    bh_lag = np.empty(n_days, dtype=float)
    bh_lag[0] = 0.0
    if n_days > 1:
        bh_lag[1:] = 1.0
    bh_delta = np.empty(n_days, dtype=float)
    bh_delta[0] = abs(bh_lag[0])
    if n_days > 1:
        bh_delta[1:] = np.abs(np.diff(bh_lag))
    bh_costs = cost_unit * bh_delta

    rng = np.random.default_rng(seed)
    shifts = rng.integers(1, n_days, size=n)
    scores = np.empty(n, dtype=float)

    from astro_market.search import _sharpe_np

    for i, k in enumerate(shifts):
        r_shift = np.roll(r_val, int(k))
        strat = pos_lag * r_shift - costs
        bh = bh_lag * r_shift - bh_costs
        sh_s = _sharpe_np(strat)
        sh_b = _sharpe_np(bh)
        excess = (sh_s - sh_b) if np.isfinite(sh_s) and np.isfinite(sh_b) else float("nan")
        scores[i] = (
            float(excess - lam_use * c) if np.isfinite(excess) else float("-inf")
        )

    return empirical_pvalue(observed, scores, alternative="greater")


def select_mc_targets(
    ranked: Sequence[SweepResult],
    *,
    n_stable: int = 30,
    n_unstable: int = 10,
) -> list[SweepResult]:
    """Top stable survivors (up to n_stable) + top unstable-but-high-fit (honesty)."""
    stable = [r for r in ranked if r.stable]
    unstable = [r for r in ranked if not r.stable]
    chosen: list[SweepResult] = []
    seen: set[str] = set()
    for r in stable[:n_stable]:
        k = canonical_rule_v2(r.rule)
        if k not in seen:
            seen.add(k)
            chosen.append(r)
    for r in unstable[:n_unstable]:
        k = canonical_rule_v2(r.rule)
        if k not in seen:
            seen.add(k)
            chosen.append(r)
    return chosen


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def run_sweep_v2(
    *,
    beam: int = DEFAULT_BEAM,
    mc_n: int = DEFAULT_MC_N,
    include_negation: bool = True,
    include_moon_or: bool = True,
    force_rebuild_atoms: bool = False,
    cfg: dict | None = None,
    verbose: bool = True,
    out_dir: Path | None = None,
) -> tuple[pd.DataFrame, list[SweepResult], dict[str, Any]]:
    """Full sweep v2 end-to-end."""
    t0 = time.perf_counter()
    cfg = cfg or load_config()
    bps = float(cfg["costs"]["bps_per_side"])
    lam = float(cfg["evaluate"].get("complexity_lambda", DEFAULT_LAMBDA))
    lam_l3 = float(cfg.get("sweep_v2", {}).get("lambda_l3", DEFAULT_LAMBDA_L3))
    seed = int(cfg.get("null", {}).get("seed", 42))
    out_dir = out_dir or (PROJECT_ROOT / "results")
    out_dir.mkdir(parents=True, exist_ok=True)
    partial_path = out_dir / "sweep_v2_partial.csv"
    n_cores = os.cpu_count() or 1

    if verbose:
        print(f"Sweep v2 | cores={n_cores} beam={beam} mc_n={mc_n} λ={lam} λ_L3={lam_l3}")
        print("Loading prices + building/loading atoms_v2...")

    prices = load_prices(cfg=cfg, auto_fetch=False)
    rets = daily_returns(prices)
    t_atoms = time.perf_counter()
    atoms = load_atoms_v2(
        cfg=cfg, dates=prices.index, auto_build=True, force_rebuild=force_rebuild_atoms
    )
    atom_build_sec = time.perf_counter() - t_atoms
    n_atoms_total = int(atoms.shape[1])
    searchable = searchable_atom_names(atoms.columns)
    n_atoms_search = len(searchable)

    common = rets.index.intersection(atoms.index)
    rets = rets.loc[common]
    atoms = atoms.loc[common]

    r_train, a_train, _ = split_returns_atoms(rets, atoms, cfg, "train")
    r_val, a_val, _ = split_returns_atoms(rets, atoms, cfg, "validation")
    r_hold, a_hold, _ = split_returns_atoms(rets, atoms, cfg, "holdout")

    if verbose:
        print(
            f"Atoms total={n_atoms_total} searchable={n_atoms_search} "
            f"(build/load {atom_build_sec:.1f}s)"
        )
        print(
            f"Splits: train={len(r_train)}d val={len(r_val)}d hold={len(r_hold)}d"
        )
        print("L1 beam search...")

    t1 = time.perf_counter()
    l1 = beam_l1(
        a_train, a_val, r_train, r_val,
        searchable=searchable,
        beam=beam,
        include_negation=include_negation,
        bps=bps, lam=lam, lam_l3=lam_l3,
        verbose=verbose,
    )
    l1_sec = time.perf_counter() - t1
    if verbose and l1:
        print(f"  L1 done in {l1_sec:.1f}s  top: {l1[0].rule} fit={l1[0].val_fitness:+.4f}")
        _write_partial(l1, partial_path, stage="L1_beam")

    if verbose:
        print("L2 beam search (AND pairs + moon ORs)...")
    t2 = time.perf_counter()
    l2 = beam_l2(
        l1, a_train, a_val, r_train, r_val,
        beam=beam, bps=bps, lam=lam, lam_l3=lam_l3,
        include_moon_or=include_moon_or,
        partial_path=partial_path,
        verbose=verbose,
    )
    l2_sec = time.perf_counter() - t2
    if verbose:
        print(f"  L2 done in {l2_sec:.1f}s")

    if verbose:
        print("L3 beam search (L2 ⊕ L1)...")
    t3 = time.perf_counter()
    l3 = beam_l3(
        l2, l1, a_train, a_val, r_train, r_val,
        beam=beam, bps=bps, lam=lam, lam_l3=lam_l3,
        partial_path=partial_path,
        verbose=verbose,
    )
    l3_sec = time.perf_counter() - t3
    if verbose:
        print(f"  L3 done in {l3_sec:.1f}s")

    # Merge all beams, dedup by canonical rule (best val fitness)
    best: dict[str, SweepResult] = {}
    for rr in list(l1) + list(l2) + list(l3):
        key = canonical_rule_v2(rr.rule)
        rr.rule = key
        prev = best.get(key)
        if prev is None or rr.val_fitness > prev.val_fitness:
            best[key] = rr
    ranked = sorted(best.values(), key=lambda x: x.val_fitness, reverse=True)

    if verbose:
        print(f"Unique ranked={len(ranked)}; attaching split-half stability...")
    ranked = attach_stability(ranked, a_val, r_val, bps=bps, lam=lam, lam_l3=lam_l3)
    n_stable = sum(1 for r in ranked if r.stable)
    if verbose:
        print(f"  Stable survivors (XS>0 in both val halves): {n_stable}")

    if verbose:
        print("Attaching holdout (report-only)...")
    ranked = attach_holdout(ranked, a_hold, r_hold, bps=bps, lam=lam, lam_l3=lam_l3)

    mc_targets = select_mc_targets(ranked, n_stable=30, n_unstable=10)
    if verbose:
        print(
            f"Monte Carlo null n={mc_n} on {len(mc_targets)} targets "
            f"({sum(1 for r in mc_targets if r.stable)} stable + "
            f"{sum(1 for r in mc_targets if not r.stable)} unstable)..."
        )
    t4 = time.perf_counter()
    for i, rr in enumerate(mc_targets):
        p = monte_carlo_fitness_pvalue(
            rr.rule, a_val, r_val, n=mc_n, seed=seed, bps=bps, lam=lam, lam_l3=lam_l3
        )
        rr.null_pvalue = p
        rr.null_n = mc_n
        for r2 in ranked:
            if canonical_rule_v2(r2.rule) == canonical_rule_v2(rr.rule):
                r2.null_pvalue = p
                r2.null_n = mc_n
                break
        if verbose and ((i + 1) % 5 == 0 or i == 0 or i + 1 == len(mc_targets)):
            print(
                f"  [{i + 1}/{len(mc_targets)}] {rr.rule[:50]:<50} "
                f"fit={rr.val_fitness:+.3f} stable={rr.stable} p={p:.4f}"
            )
        if (i + 1) % 10 == 0:
            _write_partial(ranked[:500], partial_path, stage=f"MC_{i+1}")
    mc_sec = time.perf_counter() - t4
    if verbose:
        print(f"  MC done in {mc_sec:.1f}s")

    df = pd.DataFrame([r.to_dict() for r in ranked])
    runtime = time.perf_counter() - t0
    meta = {
        "runtime_sec": runtime,
        "atom_build_sec": atom_build_sec,
        "l1_sec": l1_sec,
        "l2_sec": l2_sec,
        "l3_sec": l3_sec,
        "mc_sec": mc_sec,
        "n_cores": n_cores,
        "n_atoms_total": n_atoms_total,
        "n_atoms_searchable": n_atoms_search,
        "beam": beam,
        "n_l1_beam": len(l1),
        "n_l2_beam": len(l2),
        "n_l3_beam": len(l3),
        "n_ranked": len(ranked),
        "n_stable": n_stable,
        "n_mc_targets": len(mc_targets),
        "mc_n_sims": mc_n,
        "lambda": lam,
        "lambda_l3": lam_l3,
        "bps": bps,
        "seed": seed,
        "n_train": len(r_train),
        "n_val": len(r_val),
        "n_holdout": len(r_hold),
        "max_episode_days": MAX_EPISODE_DAYS,
        "min_episodes_val": MIN_EPISODES_VAL,
        "min_episodes_train_val": MIN_EPISODES_TRAIN_VAL,
    }
    return df, mc_targets, meta


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def protocol_pass_v2(rr: SweepResult, p_threshold: float = 0.05) -> dict[str, bool]:
    val_ok = (
        rr.val_excess_sharpe is not None
        and rr.val_fitness is not None
        and rr.val_excess_sharpe > 0
        and rr.val_fitness > 0
    )
    halves_ok = bool(rr.stable)
    null_ok = rr.null_pvalue is not None and rr.null_pvalue < p_threshold
    holdout_pos = (
        rr.holdout_excess_sharpe is not None and rr.holdout_excess_sharpe > 0
    )
    return {
        "val_interesting": bool(val_ok),
        "stable_halves": halves_ok,
        "null_reject": bool(null_ok),
        "holdout_xs_positive": bool(holdout_pos),
        "passes_stable_bar": bool(val_ok and halves_ok and null_ok),
        "passes_with_holdout": bool(val_ok and halves_ok and null_ok and holdout_pos),
    }


def write_sweep_results(
    df: pd.DataFrame,
    mc_targets: Sequence[SweepResult],
    meta: dict[str, Any],
    *,
    out_dir: Path | None = None,
    atoms_val: pd.DataFrame | None = None,
    r_val: np.ndarray | None = None,
    r_hold: np.ndarray | None = None,
    prices_index: pd.DatetimeIndex | None = None,
) -> tuple[Path, Path]:
    out_dir = out_dir or (PROJECT_ROOT / "results")
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "sweep_v2.csv"
    md_path = out_dir / "sweep_v2.md"
    df.to_csv(csv_path, index=False)

    stable_df = df[df["stable"] == True] if "stable" in df.columns else df.iloc[0:0]  # noqa: E712
    val_passers = df[(df["val_fitness"] > 0) & (df["val_excess_sharpe"] > 0)]

    lines: list[str] = []
    lines.append("# Sweep v2 — Beam search L1→L2→L3 with regime filters")
    lines.append("")
    lines.append("## Method")
    lines.append("")
    lines.append(
        "Richer tropical geocentric atoms (Uranus/Neptune/Pluto for aspects/stations/ingress; "
        "multi-orb aspect families 1°/3°/5°; 8 moon-phase buckets). "
        "**Outer-planet sign atoms** (Jupiter–Pluto) are computed for analysis but "
        "**excluded from the searchable pool** to avoid Saturn-Pisces-style multi-year regime artifacts."
    )
    lines.append("")
    lines.append(
        f"Beam search: L1 (atoms+negations) → top `{meta.get('beam')}` by validation fitness "
        f"(after regime filter) → L2 AND pairs (+ moon ORs) → L3 = L2 ⊕ one L1 literal. "
        f"Complexity λ={meta.get('lambda')} (L1/L2), λ_L3={meta.get('lambda_l3')}."
    )
    lines.append("")
    lines.append("**Regime / episode filters:**")
    lines.append("")
    lines.append(
        f"- Hard drop rules with a single validation True episode longer than "
        f"**{meta.get('max_episode_days')}** trading days"
    )
    lines.append(
        f"- Soft penalty (−0.50 fitness) if episodes < {meta.get('min_episodes_val')} on val "
        f"**and** < {meta.get('min_episodes_train_val')} on train+val"
    )
    lines.append("- Hard drop pure outer-planet sign rules")
    lines.append(
        "- **Stable survivor**: positive excess Sharpe on **both** temporal halves of validation"
    )
    lines.append("")
    lines.append("**Frozen protocol:**")
    lines.append("")
    lines.append("| Item | Value |")
    lines.append("|------|-------|")
    lines.append("| Train | 1970-01-01 → 1994-12-31 |")
    lines.append("| Validation | 1995-01-01 → 2009-12-31 |")
    lines.append("| Holdout | 2010-01-01 → present (**never used for selection**) |")
    lines.append("| Position | Long when True; flat when False; BH = always long |")
    lines.append(f"| Costs | {meta.get('bps')} bps / side |")
    lines.append(
        f"| Monte Carlo | Circular-shift, n={meta.get('mc_n_sims')}, seed={meta.get('seed')}, validation |"
    )
    lines.append("")
    lines.append(
        f"Runtime: **{meta.get('runtime_sec', float('nan')):.1f}s** "
        f"(atoms {meta.get('atom_build_sec', 0):.1f}s · L1 {meta.get('l1_sec', 0):.1f}s · "
        f"L2 {meta.get('l2_sec', 0):.1f}s · L3 {meta.get('l3_sec', 0):.1f}s · "
        f"MC {meta.get('mc_sec', 0):.1f}s) · "
        f"cores={meta.get('n_cores')} · "
        f"atoms total/searchable={meta.get('n_atoms_total')}/{meta.get('n_atoms_searchable')} · "
        f"ranked={meta.get('n_ranked')} · stable={meta.get('n_stable')} · "
        f"days train/val/hold={meta.get('n_train')}/{meta.get('n_val')}/{meta.get('n_holdout')}"
    )
    lines.append("")

    # Top all val-passers
    lines.append("## Top rules by validation fitness (all val-passers pool)")
    lines.append("")
    lines.append(
        "| Rank | Rule | L | Ep | MaxEp | Val XS | Val Fit | H1 XS | H2 XS | Stable | Hold XS | Null p |"
    )
    lines.append(
        "|-----:|------|--:|---:|------:|-------:|--------:|------:|------:|:------:|--------:|-------:|"
    )
    top_n = min(25, len(df))
    for rank, (_, row) in enumerate(df.head(top_n).iterrows(), start=1):
        h1 = row.get("val_half1_excess_sharpe")
        h2 = row.get("val_half2_excess_sharpe")
        h1s = f"{float(h1):+.3f}" if pd.notna(h1) else "n/a"
        h2s = f"{float(h2):+.3f}" if pd.notna(h2) else "n/a"
        hxs = row.get("holdout_excess_sharpe")
        hxs_s = f"{float(hxs):+.3f}" if pd.notna(hxs) else "n/a"
        p = row.get("null_pvalue")
        p_s = f"{float(p):.4f}" if pd.notna(p) else "—"
        st = "Y" if bool(row.get("stable")) else "N"
        lines.append(
            f"| {rank} | `{row['rule']}` | {int(row['length'])} | "
            f"{int(row.get('n_episodes_val', 0))} | {int(row.get('max_episode_val', 0))} | "
            f"{float(row['val_excess_sharpe']):+.3f} | {float(row['val_fitness']):+.3f} | "
            f"{h1s} | {h2s} | {st} | {hxs_s} | {p_s} |"
        )
    lines.append("")

    # Stable survivors
    lines.append("## Stable survivors (XS > 0 in both validation halves)")
    lines.append("")
    if stable_df.empty:
        lines.append("_No stable survivors._")
    else:
        lines.append(
            "| Rank | Rule | L | Val Fit | Val XS | H1 | H2 | Hold XS | Null p | Bar+HO |"
        )
        lines.append(
            "|-----:|------|--:|--------:|-------:|---:|---:|--------:|-------:|:------:|"
        )
        for rank, (_, row) in enumerate(stable_df.head(30).iterrows(), start=1):
            rr = _row_to_sweep(row)
            flags = protocol_pass_v2(rr)
            hxs = rr.holdout_excess_sharpe
            hxs_s = f"{hxs:+.3f}" if hxs is not None and np.isfinite(hxs) else "n/a"
            p_s = f"{rr.null_pvalue:.4f}" if rr.null_pvalue is not None else "—"
            bar = "YES" if flags["passes_with_holdout"] else (
                "val+null" if flags["passes_stable_bar"] else "no"
            )
            lines.append(
                f"| {rank} | `{rr.rule}` | {rr.length} | {rr.val_fitness:+.3f} | "
                f"{rr.val_excess_sharpe:+.3f} | "
                f"{rr.val_half1_excess_sharpe:+.3f} | {rr.val_half2_excess_sharpe:+.3f} | "
                f"{hxs_s} | {p_s} | {bar} |"
            )
    lines.append("")

    # MC detail
    lines.append("## Monte Carlo targets (stable + honesty unstable)")
    lines.append("")
    if not mc_targets:
        lines.append("_None._")
    else:
        lines.append(
            "| Rule | Stable | Val Fit | Hold XS | Null p | Stable bar | Bar+holdout |"
        )
        lines.append(
            "|------|:------:|--------:|--------:|-------:|:----------:|:-----------:|"
        )
        for rr in mc_targets:
            flags = protocol_pass_v2(rr)
            hxs = rr.holdout_excess_sharpe
            hxs_s = f"{hxs:+.4f}" if hxs is not None and np.isfinite(hxs) else "n/a"
            p_s = f"{rr.null_pvalue:.4f}" if rr.null_pvalue is not None else "n/a"
            lines.append(
                f"| `{rr.rule}` | {'Y' if rr.stable else 'N'} | {rr.val_fitness:+.4f} | "
                f"{hxs_s} | {p_s} | "
                f"{'YES' if flags['passes_stable_bar'] else 'no'} | "
                f"{'YES' if flags['passes_with_holdout'] else 'no'} |"
            )
    lines.append("")

    # Verdict
    lines.append("## Verdict")
    lines.append("")
    any_stable_bar = False
    any_with_holdout = False
    for rr in mc_targets:
        flags = protocol_pass_v2(rr)
        if flags["passes_stable_bar"]:
            any_stable_bar = True
        if flags["passes_with_holdout"]:
            any_with_holdout = True

    if any_with_holdout:
        lines.append(
            "**YES — at least one stable survivor clears val + both halves + null p<0.05 "
            "AND holdout excess Sharpe > 0.** Treat cautiously given multiple testing."
        )
    elif any_stable_bar:
        lines.append(
            "**Partial:** ≥1 stable survivor clears val + both halves + null p<0.05, "
            "but **none** show positive holdout excess Sharpe. "
            "Not confirmed out-of-sample alpha."
        )
    elif not stable_df.empty:
        lines.append(
            "**No:** stable survivors exist on split-half XS, but none jointly clear "
            "the null p<0.05 bar with positive val fitness (or MC was inconclusive). "
            "Honest negative / inconclusive under the tightened regime protocol."
        )
    else:
        lines.append(
            "**No stable survivors.** After episode/regime filters and split-half "
            "stability, nothing remains with positive excess Sharpe in both validation "
            "halves. This is the honest answer under sweep v2 — lattice v1's Saturn-Pisces "
            "family would have been hard-dropped here as a multi-year regime artifact."
        )
    lines.append("")
    lines.append("## Honest interpretation")
    lines.append("")
    lines.append(
        f"- Searchable atoms: **{meta.get('n_atoms_searchable')}** "
        f"(total cached including outer signs: {meta.get('n_atoms_total')})."
    )
    lines.append(
        f"- Val-passers (fit>0 & XS>0): **{len(val_passers)}**; "
        f"stable: **{meta.get('n_stable')}**."
    )
    lines.append(
        "- Selection never used holdout. Multiple testing across thousands of beam "
        "candidates means even p<0.05 is weak evidence."
    )
    lines.append(
        "- Episode filter is the main upgrade vs lattice v1: long outer-planet sign "
        "regimes no longer dominate the leaderboard."
    )
    lines.append(
        "- Costs 10 bps/side; Adj Close proxy; tropical geocentric ephemeris."
    )
    lines.append("")
    lines.append(f"Artifacts: `{csv_path.name}`, `{md_path.name}`.")
    lines.append("")

    md_path.write_text("\n".join(lines), encoding="utf-8")
    return csv_path, md_path


def _row_to_sweep(row: pd.Series) -> SweepResult:
    def mf(key: str) -> float | None:
        v = row.get(key)
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return None
        try:
            if pd.isna(v):
                return None
        except (TypeError, ValueError):
            pass
        return float(v)

    return SweepResult(
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
        n_episodes_val=int(row.get("n_episodes_val", 0) or 0),
        max_episode_val=int(row.get("max_episode_val", 0) or 0),
        val_half1_excess_sharpe=mf("val_half1_excess_sharpe"),
        val_half2_excess_sharpe=mf("val_half2_excess_sharpe"),
        stable=bool(row.get("stable", False)),
        holdout_fitness=mf("holdout_fitness"),
        holdout_excess_sharpe=mf("holdout_excess_sharpe"),
        holdout_sharpe=mf("holdout_sharpe"),
        holdout_sharpe_bh=mf("holdout_sharpe_bh"),
        null_pvalue=mf("null_pvalue"),
        null_n=int(row["null_n"]) if pd.notna(row.get("null_n")) else None,
    )


def maybe_plot_top_stable(
    rr: SweepResult,
    atoms: pd.DataFrame,
    returns: pd.Series,
    cfg: dict,
    out_path: Path,
    bps: float = 10.0,
) -> Path | None:
    """Optional equity chart: top stable rule vs BH on full common history."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    from astro_market.data import split_mask

    sig = signal_for_expr(rr.rule, atoms.reindex(returns.index).fillna(False).astype(bool))
    r = returns.to_numpy(dtype=float)
    strat, bh = strategy_and_bh_returns(sig, r, bps_per_side=bps)
    eq_s = np.cumprod(1.0 + strat)
    eq_b = np.cumprod(1.0 + bh)
    idx = returns.index

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(idx, eq_s, label=f"Rule: {rr.rule[:60]}", lw=1.2)
    ax.plot(idx, eq_b, label="Buy & Hold", lw=1.2, alpha=0.8)
    # Shade splits
    for name, color in (("train", "#d0e0ff"), ("validation", "#fff0d0"), ("holdout", "#e0ffe0")):
        start, end = cfg["splits"][name]
        m = split_mask(idx, start, end)
        if m.any():
            ax.axvspan(idx[m][0], idx[m][-1], alpha=0.15, color=color, label=name)
    ax.set_yscale("log")
    ax.set_title("Sweep v2 — top stable rule vs BH (log equity)")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def print_summary(
    df: pd.DataFrame,
    mc_targets: Sequence[SweepResult],
    meta: dict[str, Any],
    csv_path: Path,
    md_path: Path,
) -> None:
    print()
    print("=" * 72)
    print("SWEEP V2 SUMMARY")
    print("=" * 72)
    print(
        f"Runtime {meta['runtime_sec']:.1f}s | cores={meta['n_cores']} | "
        f"atoms {meta['n_atoms_total']}/{meta['n_atoms_searchable']} | "
        f"ranked {meta['n_ranked']} | stable {meta['n_stable']}"
    )
    print(f"Results: {md_path}")
    print(f"CSV:     {csv_path}")
    print()
    any_full = False
    any_bar = False
    for rr in mc_targets:
        flags = protocol_pass_v2(rr)
        if flags["passes_stable_bar"]:
            any_bar = True
        if flags["passes_with_holdout"]:
            any_full = True
    print(f"{'rule':<48} {'fit':>7} {'stab':>4} {'hold':>7} {'p':>7}  bar")
    print("-" * 90)
    for rr in mc_targets[:15]:
        flags = protocol_pass_v2(rr)
        hold = (
            f"{rr.holdout_excess_sharpe:+.3f}"
            if rr.holdout_excess_sharpe is not None
            else "nan"
        )
        p = f"{rr.null_pvalue:.4f}" if rr.null_pvalue is not None else "n/a"
        tag = (
            "YES+HO" if flags["passes_with_holdout"]
            else ("YES" if flags["passes_stable_bar"] else "no")
        )
        print(
            f"{rr.rule[:48]:<48} {rr.val_fitness:+7.3f} "
            f"{'Y' if rr.stable else 'N':>4} {hold:>7} {p:>7}  {tag}"
        )
    print()
    if any_full:
        print("Verdict: ≥1 stable survivor with val+halves+null p<0.05 AND holdout XS>0.")
    elif any_bar:
        print("Verdict: stable+null clears exist, but NONE with holdout XS>0.")
    else:
        print("Verdict: NO stable survivor clears val+halves+null p<0.05 with holdout XS>0.")
    print("=" * 72)


@click.command()
@click.option("--beam", type=int, default=DEFAULT_BEAM, show_default=True)
@click.option("--mc-n", type=int, default=DEFAULT_MC_N, show_default=True)
@click.option("--include-negation/--no-include-negation", default=True, show_default=True)
@click.option("--include-moon-or/--no-include-moon-or", default=True, show_default=True)
@click.option("--force-rebuild-atoms", is_flag=True, default=False)
@click.option("--config", "config_path", default=None)
@click.option("--out-dir", type=click.Path(), default=None)
@click.option("--plot/--no-plot", default=True, show_default=True)
def main(
    beam: int,
    mc_n: int,
    include_negation: bool,
    include_moon_or: bool,
    force_rebuild_atoms: bool,
    config_path: str | None,
    out_dir: str | None,
    plot: bool,
) -> None:
    """Sweep v2 beam search under the frozen protocol + regime filters."""
    try:
        cfg = load_config(config_path)
        out = Path(out_dir) if out_dir else (PROJECT_ROOT / "results")
        df, mc_targets, meta = run_sweep_v2(
            beam=beam,
            mc_n=mc_n,
            include_negation=include_negation,
            include_moon_or=include_moon_or,
            force_rebuild_atoms=force_rebuild_atoms,
            cfg=cfg,
            verbose=True,
            out_dir=out,
        )
        csv_path, md_path = write_sweep_results(df, mc_targets, meta, out_dir=out)

        if plot:
            stable = [r for r in mc_targets if r.stable]
            top = stable[0] if stable else (mc_targets[0] if mc_targets else None)
            if top is not None:
                try:
                    prices = load_prices(cfg=cfg, auto_fetch=False)
                    rets = daily_returns(prices)
                    atoms = load_atoms_v2(cfg=cfg, dates=prices.index, auto_build=False)
                    common = rets.index.intersection(atoms.index)
                    maybe_plot_top_stable(
                        top,
                        atoms.loc[common],
                        rets.loc[common],
                        cfg,
                        out / "sweep_v2_top_equity.png",
                        bps=float(cfg["costs"]["bps_per_side"]),
                    )
                    print(f"Chart: {out / 'sweep_v2_top_equity.png'}")
                except Exception as e:
                    print(f"(chart skipped: {e})")

        print_summary(df, mc_targets, meta, csv_path, md_path)
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
