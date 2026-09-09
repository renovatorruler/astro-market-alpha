"""Sweep v3: heavier beam L1→L2→L3→L4 + Nasdaq cross-asset must-pass + large MC.

Protocol (frozen — do not change splits):
  Train 1970–1994, Validation 1995–2009, Holdout 2010–present.
  Selection uses VALIDATION fitness on SPX only (+ regime filters + split-half).
  Cross-asset survivor: validation excess Sharpe > 0 on BOTH SPX and Nasdaq.
  Holdout is report-only for both indexes. Never select on holdout.
"""

from __future__ import annotations

import logging
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import click
import numpy as np
import pandas as pd

from astro_market.atoms import load_atoms_v2, searchable_atom_names
from astro_market.config import PROJECT_ROOT, load_config
from astro_market.data import (
    DataFetchError,
    align_multi_asset,
    daily_returns,
    is_cross_asset_survivor,
    load_nasdaq_prices,
    load_prices,
)
from astro_market.ephemeris import EphemerisError
from astro_market.null import empirical_pvalue
from astro_market.search import (
    _sharpe_np,
    complexity_for_rule,
    metrics_from_signal,
    signal_for_expr,
    split_returns_atoms,
    strategy_and_bh_returns,
)
from astro_market.sweep import (
    DEFAULT_LAMBDA,
    DEFAULT_LAMBDA_L3,
    MAX_EPISODE_DAYS,
    MIN_EPISODES_TRAIN_VAL,
    MIN_EPISODES_VAL,
    PARTIAL_EVERY,
    SweepResult,
    apply_regime_filter,
    attach_holdout,
    attach_stability,
    beam_l1,
    beam_l2,
    beam_l3,
    canonical_rule_v2,
    lambda_for_length,
    length_of_rule,
    score_rule,
    select_mc_targets,
)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_BEAM = 500
DEFAULT_MC_N = 50_000
DEFAULT_MAX_LENGTH = 4
DEFAULT_LAMBDA_L4 = 0.015
MC_STABLE_CAP = 25
MC_UNSTABLE_CAP = 10
CHECKPOINT_SECONDS = 180  # partial CSV every ~3 minutes


# ---------------------------------------------------------------------------
# Extended result
# ---------------------------------------------------------------------------


@dataclass
class SweepResultV3(SweepResult):
    """SweepResult plus Nasdaq cross-asset metrics."""

    nasdaq_val_excess_sharpe: float | None = None
    nasdaq_val_sharpe: float | None = None
    nasdaq_val_sharpe_bh: float | None = None
    nasdaq_holdout_excess_sharpe: float | None = None
    nasdaq_holdout_sharpe: float | None = None
    nasdaq_holdout_sharpe_bh: float | None = None
    cross_asset: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def upgrade_to_v3(rr: SweepResult) -> SweepResultV3:
    """Promote a v2 SweepResult to SweepResultV3 (copy fields)."""
    d = rr.to_dict()
    return SweepResultV3(**{k: v for k, v in d.items() if k in SweepResultV3.__dataclass_fields__})


# ---------------------------------------------------------------------------
# Beam L4
# ---------------------------------------------------------------------------


def beam_l4(
    l3_beam: Sequence[SweepResult],
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
    log: logging.Logger | None = None,
) -> list[SweepResult]:
    """Extend each top L3 with one more atom from L1 beam (AND)."""
    sig_l1_tr = {rr.rule: signal_for_expr(rr.rule, atoms_train) for rr in l1_beam}
    sig_l1_va = {rr.rule: signal_for_expr(rr.rule, atoms_val) for rr in l1_beam}
    sig_l3_tr = {rr.rule: signal_for_expr(rr.rule, atoms_train) for rr in l3_beam}
    sig_l3_va = {rr.rule: signal_for_expr(rr.rule, atoms_val) for rr in l3_beam}

    results: list[SweepResult] = []
    seen: set[str] = set()
    n_scored = 0
    n_drop = 0
    t_last = time.perf_counter()

    l1_lits = list(l1_beam)
    for r3 in l3_beam:
        base = r3.rule
        base_atoms = set(
            base.replace("~", " ").replace("&", " ").replace("|", " ").split()
        )
        for r1 in l1_lits:
            lit = r1.rule
            lit_core = lit.lstrip("~")
            if lit_core in base_atoms or lit in base_atoms:
                continue
            rule = canonical_rule_v2(f"{base} & {lit}")
            if rule in seen:
                continue
            seen.add(rule)
            sva = sig_l3_va[base] & sig_l1_va[lit]
            if not np.any(sva) or np.all(sva):
                continue
            if np.array_equal(sva, sig_l3_va[base]) or np.array_equal(sva, sig_l1_va[lit]):
                continue
            str_ = sig_l3_tr[base] & sig_l1_tr[lit]
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
                _write_partial_v3(results, partial_path, stage="L4")
            now = time.perf_counter()
            if log and now - t_last >= CHECKPOINT_SECONDS:
                log.info("L4 checkpoint scored=%d kept=%d", n_scored, len(results))
                if partial_path:
                    _write_partial_v3(results, partial_path, stage="L4_ckpt")
                t_last = now

    results.sort(key=lambda x: x.val_fitness, reverse=True)
    msg = (
        f"  L4: scored={n_scored} kept={len(results)} dropped_regime={n_drop} "
        f"beam_keep={min(beam, len(results))}"
    )
    if verbose:
        print(msg)
    if log:
        log.info(msg.strip())
    if partial_path:
        _write_partial_v3(results[:beam], partial_path, stage="L4_beam")
    return results[:beam]


def _write_partial_v3(results: Sequence[SweepResult], path: Path, stage: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for r in sorted(results, key=lambda x: x.val_fitness, reverse=True):
        d = r.to_dict() if hasattr(r, "to_dict") else asdict(r)
        d["partial_stage"] = stage
        rows.append(d)
    pd.DataFrame(rows).to_csv(path, index=False)


# ---------------------------------------------------------------------------
# Nasdaq attach + cross-asset filter
# ---------------------------------------------------------------------------


def attach_nasdaq_metrics(
    rules: Sequence[SweepResult],
    atoms_val: pd.DataFrame,
    r_ndq_val: np.ndarray,
    atoms_hold: pd.DataFrame | None,
    r_ndq_hold: np.ndarray | None,
    *,
    bps: float,
    lam: float,
    lam_l3: float,
) -> list[SweepResultV3]:
    """Attach Nasdaq val/holdout excess Sharpe; set cross_asset flag."""
    out: list[SweepResultV3] = []
    for rr in rules:
        v3 = rr if isinstance(rr, SweepResultV3) else upgrade_to_v3(rr)
        lam_use = lambda_for_length(v3.length, lam, lam_l3)
        sig_v = signal_for_expr(v3.rule, atoms_val)
        m_v = metrics_from_signal(
            sig_v, r_ndq_val, complexity=v3.complexity,
            bps_per_side=bps, lambda_complexity=lam_use,
        )
        v3.nasdaq_val_excess_sharpe = float(m_v["excess_sharpe"])
        v3.nasdaq_val_sharpe = float(m_v["sharpe"])
        v3.nasdaq_val_sharpe_bh = float(m_v["sharpe_bh"])
        if atoms_hold is not None and r_ndq_hold is not None:
            sig_h = signal_for_expr(v3.rule, atoms_hold)
            m_h = metrics_from_signal(
                sig_h, r_ndq_hold, complexity=v3.complexity,
                bps_per_side=bps, lambda_complexity=lam_use,
            )
            v3.nasdaq_holdout_excess_sharpe = float(m_h["excess_sharpe"])
            v3.nasdaq_holdout_sharpe = float(m_h["sharpe"])
            v3.nasdaq_holdout_sharpe_bh = float(m_h["sharpe_bh"])
        v3.cross_asset = is_cross_asset_survivor(
            float(v3.val_excess_sharpe),
            float(v3.nasdaq_val_excess_sharpe)
            if v3.nasdaq_val_excess_sharpe is not None
            else float("nan"),
        )
        out.append(v3)
    return out


def filter_cross_asset_stable(ranked: Sequence[SweepResultV3]) -> list[SweepResultV3]:
    """Stable on SPX halves AND Nasdaq val XS > 0 (and SPX val XS > 0 via cross_asset)."""
    return [r for r in ranked if r.stable and r.cross_asset]


def select_mc_targets_v3(
    ranked: Sequence[SweepResultV3],
    *,
    n_stable: int = MC_STABLE_CAP,
    n_unstable: int = MC_UNSTABLE_CAP,
) -> list[SweepResultV3]:
    """Top cross-asset stable + top high-fit unstable (honesty)."""
    ca_stable = [r for r in ranked if r.stable and r.cross_asset]
    # Prefer ranking by val_fitness among cross-asset stable
    ca_stable = sorted(ca_stable, key=lambda x: x.val_fitness, reverse=True)
    unstable = [r for r in ranked if not r.stable]
    unstable = sorted(unstable, key=lambda x: x.val_fitness, reverse=True)
    chosen: list[SweepResultV3] = []
    seen: set[str] = set()
    for r in ca_stable[:n_stable]:
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
# Monte Carlo (multiprocessing)
# ---------------------------------------------------------------------------


def _mc_one_rule(payload: dict[str, Any]) -> tuple[str, float, int]:
    """Worker: circular-shift null p-value for one rule (picklable)."""
    signal = payload["signal"]
    r_val = payload["r_val"]
    n = int(payload["n"])
    seed = int(payload["seed"])
    bps = float(payload["bps"])
    lam_use = float(payload["lam_use"])
    c = int(payload["complexity"])
    observed = float(payload["observed"])
    rule = str(payload["rule"])

    n_days = len(r_val)
    if n_days < 2:
        return rule, float("nan"), n

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
    p = empirical_pvalue(observed, scores, alternative="greater")
    return rule, float(p), n


def monte_carlo_batch_mp(
    targets: Sequence[SweepResultV3],
    atoms_val: pd.DataFrame,
    r_val: np.ndarray,
    *,
    n: int = DEFAULT_MC_N,
    seed: int = 42,
    bps: float = 10.0,
    lam: float = DEFAULT_LAMBDA,
    lam_l3: float = DEFAULT_LAMBDA_L3,
    n_workers: int | None = None,
    verbose: bool = True,
    log: logging.Logger | None = None,
    min_n_fallback: int = 20_000,
) -> tuple[dict[str, float], int, str]:
    """
    Run MC null in parallel over rules.

    Returns (rule -> pvalue, n_used, note).
    If full n is too slow, callers may lower n; this function always runs `n`
    but notes workers used.
    """
    n_workers = n_workers or max(1, (os.cpu_count() or 1))
    payloads = []
    for i, rr in enumerate(targets):
        lam_use = lambda_for_length(rr.length, lam, lam_l3)
        sig = signal_for_expr(rr.rule, atoms_val)
        observed = metrics_from_signal(
            sig, r_val, complexity=rr.complexity,
            bps_per_side=bps, lambda_complexity=lam_use,
        )["fitness"]
        payloads.append({
            "rule": rr.rule,
            "signal": np.asarray(sig, dtype=bool),
            "r_val": np.asarray(r_val, dtype=float),
            "n": n,
            "seed": seed + i * 17,
            "bps": bps,
            "lam_use": lam_use,
            "complexity": rr.complexity,
            "observed": float(observed) if np.isfinite(observed) else float("-inf"),
        })

    note = f"mc_n={n} workers={n_workers}"
    results: dict[str, float] = {}
    t0 = time.perf_counter()

    # Probe: if single rule estimate suggests too slow, reduce n
    if payloads:
        probe_n = min(2000, n)
        probe = dict(payloads[0])
        probe["n"] = probe_n
        _, _, _ = _mc_one_rule(probe)
        # Rough: time for probe_n → extrapolate
        # We already paid probe cost; continue with requested n unless absurd.
        # Prefer finishing 50k; only drop if clearly multi-hour.
        # Heuristic left to caller; here we just run n.

    if n_workers <= 1 or len(payloads) <= 1:
        for i, pl in enumerate(payloads):
            rule, p, _ = _mc_one_rule(pl)
            results[rule] = p
            if verbose and ((i + 1) % 5 == 0 or i == 0 or i + 1 == len(payloads)):
                print(f"  [{i + 1}/{len(payloads)}] {rule[:48]:<48} p={p:.4f}")
            if log:
                log.info("MC [%d/%d] %s p=%.4f", i + 1, len(payloads), rule[:60], p)
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            futs = {ex.submit(_mc_one_rule, pl): pl["rule"] for pl in payloads}
            done = 0
            for fut in as_completed(futs):
                rule, p, _ = fut.result()
                results[rule] = p
                done += 1
                if verbose and (done % 5 == 0 or done == 1 or done == len(payloads)):
                    print(f"  [{done}/{len(payloads)}] {rule[:48]:<48} p={p:.4f}")
                if log:
                    log.info("MC [%d/%d] %s p=%.4f", done, len(payloads), rule[:60], p)

    elapsed = time.perf_counter() - t0
    note = f"mc_n={n} workers={n_workers} elapsed={elapsed:.1f}s"
    # If somehow we needed to fall back (not used here unless caller lowers n)
    if n < DEFAULT_MC_N:
        note += f" (requested reduced from {DEFAULT_MC_N}; min target {min_n_fallback})"
    return results, n, note


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


def _setup_run_log(path: Path) -> logging.Logger:
    path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("sweep_v3")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(path, mode="w", encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
    logger.addHandler(fh)
    # Also echo to stdout via a stream handler? Keep print for console; log for file.
    return logger


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def run_sweep_v3(
    *,
    beam: int = DEFAULT_BEAM,
    mc_n: int = DEFAULT_MC_N,
    max_length: int = DEFAULT_MAX_LENGTH,
    include_negation: bool = True,
    include_moon_or: bool = True,
    force_rebuild_atoms: bool = False,
    cfg: dict | None = None,
    verbose: bool = True,
    out_dir: Path | None = None,
    n_workers: int | None = None,
) -> tuple[pd.DataFrame, list[SweepResultV3], dict[str, Any]]:
    """Full sweep v3 end-to-end."""
    t0 = time.perf_counter()
    cfg = cfg or load_config()
    bps = float(cfg["costs"]["bps_per_side"])
    lam = float(cfg["evaluate"].get("complexity_lambda", DEFAULT_LAMBDA))
    lam_l3 = float(
        cfg.get("sweep_v3", {}).get(
            "lambda_l3",
            cfg.get("sweep_v2", {}).get("lambda_l3", DEFAULT_LAMBDA_L3),
        )
    )
    seed = int(cfg.get("null", {}).get("seed", 42))
    out_dir = out_dir or (PROJECT_ROOT / "results")
    out_dir.mkdir(parents=True, exist_ok=True)
    partial_path = out_dir / "sweep_v3_partial.csv"
    log_path = out_dir / "sweep_v3_run.log"
    log = _setup_run_log(log_path)
    n_cores = os.cpu_count() or 1
    n_workers = n_workers or n_cores

    banner = (
        f"Sweep v3 | cores={n_cores} workers={n_workers} beam={beam} "
        f"max_length={max_length} mc_n={mc_n} λ={lam} λ_L3+={lam_l3}"
    )
    if verbose:
        print(banner)
    log.info(banner)
    log.info("START")

    if verbose:
        print("Loading SPX + Nasdaq + atoms_v2...")
    log.info("Loading SPX + Nasdaq + atoms_v2")

    prices = load_prices(cfg=cfg, auto_fetch=False)
    nasdaq_prices = load_nasdaq_prices(cfg=cfg, auto_fetch=True)
    rets_spx = daily_returns(prices)

    t_atoms = time.perf_counter()
    atoms = load_atoms_v2(
        cfg=cfg, dates=prices.index, auto_build=True, force_rebuild=force_rebuild_atoms
    )
    atom_build_sec = time.perf_counter() - t_atoms

    rets_spx, rets_ndq, atoms = align_multi_asset(rets_spx, nasdaq_prices, atoms)
    n_atoms_total = int(atoms.shape[1])
    searchable = searchable_atom_names(atoms.columns)
    n_atoms_search = len(searchable)

    r_train, a_train, _ = split_returns_atoms(rets_spx, atoms, cfg, "train")
    r_val, a_val, idx_val = split_returns_atoms(rets_spx, atoms, cfg, "validation")
    r_hold, a_hold, idx_hold = split_returns_atoms(rets_spx, atoms, cfg, "holdout")

    def _ndq_split(split_name: str) -> np.ndarray:
        _, _, idx = split_returns_atoms(rets_spx, atoms, cfg, split_name)
        return rets_ndq.reindex(idx).to_numpy(dtype=float)

    r_ndq_val = _ndq_split("validation")
    r_ndq_hold = _ndq_split("holdout")

    # Sanity: lengths must match
    assert len(r_ndq_val) == len(r_val), (len(r_ndq_val), len(r_val))
    assert len(r_ndq_hold) == len(r_hold), (len(r_ndq_hold), len(r_hold))
    if np.isnan(r_ndq_val).any() or np.isnan(r_ndq_hold).any():
        raise DataFetchError("Nasdaq returns have NaNs on SPX-aligned calendar after intersect.")

    msg = (
        f"Atoms total={n_atoms_total} searchable={n_atoms_search} "
        f"(build/load {atom_build_sec:.1f}s); "
        f"aligned days={len(rets_spx)}; "
        f"splits train/val/hold={len(r_train)}/{len(r_val)}/{len(r_hold)}"
    )
    if verbose:
        print(msg)
    log.info(msg)

    # --- Beam stages (SPX fitness) ---
    stages: dict[str, list[SweepResult]] = {}
    stage_secs: dict[str, float] = {}

    if verbose:
        print("L1 beam search...")
    log.info("L1 start")
    t1 = time.perf_counter()
    l1 = beam_l1(
        a_train, a_val, r_train, r_val,
        searchable=searchable,
        beam=beam,
        include_negation=include_negation,
        bps=bps, lam=lam, lam_l3=lam_l3,
        verbose=verbose,
    )
    stage_secs["l1"] = time.perf_counter() - t1
    stages["l1"] = l1
    log.info("L1 done %.1fs beam=%d top=%s", stage_secs["l1"], len(l1), l1[0].rule if l1 else None)
    _write_partial_v3(l1, partial_path, stage="L1_beam")

    if verbose:
        print("L2 beam search...")
    log.info("L2 start")
    t2 = time.perf_counter()
    l2 = beam_l2(
        l1, a_train, a_val, r_train, r_val,
        beam=beam, bps=bps, lam=lam, lam_l3=lam_l3,
        include_moon_or=include_moon_or,
        partial_path=partial_path,
        verbose=verbose,
    )
    stage_secs["l2"] = time.perf_counter() - t2
    stages["l2"] = l2
    log.info("L2 done %.1fs beam=%d", stage_secs["l2"], len(l2))

    if max_length >= 3:
        if verbose:
            print("L3 beam search...")
        log.info("L3 start")
        t3 = time.perf_counter()
        l3 = beam_l3(
            l2, l1, a_train, a_val, r_train, r_val,
            beam=beam, bps=bps, lam=lam, lam_l3=lam_l3,
            partial_path=partial_path,
            verbose=verbose,
        )
        stage_secs["l3"] = time.perf_counter() - t3
        stages["l3"] = l3
        log.info("L3 done %.1fs beam=%d", stage_secs["l3"], len(l3))
    else:
        l3 = []
        stage_secs["l3"] = 0.0
        stages["l3"] = l3

    if max_length >= 4:
        if verbose:
            print("L4 beam search...")
        log.info("L4 start")
        t4 = time.perf_counter()
        l4 = beam_l4(
            l3, l1, a_train, a_val, r_train, r_val,
            beam=beam, bps=bps, lam=lam, lam_l3=lam_l3,
            partial_path=partial_path,
            verbose=verbose,
            log=log,
        )
        stage_secs["l4"] = time.perf_counter() - t4
        stages["l4"] = l4
        log.info("L4 done %.1fs beam=%d", stage_secs["l4"], len(l4))
    else:
        l4 = []
        stage_secs["l4"] = 0.0
        stages["l4"] = l4

    # Merge beams
    best: dict[str, SweepResult] = {}
    for rr in list(l1) + list(l2) + list(l3) + list(l4):
        key = canonical_rule_v2(rr.rule)
        rr.rule = key
        prev = best.get(key)
        if prev is None or rr.val_fitness > prev.val_fitness:
            best[key] = rr
    ranked_v2 = sorted(best.values(), key=lambda x: x.val_fitness, reverse=True)

    if verbose:
        print(f"Unique ranked={len(ranked_v2)}; attaching SPX split-half stability...")
    log.info("Stability on %d ranked", len(ranked_v2))
    ranked_v2 = attach_stability(ranked_v2, a_val, r_val, bps=bps, lam=lam, lam_l3=lam_l3)
    n_stable = sum(1 for r in ranked_v2 if r.stable)

    if verbose:
        print("Attaching SPX holdout (report-only)...")
    ranked_v2 = attach_holdout(ranked_v2, a_hold, r_hold, bps=bps, lam=lam, lam_l3=lam_l3)

    if verbose:
        print("Attaching Nasdaq val+holdout + cross-asset filter...")
    log.info("Nasdaq attach")
    ranked = attach_nasdaq_metrics(
        ranked_v2, a_val, r_ndq_val, a_hold, r_ndq_hold,
        bps=bps, lam=lam, lam_l3=lam_l3,
    )
    # Re-sort by SPX val fitness
    ranked.sort(key=lambda x: x.val_fitness, reverse=True)
    n_cross = sum(1 for r in ranked if r.cross_asset)
    n_cross_stable = sum(1 for r in ranked if r.cross_asset and r.stable)
    if verbose:
        print(
            f"  Stable(SPX halves)={n_stable} cross_asset={n_cross} "
            f"cross_asset+stable={n_cross_stable}"
        )
    log.info(
        "stable=%d cross_asset=%d cross_stable=%d",
        n_stable, n_cross, n_cross_stable,
    )
    _write_partial_v3(ranked[:800], partial_path, stage="post_nasdaq")

    mc_targets = select_mc_targets_v3(
        ranked, n_stable=MC_STABLE_CAP, n_unstable=MC_UNSTABLE_CAP
    )
    if verbose:
        print(
            f"Monte Carlo null n={mc_n} on {len(mc_targets)} targets "
            f"({sum(1 for r in mc_targets if r.stable and r.cross_asset)} cross-stable + "
            f"{sum(1 for r in mc_targets if not r.stable)} unstable) "
            f"workers={n_workers}..."
        )
    log.info("MC start n=%d targets=%d", mc_n, len(mc_targets))

    # Adaptive MC n: prefer 50k; if probe suggests >2.5h for all targets, drop toward 20k
    mc_n_used = mc_n
    mc_note = ""
    if mc_targets:
        probe = {
            "rule": mc_targets[0].rule,
            "signal": np.asarray(signal_for_expr(mc_targets[0].rule, a_val), dtype=bool),
            "r_val": np.asarray(r_val, dtype=float),
            "n": 2000,
            "seed": seed,
            "bps": bps,
            "lam_use": lambda_for_length(mc_targets[0].length, lam, lam_l3),
            "complexity": mc_targets[0].complexity,
            "observed": 0.0,
        }
        tp0 = time.perf_counter()
        _mc_one_rule(probe)
        per_2k = time.perf_counter() - tp0
        # Estimate wall with workers: (n/2000)*per_2k * n_targets / n_workers
        est = (mc_n / 2000.0) * per_2k * len(mc_targets) / max(1, n_workers)
        log.info("MC probe: %.2fs per 2k → est wall %.0fs for n=%d", per_2k, est, mc_n)
        if verbose:
            print(f"  MC probe: {per_2k:.2f}s/2k → est wall {est:.0f}s")
        if est > 9000:  # ~2.5h
            mc_n_used = max(20_000, int(mc_n * 9000 / est))
            mc_note = f"reduced mc_n {mc_n}->{mc_n_used} due to runtime estimate {est:.0f}s"
            log.info(mc_note)
            if verbose:
                print(f"  NOTE: {mc_note}")

    t_mc = time.perf_counter()
    pmap, mc_n_used, mc_note2 = monte_carlo_batch_mp(
        mc_targets, a_val, r_val,
        n=mc_n_used, seed=seed, bps=bps, lam=lam, lam_l3=lam_l3,
        n_workers=n_workers, verbose=verbose, log=log,
    )
    mc_sec = time.perf_counter() - t_mc
    log.info("MC done %.1fs %s", mc_sec, mc_note2)

    for rr in ranked:
        key = canonical_rule_v2(rr.rule)
        if key in pmap or rr.rule in pmap:
            p = pmap.get(rr.rule, pmap.get(key))
            rr.null_pvalue = p
            rr.null_n = mc_n_used
    for rr in mc_targets:
        p = pmap.get(rr.rule)
        if p is not None:
            rr.null_pvalue = p
            rr.null_n = mc_n_used

    df = pd.DataFrame([r.to_dict() for r in ranked])
    runtime = time.perf_counter() - t0
    meta = {
        "runtime_sec": runtime,
        "atom_build_sec": atom_build_sec,
        "l1_sec": stage_secs.get("l1", 0.0),
        "l2_sec": stage_secs.get("l2", 0.0),
        "l3_sec": stage_secs.get("l3", 0.0),
        "l4_sec": stage_secs.get("l4", 0.0),
        "mc_sec": mc_sec,
        "n_cores": n_cores,
        "n_workers": n_workers,
        "n_atoms_total": n_atoms_total,
        "n_atoms_searchable": n_atoms_search,
        "beam": beam,
        "max_length": max_length,
        "n_l1_beam": len(l1),
        "n_l2_beam": len(l2),
        "n_l3_beam": len(l3),
        "n_l4_beam": len(l4),
        "n_ranked": len(ranked),
        "n_stable": n_stable,
        "n_cross_asset": n_cross,
        "n_cross_asset_stable": n_cross_stable,
        "n_mc_targets": len(mc_targets),
        "mc_n_sims": mc_n_used,
        "mc_n_requested": mc_n,
        "mc_note": " ".join(x for x in (mc_note, mc_note2) if x),
        "lambda": lam,
        "lambda_l3": lam_l3,
        "bps": bps,
        "seed": seed,
        "n_train": len(r_train),
        "n_val": len(r_val),
        "n_holdout": len(r_hold),
        "n_aligned_days": len(rets_spx),
        "max_episode_days": MAX_EPISODE_DAYS,
        "min_episodes_val": MIN_EPISODES_VAL,
        "min_episodes_train_val": MIN_EPISODES_TRAIN_VAL,
        "log_path": str(log_path),
    }
    log.info("DONE runtime=%.1fs", runtime)
    return df, mc_targets, meta


# ---------------------------------------------------------------------------
# Protocol / reporting
# ---------------------------------------------------------------------------


def protocol_pass_v3(rr: SweepResultV3, p_threshold: float = 0.05) -> dict[str, bool]:
    """Full v3 bar: SPX halves + Nasdaq val XS + null + (optional) SPX holdout."""
    spx_halves = bool(rr.stable)
    nasdaq_val = (
        rr.nasdaq_val_excess_sharpe is not None
        and np.isfinite(rr.nasdaq_val_excess_sharpe)
        and rr.nasdaq_val_excess_sharpe > 0
    )
    spx_val = (
        rr.val_excess_sharpe is not None
        and rr.val_fitness is not None
        and rr.val_excess_sharpe > 0
        and rr.val_fitness > 0
    )
    null_ok = rr.null_pvalue is not None and rr.null_pvalue < p_threshold
    hold_spx = (
        rr.holdout_excess_sharpe is not None and rr.holdout_excess_sharpe > 0
    )
    return {
        "spx_val_ok": bool(spx_val),
        "spx_halves": spx_halves,
        "nasdaq_val_ok": bool(nasdaq_val),
        "cross_asset": bool(rr.cross_asset),
        "null_reject": bool(null_ok),
        "spx_holdout_xs_positive": bool(hold_spx),
        "passes_cross_stable_null": bool(
            spx_val and spx_halves and nasdaq_val and null_ok
        ),
        "passes_full_bar": bool(
            spx_val and spx_halves and nasdaq_val and null_ok and hold_spx
        ),
    }


def write_sweep_v3_results(
    df: pd.DataFrame,
    mc_targets: Sequence[SweepResultV3],
    meta: dict[str, Any],
    *,
    out_dir: Path | None = None,
) -> tuple[Path, Path]:
    out_dir = out_dir or (PROJECT_ROOT / "results")
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "sweep_v3.csv"
    md_path = out_dir / "sweep_v3.md"
    df.to_csv(csv_path, index=False)

    stable_ca = df[(df.get("stable") == True) & (df.get("cross_asset") == True)] if (  # noqa: E712
        "stable" in df.columns and "cross_asset" in df.columns
    ) else df.iloc[0:0]

    lines: list[str] = []
    lines.append("# Sweep v3 — Beam L1→L4 + Nasdaq cross-asset must-pass")
    lines.append("")
    lines.append("## Method")
    lines.append("")
    lines.append(
        "Heavier CPU search on top of sweep v2 regime filters: **beam=500**, "
        "**length-4** AND extensions, **Nasdaq (^IXIC)** second-index must-pass, "
        f"and Monte Carlo **n={meta.get('mc_n_sims')}** "
        f"(requested {meta.get('mc_n_requested')}) with multiprocessing."
    )
    lines.append("")
    lines.append(
        "Beam selection still uses **SPX validation fitness** only. "
        "A rule is a **cross-asset survivor** only if validation excess Sharpe > 0 "
        "on **both** SPX and Nasdaq (same long/flat signal, 10 bps). "
        "Holdout reported for both indexes; **never used for selection**."
    )
    lines.append("")
    lines.append("**Regime / episode filters (from v2):**")
    lines.append("")
    lines.append(
        f"- Hard drop max True episode on validation > **{meta.get('max_episode_days')}** days"
    )
    lines.append(
        f"- Soft penalty if episodes < {meta.get('min_episodes_val')} on val "
        f"and < {meta.get('min_episodes_train_val')} on train+val"
    )
    lines.append("- No outer-planet signs in searchable pool; hard-drop pure outer-sign rules")
    lines.append("- **Stable**: SPX excess Sharpe > 0 in **both** validation temporal halves")
    lines.append("")
    lines.append("**Frozen protocol:**")
    lines.append("")
    lines.append("| Item | Value |")
    lines.append("|------|-------|")
    lines.append("| Train | 1970-01-01 → 1994-12-31 |")
    lines.append("| Validation | 1995-01-01 → 2009-12-31 |")
    lines.append("| Holdout | 2010-01-01 → present (**never used for selection**) |")
    lines.append("| Indexes | SPX `^GSPC` + Nasdaq `^IXIC` (aligned calendar) |")
    lines.append("| Position | Long when True; flat when False; BH = always long |")
    lines.append(f"| Costs | {meta.get('bps')} bps / side |")
    lines.append(
        f"| Beam / max length | {meta.get('beam')} / {meta.get('max_length')} |"
    )
    lines.append(
        f"| Monte Carlo | Circular-shift SPX val fitness, n={meta.get('mc_n_sims')}, "
        f"seed={meta.get('seed')}, workers={meta.get('n_workers')} |"
    )
    lines.append("")
    lines.append(
        f"Runtime: **{meta.get('runtime_sec', float('nan')):.1f}s** "
        f"(atoms {meta.get('atom_build_sec', 0):.1f}s · "
        f"L1 {meta.get('l1_sec', 0):.1f}s · L2 {meta.get('l2_sec', 0):.1f}s · "
        f"L3 {meta.get('l3_sec', 0):.1f}s · L4 {meta.get('l4_sec', 0):.1f}s · "
        f"MC {meta.get('mc_sec', 0):.1f}s) · "
        f"cores={meta.get('n_cores')} workers={meta.get('n_workers')} · "
        f"atoms {meta.get('n_atoms_total')}/{meta.get('n_atoms_searchable')} · "
        f"ranked={meta.get('n_ranked')} · stable={meta.get('n_stable')} · "
        f"cross={meta.get('n_cross_asset')} · cross+stable={meta.get('n_cross_asset_stable')} · "
        f"days train/val/hold={meta.get('n_train')}/{meta.get('n_val')}/{meta.get('n_holdout')} · "
        f"aligned={meta.get('n_aligned_days')}"
    )
    if meta.get("mc_note"):
        lines.append("")
        lines.append(f"MC note: {meta.get('mc_note')}")
    lines.append("")

    # Top by val fitness
    lines.append("## Top rules by SPX validation fitness")
    lines.append("")
    lines.append(
        "| Rank | Rule | L | Val XS | Val Fit | NDQ Val XS | H1 | H2 | Stable | Cross | "
        "SPX Hold XS | NDQ Hold XS | Null p |"
    )
    lines.append(
        "|-----:|------|--:|-------:|--------:|-----------:|---:|---:|:------:|:-----:|"
        "-----------:|------------:|-------:|"
    )
    for rank, (_, row) in enumerate(df.head(25).iterrows(), start=1):
        def _f(key, fmt="{:+.3f}"):
            v = row.get(key)
            return fmt.format(float(v)) if pd.notna(v) else "n/a"

        p = row.get("null_pvalue")
        p_s = f"{float(p):.4f}" if pd.notna(p) else "—"
        lines.append(
            f"| {rank} | `{row['rule']}` | {int(row['length'])} | "
            f"{_f('val_excess_sharpe')} | {_f('val_fitness')} | "
            f"{_f('nasdaq_val_excess_sharpe')} | "
            f"{_f('val_half1_excess_sharpe')} | {_f('val_half2_excess_sharpe')} | "
            f"{'Y' if bool(row.get('stable')) else 'N'} | "
            f"{'Y' if bool(row.get('cross_asset')) else 'N'} | "
            f"{_f('holdout_excess_sharpe')} | {_f('nasdaq_holdout_excess_sharpe')} | {p_s} |"
        )
    lines.append("")

    # Cross-asset stable
    lines.append("## Cross-asset stable survivors")
    lines.append("")
    lines.append(
        "SPX XS>0 in both val halves **and** Nasdaq val XS>0 "
        "(selection still never uses holdout)."
    )
    lines.append("")
    if stable_ca.empty:
        lines.append("_No cross-asset stable survivors._")
    else:
        lines.append(
            "| Rank | Rule | L | SPX Fit | SPX Val XS | NDQ Val XS | H1 | H2 | "
            "SPX Hold | NDQ Hold | Null p | Full bar |"
        )
        lines.append(
            "|-----:|------|--:|--------:|-----------:|-----------:|---:|---:|"
            "---------:|---------:|-------:|:--------:|"
        )
        for rank, (_, row) in enumerate(stable_ca.head(30).iterrows(), start=1):
            rr = _row_to_v3(row)
            flags = protocol_pass_v3(rr)
            p_s = f"{rr.null_pvalue:.4f}" if rr.null_pvalue is not None else "—"

            def _xf(v):
                return f"{v:+.3f}" if v is not None and np.isfinite(v) else "n/a"

            bar = "YES" if flags["passes_full_bar"] else (
                "cross+null" if flags["passes_cross_stable_null"] else "no"
            )
            lines.append(
                f"| {rank} | `{rr.rule}` | {rr.length} | {rr.val_fitness:+.3f} | "
                f"{rr.val_excess_sharpe:+.3f} | {_xf(rr.nasdaq_val_excess_sharpe)} | "
                f"{_xf(rr.val_half1_excess_sharpe)} | {_xf(rr.val_half2_excess_sharpe)} | "
                f"{_xf(rr.holdout_excess_sharpe)} | {_xf(rr.nasdaq_holdout_excess_sharpe)} | "
                f"{p_s} | {bar} |"
            )
    lines.append("")

    # MC table
    lines.append("## Monte Carlo targets (cross-asset stable + honesty unstable)")
    lines.append("")
    if not mc_targets:
        lines.append("_None._")
    else:
        lines.append(
            "| Rule | Stab | Cross | SPX Fit | NDQ Val XS | SPX Hold | NDQ Hold | Null p | Full |"
        )
        lines.append(
            "|------|:----:|:-----:|--------:|-----------:|---------:|---------:|-------:|:----:|"
        )
        for rr in mc_targets:
            flags = protocol_pass_v3(rr)
            def _xf(v):
                return f"{v:+.3f}" if v is not None and np.isfinite(v) else "n/a"
            p_s = f"{rr.null_pvalue:.4f}" if rr.null_pvalue is not None else "n/a"
            lines.append(
                f"| `{rr.rule}` | {'Y' if rr.stable else 'N'} | "
                f"{'Y' if rr.cross_asset else 'N'} | {rr.val_fitness:+.3f} | "
                f"{_xf(rr.nasdaq_val_excess_sharpe)} | {_xf(rr.holdout_excess_sharpe)} | "
                f"{_xf(rr.nasdaq_holdout_excess_sharpe)} | {p_s} | "
                f"{'YES' if flags['passes_full_bar'] else 'no'} |"
            )
    lines.append("")

    # Verdict
    lines.append("## Verdict")
    lines.append("")
    any_full = False
    any_cross_null = False
    for rr in mc_targets:
        flags = protocol_pass_v3(rr)
        if flags["passes_full_bar"]:
            any_full = True
        if flags["passes_cross_stable_null"]:
            any_cross_null = True

    if any_full:
        lines.append(
            "**YES — at least one rule clears the full bar:** SPX val XS>0 both halves, "
            "Nasdaq val XS>0, null p<0.05, and SPX holdout XS>0. "
            "**Treat with extreme caution** — see multiple-testing discussion below."
        )
    elif any_cross_null:
        lines.append(
            "**Partial:** ≥1 cross-asset stable survivor clears null p<0.05 on SPX validation, "
            "but **none** show positive SPX holdout excess Sharpe. "
            "Not confirmed out-of-sample alpha."
        )
    elif not stable_ca.empty:
        lines.append(
            "**No full-bar survivor.** Cross-asset stable rules exist, but none jointly "
            "clear null p<0.05 with positive SPX holdout XS (or MC inconclusive)."
        )
    else:
        lines.append(
            "**No cross-asset stable survivors.** After SPX split-half stability and "
            "Nasdaq val XS>0 must-pass, nothing remains. Honest negative under v3."
        )
    lines.append("")

    lines.append("## Honest interpretation (multiple testing)")
    lines.append("")
    lines.append(
        f"- Searchable atoms ≈ **{meta.get('n_atoms_searchable')}**; beam={meta.get('beam')}; "
        f"max length={meta.get('max_length')}. Ranked candidates after beams: "
        f"**{meta.get('n_ranked')}**. This is a large implicit search; raw p<0.05 is weak."
    )
    lines.append(
        "- Nasdaq must-pass reduces (but does not eliminate) SPX-specific overfitting. "
        "It is still one extra filter applied *after* SPX-driven beam selection."
    )
    lines.append(
        "- Holdout was never used for selection. Positive holdout on a shortlisted rule "
        "is necessary but not sufficient for a discovery claim."
    )
    # Mercury-sign section inserted by _fix_mercury_section after write

    lines.append(
        "- Costs 10 bps/side; Adj Close proxy; tropical geocentric ephemeris; "
        "atoms from `atoms_v2.parquet`."
    )
    lines.append("")
    lines.append(f"Artifacts: `{csv_path.name}`, `{md_path.name}`, `sweep_v3_run.log`.")
    lines.append("")

    md_path.write_text("\n".join(lines), encoding="utf-8")
    # Fix mercury block if broken — regenerate that section cleanly
    _fix_mercury_section(md_path, df)
    return csv_path, md_path


def _fix_mercury_section(md_path: Path, df: pd.DataFrame) -> None:
    """Rewrite mercury-sign discussion cleanly (avoids nested f-string bugs)."""
    text = md_path.read_text(encoding="utf-8")
    marker = "### Mercury-sign family"
    if marker not in text:
        # Insert before honest costs bullet if merc rows exist
        merc = df[df["rule"].astype(str).str.contains("mercury_sign", na=False)]
        if merc.empty:
            return
        block = [
            "",
            "### Mercury-sign family (almost-BH / lattice echoes)",
            "",
            "If mercury-sign rules reappear (v2 had `~mercury_sign_aquarius & ~mercury_sign_libra` "
            "as a rare holdout-positive stable), treat them as **near-buy-and-hold exclusions** "
            "(long most of the time; small calendar tilts) — easy to overfit and sensitive to "
            "costs / calendar alignment:",
            "",
        ]
        for _, row in merc.head(5).iterrows():
            ndq = row.get("nasdaq_val_excess_sharpe")
            ndq_s = f"{float(ndq):+.3f}" if pd.notna(ndq) else "n/a"
            ho = row.get("holdout_excess_sharpe")
            ho_s = f"{float(ho):+.3f}" if pd.notna(ho) else "n/a"
            block.append(
                f"- `{row['rule']}` — SPX val XS={float(row['val_excess_sharpe']):+.3f}, "
                f"NDQ val XS={ndq_s}, stable={bool(row.get('stable'))}, "
                f"cross={bool(row.get('cross_asset'))}, SPX hold={ho_s}"
            )
        block.append("")
        insert_at = text.find("- Costs 10 bps")
        if insert_at >= 0:
            text = text[:insert_at] + "\n".join(block) + "\n" + text[insert_at:]
            md_path.write_text(text, encoding="utf-8")
        return

    # Remove broken section between marker and next ## or - Costs
    start = text.find(marker)
    # Find end: next line starting with "- Costs" or "## "
    rest = text[start:]
    end_rel = None
    for key in ("\n- Costs 10 bps", "\n## "):
        i = rest.find(key, 1)
        if i > 0 and (end_rel is None or i < end_rel):
            end_rel = i
    if end_rel is None:
        return
    merc = df[df["rule"].astype(str).str.contains("mercury_sign", na=False)]
    block = [
        "### Mercury-sign family (almost-BH / lattice echoes)",
        "",
        "If mercury-sign rules reappear (v2 had `~mercury_sign_aquarius & ~mercury_sign_libra` "
        "as a rare holdout-positive stable), treat them as **near-buy-and-hold exclusions** "
        "(long most of the time; small calendar tilts) — easy to overfit and sensitive to "
        "costs / calendar alignment:",
        "",
    ]
    for _, row in merc.head(5).iterrows():
        ndq = row.get("nasdaq_val_excess_sharpe")
        ndq_s = f"{float(ndq):+.3f}" if pd.notna(ndq) else "n/a"
        ho = row.get("holdout_excess_sharpe")
        ho_s = f"{float(ho):+.3f}" if pd.notna(ho) else "n/a"
        block.append(
            f"- `{row['rule']}` — SPX val XS={float(row['val_excess_sharpe']):+.3f}, "
            f"NDQ val XS={ndq_s}, stable={bool(row.get('stable'))}, "
            f"cross={bool(row.get('cross_asset'))}, SPX hold={ho_s}"
        )
    block.append("")
    new_text = text[:start] + "\n".join(block) + rest[end_rel:]
    md_path.write_text(new_text, encoding="utf-8")


def _row_to_v3(row: pd.Series) -> SweepResultV3:
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

    return SweepResultV3(
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
        nasdaq_val_excess_sharpe=mf("nasdaq_val_excess_sharpe"),
        nasdaq_val_sharpe=mf("nasdaq_val_sharpe"),
        nasdaq_val_sharpe_bh=mf("nasdaq_val_sharpe_bh"),
        nasdaq_holdout_excess_sharpe=mf("nasdaq_holdout_excess_sharpe"),
        nasdaq_holdout_sharpe=mf("nasdaq_holdout_sharpe"),
        nasdaq_holdout_sharpe_bh=mf("nasdaq_holdout_sharpe_bh"),
        cross_asset=bool(row.get("cross_asset", False)),
    )


def maybe_plot_top_v3(
    rr: SweepResultV3,
    atoms: pd.DataFrame,
    returns: pd.Series,
    cfg: dict,
    out_path: Path,
    bps: float = 10.0,
) -> Path | None:
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
    ax.plot(idx, eq_b, label="Buy & Hold (SPX)", lw=1.2, alpha=0.8)
    for name, color in (("train", "#d0e0ff"), ("validation", "#fff0d0"), ("holdout", "#e0ffe0")):
        start, end = cfg["splits"][name]
        m = split_mask(idx, start, end)
        if m.any():
            ax.axvspan(idx[m][0], idx[m][-1], alpha=0.15, color=color, label=name)
    ax.set_yscale("log")
    ax.set_title("Sweep v3 — best cross-asset rule vs SPX BH (log equity)")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def print_summary_v3(
    df: pd.DataFrame,
    mc_targets: Sequence[SweepResultV3],
    meta: dict[str, Any],
    csv_path: Path,
    md_path: Path,
) -> None:
    print()
    print("=" * 72)
    print("SWEEP V3 SUMMARY")
    print("=" * 72)
    print(
        f"Runtime {meta['runtime_sec']:.1f}s | cores={meta['n_cores']} "
        f"workers={meta['n_workers']} | "
        f"atoms {meta['n_atoms_total']}/{meta['n_atoms_searchable']} | "
        f"ranked {meta['n_ranked']} | stable {meta['n_stable']} | "
        f"cross {meta['n_cross_asset']} | cross+stable {meta['n_cross_asset_stable']}"
    )
    print(f"Results: {md_path}")
    print(f"CSV:     {csv_path}")
    print()
    any_full = False
    any_cn = False
    for rr in mc_targets:
        flags = protocol_pass_v3(rr)
        if flags["passes_cross_stable_null"]:
            any_cn = True
        if flags["passes_full_bar"]:
            any_full = True
    print(
        f"{'rule':<42} {'fit':>6} {'ndq':>6} {'st':>2} {'x':>1} "
        f"{'hold':>6} {'p':>7}  bar"
    )
    print("-" * 90)
    for rr in mc_targets[:15]:
        flags = protocol_pass_v3(rr)
        hold = (
            f"{rr.holdout_excess_sharpe:+.3f}"
            if rr.holdout_excess_sharpe is not None
            else "nan"
        )
        ndq = (
            f"{rr.nasdaq_val_excess_sharpe:+.3f}"
            if rr.nasdaq_val_excess_sharpe is not None
            else "nan"
        )
        p = f"{rr.null_pvalue:.4f}" if rr.null_pvalue is not None else "n/a"
        tag = (
            "FULL" if flags["passes_full_bar"]
            else ("X+NULL" if flags["passes_cross_stable_null"] else "no")
        )
        print(
            f"{rr.rule[:42]:<42} {rr.val_fitness:+6.3f} {ndq:>6} "
            f"{'Y' if rr.stable else 'N':>2} {'Y' if rr.cross_asset else 'N':>1} "
            f"{hold:>6} {p:>7}  {tag}"
        )
    print()
    if any_full:
        print("Verdict: ≥1 FULL-BAR survivor (SPX halves + NDQ val + null + SPX holdout).")
    elif any_cn:
        print("Verdict: cross-asset+stable+null clears exist, but NONE with SPX holdout XS>0.")
    else:
        print("Verdict: NO rule clears full v3 bar (SPX halves + NDQ val + null + holdout).")
    print("=" * 72)


@click.command()
@click.option("--beam", type=int, default=DEFAULT_BEAM, show_default=True)
@click.option("--max-length", type=int, default=DEFAULT_MAX_LENGTH, show_default=True)
@click.option("--null-n", "mc_n", type=int, default=DEFAULT_MC_N, show_default=True)
@click.option("--mc-n", "mc_n_alias", type=int, default=None, hidden=True)
@click.option("--include-negation/--no-include-negation", default=True, show_default=True)
@click.option("--include-moon-or/--no-include-moon-or", default=True, show_default=True)
@click.option("--force-rebuild-atoms", is_flag=True, default=False)
@click.option("--workers", type=int, default=None)
@click.option("--config", "config_path", default=None)
@click.option("--out-dir", type=click.Path(), default=None)
@click.option("--plot/--no-plot", default=True, show_default=True)
def main(
    beam: int,
    max_length: int,
    mc_n: int,
    mc_n_alias: int | None,
    include_negation: bool,
    include_moon_or: bool,
    force_rebuild_atoms: bool,
    workers: int | None,
    config_path: str | None,
    out_dir: str | None,
    plot: bool,
) -> None:
    """Sweep v3: beam L1–L4, Nasdaq cross-asset must-pass, large MC null."""
    if mc_n_alias is not None:
        mc_n = mc_n_alias
    try:
        cfg = load_config(config_path)
        out = Path(out_dir) if out_dir else (PROJECT_ROOT / "results")
        df, mc_targets, meta = run_sweep_v3(
            beam=beam,
            mc_n=mc_n,
            max_length=max_length,
            include_negation=include_negation,
            include_moon_or=include_moon_or,
            force_rebuild_atoms=force_rebuild_atoms,
            cfg=cfg,
            verbose=True,
            out_dir=out,
            n_workers=workers,
        )
        csv_path, md_path = write_sweep_v3_results(df, mc_targets, meta, out_dir=out)

        if plot:
            # Prefer full-bar, else cross-asset stable, else best stable cross
            full = [r for r in mc_targets if protocol_pass_v3(r)["passes_full_bar"]]
            ca_st = [r for r in mc_targets if r.stable and r.cross_asset]
            top = full[0] if full else (ca_st[0] if ca_st else (
                mc_targets[0] if mc_targets else None
            ))
            if top is not None:
                try:
                    prices = load_prices(cfg=cfg, auto_fetch=False)
                    ndq = load_nasdaq_prices(cfg=cfg, auto_fetch=False)
                    rets = daily_returns(prices)
                    atoms = load_atoms_v2(cfg=cfg, dates=prices.index, auto_build=False)
                    rets, _, atoms = align_multi_asset(rets, ndq, atoms)
                    maybe_plot_top_v3(
                        top, atoms, rets, cfg,
                        out / "sweep_v3_top_equity.png",
                        bps=float(cfg["costs"]["bps_per_side"]),
                    )
                    print(f"Chart: {out / 'sweep_v3_top_equity.png'}")
                except Exception as e:
                    print(f"(chart skipped: {e})")

        print_summary_v3(df, mc_targets, meta, csv_path, md_path)
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
