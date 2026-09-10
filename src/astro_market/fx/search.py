"""FX search v1: beam search for short boolean rules under risk-managed FX scoring.

Primary fitness = validation avg R (expectancy) with penalties for low trade
count and complexity. Uses a fast rising-edge trade simulator for beam ranking;
full FxBrokerSim for holdout / charts. Holdout is report-only — never select.
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import asdict, dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Any, Literal, Sequence

import click
import numpy as np
import pandas as pd

from astro_market.atoms import (
    is_regime_sign_atom,
    load_atoms_v2,
    searchable_atom_names,
)
from astro_market.config import PROJECT_ROOT
from astro_market.fx.broker import FxBrokerSim
from astro_market.fx.data import (
    FX_PAIRS,
    atr as compute_atr,
    load_all_fx,
    load_fx_config,
)
from astro_market.fx.evaluate import (
    FxMetrics,
    evaluate_fx_strategy,
    evaluate_on_fx_split,
    metrics_from_sim,
    slice_ohlc,
)
from astro_market.fx.sizing import pips_to_price
from astro_market.fx.strategy import FxStrategy, rising_edge
from astro_market.null import empirical_pvalue
from astro_market.search import complexity_for_rule, signal_for_expr
from astro_market.sweep import (
    apply_regime_filter,
    canonical_rule_v2,
    count_episodes,
    episode_stats,
    length_of_rule,
    rule_is_pure_regime_sign,
)

Side = Literal["long", "short"]

DEFAULT_BEAM = 150
DEFAULT_BEAM_L3 = 80
DEFAULT_MAX_LENGTH = 3
DEFAULT_NULL_N = 2000
DEFAULT_LAMBDA = 0.02
DEFAULT_LAMBDA_L3 = 0.03
MIN_TRADES_VAL = 30
PARTIAL_EVERY = 2000


# ---------------------------------------------------------------------------
# Fitness helpers
# ---------------------------------------------------------------------------


def trade_count_penalty(n_trades: int, min_trades: int = MIN_TRADES_VAL) -> float:
    """Hard penalty below min_trades; soft ramp just above."""
    if n_trades < min_trades:
        return 10.0 + (min_trades - n_trades) * 0.05
    # Soft: small penalty until 1.5× min
    soft_target = int(1.5 * min_trades)
    if n_trades < soft_target:
        return 0.02 * (soft_target - n_trades) / max(1, soft_target - min_trades)
    return 0.0


def fx_fitness(
    avg_R: float,
    n_trades: int,
    complexity: int,
    *,
    lam: float = DEFAULT_LAMBDA,
    min_trades: int = MIN_TRADES_VAL,
    regime_penalty: float = 0.0,
) -> float:
    """
    Validation fitness for FX search.

    fitness = avg_R − λ·complexity − trade_count_penalty − regime_penalty
    Non-finite avg_R → −inf.
    """
    if not np.isfinite(avg_R):
        return float("-inf")
    return float(
        avg_R
        - lam * float(complexity)
        - trade_count_penalty(n_trades, min_trades)
        - regime_penalty
    )


def lambda_for_length(
    length: int,
    lam: float = DEFAULT_LAMBDA,
    lam_l3: float = DEFAULT_LAMBDA_L3,
) -> float:
    return lam_l3 if length >= 3 else lam


# ---------------------------------------------------------------------------
# Fast trade engine (search ranking)
# ---------------------------------------------------------------------------


@dataclass
class PairTape:
    pair: str
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    atr: np.ndarray
    spread: float
    fallback: float


@dataclass
class SplitTape:
    """Precomputed OHLC/ATR tapes on a split calendar (for fast scoring)."""

    name: str
    calendar: pd.DatetimeIndex
    pairs: list[PairTape]
    atoms: pd.DataFrame  # aligned to calendar
    start: pd.Timestamp
    end: pd.Timestamp | None

    @property
    def n(self) -> int:
        return len(self.calendar)


def _build_split_tape(
    name: str,
    ohlc_by_pair: dict[str, pd.DataFrame],
    atoms: pd.DataFrame,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp | None,
    *,
    pairs: list[str],
    atr_period: int,
    sl_pips_fallback: float,
    spread_pips: float,
) -> SplitTape:
    start_ts = pd.Timestamp(start)
    end_ts = None if end is None or (isinstance(end, float) and pd.isna(end)) else pd.Timestamp(end)

    cal: pd.DatetimeIndex | None = None
    for p in pairs:
        df = ohlc_by_pair[p]
        if end_ts is None:
            sub = df.index[df.index >= start_ts]
        else:
            sub = df.index[(df.index >= start_ts) & (df.index <= end_ts)]
        cal = sub if cal is None else cal.union(sub)
    assert cal is not None
    cal = cal.intersection(atoms.index).sort_values()

    tapes: list[PairTape] = []
    for p in pairs:
        df = ohlc_by_pair[p][["open", "high", "low", "close"]].astype(float).copy()
        # ATR with warm-up before split
        prior = df.loc[df.index < start_ts].tail(atr_period * 3 + 5)
        window = df.loc[df.index >= start_ts]
        if end_ts is not None:
            window = window.loc[window.index <= end_ts]
        full = pd.concat([prior, window]).sort_index()
        full = full[~full.index.duplicated(keep="last")]
        full["atr"] = compute_atr(full, period=atr_period)
        tapes.append(
            PairTape(
                pair=p,
                open=full["open"].reindex(cal).to_numpy(dtype=float),
                high=full["high"].reindex(cal).to_numpy(dtype=float),
                low=full["low"].reindex(cal).to_numpy(dtype=float),
                close=full["close"].reindex(cal).to_numpy(dtype=float),
                atr=full["atr"].reindex(cal).to_numpy(dtype=float),
                spread=float(pips_to_price(p, spread_pips)),
                fallback=float(pips_to_price(p, sl_pips_fallback)),
            )
        )
    return SplitTape(
        name=name,
        calendar=cal,
        pairs=tapes,
        atoms=atoms.reindex(cal).astype(bool),
        start=start_ts,
        end=end_ts,
    )


def rising_edge_indices(signal: np.ndarray) -> np.ndarray:
    """Indices where signal rises False→True (first True counts)."""
    s = np.asarray(signal, dtype=bool)
    if s.size == 0:
        return np.array([], dtype=np.int64)
    prev = np.empty_like(s)
    prev[0] = False
    if s.size > 1:
        prev[1:] = s[:-1]
    return np.flatnonzero(s & ~prev).astype(np.int64)


def simulate_trade_Rs(
    edge_idx: np.ndarray,
    tapes: Sequence[PairTape],
    *,
    side: int,
    sl_atr: float = 1.5,
    tp_R: float = 2.0,
    time_stop: int = 10,
    max_concurrent: int = 5,
) -> np.ndarray:
    """
    Fast multi-pair rising-edge trade R multiples.

    Approximates FxBrokerSim (same SL/TP/time-stop/spread half-costs, concurrent
    cap) without equity compounding — sufficient for beam ranking by avg R.
    """
    if edge_idx.size == 0 or not tapes:
        return np.array([], dtype=float)
    n = len(tapes[0].open)
    open_exits: list[int] = []
    Rs: list[float] = []

    for i in edge_idx:
        fill_i = int(i) + 1
        if fill_i >= n:
            continue
        open_exits = [e for e in open_exits if e >= fill_i]
        for tape in tapes:
            if len(open_exits) >= max_concurrent:
                break
            o = tape.open[fill_i]
            if not np.isfinite(o):
                continue
            atr_i = tape.atr[int(i)]
            dist = (
                sl_atr * atr_i
                if np.isfinite(atr_i) and atr_i > 0
                else tape.fallback
            )
            if not np.isfinite(dist) or dist <= 0:
                continue
            spread = tape.spread
            fill = float(o) + side * (spread / 2.0)
            sl = fill - side * dist
            tp = fill + side * dist * tp_R
            end = min(fill_i + time_stop, n)
            exit_bar: int | None = None
            R: float | None = None
            for j in range(fill_i, end):
                h = tape.high[j]
                if not np.isfinite(h):
                    continue
                lo = tape.low[j]
                if side > 0:
                    hit_sl = lo <= sl
                    hit_tp = h >= tp
                else:
                    hit_sl = h >= sl
                    hit_tp = lo <= tp
                if hit_sl:
                    R = -1.0 - 0.5 * (spread / dist)
                    exit_bar = j
                    break
                if hit_tp:
                    R = float(tp_R) - 0.5 * (spread / dist)
                    exit_bar = j
                    break
            if exit_bar is None:
                j = end - 1
                while j >= fill_i and not np.isfinite(tape.close[j]):
                    j -= 1
                if j >= fill_i:
                    exit_px = float(tape.close[j]) - side * (spread / 2.0)
                    R = side * (exit_px - fill) / dist
                    exit_bar = j
            if R is not None and exit_bar is not None:
                Rs.append(float(R))
                open_exits.append(int(exit_bar))
    return np.asarray(Rs, dtype=float)


def score_signal_on_tape(
    signal: np.ndarray,
    tape: SplitTape,
    *,
    side: Side,
    sl_atr: float,
    tp_R: float,
    time_stop: int,
    max_concurrent: int,
) -> tuple[float, int, np.ndarray]:
    """Return (avg_R, n_trades, Rs) for long or short rising edges."""
    sig = np.asarray(signal, dtype=bool)
    if side == "long":
        edges = rising_edge_indices(sig)
        side_i = +1
    else:
        edges = rising_edge_indices(~sig)
        side_i = -1
    Rs = simulate_trade_Rs(
        edges,
        tape.pairs,
        side=side_i,
        sl_atr=sl_atr,
        tp_R=tp_R,
        time_stop=time_stop,
        max_concurrent=max_concurrent,
    )
    n = int(Rs.size)
    avg = float(np.mean(Rs)) if n else float("nan")
    return avg, n, Rs


def circular_shift_tapes(tape: SplitTape, shift: int) -> list[PairTape]:
    """Circular-shift each pair's OHLC+ATR together (atoms/calendar fixed)."""
    k = int(shift) % tape.n
    if k == 0:
        return list(tape.pairs)
    out: list[PairTape] = []
    for p in tape.pairs:
        out.append(
            PairTape(
                pair=p.pair,
                open=np.roll(p.open, k),
                high=np.roll(p.high, k),
                low=np.roll(p.low, k),
                close=np.roll(p.close, k),
                atr=np.roll(p.atr, k),
                spread=p.spread,
                fallback=p.fallback,
            )
        )
    return out


# ---------------------------------------------------------------------------
# Candidate record
# ---------------------------------------------------------------------------


@dataclass
class FxSearchResult:
    rule: str
    side: Side
    length: int
    complexity: int
    val_avg_R: float
    val_total_R: float
    val_n_trades: int
    val_fitness: float
    val_fitness_raw: float = 0.0
    val_pf: float = float("nan")
    val_win_rate: float = float("nan")
    val_sharpe: float = float("nan")
    val_max_dd: float = float("nan")
    n_episodes_val: int = 0
    max_episode_val: int = 0
    val_half1_avg_R: float | None = None
    val_half2_avg_R: float | None = None
    stable: bool = False
    train_avg_R: float | None = None
    train_n_trades: int | None = None
    holdout_avg_R: float | None = None
    holdout_total_R: float | None = None
    holdout_n_trades: int | None = None
    holdout_pf: float | None = None
    holdout_win_rate: float | None = None
    holdout_sharpe: float | None = None
    holdout_max_dd: float | None = None
    null_pvalue: float | None = None
    null_n: int | None = None
    drop_reason: str = ""
    selected_on: str = "validation"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def key(self) -> str:
        return f"{canonical_rule_v2(self.rule)}|{self.side}"


def _pf_from_Rs(Rs: np.ndarray) -> float:
    if Rs.size == 0:
        return float("nan")
    wins = Rs[Rs > 0].sum()
    losses = np.abs(Rs[Rs < 0].sum())
    if losses <= 0:
        return float("inf") if wins > 0 else float("nan")
    return float(wins / losses)


def _win_rate_from_Rs(Rs: np.ndarray) -> float:
    if Rs.size == 0:
        return float("nan")
    return float(np.mean(Rs > 0))


# ---------------------------------------------------------------------------
# Scoring one (rule, side)
# ---------------------------------------------------------------------------


def score_candidate(
    rule: str,
    side: Side,
    sig_val: np.ndarray,
    tape_val: SplitTape,
    *,
    sig_train: np.ndarray | None = None,
    tape_train: SplitTape | None = None,
    sl_atr: float,
    tp_R: float,
    time_stop: int,
    max_concurrent: int,
    lam: float,
    lam_l3: float,
    min_trades: int,
    apply_filter: bool = True,
) -> FxSearchResult | None:
    length = length_of_rule(rule)
    c = complexity_for_rule(rule)
    lam_use = lambda_for_length(length, lam, lam_l3)

    if apply_filter:
        info = apply_regime_filter(rule, sig_val, sig_train)
        if info.dropped:
            return None
    else:
        from astro_market.sweep import RegimeInfo

        n_ep, max_ep = episode_stats(sig_val)
        info = RegimeInfo(n_episodes_val=n_ep, max_episode_val=max_ep)

    avg_R, n_tr, Rs = score_signal_on_tape(
        sig_val,
        tape_val,
        side=side,
        sl_atr=sl_atr,
        tp_R=tp_R,
        time_stop=time_stop,
        max_concurrent=max_concurrent,
    )
    raw = (
        float(avg_R - lam_use * c)
        if np.isfinite(avg_R)
        else float("-inf")
    )
    fit = fx_fitness(
        avg_R,
        n_tr,
        c,
        lam=lam_use,
        min_trades=min_trades,
        regime_penalty=info.penalty,
    )
    train_avg = None
    train_n = None
    if tape_train is not None and sig_train is not None:
        train_avg, train_n, _ = score_signal_on_tape(
            sig_train,
            tape_train,
            side=side,
            sl_atr=sl_atr,
            tp_R=tp_R,
            time_stop=time_stop,
            max_concurrent=max_concurrent,
        )

    return FxSearchResult(
        rule=canonical_rule_v2(rule),
        side=side,
        length=length,
        complexity=c,
        val_avg_R=float(avg_R) if np.isfinite(avg_R) else float("nan"),
        val_total_R=float(np.sum(Rs)) if Rs.size else float("nan"),
        val_n_trades=n_tr,
        val_fitness=fit,
        val_fitness_raw=raw,
        val_pf=_pf_from_Rs(Rs),
        val_win_rate=_win_rate_from_Rs(Rs),
        n_episodes_val=info.n_episodes_val,
        max_episode_val=info.max_episode_val,
        train_avg_R=train_avg,
        train_n_trades=train_n,
        drop_reason=info.drop_reason,
    )


# ---------------------------------------------------------------------------
# Beam stages
# ---------------------------------------------------------------------------


def beam_l1_fx(
    tape_train: SplitTape,
    tape_val: SplitTape,
    *,
    searchable: Sequence[str],
    beam: int = DEFAULT_BEAM,
    include_negation: bool = True,
    sides: Sequence[Side] = ("long", "short"),
    sl_atr: float = 1.5,
    tp_R: float = 2.0,
    time_stop: int = 10,
    max_concurrent: int = 5,
    lam: float = DEFAULT_LAMBDA,
    lam_l3: float = DEFAULT_LAMBDA_L3,
    min_trades: int = MIN_TRADES_VAL,
    verbose: bool = True,
) -> list[FxSearchResult]:
    results: list[FxSearchResult] = []
    seen: set[str] = set()
    n_drop = 0
    for col in searchable:
        cands = [col]
        if include_negation:
            cands.append(f"~{col}")
        for rule in cands:
            key_rule = canonical_rule_v2(rule)
            sig_tr = signal_for_expr(rule, tape_train.atoms)
            sig_va = signal_for_expr(rule, tape_val.atoms)
            if not np.any(sig_va) or np.all(sig_va):
                continue
            for side in sides:
                key = f"{key_rule}|{side}"
                if key in seen:
                    continue
                seen.add(key)
                rr = score_candidate(
                    rule,
                    side,
                    sig_va,
                    tape_val,
                    sig_train=sig_tr,
                    tape_train=tape_train,
                    sl_atr=sl_atr,
                    tp_R=tp_R,
                    time_stop=time_stop,
                    max_concurrent=max_concurrent,
                    lam=lam,
                    lam_l3=lam_l3,
                    min_trades=min_trades,
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


def beam_l2_fx(
    l1_beam: Sequence[FxSearchResult],
    tape_train: SplitTape,
    tape_val: SplitTape,
    *,
    beam: int = DEFAULT_BEAM,
    sides: Sequence[Side] = ("long", "short"),
    sl_atr: float = 1.5,
    tp_R: float = 2.0,
    time_stop: int = 10,
    max_concurrent: int = 5,
    lam: float = DEFAULT_LAMBDA,
    lam_l3: float = DEFAULT_LAMBDA_L3,
    min_trades: int = MIN_TRADES_VAL,
    partial_path: Path | None = None,
    verbose: bool = True,
) -> list[FxSearchResult]:
    # Unique L1 rules (ignore side) as AND seeds — keep best fitness literal set
    rule_best: dict[str, FxSearchResult] = {}
    for rr in l1_beam:
        prev = rule_best.get(rr.rule)
        if prev is None or rr.val_fitness > prev.val_fitness:
            rule_best[rr.rule] = rr
    seeds = list(rule_best.values())
    sig_tr = {rr.rule: signal_for_expr(rr.rule, tape_train.atoms) for rr in seeds}
    sig_va = {rr.rule: signal_for_expr(rr.rule, tape_val.atoms) for rr in seeds}

    results: list[FxSearchResult] = []
    seen: set[str] = set()
    n_scored = 0
    n_drop = 0

    for ra, rb in combinations(seeds, 2):
        a, b = ra.rule, rb.rule
        if a.lstrip("~") == b.lstrip("~") and (a.startswith("~") != b.startswith("~")):
            continue
        rule = canonical_rule_v2(f"{a} & {b}")
        sva = sig_va[a] & sig_va[b]
        if not np.any(sva) or np.all(sva):
            continue
        if np.array_equal(sva, sig_va[a]) or np.array_equal(sva, sig_va[b]):
            continue
        str_ = sig_tr[a] & sig_tr[b]
        for side in sides:
            key = f"{rule}|{side}"
            if key in seen:
                continue
            seen.add(key)
            rr = score_candidate(
                rule,
                side,
                sva,
                tape_val,
                sig_train=str_,
                tape_train=tape_train,
                sl_atr=sl_atr,
                tp_R=tp_R,
                time_stop=time_stop,
                max_concurrent=max_concurrent,
                lam=lam,
                lam_l3=lam_l3,
                min_trades=min_trades,
            )
            n_scored += 1
            if rr is None:
                n_drop += 1
                continue
            results.append(rr)
            if partial_path and n_scored % PARTIAL_EVERY == 0:
                _write_partial(results, partial_path, stage="L2")

    results.sort(key=lambda x: x.val_fitness, reverse=True)
    if verbose:
        print(
            f"  L2: scored={n_scored} kept={len(results)} dropped_regime={n_drop} "
            f"beam_keep={min(beam, len(results))}"
        )
    if partial_path:
        _write_partial(results[:beam], partial_path, stage="L2_beam")
    return results[:beam]


def beam_l3_fx(
    l2_beam: Sequence[FxSearchResult],
    l1_beam: Sequence[FxSearchResult],
    tape_train: SplitTape,
    tape_val: SplitTape,
    *,
    beam: int = DEFAULT_BEAM_L3,
    sides: Sequence[Side] = ("long", "short"),
    sl_atr: float = 1.5,
    tp_R: float = 2.0,
    time_stop: int = 10,
    max_concurrent: int = 5,
    lam: float = DEFAULT_LAMBDA,
    lam_l3: float = DEFAULT_LAMBDA_L3,
    min_trades: int = MIN_TRADES_VAL,
    partial_path: Path | None = None,
    verbose: bool = True,
) -> list[FxSearchResult]:
    rule_l1: dict[str, FxSearchResult] = {}
    for rr in l1_beam:
        prev = rule_l1.get(rr.rule)
        if prev is None or rr.val_fitness > prev.val_fitness:
            rule_l1[rr.rule] = rr
    rule_l2: dict[str, FxSearchResult] = {}
    for rr in l2_beam:
        prev = rule_l2.get(rr.rule)
        if prev is None or rr.val_fitness > prev.val_fitness:
            rule_l2[rr.rule] = rr

    l1_list = list(rule_l1.values())
    l2_list = list(rule_l2.values())
    sig_l1_tr = {rr.rule: signal_for_expr(rr.rule, tape_train.atoms) for rr in l1_list}
    sig_l1_va = {rr.rule: signal_for_expr(rr.rule, tape_val.atoms) for rr in l1_list}
    sig_l2_tr = {rr.rule: signal_for_expr(rr.rule, tape_train.atoms) for rr in l2_list}
    sig_l2_va = {rr.rule: signal_for_expr(rr.rule, tape_val.atoms) for rr in l2_list}

    results: list[FxSearchResult] = []
    seen: set[str] = set()
    n_scored = 0
    n_drop = 0

    for r2 in l2_list:
        base = r2.rule
        base_atoms = set(
            base.replace("~", " ").replace("&", " ").replace("|", " ").split()
        )
        for r1 in l1_list:
            lit = r1.rule
            lit_core = lit.lstrip("~")
            if lit_core in base_atoms or lit in base_atoms:
                continue
            rule = canonical_rule_v2(f"{base} & {lit}")
            sva = sig_l2_va[base] & sig_l1_va[lit]
            if not np.any(sva) or np.all(sva):
                continue
            if np.array_equal(sva, sig_l2_va[base]) or np.array_equal(sva, sig_l1_va[lit]):
                continue
            str_ = sig_l2_tr[base] & sig_l1_tr[lit]
            for side in sides:
                key = f"{rule}|{side}"
                if key in seen:
                    continue
                seen.add(key)
                rr = score_candidate(
                    rule,
                    side,
                    sva,
                    tape_val,
                    sig_train=str_,
                    tape_train=tape_train,
                    sl_atr=sl_atr,
                    tp_R=tp_R,
                    time_stop=time_stop,
                    max_concurrent=max_concurrent,
                    lam=lam,
                    lam_l3=lam_l3,
                    min_trades=min_trades,
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


def _write_partial(
    results: Sequence[FxSearchResult], path: Path, stage: str = ""
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        r.to_dict()
        for r in sorted(results, key=lambda x: x.val_fitness, reverse=True)
    ]
    df = pd.DataFrame(rows)
    df["partial_stage"] = stage
    df.to_csv(path, index=False)


# ---------------------------------------------------------------------------
# Stability / holdout / MC
# ---------------------------------------------------------------------------


def attach_split_val_stability(
    results: Sequence[FxSearchResult],
    tape_val: SplitTape,
    *,
    sl_atr: float,
    tp_R: float,
    time_stop: int,
    max_concurrent: int,
) -> list[FxSearchResult]:
    """Require avg R > 0 on both temporal halves of validation when possible."""
    n = tape_val.n
    mid = n // 2
    out: list[FxSearchResult] = []
    for rr in results:
        sig = signal_for_expr(rr.rule, tape_val.atoms)
        # Score each half with a sliced view of the same tapes
        h1_avg, h1_n, _ = _score_half(
            sig, tape_val, 0, mid, side=rr.side,
            sl_atr=sl_atr, tp_R=tp_R, time_stop=time_stop, max_concurrent=max_concurrent,
        )
        h2_avg, h2_n, _ = _score_half(
            sig, tape_val, mid, n, side=rr.side,
            sl_atr=sl_atr, tp_R=tp_R, time_stop=time_stop, max_concurrent=max_concurrent,
        )
        rr.val_half1_avg_R = h1_avg
        rr.val_half2_avg_R = h2_avg
        rr.stable = bool(
            np.isfinite(h1_avg)
            and np.isfinite(h2_avg)
            and h1_avg > 0
            and h2_avg > 0
            and h1_n >= max(5, MIN_TRADES_VAL // 4)
            and h2_n >= max(5, MIN_TRADES_VAL // 4)
        )
        out.append(rr)
    return out


def _score_half(
    signal: np.ndarray,
    tape: SplitTape,
    start: int,
    end: int,
    *,
    side: Side,
    sl_atr: float,
    tp_R: float,
    time_stop: int,
    max_concurrent: int,
) -> tuple[float, int, np.ndarray]:
    """Score trades whose signal edge falls inside [start, end)."""
    sig = np.asarray(signal, dtype=bool)
    if side == "long":
        edges = rising_edge_indices(sig)
        side_i = +1
    else:
        edges = rising_edge_indices(~sig)
        side_i = -1
    edges = edges[(edges >= start) & (edges < end)]
    # Use pair tapes sliced? Entries need fill at i+1 which may be past end —
    # allow fills slightly past end using full tape (realistic continuation).
    Rs = simulate_trade_Rs(
        edges,
        tape.pairs,
        side=side_i,
        sl_atr=sl_atr,
        tp_R=tp_R,
        time_stop=time_stop,
        max_concurrent=max_concurrent,
    )
    n = int(Rs.size)
    avg = float(np.mean(Rs)) if n else float("nan")
    return avg, n, Rs


def attach_holdout_broker(
    results: Sequence[FxSearchResult],
    ohlc: dict[str, pd.DataFrame],
    atoms: pd.DataFrame,
    *,
    cfg: dict,
    pairs: list[str],
    risk_pct: float,
    sl_atr: float,
    tp_R: float,
    atr_period: int,
    sl_pips_fallback: float,
    time_stop: int | None,
) -> list[FxSearchResult]:
    """Full broker metrics on holdout (report-only). Also refresh val sharpe/DD."""
    broker = _broker_from_cfg(cfg)
    out: list[FxSearchResult] = []
    for rr in results:
        strat = FxStrategy(
            rule=rr.rule,
            side=rr.side,
            risk_pct=risk_pct,
            sl_atr=sl_atr,
            tp_R=tp_R,
            atr_period=atr_period,
            sl_pips_fallback=sl_pips_fallback,
            time_stop=time_stop,
        )
        m_h = evaluate_on_fx_split(
            strat, ohlc, atoms, "holdout", cfg=cfg, broker=broker, pairs=pairs
        )
        rr.holdout_avg_R = m_h.avg_R
        rr.holdout_total_R = (
            float(m_h.avg_R * m_h.n_trades)
            if np.isfinite(m_h.avg_R) and m_h.n_trades
            else float("nan")
        )
        rr.holdout_n_trades = m_h.n_trades
        rr.holdout_pf = m_h.profit_factor
        rr.holdout_win_rate = m_h.win_rate
        rr.holdout_sharpe = m_h.sharpe
        rr.holdout_max_dd = m_h.max_dd

        m_v = evaluate_on_fx_split(
            strat, ohlc, atoms, "validation", cfg=cfg, broker=broker, pairs=pairs
        )
        rr.val_sharpe = m_v.sharpe
        rr.val_max_dd = m_v.max_dd
        # Prefer broker avg R / PF / trades for reported val metrics
        if m_v.n_trades > 0 and np.isfinite(m_v.avg_R):
            rr.val_avg_R = m_v.avg_R
            rr.val_pf = m_v.profit_factor
            rr.val_win_rate = m_v.win_rate
            rr.val_n_trades = m_v.n_trades
            rr.val_total_R = float(m_v.avg_R * m_v.n_trades)
        out.append(rr)
    return out


def monte_carlo_avg_R_pvalue(
    rule: str,
    side: Side,
    tape_val: SplitTape,
    *,
    n: int = DEFAULT_NULL_N,
    seed: int = 42,
    sl_atr: float = 1.5,
    tp_R: float = 2.0,
    time_stop: int = 10,
    max_concurrent: int = 5,
) -> float:
    """
    Null: circular-shift each pair's price/ATR series together; atoms fixed.
    Score = validation avg R on rising-edge trades.
    """
    sig = signal_for_expr(rule, tape_val.atoms)
    obs_avg, _, _ = score_signal_on_tape(
        sig,
        tape_val,
        side=side,
        sl_atr=sl_atr,
        tp_R=tp_R,
        time_stop=time_stop,
        max_concurrent=max_concurrent,
    )
    if not np.isfinite(obs_avg) or tape_val.n < 2:
        return float("nan")

    if side == "long":
        edges = rising_edge_indices(sig)
        side_i = +1
    else:
        edges = rising_edge_indices(~sig)
        side_i = -1

    rng = np.random.default_rng(seed)
    shifts = rng.integers(1, tape_val.n, size=n)
    scores = np.empty(n, dtype=float)
    for i, k in enumerate(shifts):
        shifted = circular_shift_tapes(tape_val, int(k))
        Rs = simulate_trade_Rs(
            edges,
            shifted,
            side=side_i,
            sl_atr=sl_atr,
            tp_R=tp_R,
            time_stop=time_stop,
            max_concurrent=max_concurrent,
        )
        scores[i] = float(np.mean(Rs)) if Rs.size else float("-inf")
    return empirical_pvalue(obs_avg, scores, alternative="greater")


def select_mc_targets(
    ranked: Sequence[FxSearchResult],
    *,
    n_stable: int = 20,
    n_unstable: int = 5,
) -> list[FxSearchResult]:
    stable = [r for r in ranked if r.stable]
    unstable = [r for r in ranked if not r.stable]
    chosen: list[FxSearchResult] = []
    seen: set[str] = set()
    for r in stable[:n_stable] + unstable[:n_unstable]:
        if r.key not in seen:
            seen.add(r.key)
            chosen.append(r)
    return chosen


def best_side_per_rule(ranked: Sequence[FxSearchResult]) -> list[FxSearchResult]:
    """Collapse to best side per canonical rule by val fitness."""
    best: dict[str, FxSearchResult] = {}
    for rr in ranked:
        prev = best.get(rr.rule)
        if prev is None or rr.val_fitness > prev.val_fitness:
            best[rr.rule] = rr
    return sorted(best.values(), key=lambda x: x.val_fitness, reverse=True)


# ---------------------------------------------------------------------------
# Broker / strategy helpers
# ---------------------------------------------------------------------------


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


def _risk_params(cfg: dict) -> dict[str, Any]:
    fx = cfg.get("fx", cfg)
    risk = fx.get("risk", {})
    costs = fx.get("costs", {})
    return {
        "risk_pct": float(risk.get("risk_pct", 0.005)),
        "sl_atr": float(risk.get("sl_atr", 1.5)),
        "tp_R": float(risk.get("tp_R", 2.0)),
        "atr_period": int(risk.get("atr_period", 14)),
        "sl_pips_fallback": float(risk.get("sl_pips_fallback", 50.0)),
        "time_stop": risk.get("time_stop", 10),
        "max_concurrent": int(risk.get("max_concurrent", 5)),
        "spread_pips": float(costs.get("spread_pips", 1.0)),
        "pairs": list(fx.get("pairs") or FX_PAIRS),
        "initial_equity": float(fx.get("initial_equity", 100_000.0)),
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def protocol_pass_fx(rr: FxSearchResult) -> bool:
    """Full bar: stable val halves, enough trades, null p<0.05, holdout avg R>0."""
    if not rr.stable:
        return False
    if rr.val_n_trades < MIN_TRADES_VAL:
        return False
    if rr.null_pvalue is None or not np.isfinite(rr.null_pvalue) or rr.null_pvalue >= 0.05:
        return False
    if rr.holdout_avg_R is None or not np.isfinite(rr.holdout_avg_R) or rr.holdout_avg_R <= 0:
        return False
    return True


def write_report(
    ranked: Sequence[FxSearchResult],
    meta: dict[str, Any],
    out_md: Path,
    out_csv: Path,
) -> None:
    out_md.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([r.to_dict() for r in ranked])
    df.to_csv(out_csv, index=False)

    survivors = [r for r in ranked if protocol_pass_fx(r)]
    top = ranked[:25]
    folklore = meta.get("folklore_rows") or []

    lines: list[str] = []
    lines.append("# FX search v1 — risk-managed beam search")
    lines.append("")
    lines.append("## Method")
    lines.append("")
    lines.append(
        "Boolean astro rules scored with the **FX risk-unit engine** "
        "(rising-edge entries, fixed fractional risk, SL/TP) — **not** the equity long/flat model."
    )
    lines.append("")
    lines.append(
        f"- Atoms: atoms_v2 searchable pool (outer-planet *sign* regimes excluded); "
        f"total={meta.get('n_atoms_total')} searchable={meta.get('n_atoms_search')}"
    )
    lines.append(
        f"- Beam: L1→L2"
        + ("→L3" if meta.get("ran_l3") else "")
        + f" · beam={meta.get('beam')} · L3_beam={meta.get('beam_l3')} · max_length={meta.get('max_length')}"
    )
    lines.append(
        f"- Fitness (validation): `avg_R − λ·complexity − trade_penalty` "
        f"(λ={meta.get('lam')}, λ_L3={meta.get('lam_l3')}, min_trades={meta.get('min_trades')})"
    )
    lines.append(
        "- Sides: **long** and **short** (short = rising edge of `~rule`) searched separately; "
        "tables show best side per rule where collapsed."
    )
    lines.append(
        "- Defaults from `configs/fx.yaml`: risk 0.5%, SL 1.5×ATR(14), TP 2R, time-stop 10, "
        "spread 1 pip, max concurrent 5."
    )
    lines.append(
        "- Split-val stability: avg R > 0 on **both** temporal halves of validation."
    )
    lines.append(
        "- Holdout **report only — never used for selection**."
    )
    lines.append(
        f"- Monte Carlo null: circular-shift each pair's OHLC+ATR together (atoms fixed), "
        f"n={meta.get('null_n')}, seed={meta.get('seed')}, score=val avg R."
    )
    lines.append("")
    lines.append("### Protocol splits (FX)")
    lines.append("")
    lines.append("| Split | Dates | Role |")
    lines.append("|-------|-------|------|")
    lines.append("| Train | 2000-01-01 → 2014-12-31 | Research (truncated by data start) |")
    lines.append("| Validation | 2015-01-01 → 2019-12-31 | Selection |")
    lines.append("| Holdout | 2020-01-01 → present | Report only |")
    lines.append("")
    lines.append(
        f"Runtime: **{meta.get('runtime_sec', float('nan')):.1f}s** "
        f"(L1 {meta.get('l1_sec', 0):.1f}s · L2 {meta.get('l2_sec', 0):.1f}s · "
        f"L3 {meta.get('l3_sec', 0):.1f}s · MC {meta.get('mc_sec', 0):.1f}s) · "
        f"cores={meta.get('n_cores')} · ranked={meta.get('n_ranked')} · "
        f"stable={meta.get('n_stable')} · "
        f"data {meta.get('data_start')} → {meta.get('data_end')}"
    )
    lines.append("")
    lines.append("## Verdict")
    lines.append("")
    if survivors:
        lines.append(
            f"**{len(survivors)} full-bar survivor(s)** "
            "(val avg R>0 both halves, ≥30 val trades, null p<0.05, holdout avg R>0)."
        )
        lines.append("")
        lines.append(
            "Honesty caveat: many rules were searched; raw p<0.05 does **not** survive "
            "naive Bonferroni across the full beam. Treat as exploratory unless holdout "
            "and pre-registered replication agree."
        )
        for s in survivors[:10]:
            lines.append(
                f"- `{s.rule}` [{s.side}] val_avg_R={s.val_avg_R:+.3f} "
                f"hold_avg_R={s.holdout_avg_R:+.3f} p={s.null_pvalue:.4f} "
                f"trades={s.val_n_trades}/{s.holdout_n_trades}"
            )
    else:
        lines.append(
            "**No full-bar survivors.** No rule simultaneously cleared: "
            "positive avg R on both validation halves, enough trades, null p<0.05, "
            "**and** holdout avg R>0."
        )
        lines.append("")
        lines.append(
            "Multiple-testing honesty: beam search over hundreds of atoms × sides × "
            "AND combinations will surface spurious val winners. Folklore moon rules "
            "remain baselines, not winners. Absence of full-bar survivors is the honest result."
        )
    lines.append("")
    lines.append("## Top rules by validation fitness")
    lines.append("")
    lines.append(
        "| Rank | Rule | Side | L | Val avgR | H1 | H2 | Stable | Val PF | Trades | "
        "Hold avgR | Hold PF | Null p |"
    )
    lines.append(
        "|-----:|------|:----:|--:|---------:|---:|---:|:------:|-------:|-------:|"
        "----------:|--------:|-------:|"
    )
    for i, rr in enumerate(top, 1):
        h1 = rr.val_half1_avg_R
        h2 = rr.val_half2_avg_R
        hp = rr.null_pvalue
        lines.append(
            f"| {i} | `{rr.rule}` | {rr.side} | {rr.length} | "
            f"{rr.val_avg_R:+.3f} | "
            f"{(h1 if h1 is not None and np.isfinite(h1) else float('nan')):+.3f} | "
            f"{(h2 if h2 is not None and np.isfinite(h2) else float('nan')):+.3f} | "
            f"{'Y' if rr.stable else 'N'} | "
            f"{rr.val_pf:.2f} | {rr.val_n_trades} | "
            f"{(rr.holdout_avg_R if rr.holdout_avg_R is not None else float('nan')):+.3f} | "
            f"{(rr.holdout_pf if rr.holdout_pf is not None else float('nan')):.2f} | "
            f"{(f'{hp:.4f}' if hp is not None and np.isfinite(hp) else '—')} |"
        )
    lines.append("")
    if folklore:
        lines.append("## Folklore baselines (not search winners)")
        lines.append("")
        lines.append("| Rule | Side | Val avgR | Val trades | Hold avgR | Hold trades |")
        lines.append("|------|:----:|---------:|-----------:|----------:|------------:|")
        for row in folklore:
            lines.append(
                f"| `{row['rule']}` | {row['side']} | {row['val_avg_R']:+.3f} | "
                f"{row['val_n']} | {row['hold_avg_R']:+.3f} | {row['hold_n']} |"
            )
        lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append(
        "- Beam ranking used a fast concurrent-capped R simulator; reported val/hold "
        "Sharpe, max DD, and final avg R/PF for survivors use full `FxBrokerSim`."
    )
    lines.append(
        "- Equity chart for the top rule (if generated): `results/fx_search_v1_top_equity.png`."
    )
    lines.append("")
    out_md.write_text("\n".join(lines), encoding="utf-8")


def plot_top_equity(
    rr: FxSearchResult,
    ohlc: dict[str, pd.DataFrame],
    atoms: pd.DataFrame,
    cfg: dict,
    pairs: list[str],
    out_path: Path,
    risk_params: dict[str, Any],
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    broker = _broker_from_cfg(cfg)
    strat = FxStrategy(
        rule=rr.rule,
        side=rr.side,
        risk_pct=risk_params["risk_pct"],
        sl_atr=risk_params["sl_atr"],
        tp_R=risk_params["tp_R"],
        atr_period=risk_params["atr_period"],
        sl_pips_fallback=risk_params["sl_pips_fallback"],
        time_stop=risk_params["time_stop"],
    )
    # Full sample equity for illustration; annotate split lines
    m, sim = evaluate_fx_strategy(
        strat, ohlc, atoms, broker=broker, pairs=pairs, cfg=cfg
    )
    if len(sim.equity_curve) == 0:
        return
    eq = sim.equity_curve
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(eq.index, eq.values, lw=1.2, label=f"{rr.rule} [{rr.side}]")
    for label, color in (
        ("2015-01-01", "#888"),
        ("2020-01-01", "#c44"),
    ):
        ax.axvline(pd.Timestamp(label), color=color, ls="--", lw=0.8, alpha=0.8)
    ax.set_title(
        f"FX search v1 top · val avgR={rr.val_avg_R:+.3f} · "
        f"hold avgR={(rr.holdout_avg_R or float('nan')):+.3f}"
    )
    ax.set_ylabel("Equity")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _folklore_baseline_rows(
    ohlc: dict[str, pd.DataFrame],
    atoms: pd.DataFrame,
    cfg: dict,
    pairs: list[str],
    risk_params: dict[str, Any],
) -> list[dict[str, Any]]:
    rules = list(cfg.get("fx", {}).get("folklore_rules") or [
        "moon_phase_new",
        "moon_phase_full",
        "mercury_retro",
    ])
    broker = _broker_from_cfg(cfg)
    rows = []
    for rule in rules:
        for side in ("long", "short"):
            strat = FxStrategy(
                rule=rule,
                side=side,  # type: ignore[arg-type]
                risk_pct=risk_params["risk_pct"],
                sl_atr=risk_params["sl_atr"],
                tp_R=risk_params["tp_R"],
                atr_period=risk_params["atr_period"],
                sl_pips_fallback=risk_params["sl_pips_fallback"],
                time_stop=risk_params["time_stop"],
            )
            mv = evaluate_on_fx_split(
                strat, ohlc, atoms, "validation", cfg=cfg, broker=broker, pairs=pairs
            )
            mh = evaluate_on_fx_split(
                strat, ohlc, atoms, "holdout", cfg=cfg, broker=broker, pairs=pairs
            )
            rows.append(
                {
                    "rule": rule,
                    "side": side,
                    "val_avg_R": mv.avg_R,
                    "val_n": mv.n_trades,
                    "hold_avg_R": mh.avg_R,
                    "hold_n": mh.n_trades,
                }
            )
    return rows


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def run_fx_search(
    *,
    beam: int = DEFAULT_BEAM,
    beam_l3: int = DEFAULT_BEAM_L3,
    max_length: int = DEFAULT_MAX_LENGTH,
    null_n: int = DEFAULT_NULL_N,
    min_trades: int = MIN_TRADES_VAL,
    lam: float = DEFAULT_LAMBDA,
    lam_l3: float = DEFAULT_LAMBDA_L3,
    include_negation: bool = True,
    cfg: dict | None = None,
    verbose: bool = True,
    out_dir: Path | None = None,
    seed: int = 42,
) -> tuple[pd.DataFrame, list[FxSearchResult], dict[str, Any]]:
    t0 = time.perf_counter()
    cfg = cfg or load_fx_config()
    rp = _risk_params(cfg)
    pairs = rp["pairs"]
    out_dir = out_dir or (PROJECT_ROOT / "results")
    out_dir.mkdir(parents=True, exist_ok=True)
    partial_path = out_dir / "fx_search_v1_partial.csv"
    log_path = out_dir / "fx_search_v1_run.log"
    log_lines: list[str] = []

    def log(msg: str) -> None:
        log_lines.append(msg)
        if verbose:
            print(msg)

    n_cores = os.cpu_count() or 1
    log(
        f"FX search v1 | cores={n_cores} beam={beam} beam_l3={beam_l3} "
        f"max_length={max_length} null_n={null_n} λ={lam} λ_L3={lam_l3} "
        f"min_trades={min_trades}"
    )

    ohlc = load_all_fx(pairs=pairs, cfg=cfg, auto_fetch=False)
    all_idx: pd.DatetimeIndex | None = None
    starts, ends = [], []
    for p, df in ohlc.items():
        all_idx = df.index if all_idx is None else all_idx.union(df.index)
        if len(df):
            starts.append(df.index.min())
            ends.append(df.index.max())
    data_start = min(starts) if starts else None
    data_end = max(ends) if ends else None
    assert all_idx is not None

    atoms = load_atoms_v2(dates=all_idx, auto_build=True)
    # Align atoms to FX overlap
    common = all_idx.intersection(atoms.index)
    atoms = atoms.loc[common].astype(bool)
    n_atoms_total = int(atoms.shape[1])
    searchable = searchable_atom_names(atoms.columns)
    # Extra safety: drop any regime sign that slipped through
    searchable = [c for c in searchable if not is_regime_sign_atom(c)]
    n_atoms_search = len(searchable)
    log(
        f"Atoms total={n_atoms_total} searchable={n_atoms_search} · "
        f"FX span {data_start} → {data_end} · pairs={pairs}"
    )

    splits = cfg["splits"]
    tape_train = _build_split_tape(
        "train",
        ohlc,
        atoms,
        splits["train"][0],
        splits["train"][1],
        pairs=pairs,
        atr_period=rp["atr_period"],
        sl_pips_fallback=rp["sl_pips_fallback"],
        spread_pips=rp["spread_pips"],
    )
    tape_val = _build_split_tape(
        "validation",
        ohlc,
        atoms,
        splits["validation"][0],
        splits["validation"][1],
        pairs=pairs,
        atr_period=rp["atr_period"],
        sl_pips_fallback=rp["sl_pips_fallback"],
        spread_pips=rp["spread_pips"],
    )
    log(f"Splits: train={tape_train.n}d val={tape_val.n}d")

    sides: tuple[Side, ...] = ("long", "short")
    ts = int(rp["time_stop"]) if rp["time_stop"] is not None else 10

    log("L1 beam search (atoms + negations × long/short)...")
    t1 = time.perf_counter()
    l1 = beam_l1_fx(
        tape_train,
        tape_val,
        searchable=searchable,
        beam=beam,
        include_negation=include_negation,
        sides=sides,
        sl_atr=rp["sl_atr"],
        tp_R=rp["tp_R"],
        time_stop=ts,
        max_concurrent=rp["max_concurrent"],
        lam=lam,
        lam_l3=lam_l3,
        min_trades=min_trades,
        verbose=verbose,
    )
    l1_sec = time.perf_counter() - t1
    log(f"  L1 done in {l1_sec:.1f}s  top: {l1[0].rule} [{l1[0].side}] fit={l1[0].val_fitness:+.4f}" if l1 else "  L1 empty")
    _write_partial(l1, partial_path, stage="L1_beam")

    log("L2 beam search (AND pairs × sides)...")
    t2 = time.perf_counter()
    l2 = beam_l2_fx(
        l1,
        tape_train,
        tape_val,
        beam=beam,
        sides=sides,
        sl_atr=rp["sl_atr"],
        tp_R=rp["tp_R"],
        time_stop=ts,
        max_concurrent=rp["max_concurrent"],
        lam=lam,
        lam_l3=lam_l3,
        min_trades=min_trades,
        partial_path=partial_path,
        verbose=verbose,
    )
    l2_sec = time.perf_counter() - t2
    log(f"  L2 done in {l2_sec:.1f}s")

    l3: list[FxSearchResult] = []
    l3_sec = 0.0
    ran_l3 = max_length >= 3
    if ran_l3:
        log("L3 beam search (L2 ⊕ L1)...")
        t3 = time.perf_counter()
        l3 = beam_l3_fx(
            l2,
            l1,
            tape_train,
            tape_val,
            beam=beam_l3,
            sides=sides,
            sl_atr=rp["sl_atr"],
            tp_R=rp["tp_R"],
            time_stop=ts,
            max_concurrent=rp["max_concurrent"],
            lam=lam,
            lam_l3=lam_l3,
            min_trades=min_trades,
            partial_path=partial_path,
            verbose=verbose,
        )
        l3_sec = time.perf_counter() - t3
        log(f"  L3 done in {l3_sec:.1f}s")

    best: dict[str, FxSearchResult] = {}
    for rr in list(l1) + list(l2) + list(l3):
        prev = best.get(rr.key)
        if prev is None or rr.val_fitness > prev.val_fitness:
            best[rr.key] = rr
    ranked = sorted(best.values(), key=lambda x: x.val_fitness, reverse=True)
    log(f"Unique ranked={len(ranked)}; attaching split-val stability...")
    ranked = attach_split_val_stability(
        ranked,
        tape_val,
        sl_atr=rp["sl_atr"],
        tp_R=rp["tp_R"],
        time_stop=ts,
        max_concurrent=rp["max_concurrent"],
    )
    n_stable = sum(1 for r in ranked if r.stable)
    log(f"  Stable survivors (avg R>0 both val halves): {n_stable}")

    # Collapse to best side per rule for reporting pool, but keep side-specific
    # stable list for MC. Prefer ranking by fitness with sides.
    collapsed = best_side_per_rule(ranked)

    # Attach holdout via full broker on top candidates (stable + top fitness)
    mc_pool = select_mc_targets(ranked, n_stable=25, n_unstable=10)
    # Also ensure top-15 by fitness get holdout even if unstable
    top_fit = [r for r in ranked[:15] if r.key not in {x.key for x in mc_pool}]
    holdout_targets = mc_pool + top_fit
    log(f"Full-broker holdout+val refresh on {len(holdout_targets)} targets...")
    holdout_targets = attach_holdout_broker(
        holdout_targets,
        ohlc,
        atoms,
        cfg=cfg,
        pairs=pairs,
        risk_pct=rp["risk_pct"],
        sl_atr=rp["sl_atr"],
        tp_R=rp["tp_R"],
        atr_period=rp["atr_period"],
        sl_pips_fallback=rp["sl_pips_fallback"],
        time_stop=rp["time_stop"],
    )
    # Merge holdout fields back into ranked
    by_key = {r.key: r for r in holdout_targets}
    for i, rr in enumerate(ranked):
        if rr.key in by_key:
            ranked[i] = by_key[rr.key]

    log(f"Monte Carlo null n={null_n} on {len(mc_pool)} targets...")
    tmc = time.perf_counter()
    for j, rr in enumerate(mc_pool):
        # use refreshed object from ranked
        target = by_key.get(rr.key, rr)
        p = monte_carlo_avg_R_pvalue(
            target.rule,
            target.side,
            tape_val,
            n=null_n,
            seed=seed + j,
            sl_atr=rp["sl_atr"],
            tp_R=rp["tp_R"],
            time_stop=ts,
            max_concurrent=rp["max_concurrent"],
        )
        target.null_pvalue = p
        target.null_n = null_n
        by_key[target.key] = target
        if verbose and (j + 1) % 5 == 0:
            print(f"  MC {j+1}/{len(mc_pool)}...")
    mc_sec = time.perf_counter() - tmc
    for i, rr in enumerate(ranked):
        if rr.key in by_key:
            ranked[i] = by_key[rr.key]
    log(f"  MC done in {mc_sec:.1f}s")

    # Folklore baselines
    log("Folklore baselines...")
    folklore_rows = _folklore_baseline_rows(ohlc, atoms, cfg, pairs, rp)

    runtime = time.perf_counter() - t0
    meta: dict[str, Any] = {
        "runtime_sec": runtime,
        "l1_sec": l1_sec,
        "l2_sec": l2_sec,
        "l3_sec": l3_sec,
        "mc_sec": mc_sec,
        "n_cores": n_cores,
        "beam": beam,
        "beam_l3": beam_l3,
        "max_length": max_length,
        "ran_l3": ran_l3,
        "null_n": null_n,
        "seed": seed,
        "lam": lam,
        "lam_l3": lam_l3,
        "min_trades": min_trades,
        "n_atoms_total": n_atoms_total,
        "n_atoms_search": n_atoms_search,
        "n_ranked": len(ranked),
        "n_stable": n_stable,
        "data_start": str(data_start.date()) if data_start is not None else None,
        "data_end": str(data_end.date()) if data_end is not None else None,
        "folklore_rows": folklore_rows,
        "n_survivors": sum(1 for r in ranked if protocol_pass_fx(r)),
    }

    # Prefer collapsed best-side ranking for the published top table, but keep
    # full side-specific CSV. Rebuild collapsed after MC/holdout merge.
    collapsed = best_side_per_rule(ranked)
    # Ensure collapsed rows carry MC/holdout from ranked
    out_md = out_dir / "fx_search_v1.md"
    out_csv = out_dir / "fx_search_v1.csv"
    write_report(collapsed, meta, out_md, out_csv)
    # Also write full side-specific CSV
    pd.DataFrame([r.to_dict() for r in ranked]).to_csv(
        out_dir / "fx_search_v1_all_sides.csv", index=False
    )

    if collapsed:
        plot_top_equity(
            collapsed[0],
            ohlc,
            atoms,
            cfg,
            pairs,
            out_dir / "fx_search_v1_top_equity.png",
            rp,
        )

    log_path.write_text("\n".join(log_lines) + f"\nTotal runtime {runtime:.1f}s\n", encoding="utf-8")
    log(f"Wrote {out_md} · {out_csv} · runtime {runtime:.1f}s · survivors={meta['n_survivors']}")

    df_out = pd.DataFrame([r.to_dict() for r in collapsed])
    return df_out, collapsed, meta


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@click.command()
@click.option("--beam", default=DEFAULT_BEAM, show_default=True, type=int)
@click.option("--beam-l3", default=DEFAULT_BEAM_L3, show_default=True, type=int)
@click.option("--max-length", default=DEFAULT_MAX_LENGTH, show_default=True, type=int)
@click.option("--null-n", default=DEFAULT_NULL_N, show_default=True, type=int)
@click.option("--min-trades", default=MIN_TRADES_VAL, show_default=True, type=int)
@click.option("--lambda", "lam", default=DEFAULT_LAMBDA, show_default=True, type=float)
@click.option("--lambda-l3", default=DEFAULT_LAMBDA_L3, show_default=True, type=float)
@click.option("--seed", default=42, show_default=True, type=int)
@click.option("--config", "config_path", default=None, help="Path to configs/fx.yaml")
@click.option("--quiet", is_flag=True, help="Suppress progress prints")
def main(
    beam: int,
    beam_l3: int,
    max_length: int,
    null_n: int,
    min_trades: int,
    lam: float,
    lambda_l3: float,
    seed: int,
    config_path: str | None,
    quiet: bool,
) -> None:
    """FX search v1: beam search under risk-managed FX scoring."""
    try:
        cfg = load_fx_config(config_path)
        df, ranked, meta = run_fx_search(
            beam=beam,
            beam_l3=beam_l3,
            max_length=max_length,
            null_n=null_n,
            min_trades=min_trades,
            lam=lam,
            lam_l3=lambda_l3,
            cfg=cfg,
            verbose=not quiet,
            seed=seed,
        )
        survivors = [r for r in ranked if protocol_pass_fx(r)]
        print()
        print(
            f"Done. ranked={len(ranked)} stable={meta['n_stable']} "
            f"full_bar_survivors={len(survivors)} runtime={meta['runtime_sec']:.1f}s"
        )
        print(f"Report: {PROJECT_ROOT / 'results' / 'fx_search_v1.md'}")
        if ranked:
            top = ranked[:5]
            print("Top 5 (best side per rule):")
            for i, r in enumerate(top, 1):
                print(
                    f"  {i}. [{r.side}] {r.rule}  valR={r.val_avg_R:+.3f} "
                    f"holdR={(r.holdout_avg_R if r.holdout_avg_R is not None else float('nan')):+.3f} "
                    f"PF={r.val_pf:.2f} n={r.val_n_trades} "
                    f"p={(r.null_pvalue if r.null_pvalue is not None else float('nan'))}"
                )
    except Exception as e:
        print(f"ERROR: {type(e).__name__}: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
