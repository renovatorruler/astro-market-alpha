"""Boolean atom features from ephemeris longitudes."""

from __future__ import annotations

from itertools import combinations
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from astro_market.config import load_config
from astro_market.ephemeris import (
    ASPECTS,
    PLANET_BODIES,
    SIGNS,
    angular_separation,
    ecliptic_longitudes,
    longitude_speeds,
    moon_phase_from_elongation,
    sign_index,
)

# Planets that can go retrograde (exclude Sun, Moon)
RETRO_PLANETS = ["mercury", "venus", "mars", "jupiter", "saturn"]


def atom_registry(
    planets: list[str] | None = None,
    ingress_windows: list[int] | None = None,
) -> list[str]:
    """Return the ordered list of all atom names produced by build_atoms."""
    planets = planets or list(PLANET_BODIES.keys())
    ingress_windows = ingress_windows or [1, 3, 5]
    names: list[str] = []

    # Moon phases
    for ph in ("new", "waxing", "full", "waning"):
        names.append(f"moon_phase_{ph}")

    # Sign placements
    for p in planets:
        for s in SIGNS:
            names.append(f"{p}_sign_{s}")

    # Retrograde
    for p in RETRO_PLANETS:
        if p in planets:
            names.append(f"{p}_retro")

    # Stations
    for p in RETRO_PLANETS:
        if p in planets:
            names.append(f"{p}_station")

    # Aspects between pairs
    active = [p for p in planets]
    for p1, p2 in combinations(sorted(active), 2):
        for asp in ASPECTS:
            names.append(f"asp_{p1}_{p2}_{asp}")

    # Ingress windows
    for p in planets:
        for w in ingress_windows:
            names.append(f"{p}_ingress_{w}d")

    return names


def build_atoms_from_longitudes(
    longitudes: pd.DataFrame,
    aspect_orb_deg: float = 3.0,
    station_window_days: int = 3,
    ingress_windows: list[int] | None = None,
) -> pd.DataFrame:
    """
    Build boolean atom DataFrame from a longitude DataFrame
    (columns = planet names, index = dates, values = ecliptic lon deg).

    Useful for unit tests with mocked longitudes (no skyfield needed).
    """
    ingress_windows = ingress_windows or [1, 3, 5]
    planets = [c for c in longitudes.columns if c in PLANET_BODIES]
    index = longitudes.index
    n = len(index)
    cols: dict[str, np.ndarray] = {}

    speeds = longitude_speeds(longitudes)

    # --- Moon phases ---
    if "sun" in longitudes.columns and "moon" in longitudes.columns:
        phases = moon_phase_from_elongation(
            longitudes["sun"].to_numpy(),
            longitudes["moon"].to_numpy(),
        )
        for ph in ("new", "waxing", "full", "waning"):
            cols[f"moon_phase_{ph}"] = phases == ph

    # --- Signs ---
    for p in planets:
        si = sign_index(longitudes[p].to_numpy())
        for i, s in enumerate(SIGNS):
            cols[f"{p}_sign_{s}"] = si == i

    # --- Retrograde ---
    for p in RETRO_PLANETS:
        if p not in longitudes.columns:
            continue
        sp = speeds[p].to_numpy()
        # First day NaN speed -> treat as not retro
        retro = np.zeros(n, dtype=bool)
        valid = ~np.isnan(sp)
        retro[valid] = sp[valid] < 0
        cols[f"{p}_retro"] = retro

    # --- Stations: speed near zero OR sign-of-speed flip within ±station_window_days ---
    for p in RETRO_PLANETS:
        if p not in longitudes.columns:
            continue
        sp = speeds[p].to_numpy()
        station = np.zeros(n, dtype=bool)
        # Near-zero speed threshold (~0.05 deg/day heuristic)
        near_zero = np.abs(sp) < 0.05
        near_zero = np.nan_to_num(near_zero.astype(float), nan=0.0).astype(bool)

        # Sign flip of speed
        sign_sp = np.sign(np.nan_to_num(sp, nan=0.0))
        flip = np.zeros(n, dtype=bool)
        flip[1:] = (sign_sp[1:] * sign_sp[:-1] < 0) & (sign_sp[1:] != 0) & (sign_sp[:-1] != 0)

        # Expand flip marks by ±window
        w = station_window_days
        expanded = near_zero | flip
        if w > 0:
            # pad and OR over ±w window
            pad = np.pad(expanded.astype(bool), (w, w), constant_values=False)
            for i in range(n):
                if pad[i : i + 2 * w + 1].any():
                    station[i] = True
        else:
            station = expanded
        cols[f"{p}_station"] = station

    # --- Aspects ---
    for p1, p2 in combinations(sorted(planets), 2):
        sep = angular_separation(
            longitudes[p1].to_numpy(),
            longitudes[p2].to_numpy(),
        )
        for asp_name, asp_angle in ASPECTS.items():
            cols[f"asp_{p1}_{p2}_{asp_name}"] = np.abs(sep - asp_angle) <= aspect_orb_deg

    # --- Ingress: sign change within next w trading days (looking ahead from today? or including day of?)
    # Spec: {planet}_ingress_{1d,3d,5d} — True if an ingress occurs within the next w days
    # (including today if sign changes from previous day).
    for p in planets:
        si = sign_index(longitudes[p].to_numpy())
        ingress_today = np.zeros(n, dtype=bool)
        ingress_today[1:] = si[1:] != si[:-1]
        for w in ingress_windows:
            flag = np.zeros(n, dtype=bool)
            for i in range(n):
                # True if any ingress in [i, i+w] (w calendar/trading steps ahead inclusive of today)
                end = min(n, i + w + 1)
                if ingress_today[i:end].any():
                    flag[i] = True
            cols[f"{p}_ingress_{w}d"] = flag

    df = pd.DataFrame(cols, index=index)
    # Ensure boolean dtype
    for c in df.columns:
        df[c] = df[c].astype(bool)
    # Reorder to registry order where possible
    registry = atom_registry(planets=planets, ingress_windows=ingress_windows)
    ordered = [c for c in registry if c in df.columns]
    extras = [c for c in df.columns if c not in ordered]
    return df[ordered + extras]


def build_atoms(
    dates: pd.DatetimeIndex | Iterable,
    cfg: dict | None = None,
    longitudes: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Build boolean atom DataFrame for the given dates.

    If `longitudes` is provided, skip ephemeris (for tests).
    Otherwise compute geocentric tropical longitudes via skyfield.
    """
    cfg = cfg or load_config()
    planets = list(cfg["atoms"]["planets"])
    orb = float(cfg["atoms"]["aspect_orb_deg"])
    station_w = int(cfg["atoms"]["station_window_days"])
    ingress_w = list(cfg["atoms"]["ingress_windows"])

    index = pd.DatetimeIndex(pd.to_datetime(list(dates))).tz_localize(None)

    if longitudes is None:
        longitudes = ecliptic_longitudes(index, planets=planets, cfg=cfg)
    else:
        longitudes = longitudes.reindex(index)

    return build_atoms_from_longitudes(
        longitudes,
        aspect_orb_deg=orb,
        station_window_days=station_w,
        ingress_windows=ingress_w,
    )


def build_and_cache_atoms(
    dates: pd.DatetimeIndex | Iterable,
    cfg: dict | None = None,
    force: bool = False,
) -> Path:
    """Build atoms for dates and write parquet cache."""
    cfg = cfg or load_config()
    path = Path(cfg["atoms"]["cache_path"])
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists() and not force:
        return path

    atoms = build_atoms(dates, cfg=cfg)
    atoms.to_parquet(path)
    return path


def load_atoms(
    cfg: dict | None = None,
    dates: pd.DatetimeIndex | None = None,
    auto_build: bool = True,
) -> pd.DataFrame:
    """Load cached atoms; optionally build if missing (needs ephemeris + dates)."""
    cfg = cfg or load_config()
    path = Path(cfg["atoms"]["cache_path"])

    if not path.exists():
        if not auto_build:
            raise FileNotFoundError(
                f"No atom cache at {path}. Run: python -m astro_market build-atoms"
            )
        if dates is None:
            from astro_market.data import load_prices

            dates = load_prices(cfg=cfg, auto_fetch=True).index
        build_and_cache_atoms(dates, cfg=cfg)

    df = pd.read_parquet(path)
    df.index = pd.to_datetime(df.index).tz_localize(None)
    for c in df.columns:
        df[c] = df[c].astype(bool)
    return df.sort_index()
