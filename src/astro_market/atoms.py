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


# ---------------------------------------------------------------------------
# Sweep v2 atoms (richer set; outer signs computed but flagged for search exclude)
# ---------------------------------------------------------------------------

from astro_market.ephemeris import (
    ALL_PLANETS_V2,
    ASPECTS as _ASPECTS,
    FAST_PLANETS,
    MOON_PHASE_8,
    REGIME_SIGN_PLANETS,
    moon_phase_8_from_elongation,
)

RETRO_PLANETS_V2 = [
    "mercury", "venus", "mars", "jupiter", "saturn", "uranus", "neptune", "pluto",
]
ASPECT_ORBS_V2 = [1.0, 3.0, 5.0]
DEFAULT_ATOMS_V2_CACHE = "data/atoms_v2.parquet"


def is_regime_sign_atom(name: str) -> bool:
    """True for outer-planet standalone sign atoms (multi-year regimes)."""
    core = name[1:] if name.startswith("~") else name
    for p in REGIME_SIGN_PLANETS:
        if core.startswith(f"{p}_sign_"):
            return True
    return False


def is_fast_planet_atom(name: str) -> bool:
    """True if atom involves a fast planet (Sun–Mars) as primary subject."""
    core = name[1:] if name.startswith("~") else name
    for p in FAST_PLANETS:
        if core.startswith(f"{p}_"):
            return True
        # Aspects: asp_p1_p2_... — either side fast counts as chopping potential
        if core.startswith("asp_"):
            parts = core.split("_")
            # asp_{p1}_{p2}_{aspect}_orb{N}
            if len(parts) >= 4:
                if parts[1] in FAST_PLANETS or parts[2] in FAST_PLANETS:
                    return True
    return False


def atom_registry_v2(
    planets: list[str] | None = None,
    ingress_windows: list[int] | None = None,
    aspect_orbs: list[float] | None = None,
    include_outer_signs: bool = True,
) -> list[str]:
    """Ordered list of v2 atom names."""
    planets = planets or list(ALL_PLANETS_V2)
    ingress_windows = ingress_windows or [1, 3, 5]
    aspect_orbs = aspect_orbs or list(ASPECT_ORBS_V2)
    names: list[str] = []

    for ph in MOON_PHASE_8:
        names.append(f"moon_phase_{ph}")

    # Fast planet signs (always in search)
    for p in FAST_PLANETS:
        if p not in planets:
            continue
        for s in SIGNS:
            names.append(f"{p}_sign_{s}")

    # Outer signs — computed for analysis; search excludes via is_regime_sign_atom
    if include_outer_signs:
        for p in REGIME_SIGN_PLANETS:
            if p not in planets:
                continue
            for s in SIGNS:
                names.append(f"{p}_sign_{s}")

    for p in RETRO_PLANETS_V2:
        if p in planets:
            names.append(f"{p}_retro")

    for p in RETRO_PLANETS_V2:
        if p in planets:
            names.append(f"{p}_station")

    active = [p for p in planets]
    for p1, p2 in combinations(sorted(active), 2):
        for asp in _ASPECTS:
            for orb in aspect_orbs:
                orb_tag = int(orb) if float(orb) == int(orb) else orb
                names.append(f"asp_{p1}_{p2}_{asp}_orb{orb_tag}")

    for p in planets:
        for w in ingress_windows:
            names.append(f"{p}_ingress_{w}d")

    return names


def searchable_atom_names(columns: Iterable[str]) -> list[str]:
    """Atom columns eligible for beam search (drop outer-planet sign regimes)."""
    return [c for c in columns if not is_regime_sign_atom(c)]


def _expand_bool_window(flags: np.ndarray, w: int) -> np.ndarray:
    """OR-expand boolean flags by ±w (inclusive). Vectorized."""
    n = len(flags)
    if w <= 0:
        return flags.astype(bool)
    x = flags.astype(bool)
    out = x.copy()
    for k in range(1, w + 1):
        out[k:] |= x[:-k]
        out[:-k] |= x[k:]
    return out


def _ingress_within(ingress_today: np.ndarray, w: int) -> np.ndarray:
    """True if any ingress in [i, i+w] inclusive. Vectorized forward OR."""
    n = len(ingress_today)
    x = ingress_today.astype(bool)
    out = x.copy()
    for k in range(1, w + 1):
        out[:-k] |= x[k:]
    return out


def build_atoms_v2_from_longitudes(
    longitudes: pd.DataFrame,
    *,
    aspect_orbs: list[float] | None = None,
    station_window_days: int = 3,
    ingress_windows: list[int] | None = None,
    include_outer_signs: bool = True,
) -> pd.DataFrame:
    """
    Build richer boolean atom DataFrame for sweep v2.

    Aspects are separate families per orb (1°, 3°, 5°) with orb in the atom id.
    Outer-planet signs are included when include_outer_signs=True (analysis),
    but searchable_atom_names() excludes them from the search pool.
    """
    aspect_orbs = aspect_orbs or list(ASPECT_ORBS_V2)
    ingress_windows = ingress_windows or [1, 3, 5]
    planets = [c for c in longitudes.columns if c in PLANET_BODIES]
    index = longitudes.index
    n = len(index)
    cols: dict[str, np.ndarray] = {}
    speeds = longitude_speeds(longitudes)

    # --- Moon phases (8 buckets) ---
    if "sun" in longitudes.columns and "moon" in longitudes.columns:
        phases = moon_phase_8_from_elongation(
            longitudes["sun"].to_numpy(),
            longitudes["moon"].to_numpy(),
        )
        for ph in MOON_PHASE_8:
            cols[f"moon_phase_{ph}"] = phases == ph

    # --- Signs ---
    sign_planets = list(FAST_PLANETS)
    if include_outer_signs:
        sign_planets = sign_planets + list(REGIME_SIGN_PLANETS)
    for p in sign_planets:
        if p not in longitudes.columns:
            continue
        si = sign_index(longitudes[p].to_numpy())
        for i, s in enumerate(SIGNS):
            cols[f"{p}_sign_{s}"] = si == i

    # --- Retrograde ---
    for p in RETRO_PLANETS_V2:
        if p not in longitudes.columns:
            continue
        sp = speeds[p].to_numpy()
        retro = np.zeros(n, dtype=bool)
        valid = ~np.isnan(sp)
        retro[valid] = sp[valid] < 0
        cols[f"{p}_retro"] = retro

    # --- Stations (vectorized window expand) ---
    for p in RETRO_PLANETS_V2:
        if p not in longitudes.columns:
            continue
        sp = speeds[p].to_numpy()
        near_zero = np.nan_to_num(np.abs(sp) < 0.05, nan=False).astype(bool)
        sign_sp = np.sign(np.nan_to_num(sp, nan=0.0))
        flip = np.zeros(n, dtype=bool)
        flip[1:] = (
            (sign_sp[1:] * sign_sp[:-1] < 0)
            & (sign_sp[1:] != 0)
            & (sign_sp[:-1] != 0)
        )
        cols[f"{p}_station"] = _expand_bool_window(near_zero | flip, station_window_days)

    # --- Aspects at multiple orbs ---
    for p1, p2 in combinations(sorted(planets), 2):
        sep = angular_separation(
            longitudes[p1].to_numpy(),
            longitudes[p2].to_numpy(),
        )
        for asp_name, asp_angle in _ASPECTS.items():
            diff = np.abs(sep - asp_angle)
            for orb in aspect_orbs:
                orb_tag = int(orb) if float(orb) == int(orb) else orb
                cols[f"asp_{p1}_{p2}_{asp_name}_orb{orb_tag}"] = diff <= float(orb)

    # --- Ingress ---
    for p in planets:
        si = sign_index(longitudes[p].to_numpy())
        ingress_today = np.zeros(n, dtype=bool)
        ingress_today[1:] = si[1:] != si[:-1]
        for w in ingress_windows:
            cols[f"{p}_ingress_{w}d"] = _ingress_within(ingress_today, w)

    df = pd.DataFrame(cols, index=index)
    for c in df.columns:
        df[c] = df[c].astype(bool)

    registry = atom_registry_v2(
        planets=planets,
        ingress_windows=ingress_windows,
        aspect_orbs=aspect_orbs,
        include_outer_signs=include_outer_signs,
    )
    ordered = [c for c in registry if c in df.columns]
    extras = [c for c in df.columns if c not in ordered]
    return df[ordered + extras]


def build_atoms_v2(
    dates: pd.DatetimeIndex | Iterable,
    cfg: dict | None = None,
    longitudes: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build v2 atoms for dates (skyfield unless longitudes injected)."""
    cfg = cfg or load_config()
    planets = list(ALL_PLANETS_V2)
    station_w = int(cfg["atoms"].get("station_window_days", 3))
    ingress_w = list(cfg["atoms"].get("ingress_windows", [1, 3, 5]))
    orbs = list(cfg.get("atoms_v2", {}).get("aspect_orbs", ASPECT_ORBS_V2))

    index = pd.DatetimeIndex(pd.to_datetime(list(dates))).tz_localize(None)
    if longitudes is None:
        longitudes = ecliptic_longitudes(index, planets=planets, cfg=cfg)
    else:
        longitudes = longitudes.reindex(index)

    return build_atoms_v2_from_longitudes(
        longitudes,
        aspect_orbs=orbs,
        station_window_days=station_w,
        ingress_windows=ingress_w,
        include_outer_signs=True,
    )


def atoms_v2_cache_path(cfg: dict | None = None) -> Path:
    cfg = cfg or load_config()
    raw = cfg.get("atoms_v2", {}).get("cache_path", DEFAULT_ATOMS_V2_CACHE)
    p = Path(raw)
    if not p.is_absolute():
        from astro_market.config import PROJECT_ROOT

        p = PROJECT_ROOT / p
    return p


def build_and_cache_atoms_v2(
    dates: pd.DatetimeIndex | Iterable,
    cfg: dict | None = None,
    force: bool = False,
) -> Path:
    """Build v2 atoms and write data/atoms_v2.parquet."""
    cfg = cfg or load_config()
    path = atoms_v2_cache_path(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not force:
        return path
    atoms = build_atoms_v2(dates, cfg=cfg)
    atoms.to_parquet(path)
    return path


def load_atoms_v2(
    cfg: dict | None = None,
    dates: pd.DatetimeIndex | None = None,
    auto_build: bool = True,
    force_rebuild: bool = False,
) -> pd.DataFrame:
    """Load cached v2 atoms; optionally build if missing."""
    cfg = cfg or load_config()
    path = atoms_v2_cache_path(cfg)
    if force_rebuild or not path.exists():
        if not auto_build and not path.exists():
            raise FileNotFoundError(
                f"No atom v2 cache at {path}. Run: python -m astro_market build-atoms-v2"
            )
        if dates is None:
            from astro_market.data import load_prices

            dates = load_prices(cfg=cfg, auto_fetch=True).index
        build_and_cache_atoms_v2(dates, cfg=cfg, force=force_rebuild)
    df = pd.read_parquet(path)
    df.index = pd.to_datetime(df.index).tz_localize(None)
    for c in df.columns:
        df[c] = df[c].astype(bool)
    return df.sort_index()
