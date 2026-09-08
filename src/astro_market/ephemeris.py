"""Skyfield-based geocentric tropical ephemeris helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from astro_market.config import load_config, PROJECT_ROOT

# Skyfield body names / SPICE kernels for de421
PLANET_BODIES = {
    "sun": "sun",
    "moon": "moon",
    "mercury": "mercury",
    "venus": "venus",
    "mars": "mars",
    "jupiter": "jupiter barycenter",
    "saturn": "saturn barycenter",
}

SIGNS = [
    "aries",
    "taurus",
    "gemini",
    "cancer",
    "leo",
    "virgo",
    "libra",
    "scorpio",
    "sagittarius",
    "capricorn",
    "aquarius",
    "pisces",
]

# Aspect angles (degrees) and names
ASPECTS = {
    "conj": 0.0,
    "sextile": 60.0,
    "square": 90.0,
    "trine": 120.0,
    "opp": 180.0,
}

DE421_URL = "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/a_old_versions/de421.bsp"
# Alternate: skyfield's loader can fetch from its CDN
DE421_SKYFIELD = "de421.bsp"


class EphemerisError(RuntimeError):
    """Raised when ephemeris file is missing and cannot be downloaded."""


def _bsp_path(cfg: dict | None = None) -> Path:
    cfg = cfg or load_config()
    return Path(cfg["ephemeris"]["bsp_path"])


def ensure_ephemeris(cfg: dict | None = None, force: bool = False) -> Path:
    """
    Ensure de421.bsp exists under data/. Download via skyfield loader if needed.
    """
    cfg = cfg or load_config()
    path = _bsp_path(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists() and not force:
        return path

    try:
        from skyfield.api import Loader
    except ImportError as e:
        raise EphemerisError(
            "skyfield is required. Install with: pip install skyfield"
        ) from e

    # Download into data/ directory using skyfield's Loader
    loader = Loader(str(path.parent))
    try:
        # Loader.download / open will fetch de421.bsp
        eph = loader(DE421_SKYFIELD)
        # skyfield writes to loader directory; ensure our expected path
        downloaded = path.parent / DE421_SKYFIELD
        if downloaded.exists() and downloaded.resolve() != path.resolve():
            # Copy/rename if names differ (they shouldn't for de421.bsp)
            import shutil

            shutil.copy2(downloaded, path)
        elif not path.exists() and downloaded.exists():
            downloaded.rename(path)
        # Touch path existence check
        if not path.exists():
            # Loader may have put it exactly at path if basename matches
            alt = list(path.parent.glob("de421*.bsp"))
            if alt:
                import shutil

                shutil.copy2(alt[0], path)
    except Exception as e:
        raise EphemerisError(
            f"Failed to download ephemeris to {path}: {e}. "
            "Manually place de421.bsp in data/ (from JPL or skyfield CDN)."
        ) from e

    if not path.exists():
        raise EphemerisError(
            f"Ephemeris still missing at {path} after download attempt."
        )
    return path


def _load_eph(cfg: dict | None = None):
    from skyfield.api import load

    path = ensure_ephemeris(cfg)
    return load(str(path))


def ecliptic_longitudes(
    dates: pd.DatetimeIndex | Iterable,
    planets: list[str] | None = None,
    cfg: dict | None = None,
) -> pd.DataFrame:
    """
    Geocentric apparent ecliptic longitudes (degrees, tropical approx)
    for each planet on each date. Uses skyfield ecliptic_latlon.

    Returns DataFrame indexed by date with columns = planet names,
    values in [0, 360).
    """
    from skyfield.api import load
    from skyfield.framelib import ecliptic_frame

    cfg = cfg or load_config()
    planets = planets or list(cfg["atoms"]["planets"])
    index = pd.DatetimeIndex(pd.to_datetime(list(dates))).tz_localize(None)

    eph = _load_eph(cfg)
    earth = eph["earth"]
    ts = load.timescale()

    # Noon UTC on each calendar day (stable for daily bars)
    t = ts.utc(
        index.year.to_numpy(),
        index.month.to_numpy(),
        index.day.to_numpy(),
        12,
        0,
        0,
    )

    out: dict[str, np.ndarray] = {}
    for name in planets:
        body_key = PLANET_BODIES[name]
        body = eph[body_key]
        astrometric = earth.at(t).observe(body).apparent()
        lat, lon, _ = astrometric.frame_latlon(ecliptic_frame)
        deg = lon.degrees % 360.0
        out[name] = np.asarray(deg, dtype=float)

    return pd.DataFrame(out, index=index)


def longitude_speeds(longitudes: pd.DataFrame) -> pd.DataFrame:
    """
    Daily ecliptic longitude speed (deg/day), handling 0/360 wrap.
    Positive = direct, negative = retrograde.
    """
    raw = longitudes.diff()
    # Unwrap jumps: if delta > 180, subtract 360; if < -180, add 360
    adj = raw.copy()
    adj = adj.where(adj <= 180, adj - 360)
    adj = adj.where(adj >= -180, adj + 360)
    return adj


def angular_separation(lon1: np.ndarray | pd.Series, lon2: np.ndarray | pd.Series) -> np.ndarray:
    """Smallest angle between two longitudes in [0, 180]."""
    d = np.abs(np.asarray(lon1, dtype=float) - np.asarray(lon2, dtype=float)) % 360.0
    return np.minimum(d, 360.0 - d)


def moon_phase_from_elongation(sun_lon: np.ndarray, moon_lon: np.ndarray) -> np.ndarray:
    """
    Elongation = (moon - sun) mod 360.
    Returns phase labels as object array: new, waxing, full, waning.
    """
    elong = (np.asarray(moon_lon) - np.asarray(sun_lon)) % 360.0
    # Buckets: new ~ [0,45) U [315,360); waxing [45,135); full [135,225); waning [225,315)
    phase = np.empty(len(elong), dtype=object)
    phase[(elong < 45) | (elong >= 315)] = "new"
    phase[(elong >= 45) & (elong < 135)] = "waxing"
    phase[(elong >= 135) & (elong < 225)] = "full"
    phase[(elong >= 225) & (elong < 315)] = "waning"
    return phase


def sign_index(lon: np.ndarray | pd.Series) -> np.ndarray:
    """Sign index 0..11 from ecliptic longitude."""
    return np.floor(np.asarray(lon, dtype=float) / 30.0).astype(int) % 12
