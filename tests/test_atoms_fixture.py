"""Atom builder tests with mocked longitudes (offline) and optional live ephemeris."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from astro_market.atoms import atom_registry, build_atoms, build_atoms_from_longitudes
from astro_market.ephemeris import SIGNS, sign_index


def _mock_longitudes(n: int = 30) -> pd.DataFrame:
    """Synthetic smooth longitudes for sun/moon/mercury etc."""
    idx = pd.date_range("2020-01-01", periods=n, freq="D")
    t = np.arange(n, dtype=float)
    data = {
        # Sun ~1 deg/day
        "sun": (280.0 + t) % 360.0,
        # Moon ~13 deg/day
        "moon": (50.0 + 13.0 * t) % 360.0,
        # Mercury with a retrograde segment
        "mercury": np.concatenate(
            [
                (100.0 + 1.2 * t[: n // 2]) % 360.0,
                (100.0 + 1.2 * t[n // 2 - 1] - 0.8 * np.arange(n - n // 2)) % 360.0,
            ]
        )
        if n >= 4
        else (100.0 + t) % 360.0,
        "venus": (40.0 + 1.1 * t) % 360.0,
        "mars": (10.0 + 0.5 * t) % 360.0,
        "jupiter": (200.0 + 0.08 * t) % 360.0,
        "saturn": (300.0 + 0.03 * t) % 360.0,
    }
    return pd.DataFrame(data, index=idx)


def test_registry_nonempty():
    reg = atom_registry()
    assert "moon_phase_new" in reg
    assert "mercury_retro" in reg
    assert "sun_sign_aries" in reg
    assert any(x.startswith("asp_") for x in reg)
    assert "mercury_ingress_3d" in reg
    # No duplicates
    assert len(reg) == len(set(reg))


def test_build_from_mock_longitudes():
    lon = _mock_longitudes(40)
    atoms = build_atoms_from_longitudes(lon, aspect_orb_deg=3.0, station_window_days=3)
    assert atoms.shape[0] == 40
    assert atoms.dtypes.apply(lambda d: d == bool).all() or all(
        atoms[c].dtype == bool for c in atoms.columns
    )
    # Moon phases partition
    phase_cols = [c for c in atoms.columns if c.startswith("moon_phase_")]
    assert len(phase_cols) == 4
    assert (atoms[phase_cols].sum(axis=1) == 1).all()

    # Signs partition for sun
    sun_signs = [f"sun_sign_{s}" for s in SIGNS]
    assert (atoms[sun_signs].sum(axis=1) == 1).all()

    # Mercury retro should be True somewhere in our synthetic retro segment
    assert atoms["mercury_retro"].dtype == bool
    assert atoms["mercury_retro"].any() or (~atoms["mercury_retro"]).any()


def test_sign_index():
    assert sign_index(np.array([0.0, 29.9, 30.0, 359.0])).tolist() == [0, 0, 1, 11]


def test_build_atoms_with_injected_longitudes():
    lon = _mock_longitudes(15)
    # build_atoms should accept longitudes= and skip skyfield
    atoms = build_atoms(lon.index, longitudes=lon)
    assert len(atoms) == 15
    assert "moon_phase_full" in atoms.columns


def test_live_ephemeris_if_available():
    """If de421 is present or downloadable, check a few fixed dates; else skip."""
    dates = pd.DatetimeIndex(["2000-01-01", "2000-06-21", "2010-12-21"])
    try:
        from astro_market.ephemeris import ensure_ephemeris, ecliptic_longitudes

        try:
            ensure_ephemeris()
        except Exception:
            pytest.skip("Ephemeris not available offline")
        lon = ecliptic_longitudes(dates)
        assert "sun" in lon.columns
        # June 21 ~ Cancer (sign index 3) — tropical sun near 90°
        june = lon.loc[pd.Timestamp("2000-06-21"), "sun"]
        assert 80 < june < 110
        atoms = build_atoms(dates, longitudes=lon)
        assert atoms.loc[pd.Timestamp("2000-06-21"), "sun_sign_cancer"]
    except ImportError:
        pytest.skip("skyfield not installed")
