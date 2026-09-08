"""Configuration loading for the research protocol."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

# Project root: .../astro-market-alpha (parent of src/)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "default.yaml"


def load_config(path: Path | str | None = None) -> dict[str, Any]:
    """Load YAML config; paths in config are resolved relative to project root."""
    cfg_path = Path(path) if path else DEFAULT_CONFIG_PATH
    with open(cfg_path, encoding="utf-8") as f:
        cfg: dict[str, Any] = yaml.safe_load(f)

    # Resolve relative data paths against project root
    for key_path in (
        ("asset", "cache_path"),
        ("ephemeris", "bsp_path"),
        ("atoms", "cache_path"),
    ):
        section, key = key_path
        if section in cfg and key in cfg[section]:
            p = Path(cfg[section][key])
            if not p.is_absolute():
                cfg[section][key] = str(PROJECT_ROOT / p)

    return cfg


def resolve_path(path: str | Path) -> Path:
    """Resolve a path relative to project root if not absolute."""
    p = Path(path)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    return p
