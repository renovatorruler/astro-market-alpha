"""python -m astro_market entrypoint."""

from __future__ import annotations

import sys


def _pop_version_flag(argv: list[str]) -> tuple[list[str], int | None]:
    """Remove --version N or --version=N from argv; return (new_argv, version|None)."""
    out: list[str] = []
    version: int | None = None
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--version" and i + 1 < len(argv):
            try:
                version = int(argv[i + 1])
            except ValueError:
                out.append(a)
                i += 1
                continue
            i += 2
            continue
        if a.startswith("--version="):
            try:
                version = int(a.split("=", 1)[1])
            except ValueError:
                out.append(a)
            i += 1
            continue
        out.append(a)
        i += 1
    return out, version


def main() -> None:
    if len(sys.argv) < 2:
        print(
            "Usage:\n"
            "  python -m astro_market folklore [args...]\n"
            "  python -m astro_market null [args...]\n"
            "  python -m astro_market search [args...]\n"
            "  python -m astro_market sweep [--version 2|3] [args...]\n"
            "  python -m astro_market sweep-v3 [args...]\n"
            "  python -m astro_market fetch-data\n"
            "  python -m astro_market build-atoms\n"
            "  python -m astro_market build-atoms-v2\n"
            "  python -m astro_market ensure-ephemeris\n"
            "  python -m astro_market fetch-fx\n"
            "  python -m astro_market fx-folklore [args...]\n"
            "  python -m astro_market fx-sim --rule ... [args...]\n"
        )
        sys.exit(1)

    cmd = sys.argv[1]
    rest = sys.argv[2:]

    if cmd in ("sweep", "sweep-v3"):
        rest, ver = _pop_version_flag(rest)
        version = 3 if cmd == "sweep-v3" else (ver if ver is not None else 2)
        sys.argv = [f"astro_market.{cmd}"] + rest
        if version >= 3:
            from astro_market.sweep_v3 import main as sweep_main
        else:
            from astro_market.sweep import main as sweep_main
        sweep_main()
        return

    # Shift argv so submodules see their own flags
    sys.argv = [f"astro_market.{cmd}"] + rest

    if cmd == "folklore":
        from astro_market.folklore import main as folklore_main

        folklore_main()
    elif cmd == "null":
        from astro_market.null import main as null_main

        null_main()
    elif cmd == "search":
        from astro_market.search import main as search_main

        search_main()
    elif cmd == "fetch-data":
        from astro_market.data import fetch_and_cache_prices

        path = fetch_and_cache_prices()
        print(f"Cached prices -> {path}")
    elif cmd == "build-atoms":
        from astro_market.atoms import build_and_cache_atoms
        from astro_market.data import load_prices

        prices = load_prices()
        path = build_and_cache_atoms(prices.index)
        print(f"Cached atoms -> {path}")
    elif cmd == "build-atoms-v2":
        from astro_market.atoms import build_and_cache_atoms_v2
        from astro_market.data import load_prices

        prices = load_prices()
        path = build_and_cache_atoms_v2(prices.index, force=True)
        print(f"Cached atoms v2 -> {path} ({__import__('pandas').read_parquet(path).shape[1]} cols)")
    elif cmd == "ensure-ephemeris":
        from astro_market.ephemeris import ensure_ephemeris

        path = ensure_ephemeris()
        print(f"Ephemeris ready -> {path}")
    elif cmd == "fetch-fx":
        from astro_market.fx.data import fetch_all_fx, load_fx_config

        cfg = load_fx_config()
        start = cfg.get("fx", {}).get("fetch_start", "2000-01-01")
        paths = fetch_all_fx(start=start, cfg=cfg)
        for pair, path in paths.items():
            print(f"Cached FX {pair} -> {path}")
    elif cmd == "fx-folklore":
        from astro_market.fx.folklore import main as fx_folklore_main

        fx_folklore_main()
    elif cmd == "fx-sim":
        from astro_market.fx.folklore import sim_main

        sim_main()
    else:
        print(f"Unknown command: {cmd}")
        sys.exit(1)


if __name__ == "__main__":
    main()
