"""python -m astro_market entrypoint."""

from __future__ import annotations

import sys


def main() -> None:
    if len(sys.argv) < 2:
        print(
            "Usage:\n"
            "  python -m astro_market folklore [args...]\n"
            "  python -m astro_market null [args...]\n"
            "  python -m astro_market fetch-data\n"
            "  python -m astro_market build-atoms\n"
            "  python -m astro_market ensure-ephemeris\n"
        )
        sys.exit(1)

    cmd = sys.argv[1]
    # Shift argv so submodules see their own flags
    sys.argv = [f"astro_market.{cmd}"] + sys.argv[2:]

    if cmd == "folklore":
        from astro_market.folklore import main as folklore_main

        folklore_main()
    elif cmd == "null":
        from astro_market.null import main as null_main

        null_main()
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
    elif cmd == "ensure-ephemeris":
        from astro_market.ephemeris import ensure_ephemeris

        path = ensure_ephemeris()
        print(f"Ephemeris ready -> {path}")
    else:
        print(f"Unknown command: {cmd}")
        sys.exit(1)


if __name__ == "__main__":
    main()
