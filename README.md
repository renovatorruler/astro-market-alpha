# Astro Market Alpha

Walk-forward evaluation of **simple astrological indicator rules** versus **S&P 500 buy-and-hold**.

This is a disciplined research scaffold (v1): fixed protocol splits, transaction costs, boolean “atoms” from a tropical geocentric ephemeris, a tiny rule DSL, folklore baselines, and a circular-shift Monte Carlo null. v1 folklore + null baselines; **v1.1 adds exhaustive Boolean lattice search (length-1/2, no GA).**

## Protocol (defaults)

| Item | Default |
|------|---------|
| Asset | S&P 500 via yfinance `^GSPC` **Adj Close**, cached under `data/` |
| Train | 1970-01-01 → 1994-12-31 |
| Validation | 1995-01-01 → 2009-12-31 |
| Holdout | 2010-01-01 → present — **NEVER tune on holdout** |
| Position | Long when rule is True; **flat (0)** when False; BH = always long |
| Costs | 10 bps per side on position changes |
| Embargo | 5 trading days (internal walk-forward helpers) |
| Sharpe | Annualized, `rf=0` |
| Fitness | `sharpe_strategy − sharpe_BH − λ · complexity` (default `λ=0.01`) |

**Caveat — returns:** Adj Close is a convenient proxy. A **total-return** series (dividends reinvested) would be better for long-horizon BH comparisons; document and swap the cache if you need that rigor.

## Install

```bash
cd /workspace/astro-market-alpha
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
# parquet I/O needs pyarrow (pulled transitively on most setups; install if missing):
# pip install pyarrow
```

## Ephemeris

Uses [skyfield](https://rhodesmill.org/skyfield/) with **de421.bsp** (bundled/downloaded into `data/`).

```bash
python -m astro_market ensure-ephemeris
```

If download fails, manually place `de421.bsp` at `data/de421.bsp` (JPL NAIF or skyfield’s CDN).

Atoms (geocentric tropical approximation):

- Ecliptic longitude from skyfield; signs = `floor(lon/30)`
- Moon phase from Sun–Moon elongation
- Retrograde when longitude derivative &lt; 0 (Mercury–Saturn)
- Station if speed near zero or speed-sign flip within ±3 days
- Aspects conj/opp/square/trine/sextile with orb ≤ 3°
- Ingress flags for 1d / 3d / 5d windows

Planets: Sun, Moon, Mercury, Venus, Mars, Jupiter, Saturn.

## Commands

```bash
# 1) Fetch & cache S&P 500 adj close
python -m astro_market fetch-data

# 2) Ensure ephemeris, then build & cache boolean atoms
python -m astro_market ensure-ephemeris
python -m astro_market build-atoms

# 3) Folklore baselines (train / validation / holdout table)
python -m astro_market folklore
# or: astro-folklore

# 4) Monte Carlo null (circular-shift returns, atoms fixed) on validation
python -m astro_market null --rule moon_phase_new --n 1000
# or: astro-null --rule moon_phase_new --n 1000

# Optional: null for best length-1 atom scan
python -m astro_market null --rule moon_phase_new --n 200 --scan-length1

# 5) Boolean lattice search (length-1 + length-2 AND; validation-only selection)
python -m astro_market search --max-length 2
# defaults: --top-k 50 (L1→L2 pairing), negations on, moon OR pairs on, MC n=1000
# writes results/lattice_v1.md and results/lattice_v1.csv
```

Network failures print a clear error and exit non-zero; prefer using the parquet caches under `data/`.

## Rule DSL

Atom names combined with `&` `|` `~` and parentheses:

```text
moon_phase_new
~mercury_retro
moon_phase_new | moon_phase_full
(moon_phase_new & mercury_retro) | ~venus_retro
```

Pre-registered folklore rules:

- `moon_phase_new`
- `moon_phase_full`
- `mercury_retro`
- `~mercury_retro`
- `moon_phase_new | moon_phase_full`

## Pass / fail criteria (illustrative)

These are **research heuristics**, not investment advice:

1. **Validation first:** a rule is interesting if `excess_sharpe > 0` and `fitness > 0` on **validation** (1995–2009).
2. **Null test:** under circular-shift of returns (atoms fixed), validation fitness should have a small empirical p-value (e.g. &lt; 0.05) to reject the null that timing is exchangeable.
3. **Holdout confirmation only:** report holdout metrics once; do **not** select or tune rules using holdout.
4. **Complexity:** prefer lower `complexity` (fitness already penalizes it).
5. **Costs:** results must use the default 10 bps/side (or explicitly higher) — zero-cost wins are not counted.

Lattice search (`search`) enumerates length-1/2 Boolean rules only — **no genetic algorithm**.

## Tests (offline)

```bash
pytest -q
```

- Rule parsing / evaluation
- `evaluate_rule` on synthetic returns & atoms (no network)
- Atom builder from mocked longitudes; live ephemeris test skipped if `de421.bsp` unavailable

## Layout

```text
astro-market-alpha/
  pyproject.toml
  README.md
  configs/default.yaml
  src/astro_market/
    __init__.py
    __main__.py
    data.py          # yfinance fetch + cache
    ephemeris.py     # skyfield longitudes, phases, aspects helpers
    atoms.py         # build_atoms + registry
    rules.py         # DSL parser + evaluator
    evaluate.py      # metrics, fitness, embargo helper
    folklore.py      # folklore CLI
    null.py          # Monte Carlo null CLI
    search.py        # Boolean lattice search (L1/L2) + MC
    config.py        # YAML loader
  tests/
    test_rules.py
    test_evaluate.py
    test_atoms_fixture.py
    test_search.py
  data/              # prices, atoms, de421.bsp caches
  results/           # lattice_v1.md / lattice_v1.csv
```

## Modules (API sketch)

| Module | Role |
|--------|------|
| `config` | Load `configs/default.yaml`, resolve paths |
| `data` | `fetch_and_cache_prices`, `load_prices`, `daily_returns` |
| `ephemeris` | `ensure_ephemeris`, `ecliptic_longitudes`, phase/sign/speed helpers |
| `atoms` | `build_atoms`, `atom_registry`, parquet cache |
| `rules` | `parse_rule`, `evaluate_ast`, `rule_complexity` |
| `evaluate` | `evaluate_rule` → metrics; `fitness`; `embargo_split` |
| `folklore` | CLI table over protocol splits |
| `null` | Circular-shift null + optional best length-1 scan hook |
| `search` | Length-1/2 Boolean lattice; val-only selection; holdout report; MC |

## Caveats

- **yfinance** requires network on first fetch; thereafter use `data/gspc_adj_close.parquet`.
- **de421.bsp** requires network once (or a manual copy into `data/`).
- Adj Close ≠ total return; dividends ignored.
- Tropical geocentric longitudes are an approximation suitable for folklore tests, not chart-service fidelity.
- Flat (cash) earns 0 in this model (no rf on cash).
