# FX search v1 — risk-managed beam search

## Method

Boolean astro rules scored with the **FX risk-unit engine** (rising-edge entries, fixed fractional risk, SL/TP) — **not** the equity long/flat model.

- Atoms: atoms_v2 searchable pool (outer-planet *sign* regimes excluded); total=849 searchable=789
- Beam: L1→L2→L3 · beam=150 · L3_beam=80 · max_length=3
- Fitness (validation): `avg_R − λ·complexity − trade_penalty` (λ=0.02, λ_L3=0.03, min_trades=30)
- Sides: **long** and **short** (short = rising edge of `~rule`) searched separately; tables show best side per rule where collapsed.
- Defaults from `configs/fx.yaml`: risk 0.5%, SL 1.5×ATR(14), TP 2R, time-stop 10, spread 1 pip, max concurrent 5.
- Split-val stability: avg R > 0 on **both** temporal halves of validation.
- Holdout **report only — never used for selection**.
- Monte Carlo null: circular-shift each pair's OHLC+ATR together (atoms fixed), n=2000, seed=42, score=val avg R.

### Protocol splits (FX)

| Split | Dates | Role |
|-------|-------|------|
| Train | 2000-01-01 → 2014-12-31 | Research (truncated by data start) |
| Validation | 2015-01-01 → 2019-12-31 | Selection |
| Holdout | 2020-01-01 → present | Report only |

Runtime: **92.1s** (L1 6.1s · L2 12.9s · L3 19.2s · MC 36.5s) · cores=8 · ranked=380 · stable=345 · data 2000-01-03 → 2026-09-09

## Verdict

**14 rules pass the raw full-bar checklist** (val avg R>0 both halves, ≥30 val trades, null p<0.05, holdout avg R>0).

After collapsing **near-duplicate L3 paddings** (extra AND-negations that do not change validation trades/avg R vs a shorter core), **7 unique cores** remain:

- `~mercury_sign_aquarius` [short] val_avg_R=+0.509 hold_avg_R=+0.095 p=0.0030 trades=30/45 PF=2.58
- `asp_mars_mercury_conj_orb5 & ~asp_mars_neptune_trine_orb3` [long] val_avg_R=+0.599 hold_avg_R=+0.169 p=0.0015 trades=30/60 PF=3.23
- `asp_mercury_uranus_square_orb5 & mercury_sign_capricorn` [short] val_avg_R=+0.549 hold_avg_R=+0.395 p=0.0025 trades=30/5 PF=2.81
- `asp_mercury_uranus_square_orb5 & ~pluto_retro` [short] val_avg_R=+0.563 hold_avg_R=+0.183 p=0.0035 trades=30/45 PF=2.93
- `~asp_jupiter_sun_opp_orb5 & asp_mars_mercury_conj_orb5` [long] val_avg_R=+0.672 hold_avg_R=+0.218 p=0.0010 trades=35/55 PF=3.73
- `~asp_jupiter_sun_opp_orb5 & asp_mars_mercury_conj_orb5 & ~asp_mercury_uranus_trine_orb1` [long] val_avg_R=+0.653 hold_avg_R=+0.218 p=0.0015 trades=39/55 PF=3.39
- `~asp_jupiter_sun_opp_orb5 & asp_mars_mercury_conj_orb5 & ~asp_neptune_sun_opp_orb3` [long] val_avg_R=+0.653 hold_avg_R=+0.218 p=0.0010 trades=39/55 PF=3.39

**Honesty / multiple testing:** Beam search scored thousands of (rule, side) candidates (L1≈2.3k kept paths, L2≈9k, L3≈17k scored). A raw null p<0.05 on ~35 MC targets does **not** survive Bonferroni across the search. Several “survivors” are cosmetic L3 clones of one L2 core (`~asp_jupiter_sun_opp_orb5 & asp_mars_mercury_conj_orb5`). Trade counts are modest (≈30–40 on val). Treat as **exploratory** — not deployable alpha — pending pre-registered replication.
Also: `asp_mercury_uranus_square_orb5 & mercury_sign_capricorn` [short] has holdout **n_trades=5** — holdout avg R is not trustworthy at that sample size despite clearing the literal checklist.


Folklore moon / mercury-retro rules are **baselines, not winners** (mixed or flat holdout). Note: folklore entry `~mercury_sign_aquarius` [short] does appear among unique cores; that is a single length-1 atom already in the folklore list, not a novel compound discovery.

## Top rules by validation fitness

| Rank | Rule | Side | L | Val avgR | H1 | H2 | Stable | Val PF | Trades | Hold avgR | Hold PF | Null p |
|-----:|------|:----:|--:|---------:|---:|---:|:------:|-------:|-------:|----------:|--------:|-------:|
| 1 | `~asp_jupiter_sun_opp_orb5 & asp_mars_mercury_conj_orb5` | long | 2 | +0.672 | +0.699 | +0.652 | Y | 3.73 | 35 | +0.218 | 1.70 | 0.0010 |
| 2 | `asp_mars_mercury_conj_orb5 & ~asp_mars_neptune_trine_orb3` | long | 2 | +0.599 | +0.670 | +0.542 | Y | 3.23 | 30 | +0.169 | 1.48 | 0.0015 |
| 3 | `asp_mercury_pluto_square_orb3 & mercury_sign_aries` | short | 2 | +0.589 | +0.561 | +0.602 | Y | 4.66 | 25 | -0.218 | 0.47 | 0.0030 |
| 4 | `~asp_jupiter_mercury_opp_orb5 & asp_jupiter_venus_opp_orb5` | short | 2 | +0.591 | +0.576 | +0.581 | Y | 3.75 | 25 | +0.085 | 1.20 | 0.0005 |
| 5 | `asp_mercury_pluto_square_orb3 & ~mercury_sign_libra` | short | 2 | +0.589 | +0.561 | +0.602 | Y | 4.66 | 25 | -0.163 | 0.63 | 0.0010 |
| 6 | `~asp_jupiter_sun_opp_orb5 & asp_mars_mercury_conj_orb5 & ~asp_mars_pluto_trine_orb1` | long | 3 | +0.672 | +0.699 | +0.652 | Y | 3.73 | 35 | +0.218 | 1.70 | 0.0010 |
| 7 | `~asp_jupiter_sun_opp_orb5 & asp_mars_mercury_conj_orb5 & ~asp_neptune_sun_sextile_orb5` | long | 3 | +0.672 | +0.699 | +0.652 | Y | 3.73 | 35 | +0.131 | 1.39 | 0.0015 |
| 8 | `~asp_jupiter_sun_opp_orb5 & asp_mars_mercury_conj_orb5 & ~asp_moon_uranus_opp_orb1` | long | 3 | +0.672 | +0.699 | +0.652 | Y | 3.73 | 35 | +0.218 | 1.70 | 0.0015 |
| 9 | `~asp_jupiter_sun_opp_orb5 & asp_mars_mercury_conj_orb5 & ~asp_saturn_sun_opp_orb3` | long | 3 | +0.672 | +0.699 | +0.652 | Y | 3.73 | 35 | +0.218 | 1.70 | 0.0025 |
| 10 | `~asp_jupiter_sun_opp_orb5 & asp_mars_mercury_conj_orb5 & ~asp_uranus_venus_trine_orb5` | long | 3 | +0.672 | +0.699 | +0.652 | Y | 3.73 | 35 | +0.215 | 1.64 | 0.0010 |
| 11 | `~asp_jupiter_sun_opp_orb5 & asp_mars_mercury_conj_orb5 & ~asp_moon_neptune_trine_orb1` | long | 3 | +0.672 | +0.699 | +0.652 | Y | 3.73 | 35 | +0.218 | 1.70 | 0.0010 |
| 12 | `~asp_jupiter_sun_opp_orb5 & asp_mars_mercury_conj_orb5 & ~mercury_sign_aquarius` | long | 3 | +0.672 | +0.699 | +0.652 | Y | 3.73 | 35 | +0.218 | 1.70 | 0.0015 |
| 13 | `asp_moon_neptune_sextile_orb1 & ~pluto_retro` | short | 2 | +0.545 | +0.627 | +0.422 | Y | 2.56 | 40 | -0.105 | 0.77 | 0.0005 |
| 14 | `asp_mercury_uranus_square_orb5 & mercury_sign_capricorn` | short | 2 | +0.549 | +0.546 | +0.479 | Y | 2.81 | 30 | +0.395 | 1.99 | 0.0025 |
| 15 | `~asp_jupiter_sun_opp_orb5 & asp_mars_mercury_conj_orb5 & ~asp_neptune_sun_opp_orb3` | long | 3 | +0.653 | +0.699 | +0.620 | Y | 3.39 | 39 | +0.218 | 1.70 | 0.0010 |
| 16 | `~asp_jupiter_sun_opp_orb5 & asp_mars_mercury_conj_orb5 & ~asp_mercury_uranus_trine_orb1` | long | 3 | +0.653 | +0.699 | +0.620 | Y | 3.39 | 39 | +0.218 | 1.70 | 0.0015 |
| 17 | `~mercury_sign_aquarius` | short | 1 | +0.509 | +0.513 | +0.479 | Y | 2.58 | 30 | +0.095 | 1.20 | 0.0030 |
| 18 | `asp_mars_saturn_trine_orb5 & ~venus_sign_aries` | short | 2 | +0.548 | +0.629 | +0.451 | Y | 2.92 | 25 | -0.185 | 0.61 | 0.0030 |
| 19 | `asp_mercury_uranus_square_orb5 & ~pluto_retro` | short | 2 | +0.563 | +0.546 | +0.520 | Y | 2.93 | 30 | +0.183 | 1.55 | 0.0035 |
| 20 | `~asp_mars_pluto_trine_orb1 & asp_mars_saturn_trine_orb5` | short | 2 | +0.534 | +0.401 | +0.686 | Y | 2.63 | 31 | -0.188 | 0.62 | 0.0045 |
| 21 | `mercury_sign_capricorn & ~venus_sign_sagittarius` | short | 2 | +0.540 | +0.428 | +1.052 | N | 2.77 | 30 | +0.096 | 1.20 | 0.0015 |
| 22 | `~asp_mars_pluto_trine_orb1 & asp_mars_saturn_trine_orb5 & ~venus_sign_aries` | short | 3 | +0.689 | +0.629 | +0.686 | Y | 3.82 | 26 | -0.185 | 0.61 | 0.0010 |
| 23 | `~asp_neptune_sun_sextile_orb5 & mercury_sign_capricorn` | short | 2 | +0.636 | +0.540 | +0.479 | Y | 3.38 | 37 | -0.006 | 0.98 | 0.0010 |
| 24 | `asp_moon_neptune_sextile_orb1 & ~asp_sun_venus_conj_orb5 & ~pluto_retro` | short | 3 | +0.676 | +0.627 | +0.699 | Y | 3.27 | 30 | -0.054 | 0.87 | 0.0005 |
| 25 | `~asp_jupiter_venus_sextile_orb3 & asp_mars_saturn_trine_orb5` | short | 2 | +0.523 | +0.401 | +0.673 | Y | 2.54 | 30 | -0.201 | 0.60 | 0.0060 |

## Folklore baselines (not search winners)

| Rule | Side | Val avgR | Val trades | Hold avgR | Hold trades |
|------|:----:|---------:|-----------:|----------:|------------:|
| `moon_phase_new` | long | -0.035 | 315 | +0.066 | 412 |
| `moon_phase_new` | short | +0.066 | 315 | -0.117 | 415 |
| `moon_phase_full` | long | +0.054 | 310 | +0.008 | 415 |
| `moon_phase_full` | short | -0.020 | 310 | -0.007 | 415 |
| `mercury_retro` | long | -0.111 | 80 | -0.040 | 105 |
| `mercury_retro` | short | -0.114 | 80 | +0.091 | 105 |
| `~mercury_sign_aquarius` | long | +0.110 | 30 | -0.120 | 45 |
| `~mercury_sign_aquarius` | short | +0.509 | 30 | +0.095 | 45 |

## Notes

- Beam ranking used a fast concurrent-capped R simulator; reported val/hold Sharpe, max DD, and final avg R/PF for survivors use full `FxBrokerSim`.
- Equity chart for the top rule (if generated): `results/fx_search_v1_top_equity.png`.
