# Sweep v2 — Beam search L1→L2→L3 with regime filters

## Method

Richer tropical geocentric atoms (Uranus/Neptune/Pluto for aspects/stations/ingress; multi-orb aspect families 1°/3°/5°; 8 moon-phase buckets). **Outer-planet sign atoms** (Jupiter–Pluto) are computed for analysis but **excluded from the searchable pool** to avoid Saturn-Pisces-style multi-year regime artifacts.

Beam search: L1 (atoms+negations) → top `200` by validation fitness (after regime filter) → L2 AND pairs (+ moon ORs) → L3 = L2 ⊕ one L1 literal. Complexity λ=0.01 (L1/L2), λ_L3=0.015.

**Regime / episode filters:**

- Hard drop rules with a single validation True episode longer than **400** trading days
- Soft penalty (−0.50 fitness) if episodes < 4 on val **and** < 8 on train+val
- Hard drop pure outer-planet sign rules
- **Stable survivor**: positive excess Sharpe on **both** temporal halves of validation

**Frozen protocol:**

| Item | Value |
|------|-------|
| Train | 1970-01-01 → 1994-12-31 |
| Validation | 1995-01-01 → 2009-12-31 |
| Holdout | 2010-01-01 → present (**never used for selection**) |
| Position | Long when True; flat when False; BH = always long |
| Costs | 10.0 bps / side |
| Monte Carlo | Circular-shift, n=5000, seed=42, validation |

Runtime: **43.5s** (atoms 0.1s · L1 0.6s · L2 13.1s · L3 19.8s · MC 9.5s) · cores=8 · atoms total/searchable=849/789 · ranked=600 · stable=61 · days train/val/hold=6318/3778/4195

## Top rules by validation fitness (all val-passers pool)

| Rank | Rule | L | Ep | MaxEp | Val XS | Val Fit | H1 XS | H2 XS | Stable | Hold XS | Null p |
|-----:|------|--:|---:|------:|-------:|--------:|------:|------:|:------:|--------:|-------:|
| 1 | `~asp_mercury_uranus_square_orb5 & asp_pluto_uranus_sextile_orb3 & ~mars_ingress_3d` | 3 | 30 | 50 | +0.669 | +0.564 | +0.875 | -0.204 | N | -0.755 | 0.0006 |
| 2 | `~asp_moon_saturn_trine_orb1 & asp_pluto_uranus_sextile_orb3 & ~mars_ingress_3d` | 3 | 33 | 76 | +0.661 | +0.556 | +0.864 | -0.204 | N | -0.755 | 0.0006 |
| 3 | `~asp_moon_saturn_trine_orb1 & asp_pluto_uranus_sextile_orb3 & ~sun_sign_cancer` | 3 | 14 | 128 | +0.655 | +0.550 | +0.855 | -0.204 | N | -0.755 | 0.0010 |
| 4 | `~asp_mercury_uranus_square_orb5 & asp_pluto_uranus_sextile_orb3 & ~sun_sign_cancer` | 3 | 12 | 106 | +0.653 | +0.548 | +0.852 | -0.204 | N | -0.755 | 0.0002 |
| 5 | `~asp_moon_saturn_trine_orb1 & asp_pluto_uranus_sextile_orb3 & ~mercury_sign_cancer` | 3 | 14 | 133 | +0.652 | +0.547 | +0.851 | -0.204 | N | -0.755 | 0.0008 |
| 6 | `~asp_mercury_uranus_square_orb5 & asp_pluto_uranus_sextile_orb3 & ~mars_ingress_5d` | 3 | 30 | 50 | +0.650 | +0.545 | +0.848 | -0.204 | N | -0.755 | 0.0020 |
| 7 | `~asp_mercury_uranus_square_orb5 & asp_pluto_uranus_sextile_orb3 & ~mercury_sign_cancer` | 3 | 12 | 106 | +0.650 | +0.545 | +0.847 | -0.204 | N | -0.755 | 0.0012 |
| 8 | `~asp_moon_saturn_trine_orb1 & asp_pluto_uranus_sextile_orb3 & ~venus_sign_aries` | 3 | 14 | 150 | +0.648 | +0.543 | +0.844 | -0.204 | N | -0.755 | 0.0014 |
| 9 | `~asp_jupiter_sun_sextile_orb5 & ~asp_moon_saturn_trine_orb1 & asp_pluto_uranus_sextile_orb3` | 3 | 17 | 164 | +0.646 | +0.541 | +0.842 | -0.204 | N | -0.755 | 0.0032 |
| 10 | `~asp_moon_saturn_trine_orb1 & asp_pluto_uranus_sextile_orb3 & ~mars_ingress_5d` | 3 | 32 | 74 | +0.644 | +0.539 | +0.840 | -0.204 | N | -0.755 | 0.0008 |
| 11 | `~asp_moon_saturn_trine_orb1 & asp_pluto_uranus_sextile_orb3` | 2 | 12 | 184 | +0.575 | +0.535 | +0.742 | -0.204 | N | -0.755 | — |
| 12 | `~asp_mercury_uranus_square_orb5 & asp_pluto_uranus_sextile_orb3` | 2 | 10 | 136 | +0.571 | +0.531 | +0.736 | -0.204 | N | -0.755 | — |
| 13 | `~asp_mercury_uranus_square_orb5 & asp_pluto_uranus_sextile_orb3 & ~asp_uranus_venus_sextile_orb5` | 3 | 15 | 136 | +0.636 | +0.531 | +0.827 | -0.204 | N | -0.755 | — |
| 14 | `~asp_mercury_uranus_square_orb1 & ~asp_moon_saturn_trine_orb1 & asp_pluto_uranus_sextile_orb3` | 3 | 16 | 150 | +0.634 | +0.529 | +0.825 | -0.204 | N | -0.755 | — |
| 15 | `~asp_mercury_uranus_square_orb3 & ~asp_moon_saturn_trine_orb1 & asp_pluto_uranus_sextile_orb3` | 3 | 17 | 132 | +0.632 | +0.527 | +0.822 | -0.204 | N | -0.755 | — |
| 16 | `~asp_moon_saturn_trine_orb1 & asp_pluto_uranus_sextile_orb3 & ~asp_uranus_venus_sextile_orb5` | 3 | 19 | 126 | +0.625 | +0.520 | +0.812 | -0.204 | N | -0.755 | — |
| 17 | `~asp_jupiter_moon_square_orb1 & ~asp_moon_saturn_trine_orb1 & asp_pluto_uranus_sextile_orb3` | 3 | 23 | 151 | +0.621 | +0.516 | +0.807 | -0.204 | N | -0.755 | — |
| 18 | `~asp_mars_neptune_square_orb5 & ~asp_moon_saturn_trine_orb1 & asp_pluto_uranus_sextile_orb3` | 3 | 13 | 150 | +0.620 | +0.515 | +0.805 | -0.204 | N | -0.755 | — |
| 19 | `~asp_mercury_uranus_square_orb5 & asp_pluto_uranus_sextile_orb3 & ~asp_saturn_sun_sextile_orb3` | 3 | 15 | 115 | +0.620 | +0.515 | +0.805 | -0.204 | N | -0.755 | — |
| 20 | `~asp_jupiter_moon_square_orb1 & ~asp_mercury_uranus_square_orb5 & asp_pluto_uranus_sextile_orb3` | 3 | 19 | 98 | +0.620 | +0.515 | +0.804 | -0.204 | N | -0.755 | — |
| 21 | `~asp_moon_saturn_trine_orb1 & asp_pluto_uranus_sextile_orb3 & ~asp_pluto_venus_opp_orb3` | 3 | 14 | 150 | +0.619 | +0.514 | +0.804 | -0.204 | N | -0.755 | — |
| 22 | `~asp_mercury_uranus_square_orb5 & asp_pluto_uranus_sextile_orb3 & ~asp_pluto_venus_sextile_orb1` | 3 | 15 | 136 | +0.618 | +0.513 | +0.802 | -0.204 | N | -0.755 | — |
| 23 | `~asp_jupiter_sun_sextile_orb5 & ~asp_mercury_uranus_square_orb5 & asp_pluto_uranus_sextile_orb3` | 3 | 14 | 136 | +0.615 | +0.510 | +0.798 | -0.204 | N | -0.755 | — |
| 24 | `~asp_mars_neptune_trine_orb1 & ~asp_moon_saturn_trine_orb1 & asp_pluto_uranus_sextile_orb3` | 3 | 16 | 129 | +0.615 | +0.510 | +0.798 | -0.204 | N | -0.755 | — |
| 25 | `~asp_mercury_uranus_square_orb5 & asp_pluto_uranus_sextile_orb3 & ~venus_sign_aries` | 3 | 12 | 136 | +0.615 | +0.510 | +0.798 | -0.204 | N | -0.755 | — |

## Stable survivors (XS > 0 in both validation halves)

| Rank | Rule | L | Val Fit | Val XS | H1 | H2 | Hold XS | Null p | Bar+HO |
|-----:|------|--:|--------:|-------:|---:|---:|--------:|-------:|:------:|
| 1 | `asp_jupiter_saturn_square_orb5 & ~mercury_sign_cancer` | 2 | +0.375 | +0.415 | +0.586 | +0.217 | -0.601 | 0.0062 | val+null |
| 2 | `asp_mercury_neptune_sextile_orb5 & ~moon_phase_waning_gibbous` | 2 | +0.339 | +0.379 | +0.625 | +0.019 | -0.539 | 0.0002 | val+null |
| 3 | `asp_jupiter_saturn_square_orb5 & ~sun_sign_cancer` | 2 | +0.336 | +0.376 | +0.669 | +0.141 | -0.563 | 0.0116 | val+null |
| 4 | `~asp_mercury_moon_trine_orb5 & asp_mercury_neptune_sextile_orb5` | 2 | +0.336 | +0.376 | +0.575 | +0.061 | -0.492 | 0.0002 | val+null |
| 5 | `asp_mercury_neptune_sextile_orb5 & ~venus_sign_taurus` | 2 | +0.335 | +0.375 | +0.629 | +0.022 | -0.355 | 0.0004 | val+null |
| 6 | `asp_mercury_neptune_sextile_orb5 & ~mars_ingress_5d` | 2 | +0.318 | +0.358 | +0.539 | +0.048 | -0.774 | 0.0006 | val+null |
| 7 | `~mercury_sign_aquarius & ~neptune_retro` | 2 | +0.313 | +0.363 | +0.413 | +0.307 | -0.114 | 0.0010 | val+null |
| 8 | `~asp_neptune_venus_sextile_orb3 & ~neptune_retro` | 2 | +0.309 | +0.359 | +0.547 | +0.195 | -0.392 | 0.0002 | val+null |
| 9 | `mars_sign_aries & ~sun_sign_aquarius` | 2 | +0.302 | +0.342 | +0.194 | +0.441 | -0.266 | 0.0010 | val+null |
| 10 | `~asp_mars_sun_sextile_orb3 & asp_mercury_neptune_sextile_orb5` | 2 | +0.300 | +0.340 | +0.424 | +0.172 | -0.424 | 0.0004 | val+null |
| 11 | `~asp_mercury_neptune_trine_orb5 & mars_sign_virgo` | 2 | +0.299 | +0.339 | +0.393 | +0.191 | -0.324 | 0.0018 | val+null |
| 12 | `asp_jupiter_pluto_square_orb3 & ~mars_ingress_5d` | 2 | +0.294 | +0.334 | +0.069 | +0.614 | -0.820 | 0.0032 | val+null |
| 13 | `asp_mercury_neptune_sextile_orb5 & ~jupiter_ingress_5d` | 2 | +0.290 | +0.330 | +0.503 | +0.061 | -0.502 | 0.0008 | val+null |
| 14 | `asp_jupiter_pluto_square_orb3 & ~asp_mars_moon_sextile_orb3` | 2 | +0.289 | +0.329 | +0.112 | +0.560 | -0.581 | 0.0044 | val+null |
| 15 | `asp_mercury_neptune_sextile_orb5 & ~asp_saturn_venus_sextile_orb3` | 2 | +0.285 | +0.325 | +0.503 | +0.043 | -0.542 | 0.0004 | val+null |
| 16 | `~mercury_sign_aquarius & ~mercury_sign_libra` | 2 | +0.285 | +0.335 | +0.151 | +0.471 | +0.046 | 0.0002 | YES |
| 17 | `~asp_jupiter_venus_trine_orb3 & asp_mercury_neptune_sextile_orb5` | 2 | +0.285 | +0.325 | +0.504 | +0.052 | -0.485 | 0.0006 | val+null |
| 18 | `~asp_mars_saturn_sextile_orb5 & asp_mercury_neptune_sextile_orb5` | 2 | +0.284 | +0.324 | +0.503 | +0.046 | -0.580 | 0.0010 | val+null |
| 19 | `asp_jupiter_saturn_square_orb5 & ~sun_sign_gemini` | 2 | +0.280 | +0.320 | +0.304 | +0.302 | -0.654 | 0.0154 | val+null |
| 20 | `asp_jupiter_saturn_square_orb5` | 1 | +0.218 | +0.228 | +0.304 | +0.160 | -0.572 | 0.0282 | val+null |
| 21 | `~mercury_sign_libra` | 1 | +0.162 | +0.182 | +0.076 | +0.248 | -0.088 | 0.0032 | val+null |
| 22 | `~neptune_retro` | 1 | +0.144 | +0.164 | +0.286 | +0.047 | -0.276 | 0.0248 | val+null |
| 23 | `asp_mars_mercury_trine_orb5` | 1 | +0.126 | +0.136 | +0.048 | +0.232 | -0.953 | 0.0118 | val+null |
| 24 | `~mercury_sign_aquarius` | 1 | +0.102 | +0.122 | +0.065 | +0.169 | +0.134 | 0.0210 | YES |
| 25 | `~asp_neptune_venus_sextile_orb3` | 1 | +0.087 | +0.107 | +0.155 | +0.074 | -0.094 | 0.0004 | val+null |
| 26 | `~uranus_retro` | 1 | +0.084 | +0.104 | +0.104 | +0.061 | -0.153 | 0.0760 | no |
| 27 | `asp_jupiter_pluto_square_orb5` | 1 | +0.078 | +0.088 | +0.063 | +0.194 | -0.348 | 0.0916 | no |
| 28 | `~sun_sign_virgo` | 1 | +0.057 | +0.077 | +0.157 | +0.024 | -0.006 | 0.0528 | no |
| 29 | `~asp_jupiter_venus_trine_orb3` | 1 | +0.043 | +0.063 | +0.105 | +0.031 | -0.011 | 0.0234 | val+null |
| 30 | `~asp_neptune_venus_sextile_orb1` | 1 | +0.038 | +0.058 | +0.093 | +0.031 | -0.063 | 0.0020 | val+null |

## Monte Carlo targets (stable + honesty unstable)

| Rule | Stable | Val Fit | Hold XS | Null p | Stable bar | Bar+holdout |
|------|:------:|--------:|--------:|-------:|:----------:|:-----------:|
| `asp_jupiter_saturn_square_orb5 & ~mercury_sign_cancer` | Y | +0.3752 | -0.6007 | 0.0062 | YES | no |
| `asp_mercury_neptune_sextile_orb5 & ~moon_phase_waning_gibbous` | Y | +0.3389 | -0.5388 | 0.0002 | YES | no |
| `asp_jupiter_saturn_square_orb5 & ~sun_sign_cancer` | Y | +0.3364 | -0.5632 | 0.0116 | YES | no |
| `~asp_mercury_moon_trine_orb5 & asp_mercury_neptune_sextile_orb5` | Y | +0.3360 | -0.4917 | 0.0002 | YES | no |
| `asp_mercury_neptune_sextile_orb5 & ~venus_sign_taurus` | Y | +0.3353 | -0.3547 | 0.0004 | YES | no |
| `asp_mercury_neptune_sextile_orb5 & ~mars_ingress_5d` | Y | +0.3178 | -0.7742 | 0.0006 | YES | no |
| `~mercury_sign_aquarius & ~neptune_retro` | Y | +0.3132 | -0.1139 | 0.0010 | YES | no |
| `~asp_neptune_venus_sextile_orb3 & ~neptune_retro` | Y | +0.3088 | -0.3919 | 0.0002 | YES | no |
| `mars_sign_aries & ~sun_sign_aquarius` | Y | +0.3019 | -0.2665 | 0.0010 | YES | no |
| `~asp_mars_sun_sextile_orb3 & asp_mercury_neptune_sextile_orb5` | Y | +0.3002 | -0.4242 | 0.0004 | YES | no |
| `~asp_mercury_neptune_trine_orb5 & mars_sign_virgo` | Y | +0.2987 | -0.3239 | 0.0018 | YES | no |
| `asp_jupiter_pluto_square_orb3 & ~mars_ingress_5d` | Y | +0.2942 | -0.8202 | 0.0032 | YES | no |
| `asp_mercury_neptune_sextile_orb5 & ~jupiter_ingress_5d` | Y | +0.2902 | -0.5023 | 0.0008 | YES | no |
| `asp_jupiter_pluto_square_orb3 & ~asp_mars_moon_sextile_orb3` | Y | +0.2894 | -0.5814 | 0.0044 | YES | no |
| `asp_mercury_neptune_sextile_orb5 & ~asp_saturn_venus_sextile_orb3` | Y | +0.2851 | -0.5423 | 0.0004 | YES | no |
| `~mercury_sign_aquarius & ~mercury_sign_libra` | Y | +0.2850 | +0.0461 | 0.0002 | YES | YES |
| `~asp_jupiter_venus_trine_orb3 & asp_mercury_neptune_sextile_orb5` | Y | +0.2848 | -0.4848 | 0.0006 | YES | no |
| `~asp_mars_saturn_sextile_orb5 & asp_mercury_neptune_sextile_orb5` | Y | +0.2837 | -0.5800 | 0.0010 | YES | no |
| `asp_jupiter_saturn_square_orb5 & ~sun_sign_gemini` | Y | +0.2805 | -0.6543 | 0.0154 | YES | no |
| `asp_jupiter_saturn_square_orb5` | Y | +0.2176 | -0.5716 | 0.0282 | YES | no |
| `~mercury_sign_libra` | Y | +0.1621 | -0.0878 | 0.0032 | YES | no |
| `~neptune_retro` | Y | +0.1442 | -0.2756 | 0.0248 | YES | no |
| `asp_mars_mercury_trine_orb5` | Y | +0.1262 | -0.9534 | 0.0118 | YES | no |
| `~mercury_sign_aquarius` | Y | +0.1022 | +0.1339 | 0.0210 | YES | YES |
| `~asp_neptune_venus_sextile_orb3` | Y | +0.0868 | -0.0944 | 0.0004 | YES | no |
| `~uranus_retro` | Y | +0.0837 | -0.1526 | 0.0760 | no | no |
| `asp_jupiter_pluto_square_orb5` | Y | +0.0784 | -0.3477 | 0.0916 | no | no |
| `~sun_sign_virgo` | Y | +0.0568 | -0.0062 | 0.0528 | no | no |
| `~asp_jupiter_venus_trine_orb3` | Y | +0.0425 | -0.0114 | 0.0234 | YES | no |
| `~asp_neptune_venus_sextile_orb1` | Y | +0.0377 | -0.0629 | 0.0020 | YES | no |
| `~asp_mercury_uranus_square_orb5 & asp_pluto_uranus_sextile_orb3 & ~mars_ingress_3d` | N | +0.5639 | -0.7553 | 0.0006 | no | no |
| `~asp_moon_saturn_trine_orb1 & asp_pluto_uranus_sextile_orb3 & ~mars_ingress_3d` | N | +0.5564 | -0.7553 | 0.0006 | no | no |
| `~asp_moon_saturn_trine_orb1 & asp_pluto_uranus_sextile_orb3 & ~sun_sign_cancer` | N | +0.5500 | -0.7553 | 0.0010 | no | no |
| `~asp_mercury_uranus_square_orb5 & asp_pluto_uranus_sextile_orb3 & ~sun_sign_cancer` | N | +0.5480 | -0.7553 | 0.0002 | no | no |
| `~asp_moon_saturn_trine_orb1 & asp_pluto_uranus_sextile_orb3 & ~mercury_sign_cancer` | N | +0.5470 | -0.7553 | 0.0008 | no | no |
| `~asp_mercury_uranus_square_orb5 & asp_pluto_uranus_sextile_orb3 & ~mars_ingress_5d` | N | +0.5449 | -0.7553 | 0.0020 | no | no |
| `~asp_mercury_uranus_square_orb5 & asp_pluto_uranus_sextile_orb3 & ~mercury_sign_cancer` | N | +0.5448 | -0.7553 | 0.0012 | no | no |
| `~asp_moon_saturn_trine_orb1 & asp_pluto_uranus_sextile_orb3 & ~venus_sign_aries` | N | +0.5427 | -0.7553 | 0.0014 | no | no |
| `~asp_jupiter_sun_sextile_orb5 & ~asp_moon_saturn_trine_orb1 & asp_pluto_uranus_sextile_orb3` | N | +0.5411 | -0.7553 | 0.0032 | no | no |
| `~asp_moon_saturn_trine_orb1 & asp_pluto_uranus_sextile_orb3 & ~mars_ingress_5d` | N | +0.5394 | -0.7553 | 0.0008 | no | no |

## Verdict

**Technically YES — 2 stable survivors clear val + both halves + null p<0.05 AND holdout XS>0:**
`~mercury_sign_aquarius & ~mercury_sign_libra` (hold XS ≈ +0.05) and `~mercury_sign_aquarius` (hold XS ≈ +0.13).

**Caveat (be honest):** these are weak holdout margins after a huge search; most other stable+null clears have **negative** holdout XS. Multiple-testing is severe (beam over ~40k L2/L3 candidates). Do **not** treat this as confirmed tradable alpha — at best a fragile curiosity that survived regime filters where Saturn-Pisces did not.

## Honest interpretation

- Searchable atoms: **789** (total cached including outer signs: 849).
- Val-passers (fit>0 & XS>0): **474**; stable: **61**.
- Selection never used holdout. Multiple testing across thousands of beam candidates means even p<0.05 is weak evidence.
- Episode filter is the main upgrade vs lattice v1: long outer-planet sign regimes no longer dominate the leaderboard.
- Highest raw val-fitness cluster is still **outer-aspect** driven (`asp_pluto_uranus_sextile_orb3` family) — many episodes so they pass the 400d cap, but they fail split-half stability (H2 XS < 0). Stable survivors are lower-fit and mostly Mercury/Mars/Jupiter–Saturn chops.
- The two holdout-positive clears are Mercury-sign exclusions (`~mercury_sign_aquarius` ± `~mercury_sign_libra`); holdout XS is small (+0.05 / +0.13).
- Costs 10 bps/side; Adj Close proxy; tropical geocentric ephemeris.

Artifacts: `sweep_v2.csv`, `sweep_v2.md`.
