# Sweep v3 — Beam L1→L4 + Nasdaq cross-asset must-pass

## Method

Heavier CPU search on top of sweep v2 regime filters: **beam=500**, **length-4** AND extensions, **Nasdaq (^IXIC)** second-index must-pass, and Monte Carlo **n=50000** (requested 50000) with multiprocessing.

Beam selection still uses **SPX validation fitness** only. A rule is a **cross-asset survivor** only if validation excess Sharpe > 0 on **both** SPX and Nasdaq (same long/flat signal, 10 bps). Holdout reported for both indexes; **never used for selection**.

**Regime / episode filters (from v2):**

- Hard drop max True episode on validation > **400** days
- Soft penalty if episodes < 4 on val and < 8 on train+val
- No outer-planet signs in searchable pool; hard-drop pure outer-sign rules
- **Stable**: SPX excess Sharpe > 0 in **both** validation temporal halves

**Frozen protocol:**

| Item | Value |
|------|-------|
| Train | 1970-01-01 → 1994-12-31 |
| Validation | 1995-01-01 → 2009-12-31 |
| Holdout | 2010-01-01 → present (**never used for selection**) |
| Indexes | SPX `^GSPC` + Nasdaq `^IXIC` (aligned calendar) |
| Position | Long when True; flat when False; BH = always long |
| Costs | 10.0 bps / side |
| Beam / max length | 500 / 4 |
| Monte Carlo | Circular-shift SPX val fitness, n=50000, seed=42, workers=8 |

Runtime: **2126.9s** (atoms 0.1s · L1 0.5s · L2 293.1s · L3 738.0s · L4 1080.9s · MC 12.3s) · cores=8 workers=8 · atoms 849/789 · ranked=2000 · stable=50 · cross=1603 · cross+stable=41 · days train/val/hold=6040/3778/4195 · aligned=14013

MC note: mc_n=50000 workers=8 elapsed=12.3s

## Top rules by SPX validation fitness

| Rank | Rule | L | Val XS | Val Fit | NDQ Val XS | H1 | H2 | Stable | Cross | SPX Hold XS | NDQ Hold XS | Null p |
|-----:|------|--:|-------:|--------:|-----------:|---:|---:|:------:|:-----:|-----------:|------------:|-------:|
| 1 | `~asp_mars_pluto_sextile_orb3 & asp_neptune_uranus_conj_orb5 & ~asp_neptune_venus_conj_orb3 & ~asp_uranus_venus_square_orb3` | 4 | +0.813 | +0.663 | +0.410 | +1.080 | -0.204 | N | Y | -0.755 | -0.818 | 0.0000 |
| 2 | `~asp_mars_pluto_sextile_orb3 & asp_neptune_uranus_conj_orb5 & ~asp_neptune_venus_conj_orb5 & ~asp_uranus_venus_square_orb3` | 4 | +0.811 | +0.661 | +0.423 | +1.077 | -0.204 | N | Y | -0.755 | -0.818 | 0.0000 |
| 3 | `~asp_mars_pluto_sextile_orb3 & asp_neptune_uranus_conj_orb5 & ~asp_neptune_venus_conj_orb3 & ~asp_sun_uranus_trine_orb5` | 4 | +0.809 | +0.659 | +0.400 | +1.075 | -0.204 | N | Y | -0.755 | -0.818 | 0.0000 |
| 4 | `~asp_mars_pluto_sextile_orb3 & asp_neptune_uranus_conj_orb5 & ~asp_neptune_venus_conj_orb5 & ~asp_sun_uranus_trine_orb5` | 4 | +0.808 | +0.658 | +0.412 | +1.072 | -0.204 | N | Y | -0.755 | -0.818 | 0.0000 |
| 5 | `~asp_mars_pluto_sextile_orb3 & asp_neptune_uranus_conj_orb5 & ~asp_neptune_venus_conj_orb3 & ~venus_sign_taurus` | 4 | +0.806 | +0.656 | +0.323 | +1.070 | -0.204 | N | Y | -0.755 | -0.818 | 0.0000 |
| 6 | `~asp_mars_pluto_sextile_orb3 & asp_neptune_uranus_conj_orb5 & ~asp_neptune_venus_conj_orb5 & ~venus_sign_taurus` | 4 | +0.805 | +0.655 | +0.333 | +1.068 | -0.204 | N | Y | -0.755 | -0.818 | 0.0002 |
| 7 | `~asp_mars_pluto_sextile_orb3 & asp_neptune_uranus_conj_orb5 & ~asp_saturn_venus_sextile_orb5 & ~asp_uranus_venus_square_orb3` | 4 | +0.804 | +0.654 | +0.492 | +1.066 | -0.204 | N | Y | -0.755 | -0.818 | 0.0000 |
| 8 | `~asp_mars_pluto_sextile_orb3 & asp_neptune_uranus_conj_orb5 & ~asp_neptune_venus_conj_orb3 & ~asp_pluto_sun_opp_orb5` | 4 | +0.803 | +0.653 | +0.340 | +1.065 | -0.204 | N | Y | -0.755 | -0.818 | 0.0000 |
| 9 | `~asp_mars_pluto_sextile_orb3 & asp_neptune_uranus_conj_orb5 & ~asp_neptune_venus_conj_orb5 & ~asp_pluto_sun_opp_orb5` | 4 | +0.801 | +0.651 | +0.350 | +1.062 | -0.204 | N | Y | -0.755 | -0.818 | 0.0000 |
| 10 | `~asp_mars_pluto_sextile_orb3 & asp_neptune_uranus_conj_orb5 & ~asp_saturn_venus_sextile_orb5 & ~asp_sun_uranus_trine_orb5` | 4 | +0.800 | +0.650 | +0.481 | +1.061 | -0.204 | N | Y | -0.755 | -0.818 | 0.0000 |
| 11 | `~asp_mars_pluto_sextile_orb3 & ~asp_neptune_sun_trine_orb1 & asp_neptune_uranus_conj_orb5 & ~asp_neptune_venus_conj_orb3` | 4 | +0.800 | +0.650 | +0.354 | +1.061 | -0.204 | N | Y | -0.755 | -0.818 | — |
| 12 | `~asp_mars_pluto_sextile_orb3 & ~asp_neptune_sun_trine_orb1 & asp_neptune_uranus_conj_orb5 & ~asp_neptune_venus_conj_orb5` | 4 | +0.798 | +0.648 | +0.364 | +1.058 | -0.204 | N | Y | -0.755 | -0.818 | — |
| 13 | `~asp_mars_pluto_sextile_orb3 & asp_neptune_uranus_conj_orb5 & ~asp_pluto_sun_opp_orb5 & ~asp_saturn_venus_sextile_orb5` | 4 | +0.793 | +0.643 | +0.419 | +1.052 | -0.204 | N | Y | -0.755 | -0.818 | — |
| 14 | `~asp_mars_pluto_sextile_orb3 & ~asp_mercury_moon_trine_orb1 & asp_neptune_uranus_conj_orb5 & ~asp_neptune_venus_conj_orb3` | 4 | +0.792 | +0.642 | +0.375 | +1.050 | -0.204 | N | Y | -0.755 | -0.818 | — |
| 15 | `~asp_mars_pluto_sextile_orb3 & ~asp_neptune_sun_trine_orb1 & asp_neptune_uranus_conj_orb5 & ~asp_saturn_venus_sextile_orb5` | 4 | +0.791 | +0.641 | +0.432 | +1.048 | -0.204 | N | Y | -0.755 | -0.818 | — |
| 16 | `~asp_jupiter_mercury_opp_orb5 & ~asp_mars_pluto_sextile_orb3 & asp_neptune_uranus_conj_orb5 & ~asp_neptune_venus_conj_orb3` | 4 | +0.787 | +0.637 | +0.219 | +1.043 | -0.204 | N | Y | -0.755 | -0.818 | — |
| 17 | `~asp_mars_pluto_sextile_orb3 & ~asp_mars_sun_square_orb1 & asp_neptune_uranus_conj_orb5 & ~asp_neptune_venus_conj_orb3` | 4 | +0.786 | +0.636 | +0.340 | +1.041 | -0.204 | N | Y | -0.755 | -0.818 | — |
| 18 | `~asp_jupiter_mercury_opp_orb5 & ~asp_mars_pluto_sextile_orb3 & asp_neptune_uranus_conj_orb5 & ~asp_neptune_venus_conj_orb5` | 4 | +0.785 | +0.635 | +0.227 | +1.040 | -0.204 | N | Y | -0.755 | -0.818 | — |
| 19 | `~asp_mars_pluto_sextile_orb3 & ~asp_moon_venus_trine_orb3 & asp_neptune_uranus_conj_orb5 & ~asp_neptune_venus_conj_orb3` | 4 | +0.784 | +0.634 | +0.411 | +1.038 | -0.204 | N | Y | -0.755 | -0.818 | — |
| 20 | `~asp_mars_pluto_sextile_orb3 & ~asp_mars_sun_square_orb1 & asp_neptune_uranus_conj_orb5 & ~asp_neptune_venus_conj_orb5` | 4 | +0.783 | +0.633 | +0.350 | +1.038 | -0.204 | N | Y | -0.755 | -0.818 | — |
| 21 | `~asp_jupiter_sun_conj_orb5 & ~asp_mars_pluto_sextile_orb3 & asp_neptune_uranus_conj_orb5 & ~asp_uranus_venus_square_orb3` | 4 | +0.782 | +0.632 | +0.432 | +1.036 | -0.204 | N | Y | -0.755 | -0.818 | — |
| 22 | `~asp_mars_pluto_sextile_orb3 & ~asp_moon_venus_trine_orb3 & asp_neptune_uranus_conj_orb5 & ~asp_neptune_venus_conj_orb5` | 4 | +0.781 | +0.631 | +0.423 | +1.035 | -0.204 | N | Y | -0.755 | -0.818 | — |
| 23 | `~asp_mars_pluto_sextile_orb3 & asp_neptune_uranus_conj_orb5 & ~asp_neptune_venus_conj_orb3 & ~venus_sign_libra` | 4 | +0.780 | +0.630 | +0.559 | +1.033 | -0.204 | N | Y | -0.755 | -0.818 | — |
| 24 | `~asp_mars_pluto_sextile_orb3 & ~asp_mars_sun_square_orb5 & asp_neptune_uranus_conj_orb5 & ~asp_neptune_venus_conj_orb3` | 4 | +0.779 | +0.629 | +0.333 | +1.031 | -0.204 | N | Y | -0.755 | -0.818 | — |
| 25 | `~asp_jupiter_sun_conj_orb5 & ~asp_mars_pluto_sextile_orb3 & asp_neptune_uranus_conj_orb5 & ~asp_sun_uranus_trine_orb5` | 4 | +0.778 | +0.628 | +0.421 | +1.030 | -0.204 | N | Y | -0.755 | -0.818 | — |

## Cross-asset stable survivors

SPX XS>0 in both val halves **and** Nasdaq val XS>0 (selection still never uses holdout).

| Rank | Rule | L | SPX Fit | SPX Val XS | NDQ Val XS | H1 | H2 | SPX Hold | NDQ Hold | Null p | Full bar |
|-----:|------|--:|--------:|-----------:|-----------:|---:|---:|---------:|---------:|-------:|:--------:|
| 1 | `asp_pluto_sun_trine_orb5 & venus_ingress_5d & ~venus_sign_aries` | 3 | +0.505 | +0.595 | +0.143 | +0.484 | +0.749 | -1.007 | -0.851 | 0.0000 | cross+null |
| 2 | `asp_jupiter_saturn_square_orb5 & ~jupiter_retro & ~mercury_sign_cancer` | 3 | +0.490 | +0.595 | +0.428 | +0.607 | +0.542 | -0.735 | -0.821 | 0.0023 | cross+null |
| 3 | `asp_jupiter_saturn_square_orb5 & ~jupiter_retro` | 2 | +0.449 | +0.489 | +0.286 | +0.607 | +0.402 | -0.699 | -0.762 | 0.0048 | cross+null |
| 4 | `~uranus_retro & venus_sign_libra` | 2 | +0.400 | +0.440 | +0.499 | +0.462 | +0.301 | -1.014 | -1.006 | 0.0000 | cross+null |
| 5 | `asp_pluto_sun_trine_orb5 & venus_ingress_5d` | 2 | +0.383 | +0.413 | +0.038 | +0.461 | +0.485 | -1.037 | -0.956 | 0.0000 | cross+null |
| 6 | `asp_jupiter_saturn_square_orb5 & ~mercury_sign_cancer` | 2 | +0.375 | +0.415 | +0.293 | +0.586 | +0.217 | -0.601 | -0.693 | 0.0060 | cross+null |
| 7 | `asp_jupiter_saturn_square_orb5` | 1 | +0.218 | +0.228 | +0.024 | +0.304 | +0.160 | -0.572 | -0.645 | 0.0279 | cross+null |
| 8 | `~mercury_sign_libra` | 1 | +0.162 | +0.182 | +0.152 | +0.076 | +0.248 | -0.088 | -0.058 | 0.0023 | cross+null |
| 9 | `~neptune_retro` | 1 | +0.144 | +0.164 | +0.041 | +0.286 | +0.047 | -0.276 | -0.250 | 0.0224 | cross+null |
| 10 | `asp_mars_mercury_trine_orb5` | 1 | +0.126 | +0.136 | +0.085 | +0.048 | +0.232 | -0.953 | -0.923 | 0.0118 | cross+null |
| 11 | `~mercury_sign_aquarius` | 1 | +0.102 | +0.122 | +0.155 | +0.065 | +0.169 | +0.134 | +0.119 | 0.0206 | YES |
| 12 | `~asp_neptune_venus_sextile_orb3` | 1 | +0.087 | +0.107 | +0.174 | +0.155 | +0.074 | -0.094 | -0.073 | 0.0009 | cross+null |
| 13 | `~sun_sign_virgo` | 1 | +0.057 | +0.077 | +0.040 | +0.157 | +0.024 | -0.006 | -0.025 | 0.0511 | no |
| 14 | `~asp_jupiter_venus_trine_orb3` | 1 | +0.043 | +0.063 | +0.086 | +0.105 | +0.031 | -0.011 | -0.014 | 0.0231 | cross+null |
| 15 | `~asp_neptune_venus_sextile_orb1` | 1 | +0.038 | +0.058 | +0.072 | +0.093 | +0.031 | -0.063 | -0.044 | 0.0016 | cross+null |
| 16 | `~asp_jupiter_venus_trine_orb5` | 1 | +0.037 | +0.057 | +0.086 | +0.117 | +0.014 | -0.051 | -0.045 | 0.0503 | no |
| 17 | `mercury_sign_sagittarius` | 1 | +0.037 | +0.047 | +0.145 | +0.008 | +0.069 | -0.590 | -0.698 | 0.0519 | no |
| 18 | `~mercury_sign_gemini` | 1 | +0.036 | +0.056 | +0.065 | +0.063 | +0.061 | -0.054 | -0.058 | 0.1318 | no |
| 19 | `~asp_mercury_pluto_sextile_orb3` | 1 | +0.031 | +0.051 | +0.075 | +0.071 | +0.032 | -0.056 | -0.033 | 0.0387 | cross+null |
| 20 | `~asp_neptune_sun_opp_orb5` | 1 | +0.030 | +0.050 | +0.083 | +0.078 | +0.028 | +0.035 | +0.031 | 0.0465 | YES |
| 21 | `~asp_mercury_pluto_sextile_orb5` | 1 | +0.027 | +0.047 | +0.073 | +0.071 | +0.028 | -0.071 | -0.056 | 0.0728 | no |
| 22 | `~asp_saturn_sun_trine_orb3` | 1 | +0.026 | +0.046 | +0.031 | +0.083 | +0.023 | +0.002 | +0.008 | 0.0576 | no |
| 23 | `~asp_saturn_venus_sextile_orb5` | 1 | +0.024 | +0.044 | +0.030 | +0.064 | +0.030 | -0.078 | -0.101 | 0.0779 | no |
| 24 | `~asp_jupiter_venus_square_orb3` | 1 | +0.022 | +0.042 | +0.013 | +0.031 | +0.049 | +0.056 | +0.053 | 0.0598 | no |
| 25 | `~asp_jupiter_sun_sextile_orb3` | 1 | +0.019 | +0.039 | +0.082 | +0.030 | +0.045 | -0.003 | -0.010 | 0.0527 | no |
| 26 | `~asp_saturn_sun_trine_orb5` | 1 | +0.016 | +0.036 | +0.012 | +0.030 | +0.046 | -0.008 | -0.022 | — | no |
| 27 | `~asp_mars_neptune_square_orb5` | 1 | +0.016 | +0.036 | +0.051 | +0.077 | +0.006 | -0.055 | -0.026 | — | no |
| 28 | `~asp_neptune_venus_sextile_orb5` | 1 | +0.015 | +0.035 | +0.067 | +0.031 | +0.046 | -0.089 | -0.060 | — | no |
| 29 | `~asp_mars_neptune_square_orb3` | 1 | +0.010 | +0.030 | +0.058 | +0.057 | +0.010 | +0.011 | +0.031 | — | no |
| 30 | `~asp_mercury_uranus_trine_orb5` | 1 | +0.008 | +0.028 | +0.014 | +0.024 | +0.036 | -0.059 | -0.062 | — | no |

## Monte Carlo targets (cross-asset stable + honesty unstable)

| Rule | Stab | Cross | SPX Fit | NDQ Val XS | SPX Hold | NDQ Hold | Null p | Full |
|------|:----:|:-----:|--------:|-----------:|---------:|---------:|-------:|:----:|
| `asp_pluto_sun_trine_orb5 & venus_ingress_5d & ~venus_sign_aries` | Y | Y | +0.505 | +0.143 | -1.007 | -0.851 | 0.0000 | no |
| `asp_jupiter_saturn_square_orb5 & ~jupiter_retro & ~mercury_sign_cancer` | Y | Y | +0.490 | +0.428 | -0.735 | -0.821 | 0.0023 | no |
| `asp_jupiter_saturn_square_orb5 & ~jupiter_retro` | Y | Y | +0.449 | +0.286 | -0.699 | -0.762 | 0.0048 | no |
| `~uranus_retro & venus_sign_libra` | Y | Y | +0.400 | +0.499 | -1.014 | -1.006 | 0.0000 | no |
| `asp_pluto_sun_trine_orb5 & venus_ingress_5d` | Y | Y | +0.383 | +0.038 | -1.037 | -0.956 | 0.0000 | no |
| `asp_jupiter_saturn_square_orb5 & ~mercury_sign_cancer` | Y | Y | +0.375 | +0.293 | -0.601 | -0.693 | 0.0060 | no |
| `asp_jupiter_saturn_square_orb5` | Y | Y | +0.218 | +0.024 | -0.572 | -0.645 | 0.0279 | no |
| `~mercury_sign_libra` | Y | Y | +0.162 | +0.152 | -0.088 | -0.058 | 0.0023 | no |
| `~neptune_retro` | Y | Y | +0.144 | +0.041 | -0.276 | -0.250 | 0.0224 | no |
| `asp_mars_mercury_trine_orb5` | Y | Y | +0.126 | +0.085 | -0.953 | -0.923 | 0.0118 | no |
| `~mercury_sign_aquarius` | Y | Y | +0.102 | +0.155 | +0.134 | +0.119 | 0.0206 | YES |
| `~asp_neptune_venus_sextile_orb3` | Y | Y | +0.087 | +0.174 | -0.094 | -0.073 | 0.0009 | no |
| `~sun_sign_virgo` | Y | Y | +0.057 | +0.040 | -0.006 | -0.025 | 0.0511 | no |
| `~asp_jupiter_venus_trine_orb3` | Y | Y | +0.043 | +0.086 | -0.011 | -0.014 | 0.0231 | no |
| `~asp_neptune_venus_sextile_orb1` | Y | Y | +0.038 | +0.072 | -0.063 | -0.044 | 0.0016 | no |
| `~asp_jupiter_venus_trine_orb5` | Y | Y | +0.037 | +0.086 | -0.051 | -0.045 | 0.0503 | no |
| `mercury_sign_sagittarius` | Y | Y | +0.037 | +0.145 | -0.590 | -0.698 | 0.0519 | no |
| `~mercury_sign_gemini` | Y | Y | +0.036 | +0.065 | -0.054 | -0.058 | 0.1318 | no |
| `~asp_mercury_pluto_sextile_orb3` | Y | Y | +0.031 | +0.075 | -0.056 | -0.033 | 0.0387 | no |
| `~asp_neptune_sun_opp_orb5` | Y | Y | +0.030 | +0.083 | +0.035 | +0.031 | 0.0465 | YES |
| `~asp_mercury_pluto_sextile_orb5` | Y | Y | +0.027 | +0.073 | -0.071 | -0.056 | 0.0728 | no |
| `~asp_saturn_sun_trine_orb3` | Y | Y | +0.026 | +0.031 | +0.002 | +0.008 | 0.0576 | no |
| `~asp_saturn_venus_sextile_orb5` | Y | Y | +0.024 | +0.030 | -0.078 | -0.101 | 0.0779 | no |
| `~asp_jupiter_venus_square_orb3` | Y | Y | +0.022 | +0.013 | +0.056 | +0.053 | 0.0598 | no |
| `~asp_jupiter_sun_sextile_orb3` | Y | Y | +0.019 | +0.082 | -0.003 | -0.010 | 0.0527 | no |
| `~asp_mars_pluto_sextile_orb3 & asp_neptune_uranus_conj_orb5 & ~asp_neptune_venus_conj_orb3 & ~asp_uranus_venus_square_orb3` | N | Y | +0.663 | +0.410 | -0.755 | -0.818 | 0.0000 | no |
| `~asp_mars_pluto_sextile_orb3 & asp_neptune_uranus_conj_orb5 & ~asp_neptune_venus_conj_orb5 & ~asp_uranus_venus_square_orb3` | N | Y | +0.661 | +0.423 | -0.755 | -0.818 | 0.0000 | no |
| `~asp_mars_pluto_sextile_orb3 & asp_neptune_uranus_conj_orb5 & ~asp_neptune_venus_conj_orb3 & ~asp_sun_uranus_trine_orb5` | N | Y | +0.659 | +0.400 | -0.755 | -0.818 | 0.0000 | no |
| `~asp_mars_pluto_sextile_orb3 & asp_neptune_uranus_conj_orb5 & ~asp_neptune_venus_conj_orb5 & ~asp_sun_uranus_trine_orb5` | N | Y | +0.658 | +0.412 | -0.755 | -0.818 | 0.0000 | no |
| `~asp_mars_pluto_sextile_orb3 & asp_neptune_uranus_conj_orb5 & ~asp_neptune_venus_conj_orb3 & ~venus_sign_taurus` | N | Y | +0.656 | +0.323 | -0.755 | -0.818 | 0.0000 | no |
| `~asp_mars_pluto_sextile_orb3 & asp_neptune_uranus_conj_orb5 & ~asp_neptune_venus_conj_orb5 & ~venus_sign_taurus` | N | Y | +0.655 | +0.333 | -0.755 | -0.818 | 0.0002 | no |
| `~asp_mars_pluto_sextile_orb3 & asp_neptune_uranus_conj_orb5 & ~asp_saturn_venus_sextile_orb5 & ~asp_uranus_venus_square_orb3` | N | Y | +0.654 | +0.492 | -0.755 | -0.818 | 0.0000 | no |
| `~asp_mars_pluto_sextile_orb3 & asp_neptune_uranus_conj_orb5 & ~asp_neptune_venus_conj_orb3 & ~asp_pluto_sun_opp_orb5` | N | Y | +0.653 | +0.340 | -0.755 | -0.818 | 0.0000 | no |
| `~asp_mars_pluto_sextile_orb3 & asp_neptune_uranus_conj_orb5 & ~asp_neptune_venus_conj_orb5 & ~asp_pluto_sun_opp_orb5` | N | Y | +0.651 | +0.350 | -0.755 | -0.818 | 0.0000 | no |
| `~asp_mars_pluto_sextile_orb3 & asp_neptune_uranus_conj_orb5 & ~asp_saturn_venus_sextile_orb5 & ~asp_sun_uranus_trine_orb5` | N | Y | +0.650 | +0.481 | -0.755 | -0.818 | 0.0000 | no |

## Verdict

**YES — two MC targets clear the full bar** (SPX val XS>0 both halves, Nasdaq val XS>0, null p<0.05, SPX holdout XS>0):

1. `~mercury_sign_aquarius` — SPX hold XS=+0.134, NDQ hold XS=+0.119, p=0.0206. This is a **near-buy-and-hold exclusion** (long ≈11/12 of the year); exactly the mercury-sign almost-BH family warned about in v2. **Do not treat as a discovery.**
2. `~asp_neptune_sun_opp_orb5` — SPX hold XS=+0.035, NDQ hold XS=+0.031, p=0.0465 (barely under 0.05). Tiny holdout edge after searching thousands of beam candidates; **weak / likely noise.**

Higher-fitness cross-asset stable rules (Jupiter–Saturn square family, Pluto–Sun trine + Venus ingress, Uranus retro + Venus Libra, etc.) all **fail holdout** (large negative SPX/NDQ holdout XS) despite tiny null p-values — classic multiple-testing overfitting.

## Honest interpretation (multiple testing)

- Searchable atoms ≈ **789**; beam=500; max length=4. Ranked candidates after beams: **2000**. This is a large implicit search; raw p<0.05 is weak.
- Nasdaq must-pass reduces (but does not eliminate) SPX-specific overfitting. It is still one extra filter applied *after* SPX-driven beam selection.
- Holdout was never used for selection. Positive holdout on a shortlisted rule is necessary but not sufficient for a discovery claim.

### Mercury-sign family (almost-BH / lattice echoes)

Mercury-sign rules **did reappear**. The only clear full-bar hit in this sweep is the almost-BH singleton `~mercury_sign_aquarius` (also positive on Nasdaq holdout). Treat as **near-buy-and-hold exclusion** (flat only while Mercury is in Aquarius) — small calendar tilt, sensitive to costs, and expected to look good under massive search. Related length-1/2 mercury-sign rules on the cross-stable list:

- `~mercury_sign_aquarius` — SPX val XS=+0.122, NDQ val XS=+0.155, stable=True, cross=True, SPX hold=+0.134, NDQ hold=+0.119, p=0.0206 (**full bar, but almost-BH**)
- `~mercury_sign_libra` — SPX val XS=+0.182, NDQ val XS=+0.152, stable=True, cross=True, SPX hold=-0.088, p=0.0023
- `asp_jupiter_saturn_square_orb5 & ~mercury_sign_cancer` — SPX val XS=+0.415, NDQ val XS=+0.293, stable=True, cross=True, SPX hold=-0.601, p=0.0060
- High-fit **unstable** L4 rules that AND-in a mercury-sign literal (e.g. `... & ~mercury_sign_leo`) show huge val XS but **holdout ≈ -0.75** — ignored for selection honesty.

- Costs 10 bps/side; Adj Close proxy; tropical geocentric ephemeris; atoms from `atoms_v2.parquet`.

Artifacts: `sweep_v3.csv`, `sweep_v3.md`, `sweep_v3_run.log`.
