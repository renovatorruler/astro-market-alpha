# Lattice search v1 — Boolean length-1 / length-2

## Method

Exhaustive length-1 evaluation of every atom and its negation (`~atom`), ranked by **validation** fitness. Length-2 = AND of the top `50` length-1 rules by validation fitness, plus folklore-style OR pairs among `moon_phase_*` atoms. Isomorphic rules (`a & b` ≡ `b & a`) are deduplicated; mutually exclusive AND pairs (never co-True on validation) are skipped.

**Frozen protocol (unchanged):**

| Item | Value |
|------|-------|
| Train | 1970-01-01 → 1994-12-31 |
| Validation | 1995-01-01 → 2009-12-31 |
| Holdout | 2010-01-01 → present (**never used for selection**) |
| Position | Long when rule True; flat when False; BH = always long |
| Costs | 10.0 bps / side |
| Fitness | `excess_sharpe − λ · complexity` with λ=0.01 |
| Monte Carlo | Circular-shift returns (atoms fixed), n=1000, seed=42, on validation |

Runtime: **3.3s** · atoms=224 · L1 evaluated=448 · unique ranked=1557 · MC survivors=20 · days train/val/hold=6318/3778/4195

## Pass / fail bar (illustrative)

A rule **passes the protocol bar** if:

1. Validation `excess_sharpe > 0` **and** `fitness > 0`
2. Circular-shift null empirical p-value `< 0.05` on validation fitness
3. Holdout metrics are reported for honesty but **must not** drive selection

## Top rules by validation fitness

| Rank | Rule | L | Cx | Train XS | Val XS | Val Fit | Hold XS | Hold Fit | Null p | Pass? |
|-----:|------|--:|---:|---------:|-------:|--------:|--------:|---------:|-------:|:------:|
| 1 | `jupiter_sign_sagittarius & saturn_sign_pisces` | 2 | 3 | -0.197 | +0.584 | +0.554 | -0.755 | -0.785 | 0.0030 | YES |
| 2 | `~jupiter_sign_capricorn & saturn_sign_pisces` | 2 | 4 | -0.573 | +0.584 | +0.544 | -0.280 | -0.320 | 0.0030 | YES |
| 3 | `~asp_saturn_sun_sextile & saturn_sign_pisces` | 2 | 4 | -0.577 | +0.518 | +0.478 | -0.297 | -0.337 | 0.0090 | YES |
| 4 | `~asp_saturn_venus_sextile & saturn_sign_pisces` | 2 | 4 | -0.558 | +0.501 | +0.461 | -0.305 | -0.345 | 0.0080 | YES |
| 5 | `~saturn_ingress_1d & saturn_sign_pisces` | 2 | 4 | -0.574 | +0.497 | +0.457 | -0.285 | -0.325 | 0.0110 | YES |
| 6 | `~asp_saturn_sun_trine & saturn_sign_pisces` | 2 | 4 | -0.620 | +0.493 | +0.453 | -0.316 | -0.356 | 0.0060 | YES |
| 7 | `~asp_jupiter_venus_square & saturn_sign_pisces` | 2 | 4 | -0.555 | +0.476 | +0.436 | -0.349 | -0.389 | 0.0140 | YES |
| 8 | `saturn_sign_pisces` | 1 | 1 | -0.573 | +0.441 | +0.431 | -0.280 | -0.290 | 0.0200 | YES |
| 9 | `~asp_mars_venus_conj & saturn_sign_aries` | 2 | 4 | -0.503 | +0.467 | +0.427 | -0.323 | -0.363 | 0.0519 | no |
| 10 | `~asp_jupiter_sun_square & saturn_sign_pisces` | 2 | 4 | -0.589 | +0.460 | +0.420 | -0.283 | -0.323 | 0.0110 | YES |
| 11 | `~mars_sign_sagittarius & saturn_sign_aries` | 2 | 4 | -0.503 | +0.453 | +0.413 | -0.323 | -0.363 | 0.0679 | no |
| 12 | `~jupiter_ingress_1d & saturn_sign_pisces` | 2 | 4 | -0.595 | +0.449 | +0.409 | -0.297 | -0.337 | 0.0200 | YES |
| 13 | `~saturn_sign_gemini & ~saturn_sign_virgo` | 2 | 5 | -0.059 | +0.454 | +0.404 | +0.040 | -0.010 | 0.0519 | no |
| 14 | `saturn_sign_pisces & ~sun_sign_cancer` | 2 | 4 | -0.607 | +0.441 | +0.401 | -0.335 | -0.375 | 0.0140 | YES |
| 15 | `mars_sign_virgo & ~saturn_sign_gemini` | 2 | 4 | -0.539 | +0.440 | +0.400 | -0.261 | -0.301 | 0.0010 | YES |
| 16 | `saturn_sign_pisces & ~sun_sign_capricorn` | 2 | 4 | -0.573 | +0.439 | +0.399 | -0.297 | -0.337 | 0.0160 | YES |
| 17 | `~asp_mars_venus_sextile & saturn_sign_pisces` | 2 | 4 | -0.629 | +0.436 | +0.396 | -0.274 | -0.314 | 0.0180 | YES |
| 18 | `saturn_sign_pisces & ~sun_sign_libra` | 2 | 4 | -0.586 | +0.428 | +0.388 | -0.279 | -0.319 | 0.0120 | YES |
| 19 | `saturn_sign_aries & ~venus_sign_sagittarius` | 2 | 4 | -0.503 | +0.426 | +0.386 | -0.323 | -0.363 | 0.0709 | no |
| 20 | `~mercury_sign_aquarius & saturn_sign_pisces` | 2 | 4 | -0.558 | +0.425 | +0.385 | -0.331 | -0.371 | 0.0130 | YES |
| 21 | `~mercury_sign_cancer & saturn_sign_pisces` | 2 | 4 | -0.589 | +0.424 | +0.384 | -0.307 | -0.347 | — | — |
| 22 | `~asp_jupiter_venus_conj & saturn_sign_aries` | 2 | 4 | -0.503 | +0.422 | +0.382 | -0.387 | -0.427 | — | — |
| 23 | `~asp_jupiter_mars_conj & saturn_sign_pisces` | 2 | 4 | -0.573 | +0.420 | +0.380 | -0.329 | -0.369 | — | — |
| 24 | `~asp_jupiter_sun_opp & saturn_sign_aries` | 2 | 4 | -0.503 | +0.414 | +0.374 | -0.323 | -0.363 | — | — |
| 25 | `~asp_mars_venus_conj & saturn_sign_pisces` | 2 | 4 | -0.573 | +0.411 | +0.371 | -0.309 | -0.349 | — | — |

## Monte Carlo survivors (detail)

| Rule | Val Fit | Val XS | Hold XS | Null p | Val interesting | Null reject | Protocol pass |
|------|--------:|-------:|--------:|-------:|:---------------:|:-----------:|:-------------:|
| `jupiter_sign_sagittarius & saturn_sign_pisces` | +0.5542 | +0.5842 | -0.7553 | 0.0030 | Y | Y | YES |
| `~jupiter_sign_capricorn & saturn_sign_pisces` | +0.5442 | +0.5842 | -0.2799 | 0.0030 | Y | Y | YES |
| `~asp_saturn_sun_sextile & saturn_sign_pisces` | +0.4780 | +0.5180 | -0.2972 | 0.0090 | Y | Y | YES |
| `~asp_saturn_venus_sextile & saturn_sign_pisces` | +0.4615 | +0.5015 | -0.3047 | 0.0080 | Y | Y | YES |
| `~saturn_ingress_1d & saturn_sign_pisces` | +0.4566 | +0.4966 | -0.2849 | 0.0110 | Y | Y | YES |
| `~asp_saturn_sun_trine & saturn_sign_pisces` | +0.4530 | +0.4930 | -0.3163 | 0.0060 | Y | Y | YES |
| `~asp_jupiter_venus_square & saturn_sign_pisces` | +0.4363 | +0.4763 | -0.3488 | 0.0140 | Y | Y | YES |
| `saturn_sign_pisces` | +0.4309 | +0.4409 | -0.2799 | 0.0200 | Y | Y | YES |
| `~asp_mars_venus_conj & saturn_sign_aries` | +0.4266 | +0.4666 | -0.3233 | 0.0519 | Y | N | no |
| `~asp_jupiter_sun_square & saturn_sign_pisces` | +0.4198 | +0.4598 | -0.2828 | 0.0110 | Y | Y | YES |
| `~mars_sign_sagittarius & saturn_sign_aries` | +0.4131 | +0.4531 | -0.3233 | 0.0679 | Y | N | no |
| `~jupiter_ingress_1d & saturn_sign_pisces` | +0.4090 | +0.4490 | -0.2968 | 0.0200 | Y | Y | YES |
| `~saturn_sign_gemini & ~saturn_sign_virgo` | +0.4041 | +0.4541 | +0.0397 | 0.0519 | Y | N | no |
| `saturn_sign_pisces & ~sun_sign_cancer` | +0.4008 | +0.4408 | -0.3348 | 0.0140 | Y | Y | YES |
| `mars_sign_virgo & ~saturn_sign_gemini` | +0.3999 | +0.4399 | -0.2614 | 0.0010 | Y | Y | YES |
| `saturn_sign_pisces & ~sun_sign_capricorn` | +0.3993 | +0.4393 | -0.2967 | 0.0160 | Y | Y | YES |
| `~asp_mars_venus_sextile & saturn_sign_pisces` | +0.3956 | +0.4356 | -0.2741 | 0.0180 | Y | Y | YES |
| `saturn_sign_pisces & ~sun_sign_libra` | +0.3879 | +0.4279 | -0.2788 | 0.0120 | Y | Y | YES |
| `saturn_sign_aries & ~venus_sign_sagittarius` | +0.3861 | +0.4261 | -0.3233 | 0.0709 | Y | N | no |
| `~mercury_sign_aquarius & saturn_sign_pisces` | +0.3850 | +0.4250 | -0.3305 | 0.0130 | Y | Y | YES |

## Verdict

**At least one rule passes the illustrative protocol bar** (val fitness & excess Sharpe > 0, and null p < 0.05). Holdout confirmation is reported separately above — do not re-select on it.

However, **none of the protocol-passers show positive holdout excess Sharpe** in this run. Treat the validation/null clears as *interesting under the bar*, not as confirmed out-of-sample alpha — especially for multi-year sign regimes.

## Honest interpretation

- Selection used **validation only**; holdout numbers are a one-shot report.
- Multiple testing is severe: hundreds of length-1 and thousands of length-2 candidates. A single-rule null p-value does **not** correct for the search. Even a p < 0.05 survivor should be treated cautiously.
- **Outer-planet sign regimes** (e.g. Saturn in Pisces / Aries) span years. High validation fitness for such atoms often reflects being long through a favorable multi-year market regime rather than high-frequency timing skill. Circular-shift nulls keep the contiguous True/False blocks intact, so they do **not** fully stress-test regime artifacts.
- Many top length-2 ANDs are near-aliases of a single strong length-1 atom (e.g. `~rare_aspect & saturn_sign_pisces` ≈ `saturn_sign_pisces`). Prefer the simpler length-1 form when holdout/val metrics are similar.
- Holdout excess Sharpe for the leading Saturn-Pisces family is typically **negative** (always-flat in holdout reads as −Sharpe_BH). That fails confirmation even when the validation+null bar is cleared.
- Costs are 10 bps/side; zero-cost wins are not counted. Flat earns 0 (no cash rf).
- Adj Close is a proxy (dividends not fully total-return). Ephemeris is tropical geocentric.
- Prefer simpler rules (lower complexity); fitness already applies a λ penalty.

Artifacts: `lattice_v1.csv`, `lattice_v1.md`.
