# ADR 0013 — DMDc modal bands from a parametric bootstrap

- Status: accepted
- Date: 2026-11-28
- Extends [ADR 0012](0012-dmdc-noise-sensitivity-and-matern-hyperprior.md)

## Context

ADR 0012 left the DMDc estimate as ordinary least squares with a point value and a maximum
relative error. That says nothing about how repeatable a mode is on a single trace, and the fastest
pole's seed-to-seed spread suggested the excitation amplitude might be the limiting factor.

## Decision

### The heater amplitude is not the lever

`heater_k` was swept over 12.5, 25, 50 and 100 K across five seeds, on the same three-pulse
schedule:

| heater | fastest pole across seeds | worst modal error | peak plate |
|---|---|---|---|
| +12.5 K | 24–33 s | 43% | 304 K |
| +25 K (default) | 25–34 s | 35% | 315 K |
| +50 K | 24–32 s | 41% | 338 K |
| +100 K | 20–25 s | 44% | 382 K |

Quadrupling the heater power does not reduce the worst modal error; it moves the error into the
other poles. The default stays at 25 K.

### Parametric bootstrap

The point fit's one-step residuals are resampled with replacement and pushed back through the fitted
ARX recursion, so each replicate keeps every mode's correlation. Each replicate is refitted with the
same order, lags and heater schedule, and the 5th and 95th percentiles of the nearest mode are
reported with the point estimate. 64 replicates run by default; all 64 recovered a stable model on
the default seed.

A moving-block bootstrap was tried first and rejected: 200 s blocks cannot carry the 420 s plate
response, and it moved the slow pole from 433 s to a 71–304 s interval that excluded the fit.

On the default seed the bands are 433 s (365–527), 74 s (57–98) and 29 s (23–39), and all three
contain the configured constants. Across five seeds, 14 of the 15 reported bands contain a
configured constant; the exception is seed 1's fastest band, 26–45 s, which excludes the 25 s
sensor pole.

The band is conditional on the fitted model. It measures sensitivity to observation noise, not
model-structure error, so it understates the uncertainty if the true device is not ARX(4).

## Consequences

- S6 shows each mode with its 5–95% band and states that the middle and fastest poles are the least
  certain.
- The bootstrap costs 64 extra least-squares fits — a few milliseconds in WASM — and is seeded from
  the run's seed, so it is reproducible.
- The coverage claim for the fitted Matérn band is now checked on five traces rather than one
  (`ukf::tests::matern_coverage_holds_across_seeds`).
- No measured data is introduced; the synthetic labels remain mandatory.
