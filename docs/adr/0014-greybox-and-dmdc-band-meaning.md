# ADR 0014 — What the DMDc band means, and a grey-box cross-check

- Status: accepted
- Date: 2026-11-28
- Extends [ADR 0013](0013-dmdc-modal-bands.md)

## Context

ADR 0013 shipped a 5–95% band from 64 parametric-bootstrap refits of the fitted ARX model, but the
band is conditional on one trace and one fitted model. Two questions were left open:

1. Does the band track how repeatable a mode actually is from trace to trace?
2. Is the fastest pole's error — 25–34 s against a 25 s constant — a defect of the blind estimator
   or a real property of the identification problem?

## Decision

### The band is deliberately wider than the seed spread

The same fit was run on five seeds (1, 7, 42, `0x5eed7e11`, 12345). For each mode the spread of
the point estimates is compared with the median bootstrap band:

| Mode | Configured | Point estimates across seeds | Span | Median 5–95% band | Span / band |
|---|---|---|---|---|---|
| plate | 420 s | 407.4–442.2 s | 34.9 s | 229.3 s | 0.15 |
| junction | 75 s | 73.8–92.1 s | 18.3 s | 43.3 s | 0.42 |
| sensor | 25 s | 25.2–33.7 s | 8.5 s | 19.0 s | 0.45 |

The band is 2.2–6.7× wider than the seed-to-seed spread. That is the honest reading of what it is:
the parametric bootstrap resamples the fitted residuals and pushes them back through the recursion,
so it carries both observation noise and the fitted model's own misfit. It is a sensitivity band for
one trace, not a calibrated confidence interval across traces, and it is conservative rather than
too narrow. It must not be quoted as a probable range for the device's constants.

### A grey-box fit separates noise from structure

`greybox::fit_output_error` fits the same three-state chain by output error: the three time
constants and the initial temperature are the free parameters, the heater schedule and ambient
temperature are known inputs, and the residual is the observed series minus the sensor output
simulated from the candidate parameters. Levenberg–Marquardt runs on log time constants and the
initial temperature with a numerical Jacobian from a fixed start of 300/100/50 s, so the result is
an estimate rather than a refinement of the constants it is compared against.

The initial temperature is estimated rather than pinned to the first observation. With the first
sample as the starting state, its 0.04 K noise biased the plate pole by 1.2 s and the 5–95% band
(420.7–423.6 s) excluded the configured 420 s. Estimating the initial temperature moves the plate
pole to 420.7 s and the band to 419.9–421.5 s, which contains it.

Default seed, 12 iterations: 420.7/75.0/25.0 s against 420/75/25 s, worst modal error 0.17%, an
output-error RMS of 0.042 K against a 0.04 K sensor noise level, and an initial temperature of
293.151 K against the configured 293.15 K.

Across the same five seeds the fastest pole is:

| Estimator | Fastest pole | Worst error |
|---|---|---|
| blind Hankel-DMDc (OLS) | 25.2–33.7 s | 34.8% |
| grey-box output error | 24.8–25.7 s | 2.9% |

The sensor pole is recovered to a few percent, which locates the blind fit's 16–35% error in the
errors-in-variables bias rather than in the chain.

### The grey-box band is conditional on the structure

Each constant carries a 5–95% band from 64 residual-bootstrap refits: the output-error residuals
are resampled with replacement, added back to the fitted response, and the fit is repeated from the
point estimate. On the default seed the bands are 420.7 s (419.9–421.5), 75.0 s (73.8–76.3) and
25.0 s (24.1–25.9), and the widest band is 8.7% of its point estimate. Across the five seeds the
90% band contains the configured constant in 13 of 15 seed × mode cases.

The band carries observation noise only. It cannot carry the error from a wrong structure or a
wrong heater schedule, because every replicate is refitted to the same assumed model. That is why
the two bands are not averaged or reconciled: the DMDc band is wide because it carries the blind
fit's misfit, and this one is narrow because it does not.

### Both are reported; neither replaces the other

The grey-box fit assumes the structure it is testing. It cannot discover that the device is not a
three-state first-order chain, and it will happily return three constants for a model that does not
fit. The DMDc fit assumes nothing about the structure but is biased by measurement noise. S6 shows
the blind estimate with its band first, and the grey-box estimate beside it as a cross-check.

## Consequences

- S6 gains a grey-box τ card with its band, a mode ladder under the plot showing both estimators'
  constants and bands on a log axis, and a sentence stating what each estimator assumes; the
  accessible trace table gains the RTS and Matérn 1σ columns that were previously only drawn on the
  canvas.
- The DMDc band's meaning is stated on the page: sensitivity to the noise draw on one trace, not a
  cross-trace confidence interval.
- Four unit tests pin the comparison (`dmd::tests::bootstrap_band_is_measured_against_the_seed_spread`,
  `greybox::tests::fastest_pole_beats_the_blind_estimate_across_seeds`,
  `greybox::tests::bootstrap_band_brackets_the_point_estimate_and_the_truth` and
  `greybox::tests::bands_cover_the_configured_constants_across_seeds`).
- No measured data is introduced; the synthetic labels remain mandatory.
- Not claimed: a calibrated uncertainty for the grey-box estimate, a structure test, or any
  statement about a real device.
