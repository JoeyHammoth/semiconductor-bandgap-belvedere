# ADR 0011 — Multi-pulse heater schedule and a Matérn-3/2 latent discrepancy

- Status: accepted
- Date: 2026-11-28
- Supersedes the "future work" notes in [ADR 0009](0009-ukf-rts-gp-residual.md) and
  [ADR 0010](0010-dmdc-and-marginal-likelihood.md)
- Superseded in part by [ADR 0012](0012-dmdc-noise-sensitivity-and-matern-hyperprior.md): the
  Matérn scales are now fitted on a grid hyperprior, and the DMDc estimator comparison is recorded
  there

## Context

ADR 0010 fitted Hankel-DMDc to a single heater step. That trace left the middle and slow poles off
by 12–15%, so the modal estimate was reporting how badly one step conditions the regression rather
than what the chain does. ADR 0009's GP was an RBF kernel fitted to the post-smoother residual: a
diagnostic over the whole trace, not a state of the model. The last Phase 4 task was the state-space
GP discrepancy.

## Decision

### Three-pulse heater schedule

`ThermalConfig.heat_pulses` is an ordered array of `HeatPulse { start_s, stop_s }`. The default
schedule turns the heater on for 90 s at 60 s, 300 s at 300 s and 720 s at 780 s of an 1800 s run,
with 10 s samples. The forward twin, the UKF transition, the DMDc control regressors and the page
all read the same `heater_target_k`, so the estimation model cannot silently disagree with the trace
it is given. Validation rejects unordered or overlapping pulses, and the stop edge is exclusive.

The narrow first pulse puts energy at the sensor's own timescale; the long third pulse drives the
plate towards steady state. On the default seed the DMDc poles move from 475/64/28 s to 433/74/29 s
against the chain's 420/75/25 s: the slow and middle relative errors fall from about 13–15% to
1.4% and 3.1%, and the fit residual from 1.10% to 0.96%.

The fastest pole stays about 16% long. That is not a schedule problem: observation noise whitens the
fastest pole, which is the errors-in-variables bias of fitting an ARX model to a noisy series. A 5 s
sample interval was tried and is worse — the slow pole drifts to about 600 s — so the 10 s interval
stays.

The DMDc residual now includes the control terms. Before, it was the residual of the autoregression
alone while the page called it the fitted model's residual.

### Matérn-3/2 latent discrepancy

`bgo-ssm::matern` carries the post-smoother residual as a two-state linear SDE. A Matérn-3/2 process
with length scale `ℓ` and stationary variance `σ²` is exactly the stationary distribution of

```
d/dt [f, f'] = [[0, 1], [-λ², -2λ]] [f, f'] + [0, 1]ᵀ w,   λ = √3/ℓ,   q = 4λ³σ²,
```

so the discrepancy is a model state, not a kernel matrix over the trace:

- the transition is the exact matrix exponential, so the discretisation is unbiased at any step;
- the process noise `∫₀^Δ Φ(s) G q Gᵀ Φ(s)ᵀ ds` is integrated with eight-point Gauss–Legendre
  quadrature;
- the filter starts from the stationary covariance `diag(σ², λ²σ²)` and runs a two-state Kalman
  filter followed by an RTS smoother, `O(1)` per sample.

Hyperparameters were fixed at the same values as the RBF diagnostic — 180 s length scale, 0.20 K
signal standard deviation — with the measurement noise taken from the configured sensor noise. The
smoothed discrepancy had an RMS of about 8 mK on the default run, with a mean standard deviation of
about 18 mK. ADR 0012 replaced the fixed scales with a grid hyperprior; the fitted signal scale
lands at the floor of that grid on this trace.

It is not a learned hyperparameter fit, not calibrated, and not evidence over models.

## Consequences

- S6 shows the Matérn discrepancy RMS in a `thermal-matern` card next to the RBF residual, the
  Kalman log-likelihood and the DMDc constants, so the two treatments of the residual sit side by
  side.
- The RBF diagnostic is kept: the kernel fit smooths the residual over the trace, the SDE treats it
  as a state that is propagated and updated. Removing the kernel would lose the comparison.
- Fixed hyperparameters mean the discrepancy band is a prior on the residual, not a posterior over
  length scales. A hierarchical fit would be separate work with its own validation.
- The fastest DMDc pole remains biased, and the page says so rather than rounding it away.
- No measured data is introduced; the synthetic labels remain mandatory.
