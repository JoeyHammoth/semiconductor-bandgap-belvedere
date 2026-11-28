# ADR 0010 — Hankel-DMDc time constants and a Kalman marginal likelihood

- Status: accepted
- Date: 2026-11-28
- Supersedes the "future work" note in [ADR 0009](0009-ukf-rts-gp-residual.md)
- Superseded in part by [ADR 0011](0011-multi-pulse-schedule-and-matern-state.md), which replaces
  the single heater step with a three-pulse schedule and makes the control terms part of the DMDc
  fit residual

## Context

ADR 0009 added the three-state UKF, the RTS smoother and a GP residual diagnostic to the S6
thermal twin, and deferred the Kalman marginal likelihood and Hankel-DMDc time constants. Both are
small enough to land without a new data source or a new dependency, and both complete the system
identification story the roadmap promised under issue #5.

The risk is over-claiming. A marginal likelihood is not model evidence, and a modal estimate from
one heater step is not a validated identification of the device.

## Decision

### Kalman marginal likelihood

The UKF's update step already forms the innovation and its predicted variance. Each step now also
returns the Gaussian log density of the observation under that predicted measurement, and the
filter sums them. The result is the filter's own predictive log-likelihood for the configured
model and noise level — the Kalman prediction-error decomposition.

It is reported as `kalmanLogLikelihood`. It is not a marginal likelihood over parameters, not a
Bayes factor, and not comparable between runs with different assumed noise without saying so.

### Hankel-DMDc time constants

The estimate fits `x_{k+1} = A x_k + B u_k` to a delay-embedded observable, where `x` is the
Hankel state of the observed sensor and `u` is the known heater target:

- embedding order 4: the three chain states plus one companion mode that the chain's direct
  feedthrough needs;
- three control lags, so the heater has the same memory as the chain;
- least squares by Householder QR on the mean-centred series — the delay columns are strongly
  collinear, and normal equations square an already large condition number;
- eigenvalues from the companion polynomial by Durand–Kerner iteration, then
  `τ = −Δt / ln|λ|` and `ω = arg(λ) / Δt`;
- modes closer than 5% in time constant are merged, and the three slowest distinct modes are
  returned with the fit's relative residual and the largest relative error against the chain's
  configured constants.

## Consequences

- S6 shows the filtered log-likelihood and the three DMDc time constants next to the chain's own
  constants, together with the fit residual and the maximum relative error. Disagreement is
  displayed, not hidden.
- The single heater step excites the plate strongly and the fastest mode weakly. The recovered
  constants land within roughly 15% of the chain's values; the unit test pinned a 20% bound rather
  than pretending to a tighter identification. ADR 0011 later replaced the single step with a
  three-pulse schedule and tightened the bound to 18%.
- Fitting the RTS-smoothed sensor instead of the raw observation was tried and rejected: the
  smoother is model-based, so its estimate is over-smoothed and the modal fit is worse (and
  circular). The raw observed series is the input.
- No learned discrepancy model, Matern-SDE state, Hankel SVD truncation or model-order search is
  claimed. Those remain separate work.
- No measured data is introduced; the synthetic labels remain mandatory.
