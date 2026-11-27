# ADR 0009 — Add UKF/RTS smoothing and a GP residual diagnostic to S6

- Status: accepted
- Date: 2026-11-27

## Context

ADR 0008 shipped a deterministic plate → junction → sensor chain so the S6 scene had an auditable
reference before adding inference. The next Phase 4 step is to estimate the hidden junction from
the noisy sensor and to expose model discrepancy without pretending the synthetic trace is a
measured device.

The full M5 target is a Kalman marginal likelihood, a state-space GP discrepancy and DMDc time
constants. Implementing all three at once would mix estimator error, discrepancy-model error and
system-identification error in one scene.

## Decision

Add two bounded pieces to the existing S6 slice:

1. A three-state unscented Kalman filter with state `[plate, junction, sensor]`, the exact
   sequential first-order transition used by the forward twin, the known heater control, and the
   noisy sensor as the only measurement.
2. A Rauch–Tung–Striebel smoother over the same trace, returning the smoothed junction and its
   marginal standard deviation.
3. A Gaussian-process residual diagnostic fitted after smoothing. It uses a fixed RBF length
   scale, fixed signal variance and the configured sensor noise. It is a diagnostic of the
   post-smoother sensor residual, not a latent state and not a learned discrepancy prior.

The WASM boundary returns the filtered junction, RTS junction, RTS standard deviation, GP mean and
standard deviation, and the junction/sensor RMSEs. The S6 page draws the RTS line, the RTS 1σ band
and the GP residual band, and includes them in the accessible table.

## Consequences

- The scene now demonstrates a real filter/smoother on the synthetic trace while keeping the
  deterministic chain visible as the reference.
- The GP hyperparameters are fixed and are not optimized against the data; the reported GP band is
  a diagnostic, not a calibrated uncertainty claim.
- The Kalman marginal likelihood, a state-space GP and Hankel-DMDc time constants remain future
  Phase 4 work.
- No measured data is introduced and the synthetic labels remain mandatory.
