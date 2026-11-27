# ADR 0007 — Ship a local, affine M0 neural posterior

- Status: accepted
- Date: 2026-11-26

## Context

The S5 MVP needs an instant posterior for the report-anchored synthetic M0 case. The full M0 prior spans `E_g` 0.05–3.5 eV and `ln B` −30–30, while the case posterior is a thin ridge close to `E_g = 1.002 eV`, `ln B = 5.504`. A conditional spline flow trained on 40,000 full-prior simulations underfit the near-singular covariance: its marginal standard deviations were several times too wide and SBC rejected it.

The Rust flow engine supports conditional affine and rational-quadratic spline coupling. The spline remains the intended engine for wider, more nonlinear posteriors.

## Decision

Ship a **local** M0 amortised model over

- `E_g ∈ [0.90, 1.10]` eV
- `ln B ∈ [4.5, 6.5]`

using a conditional affine flow with six layers. The condition is the five-summary vector `[mean, slope, curvature, residual sd, endpoint drop]`. The committed calibration report covers this local prior only. Inputs outside it are rejected by the Mahalanobis OOD gate.

The page labels the model as local and synthetic, and does not present the passes as validation of the broad M0 prior or of measured data.

## Consequences

- The local model passes SBC (p = 0.936, 0.998), TARP (max deviation 0.021) and OOD (TPR 1.00, FPR 0.01).
- The thawed broad-prior NPE remains future work; MCLMC is still the reference for that claim. A 2026-11-26 retrain over the full prior failed its SBC and OOD gates, as did a 2026-11-27 rational-quadratic spline retry, so neither was published (see `docs/validation.md`).
- The spline implementation and tests stay in `bgo-nn` for later models.
- Recalibration is required if the prior box, summary definitions, design or datasets change.
