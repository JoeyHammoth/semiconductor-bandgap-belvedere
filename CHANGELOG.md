# Changelog

## Unreleased

### Added

- Broad-prior M0 NPE experiments (affine and rational-quadratic spline) recorded as failed calibrations; the verified local v0.1.0 artifact is retained.
- First Phase 4 thermal-twin slice: deterministic plate/junction/sensor chain, WASM export and `/thermal/` S6 scene.
- S6 UKF/RTS state estimate with uncertainty band and fixed-hyperparameter GP residual diagnostic.
- S6 Kalman marginal likelihood and Hankel-DMDc time constants, with the fit residual and the modal error against the chain's configured constants.
- Three-pulse heater schedule shared by the forward twin, the UKF and DMDc; the slow and middle modal errors fall from 12–15% to 1–3%.
- S6 Matérn-3/2 latent discrepancy: a two-state linear SDE with exact discretisation, Kalman filter and RTS smoother, replacing the "future work" note for a state-space GP.
- Jeffreys hyperprior over the Matérn length and signal scales, fitted on an 8 × 6 grid with a moment-matched mixture and a reported 95% predictive coverage (95.6% on the default run).
- S6 residual strip plotting the post-smoother residual, the smoothed discrepancy and its 1σ band on a magnified axis.

### Changed

- DMDc keeps ordinary least squares: two-stage and total least squares were measured against it and both made the modal estimate worse; a new unit test pins the noise-free recovery.
- Offline-navigation Playwright coverage for the PWA service worker.

### Fixed

- Pin `@playwright/test` to 1.64.1 so the e2e suite stops hitting the Firefox 1572 content-process crash that follows an axe scan.
- Guided-story keyboard test no longer races hydration before pressing an arrow key.

## v0.1.0 — 2026-11-26

Phase 3 MVP: a calibrated local neural posterior, the Rust flow engine, an offline PWA shell and a guided story.

### Added

- JAX-trained conditional M0 flow with a committed affine export, manifest, golden vectors and calibration report.
- Rust `bgo-nn` flow engine, conditional affine and rational-quadratic spline coupling, and WASM `run_npe` protocol.
- `/npe/` page with five-summary inference, joint posterior canvas, accessible table, OOD gate and calibration receipt.
- `/story/` keyboard-operable guided tour.
- PWA manifest and SvelteKit 3 service worker with versioned precache.
- Permalink schema v2 with optional scene and case identifiers; v1 links still decode.
- ADR 0007 recording the local M0 NPE scope and the affine-vs-spline choice.

### Validation

- SBC: 1,000 synthetic simulations, 100 draws each; minimum p-value 0.936 (threshold 0.01).
- TARP: maximum deviation 0.021 (threshold 0.03).
- OOD: TPR 1.00, FPR 0.01 on the shifted-design fixture.
- `bgo-nn` golden-vector test, web unit suite, production build, size budgets and Chromium `/npe/` + `/story/` + PWA spec pass locally.

All data remains synthetic. The original recordings do not exist (ADR 0006); no measured-data or Table 1 reproduction claim is made.
