# Glossary

Terms used in code, issues and public copy. Keep this list short and precise; if a term is not here, it should not appear in a scene label.

## Physics

- **Band gap (`E_g`)** — energy between the valence-band maximum and conduction-band minimum. In this project, always state *which* gap and *at what temperature*: fitted intercept, 0 K extrapolation and room-temperature value are different numbers.
- **Varshni relation** — empirical `E_g(T) = E_g(0) − αT²/(T+β)`.
- **Bose–Einstein gap** — physically motivated single-effective-phonon form from Viña et al. (1984).
- **Ideality factor (`n`)** — deviation of the diode from the ideal Shockley exponential.
- **Series resistance (`R_s`)** — parasitic resistance that bends the `I–V` curve at high current.
- **Saturation current (`I_s`)** — extrapolated reverse current; carries the gap dependence.
- **PTAT** — proportional to absolute temperature; the two-current diode identity is PTAT and independent of `E_g`.
- **Lambert W** — inverse of `x·e^x`; gives the closed-form `I(V)` with series resistance.
- **EPM** — empirical pseudopotential method; here in real-symmetric form for Si and Ge.
- **KPM** — Kernel Polynomial Method; Chebyshev expansion for densities of states in large disordered supercells.
- **DOS** — density of states.

## Inference

- **Sloppiness** — a Fisher-information spectrum spanning many orders of magnitude, so some parameter combinations are tightly constrained and others hardly at all.
- **Fisher information** — `F = JᵀΣ⁻¹J`; its inverse bounds parameter covariance.
- **Cramér–Rao bound** — lower bound on estimator variance; used as a cross-check on Posteriors.
- **MCLMC** — microcanonical Langevin/Hamiltonian Monte Carlo; dynamics on a constant-energy surface with unit-norm velocity.
- **NUTS** — No-U-Turn Sampler; offline reference only.
- **Nested sampling** — evidence estimator that integrates likelihood over prior volume by shrinking live points.
- **Evidence (`Z`)** — marginal likelihood; the basis for Bayes factors and Occam penalties.
- **Occam factor** — automatic penalty for prior volume, reported as information `H`.
- **SVGD** — Stein variational gradient descent; deterministic particle inference with attraction and repulsion.
- **Sinkhorn** — entropic-regularised optimal transport, used for posterior morphing.
- **Sliced Wasserstein-2** — average 1D Wasserstein distance over random projections; a cheap posterior-agreement metric.
- **NPE** — neural posterior estimation; amortised inference with a conditional density estimator.
- **Rational-quadratic spline flow** — invertible transform used by the NPE density estimator.
- **SBC** — simulation-based calibration; rank histograms test posterior calibration.
- **TARP** — expected coverage test for posterior estimators.
- **PSIS-LOO** — Pareto-smoothed importance-sampling leave-one-out cross-validation.
- **EIG** — expected information gain; the objective for the next-measurement recommender.
- **D-optimal design** — design that maximises `log det F`.
- **UKF / RTS** — unscented Kalman filter and Rauch–Tung–Striebel smoother for the thermal state.
- **DMDc** — dynamic mode decomposition with control; data-driven Koopman eigenvalues.

## Project terms

- **Scene (S0–S10)** — one self-contained visual module with a documented data source and an accessible equivalent.
- **Ledger (S10)** — the evidence comparison table; also the accessible form of the model comparison.
- **Receipts** — footer showing commit hash, data hash, test status and bundle size.
- **Degradation path** — the documented fallback chain when WebGPU, workers or WASM are unavailable.
- **Simulator mode** — the site running on synthetic data because real data cannot yet be published.
- **Sloppy axis** — the Fisher eigenvector along which the posterior is widest; the target of experiment design.
