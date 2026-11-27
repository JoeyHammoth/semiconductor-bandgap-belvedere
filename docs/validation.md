# Validation plan

Nothing on the site is allowed to claim more than this document has measured. Each row states the claim, the test, the tolerance and where the evidence lives. A row is only marked **verified** when a CI run or committed reference artifact backs it.

Status legend: **pending** (no implementation yet), **wired** (test exists, no reference artifact yet), **verified** (measured, artifact committed), **unavailable** (the input needed for the claim no longer exists).

## 1. Regression against the source report

| Claim | Test | Tolerance | Status |
|---|---|---|---|
| M0 least-squares fit recovers a synthetic straight line | `cargo test -p bgo-physics fit::` | ≤ 1e-12 on intercept and slope | verified (unit) |
| M0 reproduces report Table 1 intercepts (Ge 0.646, Si 1.002, red 2.030, blue 2.796, green 2.303 eV) | none — requires the five lost recordings | n/a | unavailable (original logs lost) |
| Report Table 1 percentage errors recompute from its stated values | `cargo test -p bgo-physics --test report_arithmetic` | ≤ 0.01 percentage point | verified (unit) |
| Report wavelengths round from the measured gaps, except red | same test | blue/green ≤ 1 nm; red 613 vs 610.76 flagged | verified (unit) |
| The straight-line intercept is neither the 0 K nor the room-temperature gap | [`docs/physics-audit.md`](physics-audit.md) §3–4 + `scripts/physics-audit.py` | qualitative, citation-backed | verified (note) |

Table 1 intercept reproduction is not pending: it is **unavailable**. No authentic runs exist, and no reconstructed data will be used to manufacture a pass. See [`docs/data.md`](data.md) and [`docs/adr/0006-no-recorded-dataset.md`](adr/0006-no-recorded-dataset.md).

## 2. Physics and automatic differentiation

| Claim | Test | Tolerance | Status |
|---|---|---|---|
| AD first and second derivatives are exact | `cargo test -p bgo-ad`; seeded property tests vs five-point differences | rel. error ≤ 1e-8 (first), ≤ 1e-6 (second) | verified (unit) |
| Ideal limit reduces to the lab equation | `cargo test -p bgo-physics diode::tests::ideal_limit` | ≤ 1e-12 V | verified (unit) |
| `W(z)` satisfies `W·e^W = z` | `cargo test -p bgo-physics lambert_w::`; both branches, large and near-zero arguments | ≤ 1e-12 relative | verified (unit) |
| Lambert W reproduces the implicit diode equation | `cargo test -p bgo-physics diode::tests::lambert_form` | ≤ 1e-12 V | verified (unit) |
| Gap models return `E_g(0)` at 0 K and the audited values at 300 K | `cargo test -p bgo-physics gap::` | ≤ 5e-5 eV | verified (unit) |
| CIE fit and sRGB conversion follow the cited approximation | `cargo test -p bgo-physics colour::` | 555 nm y ±0.01; equal-energy white ±5e-4; 6500 K ±0.01 of D65 | verified (unit) |
| Two-current identity holds on noiseless data | `cargo test -p bgo-physics diode::tests::two_current` | ≤ 1e-10 V | verified (unit) |
| Hamiltonian is real symmetric | Hermiticity residual | ≤ 1e-12 | pending |

## 3. Samplers

| Claim | Test | Tolerance | Status |
|---|---|---|---|
| MCLMC matches NUTS on simulated M0–M4 | `cargo test -p bgo-infer --test reference_posteriors`; production defaults (4 chains, 5000 warmup, 4000 draws, target 0.995, 16 mean trajectory steps) against the committed NumPyro summaries | mean ≤ 0.05 σ, sd ≤ 0.10 σ, max correlation difference ≤ 0.10 | verified (CI; all six cases) |
| Coverage is nominal | 200+ simulated replications | 95% interval covers in 93–97% | pending |
| Nested sampling `ln Z` is correct | `cargo test -p bgo-infer --test nested_evidence`; closed-form Gaussian evidence | ≤ 0.2 nats | verified (integration) |
| Nested sampling agrees with `dynesty` | same test against committed [`tests/references/nested_gaussian_dynesty.json`](../crates/bgo-infer/tests/references/nested_gaussian_dynesty.json) | ≤ 0.3 nats | verified (integration; dynesty 2.1.5) |
| Replicate scatter matches theory | same test; six fixed seeds | observed sd within 0.5–2× of `sqrt(H/N)` | verified (integration) |
| Diagnostics are honest | `cargo test -p bgo-infer --test diagnostics_contract`; deliberately broken chains | R-hat/ESS/energy/divergence gates fail the converged badge | verified (unit) |
| M2 fixed-point target matches NumPyro | `cargo test -p bgo-infer --test target_parity`; latent transform, AD gradient and one log-density difference | gradient rel. error ≤ 1e-5; log-density difference ≤ 1e-5 | verified (unit) |
| M0–M4 reproduce from config + seed | `uv run pytest`; regenerate with `--all`; `uv run python -m bgo_train.reference --check`; `--compare` against the committed directory | config/seed hashes match; every summary has zero divergences, R-hat < 1.01, ESS > 400; refresh shifts stay within 0.05 sd (mean/quantiles), 10% (sd) and 0.02 (correlation) | verified (committed references + tolerance tests) |
| Energy-error trace is the recorded gate input | `cargo test -p bgo-wasm`; sampler trace is finite and has one value per post-warmup draw | trace length matches draws; no fabricated samples | verified (unit) |
| Posterior predictive is generated from posterior draws | `cargo test -p bgo-wasm`; M0 predictive intervals and S1 lines | finite values; q05 ≤ median ≤ q95; line count matches request | verified (unit) |
| Report-anchored cases are visibly synthetic | manifest and per-case `synthetic`, `anchor.label`, and notes | no measured-data or Table 1 reproduction claim | verified (committed references) |
| No reconstructed or hash-pinned raw run | manifest records config/seed hashes only; `data/manifest.json` remains empty | no data hash, no `data/raw` output | verified (audit + references) |

Phase 1 validation is simulation-only. The original recordings were lost (ADR 0006), so report-anchored synthetic cases are labelled and are not a real-data comparison.

The S5 NPE is a **local, report-anchored M0 amortised model**, not a calibrated surrogate for the full broad M0 prior. Its committed SBC/TARP/OOD report only certifies the local prior box `E_g ∈ [0.90, 1.10] eV`, `ln B ∈ [4.5, 6.5]`; outside it the OOD gate must reject the input. This limitation is visible on the page and is not hidden by the calibration badge.

A full-prior retrain was attempted on 2026-11-26 with the MCLMC prior (`E_g ∈ [0.05, 3.5]`, `ln B ~ Normal(8, 8)` clipped to `[−30, 30]`), 80,000 simulations and 12,000 steps. It **failed** the gate: SBC p-values were 0.054 and 3.6e-11, TARP passed at 0.023, and OOD TPR was 0.00. A second retry on 2026-11-27 used a 6-layer rational-quadratic spline flow (80,000 simulations, 8,000 steps, 59,250 weights); it also failed, with SBC p-values 3.3e-50 and 6.6e-10, TARP 0.023, and OOD TPR 0.00. Neither artifact was committed; the local verified model remains the shipped NPE. The experiments are recorded here so the broad-prior claim is not resurrected by a weaker test.

Nested sampling replaces a live point with a draw from the **uniform prior restricted to the likelihood threshold**, not from the likelihood-weighted posterior. The fixed-threshold tests check that this draw preserves uniform constrained means; the integration test checks the closed-form 2D Gaussian, the committed dynesty run and replicate scatter together.

Reference refresh is a **tolerance-aware** comparison, not a byte-identical diff: `--compare` accepts ordinary MCMC scatter when means and quantiles shift by at most 0.05 sd, standard deviations by at most 10%, and correlations by at most 0.02. Engine or config changes, failed diagnostics or larger shifts still fail the scheduled workflow. This does not relax the sampler-parity acceptance criteria.

## 4. Sloppiness and information geometry

| Claim | Test | Tolerance | Status |
|---|---|---|---|
| Fisher inverse matches posterior covariance | `cargo test -p bgo-info --test fisher_validation` against the committed M0 NUTS reference; committed table in [`fisher_m0_comparison.json`](../crates/bgo-info/tests/references/fisher_m0_comparison.json) | ≤ 10% on diagonal, ≤ 0.1 on correlations | verified (integration; max 2.3% sd, 0.0037 correlation) |
| Cramér–Rao bound respected | same test; 400 repeated M0 fits | empirical/Fisher sd in 0.9–1.1; correlation within 0.01 | verified (integration) |
| D-optimal design beats an even sweep | simulated benchmark suite | win rate reported; failures listed | pending |

## 5. Optimal transport, SVGD, calibration

| Claim | Test | Tolerance | Status |
|---|---|---|---|
| Sinkhorn matches exact 1D transport | analytic 1D fixture | ≤ 1e-6 after convergence | pending |
| Sliced-W2 noise floor is quantified | `cargo test -p bgo-ot`; Playwright S3 | bootstrap over random projections; reported with the point estimate | verified (unit + Chromium e2e) |
| SVGD converges on a Gaussian | `cargo test -p bgo-infer svgd` | mean within 0.05 σ; variance bias reported separately | verified (unit) |
| NPE matches the JAX reference and Rust golden vectors | `cargo test -p bgo-nn`; `python -m bgo_train.npe --check` | ≤ 1e-5 relative on log-prob | verified (unit + committed artifact) |
| SBC is uniform for the local M0 flow | 1000 synthetic simulations, 100 draws each; `web/static/data/npe-calibration.json` | chi-square p > 0.01 (observed 0.936 and 0.998) | verified (committed artifact) |
| TARP coverage is nominal for the local M0 flow | α grid 0.1–0.9 | max deviation ≤ 0.03 (observed 0.021) | verified (committed artifact) |
| OOD detector flags a shifted design | 300 shifted simulations; committed calibration report | TPR ≥ 0.9, FPR ≤ 0.02 (observed 1.00, 0.01) | verified (committed artifact) |
| Conformal coverage is nominal | held-out simulated runs | within 2% of nominal | pending |

## 6. State space and band structure

| Claim | Test | Tolerance | Status |
|---|---|---|---|
| UKF matches analytic Kalman filter | linear Gaussian model | ≤ 1e-8 | pending |
| RTS smoothing improves RMSE | simulated latent state | strictly lower RMSE | pending |
| DMDc recovers simulated eigenvalues | known plant | ≤ 1% on |λ| | pending |
| Si/Ge band edges match literature EPM | committed reference table | ≤ 0.1 eV, ordering exact | pending |
| DOS is normalised and converged | k-point refinement | integral drift ≤ 1% | pending |

## 7. UI, accessibility and budgets

| Claim | Test | Tolerance | Status |
|---|---|---|---|
| Landing page has no serious axe violations | Playwright + axe-core | zero serious/critical | verified (CI) |
| Lighthouse budgets hold | Lighthouse CI | thresholds in `lighthouserc.json` | verified (CI) |
| Size budgets hold | `scripts/size-budget.mjs` + its negative self-test | design §8.6 table; oversized asset fails | verified (CI) |
| Accent contrast holds | `scripts/contrast.mjs` in both themes | ≥ 4.5:1 for body text and links | verified (CI) |
| Permalinks round-trip | unit test | byte-identical state | verified (unit) |
| Base-path routing works | Playwright against `vite preview` | all routes reachable | verified (CI) |
| `/infer` serves the pinned synthetic M0 case | `web/src/lib/infer/case.test.ts`; SHA-256 and observation equality against `train/references/m0-report-si.json` | byte-pinned reference; no drift | verified (unit) |
| `/infer` labels the case synthetic | Playwright `/infer/`; visible synthetic chip and provenance note | no measured-data or Table 1 reproduction claim | verified (CI; e2e + a11y) |
| `/infer` renders only the Rust diagnostics verdict | Playwright Chromium production M0 run; badge `data-pass` from `SamplerResult` | pass badge only when Rust gates pass | verified (CI; e2e + a11y) |
| S1 renders posterior draws and predictive band | Playwright Chromium production M0 run; canvas labels and table | synthetic observations; 500 posterior curves | verified (CI; e2e + a11y) |
| `/infer` has no serious axe violations | Playwright + axe-core on `/infer/` | zero serious/critical | verified (CI; e2e + a11y) |
| S2 runs the Rust nested sampler and exposes a scrubbable trace | Playwright Chromium production `/evidence/`; `run_nested_gaussian` result | final `ln Z` and enabled trace scrubber | verified (Chromium e2e + a11y) |
| S4 recomputes Fisher information when the temperature range changes | Playwright `/evidence/`; low bound 220 → 260 K | summary and condition number update | verified (Chromium e2e + a11y) |
| `/evidence` has no serious axe violations | Playwright + axe-core on `/evidence/` | zero serious/critical | verified (Chromium e2e + a11y) |
| Evidence ledger publishes evidence, Bayes factor, PSIS-LOO SE, Pareto-k and prior-sensitivity flag | committed [`evidence-ledger.json`](../web/static/data/evidence-ledger.json); Playwright `/ledger/` | synthetic contract; M0 Pareto-k > 0.7 shown, not hidden | verified (artifact + Chromium e2e) |
| Committed evidence ledger matches a fresh regeneration | `.github/workflows/ledger.yml`; `python -m bgo_train.evidence --check` | engine/ArviZ and settings exact; `ln Z` 0.5 nat; ELPD 5; Pareto-k 0.1 | verified (CI; fresh regeneration) |
| S2 scrubber and S4 sliders are keyboard operable | Playwright `/evidence/`; Home/End and arrow keys | frame, temperature bounds and Fisher summary update | verified (CI e2e; Chromium, Firefox, WebKit) |
| S3 compares deterministic SVGD with MCLMC | Playwright `/evidence/`; S3 run and keyboard sliders | finite sliced-W2 with bootstrap SE, variance-ratio table, accessible equivalent | verified (unit + Chromium e2e) |
| `/ledger` has no serious axe violations | Playwright + axe-core on `/ledger/` | zero serious/critical | verified (Chromium e2e + a11y) |
| `/npe` runs the frozen local M0 flow and labels OOD | Playwright Chromium `/npe/`; `data-testid="npe"`, `npe-ood`, accessible table | draws and passing calibration report visible; OOD state explicit | verified (Chromium e2e) |
| `/npe` has no serious axe violations | Playwright + axe-core on `/npe/` | zero serious/critical | verified (Chromium e2e + a11y) |
| Guided story is keyboard operable | Playwright `/story/`; ArrowLeft/ArrowRight, retried until hydration attaches the handler | heading advances and returns | verified (CI e2e; Chromium, Firefox, WebKit) |
| PWA manifest and service worker are served | Playwright `/`; manifest link and `navigator.serviceWorker`; production build | manifest 200, worker registered under the configured base path | verified (Chromium e2e) |
| PWA serves an offline navigation after activation | Playwright Chromium: wait for `serviceWorker.ready` and an active controller, `context.setOffline(true)`, fetch `/thermal/` through the worker | worker answers 200 with the cached S6 heading and run control while the network is disabled | verified (Chromium e2e) |
| Thermal chain recovers the configured sensor lag | `cargo test -p bgo-ssm thermal::recovers_the_sensor_time_constant` | relative error ≤ 2% | verified (unit) |
| UKF/RTS recovers the latent junction from the noisy sensor | `cargo test -p bgo-ssm ukf::tests::smoother_recovers_the_latent_junction` | smoothed junction RMSE ≤ 0.1 K | verified (unit) |
| GP residual diagnostic is finite and non-negative | `cargo test -p bgo-ssm ukf::tests::gp_discrepancy_is_finite` | finite mean, finite non-negative standard deviation | verified (unit) |
| Kalman marginal likelihood responds to a noise mismatch | `cargo test -p bgo-ssm ukf::tests::marginal_likelihood_penalises_a_mismatched_noise_level` | matched-noise score exceeds a 10× tighter filter by more than 10 nats | verified (unit) |
| Hankel-DMDc recovers the chain's damping constants | `cargo test -p bgo-ssm dmd::tests::recovers_the_chain_time_constants` | three distinct positive modes, maximum relative error ≤ 20% of the configured constants, fit residual ≤ 5% | verified (unit) |
| S6 runs the Rust thermal twin and exposes RTS/GP outputs | Playwright Chromium `/thermal/`; `data-testid="thermal"`, `thermal-smoother`, `thermal-gp`, table | finite recovered lag, RTS/GP metrics, plot and accessible table | verified (Chromium e2e + a11y) |
| S6 shows the filtered likelihood and DMDc modes | Playwright `/thermal/`; `thermal-likelihood`, `thermal-modes` | finite log-likelihood and three time constants rendered next to the chain's constants | verified (Chromium e2e + a11y) |

## 8. How to run

```bash
# Rust
cargo test --workspace

# Web
cd web && npm ci && npm run check && npm run build && npm run test

# Reference refresh (offline, needs Python)
cd train && uv run python -m bgo_train.reference --all
cd train && uv run python -m bgo_train.reference --check

# Evidence ledger (synthetic M1; dynesty evidence + NUTS/PSIS-LOO)
cd train && uv run --with dynesty==3.2.0 python -m bgo_train.evidence
cd train && uv run --with dynesty==3.2.0 python -m bgo_train.evidence --check
```

`web/package.json` pins `@playwright/test` to 1.64.1. Playwright 1.65.0 and 1.65.1 ship Firefox build 1572, which segfaults in its content process when `ariaLabelledByElements` is read from both the page world and Playwright's isolated world — exactly what an `@axe-core/playwright` scan followed by a locator lookup does ([playwright#43967](https://github.com/microsoft/playwright/issues/43967)). The fix is in Firefox build 1575+, which reaches stable with Playwright 1.66; the pin can be lifted then, together with the `playwright-core` override that keeps axe and the runner on one Playwright type tree.

Reference artifacts are regenerated by `.github/workflows/refs.yml` on a schedule or manually; a regression opens an issue with the diff attached. The evidence ledger is checked against a fresh regeneration by `.github/workflows/ledger.yml` on every pull request and main push, and monthly on the default branch.

## 9. Reporting rules

- Tolerances may only be relaxed in a PR that explains why and links to the failing evidence.
- A test that is skipped must say why in the code and in this table.
- A claim whose input no longer exists is marked **unavailable**; it must not be revived with reconstructed data.
- "Verified" requires a committed artifact or a green CI run referenced from this document.
