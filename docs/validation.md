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
| MCLMC matches NUTS on simulated M0–M4 | `cargo test -p bgo-infer --test reference_posteriors`; production defaults (4 chains, 5000 warmup, 4000 draws, target 0.995, 16 mean trajectory steps) against the committed NumPyro summaries | mean ≤ 0.05 σ, sd ≤ 0.10 σ, max correlation difference ≤ 0.10 | verified (unit; all six cases pass locally) |
| Coverage is nominal | 200+ simulated replications | 95% interval covers in 93–97% | pending |
| Nested sampling `ln Z` is correct | closed-form Gaussian evidence | ≤ 0.2 nats | pending |
| Nested sampling agrees with `dynesty` | simulated data, committed dynesty output | ≤ 0.3 nats | pending |
| Replicate scatter matches theory | several independent runs | observed sd within 2× of `sqrt(H/N)` | pending |
| Diagnostics are honest | `cargo test -p bgo-infer --test diagnostics_contract`; deliberately broken chains | R-hat/ESS/energy/divergence gates fail the converged badge | verified (unit) |
| M2 fixed-point target matches NumPyro | `cargo test -p bgo-infer --test target_parity`; latent transform, AD gradient and one log-density difference | gradient rel. error ≤ 1e-5; log-density difference ≤ 1e-5 | verified (unit) |
| M0–M4 reproduce from config + seed | `uv run pytest`; regenerate with `--all`; `uv run python -m bgo_train.reference --check` | config/seed hashes match and every committed summary passes zero divergences, R-hat < 1.01, ESS > 400 | verified (committed references) |
| Energy-error trace is the recorded gate input | `cargo test -p bgo-wasm`; sampler trace is finite and has one value per post-warmup draw | trace length matches draws; no fabricated samples | verified (unit) |
| Posterior predictive is generated from posterior draws | `cargo test -p bgo-wasm`; M0 predictive intervals and S1 lines | finite values; q05 ≤ median ≤ q95; line count matches request | verified (unit) |
| Report-anchored cases are visibly synthetic | manifest and per-case `synthetic`, `anchor.label`, and notes | no measured-data or Table 1 reproduction claim | verified (committed references) |
| No reconstructed or hash-pinned raw run | manifest records config/seed hashes only; `data/manifest.json` remains empty | no data hash, no `data/raw` output | verified (audit + references) |

Phase 1 validation is simulation-only. The original recordings were lost (ADR 0006), so report-anchored synthetic cases are labelled and are not a real-data comparison.

## 4. Sloppiness and information geometry

| Claim | Test | Tolerance | Status |
|---|---|---|---|
| Fisher inverse matches posterior covariance | near-Gaussian simulated regime | ≤ 10% on diagonal, ≤ 0.1 on correlations | pending |
| Cramér–Rao bound respected | repeated simulated fits | no systematic violation beyond Monte Carlo error | pending |
| D-optimal design beats an even sweep | simulated benchmark suite | win rate reported; failures listed | pending |

## 5. Optimal transport, SVGD, calibration

| Claim | Test | Tolerance | Status |
|---|---|---|---|
| Sinkhorn matches exact 1D transport | analytic 1D fixture | ≤ 1e-6 after convergence | pending |
| Sliced-W2 noise floor is quantified | bootstrap | reported with the point estimate | pending |
| SVGD converges on a Gaussian | analytic target | mean within 0.05 σ; variance bias documented | pending |
| NPE matches PyTorch golden vectors | exported vectors | ≤ 1e-5 relative on log-prob | pending |
| SBC is uniform | 1000 simulations | chi-square p > 0.01 | pending |
| TARP coverage is nominal | α grid | max deviation ≤ 0.03 | pending |
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
| `/infer` labels the case synthetic | Playwright `/infer/`; visible synthetic chip and provenance note | no measured-data or Table 1 reproduction claim | wired (local Playwright) |
| `/infer` renders only the Rust diagnostics verdict | Playwright Chromium production M0 run; badge `data-pass` from `SamplerResult` | pass badge only when Rust gates pass | wired (local Playwright) |
| S1 renders posterior draws and predictive band | Playwright Chromium production M0 run; canvas labels and table | synthetic observations; 500 posterior curves | wired (local Playwright) |
| `/infer` has no serious axe violations | Playwright + axe-core on `/infer/` | zero serious/critical | verified (CI; e2e + a11y) |

## 8. How to run

```bash
# Rust
cargo test --workspace

# Web
cd web && npm ci && npm run check && npm run build && npm run test

# Reference refresh (offline, needs Python)
cd train && uv run python -m bgo_train.reference --all
cd train && uv run python -m bgo_train.reference --check
```

Reference artifacts are regenerated by `.github/workflows/refs.yml` on a schedule or manually; a regression opens an issue with the diff attached.

## 9. Reporting rules

- Tolerances may only be relaxed in a PR that explains why and links to the failing evidence.
- A test that is skipped must say why in the code and in this table.
- A claim whose input no longer exists is marked **unavailable**; it must not be revived with reconstructed data.
- "Verified" requires a committed artifact or a green CI run referenced from this document.
