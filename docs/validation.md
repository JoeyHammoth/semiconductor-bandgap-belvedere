# Validation plan

Nothing on the site is allowed to claim more than this document has measured. Each row states the claim, the test, the tolerance and where the evidence lives. A row is only marked **verified** when a CI run or committed reference artifact backs it.

Status legend: **pending** (no implementation yet), **wired** (test exists, no reference artifact yet), **verified** (measured, artifact committed).

## 1. Regression against the source report

| Claim | Test | Tolerance | Status |
|---|---|---|---|
| M0 reproduces report Table 1 intercepts (Ge 0.646, Si 1.002, red 2.030, blue 2.796, green 2.303 eV) | `cargo test -p bgo-physics --test report_table1` | within the report's own max–min uncertainty, and ≤ 0.02 eV absolute | pending |
| Report wavelengths follow from its measured gaps | unit test on `1239.84 / E_g` | ≤ 1 nm | pending |
| The straight-line intercept is neither the 0 K nor the room-temperature gap | [`docs/physics-audit.md`](physics-audit.md) §3–4 + `scripts/physics-audit.py` | qualitative, citation-backed | verified (note) |

## 2. Physics and automatic differentiation

| Claim | Test | Tolerance | Status |
|---|---|---|---|
| AD first and second derivatives are exact | property tests vs finite differences on random inputs | rel. error ≤ 1e-8 (first), ≤ 1e-6 (second) | pending |
| Ideal limit reduces to the lab equation | symbolic/numeric comparison | ≤ 1e-12 V | pending |
| `W(z)` satisfies `W·e^W = z` | property test over branches | ≤ 1e-12 relative | pending |
| Two-current identity holds on noiseless data | simulated fixture | ≤ 1e-10 V | pending |
| Hamiltonian is real symmetric | Hermiticity residual | ≤ 1e-12 | pending |

## 3. Samplers

| Claim | Test | Tolerance | Status |
|---|---|---|---|
| MCLMC matches NUTS on M0–M4 | golden posteriors from NumPyro (`refs.yml`) | mean ≤ 0.05 σ, sd ≤ 0.10 σ, max correlation difference ≤ 0.10 | pending |
| Coverage is nominal | 200+ simulated replications | 95% interval covers in 93–97% | pending |
| Nested sampling `ln Z` is correct | closed-form Gaussian evidence | ≤ 0.2 nats | pending |
| Nested sampling agrees with `dynesty` | real data, committed dynesty output | ≤ 0.3 nats | pending |
| Replicate scatter matches theory | several independent runs | observed sd within 2× of `sqrt(H/N)` | pending |
| Diagnostics are honest | deliberately broken chains | R-hat/ESS flags fail the converged badge | pending |

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
| Landing page has no serious axe violations | Playwright + axe-core | zero serious/critical | pending |
| Lighthouse budgets hold | Lighthouse CI | thresholds in `lighthouserc.json` | pending |
| Size budgets hold | `scripts/size-budget.mjs` | design §8.6 table | pending |
| Accent contrast holds | computed WCAG ratio in CI | ≥ 4.5:1 for body text | pending |
| Permalinks round-trip | unit test | byte-identical state | pending |
| Base-path routing works | Playwright against `vite preview` | all routes reachable | pending |

## 8. How to run

```bash
# Rust
cargo test --workspace

# Web
cd web && npm ci && npm run check && npm run build && npm run test

# Reference refresh (offline, needs Python)
cd train && uv run python -m bgo_train.reference --config configs/m0.toml
```

Reference artifacts are regenerated by `.github/workflows/refs.yml` on a schedule or manually; a regression opens an issue with the diff attached.

## 9. Reporting rules

- Tolerances may only be relaxed in a PR that explains why and links to the failing evidence.
- A test that is skipped must say why in the code and in this table.
- "Verified" requires a committed artifact or a green CI run referenced from this document.
