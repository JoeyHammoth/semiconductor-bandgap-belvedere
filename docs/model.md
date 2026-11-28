# Model specification

This is the normative description of the physics and the statistical model. If code and this document disagree, one of them is a bug. The originating design document is [`source/bandgap-observatory-design.md`](../source/bandgap-observatory-design.md); this file extracts the parts that must not be re-invented per module.

Status: extracted from design v2. Parameter provenance is audited in [`docs/physics-audit.md`](physics-audit.md); numbers there with a **parked** verdict must not reach public copy.

## 1. Forward model (constant-current direction)

```
V = I·R_s + n·V_T·ln(I / I_s(T) + 1),        V_T = k·T / q
I_s(T) = B·T^η·exp( −E_g(T) / (n_a·k·T) )
```

Ideal limit (`n = n_a = 1`, `R_s = 0`, `I ≫ I_s`):

```
V(T) = (k·T/q)·[ln(I/B) − η·ln T] + E_g(T)/q
```

With `η = 0` and constant `E_g`, this reduces to the lab's straight line (report Eq. 8), so the fitted intercept is `E_g/q` only under those assumptions. That distinction is the scientific point of the project, not an implementation detail.

### Symbols

| Symbol | Meaning | Units |
|---|---|---|
| `V` | forward voltage across the diode | V |
| `I` | forward current held by the current source | A |
| `T` | junction temperature | K |
| `E_g(T)` | band-gap energy | eV |
| `B` | prefactor (material and device dependent) | A·K^-η |
| `η` | prefactor temperature exponent | dimensionless |
| `n` | ideality factor | dimensionless |
| `n_a` | exponent in the saturation-current activation | dimensionless |
| `R_s` | series resistance | Ω |
| `k`, `q` | Boltzmann constant, elementary charge | J/K, C |

Internally all quantities are SI; `E_g` is stored in eV only at presentation boundaries. Conversions live in `bgo-physics::units`.

## 2. Gap models

Varshni:

```
E_g(T) = E_g(0) − α·T² / (T + β)
```

Bose–Einstein (single effective phonon; gap form from O'Donnell and Chen 1991, after the critical-point treatment of Viña, Logothetidis and Cardona 1984 — see [`docs/physics-audit.md`](physics-audit.md) §6):

```
E_g(T) = E_g(0) − 2·a_B / (exp(Θ_B / T) − 1),        a_B = S⟨ħω⟩
```

`E_g(0)` is the 0 K gap. The design's form `E_B − a_B[1 + 2/(exp(Θ_B/T) − 1)]` is the same curve only when `E_B = E_g(0) + a_B`; `E_B` is **not** the 0 K gap and must not be labelled as one.

The two are alternatives, not nested. `M2` and `M2b` are compared by evidence and by out-of-sample prediction, never by a likelihood-ratio test.

## 3. Model variants

| ID | Description | Free parameters (beyond noise) |
|---|---|---|
| M0 | Lab model: linear `V`–`T`, ideal diode | `E_g`, `ln B` |
| M1 | + prefactor `T^η` | + `η` |
| M2 | + Varshni `E_g(T)` | + `α`, `β` |
| M2b | Bose–Einstein `E_g(T)` instead of Varshni | + `a_B`, `Θ_B` |
| M3 | + ideality `n` | + `n` |
| M4 | + series resistance `R_s` | + `R_s` |
| M5 | + sensor lag / latent junction temperature | + thermal states |
| M6 | + state-space GP discrepancy | + GP hyperparameters |

Every variant shares the same data, likelihood and prior construction so that evidence comparisons are meaningful.

Reference convention for M3/M4: with one constant-current temperature sweep, `η` and the terminal ideality `n` are nearly degenerate. The committed M3/M4 reference cases therefore hold `η` at the illustrative central value 3.0 and infer `n` (and, for M4, `R_s`); M1/M2 infer `η`. This is a stated simulation convention, not a measured constraint. A future multi-current design can free both without that degeneracy.

### 3.1 Sampling coordinates

The browser sampler does not propose directly in `(E_g(0), α, β, n, R_s)`. For M2–M4 it samples the gap at the two temperature endpoints, `E_g(T_low)` and `E_g(T_high)`, plus `β`. For M3/M4 it also samples `n·E_g` and `n·ln B_ref` so ideality does not create a multiplicative ridge. M2 samples `η` through a probit transform on `[1.5, 5]` with its Jacobian, so a proposal cannot leave the truncated prior and be logged as a spurious divergence. The change of variables is exact and its Jacobian is included in the target; `reported` maps every draw back to the reference parameter names.

M4 has an additional near-degeneracy: at fixed current, `R_s` enters the voltage as the constant offset `I·R_s`, while `E_g(0)` also shifts the whole curve. The sampler therefore uses

```
E_g_eff = E_g(0) + I·R_s
```

as the effective gap and samples `R_s` through the prior-whitened coordinate

```
R_s = 10·Φ⁻¹((1 + Φ(z))/2),    z ~ Normal(0, 1),
```

where `Φ` is the standard-normal CDF. This is the half-normal(10) quantile transform of `Φ(z)`; the standard-normal prior on `z` is the transformed `R_s` prior, so no extra `ln R_s` Jacobian is added. The likelihood still uses the physical `R_s`. M2/M3/M4 also use a pilot run for covariance adaptation during warmup; the pilot draws are discarded.

The production sampler defaults are 4 chains, 5000 warmup draws per chain, 4000 retained draws per chain, target acceptance 0.995 and a mean trajectory length of 16 integrator steps. These are validation settings, not a claim that every model is cheap: M4 is the slowest case.

## 4. Instrument model

- **ADC (ADS1115):** signed 16-bit, `LSB = FSR / 32768` (125 µV at ±4.096 V); quantisation `σ_q = LSB/√12` plus a fitted noise floor.
- **Current:** `I = V_Rtemp / R_temp` with propagated uncertainty from resistor tolerance.
- **LM234 thermometer:** `T = I_set·R_set / γ`, `γ ≈ 227 µV/K` at 293 K, fitted per device.
- **Thermocouple and multimeter:** offset/gain priors from the meter datasheet **[fill in]**, plus a first-order lag.
- **Current-source drift:** slow multiplicative drift of `I_set`, measured per step where possible.

## 5. Priors

| Parameter | Prior | Notes |
|---|---|---|
| `E_g(0)` | wide Normal (weak) or literature-centred (informative) | both variants reported with a sensitivity flag |
| `ln B` | weak Normal scaled by simulation | scale recorded with every run |
| `η` | Normal(3, 1), truncated to [1.5, 5] | illustrative, not derived: `η = 3 + γ/2` and γ is unmeasured (audit §9). The design's σ = 0.5 is too tight — ±0.5 in η moves the M0 intercept ±15 mV, more than the report's quoted uncertainty |
| `n` | LogNormal around 1, truncated at `n ≥ 1` | by construction |
| `R_s` | HalfNormal at rectifier scale | |
| instrument terms | truncated from datasheets | no negative resistances or gains |

Prior sensitivity is a first-class output: every ledger entry carries a flag, and any conclusion that flips under the weak/informative prior swap is reported as prior-dependent.

## 6. Likelihood and noise

Voltage residuals are modelled as Gaussian with an instrument-derived scale plus an optional GP discrepancy term. When the latent-temperature model (M5) is active, the likelihood is the Kalman filter marginal likelihood over the thermal state, not a plug-in temperature.

### 6.1 First M5 implementation slice

The first Phase 4 slice is a deterministic synthetic chain:

```
T_plate    -> first-order response to heater/ambient
T_junction -> first-order response to T_plate
T_sensor   -> first-order response to T_junction
```

Each link uses the exact first-order factor `1 − exp(−Δt/τ)`. The synthetic sensor series adds
seeded Gaussian noise. The S6 page reports the configured time constants and a lag estimate from
the latent junction/sensor series.

The heater follows an ordered schedule of on-intervals (`heat_pulses`), not a single step: 90 s at
60 s, 300 s at 300 s and 720 s at 780 s of an 1800 s run at 10 s samples. The narrow first pulse
puts energy at the sensor's own timescale and the long third pulse drives the plate towards steady
state. Forward twin, filter and DMDc all read the same `heater_target_k`, and pulses must be
ordered and non-overlapping.

The same slice now runs a three-state UKF with state `[plate, junction, sensor]` and the noisy
sensor as its only measurement, followed by an RTS smoother over the full trace. It also fits a
Gaussian process with fixed RBF hyperparameters to the post-smoother sensor residual. That GP is a
diagnostic; the same residual is additionally carried as a latent Matérn-3/2 state — a two-state
linear SDE with exact discretisation, a Kalman filter and an RTS smoother — so the discrepancy is a
state of the model, not only a kernel fit ([ADR 0011](adr/0011-multi-pulse-schedule-and-matern-state.md)).
Its length and signal scales are not fixed: a Jeffreys hyperprior is placed over an 8 × 6 grid
(60–720 s, 0.005–0.16 K), the marginal likelihood is evaluated per grid point, and the reported
state is the moment-matched mixture ([ADR 0012](adr/0012-dmdc-noise-sensitivity-and-matern-hyperprior.md)).
On the default run that gives a 163 s length scale, a 5.2 mK signal scale at the grid floor and
95.6% coverage of the residual by the 95% predictive band, i.e. the post-smoother residual is
consistent with white sensor noise.

The filter also reports its own predictive log-likelihood: the sum of the Gaussian log densities of
the observations under the unscented predicted measurement. That is the Kalman prediction-error
decomposition for the configured model and noise level, not evidence over parameters.

A Hankel-DMDc fit identifies damping constants from the delay-embedded observed sensor and the
known heater input: order 4, three control lags, QR least squares on the centred series, companion
eigenvalues by Durand–Kerner, then `τ = −Δt / ln|λ|`. The three slowest distinct modes are shown
with their fit residual and their maximum relative error against the chain's configured constants:
433/74/29 s against 420/75/25 s on the default seed, so the slow and middle poles land within a few
percent and the fastest stays about 16% long because observation noise whitens it. The fit residual
includes the control terms. The regression stays ordinary least squares: two-stage and total least
squares were both measured and both were worse, and a noiseless trace is recovered to 1e-7
([ADR 0012](adr/0012-dmdc-noise-sensitivity-and-matern-hyperprior.md)). Each mode is reported with
a 5–95% band from 64 parametric-bootstrap refits of the fitted model; the heater amplitude was swept
and is not the limiting factor ([ADR 0013](adr/0013-dmdc-modal-bands.md)). That band is conditional on
one trace: it is 2.2–6.7× wider than the seed-to-seed spread, so it is a sensitivity band rather than
a cross-trace interval. A grey-box output-error fit of the same chain, which assumes the structure
instead of identifying it, recovers the fastest pole to 2.9% worst case against 34.8% for the blind
fit, which locates the residual error in the errors-in-variables bias. Its 5–95% band comes from 64
residual-bootstrap refits and is conditional on the structure; it contains the configured constant
in 13 of 15 seed × mode cases ([ADR 0014](adr/0014-greybox-and-dmdc-band-meaning.md)). See
[ADR 0010](adr/0010-dmdc-and-marginal-likelihood.md) and
[ADR 0011](adr/0011-multi-pulse-schedule-and-matern-state.md).

## 7. Identifiability

`E_g`, `ln B` and `η` are nearly collinear over the lab's 60–100 K window around room temperature. The Fisher spectrum (design §3.1) quantifies this; the app's job is to show it rather than to hide it behind a straight-line fit. Any claim that a wider temperature range would "fix" the gap must be demonstrated with the temperature-range slider before it appears in copy.

## 8. Units and numerical conventions

- Temperatures in kelvin, never Celsius, inside the model.
- `ln T` and `ln B` require positive quantities; parameters are sampled unconstrained and transformed with Jacobians.
- All random draws are seeded; seeds are part of the view state and of every validation run.
- Floating point: `f64` in Rust core and Python references; `f32` only at the WebGPU/visual boundary.
- The fail-closed diagnostic gate is `R-hat < 1.01`, bulk ESS `> 400`, mean energy error `< 0.1`, and zero divergences. A failed gate must not render a converged state.
